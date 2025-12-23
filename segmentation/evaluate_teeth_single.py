"""
道路損傷検出モデルの評価を行うスクリプト

入力画像に対してセグメンテーション推論を実行し、
正解データと比較して評価指標(IoU、Precision、Recall、Dice係数)を算出する。
また混同行列を生成し、クラス毎の予測精度を評価する。

評価結果はCSVファイルとして出力され、
セグメンテーション結果は可視化画像として保存される。
"""

from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_model, init_model
# Note: get_palette, get_classes, and show_result_pyplot have been removed or are unused
# init_model handles class information automatically via model.dataset_meta

from mmseg_custom.datasets import TeethSingleDataset
import cv2
import os.path as osp
import os,sys
import numpy as np
import torch
import time
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')  # バックエンド設定（GUIなし環境用）
import matplotlib.pyplot as plt

bgr_mean = np.array([0.406, 0.456, 0.485])
bgr_std = np.array([0.225, 0.224, 0.229])

def read_category_json(jpath):
    """カテゴリー定義JSONファイルからカテゴリーリストを読み込む

    Args:
        jpath: JSONファイルのパス
    Returns:
        カテゴリー名のリスト
    """
    jfp = open(jpath, 'r')
    jdata = json.load(jfp)

    cat_list = []
    for i, jd in enumerate(jdata):
        cat_list.append(jd[0])

    return cat_list


def create_confusion_matrix(
        index,
        args,
        images,
        gt,
        pred,
        confusion_matrix,
        pix_num,
        data_cnt,
        category_num,
        img_path_list,
        joint_label_map_value
):
    """混同行列を作成し精度評価を行う

    正解ラベルと予測ラベルの組み合わせをカウントし、
    クラス間の予測の正確さを評価するための混同行列を生成する。
    また評価結果の可視化も行う。

    Args:
        args: コマンドライン引数
        images: 入力画像
        gt: 正解ラベル
        pred: 予測ラベル 
        confusion_matrix: 混同行列
        pix_num: 総ピクセル数
        data_cnt: データ数カウンタ
        category_num: カテゴリー数
        img_path_list: 画像パスのリスト
        label_map: ラベルマップ
        joint_label_map_value: 結合クラスのラベルマップ

    Returns:
        confusion_matrix: 更新された混同行列
        pix_num: 更新された総ピクセル数
        data_cnt: 更新されたデータ数
    """

    # create confusion matrix
    # TP  gt: 2, pred: 2 --> 2 * 10 + 2 = 22
    # FN  gt: 2, pred: 3 --> 2 * 10 + 3 = 23

    encode_value = 10
    encoded = gt * encode_value + pred

    values, cnt = np.unique(encoded, return_counts=True)

    for v, c in zip(values, cnt):
        pid = v % encode_value
        gid = int((v - pid) / encode_value)
        confusion_matrix[pid][gid] += c

    height, width = pred.shape
    pix_num += height * width

    # 可視化は100件まで
    if index < 128:
        vis_batch_data(
            args,
            images[np.newaxis],
            gt[np.newaxis],
            pred[np.newaxis],
            [img_path_list],
            category_num,
            joint_label_map_value
        )

    return confusion_matrix, pix_num, data_cnt


def vis_batch_data(args, images, gt_idx, pred_idx, img_path_list, catN, joint_label_map_value):
    """バッチ単位での可視化処理

    バッチ内の各画像に対して:
    1. 予測・正解マスクをカラーマップに変換
    2. ラベルに応じた色付け
    3. 可視化画像の生成と保存

    Args:
        images: 入力画像バッチ
        gt_idx: 正解ラベルのインデックス
        pred_idx: 予測ラベルのインデックス
        img_path_list: 画像パスのリスト
        catN: カテゴリー数
    """

    batch_num = images.shape[0]

    # backgroundの色を設定
    joint_label_map_value_with_bg = [[0, 0, 0]] + joint_label_map_value

    for batch_incex in range(batch_num):
        img = denormalze_image(images[batch_incex])

        pimg = np.zeros_like(img)
        # gimg = np.zeros_like(img)

        gimg = np.repeat(gt_idx[batch_incex][:, :, np.newaxis], 3, axis=-1)
        gimg = gimg.astype(np.uint8)

        for id, cols in enumerate(joint_label_map_value_with_bg):
            imask = ((pred_idx[batch_incex]) == id)
            pimg[imask] = cols[0]

            gmask = (gt_idx[batch_incex]) == id
            gimg[gmask] = cols[0]

        create_visualization(args, img_path_list[batch_incex], img, pimg, gimg)


def denormalze_image(img):
    return (img * bgr_std + bgr_mean) * 255.0


def create_visualization(args, orig_path, img, pimg, gimg):
    """評価結果の可視化画像を生成

    4枚の画像を2x2で配置:
    - 元画像
    - ground truth
    - 推論した歯とう蝕のセグメンテーション画像を元画像にオーバーレイ表示
    - 推論した歯とう蝕のセグメンテーション画像

    Args:
        orig_path: 元画像のパス
        img: 入力画像
        pimg: 予測セグメンテーション（歯とう蝕の両方）
        gimg: 正解セグメンテーション
    """

    vis_dir = os.path.join(args.out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # Read original-size image
    orig_img = cv2.imread(os.path.join(args.data_basepath, orig_path))
    orig_h, orig_w = orig_img.shape[:2]

    # ─── 画像のリサイズと型変換 ──────────────────────────
    pred_resized = cv2.resize(pimg, (orig_w, orig_h))
    img_resized  = cv2.resize(img,  (orig_w, orig_h))

    if pred_resized.dtype != orig_img.dtype:
        pred_resized = pred_resized.astype(orig_img.dtype)

    # ─── オーバーレイ画像の作成 ──────────────────────────
    overlay = cv2.addWeighted(orig_img, 0.55, pred_resized, 0.45, 0)

    # gimg の整形
    if gimg is not None:
        if gimg.ndim == 2:  # 1ch → 3ch
            gimg = cv2.cvtColor(gimg, cv2.COLOR_GRAY2BGR)
        gimg_resized = cv2.resize(gimg, (orig_w, orig_h))
        if gimg_resized.dtype != orig_img.dtype:
            gimg_resized = gimg_resized.astype(orig_img.dtype)
    else:
        gimg_resized = np.zeros_like(orig_img)

    # ─── キャプション描画関数 ──────────────────
    def put_caption(image, text):
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.7
        thickness   = 2
        color_text  = (255, 255, 255)  # 白文字
        color_edge  = (0, 0, 0)        # 黒縁
        org = (10, 25)                 # 左上少し下
        # 縁取り（黒）
        cv2.putText(image, text, org, font, font_scale,
                    color_edge, thickness + 2, cv2.LINE_AA)
        # 本体（白）
        cv2.putText(image, text, org, font, font_scale,
                    color_text, thickness, cv2.LINE_AA)
        return image

    # 各画像にキャプション
    orig_img_cap = put_caption(orig_img.copy(), "original image")
    gimg_cap = put_caption(gimg_resized.copy(), "ground truth")
    overlay_cap = put_caption(overlay.copy(), "overlay (pred + orig)")
    pred_cap = put_caption(pred_resized.copy(), "prediction")

    # ─── 行ごとに連結 ───────────────────────────
    vis_row1 = np.hstack((orig_img_cap, gimg_cap))
    vis_row2 = np.hstack((overlay_cap, pred_cap))

    combined_vis = np.vstack((vis_row1, vis_row2))

    # Save combined visualization
    base_name = os.path.splitext(os.path.basename(orig_path))[0]
    vis_path = os.path.join(vis_dir, f"{base_name}_combined.jpg")
    cv2.imwrite(vis_path, combined_vis)


def read_category_json_value(jpath):
    jfp = open(jpath, 'r')
    jdata = json.load(jfp)

    color_list = []
    for i, jd in enumerate(jdata):
        clist = []
        for j in range(1, len(jd)):
            clist.append(np.array(jd[j], np.uint8))
        color_list.append(clist)

    return color_list

def create_gt_index_map(gt, category_num):
    # gt, pred
    # [0] background
    # [1] A1_tai_caries_confirmed
    # [2] A1_tai_caries_suspect

    # If gt is already in index format (2D), return as-is
    # If gt is one-hot encoded (3D), convert to index format
    if gt.ndim == 2:
        gt_idx = gt
    else:
        gt_idx = np.argmax(gt, axis=-1)

    return gt_idx


def calc_image_metrics(gt, pred, num_classes):
    """単一画像のメトリクスを計算

    Args:
        gt: Ground truth index map (HxW)
        pred: Prediction index map (HxW)
        num_classes: クラス数（背景を含む）

    Returns:
        dict: クラスごとのメトリクス {class_idx: {'iou': val, 'precision': val, 'recall': val, 'dice': val}}
    """
    metrics = {}

    # 混同行列を作成
    encode_value = 10
    encoded = gt * encode_value + pred
    values, cnt = np.unique(encoded, return_counts=True)

    # 小さな混同行列を初期化
    confusion_matrix_img = np.zeros((num_classes, num_classes))
    for v, c in zip(values, cnt):
        pid = v % encode_value
        gid = int((v - pid) / encode_value)
        if pid < num_classes and gid < num_classes:
            confusion_matrix_img[pid][gid] += c

    # 各クラスのメトリクスを計算
    for i in range(num_classes):
        tp = np.longlong(confusion_matrix_img[i][i])
        fn = np.longlong(confusion_matrix_img[:, i].sum()) - tp
        fp = np.longlong(confusion_matrix_img[i, :].sum()) - tp

        # IoU
        denom = tp + fn + fp
        iou = float(tp) / denom if denom > 0 else float('nan')

        # Precision
        denom = tp + fp
        precision = float(tp) / denom if denom > 0 else float('nan')

        # Recall
        denom = tp + fn
        recall = float(tp) / denom if denom > 0 else float('nan')

        # Dice
        denom = 2 * tp + fn + fp
        dice = 2.0 * float(tp) / denom if denom > 0 else float('nan')

        metrics[i] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'dice': dice
        }

    return metrics


def save_per_image_metrics_csv(per_image_metrics, class_names, save_dir):
    """画像ごとのメトリクスをCSVに保存

    Args:
        per_image_metrics: 画像ごとのメトリクスのリスト
            [{'filename': str, 'metrics': {class_idx: {'iou': val, ...}}}, ...]
        class_names: クラス名のリスト
        save_dir: 保存先ディレクトリ
    """
    import csv
    import os

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'per_image_metrics.csv')

    # CSVヘッダーを構築
    header = ['filename']
    for cls_name in class_names:
        header.extend([
            f'{cls_name}_iou',
            f'{cls_name}_precision',
            f'{cls_name}_recall',
            f'{cls_name}_dice'
        ])

    # CSVに書き込み
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_data in per_image_metrics:
            filename = img_data['filename']
            metrics = img_data['metrics']

            row = [filename]
            for class_idx in range(len(class_names)):
                if class_idx in metrics:
                    m = metrics[class_idx]
                    row.extend([
                        f"{m['iou']:.4f}" if not np.isnan(m['iou']) else 'nan',
                        f"{m['precision']:.4f}" if not np.isnan(m['precision']) else 'nan',
                        f"{m['recall']:.4f}" if not np.isnan(m['recall']) else 'nan',
                        f"{m['dice']:.4f}" if not np.isnan(m['dice']) else 'nan'
                    ])
                else:
                    row.extend(['nan', 'nan', 'nan', 'nan'])

            writer.writerow(row)

    print(f"Per-image metrics saved to: {csv_path}")


def visualize_per_image_metrics(per_image_metrics, class_names, save_dir):
    """画像ごとのメトリクスを可視化

    複数の可視化を生成:
    1. メトリクス分布のヒストグラム
    2. クラスごとのBox Plot
    3. メトリクス相関のScatter Matrix
    4. 累積分布関数 (CDF)
    5. 統計サマリーテーブル

    Args:
        per_image_metrics: 画像ごとのメトリクスのリスト
        class_names: クラス名のリスト
        save_dir: 保存先ディレクトリ
    """
    import os
    from scipy import stats

    os.makedirs(save_dir, exist_ok=True)

    # クラスごとにメトリクスを集計
    # ヒストグラム/Box plot用：個別にNaN除外
    metrics_by_class = {cls_name: {'iou': [], 'precision': [], 'recall': [], 'dice': []}
                        for cls_name in class_names}

    for img_data in per_image_metrics:
        metrics = img_data['metrics']
        for class_idx, cls_name in enumerate(class_names):
            if class_idx in metrics:
                m = metrics[class_idx]
                for metric_name in ['iou', 'precision', 'recall', 'dice']:
                    val = m[metric_name]
                    if not np.isnan(val):
                        metrics_by_class[cls_name][metric_name].append(val)

    # Scatter matrix用：画像単位でNaN除外して配列長を揃える
    metrics_by_class_aligned = {cls_name: {'iou': [], 'precision': [], 'recall': [], 'dice': []}
                                for cls_name in class_names}

    for img_data in per_image_metrics:
        metrics = img_data['metrics']
        for class_idx, cls_name in enumerate(class_names):
            if class_idx in metrics:
                m = metrics[class_idx]
                # 全メトリクスが有効な場合のみ追加（配列長を揃える）
                if all(not np.isnan(m[metric_name]) for metric_name in ['iou', 'precision', 'recall', 'dice']):
                    for metric_name in ['iou', 'precision', 'recall', 'dice']:
                        metrics_by_class_aligned[cls_name][metric_name].append(m[metric_name])

    # 1. ヒストグラム（各メトリクス × 各クラス）
    metric_names = ['iou', 'precision', 'recall', 'dice']
    for metric_name in metric_names:
        fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 4))
        if len(class_names) == 1:
            axes = [axes]

        for idx, cls_name in enumerate(class_names):
            values = metrics_by_class[cls_name][metric_name]
            if len(values) > 0:
                axes[idx].hist(values, bins=20, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{cls_name}\n{metric_name.capitalize()}')
                axes[idx].set_xlabel(metric_name.capitalize())
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)

                # 統計情報を追加
                mean_val = np.mean(values)
                std_val = np.std(values)
                axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                axes[idx].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'histogram_{metric_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Box Plot（全クラス × 全メトリクス）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        data_to_plot = [metrics_by_class[cls_name][metric_name] for cls_name in class_names]

        # 空リストチェック: 全てのクラスが空の場合はスキップ
        if all(len(d) == 0 for d in data_to_plot):
            axes[idx].text(0.5, 0.5, 'No data available', ha='center', va='center',
                          transform=axes[idx].transAxes, fontsize=12)
            axes[idx].set_title(f'{metric_name.capitalize()} Distribution by Class',
                              fontsize=12, fontweight='bold')
            continue

        # 空リストをダミー値に置き換え（boxplotエラー回避）
        data_to_plot_safe = [[0] if len(d) == 0 else d for d in data_to_plot]
        bp = axes[idx].boxplot(data_to_plot_safe, labels=class_names, patch_artist=True)

        # カラフルに
        colors = plt.cm.Set3(range(len(class_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[idx].set_title(f'{metric_name.capitalize()} Distribution by Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric_name.capitalize())
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplot_all_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Scatter Matrix（メトリクス間の相関）クラスごと
    # 配列長が揃ったmetrics_by_class_alignedを使用
    for cls_name in class_names:
        metrics_data = metrics_by_class_aligned[cls_name]

        # 全メトリクスのデータがある場合のみ
        if all(len(metrics_data[m]) > 0 for m in metric_names):
            fig, axes = plt.subplots(4, 4, figsize=(14, 14))

            for i, metric_i in enumerate(metric_names):
                for j, metric_j in enumerate(metric_names):
                    ax = axes[i, j]

                    if i == j:
                        # 対角線上はヒストグラム
                        ax.hist(metrics_data[metric_i], bins=15, alpha=0.7, edgecolor='black')
                        ax.set_ylabel('Frequency' if j == 0 else '')
                    else:
                        # 散布図（配列長が揃っているのでエラーなし）
                        ax.scatter(metrics_data[metric_j], metrics_data[metric_i],
                                 alpha=0.5, s=20)

                        # 相関係数を計算して表示
                        if len(metrics_data[metric_i]) > 1:
                            corr, _ = stats.pearsonr(metrics_data[metric_j], metrics_data[metric_i])
                            ax.text(0.05, 0.95, f'r={corr:.2f}',
                                  transform=ax.transAxes, fontsize=9,
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    # ラベル設定
                    if i == 3:
                        ax.set_xlabel(metric_j.capitalize())
                    if j == 0:
                        ax.set_ylabel(metric_i.capitalize())

                    ax.grid(True, alpha=0.3)

            plt.suptitle(f'Metric Correlation Matrix - {cls_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'scatter_matrix_{cls_name}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()

    # 4. 累積分布関数 (CDF)
    for metric_name in metric_names:
        fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 4))
        if len(class_names) == 1:
            axes = [axes]

        for idx, cls_name in enumerate(class_names):
            values = np.array(sorted(metrics_by_class[cls_name][metric_name]))
            if len(values) > 0:
                cdf = np.arange(1, len(values) + 1) / len(values)
                axes[idx].plot(values, cdf, marker='.', linestyle='-', markersize=3)
                axes[idx].set_title(f'{cls_name}\n{metric_name.capitalize()} CDF')
                axes[idx].set_xlabel(metric_name.capitalize())
                axes[idx].set_ylabel('Cumulative Probability')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim([0, 1])
                axes[idx].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cdf_{metric_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 5. 統計サマリーテーブル
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # テーブルデータを構築
    table_data = []
    header_row = ['Class', 'Metric', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Count']

    for cls_name in class_names:
        for metric_name in metric_names:
            values = np.array(metrics_by_class[cls_name][metric_name])
            if len(values) > 0:
                row = [
                    cls_name,
                    metric_name.capitalize(),
                    f'{np.mean(values):.4f}',
                    f'{np.std(values):.4f}',
                    f'{np.min(values):.4f}',
                    f'{np.percentile(values, 25):.4f}',
                    f'{np.median(values):.4f}',
                    f'{np.percentile(values, 75):.4f}',
                    f'{np.max(values):.4f}',
                    f'{len(values)}'
                ]
            else:
                row = [cls_name, metric_name.capitalize()] + ['N/A'] * 8

            table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=header_row,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # ヘッダーのスタイリング
    for i in range(len(header_row)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 行の交互色
    for i in range(1, len(table_data) + 1):
        for j in range(len(header_row)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Statistical Summary of Per-Image Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'statistics_summary_table.png'),
               dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {save_dir}")


def calc_metrics(args, joint_category_num, confusion_matrix, joint_label_map, tp_num, fp_num, fn_num):
    """評価指標を計算しCSVファイルに出力する

    混同行列から以下の評価指標を計算:
    - IoU (Intersection over Union)
    - Precision (適合率)
    - Recall (再現率) 
    - Dice係数

    計算結果はCSVファイルに保存する。
    また混同行列も別ファイルに出力する。

    Args:
        args: コマンドライン引数
        category_num: カテゴリー数
        joint_category_num: 結合カテゴリー数
        confusion_matrix: 混同行列
        category_list: カテゴリーリスト
        joint_label_map: 結合ラベルマップ
        tp_num: True Positive数
        fp_num: False Positive数
        fn_num: False Negative数
    """
    iou_list = []
    p_list = []
    r_list = []
    d_list = []
    for i in range(joint_category_num + 1):
        tp = np.longlong(confusion_matrix[i][i])
        fn = np.longlong(confusion_matrix[:, i].sum()) - tp
        fp = np.longlong(confusion_matrix[i, :].sum()) - tp

        # iou
        denom = tp + fn + fp
        if denom == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(float(tp) / denom)

        # precision
        denom = tp + fp
        if denom == 0:
            p_list.append(float('nan'))
        else:
            p_list.append(float(tp) / denom)

        # recall
        denom = tp + fn
        if denom == 0:
            r_list.append(float('nan'))
        else:
            r_list.append(float(tp) / denom)

        # Dice
        denom = 2 * tp + fn + fp
        if denom == 0:
            d_list.append(float('nan'))
        else:
            d_list.append(2.0 * float(tp) / denom)

    efp = open(os.path.join(args.out_dir, "eval.csv"), 'w')
    cfp = open(os.path.join(args.out_dir, "conf.csv"), 'w')

    cat_name_list = ["bg"]

    for jname in joint_label_map:
        cat_name_list.append(jname)

    efp.write("category,iou,precision,recall,dice\n")
    cfp.write("p/g")
    for i in range(joint_category_num + 1):
        efp.write("{0},".format(cat_name_list[i]))
        cfp.write(",{0}".format(cat_name_list[i]))
        efp.write("{0:.3f},{1:.3f},{2:.3f},{3:.3f}\n".format(iou_list[i], p_list[i], r_list[i], d_list[i]))

    cfp.write("\n")

    efp.close()

    for i in range(joint_category_num + 1):
        cfp.write("{0}".format(cat_name_list[i]))

        for j in range(joint_category_num + 1):
            cfp.write(",{0:.3f}".format(confusion_matrix[i][j]))

        cfp.write("\n")

    cfp.close()


def main():
    """メイン処理

    1. コマンドライン引数の解析
    2. モデルの構築と重みの読み込み
    3. データセットの準備
    4. 各画像に対する推論実行
    5. 評価指標の計算と結果出力
    """
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--data_list',type=str,required=True)
    parser.add_argument('--base_path',type=str,required=True)
    parser.add_argument('--out_dir', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--class_def_json', type=str)
    parser.add_argument('--joint_cls_def_json', type=str)
    parser.add_argument('--data_basepath', type=str)
    parser.add_argument('--use_joint_prob', dest='use_joint_prob', action='store_true', default=False)

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    # Note: init_model now handles checkpoint loading and class information automatically
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)

    # bgr_mean = np.array([0.406, 0.456, 0.485])
    # bgr_std = np.array([0.225, 0.224, 0.229])
    # img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadTeethAnnotations')
    ]

    dataset = TeethSingleDataset(
        test_pipeline,
        args.data_list,
        data_root=args.base_path,
        test_mode=False,
        class_color_json=args.class_def_json,
        caries_color_json=args.joint_cls_def_json
    )

    os.makedirs(args.out_dir,exist_ok=True)
    model.eval()

    time_length = 0
    img_cnt = 0

    # use_joint_prob = True

    # Get classes from model metadata (mmseg 1.2.2+ uses dataset_meta instead of CLASSES attribute)
    # Use getattr with default empty dict to avoid AttributeError
    classes = getattr(model, 'dataset_meta', {}).get('classes',
                      getattr(model, 'metainfo', {}).get('classes', []))
    confusion_matrix = np.zeros((len(classes), len(classes)))

    category_list = read_category_json(args.class_def_json)
    category_num = len(category_list)
    data_cnt = 0
    pix_num = 0

    tp_num = np.zeros(2)
    fp_num = np.zeros(2)
    fn_num = np.zeros(2)

    # label_map = read_category_json_value(args.class_def_json)
    joint_label_map = read_category_json(args.joint_cls_def_json)

    # backbroundの色を入れる
    joint_label_map_value = read_category_json_value(args.joint_cls_def_json)

    joint_category_num = 0
    if args.use_joint_prob:
        joint_category_num = len(joint_label_map)

    # 画像ごとのメトリクスを格納するリスト
    per_image_metrics = []

    for index in range(len(dataset)):
        print(f'index: {index}')
        datas = dataset.__getitem__(index)
        gt = dataset.get_gt_seg_map_by_idx(index)

        stime = time.perf_counter()

        ipath = datas['img_info']['filename']
        img = datas['img']
        # Note: inference_model now returns SegDataSample object, not numpy array
        seg_result = inference_model(model, img)
        result = seg_result.pred_sem_seg.data.cpu().numpy()[0]
        gt_idx = create_gt_index_map(gt, category_num)

        img_path = datas['img_info']['filename']
        confusion_matrix, pix_num, data_cnt = create_confusion_matrix(
            index,
            args,
            img,
            gt_idx,
            result,
            confusion_matrix,
            pix_num,
            data_cnt,
            category_num,
            img_path,
            joint_label_map_value
        )

        # 画像ごとのメトリクスを計算
        img_metrics = calc_image_metrics(gt_idx, result, len(classes))
        per_image_metrics.append({
            'filename': os.path.basename(img_path),
            'metrics': img_metrics
        })

    # Dataset全体のメトリクスを計算
    calc_metrics(args, joint_category_num, confusion_matrix, joint_label_map, tp_num, fp_num, fn_num)

    # 画像ごとのメトリクスをCSVに保存
    print("\n" + "="*80)
    print("Per-Image Metrics Summary")
    print("="*80)
    save_per_image_metrics_csv(per_image_metrics, list(classes), args.out_dir)

    # 画像ごとのメトリクスを可視化
    visualize_per_image_metrics(per_image_metrics, list(classes), args.out_dir)


if __name__ == '__main__':
    main()
    
    