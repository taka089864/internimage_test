"""
歯のう蝕検出のためのタイルベース推論スクリプト

大きな画像をタイルに分割し、各タイルで推論を実行した後、
投票方式で結果を元の画像サイズに復元する。

evaluate_teeth_single.pyのコードを最大限再利用し、
タイル分割・結合のロジックのみを追加。
"""

from argparse import ArgumentParser
import os
import glob
import json
import time

import cv2
import numpy as np
import torch
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg.apis import inference_model, init_model
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmseg.utils import get_classes, get_palette


# evaluate_teeth_single.pyから移植
bgr_mean = np.array([0.406, 0.456, 0.485])
bgr_std = np.array([0.225, 0.224, 0.229])


def get_image_files(directory):
    """ディレクトリ内の画像ファイルを取得

    Args:
        directory: 画像ディレクトリパス

    Returns:
        ソートされた画像ファイルパスのリスト
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)


def read_category_json_value(jpath):
    """カテゴリー定義JSONからカラー情報を読み込む

    evaluate_teeth_single.pyから移植

    Args:
        jpath: JSONファイルのパス

    Returns:
        カラーリスト（2次元リスト）
    """
    jfp = open(jpath, 'r')
    jdata = json.load(jfp)

    color_list = []
    for i, jd in enumerate(jdata):
        clist = []
        for j in range(1, len(jd)):
            clist.append(np.array(jd[j], np.uint8))
        color_list.append(clist)

    return color_list


def denormalze_image(img):
    """画像の正規化を解除

    evaluate_teeth_single.pyから移植

    Args:
        img: 正規化済み画像

    Returns:
        正規化解除された画像
    """
    return (img * bgr_std + bgr_mean) * 255.0


def generate_tiles(image, tile_size, overlap_ratio):
    """画像をタイルに分割

    Args:
        image: 入力画像 (H, W, C)
        tile_size: タイルサイズ（正方形）
        overlap_ratio: オーバーラップ比率（0～1）

    Returns:
        タイル情報のリスト。各要素は辞書：
        {
            'x_start': int,
            'y_start': int,
            'x_end': int,
            'y_end': int,
            'tile_img': np.ndarray (tile_size, tile_size, C)
        }
    """
    h, w = image.shape[:2]

    # overlap_ratio検証（stride計算の前に実行）
    if not (0 <= overlap_ratio < 1):
        raise ValueError(
            f"Invalid overlap ratio {overlap_ratio}: must be in [0, 1). "
            f"Values >= 1.0 would create millions of tiles and cause system freeze."
        )

    # stride計算
    stride = int(tile_size * (1 - overlap_ratio))

    # strideの最小値チェック（タイル数爆発を防ぐ）
    min_stride = max(1, tile_size // 100)  # タイルサイズの1%以上、最低1ピクセル
    if stride < min_stride:
        raise ValueError(
            f"Overlap ratio {overlap_ratio} results in stride={stride} pixels, "
            f"which is too small (< {min_stride}). This would create excessive tiles. "
            f"Use overlap < {1.0 - min_stride/tile_size:.4f} to ensure reasonable stride."
        )

    # タイル数の上限チェック（メモリ保護）
    estimated_tiles_h = (h + stride - 1) // stride
    estimated_tiles_w = (w + stride - 1) // stride
    estimated_tiles = estimated_tiles_h * estimated_tiles_w
    max_tiles = 10000  # 安全上限

    if estimated_tiles > max_tiles:
        raise ValueError(
            f"Current settings would create {estimated_tiles} tiles "
            f"({estimated_tiles_h}×{estimated_tiles_w}), exceeding safe limit of {max_tiles}. "
            f"Increase tile_size or decrease overlap ratio."
        )

    tiles = []

    # 垂直方向のタイル
    y_positions = []
    y = 0
    while y < h:
        y_end = min(y + tile_size, h)
        if y_end == h and y_end - y < tile_size:
            # 最後のタイルが小さすぎる場合、位置を調整
            y = max(0, h - tile_size)
            y_end = h
        y_positions.append((y, y_end))
        if y_end == h:
            break
        y += stride

    # 水平方向のタイル
    x_positions = []
    x = 0
    while x < w:
        x_end = min(x + tile_size, w)
        if x_end == w and x_end - x < tile_size:
            # 最後のタイルが小さすぎる場合、位置を調整
            x = max(0, w - tile_size)
            x_end = w
        x_positions.append((x, x_end))
        if x_end == w:
            break
        x += stride

    # タイルを生成
    for y_start, y_end in y_positions:
        for x_start, x_end in x_positions:
            tile_img = image[y_start:y_end, x_start:x_end].copy()

            # タイルサイズが足りない場合はパディング
            if tile_img.shape[0] < tile_size or tile_img.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, tile_img.shape[2]), dtype=tile_img.dtype)
                padded[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                tile_img = padded

            tiles.append({
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'tile_img': tile_img
            })

    return tiles


def inference_tile_with_probs(model, tile_img):
    """タイルの確率マップを取得

    新しいmmseg API（mmengine）を使用して確率マップを取得する。
    argmax前の確率マップを取得することで、オーバーラップ領域で確率を平均化でき、
    医療画像で重要な偽陰性（病変の見逃し）を最小化できる。

    Args:
        model: セグメンテーションモデル
        tile_img: タイル画像 (H, W, C) BGR形式のndarray

    Returns:
        確率マップ (num_classes, H, W)
    """
    cfg = model.cfg

    # test_pipelineの準備（LoadAnnotationsを除く）
    test_pipeline_cfg = cfg.test_pipeline.copy()
    for t in test_pipeline_cfg:
        if t.get('type') == 'LoadAnnotations':
            test_pipeline_cfg.remove(t)

    # ndarrayの場合はLoadImageFromNDArrayを使用
    if isinstance(tile_img, np.ndarray):
        test_pipeline_cfg[0]['type'] = 'LoadImageFromNDArray'

    # パイプライン構築
    pipeline = Compose(test_pipeline_cfg)

    # データ準備（新しいAPI形式）
    data = dict(img=tile_img)
    data = pipeline(data)

    # バッチ化
    inputs = data['inputs'].unsqueeze(0)
    data_samples = [data['data_samples']]

    # デバイスに移動（モデルと同じデバイスに配置）
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # 確率マップ取得
    with torch.no_grad():
        # encode_decodeでlogits取得
        logits = model.encode_decode(inputs, data_samples)
        # Softmaxで確率に変換
        probs = torch.softmax(logits, dim=1)
        probs = probs.squeeze(0).cpu().numpy()  # (C, H, W)

    return probs


def merge_tiles_with_probabilities(tiles, tile_probs_list, img_shape, num_classes):
    """確率マップベースのタイル結合

    投票方式ではなく確率平均化を使用する理由：
    - 医療画像では偽陰性（病変の見逃し）を最小化する必要がある
    - 投票方式ではタイル境界で病変が消失するリスクがある
      例：3タイルがオーバーラップし、各タイルが異なるクラスを予測した場合、
          argmaxで背景（クラス0）が選ばれ、病変が消える
    - 確率平均化により各タイルの確信度を反映した予測が可能

    Args:
        tiles: タイル情報のリスト
        tile_probs_list: 各タイルの確率マップのリスト (C, H, W)
        img_shape: 元画像のshape (H, W, C)
        num_classes: クラス数

    Returns:
        最終予測マップ (H, W) - クラスインデックス
    """
    h, w = img_shape[:2]

    # 確率累積配列
    accumulated_probs = np.zeros((h, w, num_classes), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.int32)

    # 各タイルの確率を累積
    for tile_info, tile_probs in zip(tiles, tile_probs_list):
        y_start = tile_info['y_start']
        y_end = tile_info['y_end']
        x_start = tile_info['x_start']
        x_end = tile_info['x_end']

        # 実際のタイルサイズ（パディングを除く）
        actual_h = y_end - y_start
        actual_w = x_end - x_start

        # 確率を累積 (C, H, W) -> (H, W, C)
        tile_probs_hwc = tile_probs[:, :actual_h, :actual_w].transpose(1, 2, 0)
        accumulated_probs[y_start:y_end, x_start:x_end] += tile_probs_hwc
        pred_count[y_start:y_end, x_start:x_end] += 1

    # 平均化
    pred_count_safe = np.maximum(pred_count, 1)  # ゼロ除算回避
    avg_probs = accumulated_probs / pred_count_safe[:, :, np.newaxis]

    # argmaxでクラス決定
    final_pred = np.argmax(avg_probs, axis=-1)

    return final_pred


def load_annotation_from_json(json_path, caries_color_json_path):
    """Darwin JSONアノテーションからクラスIDマスクを生成

    Darwin V7形式のJSONアノテーションをパースしてクラスIDマスク（0/1/2）を生成。
    推論結果と同じ色描画ロジックで処理できるようにする。

    Args:
        json_path: Darwin JSONファイルのパス
        caries_color_json_path: caries_colors.jsonのパス（クラス名リスト取得用）

    Returns:
        クラスIDマスク（H×W、値は0=背景、1=suspect、2=confirmed）、失敗時はNone
    """
    try:
        # JSONファイル読み込み
        with open(json_path, 'r', encoding='utf-8') as f:
            darwin_data = json.load(f)

        # 画像サイズ取得
        slots = darwin_data.get('item', {}).get('slots', [])
        if not slots:
            return None

        width = slots[0].get('width', 0)
        height = slots[0].get('height', 0)
        if width == 0 or height == 0:
            return None

        # クラス名リスト読み込み（caries_colors.jsonの順序に従う）
        class_names = []
        if os.path.exists(caries_color_json_path):
            with open(caries_color_json_path, 'r', encoding='utf-8') as f:
                caries_colors_data = json.load(f)
            # [[name, [r, g, b]], ...] 形式からクラス名リストを抽出
            # インデックス0 → クラスID 1（suspect）
            # インデックス1 → クラスID 2（confirmed）
            for item in caries_colors_data:
                if len(item) >= 1:
                    class_names.append(item[0])

        # 空白マスク作成（背景=0）
        mask = np.zeros((height, width), dtype=np.uint8)

        # アノテーションを描画
        annotations = darwin_data.get('annotations', [])
        for ann in annotations:
            # タグアノテーションはスキップ
            if 'tag' in ann:
                continue

            # アノテーション名取得
            ann_name = ann.get('name', '')
            if not ann_name:
                continue

            # ポリゴンがない場合はスキップ
            if 'polygon' not in ann:
                continue

            # クラス名からクラスIDを決定
            # caries_colors.jsonのクラス名に部分一致するかチェック
            class_id = 0  # デフォルトは背景
            for idx, class_name in enumerate(class_names):
                # 部分一致チェック（両方向）
                if class_name in ann_name or ann_name in class_name:
                    class_id = idx + 1  # インデックス0 → クラスID 1
                    break

            # 背景でない場合のみ描画
            if class_id > 0:
                # ポリゴンを塗りつぶし
                polygon_paths = ann['polygon'].get('paths', [])
                for path in polygon_paths:
                    points = np.array([[int(p['x']), int(p['y'])] for p in path], dtype=np.int32)
                    cv2.fillPoly(mask, [points], class_id)

        return mask

    except Exception as e:
        print(f"Warning: Failed to load annotation from JSON {json_path}: {e}")
        return None


def load_annotation_image(annotation_dirs, img_filename, caries_color_json_path=None, verbose=True):
    """アノテーション画像またはJSONを複数ディレクトリから検索して読み込む

    元画像のファイル名と同じベース名を持つアノテーション画像またはJSONファイルを検索する。
    複数のディレクトリが指定された場合、優先順に検索し、最初に見つかったものを返す。
    JSONファイルが見つかった場合は、Darwin形式のアノテーションをパースしてマスク画像を生成。

    Args:
        annotation_dirs: アノテーション画像が格納されているディレクトリ（単一文字列またはリスト）
                        Noneまたは空リストの場合は読み込みなし
        img_filename: 元画像のファイル名（ベース名の抽出に使用）
        caries_color_json_path: caries_colors.jsonのパス（JSON読み込み時に使用）
        verbose: デバッグメッセージを表示するかどうか

    Returns:
        アノテーション画像（BGR形式）、見つからない場合はNone
    """
    # None チェック
    if annotation_dirs is None:
        if verbose:
            print(f"  [GT] Annotation directory not specified")
        return None

    # 単一文字列の場合はリストに変換（後方互換性）
    if isinstance(annotation_dirs, str):
        annotation_dirs = [annotation_dirs]

    # 空リストチェック
    if len(annotation_dirs) == 0:
        if verbose:
            print(f"  [GT] Annotation directory list is empty")
        return None

    # ベース名取得（拡張子なし）
    base_name = os.path.splitext(os.path.basename(img_filename))[0]

    # 各ディレクトリを優先順に検索
    image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

    for annotation_dir in annotation_dirs:
        # ディレクトリの存在確認
        if not os.path.exists(annotation_dir):
            if verbose:
                print(f"  [GT] Directory not found (skipping): {annotation_dir}")
            continue

        # 画像ファイルを優先して試行
        for ext in image_extensions:
            annotation_path = os.path.join(annotation_dir, base_name + ext)
            if os.path.exists(annotation_path):
                annotation_img = cv2.imread(annotation_path)
                if annotation_img is not None:
                    if verbose:
                        print(f"  [GT] Found annotation image in {os.path.basename(annotation_dir)}: {base_name}{ext}")
                    return annotation_img

        # 画像が見つからない場合、JSONファイルを試行
        json_path = os.path.join(annotation_dir, base_name + '.json')
        if os.path.exists(json_path):
            if caries_color_json_path is None:
                if verbose:
                    print(f"  [GT] Warning: JSON annotation found but caries_color_json_path not provided: {json_path}")
                continue  # 次のディレクトリを試行
            # Darwin JSONからマスク画像を生成
            if verbose:
                print(f"  [GT] Found annotation JSON in {os.path.basename(annotation_dir)}: {base_name}.json")
            return load_annotation_from_json(json_path, caries_color_json_path)

    # すべてのディレクトリで見つからなかった場合
    if verbose:
        print(f"  [GT] No annotation found in any directory for: {base_name}")
    return None


def create_visualization(args, img_path, img, pimg, joint_label_map_value, gimg=None):
    """推論結果の可視化画像を生成（4パネルレイアウト）

    evaluate_teeth_single.pyのcreate_visualization()を基に修正。

    レイアウト：
    - 左上：GTオーバーレイ（元画像 + GTアノテーション）
    - 右上：ground truth（アノテーション画像、ない場合は空白）
    - 左下：予測オーバーレイ（元画像 + 推論結果）
    - 右下：推論結果

    Args:
        args: コマンドライン引数
        img_path: 元画像のパス
        img: 入力画像（正規化済み）
        pimg: 予測セグメンテーション画像
        joint_label_map_value: カラーマップ
        gimg: 正解アノテーション画像（オプション）
    """
    vis_dir = os.path.join(args.out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # 元画像読み込み
    orig_img = cv2.imread(img_path)
    orig_h, orig_w = orig_img.shape[:2]

    # 予測画像のリサイズと型変換
    pred_resized = cv2.resize(pimg, (orig_w, orig_h))
    if pred_resized.dtype != orig_img.dtype:
        pred_resized = pred_resized.astype(orig_img.dtype)

    # 予測オーバーレイ画像の作成
    pred_overlay = cv2.addWeighted(orig_img, 0.55, pred_resized, 0.45, 0)

    # gimg（GT）の整形（evaluate_teeth_single.pyから移植）
    gt_available = gimg is not None
    if gt_available:
        if gimg.ndim == 2:  # 1ch → 3ch
            gimg = cv2.cvtColor(gimg, cv2.COLOR_GRAY2BGR)
        gimg_resized = cv2.resize(gimg, (orig_w, orig_h))
        if gimg_resized.dtype != orig_img.dtype:
            gimg_resized = gimg_resized.astype(orig_img.dtype)
    else:
        # GTが利用できない場合、元画像をコピーして"No GT Available"と表示
        gimg_resized = orig_img.copy()

    # GTオーバーレイ画像の作成（元画像 + GTアノテーション）
    if gt_available:
        gt_overlay = cv2.addWeighted(orig_img, 0.55, gimg_resized, 0.45, 0)
    else:
        # GTがない場合は元画像をそのまま使用
        gt_overlay = orig_img.copy()

    # キャプション描画関数（evaluate_teeth_single.pyから移植）
    def put_caption(image, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color_text = (255, 255, 255)  # 白文字
        color_edge = (0, 0, 0)  # 黒縁
        org = (10, 25)  # 左上少し下
        # 縁取り（黒）
        cv2.putText(image, text, org, font, font_scale,
                    color_edge, thickness + 2, cv2.LINE_AA)
        # 本体（白）
        cv2.putText(image, text, org, font, font_scale,
                    color_text, thickness, cv2.LINE_AA)
        return image

    # 各画像にキャプション
    if gt_available:
        gt_overlay_cap = put_caption(gt_overlay.copy(), "overlay (gt + orig)")
        gimg_cap = put_caption(gimg_resized.copy(), "ground truth")
    else:
        gt_overlay_cap = put_caption(gt_overlay.copy(), "No GT Available")
        gimg_cap = put_caption(gimg_resized.copy(), "No GT Available")

    pred_cap = put_caption(pred_resized.copy(), "prediction")
    pred_overlay_cap = put_caption(pred_overlay.copy(), "overlay (pred + orig)")

    # 行ごとに連結
    vis_row1 = np.hstack((gt_overlay_cap, gimg_cap))
    vis_row2 = np.hstack((pred_overlay_cap, pred_cap))

    combined_vis = np.vstack((vis_row1, vis_row2))

    # 保存
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    vis_path = os.path.join(vis_dir, f"{base_name}_combined.jpg")
    cv2.imwrite(vis_path, combined_vis)


def main():
    """メイン処理

    evaluate_teeth_single.pyの構造を踏襲しつつ、
    タイル分割・結合のロジックを追加。
    """
    parser = ArgumentParser(description='Tiled inference for dental caries detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Model config file (e.g., configs/medical_shift/internimage_l_640_teeth_single.py)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Trained model checkpoint (e.g., results/train/latest.pth)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--out_dir', type=str, default='tiled_inference_output',
                        help='Output directory for predictions and visualizations')
    parser.add_argument('--tile_size', type=int, default=640,
                        help='Tile size for splitting large images (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap ratio between adjacent tiles, range [0, 1) (default: 0.2)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference (default: cuda:0)')
    parser.add_argument('--class_def_json', type=str, default=None,
                        help='Class definition JSON file (optional, will be read from config if not specified)')
    parser.add_argument('--joint_cls_def_json', type=str, default=None,
                        help='Joint class (caries) definition JSON file (optional, will be read from config if not specified)')
    parser.add_argument('--annotation', type=str, nargs='+', default=None,
                        help='Directory(ies) containing ground truth annotation images (searched in order, optional)')

    args = parser.parse_args()

    # モデル初期化（新しいAPI: init_modelはcheckpointも一緒に読み込む）
    print("=== Loading model ===")
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))

    # init_modelがcheckpoint読み込みとクラス情報の設定を自動的に行う
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)

    # dataset_metaからクラス情報を取得（新しいmmseg 1.xの形式）
    if hasattr(model, 'dataset_meta') and model.dataset_meta is not None:
        model_classes = model.dataset_meta.get('classes', None)
        if model_classes is not None:
            num_model_classes = len(model_classes)
        else:
            num_model_classes = model.decode_head.num_classes
    else:
        # dataset_metaがない場合はdecode_headから取得
        num_model_classes = model.decode_head.num_classes

    print(f"Model loaded: {num_model_classes} classes")

    # configから設定を取得
    cfg = model.cfg

    # JSON pathsをconfigから取得（引数でオーバーライド可能）
    joint_cls_def_json = args.joint_cls_def_json
    if joint_cls_def_json is None:
        joint_cls_def_json = cfg.data.test.get('caries_color_json')
        if joint_cls_def_json is None:
            raise ValueError("joint_cls_def_json not found in config and not specified in arguments")

    class_def_json = args.class_def_json
    if class_def_json is None:
        class_def_json = cfg.data.test.get('class_color_json')

    print(f"Using joint_cls_def_json: {joint_cls_def_json}")
    if class_def_json:
        print(f"Using class_def_json: {class_def_json}")

    # 画像ファイル取得
    print("\n=== Loading images ===")
    image_files = get_image_files(args.input_dir)
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {args.input_dir}")

    print(f"Found {len(image_files)} images")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap ratio: {args.overlap}")

    # 出力ディレクトリ作成
    os.makedirs(args.out_dir, exist_ok=True)

    # カラーパレット読み込み（evaluate_teeth_single.pyと同じ）
    joint_label_map_value = read_category_json_value(joint_cls_def_json)
    num_classes = len(joint_label_map_value) + 1  # +1 for background

    # backgroundの色を設定（evaluate_teeth_single.pyと同じ）
    joint_label_map_value_with_bg = [[0, 0, 0]] + joint_label_map_value

    print(f"Number of classes: {num_classes}")

    # アノテーション設定の表示
    if args.annotation:
        if isinstance(args.annotation, list):
            print(f"Annotation directories ({len(args.annotation)} dirs, searched in order):")
            for i, annotation_dir in enumerate(args.annotation, 1):
                exists_status = "✓" if os.path.exists(annotation_dir) else "✗"
                print(f"  [{i}] {exists_status} {annotation_dir}")
        else:
            print(f"Annotation directory: {args.annotation}")
            if not os.path.exists(args.annotation):
                print(f"  Warning: Annotation directory does not exist!")

    # 各画像を処理
    print("\n=== Processing images ===")
    total_start_time = time.time()
    annotation_found_count = 0
    annotation_not_found_count = 0

    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_name}")

        img_start_time = time.time()

        # 画像読み込み
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Warning: Failed to load image, skipping...")
            continue

        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h}")

        # タイル生成
        tiles = generate_tiles(image, args.tile_size, args.overlap)
        print(f"  Generated {len(tiles)} tiles")

        # 各タイルで推論（確率マップを取得）
        tile_probs_list = []
        for tile_idx, tile_info in enumerate(tiles):
            print(f"\r  Inferring tile {tile_idx+1}/{len(tiles)}...", end='', flush=True)

            # 確率マップ取得（evaluate_teeth_single.pyと同じパイプライン処理）
            tile_probs = inference_tile_with_probs(model, tile_info['tile_img'])
            tile_probs_list.append(tile_probs)

            # メモリクリア（CUDA使用時のみ）
            if (tile_idx + 1) % 10 == 0 and torch.cuda.is_available() and 'cuda' in args.device:
                torch.cuda.synchronize()  # GPU操作の完了を待機（race condition防止）
                torch.cuda.empty_cache()
                print(f" [GPU cache cleared at tile {tile_idx+1}]", end='', flush=True)

        print()  # 改行

        # タイル結果を確率ベースで結合（医療画像では偽陰性を最小化）
        print("  Merging tiles with probability averaging...")
        final_pred = merge_tiles_with_probabilities(tiles, tile_probs_list, (h, w, 3), num_classes)

        # カラーマップ適用（evaluate_teeth_single.pyと同じ方式）
        pimg = np.zeros((h, w, 3), dtype=np.uint8)
        for id, cols in enumerate(joint_label_map_value_with_bg):
            imask = (final_pred == id)
            pimg[imask] = cols[0]

        # アノテーション画像の読み込み（--annotationが指定されている場合）
        gt_mask = load_annotation_image(args.annotation, img_name, joint_cls_def_json, verbose=True)

        # GTマスクをBGR画像に変換（推論結果と同じカラーマップを使用）
        gimg = None
        if gt_mask is not None:
            annotation_found_count += 1
            gimg = np.zeros((h, w, 3), dtype=np.uint8)
            for id, cols in enumerate(joint_label_map_value_with_bg):
                imask = (gt_mask == id)
                gimg[imask] = cols[0]
        else:
            annotation_not_found_count += 1

        # 可視化（evaluate_teeth_single.pyの方式を使用）
        create_visualization(args, img_path, image, pimg, joint_label_map_value, gimg)

        # 予測マップも保存
        pred_path = os.path.join(args.out_dir, f"{os.path.splitext(img_name)[0]}_pred.png")
        cv2.imwrite(pred_path, pimg)

        img_elapsed = time.time() - img_start_time
        print(f"  Time: {img_elapsed:.2f}s")
        print(f"  Saved to: {args.out_dir}")

    # 処理時間サマリー
    total_elapsed = time.time() - total_start_time
    avg_time = total_elapsed / len(image_files) if len(image_files) > 0 else 0

    print(f"\n=== Summary ===")
    print(f"Total images processed: {len(image_files)}")
    if args.annotation:
        print(f"Annotations found: {annotation_found_count}")
        print(f"Annotations not found: {annotation_not_found_count}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average time per image: {avg_time:.2f}s")
    print(f"Output directory: {args.out_dir}")


if __name__ == '__main__':
    main()
