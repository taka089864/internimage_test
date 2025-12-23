# 環境設定
開発を行った環境は下記になります。  
python: 3.11.14  
CUDA: 12.9  
## mmcv
```
cd mmcv

export CUDA_HOME=/usr/local/cuda-12.9   # あなたの nvcc に合わせる
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=12.0
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

# editable install(コードの修正がすぐに反映される)
pip install -v -e . --no-build-isolation
```
## mmdet
```
pip install -U openmim
mim install mmengine                     # 要件: >=0.7.1, <1.0.0
pip install -U "mmdet==3.3.0"            # or: mim install mmdet==3.3.0
```

## mmsegmentationのインストール
```
pip install "mmsegmentation>=1.0.0"
```

## PyTorchのインストール
下記ページを参照し、v2.8.0、CUDAは環境に合わせて選択してください  
https://pytorch.org/get-started/previous-versions/

LINUX, CUDA 12.8の場合の例
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## DCNv3のビルド
```
cd segmentation/ops_dcnv3

# 環境変数設定（mmcvと同じ設定を使用）
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ビルド
./make.sh
# または python setup.py install
```

## その他のモジュール
```
pip install -r requirements.txt
```


# コードについて
## inference
### infer_teeth_tiled.py

大きな画像をタイル分割して推論するスクリプトです。メモリ制約のある環境で高解像度画像を処理できます。

#### 基本的な使い方

```bash
cd segmentation

python infer_teeth_tiled.py \
  --config configs/medical_shift/internimage_xl_640_teeth_single.py \
  --checkpoint ../results/042_caries_seg_2025.10.01_03_all_internimage/train/best_mDice_iter_35000.pth \
  --input_dir /path/to/input/images \
  --out_dir ./inference_output
```

#### 主要なオプション

| オプション | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--config` | ✅ | - | モデル設定ファイル |
| `--checkpoint` | ✅ | - | 学習済みモデルのチェックポイント |
| `--input_dir` | ✅ | - | 入力画像のディレクトリ |
| `--out_dir` | | `tiled_inference_output` | 出力ディレクトリ |
| `--tile_size` | | 640 | タイルサイズ（画像分割の単位） |
| `--overlap` | | 0.2 | 隣接タイル間のオーバーラップ比率 (0-1) |
| `--device` | | `cuda:0` | 使用するデバイス |
| `--annotation` | | None | Ground Truthアノテーションディレクトリ（複数指定可） |

- configファイルについて
  - large model: internimage_l_640_teeth_single.py
  - xlarge model: internimage_xl_640_teeth_single.py
  - huge model: internimage_h_896_teeth_single.py

- checkpointファイルについて
  - 下記の資料を参考にして使用したいモデルのNoを確認する
    - https://docs.google.com/spreadsheets/d/1aqN-dNW5HF5KcdLBUGwPSjQdYgymq0HphxjzXQcOMos/edit?pli=1&gid=1985095465#gid=1985095465
  - 使用するモデルが決まったら、下記のURLよりchedkpointファイルをダウンロードする
    - https://drive.google.com/drive/folders/1bJlE2cTsCd1NfrxzwnFLOMquROtjw6Jf?usp=drive_link
    - (ex) 042のcheckpointsファイルを使用したい場合
      - 042/train/iter_150000.pthをダウンロード
      - https://drive.google.com/drive/folders/1sq4j42XYHNP7WDHDOq0GUFaLKVRW8wGi?usp=drive_link

#### 使用例

**1. 基本的な推論（タイルサイズ640, オーバーラップ20%）**
```bash
python infer_teeth_tiled.py \
  --config configs/medical_shift/internimage_l_640_teeth_single.py \
  --checkpoint ../results/train/best_mDice_iter_50000.pth \
  --input_dir path/.darwin/medicalshift/pb-rf-bubaigawara/infer_test_images \
  --out_dir ./output_640_overlap20
```

**2. タイルサイズとオーバーラップをカスタマイズ**
```bash
python infer_teeth_tiled.py \
  --config configs/medical_shift/internimage_xl_640_teeth_single.py \
  --checkpoint ../results/train/best_mDice_iter_80000.pth \
  --input_dir /path/to/images \
  --out_dir ./output_896_overlap30 \
  --tile_size 896 \
  --overlap 0.3
```

**3. アノテーション（GT）との比較表示**
```bash
python infer_teeth_tiled.py \
  --config configs/medical_shift/internimage_l_640_teeth_single.py \
  --checkpoint ../results/train/best_mDice_iter_50000.pth \
  --input_dir path/.darwin/medicalshift/pb-rf-bubaigawara/infer_test_images \
  --out_dir ./output_with_gt \
  --annotation path/.darwin/medicalshift/infer_standard_B/annotations \
              path/.darwin/medicalshift/pb-rf-bubaigawara/releases/2025.10.01_01/annotations
```

**4. 特定のGPUを指定**
```bash
CUDA_VISIBLE_DEVICES=1 python infer_teeth_tiled.py \
  --config configs/medical_shift/internimage_h_896_teeth_single.py \
  --checkpoint ../results/train/best_mDice_iter_100000.pth \
  --input_dir /path/to/images \
  --out_dir ./output_gpu1 \
  --device cuda:0
```

#### 出力ファイル

出力ディレクトリには以下が生成されます：

```
output_dir/
├── PBN546_201214100039_pred.png      # 予測結果の画像ファイル
├── PBN546_201214100332_pred.png
├── PBN553_201214153521_pred.png
├── ...
└── vis/                               # 可視化画像（4パネルレイアウト）
    ├── PBN546_201214100039_combined.jpg
    ├── PBN546_201214100332_combined.jpg
    ├── PBN553_201214153521_combined.jpg
    └── ...
```

**可視化画像（`vis/*_combined.jpg`）のレイアウト**:
```
┌─────────────┬─────────────┐
│ GT Overlay  │   GT Mask   │  ← Ground Truth（--annotation指定時）
├─────────────┼─────────────┤
│ Pred Overlay│  Pred Mask  │  ← 予測結果
└─────────────┴─────────────┘
```

#### 注意事項

- `--tile_size`はモデルの入力サイズ以下にする必要があります
- `--overlap`を大きくすると境界での予測が改善されますが、処理時間が増加します
- 複数の`--annotation`ディレクトリを指定すると、最初に見つかったアノテーションが使用されます

## evaluate
### evaluate_teeth_single.py

学習済みモデルの性能を評価するスクリプトです。テストデータに対して推論を実行し、IoU、Dice、Precision、Recallなどの評価指標を計算します。

#### 基本的な使い方

```bash
cd segmentation

python evaluate_teeth_single.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  ../results/042_caries_seg_2025.10.01_03_all_internimage/train/iter_150000.pth \
  --data_list /path/to/test.list \
  --base_path /path/to/dataset \
  --out_dir ./eval_results
```

#### 主要なオプション

| オプション | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `config` | ✅ | - | モデル設定ファイル（位置引数） |
| `checkpoint` | ✅ | - | 評価するチェックポイント（位置引数） |
| `--data_list` | ✅ | - | 評価対象画像のリストファイル (.list) |
| `--base_path` | ✅ | - | データセットのベースパス |
| `--out_dir` | | `demo` | 評価結果の出力ディレクトリ |
| `--device` | | `cuda:0` | 使用するデバイス |

#### 使用例

**1. 基本的な評価（Large モデル）**
```bash
python evaluate_teeth_single.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  ../results/042_caries_seg_2025.10.01_03_all_internimage/train/iter_150000.pth \
  --data_list path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02/valid.list \
  --base_path path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02 \
  --out_dir ./eval_042_iter150k
```

**2. XLarge モデルの評価**
```bash
python evaluate_teeth_single.py \
  configs/medical_shift/internimage_xl_640_teeth_single.py \
  ../results/043_caries_seg_2025.10.01_03_all_internimage/train/best_mDice_iter_80000.pth \
  --data_list path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02/valid.list \
  --base_path path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02 \
  --out_dir ./eval_043_xl_best
```

**3. Huge モデルの評価（特定のGPU使用）**
```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_teeth_single.py \
  configs/medical_shift/internimage_h_896_teeth_single.py \
  ../results/044_caries_seg_2025.10.01_03_all_internimage/train/iter_100000.pth \
  --data_list path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02/test.list \
  --base_path path/dataset/medical_shift/caries_seg_2025.10.01_03_all_internimage_02 \
  --out_dir ./eval_044_h_test \
  --device cuda:0
```

#### 出力ファイル

評価完了後、以下のファイルが生成されます：

```
out_dir/
├── vis/                                          # 画像ごとの可視化（4パネルレイアウト）
│   ├── image001_combined.jpg
│   ├── image002_combined.jpg
│   └── ...
│
├── eval.csv                                      # 全体評価結果サマリ
├── conf.csv                                      # 混同行列データ
├── per_image_metrics.csv                         # 画像ごとの詳細評価指標
│
├── statistics_summary_table.png                  # 統計サマリテーブル（可視化）
│
├── cdf_dice.png                                  # Dice係数の累積分布関数
├── cdf_iou.png                                   # IoUの累積分布関数
├── cdf_precision.png                             # Precisionの累積分布関数
├── cdf_recall.png                                # Recallの累積分布関数
│
├── histogram_dice.png                            # Diceヒストグラム
├── histogram_iou.png                             # IoUヒストグラム
├── histogram_precision.png                       # Precisionヒストグラム
├── histogram_recall.png                          # Recallヒストグラム
│
├── scatter_matrix_bg.png                         # 背景クラスの散布図行列
├── scatter_matrix_A1_tai_caries_confirmed.png   # 確定う蝕の散布図行列
└── scatter_matrix_A1_tai_caries_suspect.png     # 疑いう蝕の散布図行列
```

#### 出力ファイルの詳細

**CSVファイル**:
- `eval.csv`: クラスごとの平均IoU、Dice、Precision、Recallなど全体サマリ
- `conf.csv`: 混同行列（クラス間の予測と正解の対応関係）
- `per_image_metrics.csv`: 画像ごとの詳細評価指標

**統計可視化**:
- `statistics_summary_table.png`: 評価指標の統計サマリテーブル（平均、中央値、標準偏差など）

**分布の可視化**:
- `cdf_*.png`: 各指標の累積分布関数（CDF）グラフ
- `histogram_*.png`: 各指標のヒストグラム

**散布図行列**:
- `scatter_matrix_*.png`: クラスごとの4指標（IoU、Dice、Precision、Recall）の相関を示す散布図行列

#### 評価指標

以下の指標がクラスごとに計算されます：

- **IoU (Intersection over Union)**: 予測と正解の重なり度合い
- **Dice係数**: F1スコアに相当する指標
- **Precision**: 予測したピクセルのうち正解の割合
- **Recall**: 正解ピクセルのうち検出できた割合

#### data_listファイルの形式

```
# 各行に画像ファイル名とアノテーションファイル名を記載
# 形式: 画像ファイル アノテーションファイル（または画像ファイルのみ）
image001.jpg
image002.jpg
image003.jpg
```

#### 注意事項

- `--data_list`で指定するファイルには、評価対象の画像ファイル名を1行ずつ記載します
- `--base_path`には、画像とアノテーションが格納されているディレクトリを指定します
- 評価には対応するGround Truthアノテーションが必要です

## train
### train_for_teeth_single_head.py

モデルを学習するスクリプトです。設定ファイルに基づいて学習を実行し、チェックポイントとログを保存します。

#### 基本的な使い方

```bash
cd segmentation

python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  --work-dir ../results/042_caries_seg_2025.10.01_03_all_internimage/train \
  --gpu-id 0
```

#### 主要なオプション

| オプション | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `config` | ✅ | - | 学習設定ファイル（位置引数） |
| `--work-dir` | | 設定ファイル内 | ログとチェックポイントの保存先 |
| `--load-from` | | None | 初期化用チェックポイント（事前学習済みモデル） |
| `--resume-from` | | None | 中断した学習の再開用チェックポイント |
| `--gpu-id` | | 0 | 使用するGPU ID |
| `--no-validate` | | False | 検証を無効化（学習のみ実行） |
| `--options` | | - | 設定を動的にオーバーライド |

#### 使用例

**1. 新規学習（Large モデル）**
```bash
python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  --work-dir ../results/042_caries_seg_2025.10.01_03_all_internimage/train \
  --gpu-id 0
```

**2. XLarge モデルの学習**
```bash
CUDA_VISIBLE_DEVICES=0 python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_xl_640_teeth_single.py \
  --work-dir ../results/043_caries_seg_2025.10.01_03_all_internimage/train \
  --gpu-id 0
```

**3. Huge モデルの学習（メモリ最適化）**
```bash
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256 \
CUDA_VISIBLE_DEVICES=0 python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_h_896_teeth_single.py \
  --work-dir ../results/044_caries_seg_2025.10.01_03_all_internimage/train \
  --gpu-id 0
```

**4. 学習の再開（resume）**
```bash
python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  --work-dir ../results/042_caries_seg_2025.10.01_03_all_internimage/train \
  --resume-from ../results/042_caries_seg_2025.10.01_03_all_internimage/train/iter_35000.pth \
  --gpu-id 0
```

**5. 設定の動的オーバーライド（バッチサイズ変更）**
```bash
python train_for_teeth_single_head.py \
  configs/medical_shift/internimage_l_640_teeth_single.py \
  --work-dir ../results/042_caries_seg_2025.10.01_03_all_internimage/train \
  --gpu-id 0 \
  --options train_dataloader.batch_size=8
```

#### 出力ファイル

学習中、work-dirに以下が生成されます：

```
work_dir/
├── {config_name}.py              # 使用した設定ファイルのコピー
├── iter_1000.pth                 # 定期的なチェックポイント
├── iter_2000.pth
├── ...
├── best_mDice_iter_XXXXX.pth     # 最良のDiceスコアモデル
├── latest.pth                    # 最新のチェックポイント（シンボリックリンク）
│
├── {timestamp}.log               # 学習ログ
│
├── vis_data/                     # TensorBoard用ログ
│   └── events.out.tfevents.*
│
└── tf_logs/                      # TensorBoard追加ログ
    └── ...
```

#### TensorBoardでの学習監視

```bash
tensorboard --logdir=../results/042_caries_seg_2025.10.01_03_all_internimage/train/vis_data
```

ブラウザで `http://localhost:6006` にアクセスして学習曲線を確認できます。

#### 学習設定について

各モデルサイズに対応する設定ファイル：

- **Large**: `configs/medical_shift/internimage_l_640_teeth_single.py`
  - パラメータ: 223M
  - 入力サイズ: 640×640
  - 推奨batch_size: 12

- **XLarge**: `configs/medical_shift/internimage_xl_640_teeth_single.py`
  - パラメータ: 368M
  - 入力サイズ: 640×640
  - 推奨batch_size: 12

- **Huge**: `configs/medical_shift/internimage_h_896_teeth_single.py`
  - パラメータ: 1.12B
  - 入力サイズ: 896×896
  - 推奨batch_size: 1（with_cp=True必須）

#### 注意事項

- **resume vs load-from**:
  - `--resume-from`: 学習を中断した地点から再開（optimizer状態も復元）
  - `--load-from`: モデルの重みのみ読み込み（新規学習の初期化用）

- **Huge モデルの場合**:
  - VRAM不足を避けるため`PYTORCH_CUDA_ALLOC_CONF`の設定を推奨
  - `with_cp=True`（gradient checkpointing）が必須
  - batch_size=1を推奨

- **学習の中断**:
  - Ctrl+C で安全に中断可能
  - 最新のチェックポイントから`--resume-from`で再開

- **検証の無効化**:
  - `--no-validate`を指定すると学習のみ実行（検証をスキップして高速化）

## 設定ファイルについて

### 設定ファイルの構成

InternImageの学習では、MMEngine/MMSegmentationの設定システムを使用しています。設定ファイルは階層的な継承構造を持ち、基本設定を再利用しながらモデル固有のパラメータを定義できます。

#### 設定ファイルの関係

```
configs/medical_shift/internimage_l_640_teeth_single.py  ← モデル固有設定
├─ _base_ = [
│   ├─ '../_base_/datasets/teeth_single.py'              ← データセット設定（継承）
│   ├─ '../_base_/default_runtime.py'                    ← 実行環境設定（継承）
│   └─ '../_base_/schedules/schedule_80k.py'             ← スケジューラ設定（継承）
│  ]
└─ モデル固有のパラメータ（backbone, decode_head, optimizer, etc.）
```

#### 主要な設定ファイル

| ファイル | 役割 | 主な内容 |
|---------|------|---------|
| `medical_shift/internimage_l_640_teeth_single.py` | Large モデル設定 | モデルアーキテクチャ、最適化、学習パラメータ |
| `medical_shift/internimage_xl_640_teeth_single.py` | XLarge モデル設定 | XL固有のアーキテクチャパラメータ |
| `medical_shift/internimage_h_896_teeth_single.py` | Huge モデル設定 | H固有のアーキテクチャ、大モデル用最適化 |
| `_base_/datasets/teeth_single.py` | データセット設定 | データパス、前処理パイプライン、train/valid/test分割 |
| `_base_/default_runtime.py` | 実行環境設定 | ログ、チェックポイント、分散学習設定 |
| `_base_/schedules/schedule_80k.py` | スケジューラ設定 | 学習率スケジュール、epoch/iteration設定 |

### データセット設定（teeth_single.py）

#### データパスの指定

`configs/_base_/datasets/teeth_single.py`では、以下の3つの要素でデータセットを定義します：

```python
# データセットのルートディレクトリ
data_root = 'dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage'

# アノテーション用のカラーマッピングファイル
class_color_json = 'dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage/teeth_colors.json'
caries_color_json = 'dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage/caries_colors.json'

# train/valid/testデータのリストファイル
data = dict(
    train=dict(
        type='TeethSingleDataset',
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage/train.list",
        pipeline=train_pipeline),

    val=dict(
        type='TeethSingleDataset',
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage/valid.list",
        pipeline=test_pipeline),

    test=dict(
        type='TeethSingleDataset',
        data_root=data_root,
        class_color_json=class_color_json,
        caries_color_json=caries_color_json,
        data_list="dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage/valid.list",
        pipeline=test_pipeline)
)
```

#### データリストファイル（.list）の形式

`train.list`, `valid.list`, `test.list`の各ファイルには、学習・検証・テストに使用する画像のファイル名を1行ずつ記載します：

```
# train.list の例
PBN546_201214100039.jpg
PBN546_201214100332.jpg
PBN553_201214153521.jpg
...
```

#### データセットのディレクトリ構造

```
data_root/
├── images/                          # 画像ファイル
│   ├── PBN546_201214100039.jpg
│   ├── PBN546_201214100332.jpg
│   └── ...
│
├── annotations/                     # アノテーションファイル
│   ├── PBN546_201214100039.png     # カラーマスク画像
│   ├── PBN546_201214100332.png
│   └── ...
│
├── train.list                       # 学習データリスト
├── valid.list                       # 検証データリスト
├── test.list                        # テストデータリスト（オプション）
│
├── teeth_colors.json                # 歯のクラスカラーマッピング
└── caries_colors.json               # う蝕のクラスカラーマッピング
```

### データセットの変更方法

#### 基本設定ファイルを直接編集

`configs/_base_/datasets/teeth_single.py`を編集してデータパスを変更：

```python
# 既存のdata_rootをコメントアウトし、新しいパスを追加
# data_root = 'dataset/medical_shift/caries_seg_2025.11.07_01_all_internimage'
data_root = '/path/to/your/new/dataset'

# 対応するカラーマッピングファイルも変更
class_color_json = '/path/to/your/new/dataset/teeth_colors.json'
caries_color_json = '/path/to/your/new/dataset/caries_colors.json'
```
