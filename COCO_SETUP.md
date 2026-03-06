# DINOv3 COCO Segmentation Setup

このドキュメントでは、DINOv3を使用してCOCO形式データセットでセマンティックセグメンテーションを実行する方法を説明します。

## 📋 追加された機能

公式DINOv3リポジトリに以下を追加しました：

1. **COCOセグメンテーションデータローダー**
   - ファイル: `dinov3/data/datasets/coco_segmentation.py`
   - COCO形式のインスタンスアノテーションをセマンティックマスクに変換
   - ポリゴンとRLE形式の両方に対応

2. **COCO用設定ファイル**
   - ファイル: `dinov3/eval/segmentation/configs/config-coco-linear-training.yaml`
   - 81クラス（80オブジェクト + 背景）
   - 線形プローブ（Linear Head）を使用

## 🚀 セットアップ手順

### 1. 環境構築

```bash
# 依存関係のインストール
cd /Users/sou/workspace/dinov3_official
pip install -e .
pip install pycocotools
```

### 2. COCOデータセットの準備

COCO 2017データセットをダウンロードして配置：

```
your_data_dir/
├── train2017/          # トレーニング画像
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017/            # 検証画像
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
└── annotations/        # アノテーション
    ├── instances_train2017.json
    └── instances_val2017.json
```

**ダウンロード方法:**
```bash
# COCO 2017をダウンロード
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解凍
unzip train2017.zip -d /path/to/your_data_dir/
unzip val2017.zip -d /path/to/your_data_dir/
unzip annotations_trainval2017.zip -d /path/to/your_data_dir/
```

### 3. 設定ファイルの編集

`dinov3/eval/segmentation/configs/config-coco-linear-training.yaml` を編集：

```yaml
datasets:
  root: "/path/to/your_data_dir"  # COCOデータセットのパス
  train: "COCOSegmentation:split=TRAIN"
  val: "COCOSegmentation:split=VAL"
```

### 4. トレーニング実行

```bash
# 単一ノード、8 GPU
python -m dinov3.run.submit \
    dinov3/eval/segmentation/run.py \
    --config dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
    --backbone dinov3_convnext_small \
    --output-dir ./outputs/coco_segmentation
```

## 📊 モデル選択

### バックボーン

| モデル | パラメータ | 推奨用途 |
|--------|-----------|---------|
| `dinov3_convnext_tiny` | 29M | 小規模データセット、高速推論 |
| `dinov3_convnext_small` | 50M | **推奨**（バランス） |
| `dinov3_convnext_base` | 89M | 高精度要求 |
| `dinov3_convnext_large` | 198M | 最高精度 |

```bash
# 異なるバックボーンを使用
python -m dinov3.run.submit \
    dinov3/eval/segmentation/run.py \
    --config config-coco-linear-training.yaml \
    --backbone dinov3_convnext_base
```

## 🔧 設定のカスタマイズ

### ハイパーパラメータ調整

```yaml
# config-coco-linear-training.yaml

# バッチサイズ（GPU1台あたり）
bs: 4

# GPU数
n_gpus: 8

# 学習率
optimizer:
  lr: 1e-3  # 必要に応じて調整

# トレーニング反復回数
scheduler:
  total_iter: 40000

# デコーダー設定
decoder_head:
  type: "linear"  # 軽量な線形プローブ
  num_classes: 81  # COCO: 80 + 背景
```

### 小規模データセット向け設定

データセットが3000枚程度の場合：

```yaml
scheduler:
  total_iter: 10000  # 反復回数を減らす
  constructor_kwargs:
    warmup_iters: 500

optimizer:
  lr: 5e-4  # 学習率を下げる
  weight_decay: 5e-4
```

## 📈 評価

```bash
# 学習済みモデルで評価
python -m dinov3.eval.segmentation.eval \
    --config config-coco-linear-training.yaml \
    --checkpoint ./outputs/coco_segmentation/checkpoint_best.pth \
    --backbone dinov3_convnext_small
```

## 🎯 期待される結果

### Linear Head（線形プローブ）

- **パラメータ数**: 最小（デコーダーのみ学習）
- **トレーニング時間**: 短い
- **mIoU**: 30-40%程度（COCOは難しいデータセット）

### カスタムデータセット

3000枚程度の小規模データセットの場合：
- 過学習を避けるため、線形プローブ推奨
- データ拡張を強化
- Early stopping使用

## 🔍 トラブルシューティング

### 問題: `FileNotFoundError: Annotation file not found`

**解決策:**
```bash
# ディレクトリ構造を確認
ls -la /path/to/your_data_dir/
# 以下が存在することを確認：
#   annotations/instances_train2017.json
#   annotations/instances_val2017.json
#   train2017/
#   val2017/
```

### 問題: `ImportError: No module named 'pycocotools'`

**解決策:**
```bash
pip install pycocotools
```

### 問題: GPU メモリ不足

**解決策:**
```yaml
# config.yaml
bs: 2  # バッチサイズを減らす
```

### 問題: クラス数のミスマッチ

COCOは80のオブジェクトクラス + 背景 = 81クラスです。

```yaml
decoder_head:
  num_classes: 81  # 確認
```

## 📚 参考

- [公式DINOv3リポジトリ](https://github.com/facebookresearch/dinov3)
- [COCO Dataset](https://cocodataset.org/)
- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)

## 🆚 ADE20K vs COCO

| 特徴 | ADE20K | COCO |
|------|--------|------|
| クラス数 | 150 | 81 (80+1) |
| トレーニング画像 | 20,210 | 118,287 |
| 検証画像 | 2,000 | 5,000 |
| アノテーション形式 | PNG mask | JSON (polygon/RLE) |
| タスク | Scene parsing | Object segmentation |

## 💡 ベストプラクティス

1. **小規模データセット（<5000枚）**
   - 線形プローブ（Linear Head）を使用
   - バックボーンを凍結
   - データ拡張を強化

2. **大規模データセット（>50000枚）**
   - Mask2Former Headの使用を検討
   - バックボーンのファインチューニングも可能

3. **本番環境**
   - モデルを事前ダウンロード
   - Mixed precision (FP16) を使用
   - 分散学習で高速化

---

**作成日**: 2026-03-06
**ベース**: 公式DINOv3リポジトリ
**追加**: COCOセグメンテーション対応
