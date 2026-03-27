# M2F デコーダー実験ガイド

**作成日**: 2026-03-27
**前提資料**: [bepli_v2_segmentation_report.md](./bepli_v2_segmentation_report.md) のセクション 8〜11

---

## 目次

1. [背景と目的](#1-背景と目的)
2. [事前学習済み重みの取得](#2-事前学習済み重みの取得)
3. [環境セットアップ](#3-環境セットアップ)
4. [実験計画](#4-実験計画)
5. [実行コマンド](#5-実行コマンド)
6. [設定パラメータの解説](#6-設定パラメータの解説)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 背景と目的

前回の実験（Linear Head, mIoU 1.8%）の失敗分析から、**デコーダーの表現力不足が主因**であると判明した（詳細は引き継ぎ資料セクション 8〜9 参照）。

本ガイドでは、M2F (Mask2Former) デコーダーへの変更を軸とした改善実験の実行手順をまとめる。

### 実験の優先順位

| 順番 | 実験名 | バックボーン | デコーダー | 目的 |
|------|--------|------------|-----------|------|
| 1 | step4_m2f | ViT-B/16 | M2F | デコーダー強化の効果確認 |
| 2 | step5_convnext_m2f | ConvNeXt Small | M2F | バックボーン比較 |
| 3 | step6_convnext_m2f_finetune | ConvNeXt Small | M2F | 部分 Fine-tuning の効果確認 |

step4 で改善が確認できたら step5 に進む。step4 の時点で結果が芳しくない場合は、先に損失関数やハイパーパラメータの調整を検討する。

---

## 2. 事前学習済み重みの取得

### 2.1 Meta への申請

DINOv3 の重みは **Meta のライセンス承認が必要**。

1. **申請ページ**: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. ライセンス契約（`LICENSE.md` 参照）に同意してフォームを送信
3. **承認後、メールでダウンロード URL 一覧が届く**
4. メール記載の URL から `wget` でダウンロードする（ブラウザではなく `wget` を推奨）

> **注意**: 前回の実験で取得した重みはアーカイブ済みのため、再度申請が必要。

### 2.2 ダウンロード方法

申請承認メールに記載された URL を使用する。メール内の URL はユーザー固有の認証トークン付きの場合がある。

```bash
# メール記載の URL を使用してダウンロード（wget 推奨）
wget -O dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth "<メールに記載されたURL>"
```

> **重要**: 以下に記載するファイル名は、コード内の hash 値から特定した正確なファイル名である。メールに記載される URL のパスが異なる場合でも、ダウンロード後にこのファイル名でリネームすれば動作する。

### 2.3 必要な重みファイル

**step4 用（必須）:**

| モデル | ファイル名 | hub 名 |
|--------|-----------|--------|
| ViT-B/16 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | `dinov3_vitb16` |

**step5 用（ConvNeXt、推奨は Small）:**

| モデル | ファイル名 | hub 名 |
|--------|-----------|--------|
| ConvNeXt Small (50M) | `dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth` | `dinov3_convnext_small` |
| ConvNeXt Tiny (29M) | `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` | `dinov3_convnext_tiny` |

### 2.4 代替手段: HuggingFace Hub

Meta 申請の代わりに、HuggingFace Hub からもモデルを取得できる可能性がある。

**HuggingFace コレクション**: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009

| モデル | HuggingFace モデル ID |
|--------|----------------------|
| ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| ConvNeXt Tiny | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` |
| ConvNeXt Small | `facebook/dinov3-convnext-small-pretrain-lvd1689m` |
| ConvNeXt Base | `facebook/dinov3-convnext-base-pretrain-lvd1689m` |
| ConvNeXt Large | `facebook/dinov3-convnext-large-pretrain-lvd1689m` |

HuggingFace から重みをダウンロードしてローカルに配置する場合:

```bash
# pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/dinov3-vitb16-pretrain-lvd1689m',
    filename='dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    local_dir='./weights'
)
"
```

> **注意**: HuggingFace のモデルもライセンス承認（Gated Model）が必要な場合がある。その場合は HuggingFace アカウントでログインし、モデルページで承認後にダウンロードする。

### 2.5 全モデル一覧（参考）

| モデル | ファイル名 | hash | サイズ目安 |
|--------|-----------|------|-----------|
| ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | `08c60483` | - |
| ViT-B/16 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | `73cec8be` | ~334 MB |
| ViT-L/16 | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `8aa4cbdd` | - |
| ViT-L/16 (SAT) | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | `eadcf0ff` | - |
| ViT-7B/16 | `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` | `a955f4ea` | - |
| ConvNeXt Tiny | `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` | `21b726bb` | ~112 MB |
| ConvNeXt Small | `dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth` | `296db49d` | ~192 MB |
| ConvNeXt Base | `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` | `801f2ba9` | ~340 MB |
| ConvNeXt Large | `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` | `61fa432d` | ~756 MB |

> ファイル名と hash はコード（`dinov3/hub/backbones.py`）から抽出した正確な値。

### 2.6 サーバーへの配置

ダウンロードしたファイルをサーバーの `weights/` ディレクトリに配置する:

```
dinov3-finetune/
  weights/
    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth           # step4 用
    dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth    # step5 用
```

> **注意**: 学習サーバーから外部ネットワークへのアクセスが制限されているため、ローカル PC 等でダウンロードした後に `scp` 等で転送する必要がある。

---

## 3. 環境セットアップ

### 3.1 追加パッケージのインストール

```bash
pip install torchmetrics
```

### 3.2 Deformable Attention CUDA ops（必要な場合のみ）

M2F の Multi-Scale Deformable Attention は CUDA カスタムオペレータを使用する。実行時にエラーが出た場合のみビルドする:

```bash
cd dinov3/eval/segmentation/models/utils/ops
pip install -e .
cd -
```

### 3.3 データセットの確認

bepli_v2 データセットが以下の構造で配置されていることを確認:

```
/data4/hirawatas/bepli_coco/
  train/
    _annotations.coco.json
    images/
      *.jpg
  val/
    _annotations.coco.json
    images/
      *.jpg
```

---

## 4. 実験計画

### step4: ViT-B/16 + M2F（デコーダー強化の効果確認）

前回実験（Linear Head, mIoU 1.8%）との比較が目的。バックボーンは同じ ViT-B/16 を使い、デコーダーのみを M2F に変更する。

- **期待される改善**: mIoU 10〜30% 程度（空間的文脈 + マルチスケール特徴の効果）
- **注意**: 実用水準 (50%+) に届くかは不確実。bepli_v2 のデブリは特徴マップ上で 1 ピクセル未満の極小物体が多い

### step5: ConvNeXt Small + M2F（バックボーン比較）

ConvNeXt はネイティブなマルチスケール特徴を出力する畳み込みベースのモデル。小データ・小物体への帰納バイアスが ViT より有利。

- **注意**: M2F の DINOv3_Adapter は ViT を前提に設計されている。ConvNeXt では adapter を使わずに直接マルチスケール特徴を渡す必要がある可能性がある。step4 の結果を見てから追加対応を検討する。

---

## 5. 実行コマンド

### step4: ViT-B/16 + M2F

```bash
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step4_m2f \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/data4/hirawatas/bepli_coco
```

### step4 (VRAM 不足の場合): バッチサイズ縮小

```bash
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step4_m2f \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/data4/hirawatas/bepli_coco \
  bs=1
```

### step5: ConvNeXt Small + M2F

```bash
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step5_convnext_m2f \
  model.dino_hub=dinov3_convnext_small \
  model.dino_hub_weights=./weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth \
  datasets.root=/data4/hirawatas/bepli_coco
```

### step5 代替: ConvNeXt Tiny + M2F（より軽量）

```bash
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step5_convnext_tiny_m2f \
  model.dino_hub=dinov3_convnext_tiny \
  model.dino_hub_weights=./weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
  datasets.root=/data4/hirawatas/bepli_coco
```

### 複数 GPU で実行する場合

```bash
# 例: 3 GPU
torchrun --nproc_per_node=3 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step4_m2f \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/data4/hirawatas/bepli_coco \
  n_gpus=3
```

---

## 6. 設定パラメータの解説

設定ファイル: `dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml`

### 主要パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `decoder_head.type` | `m2f` | Mask2Former デコーダーを使用 |
| `decoder_head.backbone_out_layers` | `FOUR_EVEN_INTERVALS` | ViT-B/16 の場合 [2,5,8,11] 層を抽出 |
| `decoder_head.hidden_dim` | `256` | M2F の隠れ次元。デフォルト 2048 から縮小してメモリ節約 |
| `decoder_head.num_classes` | `12` | bepli_v2: 背景 + 11 デブリ種別 |
| `bs` | `2` | GPU あたりのバッチサイズ。VRAM 不足なら 1 に |
| `optimizer.lr` | `1e-4` | Linear Head (1e-3) より低い。M2F は学習パラメータが多いため |
| `scheduler.total_iter` | `20000` | 総学習ステップ数 |
| `eval.eval_interval` | `2000` | バリデーション間隔 |
| `train.diceloss_weight` | `0.5` | Dice Loss の重み（前回実験では未使用） |
| `train.celoss_weight` | `1.0` | CE Loss の重み |
| `train.class_weight` | `[0.1, 0.60, ...]` | クラスごとの重み（背景抑制） |

### CLI でのパラメータ上書き

config ファイルのパラメータは CLI で上書き可能:

```bash
# 例: 学習率とステップ数を変更
torchrun ... \
  optimizer.lr=5e-5 \
  scheduler.total_iter=40000 \
  eval.eval_interval=5000
```

---

## 7. トラブルシューティング

### `ModuleNotFoundError: No module named 'torchmetrics'`

```bash
pip install torchmetrics
```

### `ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'`

Deformable Attention の CUDA ops が未ビルド:

```bash
cd dinov3/eval/segmentation/models/utils/ops
pip install -e .
cd -
```

CUDA バージョンが合わない場合は、PyTorch の純粋 Python 実装にフォールバックされる可能性がある。`ms_deform_attn.py` 内のエラーメッセージを確認。

### CUDA out of memory

1. `bs=1` に変更
2. `decoder_head.hidden_dim=128` に縮小
3. `transforms.train.img_size=384` + `eval.crop_size=384` で入力解像度を下げる

```bash
torchrun ... bs=1 decoder_head.hidden_dim=128
```

### ConvNeXt + M2F でエラーが出る場合

M2F の DINOv3_Adapter は ViT の `get_intermediate_layers(return_class_token=True)` を前提としている。ConvNeXt の `get_intermediate_layers` も同じインターフェースを実装しているが、内部の特徴マップ構造が異なるため、adapter の forward でエラーが出る可能性がある。

その場合は、ConvNeXt + Linear Head（`decoder_head.type=linear`）で先にバックボーンの効果を確認する:

```bash
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
  output_dir=./outputs/step5_convnext_linear \
  model.dino_hub=dinov3_convnext_small \
  model.dino_hub_weights=./weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth \
  datasets.root=/data4/hirawatas/bepli_coco \
  decoder_head.backbone_out_layers=LAST \
  scheduler.total_iter=5000 \
  eval.eval_interval=1000
```
