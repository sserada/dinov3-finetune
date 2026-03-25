# DINOv3 x bepli_v2 セグメンテーション 引き継ぎ資料

**作成日**: 2026-03-25
**対象リポジトリ**: dinov3_official（Facebook Research DINOv3 の fork）

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [フォークで行った変更の全体像](#2-フォークで行った変更の全体像)
3. [変更内容の詳細](#3-変更内容の詳細)
4. [実験環境](#4-実験環境)
5. [データセット（bepli_v2）](#5-データセットbepli_v2)
6. [実験一覧と結果](#6-実験一覧と結果)
7. [失敗の原因分析](#7-失敗の原因分析)
8. [未実施の改善案](#8-未実施の改善案)
9. [ファイル構成とコマンド例](#9-ファイル構成とコマンド例)

---

## 1. プロジェクト概要

### 目的

DINOv3（Facebook Research の自己教師あり学習 Vision Transformer）を使い、**海洋デブリ（漂流ごみ）の意味セグメンテーション**を行う。

### やったこと

DINOv3 公式リポジトリを fork し、以下を実施した：

1. COCO フォーマットのセグメンテーションデータセットを読み込む仕組みを追加
2. bepli_v2（Roboflow 形式の海洋デブリデータセット）に対応するよう調整
3. クラス不均衡に対処するためのクラス重み付き損失関数を実装
4. 複数パターンの実験を実行

### 結果

**全ての実験で mIoU 1.5〜1.8% と極めて低い精度に留まり、実用水準には達しなかった。**

---

## 2. フォークで行った変更の全体像

fork 上の独自コミットは以下の 3 つ：

- `f4394f1`（2026-03-06）: COCO セグメンテーションデータセットの読み込み機能を新規追加
- `857e1a4`（2026-03-18）: bepli_v2 (Roboflow) 形式への対応、学習パイプラインの強化
- `bb58f46`（2026-03-18）: CE Loss の正規化バグ修正、クラス重みの二重適用バグ修正

---

## 3. 変更内容の詳細

### 3.1 コミット `f4394f1`: COCO データセット対応

**新規追加ファイル:**

- `dinov3/data/datasets/coco_segmentation.py` — COCO 形式のセグメンテーションデータローダー。ポリゴン/RLE 両方の annotation 形式に対応
- `dinov3/eval/segmentation/configs/config-coco-linear-training.yaml` — COCO セグメンテーション用の学習設定ファイル
- `test_coco_dataset.py` — データセット読み込みの動作確認スクリプト
- `COCO_SETUP.md` — セットアップガイド（日本語）

**変更ファイル:**

- `dinov3/data/datasets/__init__.py` — COCOSegmentation の import 追加

### 3.2 コミット `857e1a4`: bepli_v2 対応 + パイプライン強化

**主要な変更:**

- `dinov3/data/datasets/coco_segmentation.py` — ディレクトリ構造を Roboflow 形式に変更（`_annotations.coco.json` + `images/`）。クラス数を 81→12 に変更
- `dinov3/data/loaders.py` — COCOSegmentation のパーサー対応
- `dinov3/eval/segmentation/loss.py` — CE Loss + Dice Loss の同時使用に対応。`class_weight` パラメータの追加
- `dinov3/eval/segmentation/config.py` — `class_weight` 設定項目の追加
- `dinov3/eval/segmentation/train.py` — TensorBoard ログ出力追加。`gt.squeeze(1)` の次元処理修正。`persistent_workers` の条件分岐修正
- `dinov3/eval/setup.py` — ローカルの torch.hub 重みファイルからのモデル読み込みに対応
- `dinov3/run/submit.py` — クラスタ外（ローカル環境）でのパス解決の修正
- `dinov3/utils/cluster.py` — Slurm 環境検出の修正（ローカル実行時に None を返す）
- `requirements.txt` — pycocotools, tensorboard, pandas, openpyxl を追加
- `configs/config-coco-linear-training-v2.yaml` — クラス重み付き + CE+Dice 併用の設定ファイルを新規追加

### 3.3 コミット `bb58f46`: Loss 関数のバグ修正

**修正した 2 つのバグ:**

#### バグ 1: CE Loss の正規化方法が不正確

- **問題**: `reduction="none"` で計算した後に単純に平均を取っていたため、`class_weight` を指定した場合の正規化が不正確だった
- **修正**: `avg_non_ignore=True` の場合は非 ignore ピクセル数で割る。それ以外は PyTorch 組み込みの `reduction="mean"` を使用（`sum(w*loss) / sum(w[target])` で正しく正規化される）

#### バグ 2: Dice Loss にもクラス重みが適用されていた

- **問題**: CE + Dice を併用する際、`class_weight` が両方に適用され二重重み付けになっていた
- **修正**: `class_weight` は CE Loss にのみ適用し、Dice Loss には `class_weight=None` を渡すよう変更

#### 背景クラス重みの調整

- `class_weight[0]`（background）を `0.0008` → `0.1` に変更
- 理由: 元の値は最大クラス重み（4.1945）との比が約 5000 倍あり、勾配の分散が大きすぎて学習が不安定になった

---

## 4. 実験環境

- **OS**: Ubuntu (Linux 5.15.0-105-generic)
- **GPU**: NVIDIA RTX A6000 x 3（各 48 GB VRAM）
- **CUDA**: 12.4
- **PyTorch**: 2.6.0+cu124
- **Python**: 3.11（uv 仮想環境）
- **GPU メモリ使用量**: 約 654〜772 MiB / GPU

---

## 5. データセット（bepli_v2）

- **形式**: Roboflow COCO フォーマット
- **訓練画像数**: 2,234 枚
- **検証画像数**: 744 枚
- **クラス数**: 12（背景 + 11 デブリ種別）
- **総アノテーション数**: 19,019 件
- **平均アノテーション数/画像**: 9.8 件

### クラス一覧

- 0: background（アノテーションなし）
- 1: pet_bottle（2,740 件）
- 2: other_bottle（1,488 件）
- 3: plastic_bag（1,848 件）
- 4: box_shaped_case（630 件）
- 5: other_container（740 件）
- 6: rope（1,656 件）
- 7: other_string（617 件）
- 8: fishing_net（1,242 件）
- 9: buoy（1,503 件）
- 10: other_fishing_gear（631 件）
- 11: styrene_foam（5,924 件）

### データセットの特徴（重要）

- **背景が画像面積の 98.2% を占める極端な不均衡**
- デブリの中央値面積は画像全体の 0.03〜0.07%（512x512 換算で約 79〜183 px²）
- ViT-B/16 の 1 パッチ（16x16=256 px²）よりも小さい物体が多い

### ディレクトリ構造

```
bepli_v2/
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

## 6. 実験一覧と結果

### モデル共通設定

- **バックボーン**: DINOv3 ViT-B/16（LVD-1689M 事前学習、**完全凍結**）
- **デコーダー**: 線形ヘッド（BN + 1x1 Conv、学習パラメータ **10,764 個のみ**）
- **バッチサイズ**: 4/GPU
- **学習率**: 1e-3（WarmupOneCycleLR）
- **入力解像度**: 512x512
- **評価方式**: スライドウィンドウ（crop=512, stride=341）

---

### 実験 1: step1_smoke（スモークテスト）

- **設定ファイル**: config-coco-linear-training.yaml
- **ステップ数**: 200
- **損失関数**: CE のみ（重みなし）
- **GPU**: 1

**結果:**

- **mIoU: 1.48%** / acc: 8.93% / aAcc: 6.76% / dice: 2.79% / fscore: 4.78% / precision: 4.71% / recall: 8.93%

**クラス別 mIoU:**

```
 0: background ........... 8.49%
 1: pet_bottle ........... 2.79%
 2: other_bottle ......... 0.00%
 3: plastic_bag .......... 0.15%
 4: box_shaped_case ...... 5.01%
 5: other_container ...... 0.05%
 6: rope ................. 0.02%
 7: other_string ......... 0.00%
 8: fishing_net .......... 1.20%
 9: buoy ................. 0.00%
10: other_fishing_gear ... 0.00%
11: styrene_foam ......... 0.00%
```

---

### 実験 2: step2_weighted（クラス重み付き短期学習）

- **設定ファイル**: config-coco-linear-training.yaml（CLI でクラス重みを上書き）
- **ステップ数**: 200
- **損失関数**: CE のみ（**クラス重みあり**）
- **GPU**: 1
- **class_weight**: [0.1, 0.6047, 1.2736, 0.3393, 0.6696, 0.5980, 0.2457, 2.9051, 0.1825, 0.6214, 4.1945, 0.3648]

**結果:**

- **mIoU: 1.59%** / acc: 6.45% / aAcc: 4.89% / dice: 3.04% / fscore: 4.05% / precision: 4.10% / recall: 6.45%

**クラス別 mIoU:**

```
 0: background ........... 7.75%
 1: pet_bottle ........... 3.76%
 2: other_bottle ......... 2.99%
 3: plastic_bag .......... 0.00%
 4: box_shaped_case ...... 3.03%
 5: other_container ...... 0.18%
 6: rope ................. 0.25%
 7: other_string ......... 0.02%
 8: fishing_net .......... 0.81%
 9: buoy ................. 0.00%
10: other_fishing_gear ... 0.32%
11: styrene_foam ......... 0.00%
```

**step1 との比較**: クラス重みにより mIoU が 1.48% → 1.59% にわずかに改善。other_bottle（0→2.99%）など一部クラスで改善が見られるが、背景の支配は解消されていない。

---

### 実験 3: step3_final（クラス重み付き長期学習）

- **設定ファイル**: config-coco-linear-training.yaml（CLI でクラス重みを上書き）
- **ステップ数**: **40,000**（約 215 epoch 相当）
- **損失関数**: CE のみ（**クラス重みあり**）
- **GPU**: 1
- **学習時間**: 1 時間 42 分
- **class_weight**: 実験 2 と同じ

**バリデーション指標の推移:**

```
Step  5,000 → mIoU: 1.45% / acc: 5.98% / aAcc: 4.24% / fscore: 3.65%
Step 10,000 → mIoU: 1.58% / acc: 6.37% / aAcc: 4.33% / fscore: 3.21%
Step 15,000 → mIoU: 1.78% / acc: 6.45% / aAcc: 4.41% / fscore: 3.62%  ← best
Step 20,000 → mIoU: 1.69% / acc: 6.45% / aAcc: 4.41% / fscore: 3.43%
Step 25,000 → mIoU: 1.69% / acc: 6.71% / aAcc: 4.41% / fscore: 3.78%
Step 30,000 → mIoU: 1.65% / acc: 6.27% / aAcc: 4.33% / fscore: 3.71%
Step 35,000 → mIoU: 1.66% / acc: 6.49% / aAcc: 4.38% / fscore: 3.72%
Step 40,000 → mIoU: 1.69% / acc: 6.40% / aAcc: 4.36% / fscore: 3.78%
```

**最終結果（best @ step 15,000）:**

- **mIoU: 1.78%** / acc: 6.45% / aAcc: 4.41% / dice: 3.32% / fscore: 3.62% / precision: 4.32% / recall: 6.45%

**最終クラス別 mIoU（step 40,000）:**

```
 0: background ........... 8.38%
 1: pet_bottle ........... 8.57%
 2: other_bottle ......... 0.81%
 3: plastic_bag .......... 0.00%
 4: box_shaped_case ...... 1.13%
 5: other_container ...... 0.08%
 6: rope ................. 0.15%
 7: other_string ......... 0.20%
 8: fishing_net .......... 0.62%
 9: buoy ................. 0.00%
10: other_fishing_gear ... 0.30%
11: styrene_foam ......... 0.00%
```

---

### 全実験の比較まとめ

```
実験名           ステップ数  損失関数  クラス重み  best mIoU
step1_smoke        200       CE       なし        1.48%
step2_weighted     200       CE       あり        1.59%
step3_final     40,000       CE       あり        1.78%
```

**結論: クラス重みの追加で +0.1%、学習ステップの大幅増加で +0.2%。いずれも誤差レベルの改善に留まった。**

---

## 7. 失敗の原因分析

### 7.1 根本原因: 背景クラスの圧倒的支配

画像面積の **98.2% が背景**。CE Loss はクラス頻度に不均衡があると支配的クラスを予測するだけで損失を最小化できてしまう。

- loss は step 1,000 で 0.019 まで急落 → 以降横ばい
- **損失は下がっているのに mIoU が改善しない** = 「背景を全て背景と予測する」ことで loss を稼いでいる
- クラス重みを入れても、背景の圧倒的ピクセル数の前では効果が限定的

### 7.2 物体サイズと ViT パッチサイズの不整合

- ViT-B/16 の 1 パッチ: 16x16 = 256 px²
- デブリの中央値面積: 79〜183 px²
- 特徴マップ解像度: 32x32（512px 入力時）

多くのデブリが **1 パッチ以下**のサイズであり、特徴マップ上で 1 ピクセル（あるいは 0 ピクセル）にしかならない。線形ヘッドはこの粗い特徴マップからピクセルごとに分類するだけなので、小物体を検出する空間的手がかりが不足している。

### 7.3 線形ヘッドの表現力不足

- 学習可能パラメータは **10,764 個のみ**（BatchNorm 1,536 + Conv1x1 9,228）
- 各ピクセル位置の特徴ベクトルを独立に 12 クラスに分類するだけの能力
- 周辺ピクセルとの空間的文脈を一切利用できない

DINOv3 公式の線形セグメンテーションは ADE20K（150 クラス、大型物体が多い）で設計されており、bepli_v2（12 クラス、極小物体）とは前提が根本的に異なる。

### 7.4 ドメインギャップ

DINOv3 は Web 上の自然画像（LVD-1689M データセット）で事前学習されている。bepli_v2 は：

- **海洋環境**: 水面反射、一様な背景
- **特殊な物体外観**: 劣化・変形したプラスチック、ロープ、網など
- **特殊な視点**: ドローンや船舶からの俯瞰撮影

凍結バックボーンが出力する特徴表現は海洋デブリに特化しておらず、線形ヘッドだけではこのギャップを埋められない。

### 7.5 バリデーション指標の頭打ち

step3_final（40,000 steps）では step 15,000 で best mIoU 1.78% に到達した後、残り 25,000 steps で改善が見られない。これは：

- 線形ヘッドの容量上限に到達している
- 凍結特徴量 + 線形プローブという設定での理論的限界に近い

---

## 8. 未実施の改善案

以下は EXPERIMENT_REPORT.md に記載された改善案のうち、未実施のものである。

### 優先度 高

- **Dice Loss の有効化** — config-v2.yaml に設定済み（`diceloss_weight: 0.5`）だが、**この設定での実験は未実施**
- **バックボーンの部分 Fine-tuning** — 凍結を解除して最終数層のみ学習させる

### 優先度 中

- **M2F デコーダーの使用** — 設定変更のみで試行可能
- **FOUR_EVEN_INTERVALS 層の使用** — 設定変更のみで試行可能
- **入力解像度の拡大**（例: 768x768）
- **データ拡張の強化**

### 優先度 低

- **SAT-493M 重みの試用** — 衛星画像で学習済みのためドメインが近い可能性
- **インスタンスセグメンテーションへの転換**

---

## 9. ファイル構成とコマンド例

### 主要ファイル

```
dinov3_official/
  dinov3/
    data/
      datasets/
        coco_segmentation.py       # COCO/Roboflow データローダー
      loaders.py                   # データセットパーサー（COCOSegmentation 追加）
    eval/
      segmentation/
        configs/
          config-coco-linear-training.yaml     # 基本設定（CE のみ）
          config-coco-linear-training-v2.yaml  # v2 設定（CE+Dice+class_weight）
        loss.py          # 損失関数（CE + Dice 併用対応済み）
        config.py        # 設定データクラス（class_weight 追加済み）
        train.py         # 学習ループ（TensorBoard 対応済み）
        eval.py          # 評価ループ
        metrics.py       # mIoU 等のメトリクス計算
      setup.py           # モデル読み込み（ローカル重み対応済み）
    utils/
      cluster.py         # クラスタ検出（ローカル環境対応済み）
    run/
      submit.py          # ジョブ投入（ローカル環境対応済み）
  weights/
    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  # 事前学習済み重み（334MB）
  outputs/
    step1_smoke/         # 実験 1 の結果
    step2_weighted/      # 実験 2 の結果
    step3_final/         # 実験 3 の結果
  EXPERIMENT_REPORT.md   # 詳細な実験レポート（日本語）
  COCO_SETUP.md          # セットアップガイド（日本語）
```

### 実行コマンド例

```bash
# 基本実行（CE のみ、重みなし）
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
  output_dir=./outputs/step1_smoke \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=/path/to/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/path/to/bepli_v2 \
  scheduler.total_iter=200 \
  scheduler.constructor_kwargs.warmup_iters=20 \
  eval.eval_interval=200

# クラス重み付き（CE のみ）
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
  output_dir=./outputs/step2_weighted \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=/path/to/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/path/to/bepli_v2 \
  scheduler.total_iter=200 \
  scheduler.constructor_kwargs.warmup_iters=20 \
  eval.eval_interval=200 \
  'train.class_weight=[0.1,0.6047,1.2736,0.3393,0.6696,0.5980,0.2457,2.9051,0.1825,0.6214,4.1945,0.3648]'

# CE + Dice 併用（config-v2 を使用、未実験）
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training-v2.yaml \
  output_dir=./outputs/experiment_name \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=/path/to/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/path/to/bepli_v2
```

### 依存パッケージ（追加分）

```
pycocotools
tensorboard
pandas
openpyxl
```

---

## 補足: この実験から得られた知見

1. **パイプラインとしては正常動作している** — 問題はコードではなく、タスクとモデル設定の相性
2. **凍結バックボーン + 線形ヘッド（Linear Probe）は、このデータセットには不向き** — 背景 98.2%・極小物体・ドメインギャップの三重苦
3. **クラス重みだけでは不十分** — ピクセル数の偏りが極端すぎて、重み調整のみでは根本解決にならない
4. **次のステップとしては、バックボーンの Fine-tuning か、より表現力の高いデコーダー（M2F）の使用が最有力**
