# DINOv3 x bepli_v2 セグメンテーション 引き継ぎ資料

**作成日**: 2026-03-25
**最終更新日**: 2026-03-27
**対象リポジトリ**: dinov3_official（Facebook Research DINOv3 の fork）

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [フォークで行った変更の全体像](#2-フォークで行った変更の全体像)
3. [変更内容の詳細](#3-変更内容の詳細)
4. [実験環境](#4-実験環境)
5. [データセット（bepli_v2）](#5-データセットbepli_v2)
6. [実験一覧と結果](#6-実験一覧と結果)
7. [失敗の原因分析（初期）](#7-失敗の原因分析初期)
8. [外部事例との比較分析（DINOv3 SAT-Bench）](#8-外部事例との比較分析dinov3-sat-bench)
9. [失敗の真の原因: デコーダーの表現力不足](#9-失敗の真の原因-デコーダーの表現力不足)
10. [改善案（優先度見直し済み）](#10-改善案優先度見直し済み)
11. [ConvNeXt 蒸留モデルの利用可能性調査](#11-convnext-蒸留モデルの利用可能性調査)
12. [ファイル構成とコマンド例](#12-ファイル構成とコマンド例)

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

## 7. 失敗の原因分析（初期）

> **注意**: 本セクションは実験直後に記載した初期分析である。追加調査により、**真の主因はデコーダーの表現力不足**であることが判明した。詳細はセクション 8〜9 を参照。

### 7.1 背景クラスの圧倒的支配

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

## 8. 外部事例との比較分析（DINOv3 SAT-Bench）

### 8.1 参照資料

DINOv3 の凍結バックボーン + デコーダー学習でセグメンテーションに成功している外部事例が存在する。

- **発表資料**: https://www.docswell.com/s/motokimura/KVM382-dinov3-sat-bench#p9
- **タスク**: SpaceNet 2 建物セグメンテーション（衛星画像）
- **データ規模**: 9,783 画像（5-fold の 3 fold 使用）、解像度 320x320

### 8.2 成功事例の実験設定

| 項目 | SAT-Bench (成功) | 今回の実験 (失敗) |
|------|------------------|-------------------|
| **バックボーン** | ViT-L/16 (300M params) | ViT-B/16 (86M params) |
| **バックボーン状態** | 完全凍結 | 完全凍結 |
| **デコーダー** | **Feature2Pyramid + U-Net** | **BN + Conv1x1 (Linear Head)** |
| **学習可能パラメータ** | **約 20M** | **10,764** |
| **マルチスケール特徴** | 1/4〜1/32 の 4 段階 | 最終層のみ (1/16) |
| **対象物体サイズ** | 建物（大きい、画像の数%〜数十%） | デブリ（極小、画像の 0.03〜0.07%） |
| **背景比率** | 適度 | 98.2% |

### 8.3 SAT-Bench から得られた知見

1. **ViT バックボーンの凍結でも、十分なデコーダーがあれば小データで高精度を達成できる**
   - 学習サンプル約 1,000 枚で、DINOv3（Web 学習版）が既存手法を上回った
   - 2,000 枚を超えると、学習済み MaxViT-Small に追いつかれる傾向
2. **意外にも、衛星画像特化重み（SAT-493M）より Web 学習重み（LVD-1689M）の方が高精度だった**
3. **課題は建物輪郭周辺の精度不足** — データ量が増えると従来手法に劣化する場面がある

### 8.4 今回の実験との決定的な差

**デコーダーの学習可能パラメータ量が約 2,000 倍異なる**（20M vs 10,764）。

SAT-Bench のデコーダー構成:

```
ViT 中間層出力
  ↓ Feature2Pyramid（ViT の単一解像度出力を 4 段階のマルチスケール特徴ピラミッドに変換）
  ↓ U-Net デコーダー（skip connection 付き段階的アップサンプリング）
  ↓ 最終出力: 1/4 解像度 → 4 倍バイリニア補間
```

今回の Linear Head:

```
ViT 最終層出力 (32x32)
  ↓ BatchNorm
  ↓ Conv1x1（768ch → 12ch）← 唯一の学習層
  ↓ 16 倍バイリニア補間 (32x32 → 512x512)
```

---

## 9. 失敗の真の原因: デコーダーの表現力不足

セクション 7 の初期分析と、セクション 8 の外部事例比較を統合した結論を示す。

### 9.1 主因: Linear Head は実用的なセグメンテーションに設計されていない

Linear Head（Conv1x1）は本来、**バックボーンの特徴品質を評価するためのプローブ（探針）**として設計されたものであり、実用的なセグメンテーション精度を出すことを目的としていない。

Linear Head の構造的限界:

1. **空間的文脈を一切見ない** — Conv1x1 は receptive field が 1 ピクセル分しかない。位置 (i, j) の予測はその位置の特徴ベクトルのみで決まり、「隣に何があるか」を考慮できない
2. **粗い特徴マップからの単純補間** — 32x32 → 512x512 の 16 倍 bilinear 補間は学習不要のただの拡大処理であり、粗い予測をぼんやり引き伸ばすだけ
3. **パラメータ数の絶対的不足** — 10,764 パラメータでは、12 クラスの空間パターンを学習する容量が根本的に足りない

### 9.2 U-Net 型 / M2F 型デコーダーとの構造比較

```
Linear Head (今回):
  特徴(32x32) → [Conv1x1] → bilinear補間 → 予測(512x512)

U-Net 型デコーダー (SAT-Bench):
  特徴(32x32) ──→ [Conv3x3 + 文脈統合]  → 64x64
  特徴(64x64) ──→ [Conv3x3 + skip 結合] → 128x128
  特徴(128x128)→ [Conv3x3 + skip 結合]  → 256x256
                  [Conv3x3 + skip 結合]  → 512x512 → 予測
```

U-Net 型 / M2F 型の利点:
- **Conv3x3 以上**で周辺ピクセルの空間的文脈を利用できる
- **段階的アップサンプリング**で高解像度の空間情報を復元できる
- **skip connection** でバックボーン浅い層の高解像度特徴を活用できる
- 1 ピクセル未満の小物体でも、周辺の文脈パターンから推論が可能

### 9.3 原因の優先度整理（見直し後）

| 優先度 | 原因 | 寄与度 | 対策 |
|--------|------|--------|------|
| **1（最大）** | デコーダーの表現力不足（Linear Head） | 極めて高い | M2F デコーダーへの変更 |
| **2** | マルチスケール特徴の不使用 | 高い | FOUR_EVEN_INTERVALS + M2F |
| **3** | 背景 98.2% の極端な不均衡 | 中程度 | Dice Loss、クラス重み |
| **4** | 物体サイズと特徴マップ解像度の不整合 | 中程度 | 入力解像度拡大、ConvNeXt |
| **5** | ドメインギャップ | 低〜中程度 | バックボーン部分 Fine-tuning |

> **重要**: 初期分析（セクション 7）では「背景の不均衡」と「ViT パッチサイズとの不整合」を主因としていたが、外部事例の比較により、**それ以前にデコーダーの表現力が決定的に不足していた**ことが判明した。SAT-Bench では ViT バックボーン凍結のまま、デコーダーを強化するだけで少量データでも高精度を達成している。

### 9.4 期待される改善効果と注意点

M2F デコーダーへの変更により、mIoU 1.8% → **10〜30% 程度**への改善は十分ありえる。ただし以下の点に留意:

- SAT-Bench の対象（建物）は画像の数%〜数十%を占める大きな物体であり、bepli_v2 のデブリ（0.03〜0.07%）とは物体サイズが根本的に異なる
- デコーダー強化だけで「実用水準」（mIoU 50%+）に到達するかは不確実
- デコーダー強化で改善が見られた場合、次にバックボーン側（ConvNeXt、部分 Fine-tuning）を重ねることで更なる改善が期待できる

---

## 10. 改善案（優先度見直し済み）

> セクション 9 の分析結果を踏まえ、旧セクション 8 の優先度を見直した。**デコーダー強化を最優先に変更**。

### 優先度 最高（次に試すべきタスク）

- **M2F (Mask2Former) デコーダーへの変更** — 本リポジトリに実装済み。設定変更のみで試行可能。マルチスケール特徴 + Transformer デコーダーにより、空間的文脈の利用と段階的なアップサンプリングが実現される。SAT-Bench の成功事例から、デコーダー強化が最も効果的な改善策と判断

  ```yaml
  # config の変更箇所
  decoder_head:
    type: m2f                                    # linear → m2f
    backbone_out_layers: FOUR_EVEN_INTERVALS     # LAST → FOUR_EVEN_INTERVALS
  ```

  - M2F の実装: `dinov3/eval/segmentation/models/heads/mask2former_head.py`
  - Adapter の実装: `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`
  - 学習用 config: `dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml`
  - **注意**: M2F + Adapter は Linear Head に比べて VRAM 使用量が大幅に増加する。バッチサイズの調整が必要になる可能性がある

### 優先度 高

- **Dice Loss の有効化** — config-v2.yaml に設定済み（`diceloss_weight: 0.5`）だが、**この設定での実験は未実施**。M2F と併用することで、背景支配の問題を軽減できる可能性がある
- **ConvNeXt 蒸留モデルへの切り替え** — 畳み込みベースでネイティブなマルチスケール特徴を持つ。M2F との相性が良い。詳細はセクション 11 参照

### 優先度 中

- **バックボーンの部分 Fine-tuning** — 凍結を解除して最終数層のみ学習させることでドメインギャップを軽減
- **入力解像度の拡大**（例: 768x768）— 特徴マップ上の物体サイズを増加させる
- **データ拡張の強化**

### 優先度 低

- **SAT-493M 重みの試用** — SAT-Bench では Web 学習版（LVD-1689M）の方が高精度だったため、優先度を下げた
- **インスタンスセグメンテーションへの転換**

---

## 11. ConvNeXt 蒸留モデルの利用可能性調査

### 11.1 事前学習済み重みの公開状況

4 サイズ全て Facebook 公式サーバー（`https://dl.fbaipublicfiles.com/dinov3`）から自動ダウンロード可能。ローカルに重みファイルを配置する必要はない。

| モデル | hub 名 | embed_dim (最終層) | パラメータ数 | hash |
|--------|--------|-------------------|-------------|------|
| ConvNeXt Tiny | `dinov3_convnext_tiny` | 768 | 29M | `21b726bb` |
| ConvNeXt Small | `dinov3_convnext_small` | 768 | 50M | `296db49d` |
| ConvNeXt Base | `dinov3_convnext_base` | 1024 | 89M | `801f2ba9` |
| ConvNeXt Large | `dinov3_convnext_large` | 1536 | 198M | `61fa432d` |

### 11.2 既存パイプラインとの互換性

コード調査の結果、**現在の学習パイプラインは ConvNeXt と互換性がある**:

1. `hubconf.py` に `dinov3_convnext_tiny/small/base/large` が登録済み
2. `dinov3/eval/setup.py` の `load_model_and_context()` で `"dinov3" in model_config.dino_hub` の判定があり、ConvNeXt モデル名も通過する
3. `dinov3/models/convnext.py` の `ConvNeXt` クラスが ViT と同じインターフェース（`get_intermediate_layers()`, `embed_dim`, `n_blocks`）を実装済み
4. Linear Head / M2F 両方のデコーダーでバックボーン非依存の設計になっている

### 11.3 ConvNeXt の技術的特徴

- **ネイティブなマルチスケール特徴**: 4 ステージで [H/4, H/8, H/16, H/32] の特徴マップを自然に出力。ViT + Feature2Pyramid のような変換が不要
- **空間的局所性の帰納バイアス**: 畳み込みベースのため、少量データでの学習に有利
- **各ステージの次元**: Tiny/Small = [96, 192, 384, 768]、Base = [128, 256, 512, 1024]、Large = [192, 384, 768, 1536]

### 11.4 注意点

- ConvNeXt の `embed_dim` は最終ステージの次元のみを返す（例: Small = 768）。`FOUR_EVEN_INTERVALS` で 4 層抽出する場合、`models/__init__.py:117` の処理で全層が同一次元として扱われる。`get_intermediate_layers` 内でリサイズされるため Linear Head では動作するが、M2F (DINOv3_Adapter) は ViT を前提とした設計のため、ConvNeXt との組み合わせには追加の検証・改修が必要
- `dino_hub_weights` を省略すれば Facebook サーバーから自動ダウンロードされる。ローカルに保存したい場合は `weights/` ディレクトリに配置し、パスを指定する

### 11.5 実行コマンド例

```bash
# ConvNeXt Small + Linear Head（最終層のみ、最小構成での動作確認用）
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
  output_dir=./outputs/convnext_small_linear \
  model.dino_hub=dinov3_convnext_small \
  datasets.root=/path/to/bepli_v2 \
  decoder_head.backbone_out_layers=LAST \
  scheduler.total_iter=5000 \
  eval.eval_interval=1000
```

---

## 12. ファイル構成とコマンド例

### M2F デコーダー関連ファイル

```
dinov3/eval/segmentation/
  models/
    __init__.py                    # build_segmentation_decoder(): デコーダー構築のエントリポイント
    backbone/
      dinov3_adapter.py            # DINOv3_Adapter: ViT → マルチスケール特徴変換（M2F 用）
    heads/
      linear_head.py               # LinearHead: Conv1x1 のみ（実験 1〜3 で使用）
      mask2former_head.py          # Mask2FormerHead: M2F デコーダー
      pixel_decoder.py             # PixelDecoder: M2F 内部のピクセルデコーダー
    utils/
      ms_deform_attn.py            # Multi-Scale Deformable Attention（M2F の中核モジュール）
      transformer.py               # Transformer デコーダー層
```

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
          config-coco-m2f-training.yaml        # M2F デコーダー学習用
        loss.py          # 損失関数（CE + Dice 併用対応済み）
        config.py        # 設定データクラス（class_weight 追加済み）
        train.py         # 学習ループ（TensorBoard 対応済み、M2F 対応済み）
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
```

### 実行コマンド例

```bash
# M2F デコーダー学習（ViT-B/16 バックボーン凍結、次に試すべき実験）
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-m2f-training.yaml \
  output_dir=./outputs/step4_m2f \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=/path/to/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/path/to/bepli_v2

# 基本実行（CE のみ、重みなし）— 実験 1 で実施済み
torchrun --nproc_per_node=1 -m dinov3.eval.segmentation.run \
  config=dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \
  output_dir=./outputs/step1_smoke \
  model.dino_hub=dinov3_vitb16 \
  model.dino_hub_weights=/path/to/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  datasets.root=/path/to/bepli_v2 \
  scheduler.total_iter=200 \
  scheduler.constructor_kwargs.warmup_iters=20 \
  eval.eval_interval=200

# クラス重み付き（CE のみ）— 実験 2 で実施済み
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
```

### 依存パッケージ（追加分）

```
pycocotools
tensorboard
pandas
openpyxl
torchmetrics
```

---

## 補足: この実験から得られた知見

1. **パイプラインとしては正常動作している** — 問題はコードではなく、タスクとモデル設定の相性
2. **Linear Head はプローブであり、実用的なセグメンテーション用途には不向き** — 空間的文脈を持たない Conv1x1 は、背景 98.2%・極小物体のタスクでは原理的に機能しない
3. **クラス重みだけでは不十分** — ピクセル数の偏りが極端すぎて、重み調整のみでは根本解決にならない
4. **外部事例（DINOv3 SAT-Bench）から、凍結バックボーンでもデコーダーを強化すれば少量データで高精度を達成できることが確認済み**
5. **次のステップとしては M2F デコーダーの使用が最優先**。バックボーン変更（ConvNeXt、部分 Fine-tuning）はその後に検討
