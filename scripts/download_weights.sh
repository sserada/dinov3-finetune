#!/bin/bash
# DINOv3 事前学習済み重みを HuggingFace Hub からダウンロードするスクリプト
#
# 前提条件:
#   1. pip install huggingface_hub safetensors
#   2. python -c "from huggingface_hub import login; login()"  でログイン済み
#   3. 各モデルページで Gated Model の承認済み:
#      - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
#      - https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m
#      - https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
#
# 使い方:
#   ./scripts/download_weights.sh                  # 全モデルをダウンロード
#   ./scripts/download_weights.sh vitb16            # ViT-B/16 のみ
#   ./scripts/download_weights.sh convnext_small    # ConvNeXt Small のみ
#   ./scripts/download_weights.sh convnext_tiny     # ConvNeXt Tiny のみ

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/../weights"
mkdir -p "${WEIGHTS_DIR}"

download_model() {
    local repo_id="$1"
    local output_name="$2"
    local output_path="${WEIGHTS_DIR}/${output_name}"

    if [ -f "${output_path}" ]; then
        echo "[SKIP] ${output_name} already exists"
        return 0
    fi

    echo "[DOWNLOAD] ${repo_id} -> ${output_name}"
    python3 -c "
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

path = hf_hub_download(
    repo_id='${repo_id}',
    filename='model.safetensors',
)
print('  Converting safetensors -> pth ...')
state_dict = load_file(path)
torch.save(state_dict, '${output_path}')
print('  Saved: ${output_path}')
"
    echo "[DONE] ${output_name}"
}

# --- モデル定義 ---

download_vitb16() {
    download_model \
        "facebook/dinov3-vitb16-pretrain-lvd1689m" \
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
}

download_convnext_small() {
    download_model \
        "facebook/dinov3-convnext-small-pretrain-lvd1689m" \
        "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
}

download_convnext_tiny() {
    download_model \
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m" \
        "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
}

# --- メイン ---

TARGET="${1:-all}"

case "${TARGET}" in
    vitb16)
        download_vitb16
        ;;
    convnext_small)
        download_convnext_small
        ;;
    convnext_tiny)
        download_convnext_tiny
        ;;
    all)
        download_vitb16
        download_convnext_small
        download_convnext_tiny
        ;;
    *)
        echo "Usage: $0 [vitb16|convnext_small|convnext_tiny|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Weights directory ==="
ls -lh "${WEIGHTS_DIR}"/*.pth 2>/dev/null || echo "(empty)"
