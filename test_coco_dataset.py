"""
Test script for COCO Segmentation dataset loader
"""

import sys
import torch
from pathlib import Path

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent))

from dinov3.data.datasets import COCOSegmentation


def test_coco_dataset():
    """Test COCO Segmentation dataset loader"""

    print("="*80)
    print("Testing COCO Segmentation Dataset Loader")
    print("="*80)
    print()

    # Test configuration
    # NOTE: Update this path to your COCO dataset location
    coco_root = "/path/to/coco"  # UPDATE THIS

    print(f"COCO root: {coco_root}")
    print()

    # Check if path exists
    if not Path(coco_root).exists():
        print("⚠️  WARNING: COCO dataset path does not exist!")
        print(f"   Please update the 'coco_root' variable in this script.")
        print(f"   Expected structure:")
        print(f"     {coco_root}/")
        print(f"       train2017/")
        print(f"       val2017/")
        print(f"       annotations/")
        print(f"         instances_train2017.json")
        print(f"         instances_val2017.json")
        print()
        print("Skipping dataset test...")
        return

    try:
        # Load train split
        print("Loading TRAIN split...")
        train_dataset = COCOSegmentation(
            split=COCOSegmentation.Split.TRAIN,
            root=coco_root,
            num_classes=81  # COCO: 80 objects + background
        )
        print(f"✓ Train dataset loaded: {len(train_dataset)} images")
        print()

        # Load val split
        print("Loading VAL split...")
        val_dataset = COCOSegmentation(
            split=COCOSegmentation.Split.VAL,
            root=coco_root,
            num_classes=81
        )
        print(f"✓ Val dataset loaded: {len(val_dataset)} images")
        print()

        # Test loading a single item
        print("Testing data loading...")
        image, mask = train_dataset[0]
        print(f"✓ Image type: {type(image)}")
        print(f"✓ Mask type: {type(mask)}")
        print(f"✓ Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
        print(f"✓ Mask size: {mask.size if hasattr(mask, 'size') else 'N/A'}")
        print()

        # Check mask values
        import numpy as np
        from PIL import Image

        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            print(f"✓ Unique mask values: {unique_values}")
            print(f"✓ Min class: {unique_values.min()}")
            print(f"✓ Max class: {unique_values.max()}")
            print()

        print("="*80)
        print("✓ All tests passed!")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Update config-coco-linear-training.yaml with your dataset path")
        print("2. Run training:")
        print("   python -m dinov3.run.submit dinov3/eval/segmentation/run.py \\")
        print("       --config dinov3/eval/segmentation/configs/config-coco-linear-training.yaml \\")
        print("       --backbone dinov3_convnext_small")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print()
        print("Please check:")
        print("1. COCO dataset path is correct")
        print("2. Directory structure matches expected format")
        print("3. Annotation files exist")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_coco_dataset()
