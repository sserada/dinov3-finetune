# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
import numpy as np
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
except ImportError:
    raise ImportError("Please install pycocotools: pip install pycocotools")

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        return {
            _Split.TRAIN: "train",
            _Split.VAL: "val",
        }[self]


def _create_semantic_mask_from_annotations(
    coco: COCO,
    img_info: dict,
    cat_id_to_class_idx: dict,
    num_classes: int
) -> np.ndarray:
    """
    Create semantic segmentation mask from COCO annotations.

    Args:
        coco: COCO object
        img_info: Image information dictionary
        cat_id_to_class_idx: Mapping from category ID to class index
        num_classes: Number of classes

    Returns:
        Semantic mask as numpy array
    """
    # Initialize mask with background class (0)
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)

    # Sort by area (larger objects first, so smaller objects can overlay)
    anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)

    for ann in anns:
        # Get category class index
        cat_id = ann['category_id']
        class_idx = cat_id_to_class_idx.get(cat_id, 0)

        if class_idx >= num_classes:
            continue

        # Get binary mask from annotation
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rle = coco_mask.frPyObjects(
                    ann['segmentation'],
                    img_info['height'],
                    img_info['width']
                )
                binary_mask = coco_mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = binary_mask.sum(axis=2) > 0
            elif isinstance(ann['segmentation'], dict):
                # RLE format
                binary_mask = coco_mask.decode(ann['segmentation'])
            else:
                continue

            # Apply to semantic mask
            mask[binary_mask > 0] = class_idx

    return mask


class COCOSegmentation(ExtendedVisionDataset):
    """
    COCO dataset for semantic segmentation.

    Converts instance annotations to semantic segmentation masks.
    """

    Split = Union[_Split]
    Labels = Union[Image.Image]

    def __init__(
        self,
        split: "COCOSegmentation.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
        num_classes: int = 12,  # bepli_v2: 11 debris classes + background
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        self.split = split
        self.num_classes = num_classes

        # Load COCO annotations (bepli_v2 / Roboflow COCO format)
        ann_file = os.path.join(root, split.dirname, "_annotations.coco.json")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Expected bepli_v2 directory structure:\n"
                f"  {root}/\n"
                f"    train/\n"
                f"      _annotations.coco.json\n"
                f"      images/\n"
                f"    val/\n"
                f"      _annotations.coco.json\n"
                f"      images/"
            )

        self.coco = COCO(ann_file)

        # Get all image IDs
        self.image_ids = list(self.coco.imgs.keys())

        # Build category ID to class index mapping
        self.cat_id_to_class_idx = self._build_category_mapping()

        # Image directory (bepli_v2 / Roboflow COCO format)
        self.img_dir = os.path.join(root, split.dirname, "images")

        print(f"COCO Segmentation dataset initialized:")
        print(f"  Split: {split.value}")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Classes: {num_classes}")

    def _build_category_mapping(self):
        """Build mapping from COCO category IDs to contiguous class indices."""
        cat_ids = sorted(self.coco.getCatIds())
        # Map category IDs to contiguous indices starting from 1 (0 is background)
        cat_id_to_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        cat_id_to_idx[0] = 0  # Background
        return cat_id_to_idx

    def get_image_data(self, index: int) -> bytes:
        """Get image data as bytes."""
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        image_path = os.path.join(self.img_dir, img_info['file_name'])

        with open(image_path, mode="rb") as f:
            image_data = f.read()

        return image_data

    def get_target(self, index: int) -> Any:
        """Get segmentation mask."""
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        # Create semantic mask from COCO annotations
        mask = _create_semantic_mask_from_annotations(
            self.coco,
            img_info,
            self.cat_id_to_class_idx,
            self.num_classes
        )

        # Convert to PIL Image
        mask_image = Image.fromarray(mask, mode='L')

        # Convert PIL Image to bytes (PNG format)
        import io
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_data = buffer.getvalue()

        return mask_data

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item with proper error handling."""
        try:
            return super().__getitem__(index)
        except Exception as e:
            # Log error and try next item
            print(f"Error loading image at index {index}: {e}")
            # Return next valid item
            return self.__getitem__((index + 1) % len(self))
