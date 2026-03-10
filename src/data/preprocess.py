"""Validate and clean images/labels for YOLO training."""

import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def validate_image(image_path: Path) -> bool:
    """Check if an image file is valid and readable."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_label(label_path: Path) -> bool:
    """Check YOLO label format: each line must be 'class x_center y_center width height'."""
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                int(parts[0])  # class id
                for v in parts[1:]:
                    val = float(v)
                    if not (0.0 <= val <= 1.0):
                        return False
        return True
    except Exception:
        return False


def clean_split(split_dir: Path) -> dict:
    """Remove invalid image/label pairs from a split directory. Returns stats."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    removed = 0
    total = 0

    if not images_dir.exists():
        return {"total": 0, "removed": 0}

    for img_path in list(images_dir.glob("*")):
        total += 1
        label_path = labels_dir / (img_path.stem + ".txt")

        if not validate_image(img_path) or (label_path.exists() and not validate_label(label_path)):
            img_path.unlink()
            if label_path.exists():
                label_path.unlink()
            removed += 1
            logger.warning("Removed invalid pair: %s", img_path.name)

    logger.info("Cleaned %s: %d total, %d removed, %d kept", split_dir.name, total, removed, total - removed)
    return {"total": total, "removed": removed, "kept": total - removed}
