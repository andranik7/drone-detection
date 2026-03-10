"""Download and organize the YOLO drone detection dataset from Kaggle."""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATASET_SLUG = "muki2003/yolo-drone-detection-dataset"


def download_dataset(dest: Path = DATA_DIR) -> Path:
    """Download dataset via Kaggle CLI. Requires KAGGLE_USERNAME & KAGGLE_KEY env vars."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset %s to %s", DATASET_SLUG, dest)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "--unzip", "-p", str(dest)],
        check=True,
    )
    logger.info("Dataset downloaded successfully")
    return dest


def verify_structure(data_dir: Path = DATA_DIR) -> dict:
    """Verify dataset has expected train/valid splits with images and labels."""
    dataset_dir = data_dir / "drone_dataset"
    stats = {}
    for split in ["train", "valid"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        lbl_count = len(list(labels_dir.glob("*"))) if labels_dir.exists() else 0
        stats[split] = {"images": img_count, "labels": lbl_count}
    logger.info("Dataset structure: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_dataset()
    stats = verify_structure()
    for split, counts in stats.items():
        logger.info("%s: %d images, %d labels", split, counts["images"], counts["labels"])
