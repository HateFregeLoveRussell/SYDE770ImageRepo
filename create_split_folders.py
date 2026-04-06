"""
create_split_folders.py

Create v2_train/ and v2_test/ folders by symlinking (or copying) images
from the augmented_dataset_v2 using the saved split metadata.

Run this on any machine after: git pull && dvc pull

Usage:
  python create_split_folders.py
  python create_split_folders.py --output-dir splits
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="Create train/test image folders from split CSV")
    p.add_argument("--split-csv", default="models/v2_train_test_split.csv",
                   help="Path to split CSV (default: models/v2_train_test_split.csv)")
    p.add_argument("--images-dir", default="data/augmented_dataset_v2/images",
                   help="Source images directory")
    p.add_argument("--output-dir", default="splits",
                   help="Output directory for v2_train/ and v2_test/ (default: splits/)")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlinking (slower but more portable)")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = pathlib.Path(args.images_dir)
    output_dir = pathlib.Path(args.output_dir)
    train_dir = output_dir / "v2_train"
    test_dir = output_dir / "v2_test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {args.split_csv}")
    print(f"Source images: {images_dir}")
    print(f"Output: {output_dir}")

    train_count, test_count, missing = 0, 0, 0
    for row in rows:
        key = row["image_key"]
        split = row["split"]
        dst_dir = train_dir if split == "train" else test_dir

        src = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = images_dir / (key + ext)
            if candidate.exists():
                src = candidate
                break

        if src is None:
            missing += 1
            continue

        dst = dst_dir / src.name
        if dst.exists():
            continue

        if args.copy:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

        if split == "train":
            train_count += 1
        else:
            test_count += 1

    print(f"\nDone!")
    print(f"  Train: {train_count} images -> {train_dir}")
    print(f"  Test:  {test_count} images -> {test_dir}")
    if missing:
        print(f"  Missing: {missing} (image not found in source dir)")


if __name__ == "__main__":
    main()
