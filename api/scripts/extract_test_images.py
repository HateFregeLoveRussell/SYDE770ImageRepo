"""
extract_test_images.py

Recreates the exact train/test splits used for v1 and v2 model training,
then symlinks the test images into organized folders for API testing.

Uses the same seed=42, test_size=0.20, stratified split as prepare_yolo_split.py,
including hard negatives (images with no detections).

Usage:
  python scripts/extract_test_images.py
  python scripts/extract_test_images.py --output ../data/test_splits
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


DATASETS = {
    "v1": {
        "images": REPO_ROOT / "data" / "augmented_dataset_v1" / "images",
        "samples": REPO_ROOT / "data" / "augmented_dataset_v1" / "samples.parquet",
        "labels": REPO_ROOT / "data" / "derived" / "yolo" / "labels_v1",
    },
    "v2": {
        "images": REPO_ROOT / "data" / "augmented_dataset_v2" / "images",
        "samples": REPO_ROOT / "data" / "augmented_dataset_v2" / "samples.parquet",
        "labels": REPO_ROOT / "data" / "derived" / "yolo" / "labels_v2",
    },
}


def find_image(key: str, images_dir: pathlib.Path) -> pathlib.Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / (key + ext)
        if candidate.exists():
            return candidate
    return None


def extract_split(
    dataset_name: str,
    images_dir: pathlib.Path,
    samples_pq: pathlib.Path,
    labels_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 42,
    test_size: float = 0.20,
) -> None:
    """Extract test images using the same logic as prepare_yolo_split.py."""

    # Collect ALL keys that have both a label file AND an image,
    # plus hard negatives from samples.parquet that have images but no labels.
    label_keys = {f.stem for f in labels_dir.glob("*.txt") if f.is_file()}

    df = pd.read_parquet(samples_pq, columns=["image_key", "has_tim_bbox", "n_det"])
    df["stem"] = (
        df["image_key"]
        .str.replace(r"^images/", "", regex=True)
        .str.replace(r"\.(jpg|jpeg|png|webp)$", "", regex=True)
    )

    # All image keys from samples.parquet that exist on disk
    all_keys = []
    for stem in sorted(df["stem"].unique()):
        if find_image(stem, images_dir) is not None:
            all_keys.append(stem)

    # Also include labelled keys not in samples.parquet
    for k in sorted(label_keys):
        if k not in set(df["stem"]) and find_image(k, images_dir) is not None:
            all_keys.append(k)

    all_keys = sorted(set(all_keys))

    if not all_keys:
        print(f"  WARNING: No matching images found")
        return

    # Stratification: 0=hard_negative, 1=non-tim cups, 2=tim hortons
    key_to_tim = dict(zip(df["stem"], df["has_tim_bbox"]))
    key_to_ndet = dict(zip(df["stem"], df["n_det"]))

    def strat_class(key):
        n = key_to_ndet.get(key, 0)
        if n == 0:
            return 0
        return 2 if key_to_tim.get(key, False) else 1

    strat = [strat_class(k) for k in all_keys]

    hn = strat.count(0)
    nontim = strat.count(1)
    tim = strat.count(2)
    print(f"  Total: {len(all_keys)} (Tim: {tim}, Non-Tim: {nontim}, Hard neg: {hn})")

    tv_keys, test_keys, _, _ = train_test_split(
        all_keys, strat,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    test_dir = output_dir / dataset_name / "test"
    train_dir = output_dir / dataset_name / "train_sample"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    copied_test = 0
    for key in test_keys:
        img = find_image(key, images_dir)
        if img:
            dst = test_dir / img.name
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst)
            copied_test += 1

    copied_train = 0
    for key in tv_keys[:50]:
        img = find_image(key, images_dir)
        if img:
            dst = train_dir / img.name
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst)
            copied_train += 1

    print(f"  {dataset_name}: {copied_test} test images -> {test_dir}")
    print(f"  {dataset_name}: {copied_train} train samples -> {train_dir}")
    print(f"  (split: {len(tv_keys)} train+cv, {len(test_keys)} test)")


def main():
    parser = argparse.ArgumentParser(description="Extract test/train images into folders")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "test_splits"),
                        help="Output directory (default: data/test_splits)")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)

    for name, cfg in DATASETS.items():
        if cfg["images"].exists() and cfg["samples"].exists() and cfg["labels"].exists():
            print(f"Extracting {name} split...")
            extract_split(
                name,
                cfg["images"],
                cfg["samples"],
                cfg["labels"],
                output_dir,
            )
        else:
            missing = []
            if not cfg["images"].exists():
                missing.append("images")
            if not cfg["samples"].exists():
                missing.append("samples.parquet")
            if not cfg["labels"].exists():
                missing.append(f"labels ({cfg['labels']})")
            print(f"Skipping {name}: missing {', '.join(missing)}")

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\nDone! Test splits are in: {output_dir}")
        print("\nTo load test against the API:")
        print(f"  python scripts/load_test.py --images {output_dir}/v2/test --url http://127.0.0.1:8080")
        print(f"  python scripts/load_test.py --images {output_dir}/v1/test --url http://127.0.0.1:8080")


if __name__ == "__main__":
    main()
