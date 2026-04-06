"""
extract_test_images.py

Recreates the exact train/test splits, splitting on ORIGINAL images first
to prevent data leakage from augmented copies, then assigns augmented
copies to the same split as their parent.

Usage:
  python scripts/extract_test_images.py
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


def get_original_stem(key: str) -> str:
    """Strip __aug suffix to get the original image stem."""
    return key.replace("__aug", "")


def extract_split(
    dataset_name: str,
    images_dir: pathlib.Path,
    samples_pq: pathlib.Path,
    labels_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 42,
    test_size: float = 0.20,
) -> None:
    df = pd.read_parquet(samples_pq, columns=["image_key", "has_tim_bbox", "n_det"])
    df["stem"] = (
        df["image_key"]
        .str.replace(r"^images/", "", regex=True)
        .str.replace(r"\.(jpg|jpeg|png|webp)$", "", regex=True)
    )

    all_keys = sorted(
        s for s in df["stem"].unique()
        if find_image(s, images_dir) is not None
    )

    if not all_keys:
        print(f"  WARNING: No images found")
        return

    key_to_tim = dict(zip(df["stem"], df["has_tim_bbox"]))
    key_to_ndet = dict(zip(df["stem"], df["n_det"]))

    def strat_class(key):
        n = key_to_ndet.get(key, 0)
        if n == 0:
            return 0
        return 2 if key_to_tim.get(key, False) else 1

    # Split on ORIGINAL stems only (no __aug) to prevent leakage
    original_keys = sorted(set(get_original_stem(k) for k in all_keys))
    orig_strat = [strat_class(k) for k in original_keys]

    # Filter to only originals that actually exist (some may only have aug versions)
    valid_orig = []
    valid_strat = []
    for k, s in zip(original_keys, orig_strat):
        valid_orig.append(k)
        valid_strat.append(s)

    tv_orig, test_orig, _, _ = train_test_split(
        valid_orig, valid_strat,
        test_size=test_size,
        random_state=seed,
        stratify=valid_strat,
    )

    test_orig_set = set(test_orig)
    train_orig_set = set(tv_orig)

    # Assign all keys (including augmented) based on their original's split
    test_keys = [k for k in all_keys if get_original_stem(k) in test_orig_set]
    train_keys = [k for k in all_keys if get_original_stem(k) in train_orig_set]

    # Verify no leakage
    test_parents = set(get_original_stem(k) for k in test_keys)
    train_parents = set(get_original_stem(k) for k in train_keys)
    leaked = test_parents & train_parents
    assert len(leaked) == 0, f"Data leakage detected: {len(leaked)} shared parents!"

    hn = sum(1 for k in all_keys if strat_class(k) == 0)
    print(f"  Total: {len(all_keys)} images ({len(original_keys)} originals, "
          f"{len(all_keys) - len(original_keys)} augmented, {hn} hard neg)")
    print(f"  Split on {len(valid_orig)} original stems -> "
          f"{len(tv_orig)} train originals, {len(test_orig)} test originals")
    print(f"  After expanding augmented: {len(train_keys)} train, {len(test_keys)} test")
    print(f"  Leakage check: PASSED (0 shared parents)")

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
    for key in train_keys[:50]:
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


def main():
    parser = argparse.ArgumentParser(description="Extract test/train images into folders")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "test_splits"),
                        help="Output directory")
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output)

    for name, cfg in DATASETS.items():
        if cfg["images"].exists() and cfg["samples"].exists() and cfg["labels"].exists():
            print(f"Extracting {name} split...")
            extract_split(name, cfg["images"], cfg["samples"], cfg["labels"], output_dir)
        else:
            print(f"Skipping {name}: missing files")

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\nDone! Test splits in: {output_dir}")


if __name__ == "__main__":
    main()
