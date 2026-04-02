"""
generate_yolo_labels.py

Convert bounding-box annotations from detections.parquet into YOLO-format
.txt label files (one per image).

YOLO format per line:  class_id  center_x  center_y  width  height
All values normalised to [0, 1].

Class mapping:
  0 = NonTimHortonsCup  (label == "Cup")
  1 = TimHortonsCup     (label == "Tim Hortons")

Usage:
  python generate_yolo_labels.py                      # defaults
  python generate_yolo_labels.py --detections path/to/detections.parquet
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).parent.resolve()

LABEL_MAP = {
    "Cup": 0,
    "Tim Hortons": 1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate YOLO .txt labels from detections.parquet"
    )
    p.add_argument(
        "--detections",
        default=str(REPO_ROOT / "data" / "index" / "detections.parquet"),
        help="Path to detections.parquet",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "derived" / "yolo" / "labels"),
        help="Directory to write .txt label files",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Remove all existing .txt files in output dir before writing",
    )
    return p.parse_args()


def main():
    args = parse_args()

    det_path = pathlib.Path(args.detections)
    out_dir = pathlib.Path(args.output_dir)

    if not det_path.exists():
        raise FileNotFoundError(f"Detections parquet not found: {det_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        removed = 0
        for f in out_dir.glob("*.txt"):
            f.unlink()
            removed += 1
        if removed:
            print(f"Cleaned {removed} existing .txt files from {out_dir}")

    det = pd.read_parquet(det_path)
    print(f"Loaded {len(det)} detections across {det['sample_id'].nunique()} images")

    skipped_labels = set()
    written = 0
    skipped_rows = 0

    for sample_id, group in det.groupby("sample_id"):
        stem = pathlib.Path(str(sample_id)).stem
        lines = []

        for _, row in group.iterrows():
            label = row["label"]
            class_id = LABEL_MAP.get(label)
            if class_id is None:
                skipped_labels.add(label)
                skipped_rows += 1
                continue

            cx = (row["x_pct"] + row["w_pct"] / 2.0) / 100.0
            cy = (row["y_pct"] + row["h_pct"] / 2.0) / 100.0
            w = row["w_pct"] / 100.0
            h = row["h_pct"] / 100.0

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            written += 1

    print(f"\nWrote {written} label files to {out_dir}")
    if skipped_labels:
        print(f"Skipped {skipped_rows} rows with unknown labels: {skipped_labels}")

    label_counts = det["label"].value_counts()
    for label, count in label_counts.items():
        cls = LABEL_MAP.get(label, "?")
        print(f"  {label} (class {cls}): {count} boxes")


if __name__ == "__main__":
    main()
