#!/usr/bin/env python3
"""
Build a FiftyOne dataset from samples.parquet + detections.parquet.

Example:
  python build_fiftyone_dataset.py \
    --samples data/index/samples.parquet \
    --detections data/index/detections.parquet \
    --images-root data/ \
    --dataset th_cups_v1 \
    --overwrite \
    --persistent
"""

import argparse
import os
from typing import Dict, List

import pandas as pd
import fiftyone as fo
from datetime import datetime, timezone

#dumb datetime fix helper
def parse_dt(v):
    """
    Accepts:
      - pandas Timestamp
      - python datetime
      - ISO-8601 string like '2026-02-19T04:18:14.701513Z'
    Returns:
      - timezone-aware datetime (UTC) or None
    """
    if v is None:
        return None

    # pandas Timestamp has .to_pydatetime()
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()

    if isinstance(v, datetime):
        # ensure tz-aware (store in UTC)
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Handle trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return None

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def pct_bbox_to_rel(x_pct: float, y_pct: float, w_pct: float, h_pct: float):
    # Convert percent to [0,1] and clamp for safety
    x = clamp01(float(x_pct) / 100.0)
    y = clamp01(float(y_pct) / 100.0)
    w = clamp01(float(w_pct) / 100.0)
    h = clamp01(float(h_pct) / 100.0)

    # Ensure bbox stays in bounds (optional but helps downstream)
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)

    return [x, y, w, h]


def build_detection_label(row: pd.Series) -> str:
    return "TimHortonsCup" if bool(row.get("is_tim", False)) else "NonTimHortonsCup"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to samples.parquet")
    ap.add_argument("--detections", required=True, help="Path to detections.parquet")
    ap.add_argument("--images-root", required=True, help="Root folder that contains the 'images/' dir (e.g. data/raw)")
    ap.add_argument("--dataset", required=True, help="FiftyOne dataset name to create/update")
    ap.add_argument("--field", default="ground_truth", help="Label field name (default: ground_truth)")
    ap.add_argument("--overwrite", action="store_true", help="Delete existing dataset with same name")
    ap.add_argument("--persistent", action="store_true", help="Make dataset persistent in FiftyOne DB")
    ap.add_argument("--skip-missing-images", action="store_true", help="Skip samples whose image file is missing")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    return ap.parse_args()


def main():
    args = parse_args()

    samples_df = pd.read_parquet(args.samples)
    det_df = pd.read_parquet(args.detections)

    if args.max_samples is not None:
        samples_df = samples_df.head(args.max_samples)

    # Build a map: sample_id -> list[Detection]
    det_df = det_df[det_df["sample_id"].isin(set(samples_df["sample_id"]))].copy()

    dets_by_sample: Dict[str, List[fo.Detection]] = {}
    if not det_df.empty:
        for row in det_df.itertuples(index=False):
            sample_id = row.sample_id

            bbox = pct_bbox_to_rel(row.x_pct, row.y_pct, row.w_pct, row.h_pct)
            label = "TimHortonsCup" if bool(row.is_tim) else "NonTimHortonsCup"

            det = fo.Detection(
                label=label,
                bounding_box=bbox,
            )

            # Preserve rich per-box metadata
            det["ls_task_id"] = int(row.ls_task_id) if row.ls_task_id is not None else None
            det["ls_annotation_id"] = int(row.ls_annotation_id) if row.ls_annotation_id is not None else None
            det["ls_result_id"] = str(row.ls_result_id) if row.ls_result_id is not None else None
            det["image_key"] = str(row.image_key)
            det["is_cup"] = bool(row.is_cup)
            det["is_tim"] = bool(row.is_tim)
            det["area_frac"] = float(row.area_frac) if row.area_frac is not None else None

            dets_by_sample.setdefault(sample_id, []).append(det)

    # Dataset lifecycle
    if args.overwrite and fo.dataset_exists(args.dataset):
        fo.delete_dataset(args.dataset)

    dataset = fo.Dataset(args.dataset) if not fo.dataset_exists(args.dataset) else fo.load_dataset(args.dataset)

    # If we want to clean rebuild without deleting uncomment
    # dataset.clear()

    images_root = os.path.abspath(args.images_root)

    n_added = 0
    n_skipped_missing = 0

    samples_to_add = []
    for row in samples_df.itertuples(index=False):
        sample_id = row.sample_id
        filepath = os.path.join(images_root, sample_id)

        exists = os.path.exists(filepath)
        if not exists and args.skip_missing_images:
            n_skipped_missing += 1
            print(f"[fiftyone] Skipping missing image: {filepath}")
            continue

        s = fo.Sample(filepath=filepath)

        s["sample_id"] = str(sample_id)
        s["image_key"] = str(getattr(row, "image_key", sample_id))
        s["image_raw"] = str(getattr(row, "image_raw", ""))

        # Provenance
        s["ls_task_id"] = int(getattr(row, "ls_task_id", -1))
        s["created_at"] = parse_dt(getattr(row, "created_at", None))
        s["updated_at"] = parse_dt(getattr(row, "updated_at", None))

        # Diagnostics
        for field in [
            "type",
            "background",
            "cup_percentage",
            "orientation",
            "deform",
            "blur",
            "occluded",
            "count",
            "brand",
            "n_det",
            "n_cup",
            "computed_cup_percantage",
            "cup_class",
        ]:
            if hasattr(row, field):
                s[field] = getattr(row, field)

        # Attach detections
        det_list = dets_by_sample.get(sample_id, [])
        s[args.field] = fo.Detections(detections=det_list)

        samples_to_add.append(s)
        n_added += 1

    dataset.add_samples(samples_to_add)
    dataset.persistent = bool(args.persistent)

    # Basic sanity output for DVC logs
    print(f"[fiftyone] dataset='{dataset.name}'")
    print(f"[fiftyone] samples_added={n_added}")
    print(f"[fiftyone] samples_skipped_missing={n_skipped_missing}")
    print(f"[fiftyone] label_field='{args.field}'")
    print(f"[fiftyone] persistent={dataset.persistent}")

    print(f"[fiftyone] total_samples_in_dataset={len(dataset)}")


if __name__ == "__main__":
    main()