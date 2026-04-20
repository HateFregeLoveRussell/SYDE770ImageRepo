"""
build_fiftyone_predictions.py

Build a FiftyOne dataset with both ground_truth and predictions label fields
for interactive comparison. Runs evaluate_detections() to tag TP/FP/FN on
each detection and compute mAP, then launches the FiftyOne App.

Usage:
  python build_fiftyone_predictions.py \
      --predictions predictions/seed0 \
      --images-root data/

  python build_fiftyone_predictions.py \
      --predictions predictions/seed0 \
      --images-root data/ \
      --dataset test_eval_v1 \
      --no-launch
"""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Dict, List

import pandas as pd
import fiftyone as fo


def parse_args():
    p = argparse.ArgumentParser(
        description="Build FiftyOne dataset with GT + predictions for comparison"
    )
    p.add_argument(
        "--predictions", default="predictions/seed0",
        help="Directory containing samples.parquet, detections.parquet, "
             "and ground_truth.parquet",
    )
    p.add_argument(
        "--images-root", default="data/",
        help="Root folder containing the 'images/' directory",
    )
    p.add_argument(
        "--dataset", default="test_predictions_v1",
        help="FiftyOne dataset name (default: test_predictions_v1)",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing dataset with same name")
    p.add_argument("--persistent", action="store_true", default=True,
                   help="Make dataset persistent in FiftyOne DB (default: True)")
    p.add_argument("--no-launch", action="store_true",
                   help="Skip launching the FiftyOne App")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for evaluation (default: 0.5)")
    return p.parse_args()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def pct_bbox_to_rel(x_pct: float, y_pct: float, w_pct: float, h_pct: float):
    x = clamp01(float(x_pct) / 100.0)
    y = clamp01(float(y_pct) / 100.0)
    w = clamp01(float(w_pct) / 100.0)
    h = clamp01(float(h_pct) / 100.0)
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    return [x, y, w, h]


def to_str_or_none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v)


def to_bool_or_none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def build_detections_map(det_df: pd.DataFrame, with_confidence: bool = False
                         ) -> Dict[str, List[fo.Detection]]:
    """Group a detections dataframe into {sample_id: [fo.Detection, ...]}."""
    dets_by_sample: Dict[str, List[fo.Detection]] = {}
    for row in det_df.itertuples(index=False):
        bbox = pct_bbox_to_rel(row.x_pct, row.y_pct, row.w_pct, row.h_pct)
        det = fo.Detection(label=row.label, bounding_box=bbox)
        det["is_cup"] = bool(row.is_cup)
        det["is_tim"] = bool(row.is_tim)
        det["area_frac"] = float(row.area_frac) if row.area_frac is not None else None

        if with_confidence and hasattr(row, "confidence"):
            det.confidence = float(row.confidence)

        dets_by_sample.setdefault(row.sample_id, []).append(det)
    return dets_by_sample


def main():
    args = parse_args()
    pred_dir = pathlib.Path(args.predictions)
    images_root = os.path.abspath(args.images_root)

    samples_df = pd.read_parquet(pred_dir / "samples.parquet")
    gt_df = pd.read_parquet(pred_dir / "ground_truth.parquet")
    pred_df = pd.read_parquet(pred_dir / "detections.parquet")

    print(f"Loaded {len(samples_df)} samples, "
          f"{len(gt_df)} GT detections, {len(pred_df)} predictions")

    gt_map = build_detections_map(gt_df, with_confidence=False)
    pred_map = build_detections_map(pred_df, with_confidence=True)

    if args.overwrite and fo.dataset_exists(args.dataset):
        fo.delete_dataset(args.dataset)

    dataset = (fo.Dataset(args.dataset) if not fo.dataset_exists(args.dataset)
               else fo.load_dataset(args.dataset))

    diag_str_fields = [
        "type", "background", "cup_percantage", "orientation",
        "deform", "blur", "occluded", "count", "brand",
    ]
    diag_bool_fields = ["lid", "sleeve"]

    fo_samples = []
    n_skipped = 0
    for row in samples_df.itertuples(index=False):
        sid = row.sample_id
        filepath = os.path.join(images_root, sid)

        if not os.path.exists(filepath):
            n_skipped += 1
            continue

        s = fo.Sample(filepath=filepath)
        s["sample_id"] = str(sid)

        for field in diag_str_fields:
            if hasattr(row, field):
                s[field] = to_str_or_none(getattr(row, field))
        for field in diag_bool_fields:
            if hasattr(row, field):
                s[field] = to_bool_or_none(getattr(row, field))

        s["ground_truth"] = fo.Detections(detections=gt_map.get(sid, []))
        s["predictions"] = fo.Detections(detections=pred_map.get(sid, []))
        fo_samples.append(s)

    dataset.add_samples(fo_samples)
    dataset.persistent = args.persistent

    print(f"Added {len(fo_samples)} samples to dataset '{args.dataset}' "
          f"(skipped {n_skipped} missing images)")

    print(f"\nRunning evaluate_detections (IoU={args.iou})...")
    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou=args.iou,
        compute_mAP=True,
    )

    print("\n--- Evaluation Results ---")
    results.print_report()
    print(f"\nmAP@{args.iou}: {results.mAP():.4f}")

    print(f"\nDataset: '{args.dataset}' ({len(dataset)} samples)")
    print("Fields: ground_truth, predictions, eval, eval_tp, eval_fp, eval_fn")
    print("\nUseful FiftyOne filters:")
    print("  - False positives: dataset.filter_labels('predictions', F('eval') == 'fp')")
    print("  - False negatives: dataset.filter_labels('ground_truth', F('eval') == 'fn')")
    print("  - Low confidence:  dataset.filter_labels('predictions', F('confidence') < 0.5)")

    if not args.no_launch:
        print("\nLaunching FiftyOne App...")
        session = fo.launch_app(dataset)
        session.wait()
    else:
        print("\nSkipped app launch (--no-launch). Open later with:")
        print(f"  import fiftyone as fo")
        print(f"  dataset = fo.load_dataset('{args.dataset}')")
        print(f"  session = fo.launch_app(dataset)")


if __name__ == "__main__":
    main()
