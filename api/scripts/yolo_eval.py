"""
yolo_eval.py

Run proper YOLO mAP evaluation on the v2 test set for all 4 models.
Uses ultralytics model.val() which computes COCO-standard metrics
with IoU-based box matching.

Usage:
  python scripts/yolo_eval.py
"""

from __future__ import annotations

import json
import pathlib
from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
EVAL_YAML = REPO_ROOT / "data" / "eval_v2" / "data.yaml"

MODELS = [
    "yolov8s_v2_extended",
    "yolov8s_v2",
    "yolov8s_v1_extended",
    "yolov8s_v1",
]


def main():
    if not EVAL_YAML.exists():
        print(f"ERROR: {EVAL_YAML} not found. Run extract_test_images.py first.")
        return

    all_results = {}

    for model_name in MODELS:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            print(f"Skipping {model_name}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        model = YOLO(str(model_path))
        results = model.val(
            data=str(EVAL_YAML),
            imgsz=640,
            conf=0.25,
            iou=0.7,
            verbose=False,
            plots=False,
        )

        rd = results.results_dict
        box = results.box

        per_class = {}
        if hasattr(box, 'ap') and box.ap is not None:
            for i, cls_name in enumerate(results.names.values()):
                if i < len(box.ap):
                    per_class[cls_name] = {
                        "ap50": float(box.ap50[i]) if hasattr(box, 'ap50') else 0,
                        "ap": float(box.ap[i]) if i < len(box.ap) else 0,
                        "precision": float(box.p[i]) if hasattr(box, 'p') and i < len(box.p) else 0,
                        "recall": float(box.r[i]) if hasattr(box, 'r') and i < len(box.r) else 0,
                    }

        model_results = {
            "mAP50": float(rd.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall": float(rd.get("metrics/recall(B)", 0)),
            "per_class": per_class,
        }

        f1 = 2 * model_results["precision"] * model_results["recall"] / (
            model_results["precision"] + model_results["recall"] + 1e-9)
        model_results["f1"] = f1

        all_results[model_name] = model_results

        print(f"  mAP50:     {model_results['mAP50']:.4f}")
        print(f"  mAP50-95:  {model_results['mAP50-95']:.4f}")
        print(f"  Precision: {model_results['precision']:.4f}")
        print(f"  Recall:    {model_results['recall']:.4f}")
        print(f"  F1:        {f1:.4f}")
        for cls_name, cls_data in per_class.items():
            print(f"  {cls_name:10s}  AP50={cls_data['ap50']:.4f}  AP50-95={cls_data['ap']:.4f}  "
                  f"P={cls_data['precision']:.4f}  R={cls_data['recall']:.4f}")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  ALL MODELS COMPARISON (v2 test set, 882 images)")
    print(f"{'='*80}")
    print(f"  {'Model':<25s} {'mAP50':>8s} {'mAP50-95':>10s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    best_model = None
    best_map = 0
    for name, r in all_results.items():
        marker = ""
        if r["mAP50-95"] > best_map:
            best_map = r["mAP50-95"]
            best_model = name
        print(f"  {name:<25s} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")

    print(f"\n  BEST MODEL: {best_model} (mAP50-95 = {best_map:.4f})")

    # Per-class breakdown
    print(f"\n  PER-CLASS BREAKDOWN:")
    print(f"  {'Model':<25s} {'Class':>10s} {'AP50':>8s} {'AP50-95':>10s} {'P':>8s} {'R':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for name, r in all_results.items():
        for cls_name, cls_data in r["per_class"].items():
            print(f"  {name:<25s} {cls_name:>10s} {cls_data['ap50']:>8.4f} "
                  f"{cls_data['ap']:>10.4f} {cls_data['precision']:>8.4f} {cls_data['recall']:>8.4f}")

    # Save results JSON
    out_path = REPO_ROOT / "api" / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
