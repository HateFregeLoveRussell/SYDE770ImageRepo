"""
yolo_eval.py

Run proper YOLO mAP evaluation + confusion matrix breakdown on all 4 models.
Reports mAP50, mAP50-95, and per-class TP/FP/FN counts.

Usage:
  python scripts/yolo_eval.py
"""

from __future__ import annotations

import json
import pathlib
from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
EVAL_YAML = REPO_ROOT / "data" / "eval_v2_split_actual" / "data.yaml"
LABELS_DIR = REPO_ROOT / "data" / "eval_v2_split_actual" / "labels" / "test"
IMAGES_DIR = REPO_ROOT / "data" / "eval_v2_split_actual" / "images" / "test"

MODELS = [
    "yolov8s_v2_extended",
    "yolov8s_v2",
    "yolov8s_v1_extended",
    "yolov8s_v1",
]

CLASS_NAMES = {0: "cup", 1: "timmies"}


def count_gt_labels(labels_dir: pathlib.Path) -> dict:
    """Count ground truth detections."""
    counts = {"cup": 0, "timmies": 0, "images_with_cup": 0, "images_with_timmies": 0, "total_images": 0}
    for txt in sorted(labels_dir.glob("*.txt")):
        counts["total_images"] += 1
        has_cup = False
        has_tim = False
        for line in txt.read_text().strip().split("\n"):
            if not line.strip():
                continue
            cls = int(line.strip().split()[0])
            if cls == 0:
                counts["cup"] += 1
                has_cup = True
            else:
                counts["timmies"] += 1
                has_tim = True
        if has_cup:
            counts["images_with_cup"] += 1
        if has_tim:
            counts["images_with_timmies"] += 1
    return counts


def run_detailed_eval(model_name: str, images_dir: pathlib.Path,
                      labels_dir: pathlib.Path, conf: float = 0.25) -> dict:
    """Run model on each image and compare detections to GT by class count."""
    model_path = MODELS_DIR / f"{model_name}.pt"
    model = YOLO(str(model_path))

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    stats = {
        "cup": {"tp": 0, "fp": 0, "fn": 0},
        "timmies": {"tp": 0, "fp": 0, "fn": 0},
    }
    total_gt = 0
    total_pred = 0
    images_perfect = 0

    for img_path in image_files:
        gt_path = labels_dir / f"{img_path.stem}.txt"
        gt = {0: 0, 1: 0}
        if gt_path.exists():
            for line in gt_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                cls = int(line.strip().split()[0])
                gt[cls] += 1

        results = model.predict(source=str(img_path), conf=conf, iou=0.7,
                                imgsz=640, verbose=False)
        pred = {0: 0, 1: 0}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.item())
                pred[cls] = pred.get(cls, 0) + 1

        perfect = True
        for cls_id, cls_name in CLASS_NAMES.items():
            matched = min(pred[cls_id], gt[cls_id])
            fp = max(0, pred[cls_id] - gt[cls_id])
            fn = max(0, gt[cls_id] - pred[cls_id])
            stats[cls_name]["tp"] += matched
            stats[cls_name]["fp"] += fp
            stats[cls_name]["fn"] += fn
            if fp > 0 or fn > 0:
                perfect = False

        total_gt += gt[0] + gt[1]
        total_pred += pred[0] + pred[1]
        if perfect:
            images_perfect += 1

    return {
        "per_class": stats,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "images_total": len(image_files),
        "images_perfect": images_perfect,
    }


def main():
    if not EVAL_YAML.exists():
        print(f"ERROR: {EVAL_YAML} not found")
        return

    gt_counts = count_gt_labels(LABELS_DIR)
    print(f"Ground Truth Summary ({gt_counts['total_images']} labelled images):")
    print(f"  Cup detections:     {gt_counts['cup']} (in {gt_counts['images_with_cup']} images)")
    print(f"  Timmies detections: {gt_counts['timmies']} (in {gt_counts['images_with_timmies']} images)")
    print(f"  Hard negatives:     80 images (no labels)")

    all_results = {}

    for model_name in MODELS:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            continue

        print(f"\n{'='*70}")
        print(f"  {model_name}")
        print(f"{'='*70}")

        # mAP evaluation
        model = YOLO(str(model_path))
        val_results = model.val(data=str(EVAL_YAML), imgsz=640, conf=0.25,
                                iou=0.7, verbose=False, plots=False)
        rd = val_results.results_dict
        box = val_results.box

        map50 = float(rd.get("metrics/mAP50(B)", 0))
        map50_95 = float(rd.get("metrics/mAP50-95(B)", 0))
        precision = float(rd.get("metrics/precision(B)", 0))
        recall = float(rd.get("metrics/recall(B)", 0))
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"  mAP50:     {map50:.4f}")
        print(f"  mAP50-95:  {map50_95:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

        per_class_ap = {}
        if hasattr(box, 'ap') and box.ap is not None:
            for i, cls_name in enumerate(val_results.names.values()):
                if i < len(box.ap):
                    per_class_ap[cls_name] = {
                        "ap50": float(box.ap50[i]),
                        "ap50_95": float(box.ap[i]),
                        "precision": float(box.p[i]),
                        "recall": float(box.r[i]),
                    }
                    print(f"  {cls_name:10s}  AP50={box.ap50[i]:.4f}  AP50-95={box.ap[i]:.4f}  "
                          f"P={box.p[i]:.4f}  R={box.r[i]:.4f}")

        # Detailed TP/FP/FN counts
        print(f"\n  Detection Breakdown (count-based matching):")
        detail = run_detailed_eval(model_name, IMAGES_DIR, LABELS_DIR)

        for cls_name in ["cup", "timmies"]:
            s = detail["per_class"][cls_name]
            total_cls = s["tp"] + s["fn"]
            p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
            r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
            print(f"    {cls_name:10s}  TP={s['tp']:4d}  FP={s['fp']:4d}  FN={s['fn']:4d}  "
                  f"(GT={total_cls}, Pred={s['tp']+s['fp']})")

        tp_all = sum(s["tp"] for s in detail["per_class"].values())
        fp_all = sum(s["fp"] for s in detail["per_class"].values())
        fn_all = sum(s["fn"] for s in detail["per_class"].values())
        print(f"    {'TOTAL':10s}  TP={tp_all:4d}  FP={fp_all:4d}  FN={fn_all:4d}  "
              f"(GT={detail['total_gt']}, Pred={detail['total_pred']})")
        print(f"    Perfect images (exact match): {detail['images_perfect']}/{detail['images_total']} "
              f"({detail['images_perfect']/detail['images_total']*100:.1f}%)")

        all_results[model_name] = {
            "mAP50": map50, "mAP50-95": map50_95,
            "precision": precision, "recall": recall, "f1": f1,
            "per_class_ap": per_class_ap,
            "detection_counts": detail,
        }

    # Comparison table
    print(f"\n\n{'='*70}")
    print(f"  FINAL MODEL COMPARISON (v2 test set, leak-free split)")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'mAP50':>7s} {'mAP50-95':>9s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>5s} {'FP':>4s} {'FN':>4s} {'Perfect':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*8}")

    best_model = None
    best_map = 0
    for name, r in all_results.items():
        d = r["detection_counts"]
        tp = sum(s["tp"] for s in d["per_class"].values())
        fp = sum(s["fp"] for s in d["per_class"].values())
        fn = sum(s["fn"] for s in d["per_class"].values())
        pct = f"{d['images_perfect']}/{d['images_total']}"
        if r["mAP50-95"] > best_map:
            best_map = r["mAP50-95"]
            best_model = name
        print(f"  {name:<25s} {r['mAP50']:>7.4f} {r['mAP50-95']:>9.4f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{tp:>5d} {fp:>4d} {fn:>4d} {pct:>8s}")

    print(f"\n  BEST: {best_model} (mAP50-95 = {best_map:.4f})")

    out_path = REPO_ROOT / "api" / "eval_results.json"
    serializable = {}
    for name, r in all_results.items():
        serializable[name] = {
            "mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"],
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "per_class_ap": r["per_class_ap"],
            "total_gt": r["detection_counts"]["total_gt"],
            "total_pred": r["detection_counts"]["total_pred"],
            "images_perfect": r["detection_counts"]["images_perfect"],
            "images_total": r["detection_counts"]["images_total"],
            "cup_tp": r["detection_counts"]["per_class"]["cup"]["tp"],
            "cup_fp": r["detection_counts"]["per_class"]["cup"]["fp"],
            "cup_fn": r["detection_counts"]["per_class"]["cup"]["fn"],
            "timmies_tp": r["detection_counts"]["per_class"]["timmies"]["tp"],
            "timmies_fp": r["detection_counts"]["per_class"]["timmies"]["fp"],
            "timmies_fn": r["detection_counts"]["per_class"]["timmies"]["fn"],
        }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
