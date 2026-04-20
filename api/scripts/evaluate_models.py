"""
evaluate_models.py

Run both v2 models on the test set and compare predictions against
ground truth YOLO labels. Reports precision, recall, F1, and detection counts.

Usage:
  python scripts/evaluate_models.py
  python scripts/evaluate_models.py --conf 0.25 --iou 0.5
"""

from __future__ import annotations

import argparse
import pathlib

from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LABELS_V2 = REPO_ROOT / "data" / "derived" / "yolo" / "labels_v2"
TEST_IMAGES = REPO_ROOT / "data" / "test_splits" / "v2" / "test"
MODELS_DIR = REPO_ROOT / "models"

MODELS = [
    "yolov8s_v2_extended",
    "yolov8s_v2",
]


def load_gt_labels(label_path: pathlib.Path) -> list[dict]:
    """Load YOLO format ground truth: class_id cx cy w h"""
    if not label_path.exists():
        return []
    detections = []
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        detections.append({"class_id": cls_id})
    return detections


def evaluate_model(model_name: str, images_dir: pathlib.Path,
                   labels_dir: pathlib.Path, conf: float, iou: float) -> dict:
    """Run model on images and compare with ground truth labels."""
    model_path = MODELS_DIR / f"{model_name}.pt"
    model = YOLO(str(model_path))

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    tp_cup, fp_cup, fn_cup = 0, 0, 0
    tp_tim, fp_tim, fn_tim = 0, 0, 0
    total_images = 0
    images_with_gt = 0
    images_correct_count = 0
    total_pred_dets = 0
    total_gt_dets = 0

    for img_path in image_files:
        total_images += 1
        stem = img_path.stem

        # Ground truth
        gt_label_path = labels_dir / f"{stem}.txt"
        gt_dets = load_gt_labels(gt_label_path)
        gt_cups = sum(1 for d in gt_dets if d["class_id"] == 0)
        gt_tims = sum(1 for d in gt_dets if d["class_id"] == 1)
        total_gt_dets += len(gt_dets)

        if gt_dets:
            images_with_gt += 1

        # Predictions
        results = model.predict(source=str(img_path), conf=conf, iou=iou,
                                imgsz=640, verbose=False)
        result = results[0]
        pred_cups = 0
        pred_tims = 0
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id == 0:
                    pred_cups += 1
                elif cls_id == 1:
                    pred_tims += 1
        total_pred_dets += pred_cups + pred_tims

        # Count-based matching (simplified: match by class count)
        matched_cups = min(pred_cups, gt_cups)
        matched_tims = min(pred_tims, gt_tims)

        tp_cup += matched_cups
        fp_cup += max(0, pred_cups - gt_cups)
        fn_cup += max(0, gt_cups - pred_cups)

        tp_tim += matched_tims
        fp_tim += max(0, pred_tims - gt_tims)
        fn_tim += max(0, gt_tims - pred_tims)

        if pred_cups == gt_cups and pred_tims == gt_tims:
            images_correct_count += 1

    # Compute metrics
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    p_cup, r_cup, f1_cup = prf(tp_cup, fp_cup, fn_cup)
    p_tim, r_tim, f1_tim = prf(tp_tim, fp_tim, fn_tim)

    tp_all = tp_cup + tp_tim
    fp_all = fp_cup + fp_tim
    fn_all = fn_cup + fn_tim
    p_all, r_all, f1_all = prf(tp_all, fp_all, fn_all)

    return {
        "model": model_name,
        "total_images": total_images,
        "images_with_gt": images_with_gt,
        "images_correct_count": images_correct_count,
        "total_gt_dets": total_gt_dets,
        "total_pred_dets": total_pred_dets,
        "cup": {"tp": tp_cup, "fp": fp_cup, "fn": fn_cup,
                "precision": p_cup, "recall": r_cup, "f1": f1_cup},
        "timmies": {"tp": tp_tim, "fp": fp_tim, "fn": fn_tim,
                    "precision": p_tim, "recall": r_tim, "f1": f1_tim},
        "overall": {"tp": tp_all, "fp": fp_all, "fn": fn_all,
                    "precision": p_all, "recall": r_all, "f1": f1_all},
    }


def print_results(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  MODEL: {r['model']}")
    print(f"{'='*60}")
    print(f"  Images: {r['total_images']} ({r['images_with_gt']} with GT detections)")
    print(f"  GT detections: {r['total_gt_dets']}  |  Predicted: {r['total_pred_dets']}")
    print(f"  Images with exact count match: {r['images_correct_count']}/{r['total_images']} "
          f"({r['images_correct_count']/r['total_images']*100:.1f}%)")
    print()
    for cls_name in ["cup", "timmies", "overall"]:
        c = r[cls_name]
        print(f"  {cls_name:10s}  P={c['precision']:.3f}  R={c['recall']:.3f}  "
              f"F1={c['f1']:.3f}  (TP={c['tp']} FP={c['fp']} FN={c['fn']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Evaluating on {TEST_IMAGES}")
    print(f"Ground truth: {LABELS_V2}")
    print(f"Conf threshold: {args.conf}, IoU threshold: {args.iou}")

    all_results = []
    for model_name in MODELS:
        print(f"\nRunning {model_name}...")
        r = evaluate_model(model_name, TEST_IMAGES, LABELS_V2, args.conf, args.iou)
        print_results(r)
        all_results.append(r)

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s}", end="")
    for r in all_results:
        print(f"  {r['model']:>20s}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in all_results:
        print(f"  {'-'*20}", end="")
    print()

    for metric_name, key_path in [
        ("Predicted detections", "total_pred_dets"),
        ("GT detections", "total_gt_dets"),
        ("Overall Precision", ("overall", "precision")),
        ("Overall Recall", ("overall", "recall")),
        ("Overall F1", ("overall", "f1")),
        ("Cup Precision", ("cup", "precision")),
        ("Cup Recall", ("cup", "recall")),
        ("Cup F1", ("cup", "f1")),
        ("Timmies Precision", ("timmies", "precision")),
        ("Timmies Recall", ("timmies", "recall")),
        ("Timmies F1", ("timmies", "f1")),
    ]:
        print(f"  {metric_name:<25s}", end="")
        for r in all_results:
            if isinstance(key_path, tuple):
                val = r[key_path[0]][key_path[1]]
                print(f"  {val:>20.3f}", end="")
            else:
                val = r[key_path]
                print(f"  {val:>20d}", end="")
        print()


if __name__ == "__main__":
    main()
