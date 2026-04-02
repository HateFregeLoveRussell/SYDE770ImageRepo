"""
predict_to_parquet.py

Run YOLO inference on a set of images and save predictions as:
  - predictions.parquet  (one row per detection, FiftyOne-compatible schema)
  - labels/*.txt         (YOLO-format .txt files per image)

Usage:
  python predict_to_parquet.py \
      --weights runs/final/seed0/weights/best.pt \
      --images data/derived/yolo/test/images \
      --output predictions/seed0

  python predict_to_parquet.py \
      --weights runs/final/seed0/weights/best.pt \
      --images data/derived/yolo/test/images \
      --output predictions/seed0 \
      --conf 0.25 --iou 0.7
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from ultralytics import YOLO

CLASS_NAMES = {0: "NonTimHortonsCup", 1: "TimHortonsCup"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run YOLO inference and save predictions as parquet + YOLO .txt"
    )
    p.add_argument("--weights", required=True,
                   help="Path to YOLO checkpoint (e.g. best.pt)")
    p.add_argument("--images", required=True,
                   help="Directory of images to run inference on")
    p.add_argument("--output", required=True,
                   help="Output directory for predictions.parquet and labels/")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU threshold (default: 0.7)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (default: 640)")
    p.add_argument("--source-samples",
                   default="data/augmented_dataset_v1/samples.parquet",
                   help="Path to source samples.parquet to join diagnostic labels from "
                        "(default: data/augmented_dataset_v1/samples.parquet)")
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = pathlib.Path(args.weights)
    images_dir = pathlib.Path(args.images)
    output_dir = pathlib.Path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    print(f"Running inference on {len(image_files)} images...")
    print(f"  Weights : {weights_path}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print(f"  ImgSz   : {args.imgsz}")

    results = model.predict(
        source=str(images_dir),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )

    rows = []
    for result in results:
        img_path = pathlib.Path(result.path)
        stem = img_path.stem
        sample_id = f"images/{img_path.name}"

        yolo_lines = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                img_h, img_w = result.orig_shape
                x_pct = (x1 / img_w) * 100.0
                y_pct = (y1 / img_h) * 100.0
                w_pct = ((x2 - x1) / img_w) * 100.0
                h_pct = ((y2 - y1) / img_h) * 100.0

                cx_norm = ((x1 + x2) / 2) / img_w
                cy_norm = ((y1 + y2) / 2) / img_h
                w_norm = (x2 - x1) / img_w
                h_norm = (y2 - y1) / img_h

                label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                rows.append({
                    "sample_id": sample_id,
                    "image_key": sample_id,
                    "label": label,
                    "confidence": conf,
                    "class_id": cls_id,
                    "x_pct": x_pct,
                    "y_pct": y_pct,
                    "w_pct": w_pct,
                    "h_pct": h_pct,
                    "x1_abs": x1,
                    "y1_abs": y1,
                    "x2_abs": x2,
                    "y2_abs": y2,
                    "img_width": img_w,
                    "img_height": img_h,
                })

                yolo_lines.append(
                    f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} "
                    f"{w_norm:.6f} {h_norm:.6f} {conf:.4f}"
                )

        txt_path = labels_dir / f"{stem}.txt"
        if yolo_lines:
            txt_path.write_text("\n".join(yolo_lines) + "\n")
        else:
            txt_path.write_text("")

    df = pd.DataFrame(rows)

    # -- Save raw predictions parquet --
    raw_path = output_dir / "predictions.parquet"
    df.to_parquet(raw_path, index=False)

    # -- Save detections.parquet (matching augmented_dataset_v1 schema) --
    det_rows = []
    if len(df) > 0:
        for _, row in df.iterrows():
            is_tim = row["label"] == "TimHortonsCup"
            area = (row["w_pct"] / 100.0) * (row["h_pct"] / 100.0)
            det_rows.append({
                "sample_id": row["sample_id"],
                "image_key": row["image_key"],
                "ls_task_id": pd.NA,
                "label": row["label"],
                "x_pct": row["x_pct"],
                "y_pct": row["y_pct"],
                "w_pct": row["w_pct"],
                "h_pct": row["h_pct"],
                "ls_annotation_id": pd.NA,
                "ls_result_id": pd.NA,
                "is_cup": True,
                "is_tim": is_tim,
                "area_frac": area,
                "confidence": row["confidence"],
            })
    det_df = pd.DataFrame(det_rows)
    det_df.to_parquet(output_dir / "detections.parquet", index=False)

    # -- Build samples.parquet (matching augmented_dataset_v1 schema) --
    # Compute per-image prediction stats
    all_image_ids = sorted(f"images/{f.name}" for f in image_files)
    pred_stats = []
    for img_id in all_image_ids:
        img_dets = df[df["sample_id"] == img_id] if len(df) > 0 else pd.DataFrame()
        n_det = len(img_dets)
        n_cup = n_det
        has_tim = bool((img_dets["label"] == "TimHortonsCup").any()) if n_det > 0 else False
        cup_areas = (
            ((img_dets["w_pct"] / 100.0) * (img_dets["h_pct"] / 100.0)).tolist()
            if n_det > 0 else []
        )
        pred_stats.append({
            "sample_id": img_id,
            "image_key": img_id,
            "image_exists": True,
            "n_det": n_det,
            "n_cup": n_cup,
            "has_tim_bbox": has_tim,
            "max_cup_area": max(cup_areas) if cup_areas else 0.0,
            "sum_cup_area": sum(cup_areas) if cup_areas else 0.0,
            "cup_area_frac": max(cup_areas) if cup_areas else 0.0,
            "cup_class": "TimHortonsCup" if has_tim else (
                "NonTimHortonsCup" if n_cup > 0 else pd.NA),
        })
    samples_df = pd.DataFrame(pred_stats)

    # Join diagnostic labels from source samples
    src_path = pathlib.Path(args.source_samples)
    if src_path.exists():
        src = pd.read_parquet(src_path)
        diag_cols = [
            "sample_id", "image_raw", "ls_task_id", "created_at",
            "updated_at", "ls_project", "total_annotations",
            "cancelled_annotations", "type", "background",
            "cup_percantage", "orientation", "deform", "blur",
            "occluded", "count", "brand", "expected_count_from_boxes",
            "computed_cup_percantage", "is_tim_hortons_bbox",
            "lid", "sleeve", "lid_ls_task_id", "lid_ls_project_id",
            "lid_completed_by", "lid_updated_at", "sleeve_ls_task_id",
            "sleeve_ls_project_id", "sleeve_completed_by",
            "sleeve_updated_at", "source_sample_id",
            "augmentation_name", "crop_variant", "orig_max_area_frac",
            "post_geom_max_area_frac", "target_met_after_geom",
            "photo_group_applied", "photo_choice",
            "degrade_group_applied", "degrade_choice",
        ]
        available = [c for c in diag_cols if c in src.columns]
        src_subset = src[available].drop_duplicates(subset="sample_id")
        samples_df = samples_df.merge(src_subset, on="sample_id", how="left")

        matched = samples_df["type"].notna().sum()
        unmatched = len(samples_df) - matched
        print(f"  Joined diagnostic labels from {src_path}: "
              f"{matched}/{len(samples_df)} matched")
        if unmatched > 0:
            print(f"  WARNING: {unmatched} samples had no diagnostic labels "
                  f"(not found in source parquet)")
    else:
        print(f"  WARNING: source samples not found: {src_path}, "
              f"diagnostic labels will be missing")

    samples_df.to_parquet(output_dir / "samples.parquet", index=False)

    n_images_with_dets = df["sample_id"].nunique() if len(df) > 0 else 0
    print(f"\nDone!")
    print(f"  Total detections : {len(df)}")
    print(f"  Images with dets : {n_images_with_dets}/{len(image_files)}")
    if len(df) > 0:
        print(f"  Label distribution:")
        for label, count in df["label"].value_counts().items():
            print(f"    {label}: {count}")
        print(f"  Mean confidence  : {df['confidence'].mean():.3f}")
    print(f"\nOutput files:")
    print(f"  {output_dir / 'predictions.parquet'}  (raw predictions)")
    print(f"  {output_dir / 'detections.parquet'}   (FiftyOne-compatible)")
    print(f"  {output_dir / 'samples.parquet'}      (per-image summary)")
    print(f"  {labels_dir}                  (YOLO .txt files)")


if __name__ == "__main__":
    main()
