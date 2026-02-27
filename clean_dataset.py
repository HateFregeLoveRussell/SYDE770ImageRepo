#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

"""
Example usage:
python clean_dataset.py \
  --images-dir data/images \
  --report ./violations_report.parquet
Example usage, with custom config:
python clean_dataset.py \
    --images-dir data/images\
    --config ./clean_config.json
"""

DEFAULT_CONFIG: Dict[str, Any] = {
    "cup_area_threshold": 0.15,  # fraction of image area
    # - max: uses the largest bbox area
    # - sum: sums areas of all boxes
    "cup_area_mode": "max",

    # Rule actions: "drop" | "warn" | "fix"
    "rules": {
        "MISSING_IMAGE_FILE": {"action": "drop"},
        "NO_DETECTIONS": {"action": "drop"},

        # bbox validity includes out-of-range, non-positive width/height, x+y overflow
        "BBOX_INVALID": {"action": "drop"},

        # Count mismatch: diagnostics vs number of Cup boxes
        "COUNT_MISMATCH": {"action": "fix"},

        # cup_percantage mismatch: overwrite with computed value
        "CUP_PERCENTAGE_MISMATCH": {"action": "fix"},

        # brand vs Tim Hortons bbox label mismatch
        "BRAND_TIM_MISMATCH": {"action": "warn"},
    },

    # Labels used in detections.parquet
    "labels": {
        "cup": "Cup",
        "tim": "Tim Hortons",
    },

    # Output class names
    "cup_class_names": {
        "tim": "TimHortonsCup",
        "non_tim": "NonTimHortonsCup",
    },
}


VALID_ACTIONS = {"drop", "warn", "fix"}


@dataclass
class Violation:
    sample_id: str
    rule: str
    detail: str


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    cfg.update({k: v for k, v in user_cfg.items() if k not in ("rules", "labels", "cup_class_names")})
    if "rules" in user_cfg:
        cfg["rules"].update(user_cfg["rules"])
    if "labels" in user_cfg:
        cfg["labels"].update(user_cfg["labels"])
    if "cup_class_names" in user_cfg:
        cfg["cup_class_names"].update(user_cfg["cup_class_names"])

    for rule, spec in cfg["rules"].items():
        act = spec.get("action")
        if act not in VALID_ACTIONS:
            raise ValueError(f"Config error: rules.{rule}.action must be one of {sorted(VALID_ACTIONS)}, got {act!r}")

    if cfg["cup_area_mode"] not in ("max", "sum"):
        raise ValueError("Config error: cup_area_mode must be 'max' or 'sum'")

    return cfg


def _images_fs_path(images_dir: str, image_key: str) -> str:
    """
    image_key: images/foo.jpg
    images_dir should point to the directory that CONTAINS the images (not including 'images/').
               Example: if files are at data/raw/images/foo.jpg -> images_dir=data/raw/images
    """
    rel = image_key
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return os.path.join(images_dir, rel)


def _get_cup_percentage_col(samples: pd.DataFrame) -> str:
    if "cup_percantage" in samples.columns:
        return "cup_percantage"
    if "cup_percentage" in samples.columns:
        return "cup_percentage"
    samples["cup_percantage"] = None
    return "cup_percantage"


def _bbox_invalid_mask(det: pd.DataFrame) -> pd.Series:
    """
    Invalid if:
      - any coord missing
      - w/h <= 0
      - x/y < 0
      - x+w > 100 or y+h > 100
      - x/y > 100
    """
    for c in ("x_pct", "y_pct", "w_pct", "h_pct"):
        if c not in det.columns:
            # If columns missing, everything is invalid
            return pd.Series([True] * len(det), index=det.index)

    x = det["x_pct"]
    y = det["y_pct"]
    w = det["w_pct"]
    h = det["h_pct"]

    invalid = (
        x.isna() | y.isna() | w.isna() | h.isna()
        | (w <= 0) | (h <= 0)
        | (x < 0) | (y < 0)
        | (x > 100) | (y > 100)
        | ((x + w) > 100.000001) | ((y + h) > 100.000001)
    )
    return invalid


def _area_fraction_from_pct(w_pct: float, h_pct: float) -> float:
    return (w_pct * h_pct) / 10000.0


def clean(
    samples_path: str,
    detections_path: str,
    images_dir: str,
    cfg: Dict[str, Any],
    *,
    overwrite: bool = True,
    report_path: Optional[str] = None,
) -> None:
    samples = pd.read_parquet(samples_path)
    det = pd.read_parquet(detections_path)

    if "sample_id" not in samples.columns:
        raise ValueError(f"{samples_path} missing required column 'sample_id'")
    if "sample_id" not in det.columns:
        # allow empty detections table
        if len(det) > 0:
            raise ValueError(f"{detections_path} missing required column 'sample_id'")

    cup_label = cfg["labels"]["cup"]
    tim_label = cfg["labels"]["tim"]
    cup_area_threshold = float(cfg["cup_area_threshold"])
    cup_area_mode = cfg["cup_area_mode"]

    cup_pct_col = _get_cup_percentage_col(samples)

    violations: List[Violation] = []

    # Missing image file
    samples["image_exists"] = samples["sample_id"].apply(lambda k: os.path.exists(_images_fs_path(images_dir, k)))
    missing_img_ids = samples.loc[~samples["image_exists"], "sample_id"].tolist()
    for sid in missing_img_ids:
        violations.append(Violation(sid, "MISSING_IMAGE_FILE", "Image file not found on disk"))

    # No detections
    det_counts = det.groupby("sample_id").size().rename("n_det")
    samples = samples.merge(det_counts, how="left", left_on="sample_id", right_index=True)
    samples["n_det"] = samples["n_det"].fillna(0).astype(int)

    no_det_ids = samples.loc[samples["n_det"] == 0, "sample_id"].tolist()
    for sid in no_det_ids:
        violations.append(Violation(sid, "NO_DETECTIONS", "No detections for sample"))

    # BBox validity
    det_invalid = det[_bbox_invalid_mask(det)].copy() if len(det) else det.head(0).copy()
    invalid_bbox_sample_ids = det_invalid["sample_id"].unique().tolist() if len(det_invalid) else []
    for sid in invalid_bbox_sample_ids:
        violations.append(Violation(sid, "BBOX_INVALID", "One or more bboxes invalid (range/size)"))

    # Derived helpers: cup/tim presence, cup area
    if len(det):
        det["is_cup"] = det["label"] == cup_label
        det["is_tim"] = det["label"] == tim_label
        det["area_frac"] = det.apply(lambda r: _area_fraction_from_pct(float(r["w_pct"]), float(r["h_pct"])), axis=1)

        cup_stats = det[det["is_cup"]].groupby("sample_id").agg(
            n_cup=("is_cup", "size"),
            max_cup_area=("area_frac", "max"),
            sum_cup_area=("area_frac", "sum"),
        )
        tim_present = det.groupby("sample_id")["is_tim"].any().rename("has_tim_bbox")
    else:
        cup_stats = pd.DataFrame(columns=["n_cup", "max_cup_area", "sum_cup_area"])
        tim_present = pd.Series(dtype=bool, name="has_tim_bbox")

    samples = samples.merge(cup_stats, how="left", left_on="sample_id", right_index=True)
    samples = samples.merge(tim_present, how="left", left_on="sample_id", right_index=True)

    samples["n_cup"] = samples["n_cup"].fillna(0).astype(int)
    samples["max_cup_area"] = samples["max_cup_area"].fillna(0.0)
    samples["sum_cup_area"] = samples["sum_cup_area"].fillna(0.0)
    samples["has_tim_bbox"] = samples["has_tim_bbox"].fillna(False)

    # Count mismatch (Single/Multiple vs number of Cup boxes)
    if "count" in samples.columns:
        def expected_count(n_cup: int) -> Optional[str]:
            if n_cup <= 0:
                return None
            return "Single" if n_cup == 1 else "Multiple"

        samples["expected_count_from_boxes"] = samples["n_cup"].apply(expected_count)
        mismatch = (
            samples["expected_count_from_boxes"].notna()
            & samples["count"].notna()
            & (samples["count"] != samples["expected_count_from_boxes"])
        )
        for sid, got, exp in samples.loc[mismatch, ["sample_id", "count", "expected_count_from_boxes"]].itertuples(index=False):
            violations.append(Violation(str(sid), "COUNT_MISMATCH", f"count={got!r} but boxes imply {exp!r}"))

    # cup_percantage mismatch
    if cup_area_mode == "max":
        samples["cup_area_frac"] = samples["max_cup_area"]
    else:
        samples["cup_area_frac"] = samples["sum_cup_area"]

    samples["computed_cup_percantage"] = samples["cup_area_frac"].apply(lambda a: "True" if float(a) > cup_area_threshold else "False")

    # Only compare when there is at least one cup box otherwise it's ambiguous
    comparable = samples["n_cup"] > 0
    existing = samples[cup_pct_col].astype("object")

    mismatch = comparable & existing.notna() & (existing != samples["computed_cup_percantage"])
    for sid, got, comp, area in samples.loc[mismatch, ["sample_id", cup_pct_col, "computed_cup_percantage", "cup_area_frac"]].itertuples(index=False):
        violations.append(Violation(str(sid), "CUP_PERCENTAGE_MISMATCH", f"{cup_pct_col}={got!r} computed={comp!r} area_frac={area:.4f}"))

    # brand vs Tim Hortons bbox mismatch
    if "brand" in samples.columns:
        brand_is_tim = samples["brand"] == "Tim Hortons"
        mismatch_brand = brand_is_tim ^ samples["has_tim_bbox"]
        mismatch_brand = mismatch_brand & (samples["brand"].notna() | samples["has_tim_bbox"])
        for sid, brand, has in samples.loc[mismatch_brand, ["sample_id", "brand", "has_tim_bbox"]].itertuples(index=False):
            violations.append(Violation(str(sid), "BRAND_TIM_MISMATCH", f"brand={brand!r} has_tim_bbox={bool(has)}"))

    # Apply actions
    rules = cfg["rules"]

    # Collect drop set
    drop_ids = set()
    for v in violations:
        action = rules.get(v.rule, {"action": "warn"}).get("action", "warn")
        if action == "drop":
            drop_ids.add(v.sample_id)

    # Apply fixes
    # Fix COUNT_MISMATCH
    count_action = rules.get("COUNT_MISMATCH", {"action": "warn"}).get("action", "warn")
    if count_action == "fix":
        if "count" not in samples.columns:
            samples["count"] = None

        # Only set when we have at least one cup box.
        fix_mask = samples["n_cup"] > 0
        samples.loc[fix_mask & (samples["n_cup"] == 1), "count"] = "Single"
        samples.loc[fix_mask & (samples["n_cup"] > 1), "count"] = "Multiple"

    # Fix CUP_PERCENTAGE_MISMATCH
    cup_pct_action = rules.get("CUP_PERCENTAGE_MISMATCH", {"action": "warn"}).get("action", "warn")
    if cup_pct_action == "fix":
        fix_mask = samples["n_cup"] > 0
        samples.loc[fix_mask, cup_pct_col] = samples.loc[fix_mask, "computed_cup_percantage"]

    samples["is_tim_hortons_bbox"] = samples["has_tim_bbox"].astype(bool)
    samples["cup_class"] = samples["is_tim_hortons_bbox"].apply(
        lambda b: cfg["cup_class_names"]["tim"] if b else cfg["cup_class_names"]["non_tim"]
    )

    if drop_ids:
        samples = samples[~samples["sample_id"].isin(drop_ids)].copy()
        det = det[~det["sample_id"].isin(drop_ids)].copy()

    # Emit report
    by_rule: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    for v in violations:
        by_rule[v.rule] = by_rule.get(v.rule, 0) + 1
        act = rules.get(v.rule, {"action": "warn"}).get("action", "warn")
        by_action[act] = by_action.get(act, 0) + 1

    print("Cleaning summary:", file=sys.stderr)
    print(f"  Violations total: {len(violations)}", file=sys.stderr)
    print(f"  By rule: {by_rule}", file=sys.stderr)
    print(f"  By action: {by_action}", file=sys.stderr)
    print(f"  Dropped samples: {len(drop_ids)}", file=sys.stderr)
    print(f"  Remaining samples: {len(samples)}", file=sys.stderr)
    print(f"  Remaining detections: {len(det)}", file=sys.stderr)

    if report_path:
        rep = pd.DataFrame([v.__dict__ for v in violations])
        rep.to_parquet(report_path, index=False)
        print(f"Wrote report: {report_path}", file=sys.stderr)

    if overwrite:
        samples.to_parquet(samples_path, index=False)
        det.to_parquet(detections_path, index=False)
        print(f"Overwrote: {samples_path}", file=sys.stderr)
        print(f"Overwrote: {detections_path}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Basic cleaning for Label Studio-derived index tables.")
    p.add_argument("--samples", default="data/index/samples.parquet", help="Path to samples.parquet")
    p.add_argument("--detections", default="data/index/detections.parquet", help="Path to detections.parquet")
    p.add_argument("--images-dir", required=True, help="Directory containing the actual image files.")
    p.add_argument("--config", default=None, help="Optional JSON config overriding rule actions.")
    p.add_argument("--report", default=None, help="Optional path to write violations report parquet.")
    args = p.parse_args(argv)

    try:
        cfg = _load_config(args.config)
        clean(
            samples_path=args.samples,
            detections_path=args.detections,
            images_dir=args.images_dir,
            cfg=cfg,
            overwrite=True,
            report_path=args.report,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())