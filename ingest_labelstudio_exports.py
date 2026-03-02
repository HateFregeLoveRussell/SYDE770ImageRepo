#!/usr/bin/env python3
"""
ingest_new_ls_exports.py

Append new tasks from one or more Label Studio JSON exports into:
  - data/index/samples.parquet  (1 row per image, keyed by sample_id)
  - data/index/detections.parquet (1 row per bbox, FK = sample_id)

Canonical key rule:
  sample_id = "images/<basename-of-image-file>"

This handles LS exports whose task["data"]["image"] looks like:
  /data/local-files/?d=images%5CNew_Timmies%5CPXL_....jpg
  /data/local-files/?d=2026-02-26T.../openverse_....jpg
etc.

The script is conservative:
  - by default it only appends truly new sample_ids
  - it does not try to perfectly re-derive *all* downstream computed columns;
    it fills what it can, and leaves the rest as NA so your cleaning/derivation
    stage can recompute deterministically.

Requires: pandas, pyarrow
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None

def _extract_basename_from_ls_image_field(image_field: str) -> str:
    """
    LS image field commonly:
      "/data/local-files/?d=images%5CNew_Timmies%5Cfoo.jpg"
      "/data/local-files/?d=2026-02-26T.../openverse_x.jpg"
    We:
      - parse query param d if present
      - url-decode
      - normalize slashes
      - take basename
    """
    if not image_field:
        return ""

    # Pull out query param d=... if it exists
    if "?d=" in image_field:
        # Split once to keep everything after ?d=
        d = image_field.split("?d=", 1)[1]
    else:
        d = image_field

    d = urllib.parse.unquote(d)              # decode %5C etc
    d = d.replace("\\", "/")                 # normalize windows slashes
    d = re.sub(r"/+", "/", d)                # collapse repeated slashes
    base = os.path.basename(d)
    return base

def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


# -------------------------
# Parsing LS task
# -------------------------

@dataclass
class ParsedSample:
    sample_row: Dict[str, Any]
    det_rows: List[Dict[str, Any]]

def _parse_task(task: Dict[str, Any], images_root: Path) -> ParsedSample:
    ls_task_id = task.get("id")
    project_id = task.get("project") or task.get("project_id")

    image_field = (task.get("data") or {}).get("image") or ""
    basename = _extract_basename_from_ls_image_field(image_field)
    sample_id = f"images/{basename}" if basename else None

    # Choose the "best" annotation (you said 1 per task typically)
    annotations = task.get("annotations") or []
    ann = annotations[0] if annotations else None

    created_at = (ann or {}).get("created_at") or task.get("created_at")
    updated_at = (ann or {}).get("updated_at") or task.get("updated_at")

    total_annotations = task.get("total_annotations")
    cancelled_annotations = task.get("cancelled_annotations")

    # Parse results -> bboxes + choices
    results = (ann or {}).get("result") or []
    choices: Dict[str, Any] = {}

    det_rows: List[Dict[str, Any]] = []
    for r in results:
        rtype = r.get("type")
        from_name = r.get("from_name")
        to_name = r.get("to_name")
        val = r.get("value") or {}

        # Rectangle labels -> one detection row
        if rtype == "rectanglelabels":
            rect_labels = val.get("rectanglelabels") or []
            label = rect_labels[0] if rect_labels else None

            x = float(val.get("x", 0.0))
            y = float(val.get("y", 0.0))
            w = float(val.get("width", 0.0))
            h = float(val.get("height", 0.0))

            # Derived flags
            is_tim = (label == "Tim Hortons")
            is_cup = (label in ("Cup", "Tim Hortons"))  # treat Tim Hortons as a cup subtype

            area_frac = (w / 100.0) * (h / 100.0)

            det_rows.append({
                "sample_id": sample_id,
                "image_key": sample_id,  # keep compatibility with your existing schema
                "ls_task_id": ls_task_id,
                "label": label,
                "x_pct": x,
                "y_pct": y,
                "w_pct": w,
                "h_pct": h,
                "ls_annotation_id": (ann or {}).get("id"),
                "ls_result_id": r.get("id"),
                "is_cup": bool(is_cup),
                "is_tim": bool(is_tim),
                "area_frac": float(area_frac),
            })

        # Choices -> store first choice
        elif rtype == "choices":
            ch = val.get("choices") or []
            if from_name and ch:
                choices[from_name] = ch[0]

    # Minimal sample-level derived stats (safe + cheap)
    n_det = len(det_rows)
    n_cup = sum(1 for d in det_rows if d.get("is_cup"))
    has_tim_bbox = any(d.get("is_tim") for d in det_rows)
    cup_areas = [d["area_frac"] for d in det_rows if d.get("is_cup")]
    max_cup_area = max(cup_areas) if cup_areas else 0.0
    sum_cup_area = float(sum(cup_areas)) if cup_areas else 0.0

    # image existence check on disk
    img_path = images_root / sample_id if sample_id else None
    image_exists = bool(img_path and img_path.exists())

    # lid/sleeve are stored as choices like "True"/"False" in the exports :contentReference[oaicite:2]{index=2}
    lid = _to_bool(choices.get("lid"))
    sleeve = _to_bool(choices.get("sleeve"))

    # You can keep your older naming (cup_percantage) to match current parquet column
    sample_row = {
        "sample_id": sample_id,
        "image_key": sample_id,
        "image_raw": image_field,
        "ls_task_id": ls_task_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "ls_project": project_id,
        "total_annotations": total_annotations,
        "cancelled_annotations": cancelled_annotations,

        # diagnostic fields (from LS choices)
        "type": choices.get("type"),
        "background": choices.get("background"),
        "cup_percantage": choices.get("cup_percantage"),
        "orientation": choices.get("orientation"),
        "deform": choices.get("deform"),
        "blur": choices.get("blur"),
        "occluded": choices.get("occluded"),
        "count": choices.get("count"),
        "brand": choices.get("brand"),

        # new binary columns
        "lid": lid,
        "sleeve": sleeve,

        # cheap derivations
        "image_exists": image_exists,
        "n_det": n_det,
        "n_cup": n_cup,
        "max_cup_area": max_cup_area,
        "sum_cup_area": sum_cup_area,
        "has_tim_bbox": bool(has_tim_bbox),

        # optional “best guess” class; feel free to recompute later in your cleaning stage
        "cup_class": ("TimHortonsCup" if has_tim_bbox else ("NonTimHortonsCup" if n_cup > 0 else pd.NA)),
    }

    return ParsedSample(sample_row=sample_row, det_rows=det_rows)


# -------------------------
# Main
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to samples.parquet")
    ap.add_argument("--detections", required=True, help="Path to detections.parquet")
    ap.add_argument("--images-root", required=True, help="Repo root that contains the images/ directory (e.g. data)")
    ap.add_argument("--ls-json", action="append", required=True, help="Label Studio export JSON (repeatable)")
    ap.add_argument("--overwrite-existing", action="store_true",
                    help="If set, replace existing rows for any sample_id found in the export")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    det_path = Path(args.detections)
    images_root = Path(args.images_root)

    if not samples_path.exists():
        print(f"ERROR: samples parquet not found: {samples_path}", file=sys.stderr)
        return 2
    if not det_path.exists():
        print(f"ERROR: detections parquet not found: {det_path}", file=sys.stderr)
        return 2

    samples_df = pd.read_parquet(samples_path)
    det_df = pd.read_parquet(det_path)

    # Parse all tasks from all exports
    new_samples: List[Dict[str, Any]] = []
    new_dets: List[Dict[str, Any]] = []

    for p in args.ls_json:
        export = _read_json(p)
        if not isinstance(export, list):
            print(f"ERROR: expected list-of-tasks JSON export: {p}", file=sys.stderr)
            return 2

        for task in export:
            parsed = _parse_task(task, images_root=images_root)
            if not parsed.sample_row.get("sample_id"):
                continue
            new_samples.append(parsed.sample_row)
            new_dets.extend(parsed.det_rows)

    if not new_samples:
        print("No samples parsed from exports (nothing to do).")
        return 0

    new_samples_df = pd.DataFrame(new_samples)
    new_dets_df = pd.DataFrame(new_dets)

    # Align columns to existing schemas (add missing cols as NA; drop unknown extra cols)
    samples_df = _ensure_columns(samples_df, list(new_samples_df.columns))
    new_samples_df = _ensure_columns(new_samples_df, list(samples_df.columns))
    new_samples_df = new_samples_df[samples_df.columns]

    det_df = _ensure_columns(det_df, list(new_dets_df.columns))
    new_dets_df = _ensure_columns(new_dets_df, list(det_df.columns))
    new_dets_df = new_dets_df[det_df.columns]

    # Handle duplicates by sample_id
    existing_ids = set(samples_df["sample_id"].astype(str).tolist())
    incoming_ids = new_samples_df["sample_id"].astype(str).tolist()

    if args.overwrite_existing:
        # remove any existing rows for incoming ids
        keep_mask = ~samples_df["sample_id"].astype(str).isin(incoming_ids)
        samples_df = samples_df.loc[keep_mask].copy()

        keep_det_mask = ~det_df["sample_id"].astype(str).isin(incoming_ids)
        det_df = det_df.loc[keep_det_mask].copy()

        appended_samples = len(new_samples_df)
    else:
        # append only those not already present
        new_mask = ~new_samples_df["sample_id"].astype(str).isin(existing_ids)
        new_samples_df = new_samples_df.loc[new_mask].copy()

        # keep only dets whose sample_id is genuinely new
        new_ids = set(new_samples_df["sample_id"].astype(str).tolist())
        new_dets_df = new_dets_df.loc[new_dets_df["sample_id"].astype(str).isin(new_ids)].copy()

        appended_samples = len(new_samples_df)

    if appended_samples == 0:
        print("All incoming sample_ids already exist; nothing appended.")
        return 0

    # Append + write
    out_samples = pd.concat([samples_df, new_samples_df], ignore_index=True)
    out_dets = pd.concat([det_df, new_dets_df], ignore_index=True)

    _atomic_write_parquet(out_samples, samples_path)
    _atomic_write_parquet(out_dets, det_path)

    print(f"Appended samples: {appended_samples}")
    print(f"Appended detections: {len(new_dets_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())