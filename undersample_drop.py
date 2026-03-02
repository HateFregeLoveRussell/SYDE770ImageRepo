#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


def parse_constraints(s: str) -> Dict[str, Any]:
    """
    Accept JSON object, e.g.
      '{"brand":"other","occlusion":"fully-occluded","sleeve":true}'
    """
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse --constraints as JSON: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--constraints must be a JSON object")
    return obj


def apply_equality_constraints(df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in constraints.items():
        if col not in df.columns:
            raise SystemExit(f"Constraint column '{col}' not found in samples.parquet columns")

        # Equality only Handle null explicitly via val=None if needed.
        if val is None:
            mask &= df[col].isna()
        else:
            mask &= (df[col] == val)
    return df[mask]


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Undersample by dropping N samples matching equality constraints")
    ap.add_argument("--n-drop", type=int, required=True)
    ap.add_argument("--constraints", type=str, required=True, help='JSON object, e.g. \'{"brand":"other","sleeve":true}\'')
    ap.add_argument("--samples", type=str, default="data/index/samples.parquet")
    ap.add_argument("--detections", type=str, default="data/index/detections.parquet")
    ap.add_argument("--images-root", type=str, default="data", help="Root containing images/<...> paths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delete-images", action="store_true")
    ap.add_argument("--strict-missing-files", action="store_true",
                    help="If deleting images, fail if any file is missing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--record", type=str, default="",
                    help="Optional path to write dropped sample_ids parquet (audit record)")
    args = ap.parse_args()

    if args.n_drop <= 0:
        raise SystemExit("--n-drop must be > 0")

    constraints = parse_constraints(args.constraints)

    samples_path = Path(args.samples)
    det_path = Path(args.detections)
    images_root = Path(args.images_root)

    samples = pd.read_parquet(samples_path)
    det = pd.read_parquet(det_path)

    if "sample_id" not in samples.columns:
        raise SystemExit(f"{samples_path} missing 'sample_id'")
    if "sample_id" not in det.columns:
        raise SystemExit(f"{det_path} missing 'sample_id'")

    eligible = apply_equality_constraints(samples, constraints)

    n_eligible = len(eligible)
    if n_eligible < args.n_drop:
        raise SystemExit(
            f"Infeasible undersampling: eligible={n_eligible} < n_drop={args.n_drop} "
            f"for constraints={constraints}"
        )

    # Deterministic selection: seeded shuffle of eligible rows
    eligible = eligible.sample(frac=1.0, random_state=args.seed)
    drop_df = eligible.head(args.n_drop)
    drop_ids: List[str] = drop_df["sample_id"].astype(str).tolist()
    drop_set: Set[str] = set(drop_ids)

    # Stats
    det_drop_count = int(det["sample_id"].isin(drop_set).sum())
    print(f"Eligible pool: {n_eligible}")
    print(f"Dropping: {len(drop_ids)} samples (seed={args.seed})")
    print(f"Detections to drop: {det_drop_count}")
    print(f"Constraints: {constraints}")

    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec = drop_df[["sample_id"]].copy()
        # store metadata as columns for audit
        rec["seed"] = args.seed
        rec["n_drop"] = args.n_drop
        rec["constraints_json"] = json.dumps(constraints, sort_keys=True)
        if args.dry_run:
            print(f"[dry-run] Would write record to {rec_path}")
        else:
            rec.to_parquet(rec_path, index=False)
            print(f"Wrote record: {rec_path}")

    # Optionally delete images
    if args.delete_images:
        missing: List[str] = []
        deleted = 0
        for sid in drop_ids:
            p = images_root / sid  # sid is like "images/xxx.jpg"
            if p.exists():
                p.unlink()
                deleted += 1
            else:
                missing.append(str(p))

        print(f"Deleted image files: {deleted}")
        if missing:
            msg = f"Missing {len(missing)} image files. First 10:\n" + "\n".join(missing[:10])
            if args.strict_missing_files:
                raise SystemExit(msg)
            else:
                print("WARNING:", msg)

    # Update parquets (remove dropped sample_ids)
    new_samples = samples[~samples["sample_id"].isin(drop_set)].copy()
    new_det = det[~det["sample_id"].isin(drop_set)].copy()

    if args.dry_run:
        print("[dry-run] Would write updated samples.parquet and detections.parquet")
        print(f"[dry-run] samples: {len(samples)} -> {len(new_samples)}")
        print(f"[dry-run] detections: {len(det)} -> {len(new_det)}")
        return 0

    atomic_write_parquet(new_samples, samples_path)
    atomic_write_parquet(new_det, det_path)
    print(f"Wrote updated: {samples_path}")
    print(f"Wrote updated: {det_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())