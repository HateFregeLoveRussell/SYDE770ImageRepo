#!/usr/bin/env python3
"""
compute_embeddings_fo.py

Compute DINOv2 + CLIP embeddings using FiftyOne Model Zoo, but **feed resized cached images**
(from cache_manifest parquet) so you have a hard upper bound on compute.

Outputs (DVC-tracked):
  <out_dir>/dinov2_embeddings.parquet
  <out_dir>/clip_embeddings.parquet

Optionally also writes embeddings back into your *main* FiftyOne dataset fields
(keyed by sample_id), so you can use FiftyOne Brain/UMAP tooling interactively.

Expected manifest schema (from cache_resize_images.py):
  sample_id, cached_path, ok, reason, ...

Example:
  python scripts/compute_embeddings_fo.py \
    --dataset th_cups_v1 \
    --cache-manifest data/derived/cache/cache_manifest_512.parquet \
    --out-dir data/derived/embeddings \
    --dinov2-model dinov2-vitg14-torch \
    --clip-model clip-vit-base32-torch \
    --batch-size 32 \
    --num-workers 8 \
    --write-to-fo \
    --overwrite-parquet

Notes:
- This script does NOT require cached images to be DVC-tracked.
- It uses a temporary in-memory FiftyOne dataset whose filepaths point at cached_path.
- It exports Parquet keyed by your canonical sample_id.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import fiftyone as fo
import fiftyone.zoo as foz


def str2bool(x: str) -> bool:
    x = x.strip().lower()
    if x in ("1", "true", "t", "yes", "y", "on"):
        return True
    if x in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got '{x}'")

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, help="Existing FiftyOne dataset name (canonical dataset)")
    ap.add_argument("--cache-manifest", required=True, help="Parquet manifest with sample_id -> cached_path")
    ap.add_argument("--out-dir", required=True, help="Output directory for parquet artifacts (DVC-tracked)")

    ap.add_argument("--sample-id-field", default="sample_id",
                    help="Field in main dataset that stores canonical key (default: sample_id)")

    ap.add_argument("--dinov2-model", default="dinov2-vitg14-torch", help="FiftyOne Model Zoo name for DINOv2")
    ap.add_argument("--clip-model", default="clip-vit-base32-torch", help="FiftyOne Model Zoo name for CLIP")

    ap.add_argument("--dinov2-field", default="emb_dinov2", help="Field name to store DINOv2 embeddings in FO")
    ap.add_argument("--clip-field", default="emb_clip", help="Field name to store CLIP embeddings in FO")

    ap.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding computation")
    ap.add_argument("--num-workers", type=int, default=None, help="Num dataloader workers (Torch models)")

    ap.add_argument(
        "--write-to-fo",
        type=str2bool,
        default=True,
        help="Write embeddings into the main FiftyOne dataset fields (true/false). Default: true",
    )
    ap.add_argument("--skip-existing-parquet", action="store_true",
                    help="If output parquet exists, skip recompute for that model")
    ap.add_argument("--overwrite-parquet", action="store_true", help="Overwrite existing parquet outputs")

    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")
    return ap.parse_args()


def _require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_cache_manifest(path: str) -> pd.DataFrame:
    m = pd.read_parquet(path)
    _require_cols(m, ["sample_id", "cached_path"], "cache_manifest")
    if "ok" in m.columns:
        m = m[m["ok"] == True].copy()
    m["sample_id"] = m["sample_id"].astype(str)
    m["cached_path"] = m["cached_path"].astype(str)
    return m


def build_cached_dataset(cache_df: pd.DataFrame) -> fo.Dataset:
    """
    Create a temporary FO dataset pointing at cached images.
    Stores canonical sample_id on each sample for later join.
    """
    tmp = fo.Dataset()  # unnamed temp dataset (exists in FO DB, but you can delete after)
    samples = []
    for row in cache_df.itertuples(index=False):
        s = fo.Sample(filepath=row.cached_path)
        s["sample_id"] = row.sample_id
        samples.append(s)
    tmp.add_samples(samples)
    return tmp


def compute_embeddings_on_cached_dataset(
    cached_ds: fo.Dataset,
    model_name: str,
    batch_size: Optional[int],
    num_workers: Optional[int],
) -> Tuple[np.ndarray, List[str], int]:
    """
    Returns:
      embs: np.ndarray [N, D]
      sample_ids: list[str] aligned with embs
      D: embedding dim
    """
    model = foz.load_zoo_model(model_name)

    # Returns embeddings aligned with the dataset order
    embs = cached_ds.compute_embeddings(
        model,
        embeddings_field=None,   # we do not need to store on cached_ds
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=False,     # if cached image exists, failures should be rare; better to fail loudly
    )

    embs = np.asarray(embs, dtype=np.float32)
    sample_ids = cached_ds.values("sample_id")

    if len(sample_ids) != len(embs):
        raise RuntimeError(f"sample_ids ({len(sample_ids)}) != embeddings ({len(embs)}) for model={model_name}")

    emb_dim = int(embs.shape[1]) if embs.ndim == 2 else 0
    return embs, sample_ids, emb_dim


def export_embeddings_parquet(
    out_path: Path,
    model_name: str,
    sample_ids: List[str],
    embs: np.ndarray,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "model": model_name,
        "embedding_dim": int(embs.shape[1]) if embs.ndim == 2 else None,
        "embedding": [v.tolist() for v in embs],
    })
    df.to_parquet(out_path, index=False)


def write_embeddings_to_main_dataset(
    main_ds: fo.Dataset,
    sample_id_field: str,
    embeddings_field: str,
    sample_ids: List[str],
    embs: np.ndarray,
):
    """
    Writes embeddings to `main_ds[embeddings_field]` keyed by `sample_id_field`.
    This is O(N) updates and fine for ~2500 images.
    """
    # Build mapping sample_id -> embedding list
    emb_map: Dict[str, List[float]] = {sid: embs[i].tolist() for i, sid in enumerate(sample_ids)}

    # Ensure the key field exists
    schema = main_ds.get_field_schema()
    if sample_id_field not in schema:
        raise ValueError(
            f"Main dataset '{main_ds.name}' missing field '{sample_id_field}'. "
            f"Your importer should set this. Alternatively, change --sample-id-field."
        )

    # Efficient-ish bulk loop
    view = main_ds.select_fields([sample_id_field])  # keeps reads light
    updates = 0
    missing = 0

    for s in view.iter_samples(progress=True):
        sid = s[sample_id_field]
        emb = emb_map.get(str(sid))
        if emb is None:
            missing += 1
            continue
        s[embeddings_field] = emb
        s.save()
        updates += 1

    print(f"[fiftyone] wrote field='{embeddings_field}' updates={updates} missing_in_manifest={missing}")


def main():
    args = parse_args()

    main_ds = fo.load_dataset(args.dataset)

    if args.max_samples is not None:
        # For testing: limit both main and cached via intersection on sample_id
        # We'll filter cache manifest after reading main.
        pass

    cache_df = load_cache_manifest(args.cache_manifest)

    # If max-samples, intersect with main dataset sample_ids and cap deterministically
    if args.max_samples is not None:
        main_ids = set(map(str, main_ds.values(args.sample_id_field)))
        cache_df = cache_df[cache_df["sample_id"].isin(main_ids)].sort_values("sample_id").head(args.max_samples)

    if cache_df.empty:
        raise RuntimeError("Cache manifest produced zero usable rows after filtering. Check 'ok' flags and paths.")

    # Build temp dataset pointing to cached images
    cached_ds = build_cached_dataset(cache_df)

    out_dir = Path(args.out_dir)
    dinov2_out = out_dir / "dinov2_embeddings.parquet"
    clip_out = out_dir / "clip_embeddings.parquet"

    # DINOv2
    if dinov2_out.exists() and args.skip_existing_parquet and not args.overwrite_parquet:
        print(f"[skip] DINOv2 parquet exists: {dinov2_out}")
    else:
        if dinov2_out.exists() and not args.overwrite_parquet:
            raise FileExistsError(f"{dinov2_out} exists. Use --overwrite-parquet or --skip-existing-parquet")

        embs, sample_ids, dim = compute_embeddings_on_cached_dataset(
            cached_ds, args.dinov2_model, args.batch_size, args.num_workers
        )
        export_embeddings_parquet(dinov2_out, args.dinov2_model, sample_ids, embs)
        print(f"[ok] DINOv2: N={len(sample_ids)} dim={dim} -> {dinov2_out}")

        if args.write_to_fo:
            write_embeddings_to_main_dataset(
                main_ds,
                sample_id_field=args.sample_id_field,
                embeddings_field=args.dinov2_field,
                sample_ids=sample_ids,
                embs=embs,
            )

    # CLIP
    if clip_out.exists() and args.skip_existing_parquet and not args.overwrite_parquet:
        print(f"[skip] CLIP parquet exists: {clip_out}")
    else:
        if clip_out.exists() and not args.overwrite_parquet:
            raise FileExistsError(f"{clip_out} exists. Use --overwrite-parquet or --skip-existing-parquet")

        embs, sample_ids, dim = compute_embeddings_on_cached_dataset(
            cached_ds, args.clip_model, args.batch_size, args.num_workers
        )
        export_embeddings_parquet(clip_out, args.clip_model, sample_ids, embs)
        print(f"[ok] CLIP: N={len(sample_ids)} dim={dim} -> {clip_out}")

        if args.write_to_fo:
            write_embeddings_to_main_dataset(
                main_ds,
                sample_id_field=args.sample_id_field,
                embeddings_field=args.clip_field,
                sample_ids=sample_ids,
                embs=embs,
            )

    if args.write_to_fo:
        main_ds.save()

    # Cleanup temp dataset to keep FO DB tidy
    tmp_name = cached_ds.name
    fo.delete_dataset(tmp_name)
    print(f"[cleanup] deleted temp cached dataset='{tmp_name}'")

    print("[done]")


if __name__ == "__main__":
    main()