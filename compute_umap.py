#!/usr/bin/env python3
"""
compute_umap.py

Run UMAP over precomputed embedding Parquets and export 2D coordinates.

Input embedding parquet schema (from compute_embeddings_fo.py):
  sample_id: str
  model: str
  embedding_dim: int
  embedding: list[float]

Outputs:
  <out_path> parquet with:
    sample_id: str
    model: str
    embedding_dim: int
    umap_x: float
    umap_y: float
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    umap_seed: int

Optionally sync back to FiftyOne (field = --fo-field) as [x, y] vector.

Example:
  python scripts/compute_umap.py \
    --embeddings data/derived/embeddings/dinov2_embeddings.parquet \
    --out data/derived/umap/dinov2_umap.parquet \
    --n-neighbors 30 --min-dist 0.1 --metric cosine --seed 42 \
    --sync-fo --dataset th_cups_v1 --fo-field umap_dinov2
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import umap  # umap-learn
except Exception as e:
    raise RuntimeError(
        "Missing dependency: umap-learn. Install via: pip install umap-learn"
    ) from e


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--embeddings", required=True, help="Embedding parquet path")
    ap.add_argument("--out", required=True, help="Output parquet path for UMAP coords")

    ap.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors (default 30)")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default 0.1)")
    ap.add_argument("--metric", default="cosine", help="UMAP metric (default cosine)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    ap.add_argument("--normalize", choices=["none", "l2"], default="l2",
                    help="Optional embedding normalization before UMAP (default l2)")

    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")

    # Optional FiftyOne sync
    ap.add_argument("--sync-fo", action="store_true", help="Sync coords back into a FiftyOne dataset")
    ap.add_argument("--dataset", default=None, help="FiftyOne dataset name (required if --sync-fo)")
    ap.add_argument("--sample-id-field", default="sample_id",
                    help="Sample id field in FiftyOne dataset (default sample_id)")
    ap.add_argument("--fo-field", default=None,
                    help="Field name to store [x,y] in FiftyOne (required if --sync-fo)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite output parquet if exists")
    return ap.parse_args()


def _require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def load_embeddings(path: str, max_samples: int = None) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    _require_cols(df, ["sample_id", "embedding"], "embeddings parquet")
    df["sample_id"] = df["sample_id"].astype(str)

    # Deterministic order
    df = df.sort_values("sample_id").reset_index(drop=True)

    if max_samples is not None:
        df = df.head(max_samples)

    # Convert list column to ndarray
    X = np.asarray(df["embedding"].tolist(), dtype=np.float32)

    # Optional: embed_dim present
    if "embedding_dim" in df.columns:
        emb_dim = int(df["embedding_dim"].iloc[0]) if len(df) else X.shape[1]
        if X.ndim != 2 or X.shape[1] != emb_dim:
            # Don't hard fail, but warn via print
            print(f"[warn] embedding_dim mismatch: parquet={emb_dim}, array={X.shape}")
    return df, X


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
    )
    Z = reducer.fit_transform(X)
    return np.asarray(Z, dtype=np.float32)


def write_umap_parquet(
    out_path: Path,
    df_in: pd.DataFrame,
    Z: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = df_in["model"].iloc[0] if "model" in df_in.columns and len(df_in) else None
    emb_dim = int(df_in["embedding_dim"].iloc[0]) if "embedding_dim" in df_in.columns and len(df_in) else int(Z.shape[1])

    out_df = pd.DataFrame({
        "sample_id": df_in["sample_id"].tolist(),
        "model": model,
        "embedding_dim": emb_dim,
        "umap_x": Z[:, 0],
        "umap_y": Z[:, 1],
        "umap_n_neighbors": n_neighbors,
        "umap_min_dist": float(min_dist),
        "umap_metric": str(metric),
        "umap_seed": int(seed),
    })

    out_df.to_parquet(out_path, index=False)


def sync_to_fiftyone(
    dataset_name: str,
    sample_id_field: str,
    fo_field: str,
    umap_df: pd.DataFrame,
):
    import fiftyone as fo

    ds = fo.load_dataset(dataset_name)
    schema = ds.get_field_schema()

    if sample_id_field not in schema:
        raise ValueError(f"Dataset '{dataset_name}' missing field '{sample_id_field}'")

    # Map sample_id -> [x,y]
    coords = {
        str(sid): [float(x), float(y)]
        for sid, x, y in zip(umap_df["sample_id"], umap_df["umap_x"], umap_df["umap_y"])
    }

    updates = 0
    missing = 0

    view = ds.select_fields([sample_id_field])
    for s in view.iter_samples(progress=True):
        sid = str(s[sample_id_field])
        v = coords.get(sid)
        if v is None:
            missing += 1
            continue
        s[fo_field] = v
        s.save()
        updates += 1

    ds.save()
    print(f"[fiftyone] synced field='{fo_field}' updates={updates} missing={missing}")


def main():
    args = parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")

    df, X = load_embeddings(args.embeddings, max_samples=args.max_samples)

    if args.normalize == "l2":
        X = l2_normalize(X)

    Z = run_umap(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    write_umap_parquet(
        out_path=out_path,
        df_in=df,
        Z=Z,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    print(f"[ok] wrote umap -> {out_path} (N={len(df)})")

    if args.sync_fo:
        if not args.dataset or not args.fo_field:
            raise ValueError("--sync-fo requires --dataset and --fo-field")
        umap_df = pd.read_parquet(out_path)
        sync_to_fiftyone(
            dataset_name=args.dataset,
            sample_id_field=args.sample_id_field,
            fo_field=args.fo_field,
            umap_df=umap_df,
        )

    print("[done]")


if __name__ == "__main__":
    main()