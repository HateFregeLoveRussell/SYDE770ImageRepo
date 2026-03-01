#!/usr/bin/env python3
"""
compute_brain_visualization.py

Create a FiftyOne Brain visualization (e.g. UMAP) that shows up in the App's
Embeddings panel via a `brain_key`.

This is intentionally *not* DVC-tracked as a file artifact: Brain runs live in
the FiftyOne dataset DB. Treat this as an "interactive analysis stage" that can
be rerun deterministically from your versioned embeddings + params.

Typical usage:
  python scripts/compute_brain_visualization.py \
    --dataset th_cups_v1 \
    --embeddings-field emb_clip \
    --method umap \
    --brain-key umap_clip_512_v1 \
    --metric cosine \
    --seed 42 \
    --overwrite

Then launch:
  fiftyone app launch --dataset th_cups_v1
"""

import argparse
import sys

import fiftyone as fo
import fiftyone.brain as fob


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, help="FiftyOne dataset name")
    ap.add_argument(
        "--embeddings-field",
        required=True,
        help="Sample field containing embeddings (e.g. emb_clip or emb_dinov2)",
    )

    ap.add_argument(
        "--method",
        default="umap",
        choices=["umap", "tsne", "pca"],
        help="Visualization method (default: umap)",
    )
    ap.add_argument("--num-dims", type=int, default=2, help="Output dims (default: 2)")

    ap.add_argument(
        "--brain-key",
        required=True,
        help="Name for the Brain run (shows up in Embeddings panel dropdown)",
    )

    ap.add_argument("--metric", default="cosine", help="Distance metric (default: cosine)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # UMAP-specific knobs (safe to pass; ignored by other methods if not applicable)
    ap.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors (default: 30)")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing run with same brain_key")
    ap.add_argument("--view", default=None, help="Optional saved view name to visualize instead of whole dataset")

    return ap.parse_args()


def main():
    args = parse_args()

    ds = fo.load_dataset(args.dataset)

    # Optionally operate on a saved view for a subset
    if args.view:
        try:
            view = ds.load_saved_view(args.view)
        except Exception as e:
            raise RuntimeError(f"Could not load saved view '{args.view}': {e}") from e
    else:
        view = ds

    # Validate embeddings field exists
    schema = view.get_field_schema()
    if args.embeddings_field not in schema:
        raise ValueError(
            f"Dataset/view is missing embeddings field '{args.embeddings_field}'. "
            f"Did you run compute_embeddings_fo.py with --write-to-fo?"
        )

    # Handle overwrite
    existing = set(ds.list_brain_runs())
    if args.brain_key in existing:
        if args.overwrite:
            ds.delete_brain_run(args.brain_key)
            print(f"[brain] deleted existing run brain_key='{args.brain_key}'")
        else:
            raise FileExistsError(
                f"Brain run '{args.brain_key}' already exists. Use --overwrite to replace it."
            )

    # Compute visualization
    kwargs = dict(
        method=args.method,
        num_dims=args.num_dims,
        brain_key=args.brain_key,
        metric=args.metric,
        seed=args.seed,
    )

    # UMAP knobs are only meaningful for UMAP; harmless to include conditionally
    if args.method == "umap":
        kwargs.update(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

    print(
        f"[brain] computing visualization method={args.method} "
        f"embeddings='{args.embeddings_field}' brain_key='{args.brain_key}' "
        f"metric={args.metric} seed={args.seed}"
    )

    fob.compute_visualization(
        view,
        embeddings=args.embeddings_field,
        **kwargs,
    )

    # Persist run
    ds.save()

    print(f"[brain] done. brain_key='{args.brain_key}'")
    print("[hint] Open FiftyOne App -> Embeddings panel -> select this brain_key from dropdown")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        raise