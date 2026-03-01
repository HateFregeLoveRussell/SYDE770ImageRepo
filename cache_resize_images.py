#!/usr/bin/env python3
"""
cache_resize_images.py

Build a local cache of resized images
emit a manifest parquet mapping sample_id -> cached_filepath.

Example:
  python cache_resize_images.py \
    --samples data/index/samples.parquet \
    --images-root data/ \
    --cache-root .cache/resized_512 \
    --max-side 512 \
    --format jpg \
    --jpeg-quality 92 \
    --manifest-out data/derived/cache/cache_manifest_512.parquet

- Uses PIL. For speed, set --num-workers > 1.
- "max-side" means the longest side after resizing equals max_side (aspect preserved).
- "short-side" means the shortest side after resizing equals short_side (aspect preserved).
  Choose ONE.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Path to samples.parquet")
    ap.add_argument(
        "--images-root",
        required=True,
        help="Root folder that contains the 'images/' dir (e.g. data/raw). "
             "Input path becomes images_root/<sample_id>",
    )
    ap.add_argument(
        "--cache-root",
        required=True,
        help="Local cache root (NOT tracked by DVC), e.g. .cache/resized_512",
    )

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--max-side", type=int, help="Resize so longest side == this many pixels")
    g.add_argument("--short-side", type=int, help="Resize so shortest side == this many pixels")

    ap.add_argument(
        "--format",
        choices=["jpg", "png", "webp"],
        default="jpg",
        help="Output image format/extension (default: jpg)",
    )
    ap.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality (default 92)")
    ap.add_argument("--webp-quality", type=int, default=90, help="WebP quality (default 90)")
    ap.add_argument(
        "--resample",
        choices=["lanczos", "bilinear", "bicubic", "nearest"],
        default="lanczos",
        help="Resampling filter",
    )
    ap.add_argument(
        "--exif-orient",
        action="store_true",
        help="Apply EXIF orientation before resizing",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached files even if they already exist",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows where the canonical image file does not exist",
    )
    ap.add_argument("--num-workers", type=int, default=8, help="Thread workers (default 8)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick tests")

    ap.add_argument(
        "--manifest-out",
        required=True,
        help="Output parquet path for manifest (DVC-tracked)",
    )
    return ap.parse_args()


def get_resample(name: str):
    # Pillow constants
    name = name.lower()
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "nearest":
        return Image.Resampling.NEAREST
    raise ValueError(f"Unknown resample: {name}")


def compute_new_size(w: int, h: int, max_side: Optional[int], short_side: Optional[int]) -> Tuple[int, int]:
    if max_side is not None:
        if max(w, h) <= max_side:
            return w, h
        scale = max_side / float(max(w, h))
    else:
        # short_side
        if min(w, h) <= short_side:
            return w, h
        scale = short_side / float(min(w, h))

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return nw, nh


@dataclass
class ResizeResult:
    sample_id: str
    src_path: str
    cached_path: str
    ok: bool
    reason: str
    src_w: Optional[int] = None
    src_h: Optional[int] = None
    out_w: Optional[int] = None
    out_h: Optional[int] = None


def save_image(img: Image.Image, out_path: Path, fmt: str, jpeg_quality: int, webp_quality: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to RGB for JPEG to avoid errors with alpha/P modes
    if fmt == "jpg":
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    elif fmt == "png":
        img.save(out_path, format="PNG", optimize=True)
    elif fmt == "webp":
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(out_path, format="WEBP", quality=webp_quality, method=6)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def process_one(
    sample_id: str,
    images_root: Path,
    cache_root: Path,
    out_ext: str,
    max_side: Optional[int],
    short_side: Optional[int],
    resample,
    exif_orient: bool,
    overwrite: bool,
    skip_missing: bool,
    jpeg_quality: int,
    webp_quality: int,
) -> ResizeResult:
    src_path = images_root / sample_id
    # Mirror sample_id under cache root, but force extension
    cached_rel = Path(sample_id).with_suffix(f".{out_ext}")
    out_path = cache_root / cached_rel

    if not src_path.exists():
        if skip_missing:
            return ResizeResult(sample_id, str(src_path), str(out_path), False, "missing_source")
        return ResizeResult(sample_id, str(src_path), str(out_path), False, "missing_source")

    if out_path.exists() and not overwrite:
        # Return dims unknown without opening; keep cheap
        return ResizeResult(sample_id, str(src_path), str(out_path), True, "exists_skipped")

    try:
        with Image.open(src_path) as img:
            if exif_orient:
                img = ImageOps.exif_transpose(img)

            w, h = img.size
            nw, nh = compute_new_size(w, h, max_side=max_side, short_side=short_side)

            # If already small enough, we still re-encode to cache format for consistency
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), resample=resample)

            save_image(img, out_path, out_ext, jpeg_quality=jpeg_quality, webp_quality=webp_quality)

            return ResizeResult(
                sample_id=sample_id,
                src_path=str(src_path),
                cached_path=str(out_path),
                ok=True,
                reason="resized" if (nw, nh) != (w, h) else "copied_reencoded",
                src_w=w,
                src_h=h,
                out_w=nw,
                out_h=nh,
            )
    except Exception as e:
        return ResizeResult(sample_id, str(src_path), str(out_path), False, f"error:{type(e).__name__}:{e}")


def main():
    args = parse_args()

    samples_df = pd.read_parquet(args.samples)
    if args.max_samples is not None:
        samples_df = samples_df.head(args.max_samples)

    if "sample_id" not in samples_df.columns:
        raise ValueError("samples.parquet must contain a 'sample_id' column")

    images_root = Path(os.path.abspath(args.images_root))
    cache_root = Path(os.path.abspath(args.cache_root))
    manifest_out = Path(args.manifest_out)

    resample = get_resample(args.resample)

    sample_ids = samples_df["sample_id"].astype(str).tolist()

    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = [
            ex.submit(
                process_one,
                sample_id=sid,
                images_root=images_root,
                cache_root=cache_root,
                out_ext=args.format,
                max_side=args.max_side,
                short_side=args.short_side,
                resample=resample,
                exif_orient=args.exif_orient,
                overwrite=args.overwrite,
                skip_missing=args.skip_missing,
                jpeg_quality=args.jpeg_quality,
                webp_quality=args.webp_quality,
            )
            for sid in sample_ids
        ]

        for fut in as_completed(futs):
            results.append(fut.result())

    # Build manifest
    rows = []
    n_ok = 0
    n_missing = 0
    n_err = 0
    n_skipped = 0

    for r in results:
        if r.ok:
            n_ok += 1
            if r.reason == "exists_skipped":
                n_skipped += 1
        else:
            if r.reason == "missing_source":
                n_missing += 1
            else:
                n_err += 1

        rows.append({
            "sample_id": r.sample_id,
            "src_path": r.src_path,
            "cached_path": r.cached_path,
            "ok": r.ok,
            "reason": r.reason,
            "src_w": r.src_w,
            "src_h": r.src_h,
            "out_w": r.out_w,
            "out_h": r.out_h,
            "resize_mode": "max_side" if args.max_side is not None else "short_side",
            "target_pixels": args.max_side if args.max_side is not None else args.short_side,
            "format": args.format,
            "resample": args.resample,
            "exif_orient": bool(args.exif_orient),
        })

    manifest_df = pd.DataFrame(rows).sort_values("sample_id")

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(manifest_out, index=False)

    # Console summary (useful for DVC logs)
    print(f"[cache] images_root={images_root}")
    print(f"[cache] cache_root={cache_root}")
    print(f"[cache] resize_mode={'max_side' if args.max_side is not None else 'short_side'} "
          f"target={args.max_side if args.max_side is not None else args.short_side}")
    print(f"[cache] format={args.format} resample={args.resample} exif_orient={args.exif_orient}")
    print(f"[cache] total={len(sample_ids)} ok={n_ok} skipped_existing={n_skipped} missing={n_missing} errors={n_err}")
    print(f"[cache] manifest_out={manifest_out}")


if __name__ == "__main__":
    main()