#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif
import piexif

pillow_heif.register_heif_opener()


def convert_one(src: Path, *, quality: int = 100, dry_run: bool = False) -> bool:
    """
    Convert one HEIC/HEIF file to JPG:
      - preserve EXIF + ICC profile when possible
      - apply EXIF orientation (so pixels are upright)
      - set Orientation tag to 1 after transpose
      - save JPEG at maximum quality (quality=100, subsampling=0)
      - delete source only after success
    Returns True if converted (or would convert in dry-run), False if skipped/failed.
    """
    dst = src.with_suffix(".jpg")
    if dst.exists():
        print(f"SKIP (jpg exists): {dst}")
        return False

    try:
        with Image.open(src) as im:
            # Preserve ICC profile if present
            icc_profile = im.info.get("icc_profile", None)

            # Extract EXIF bytes (may be missing)
            exif_bytes = im.info.get("exif", b"")

            # Ensure the pixels are correctly oriented
            im = ImageOps.exif_transpose(im)

            # JPEG needs RGB (drop alpha if any)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            # If we preserved EXIF, set Orientation=1 because we've already transposed pixels
            out_exif_bytes = b""
            if exif_bytes:
                try:
                    exif_dict = piexif.load(exif_bytes)
                    # 274 is the Orientation tag in 0th IFD
                    if "0th" in exif_dict:
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                    out_exif_bytes = piexif.dump(exif_dict)
                except Exception as e:
                    print(f"WARNING: Could not parse/modify EXIF for {src.name}: {e}")
                    # Fall back to saving without EXIF rather than failing conversion
                    out_exif_bytes = b""

            if dry_run:
                print(f"DRY RUN: would convert {src} -> {dst} and delete source")
                return True

            # Save JPEG with maximum quality (subsampling=0 preserves chroma detail)
            save_kwargs = {
                "format": "JPEG",
                "quality": quality,
                "subsampling": 0,
                "optimize": False,   # optimize can slightly change encoding; keep simple
            }
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            if out_exif_bytes:
                save_kwargs["exif"] = out_exif_bytes

            im.save(dst, **save_kwargs)

        # Basic sanity check: file exists and non-trivial size
        if not dst.exists() or dst.stat().st_size < 1024:
            print(f"ERROR: Output JPG seems invalid: {dst}")
            if dst.exists():
                dst.unlink(missing_ok=True)
            return False

        # Delete original only after successful save
        src.unlink()
        print(f"OK: {src} -> {dst} (deleted source)")
        return True

    except Exception as e:
        print(f"ERROR converting {src}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HEIC/HEIF to high-quality JPG, preserving EXIF/ICC, then delete source.")
    ap.add_argument("root", type=Path, help="Root directory to scan recursively")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality (default: 100)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without writing/deleting")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        return 2

    exts = {".heic", ".heif"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print("No .heic/.heif files found.")
        return 0

    converted = 0
    failed = 0
    skipped = 0

    for src in sorted(files):
        ok = convert_one(src, quality=args.quality, dry_run=args.dry_run)
        if ok:
            converted += 1
        else:
            # either skipped or failed; distinguish by whether jpg exists
            if src.with_suffix(".jpg").exists():
                skipped += 1
            else:
                failed += 1

    print(f"\nDone. converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
