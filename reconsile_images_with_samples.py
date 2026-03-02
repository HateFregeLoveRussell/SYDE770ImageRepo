#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Make data/images consistent with data/index/samples.parquet by deleting orphan image files"
    )
    ap.add_argument("--samples", default="data/index/samples.parquet", help="Path to samples.parquet")
    ap.add_argument("--images-dir", default="data/images", help="Images directory to reconcile")
    ap.add_argument("--dry-run", action="store_true", help="Report actions, do not delete")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if samples.parquet references files that do not exist on disk")
    ap.add_argument("--ext", nargs="*", default=[],
                    help="Optional whitelist of extensions (e.g. --ext .jpg .jpeg .png). If omitted, all files count.")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    images_dir = Path(args.images_dir)

    if not samples_path.exists():
        raise SystemExit(f"Missing: {samples_path}")
    if not images_dir.exists():
        raise SystemExit(f"Missing: {images_dir}")

    df = pd.read_parquet(samples_path)
    if "sample_id" not in df.columns:
        raise SystemExit(f"{samples_path} missing required column 'sample_id'")

    # Canonical expected paths (as stored): e.g. "images/foo.jpg"
    expected: Set[str] = set(df["sample_id"].astype(str).tolist())

    # Build actual paths relative to the "data/" root (so they match sample_id format)
    # images_dir is usually "data/images" so root is its parent ("data")
    data_root = images_dir.parent

    def ext_ok(p: Path) -> bool:
        if not args.ext:
            return True
        return p.suffix.lower() in {e.lower() for e in args.ext}

    actual_files = [p for p in images_dir.rglob("*") if p.is_file() and ext_ok(p)]

    # Orphans = on disk but not in samples.parquet
    orphan_files = []
    for p in actual_files:
        rel = p.relative_to(data_root).as_posix()  # -> "images/..."
        if rel not in expected:
            orphan_files.append(p)

    # Missing expected = in samples.parquet but not on disk
    missing_expected = []
    for sid in expected:
        p = data_root / sid
        if not p.exists():
            missing_expected.append(sid)

    print(f"Images dir: {images_dir}")
    print(f"Samples rows: {len(df)}")
    print(f"Actual image files found: {len(actual_files)}")
    print(f"Orphan files to delete: {len(orphan_files)}")
    print(f"Missing expected files: {len(missing_expected)}")

    if missing_expected:
        preview = "\n".join(missing_expected[:20])
        msg = f"Missing expected files (first 20):\n{preview}"
        if args.strict:
            raise SystemExit(msg)
        else:
            print("WARNING:", msg)

    if orphan_files:
        print("Preview orphan deletions (first 20):")
        for p in orphan_files[:20]:
            print("  ", p.as_posix())

    if args.dry_run:
        print("[dry-run] No files deleted")
        return 0

    deleted = 0
    for p in orphan_files:
        p.unlink()
        deleted += 1

    print(f"Deleted {deleted} orphan files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())