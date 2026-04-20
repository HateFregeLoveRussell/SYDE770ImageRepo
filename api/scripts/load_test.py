"""
load_test.py

Send all images in a directory to the API's /predict endpoint.
Generates traffic for Prometheus + Grafana dashboards.

Usage:
  python scripts/load_test.py --images ../../data/derived/yolo/test/images/
  python scripts/load_test.py --images ../../data/images/ --url http://localhost:6000 --models yolov8s_v2 yolov8s_v2_extended
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def send_image(url: str, image_path: Path, model: str | None = None) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, f"image/{image_path.suffix.lstrip('.')}")}
        data = {"model": model} if model else {}
        resp = requests.post(f"{url}/predict", files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the detection API")
    parser.add_argument("--images", required=True, help="Directory of images to send")
    parser.add_argument("--url", default="http://127.0.0.1:6000", help="API base URL")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model(s) to cycle through (omit for default only)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to send")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests (s)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} is not a directory")
        sys.exit(1)

    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in SUPPORTED
    )
    if args.limit:
        image_files = image_files[:args.limit]

    if not image_files:
        print(f"No supported images found in {images_dir}")
        sys.exit(1)

    # Verify API is up
    try:
        r = requests.get(f"{args.url}/health-status", timeout=5)
        r.raise_for_status()
        print(f"API is healthy: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.url}: {e}")
        sys.exit(1)

    models = args.models or [None]
    print(f"\nSending {len(image_files)} images to {args.url}/predict")
    if args.models:
        print(f"Cycling through models: {args.models}")
    print("-" * 60)

    total_dets = 0
    total_timmies = 0
    total_cups = 0
    latencies: list[float] = []
    errors = 0
    t_start = time.time()

    for i, img_path in enumerate(image_files):
        model = models[i % len(models)]
        try:
            t0 = time.perf_counter()
            result = send_image(args.url, img_path, model)
            elapsed = (time.perf_counter() - t0) * 1000

            n_det = len(result["predictions"])
            total_dets += n_det
            for d in result["predictions"]:
                if d["label"] == "timmies":
                    total_timmies += 1
                else:
                    total_cups += 1

            latencies.append(elapsed)
            model_tag = result["model_used"]
            det_summary = ", ".join(
                f"{d['label']}:{d['confidence']:.2f}" for d in result["predictions"]
            ) or "none"

            print(f"  [{i+1}/{len(image_files)}] {img_path.name:40s} "
                  f"{elapsed:6.0f}ms  dets={n_det}  model={model_tag}  [{det_summary}]")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(image_files)}] {img_path.name:40s}  ERROR: {e}")

        if args.delay > 0:
            time.sleep(args.delay)

    elapsed_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"  Images sent       : {len(image_files)}")
    print(f"  Errors            : {errors}")
    print(f"  Total detections  : {total_dets}")
    print(f"    timmies         : {total_timmies}")
    print(f"    cup             : {total_cups}")
    if latencies:
        latencies.sort()
        print(f"  Avg latency       : {sum(latencies)/len(latencies):.1f} ms")
        print(f"  p50 latency       : {latencies[len(latencies)//2]:.1f} ms")
        print(f"  p95 latency       : {latencies[int(len(latencies)*0.95)]:.1f} ms")
        print(f"  Max latency       : {max(latencies):.1f} ms")
    print(f"  Total time        : {elapsed_total:.1f} s")
    print(f"  Throughput        : {len(image_files)/elapsed_total:.1f} img/s")

    # Print final /metrics
    try:
        m = requests.get(f"{args.url}/metrics", timeout=5).json()
        print(f"\n  API /metrics:")
        for k, v in m.items():
            print(f"    {k}: {v}")
    except Exception:
        pass

    print()


if __name__ == "__main__":
    main()
