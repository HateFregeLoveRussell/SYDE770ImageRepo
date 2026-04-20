"""
Rename MLflow runs in the Ultralytics optuna_cv5 experiment from random names
(e.g. dashing-snipe-233) to the YOLO run name (trial_<n>_fold_<k>).

The correct name is already logged as param 'name' by Ultralytics.

Usage:
  python rename_mlflow_cv_runs.py --dry-run    # preview
  python rename_mlflow_cv_runs.py              # apply
"""

import argparse
import re

from mlflow.tracking import MlflowClient

# Default: experiment backed by runs/optuna_cv5 (Ultralytics auto-logging)
DEFAULT_EXP_ID = "6"
NAME_PARAM_RE = re.compile(r"^trial_(\d+)_fold_(\d+)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-id",
        default=DEFAULT_EXP_ID,
        help="MLflow experiment id for runs/optuna_cv5 (default: 6)",
    )
    p.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print renames only, do not write",
    )
    args = p.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        max_results=50000,
    )

    renamed = 0
    skipped = 0
    for run in runs:
        rid = run.info.run_id
        old = run.info.run_name
        params = run.data.params
        yolo_name = params.get("name", "")
        if not NAME_PARAM_RE.match(yolo_name):
            skipped += 1
            continue
        if old == yolo_name:
            skipped += 1
            continue
        print(f"  {old!r} -> {yolo_name!r}  ({rid[:8]}...)")
        if not args.dry_run:
            client.set_tag(rid, "mlflow.runName", yolo_name)
        renamed += 1

    print()
    print(f"Renamed: {renamed}  |  unchanged/skipped: {skipped}  |  total runs: {len(runs)}")
    if args.dry_run and renamed:
        print("\nRun without --dry-run to apply.")


if __name__ == "__main__":
    main()
