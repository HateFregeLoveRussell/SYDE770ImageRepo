"""
optuna_search.py

Multi-objective hyperparameter search for YOLOv8 using NSGA-II with k-fold
cross-validation. Each trial trains k folds and optimises mean F1 and mean
mAP50 across folds.

Usage:
  python optuna_search.py --n-trials 1  --epochs-per-trial 2 --k-folds 2   # smoke
  python optuna_search.py --n-trials 50 --epochs-per-trial 20 --k-folds 5  # full
  python optuna_search.py --n-trials 50 --epochs-per-trial 20 --k-folds 5 --resume
"""

import argparse
import csv
import datetime
import pathlib
import platform
import traceback

import cv2
import mlflow
import numpy as np
import optuna
import torch
from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
YOLO_DIR = REPO_ROOT / "data" / "derived" / "yolo"


def best_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def safe_key(k: str) -> str:
    """Sanitise a YOLO metric key for MLflow (no parentheses/slashes)."""
    return k.replace("(", "").replace(")", "").replace("/", "_")


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    """Sample the full 24-param search space."""
    return {
        # training / optimisation
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
        "lrf": trial.suggest_float("lrf", 1e-3, 0.1, log=True),
        "momentum": trial.suggest_float("momentum", 0.6, 0.98),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"]),
        "cos_lr": trial.suggest_categorical("cos_lr", [True, False]),
        # model / resolution
        "model": trial.suggest_categorical("model", ["yolov8n.pt", "yolov8s.pt"]),
        "imgsz": trial.suggest_categorical("imgsz", [480, 640]),
        # loss weights
        "box": trial.suggest_float("box", 0.02, 0.2),
        "cls": trial.suggest_float("cls", 0.2, 4.0),
        "dfl": trial.suggest_float("dfl", 0.5, 2.0),
        # augmentation
        "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
        "hsv_s": trial.suggest_float("hsv_s", 0.0, 0.9),
        "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9),
        "degrees": trial.suggest_float("degrees", 0.0, 45.0),
        "translate": trial.suggest_float("translate", 0.0, 0.2),
        "scale": trial.suggest_float("scale", 0.1, 0.9),
        "fliplr": trial.suggest_float("fliplr", 0.0, 0.5),
        "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        "mixup": trial.suggest_float("mixup", 0.0, 0.3),
        "close_mosaic": trial.suggest_int("close_mosaic", 0, 10),
        "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.3),
    }


def run_fold(
    hp: dict,
    epochs: int,
    batch: int,
    device: str,
    data_yaml: str,
    project: str,
    trial_num: int,
    fold_idx: int,
) -> tuple[float, float]:
    """Train one fold and return (F1, mAP50)."""
    model_name = hp["model"]
    imgsz = hp["imgsz"]
    train_hp = {k: v for k, v in hp.items() if k not in ("model", "imgsz")}

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=f"trial_{trial_num}_fold_{fold_idx}",
        exist_ok=True,
        verbose=False,
        **train_hp,
    )

    rd = results.results_dict if hasattr(results, "results_dict") else {}
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall = float(rd.get("metrics/recall(B)", 0.0))
    map50 = float(rd.get("metrics/mAP50(B)", 0.0))
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, map50


def save_pareto(study: optuna.Study, csv_path: pathlib.Path):
    """Write the Pareto frontier trials to CSV."""
    trials = study.best_trials
    if not trials:
        return

    rows = []
    for t in trials:
        row = {"trial": t.number, "avg_f1": t.values[0], "avg_map50": t.values[1]}
        row.update(t.params)
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nPareto frontier ({len(rows)} trials) saved to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Optuna k-fold CV YOLOv8 search")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--epochs-per-trial", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--k-folds", type=int, default=5,
                   help="Number of CV folds (must match prepare_yolo_split.py)")
    p.add_argument("--device", default=None, help="Force device (cuda/mps/cpu)")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study from SQLite")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or best_device()
    k = args.k_folds

    # Validate that fold data.yaml files exist
    fold_yamls: list[str] = []
    for i in range(k):
        p = YOLO_DIR / "folds" / f"fold_{i}" / "data.yaml"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\n"
                f"Run: python prepare_yolo_split.py --k-folds {k}"
            )
        fold_yamls.append(str(p))

    study_db = REPO_ROOT / f"optuna_study_cv{k}.db"
    pareto_csv = REPO_ROOT / f"pareto_frontier_cv{k}.csv"
    storage = f"sqlite:///{study_db}"
    study_name = f"yolov8_nsga2_cv{k}"
    project = str(REPO_ROOT / "runs" / f"optuna_cv{k}")

    if args.resume:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Resumed study with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            load_if_exists=True,
        )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"YOLOv8-TimHortons-CV{k}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'='*60}")
    print(f"  Trials  : {args.n_trials}")
    print(f"  Epochs  : {args.epochs_per_trial}")
    print(f"  Folds   : {k}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {device}")
    print(f"  Study   : {study_name}")
    print(f"  Storage : {storage}")
    print(f"  Resume  : {args.resume}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=f"optuna_cv{k}_{ts}") as parent_run:
        mlflow.log_params({
            "n_trials": args.n_trials,
            "epochs_per_trial": args.epochs_per_trial,
            "k_folds": k,
            "batch": args.batch,
            "device": device,
            "sampler": "NSGA-II",
        })

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            hp = suggest_hyperparams(trial)

            with mlflow.start_run(
                run_name=f"trial_{trial.number}", nested=True
            ):
                mlflow.log_params({safe_key(key): v for key, v in hp.items()})
                mlflow.log_param("trial_number", trial.number)

                fold_f1s: list[float] = []
                fold_map50s: list[float] = []

                for fold_i, fold_yaml in enumerate(fold_yamls):
                    try:
                        f1, map50 = run_fold(
                            hp, args.epochs_per_trial, args.batch, device,
                            fold_yaml, project, trial.number, fold_i,
                        )
                    except (RuntimeError, ValueError, OSError, cv2.error):
                        traceback.print_exc()
                        f1, map50 = 0.0, 0.0

                    fold_f1s.append(f1)
                    fold_map50s.append(map50)
                    mlflow.log_metrics({
                        f"fold_{fold_i}_f1": f1,
                        f"fold_{fold_i}_map50": map50,
                    })
                    print(f"    Fold {fold_i}: F1={f1:.4f}  mAP50={map50:.4f}")

                avg_f1 = float(np.mean(fold_f1s))
                avg_map50 = float(np.mean(fold_map50s))
                std_f1 = float(np.std(fold_f1s, ddof=1)) if k > 1 else 0.0
                std_map50 = float(np.std(fold_map50s, ddof=1)) if k > 1 else 0.0

                mlflow.log_metrics({
                    "avg_f1": avg_f1,
                    "avg_map50": avg_map50,
                    "std_f1": std_f1,
                    "std_map50": std_map50,
                })

                print(
                    f"  Trial {trial.number}: "
                    f"avg_F1={avg_f1:.4f}+-{std_f1:.4f}  "
                    f"avg_mAP50={avg_map50:.4f}+-{std_map50:.4f}"
                )

            return avg_f1, avg_map50

        study.optimize(objective, n_trials=args.n_trials)

        save_pareto(study, pareto_csv)
        if pareto_csv.exists():
            mlflow.log_artifact(str(pareto_csv))

    print(f"\nDone. {len(study.trials)} total trials in study.")
    print(f"MLflow parent run: {parent_run.info.run_id}")
    print("View results: http://localhost:5000")


if __name__ == "__main__":
    main()
