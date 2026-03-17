"""
optuna_search.py

Multi-objective hyperparameter search for YOLOv8 using NSGA-II.
Optimises two objectives: F1 score and mAP50.
Each trial trains a short run, logs to MLflow, and the study persists to SQLite.

Usage:
  python optuna_search.py --n-trials 2  --epochs-per-trial 3          # smoke test
  python optuna_search.py --n-trials 50 --epochs-per-trial 20 --device cuda
  python optuna_search.py --n-trials 50 --epochs-per-trial 20 --resume # continue
"""

import argparse
import datetime
import pathlib
import platform
import traceback

import mlflow
import optuna
import torch
from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
DATA_YAML = REPO_ROOT / "data" / "derived" / "yolo" / "data.yaml"
STUDY_DB = REPO_ROOT / "optuna_study.db"
PARETO_CSV = REPO_ROOT / "pareto_frontier.csv"


def best_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def safe_key(k: str) -> str:
    """Sanitise a YOLO metric key for MLflow (no parentheses)."""
    return k.replace("(", "").replace(")", "").replace("/", "_")


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    """Sample the full 25-param search space."""
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


def run_trial(hp: dict, epochs: int, batch: int, device: str, data: str) -> tuple[float, float]:
    """Train YOLOv8 with the given hyperparams and return (F1, mAP50)."""
    model_name = hp.pop("model")
    imgsz = hp.pop("imgsz")

    model = YOLO(model_name)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(REPO_ROOT / "runs" / "optuna"),
        name="trial",
        exist_ok=True,
        verbose=False,
        **hp,
    )

    rd = results.results_dict if hasattr(results, "results_dict") else {}
    precision = float(rd.get("metrics/precision(B)", 0.0))
    recall = float(rd.get("metrics/recall(B)", 0.0))
    map50 = float(rd.get("metrics/mAP50(B)", 0.0))
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, map50


def save_pareto(study: optuna.Study):
    """Write the Pareto frontier trials to CSV."""
    trials = study.best_trials
    if not trials:
        return

    rows = []
    for t in trials:
        row = {"trial": t.number, "f1": t.values[0], "map50": t.values[1]}
        row.update(t.params)
        rows.append(row)

    import csv
    fieldnames = rows[0].keys()
    with open(PARETO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nPareto frontier ({len(rows)} trials) saved to {PARETO_CSV}")


def parse_args():
    p = argparse.ArgumentParser(description="Optuna multi-objective YOLOv8 search")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--epochs-per-trial", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="Force device (cuda/mps/cpu)")
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--resume", action="store_true",
                   help="Load existing study from SQLite instead of creating new")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or best_device()

    if not pathlib.Path(args.data).exists():
        raise FileNotFoundError(
            f"data.yaml not found at {args.data}\n"
            "Run prepare_yolo_split.py first."
        )

    storage = f"sqlite:///{STUDY_DB}"
    study_name = "yolov8_nsga2"

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
    mlflow.set_experiment("YOLOv8-TimHortons")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    parent_run_name = f"optuna_search_{ts}"

    print(f"\n{'='*60}")
    print(f"  Trials  : {args.n_trials}")
    print(f"  Epochs  : {args.epochs_per_trial}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {device}")
    print(f"  Storage : {storage}")
    print(f"  Resume  : {args.resume}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.log_params({
            "n_trials": args.n_trials,
            "epochs_per_trial": args.epochs_per_trial,
            "batch": args.batch,
            "device": device,
            "sampler": "NSGA-II",
        })

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            hp = suggest_hyperparams(trial)

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params({safe_key(k): v for k, v in hp.items()})
                mlflow.log_param("trial_number", trial.number)

                try:
                    f1, map50 = run_trial(
                        hp.copy(), args.epochs_per_trial, args.batch, device, args.data
                    )
                except (RuntimeError, ValueError, OSError):
                    traceback.print_exc()
                    f1, map50 = 0.0, 0.0

                mlflow.log_metrics({"val_f1": f1, "val_map50": map50})
                print(f"  Trial {trial.number}: F1={f1:.4f}  mAP50={map50:.4f}")

            return f1, map50

        study.optimize(objective, n_trials=args.n_trials)

        save_pareto(study)

        if PARETO_CSV.exists():
            mlflow.log_artifact(str(PARETO_CSV))

    print(f"\nDone. {len(study.trials)} total trials in study.")
    print(f"MLflow parent run: {parent_run.info.run_id}")
    print("View results: http://localhost:5000")


if __name__ == "__main__":
    main()
