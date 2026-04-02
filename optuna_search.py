"""
optuna_search.py

3-phase Bayesian hyperparameter search for YOLOv8 with k-fold
cross-validation.  Single objective: maximise mean mAP50-95.

Phases
  1 - Architecture : model, imgsz, freeze       (18 trials, 40 ep)
  2 - Optimizer/LR : optimizer, lr0, lrf, wd    (24 trials, 60 ep)
  3 - Loss weights : box, cls, dfl              (14 trials, 80 ep)

Usage
  python optuna_search.py --run-all --k-folds 2 --dry-run   # smoke test
  python optuna_search.py --run-all --k-folds 5              # full pipeline
  python optuna_search.py --phase 1 --k-folds 5              # single phase
"""

from __future__ import annotations

import argparse
import datetime
import gc
import pathlib
import platform
import traceback

import cv2
import mlflow
import numpy as np
import optuna
import torch
import yaml
from ultralytics import YOLO, settings as yolo_settings

yolo_settings.update({"mlflow": False})

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
YOLO_DIR = REPO_ROOT / "data" / "derived" / "yolo"

FIXED_AUG: dict[str, float | int] = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0,
    "close_mosaic": 0, "copy_paste": 0.0,
}

PHASE_CFG = {
    1: {"n_trials": 18, "epochs": 20, "patience": 10, "k_folds": 3},
    2: {"n_trials": 24, "epochs": 30, "patience": 15, "k_folds": 3},
    3: {"n_trials": 14, "epochs": 40, "patience": 20, "k_folds": 5},
}


# -- helpers ------------------------------------------------------------------

def best_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_key(k: str) -> str:
    return k.replace("(", "").replace(")", "").replace("/", "_")


# -- phase suggest functions --------------------------------------------------

def suggest_phase1(trial: optuna.Trial) -> dict:
    """Phase 1: architecture - model, imgsz, freeze."""
    return {
        "model": trial.suggest_categorical("model", ["yolov8n.pt", "yolov8s.pt"]),
        "imgsz": trial.suggest_categorical("imgsz", [480, 512, 640]),
        "freeze": trial.suggest_categorical("freeze", [0, 10, 20]),
        "optimizer": "AdamW", "lr0": 1e-3, "lrf": 1e-2,
        "weight_decay": 1e-4, "warmup_epochs": 2,
        "momentum": 0.937, "cos_lr": False, "dropout": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
        **FIXED_AUG,
    }


def suggest_phase2(trial: optuna.Trial, p1: dict) -> dict:
    """Phase 2: optimizer / LR with conditional lr0 ranges."""
    opt = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    lr0 = (trial.suggest_float("lr0_sgd", 1e-3, 3e-2, log=True)
           if opt == "SGD"
           else trial.suggest_float("lr0_adamw", 1e-4, 3e-3, log=True))

    return {
        "model": p1["model"], "imgsz": p1["imgsz"], "freeze": p1["freeze"],
        "optimizer": opt, "lr0": lr0,
        "lrf": trial.suggest_float("lrf", 1e-2, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 3),
        "momentum": 0.937, "cos_lr": False, "dropout": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
        **FIXED_AUG,
    }


def suggest_phase3(trial: optuna.Trial, p1: dict, p2: dict) -> dict:
    """Phase 3: loss weights - box, cls, dfl."""
    return {
        "model": p1["model"], "imgsz": p1["imgsz"], "freeze": p1["freeze"],
        "optimizer": p2["optimizer"], "lr0": p2["lr0"],
        "lrf": p2["lrf"], "weight_decay": p2["weight_decay"],
        "warmup_epochs": p2["warmup_epochs"],
        "momentum": 0.937, "cos_lr": False, "dropout": 0.0,
        "box": trial.suggest_float("box", 0.03, 0.15),
        "cls": trial.suggest_float("cls", 0.3, 2.0),
        "dfl": trial.suggest_float("dfl", 0.8, 1.5),
        **FIXED_AUG,
    }


# -- fold training ------------------------------------------------------------

def run_fold(
    hp: dict, epochs: int, patience: int, batch: int, device: str,
    data_yaml: str, project: str, phase: int, trial_num: int, fold_idx: int,
) -> dict:
    """Train one fold and return metrics dict."""
    model_name = hp.pop("model")
    if hp.get("freeze") == 0:
        hp["freeze"] = None

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml, epochs=epochs, patience=patience,
        batch=batch, device=device, project=project,
        name=f"p{phase}_t{trial_num}_f{fold_idx}",
        exist_ok=True, verbose=False, **hp,
    )

    rd = results.results_dict if hasattr(results, "results_dict") else {}
    prec = float(rd.get("metrics/precision(B)", 0.0))
    rec = float(rd.get("metrics/recall(B)", 0.0))
    metrics = {
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec + 1e-9),
    }

    del results, model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# -- study helpers ------------------------------------------------------------

def _study_name(phase: int, k: int, dry: bool) -> str:
    return f"yolov8_tpe_cv{k}_p{phase}" + ("_dry" if dry else "")


def _db_path(phase: int, k: int, dry: bool) -> pathlib.Path:
    return REPO_ROOT / f"optuna_study_cv{k}_p{phase}{'_dry' if dry else ''}.db"


def load_phase_best(phase: int, k: int, dry: bool = False) -> dict:
    """Load best trial params from a completed phase's SQLite DB."""
    db = _db_path(phase, k, dry)
    if not db.exists():
        raise FileNotFoundError(f"Phase {phase} DB not found: {db}")
    study = optuna.load_study(
        study_name=_study_name(phase, k, dry),
        storage=f"sqlite:///{db}",
    )
    best = study.best_trial
    print(f"  Loaded Phase {phase} best: trial {best.number}, "
          f"mAP50-95={best.value:.4f}, params={best.params}")
    return best.params


def normalize_p2(raw: dict) -> dict:
    """Resolve conditional lr0 name from Phase 2 best trial params."""
    return {
        "optimizer": raw["optimizer"],
        "lr0": raw.get("lr0_sgd") or raw.get("lr0_adamw"),
        "lrf": raw["lrf"],
        "weight_decay": raw["weight_decay"],
        "warmup_epochs": raw["warmup_epochs"],
    }


# -- phase runner -------------------------------------------------------------

def run_phase(
    phase: int, k: int, fold_yamls: list[str], batch: int, device: str,
    project: str, *, p1: dict | None = None, p2: dict | None = None,
    dry_run: bool = False, nested: bool = True,
) -> dict:
    """Run one Optuna phase. Returns best trial's raw suggested params."""
    cfg = PHASE_CFG[phase]
    n_trials = 2 if dry_run else cfg["n_trials"]
    epochs = 2 if dry_run else cfg["epochs"]
    phase_k = 2 if dry_run else cfg.get("k_folds", k)
    phase_folds = fold_yamls[:phase_k]
    patience = 0 if dry_run else cfg["patience"]

    sname = _study_name(phase, k, dry_run)
    storage = f"sqlite:///{_db_path(phase, k, dry_run)}"

    study = optuna.create_study(
        study_name=sname, storage=storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)

    print(f"\n{'='*60}")
    print(f"  Phase     : {phase}  ({'DRY RUN' if dry_run else 'FULL'})")
    print(f"  Trials    : {n_trials} ({done} done, {remaining} remaining)")
    print(f"  Epochs    : {epochs}")
    print(f"  Patience  : {patience}")
    print(f"  Folds     : {phase_k}")
    print(f"  Study     : {sname}")
    if p1:
        print(f"  P1 locked : {p1}")
    if p2:
        print(f"  P2 locked : {p2}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print(f"  Phase {phase} already complete ({done} trials), skipping.")
        return study.best_trial.params

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    with mlflow.start_run(run_name=f"phase{phase}_{ts}", nested=nested):
        mlflow.log_params({
            "phase": phase, "n_trials": n_trials, "epochs": epochs,
            "patience": patience, "k_folds": phase_k, "dry_run": dry_run,
        })

        def objective(trial: optuna.Trial) -> float:
            if phase == 1:
                hp = suggest_phase1(trial)
            elif phase == 2:
                hp = suggest_phase2(trial, p1)
            else:
                hp = suggest_phase3(trial, p1, p2)

            with mlflow.start_run(
                run_name=f"p{phase}_trial_{trial.number}", nested=True
            ):
                mlflow.log_params({safe_key(key): v for key, v in hp.items()})

                fold_results: list[dict] = []
                for fi, fy in enumerate(phase_folds):
                    try:
                        m = run_fold(
                            hp.copy(), epochs, patience, batch, device,
                            fy, project, phase, trial.number, fi,
                        )
                    except (RuntimeError, ValueError, OSError, cv2.error):
                        traceback.print_exc()
                        m = dict.fromkeys(
                            ["map50_95", "map50", "precision", "recall", "f1"],
                            0.0,
                        )

                    fold_results.append(m)
                    mlflow.log_metrics(
                        {f"fold_{fi}_{mk}": mv for mk, mv in m.items()}
                    )
                    print(f"    Fold {fi}: mAP50-95={m['map50_95']:.4f}  "
                          f"F1={m['f1']:.4f}")

                avgs = {mk: float(np.mean([r[mk] for r in fold_results]))
                        for mk in fold_results[0]}
                std = (float(np.std(
                    [r["map50_95"] for r in fold_results], ddof=1
                )) if phase_k > 1 else 0.0)

                mlflow.log_metrics({f"avg_{mk}": mv for mk, mv in avgs.items()})
                mlflow.log_metrics({"std_map50_95": std})

                print(f"  Trial {trial.number}: avg_mAP50-95="
                      f"{avgs['map50_95']:.4f}\u00b1{std:.4f}")

            return avgs["map50_95"]

        if done > 0:
            print(f"  Resuming: {done}/{n_trials} done, running {remaining} more")
        study.optimize(objective, n_trials=remaining)

    best = study.best_trial
    print(f"\nPhase {phase} done. Best: trial {best.number}, "
          f"mAP50-95={best.value:.4f}")
    print(f"  Params: {best.params}")
    return best.params


# -- final config builder -----------------------------------------------------

def build_final_config(p1_raw: dict, p2_raw: dict, p3_raw: dict) -> dict:
    """Combine all phase winners into a single training config."""
    p2 = normalize_p2(p2_raw)
    return {
        "model": p1_raw["model"], "imgsz": p1_raw["imgsz"],
        "freeze": p1_raw["freeze"],
        **p2,
        "momentum": 0.937, "cos_lr": False, "dropout": 0.0,
        "box": p3_raw["box"], "cls": p3_raw["cls"], "dfl": p3_raw["dfl"],
        **FIXED_AUG,
    }


# -- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-phase Bayesian HP search for YOLOv8"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-all", action="store_true",
                   help="Run all 3 phases back-to-back")
    g.add_argument("--phase", type=int, choices=[1, 2, 3],
                   help="Run a single phase")
    p.add_argument("--dry-run", action="store_true",
                   help="Smoke test (2 trials, 2 epochs per phase)")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    return p.parse_args()


# -- main ---------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device or best_device()
    k = args.k_folds
    dry = args.dry_run

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_yamls: list[str] = []
    for i in range(k):
        p = YOLO_DIR / "folds" / f"fold_{i}" / "data.yaml"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\n"
                f"Run: python prepare_yolo_split.py --k-folds {k}"
            )
        fold_yamls.append(str(p))

    project = str(REPO_ROOT / "runs" / "optuna_phased")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(
        f"YOLOv8-Phased-CV{k}" + ("-DryRun" if dry else "")
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if args.run_all:
        with mlflow.start_run(run_name=f"phased_search_{ts}"):
            p1_raw = run_phase(
                1, k, fold_yamls, args.batch, device, project,
                dry_run=dry, nested=True,
            )

            p2_raw = run_phase(
                2, k, fold_yamls, args.batch, device, project,
                p1=p1_raw, dry_run=dry, nested=True,
            )

            p2_clean = normalize_p2(p2_raw)
            p3_raw = run_phase(
                3, k, fold_yamls, args.batch, device, project,
                p1=p1_raw, p2=p2_clean, dry_run=dry, nested=True,
            )

            config = build_final_config(p1_raw, p2_raw, p3_raw)
            cfg_path = REPO_ROOT / "final_config.yaml"
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
            mlflow.log_artifact(str(cfg_path))

            print(f"\n{'='*60}")
            print("  ALL PHASES COMPLETE")
            print(f"  Config : {cfg_path}")
            print(f"  Next   : python train_yolov8_mlflow.py "
                  f"--config final_config.yaml --epochs 200")
            print(f"{'='*60}")
    else:
        phase = args.phase
        p1 = load_phase_best(1, k, dry) if phase >= 2 else None
        p2_raw_loaded = load_phase_best(2, k, dry) if phase >= 3 else None
        p2 = normalize_p2(p2_raw_loaded) if p2_raw_loaded else None

        best = run_phase(
            phase, k, fold_yamls, args.batch, device, project,
            p1=p1, p2=p2, dry_run=dry, nested=False,
        )

        if phase == 3 and p1 is not None and p2_raw_loaded is not None:
            config = build_final_config(p1, p2_raw_loaded, best)
            cfg_path = REPO_ROOT / "final_config.yaml"
            cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
            print(f"\nFinal config saved: {cfg_path}")


if __name__ == "__main__":
    main()
