"""
train_yolov8_mlflow.py

Train the final YOLOv8 model on the full Train+CV (80%) set using the best
hyperparameters from Optuna, then evaluate on the held-out 20% test set.

Usage:
  python train_yolov8_mlflow.py                             # defaults
  python train_yolov8_mlflow.py --epochs 5                  # quick sanity
  python train_yolov8_mlflow.py --model yolov8s.pt --epochs 50 \\
      --lr0 0.005 --lrf 0.01 --optimizer AdamW              # best config
"""

import argparse
import pathlib
import platform

import mlflow
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
    return k.replace("(", "").replace(")", "").replace("/", "_")


def make_mlflow_callback(_run):
    """Return an Ultralytics on_fit_epoch_end callback that logs to MLflow."""

    def on_fit_epoch_end(trainer):
        metrics = {}

        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass

        if hasattr(trainer, "loss_names") and hasattr(trainer, "tloss"):
            losses = (
                trainer.tloss
                if hasattr(trainer.tloss, "__iter__")
                else [trainer.tloss]
            )
            for name, val in zip(trainer.loss_names, losses):
                try:
                    metrics[f"train/{name}"] = float(val)
                except (TypeError, ValueError):
                    pass

        if hasattr(trainer, "optimizer") and trainer.optimizer:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                metrics[f"lr/pg{i}"] = pg.get("lr", 0.0)

        epoch = trainer.epoch + 1
        if metrics:
            mlflow.log_metrics(metrics, step=epoch)

    return on_fit_epoch_end


def parse_args():
    p = argparse.ArgumentParser(
        description="Train final YOLOv8 on trainval (80%) and evaluate on test (20%)"
    )

    # core
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base weights (yolov8n/s/m/l/x.pt or path)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--data", default=str(YOLO_DIR / "trainval" / "data.yaml"),
                   help="Training data.yaml (default: trainval/data.yaml)")
    p.add_argument("--test-data", default=str(YOLO_DIR / "test" / "data.yaml"),
                   help="Test data.yaml for held-out evaluation")
    p.add_argument("--run-name", default=None,
                   help="MLflow run name (auto-generated if omitted)")
    p.add_argument("--project", default=str(REPO_ROOT / "runs" / "final"))

    # training / optimisation
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)

    # loss weights
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)

    # augmentation
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--copy-paste", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()
    device = best_device()

    data_yaml = pathlib.Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}\n"
            "Run prepare_yolo_split.py first."
        )

    hp = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
    }

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLOv8-TimHortons-Final")

    run_name = args.run_name or f"{pathlib.Path(args.model).stem}_ep{args.epochs}_img{args.imgsz}"

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Data    : {data_yaml}")
    print(f"  Run     : {run_name}")
    for k, v in hp.items():
        print(f"  {k:15s}: {v}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": device,
            "data": str(data_yaml),
            **{safe_key(k): v for k, v in hp.items()},
        })

        model = YOLO(args.model)
        model.add_callback("on_fit_epoch_end", make_mlflow_callback(_run=run))

        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=run_name,
            exist_ok=True,
            verbose=True,
            **hp,
        )

        # Log final training metrics
        if hasattr(results, "results_dict"):
            trainval_metrics = {}
            for k, v in results.results_dict.items():
                try:
                    trainval_metrics[f"trainval/{safe_key(k)}"] = float(v)
                except (TypeError, ValueError):
                    pass
            if trainval_metrics:
                mlflow.log_metrics(trainval_metrics)
                print("\nTrain+CV metrics:")
                for k, v in trainval_metrics.items():
                    print(f"  {k}: {v:.4f}")

        # ---- Evaluate on held-out test set ------------------------------------
        test_yaml = pathlib.Path(args.test_data)
        if test_yaml.exists():
            print(f"\nEvaluating on held-out test set: {test_yaml}")
            test_results = model.val(data=str(test_yaml))

            if hasattr(test_results, "results_dict"):
                test_metrics = {}
                for k, v in test_results.results_dict.items():
                    try:
                        test_metrics[f"test/{safe_key(k)}"] = float(v)
                    except (TypeError, ValueError):
                        pass

                p = float(test_results.results_dict.get("metrics/precision(B)", 0))
                r = float(test_results.results_dict.get("metrics/recall(B)", 0))
                test_metrics["test/f1"] = 2 * p * r / (p + r + 1e-9)

                if test_metrics:
                    mlflow.log_metrics(test_metrics)
                    print("\nTest set metrics:")
                    for k, v in test_metrics.items():
                        print(f"  {k}: {v:.4f}")
        else:
            print(f"\nWARNING: test data.yaml not found at {test_yaml}, skipping test eval")

        # ---- Save model weights -----------------------------------------------
        weights_dir = pathlib.Path(args.project) / run_name / "weights"
        for name in ("best.pt", "last.pt"):
            pt = weights_dir / name
            if pt.exists():
                mlflow.log_artifact(str(pt), artifact_path="weights")
                print(f"Saved {name} -> MLflow artifact")

        print(f"\nMLflow run ID : {run.info.run_id}")
        print("View results  : http://localhost:5000")


if __name__ == "__main__":
    main()
