"""
train_yolov8_mlflow.py

Train a YOLOv8 model and log every metric to a local MLflow server.

Usage:
  python train_yolov8_mlflow.py                          # defaults
  python train_yolov8_mlflow.py --epochs 5               # quick sanity run
  python train_yolov8_mlflow.py --model yolov8s.pt --epochs 50 --imgsz 640
"""

import argparse
import pathlib
import platform

import mlflow
import torch
from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).parent.resolve()


def best_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    p = argparse.ArgumentParser(description="Train YOLOv8 and log to MLflow")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base weights (yolov8n/s/m/l/x.pt or path)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--data",
                   default=str(REPO_ROOT / "data" / "derived" / "yolo" / "data.yaml"))
    p.add_argument("--run-name", default=None,
                   help="MLflow run name (auto-generated if omitted)")
    p.add_argument("--project",
                   default=str(REPO_ROOT / "runs" / "yolov8"))
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

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLOv8-TimHortons")

    run_name = args.run_name or f"{pathlib.Path(args.model).stem}_ep{args.epochs}_img{args.imgsz}"

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  lr0     : {args.lr0}")
    print(f"  Data    : {data_yaml}")
    print(f"  Run     : {run_name}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "device": device,
            "data": str(data_yaml),
        })

        model = YOLO(args.model)
        model.add_callback("on_fit_epoch_end", make_mlflow_callback(_run=run))

        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            device=device,
            project=args.project,
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        final_metrics = {}
        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                try:
                    final_metrics[f"final/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass

        if final_metrics:
            mlflow.log_metrics(final_metrics)
            print("\nFinal metrics logged to MLflow:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")

        best_pt = pathlib.Path(args.project) / run_name / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print("\nSaved best.pt -> MLflow artifact")
        else:
            print(f"\nWARNING: best.pt not found at {best_pt}")  # noqa: G004

        print(f"\nMLflow run ID : {run.info.run_id}")
        print("View results  : http://localhost:5000")


if __name__ == "__main__":
    main()
