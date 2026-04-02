"""
train_yolov8_mlflow.py

Train the final YOLOv8 model on the full Train+CV (80%) set using the best
hyperparameters from Optuna, then evaluate on the held-out 20% test set.

Usage:
  python train_yolov8_mlflow.py --config final_config.yaml --epochs 200
  python train_yolov8_mlflow.py --epochs 5                  # quick sanity
  python train_yolov8_mlflow.py --model yolov8s.pt --epochs 50 \
      --lr0 0.005 --lrf 0.01 --optimizer AdamW              # manual config
"""

import argparse
import pathlib
import platform

import mlflow
import torch
import yaml
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


def make_mlflow_callback(_run, train_eval_yaml: str | None = None,
                         train_eval_every: int = 2):
    """Return an Ultralytics on_fit_epoch_end callback that logs to MLflow.

    If train_eval_yaml is provided, runs detection metrics on the training
    data every `train_eval_every` epochs and logs them with a 'train_det/' prefix.
    """

    def on_fit_epoch_end(trainer):
        metrics = {}

        if hasattr(trainer, "metrics") and trainer.metrics:
            for k, v in trainer.metrics.items():
                try:
                    metrics[f"test_metrics/{safe_key(k)}"] = float(v)
                except (TypeError, ValueError):
                    pass

        if hasattr(trainer, "validator") and hasattr(trainer.validator, "loss"):
            val_loss = trainer.validator.loss
            if val_loss is not None and hasattr(trainer, "loss_names"):
                val_losses = (val_loss.cpu().tolist()
                              if hasattr(val_loss, "tolist")
                              else [float(val_loss)])
                for name, val in zip(trainer.loss_names, val_losses):
                    try:
                        metrics[f"test/{name}"] = float(val)
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

        if train_eval_yaml and epoch % train_eval_every == 0:
            try:
                train_results = trainer.model.val(data=train_eval_yaml)
                if hasattr(train_results, "results_dict"):
                    for k, v in train_results.results_dict.items():
                        try:
                            metrics[f"train_metrics/{safe_key(k)}"] = float(v)
                        except (TypeError, ValueError):
                            pass
                    p = float(train_results.results_dict.get(
                        "metrics/precision(B)", 0))
                    r = float(train_results.results_dict.get(
                        "metrics/recall(B)", 0))
                    metrics["train_metrics/f1"] = (
                        2 * p * r / (p + r + 1e-9))
            except Exception:
                pass

        if metrics:
            mlflow.log_metrics(metrics, step=epoch)

    return on_fit_epoch_end


def parse_args():
    p = argparse.ArgumentParser(
        description="Train final YOLOv8 on trainval (80%) and evaluate on test (20%)"
    )

    p.add_argument("--config", default=None,
                   help="Path to YAML config from Optuna (e.g. final_config.yaml)")

    # core / training schedule
    p.add_argument("--model", default="yolov8n.pt",
                   help="Base weights (yolov8n/s/m/l/x.pt or path)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for reproducibility")
    p.add_argument("--freeze", type=int, default=10,
                   help="Freeze first N layers (0 = no freezing)")
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

    # augmentation (all default to 0 - relying on pre-augmented dataset)
    p.add_argument("--hsv-h", type=float, default=0.0)
    p.add_argument("--hsv-s", type=float, default=0.0)
    p.add_argument("--hsv-v", type=float, default=0.0)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.0)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=0)
    p.add_argument("--copy-paste", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()
    device = best_device()

    # ---- Build hyperparameters from config file or CLI args -------------------
    if args.config:
        cfg_path = pathlib.Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        model_name = config.pop("model", args.model)
        imgsz = config.pop("imgsz", args.imgsz)

        freeze_val = config.pop("freeze", args.freeze)
        if freeze_val and freeze_val > 0:
            config["freeze"] = freeze_val

        hp = config
    else:
        model_name = args.model
        imgsz = args.imgsz
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
        if args.freeze > 0:
            hp["freeze"] = args.freeze

    data_yaml = pathlib.Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}\n"
            "Run prepare_yolo_split.py first."
        )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLOv8-TimHortons-Final")

    run_name = (args.run_name
                or f"{pathlib.Path(model_name).stem}_ep{args.epochs}_img{imgsz}")

    print(f"\n{'='*60}")
    print(f"  Model    : {model_name}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Patience : {args.patience}")
    print(f"  ImgSz    : {imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Data     : {data_yaml}")
    print(f"  Config   : {args.config or '(CLI args)'}")
    print(f"  Run      : {run_name}")
    for hp_key, hp_val in hp.items():
        print(f"  {hp_key:15s}: {hp_val}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "model": model_name,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "imgsz": imgsz,
            "batch": args.batch,
            "device": device,
            "data": str(data_yaml),
            "config_file": str(args.config) if args.config else "none",
            **{safe_key(k): v for k, v in hp.items()},
        })

        train_eval_yaml_path = pathlib.Path(args.project) / run_name / "train_eval.yaml"
        train_eval_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        train_eval_yaml_path.write_text(
            f"path: {YOLO_DIR}\n"
            f"train: trainval/images/train\n"
            f"val: trainval/images/train\n\n"
            f"nc: 2\nnames:\n  0: NonTimHortonsCup\n  1: TimHortonsCup\n"
        )

        model = YOLO(model_name)
        model.add_callback("on_fit_epoch_end", make_mlflow_callback(
            _run=run,
            train_eval_yaml=str(train_eval_yaml_path),
            train_eval_every=2,
        ))

        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            imgsz=imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=run_name,
            exist_ok=True,
            verbose=True,
            **hp,
        )

        # Log final training summary metrics
        if hasattr(results, "results_dict"):
            final_train = {}
            for k, v in results.results_dict.items():
                try:
                    final_train[f"train_final/{safe_key(k)}"] = float(v)
                except (TypeError, ValueError):
                    pass
            if final_train:
                mlflow.log_metrics(final_train)
                print("\nTrain final metrics:")
                for k, v in final_train.items():
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
                        test_metrics[f"test_final/{safe_key(k)}"] = float(v)
                    except (TypeError, ValueError):
                        pass

                p_val = float(test_results.results_dict.get(
                    "metrics/precision(B)", 0))
                r_val = float(test_results.results_dict.get(
                    "metrics/recall(B)", 0))
                test_metrics["test_final/f1"] = (
                    2 * p_val * r_val / (p_val + r_val + 1e-9))

                if test_metrics:
                    mlflow.log_metrics(test_metrics)
                    print("\nTest set metrics:")
                    for k, v in test_metrics.items():
                        print(f"  {k}: {v:.4f}")
        else:
            print(f"\nWARNING: test data.yaml not found at {test_yaml}, "
                  "skipping test eval")

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
