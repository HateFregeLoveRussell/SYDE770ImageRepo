# Windows 5080 Runbook — Optuna YOLOv8 Search

Quick setup guide to run the hyperparameter search on Windows with an RTX 5080.

## Prerequisites

- **Git** — https://git-scm.com/download/win
- **Python 3.11+** — https://www.python.org/downloads/ (check "Add to PATH")
- **NVIDIA drivers** — latest Game Ready or Studio drivers for the 5080
- **CUDA Toolkit 12.x** — https://developer.nvidia.com/cuda-downloads (match your driver)

## 1. Clone & checkout the branch

```powershell
git clone https://github.com/HateFregeLoveRussell/SYDE770-DL_Deployment.git
cd SYDE770-DL_Deployment\SYDE770ImageRepo
git checkout v3_yolo_optuna
```

## 2. Pull data with DVC

Images and index files are tracked by DVC, not git. You need them locally.

```powershell
pip install dvc dvc-s3
dvc pull
```

> If DVC credentials aren't set up on this machine, copy `liam_r2_credentials.json`
> from your Mac and configure `dvc remote modify` accordingly.

## 3. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install PyTorch with CUDA

Go to https://pytorch.org/get-started/locally/ and pick:
- OS: Windows
- Package: pip
- CUDA: 12.x (match your installed toolkit)

It will give you something like:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 5. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

## 6. Verify CUDA is visible

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 5080`

## 7. Prepare the train/val split

`data.yaml` contains absolute paths from the Mac. Regenerate it:

```powershell
python prepare_yolo_split.py
```

This recreates `data/derived/yolo/data.yaml` with Windows-correct absolute paths
and symlinks (or copies) the images into `images/train/` and `images/val/`.

> **Note on symlinks:** Windows requires Developer Mode enabled or an admin terminal
> for `os.symlink()`. If it fails, the script will need a small tweak to copy
> instead of symlink. Run as admin or enable Developer Mode in Settings > For Developers.

## 8. Start MLflow (optional but recommended)

In a separate terminal:

```powershell
.venv\Scripts\activate
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser.

## 9. Run the Optuna search

```powershell
python optuna_search.py --n-trials 50 --epochs-per-trial 20 --batch 32 --device cuda
```

With the 5080 (~16 GB VRAM), `--batch 32` should be fine. Each 20-epoch trial
will take roughly 3-5 minutes, so 50 trials ≈ 3-4 hours.

**If it crashes or you want more trials later:**

```powershell
python optuna_search.py --n-trials 75 --epochs-per-trial 20 --batch 32 --device cuda --resume
```

The `--resume` flag loads the existing study from `optuna_study.db` (SQLite)
and continues where it left off.

## 10. Check results

- **MLflow UI** at http://localhost:5000 — all trials are nested runs
- **Pareto frontier** saved to `pareto_frontier.csv` — the best F1 vs mAP50 tradeoffs
- **Optuna study** in `optuna_study.db` — can be loaded in a notebook for analysis

## 11. Retrain the winner at full epochs

Pick the best config from the Pareto frontier and run a full 50-epoch training:

```powershell
python train_yolov8_mlflow.py --model yolov8n.pt --epochs 50 --lr0 <best_lr0> --batch 32
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` returns `False` | Reinstall PyTorch with the correct CUDA version |
| `dvc pull` fails | Check credentials / remote config with `dvc remote list` |
| Symlink errors in `prepare_yolo_split.py` | Enable Developer Mode or run terminal as Admin |
| OOM on `--batch 32` | Drop to `--batch 16` |
| MLflow connection refused | Start `mlflow ui --port 5000` in a separate terminal first |
