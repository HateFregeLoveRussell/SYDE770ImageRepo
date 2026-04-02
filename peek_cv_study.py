"""Quick read-only peek at the CV Optuna study progress."""
import optuna, pathlib

db = pathlib.Path("optuna_study_cv5.db")
study = optuna.load_study(study_name="yolov8_nsga2_cv5", storage=f"sqlite:///{db}")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]

print(f"Completed: {len(completed)}  |  Running: {len(running)}  |  Total: {len(study.trials)}")
print()

completed.sort(key=lambda t: t.values[0], reverse=True)
print("=== Top 10 by avg F1 ===")
header = f"{'Trial':>6}  {'avg_F1':>8}  {'avg_mAP50':>10}  {'model':>12}  {'imgsz':>5}  {'optim':>6}  {'lr0':>10}  {'dropout':>7}"
print(header)
print("-" * len(header))
for t in completed[:10]:
    p = t.params
    print(f"{t.number:>6}  {t.values[0]:>8.4f}  {t.values[1]:>10.4f}  {p['model']:>12}  {p['imgsz']:>5}  {p['optimizer']:>6}  {p['lr0']:>10.6f}  {p['dropout']:>7.3f}")

print()
completed.sort(key=lambda t: t.values[1], reverse=True)
print("=== Top 10 by avg mAP50 ===")
print(header)
print("-" * len(header))
for t in completed[:10]:
    p = t.params
    print(f"{t.number:>6}  {t.values[0]:>8.4f}  {t.values[1]:>10.4f}  {p['model']:>12}  {p['imgsz']:>5}  {p['optimizer']:>6}  {p['lr0']:>10.6f}  {p['dropout']:>7.3f}")

print()
pareto = study.best_trials
print(f"=== Pareto frontier ({len(pareto)} trials) ===")
for t in sorted(pareto, key=lambda t: t.values[0], reverse=True):
    print(f"  Trial {t.number}: avg_F1={t.values[0]:.4f}, avg_mAP50={t.values[1]:.4f}")

print()
f1s = [t.values[0] for t in completed]
maps = [t.values[1] for t in completed]
print(f"F1  range: {min(f1s):.4f} - {max(f1s):.4f}  (mean={sum(f1s)/len(f1s):.4f})")
print(f"mAP range: {min(maps):.4f} - {max(maps):.4f}  (mean={sum(maps)/len(maps):.4f})")

import statistics as st

by_f1 = sorted(completed, key=lambda t: t.values[0], reverse=True)

def avg_top(k, trials):
    s = trials[:k]
    return sum(t.values[0] for t in s) / k, sum(t.values[1] for t in s) / k

print()
print("=== Averages ===")
print(f"All {len(completed)} trials: mean F1={st.mean(f1s):.4f}, mean mAP50={st.mean(maps):.4f}")
for k in (3, 5, 10):
    if len(by_f1) >= k:
        af, am = avg_top(k, by_f1)
        print(f"Mean of top-{k} by F1:  F1={af:.4f}, mAP50={am:.4f}")
by_map = sorted(completed, key=lambda t: t.values[1], reverse=True)
for k in (3, 5, 10):
    if len(by_map) >= k:
        af, am = avg_top(k, by_map)
        print(f"Mean of top-{k} by mAP50: F1={af:.4f}, mAP50={am:.4f}")
