"""Deeper look at the Ultralytics MLflow experiment."""
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

runs = mlflow.search_runs(experiment_ids=["2"])
print(f"Total runs in optuna experiment: {len(runs)}")
print(f"Finished: {len(runs[runs['status'] == 'FINISHED'])}")

metric_cols = sorted([c for c in runs.columns if c.startswith("metrics.")])
print(f"\nAll metric columns ({len(metric_cols)}):")
for c in metric_cols:
    non_null = runs[c].notna().sum()
    print(f"  {c:45s}  ({non_null} runs have data)")

print("\n--- Top 5 runs by metrics.metrics/mAP50B ---")
col = "metrics.metrics/mAP50B"
if col in runs.columns:
    valid = runs[runs[col].notna()].sort_values(col, ascending=False).head(5)
    for _, row in valid.iterrows():
        name = row.get("tags.mlflow.runName", "?")
        map50 = row[col]
        prec = row.get("metrics.metrics/precisionB", 0)
        rec = row.get("metrics.metrics/recallB", 0)
        print(f"  {name:20s}  mAP50={map50:.4f}  P={prec:.4f}  R={rec:.4f}")
