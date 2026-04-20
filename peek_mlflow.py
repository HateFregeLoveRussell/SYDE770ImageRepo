"""Quick peek at MLflow experiments and runs."""
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

for exp in mlflow.search_experiments():
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    n_runs = len(runs)
    print(f"  Total runs: {n_runs}")
    if n_runs > 0:
        status_counts = runs["status"].value_counts().to_dict()
        print(f"  Status: {status_counts}")
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in runs.columns if c.startswith("params.")]
        print(f"  Metric columns ({len(metric_cols)}): {metric_cols[:8]}")
        print(f"  Param columns ({len(param_cols)}): {param_cols[:8]}")

        if "metrics.val_f1" in runs.columns:
            valid = runs[runs["metrics.val_f1"].notna()]
            print(f"  Runs with val_f1: {len(valid)}")
            if len(valid) > 0:
                best = valid.sort_values("metrics.val_f1", ascending=False).head(3)
                for _, row in best.iterrows():
                    print(f"    {row['tags.mlflow.runName']}: F1={row['metrics.val_f1']:.4f} mAP50={row.get('metrics.val_map50', 'N/A')}")
    print()
