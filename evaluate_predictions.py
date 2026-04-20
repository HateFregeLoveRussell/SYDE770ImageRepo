"""
evaluate_predictions.py

Print a comprehensive performance report comparing model predictions against
ground truth on the test set.

Usage:
  python evaluate_predictions.py
  python evaluate_predictions.py --predictions predictions/seed0
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate YOLO predictions against ground truth"
    )
    p.add_argument(
        "--predictions", default="predictions/seed0",
        help="Directory containing samples.parquet, detections.parquet, "
             "and ground_truth.parquet (default: predictions/seed0)",
    )
    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _print_section(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _classification_metrics(tp, fp, fn, tn, total):
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    acc = _safe_div(tp + tn, total)
    return prec, rec, f1, acc


def _print_breakdown(comp: pd.DataFrame, attr: str):
    """Print Tim Hortons classification accuracy grouped by a diagnostic attribute."""
    if attr not in comp.columns or comp[attr].isna().all():
        print(f"  {attr}: (no data)")
        return

    col = comp[attr].astype(str).where(comp[attr].notna(), other="(null)")
    groups = comp.groupby(col, dropna=False)
    rows = []
    for val, sub in sorted(groups, key=lambda x: -len(x[1])):
        display = str(val) if pd.notna(val) else "(null)"
        n = len(sub)
        gt_t = (sub["gt_tim"] > 0).sum()
        pred_t = (sub["pred_tim"] > 0).sum()
        correct = ((sub["gt_tim"] > 0) == (sub["pred_tim"] > 0)).sum()
        pct = 100 * _safe_div(correct, n)
        rows.append((display, n, gt_t, pred_t, correct, pct))

    label_w = max(len(r[0]) for r in rows)
    header = f"  {'Value':<{label_w}}  {'Imgs':>5}  {'GT Tim':>6}  {'Pred Tim':>8}  {'Correct':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for display, n, gt_t, pred_t, correct, pct in rows:
        print(f"  {display:<{label_w}}  {n:>5}  {gt_t:>6}  {pred_t:>8}  "
              f"{correct:>4}/{n:<4} ({pct:>5.1f}%)")


def main():
    args = parse_args()
    pred_dir = pathlib.Path(args.predictions)

    samples = pd.read_parquet(pred_dir / "samples.parquet")
    gt = pd.read_parquet(pred_dir / "ground_truth.parquet")
    pred = pd.read_parquet(pred_dir / "detections.parquet")

    # ── Overall detection counts ──────────────────────────────
    _print_section("OVERALL DETECTION COUNTS")
    print(f"  Ground truth detections : {len(gt)}")
    for label, cnt in gt["label"].value_counts().items():
        print(f"    {label}: {cnt}")
    print(f"  Prediction detections   : {len(pred)}")
    for label, cnt in pred["label"].value_counts().items():
        print(f"    {label}: {cnt}")
    print(f"  Test images             : {len(samples)}")

    # ── Per-image detection count comparison ───────────────────
    _print_section("PER-IMAGE DETECTION COUNT COMPARISON")

    gt_per = gt.groupby("sample_id").agg(
        gt_count=("label", "size"), gt_tim=("is_tim", "sum"),
    ).reset_index()
    pred_per = pred.groupby("sample_id").agg(
        pred_count=("label", "size"), pred_tim=("is_tim", "sum"),
        mean_conf=("confidence", "mean"),
    ).reset_index()

    comp = samples[["sample_id"]].merge(gt_per, on="sample_id", how="left")
    comp = comp.merge(pred_per, on="sample_id", how="left")
    for c in ["gt_count", "pred_count", "gt_tim", "pred_tim"]:
        comp[c] = comp[c].fillna(0).astype(int)

    exact = (comp["gt_count"] == comp["pred_count"]).sum()
    over = (comp["pred_count"] > comp["gt_count"]).sum()
    under = (comp["pred_count"] < comp["gt_count"]).sum()
    print(f"  Exact count match : {exact}/{len(comp)} images ({100*exact/len(comp):.1f}%)")
    print(f"  Over-detected     : {over} images (model found extra boxes)")
    print(f"  Under-detected    : {under} images (model missed boxes)")

    # ── Tim Hortons image-level classification ─────────────────
    _print_section("TIM HORTONS CLASSIFICATION (per image)")

    gt_has = comp["gt_tim"] > 0
    pr_has = comp["pred_tim"] > 0
    tp = int((gt_has & pr_has).sum())
    fp = int((~gt_has & pr_has).sum())
    fn = int((gt_has & ~pr_has).sum())
    tn = int((~gt_has & ~pr_has).sum())
    prec, rec, f1, acc = _classification_metrics(tp, fp, fn, tn, len(comp))

    print(f"  TP = {tp:>4}   (GT has Tim, model detected Tim)")
    print(f"  FP = {fp:>4}   (GT no Tim, model predicted Tim)")
    print(f"  FN = {fn:>4}   (GT has Tim, model missed Tim)")
    print(f"  TN = {tn:>4}   (GT no Tim, model correctly absent)")
    print()
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # ── Breakdowns by diagnostic attributes ────────────────────
    diag_attrs = [
        "brand", "type", "background", "orientation", "deform",
        "blur", "occluded", "count", "lid", "sleeve",
    ]
    comp_with_diag = comp.merge(
        samples[["sample_id"] + [a for a in diag_attrs if a in samples.columns]],
        on="sample_id", how="left",
    )

    _print_section("BREAKDOWN BY DIAGNOSTIC ATTRIBUTES")
    for attr in diag_attrs:
        print(f"\n  --- {attr.upper()} ---")
        _print_breakdown(comp_with_diag, attr)

    # ── Confidence statistics ──────────────────────────────────
    _print_section("CONFIDENCE STATISTICS")
    print(f"  Mean (all predictions) : {pred['confidence'].mean():.4f}")
    print(f"  Std                    : {pred['confidence'].std():.4f}")
    print(f"  Min                    : {pred['confidence'].min():.4f}")
    print(f"  Max                    : {pred['confidence'].max():.4f}")
    print()
    for label in sorted(pred["label"].unique()):
        subset = pred[pred["label"] == label]
        print(f"  {label}:")
        print(f"    Mean confidence : {subset['confidence'].mean():.4f}")
        print(f"    Count           : {len(subset)}")

    # ── Confidence distribution buckets ────────────────────────
    print()
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels_b = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    pred["conf_bucket"] = pd.cut(pred["confidence"], bins=bins, labels=labels_b, include_lowest=True)
    print("  Confidence distribution:")
    for bucket, cnt in pred["conf_bucket"].value_counts().sort_index().items():
        bar = "#" * int(cnt / len(pred) * 40)
        print(f"    {bucket}: {cnt:>5}  {bar}")

    print()
    print("=" * 60)
    print("  Report complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
