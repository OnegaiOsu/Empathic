"""
Compare results across all trained models.

Reads per-model metrics.csv files from results/<model>/ and produces:
  - A combined comparison table (CSV)
  - An accuracy bar chart with error bars
  - A combined ROC curve overlay

Run this AFTER all 5 individual training scripts have completed.

Usage:
    python compare_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation import RESULTS_DIR, _ensure_dir

MODEL_NAMES = [
    "random_forest",
    "xgboost",
    "svm",
    "logistic_regression",
    "mlp",
    "enhanced_ensemble",
    "transformer",
]

COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")


def load_model_metrics(model_name: str) -> pd.DataFrame | None:
    """Load metrics.csv for a model, return per-fold rows (exclude summary)."""
    path = os.path.join(RESULTS_DIR, model_name, "metrics.csv")
    if not os.path.exists(path):
        print(f"[compare] WARNING: {path} not found — skipping {model_name}")
        return None
    df = pd.read_csv(path)
    # Drop the summary row (MEAN±STD)
    df = df[df["fold_participant"] != "MEAN±STD"].copy()
    # Convert metric columns to numeric
    metric_cols = [c for c in df.columns if c != "fold_participant"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    _ensure_dir(COMPARISON_DIR)

    # ------------------------------------------------------------------
    # 1. Load all model metrics
    # ------------------------------------------------------------------
    all_summaries = []
    model_data = {}

    for model_name in MODEL_NAMES:
        df = load_model_metrics(model_name)
        if df is None:
            continue
        model_data[model_name] = df

        metric_cols = [c for c in df.columns if c != "fold_participant"]
        summary = {"model": model_name}
        for col in metric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            summary[f"{col}_mean"] = mean_val
            summary[f"{col}_std"] = std_val
            summary[col] = f"{mean_val:.4f} ± {std_val:.4f}"
        all_summaries.append(summary)

    if not all_summaries:
        print("[compare] No model results found. Run training scripts first.")
        return

    summary_df = pd.DataFrame(all_summaries)

    # Save comparison CSV (human-readable columns)
    display_cols = ["model"] + [
        c for c in summary_df.columns
        if not c.endswith("_mean") and not c.endswith("_std") and c != "model"
    ]
    comparison_path = os.path.join(COMPARISON_DIR, "model_comparison.csv")
    summary_df[display_cols].to_csv(comparison_path, index=False)
    print(f"[compare] Comparison table saved to {comparison_path}")

    # Print to console
    print(f"\n{'='*80}")
    print("  Model Comparison — SWELL-KW Stress Classification (LOSO)")
    print(f"{'='*80}")
    print(summary_df[display_cols].to_string(index=False))
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # 2. Accuracy bar chart
    # ------------------------------------------------------------------
    models_present = [s["model"] for s in all_summaries]
    acc_means = [s["accuracy_mean"] for s in all_summaries]
    acc_stds = [s["accuracy_std"] for s in all_summaries]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(models_present))
    bars = ax.bar(x_pos, acc_means, yerr=acc_stds, capsize=5,
                  color=sns.color_palette("viridis", len(models_present)),
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("_", "\n") for m in models_present], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Comparison — Accuracy (LOSO, mean ± std)", fontsize=13)
    ax.set_ylim(0, 1.05)
    # Add value labels on bars
    for bar, mean, std in zip(bars, acc_means, acc_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    bar_path = os.path.join(COMPARISON_DIR, "accuracy_bar_chart.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Accuracy bar chart saved to {bar_path}")

    # ------------------------------------------------------------------
    # 3. F1 macro bar chart
    # ------------------------------------------------------------------
    f1_means = [s["f1_macro_mean"] for s in all_summaries]
    f1_stds = [s["f1_macro_std"] for s in all_summaries]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5,
                  color=sns.color_palette("mako", len(models_present)),
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("_", "\n") for m in models_present], fontsize=10)
    ax.set_ylabel("F1 Score (macro)", fontsize=12)
    ax.set_title("Model Comparison — F1 Macro (LOSO, mean ± std)", fontsize=13)
    ax.set_ylim(0, 1.05)
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    f1_path = os.path.join(COMPARISON_DIR, "f1_macro_bar_chart.png")
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)
    print(f"[compare] F1 macro bar chart saved to {f1_path}")

    # ------------------------------------------------------------------
    # 4. Combined metrics table (all metrics side by side)
    # ------------------------------------------------------------------
    metrics_to_plot = ["accuracy", "f1_macro", "auc_roc", "precision", "recall"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
    colors = sns.color_palette("Set2", len(models_present))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = [s.get(f"{metric}_mean", 0) for s in all_summaries]
        stds = [s.get(f"{metric}_std", 0) for s in all_summaries]
        ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m[:6] for m in models_present], fontsize=8, rotation=30)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(0, 1.05)

    fig.suptitle("SWELL-KW Stress Classification — All Metrics (LOSO)", fontsize=14, y=1.02)
    fig.tight_layout()
    all_metrics_path = os.path.join(COMPARISON_DIR, "all_metrics_comparison.png")
    fig.savefig(all_metrics_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] All metrics comparison saved to {all_metrics_path}")

    print(f"\n[compare] All comparison outputs saved to {COMPARISON_DIR}/")


if __name__ == "__main__":
    main()
