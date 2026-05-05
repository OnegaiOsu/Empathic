"""
Evaluation utilities for SWELL-KW stress classification.

Computes per-fold and aggregated metrics, generates plots, and saves results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

# ---------------------------------------------------------------------------
# Results directory helper
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_BASE_DIR, "results")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_model_results_dir(model_name: str) -> str:
    """Return and create results/<model_name>/ directory."""
    d = os.path.join(RESULTS_DIR, model_name)
    _ensure_dir(d)
    return d


# ---------------------------------------------------------------------------
# Per-fold metric computation
# ---------------------------------------------------------------------------
def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute classification metrics for a single LOSO fold.

    Parameters
    ----------
    y_true  : true labels (0/1)
    y_pred  : predicted labels (0/1)
    y_proba : predicted probability of class 1 (for AUC-ROC). Can be None.

    Returns
    -------
    dict with keys: accuracy, f1_stressed, f1_not_stressed, f1_macro,
                    precision, recall, auc_roc
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_stressed": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_not_stressed": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["auc_roc"] = np.nan
    return metrics


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------
def aggregate_fold_metrics(fold_results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-fold metrics into a summary DataFrame with mean ± std.

    Parameters
    ----------
    fold_results : list of dicts (one per LOSO fold), each from compute_fold_metrics()

    Returns
    -------
    pd.DataFrame with columns: metric, mean, std
    """
    df = pd.DataFrame(fold_results)
    summary = pd.DataFrame(
        {
            "metric": df.columns,
            "mean": df.mean().values,
            "std": df.std().values,
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_summary(fold_results: list[dict], model_name: str):
    """Print a formatted summary table to console."""
    summary = aggregate_fold_metrics(fold_results)
    print(f"\n{'='*60}")
    print(f"  {model_name} — LOSO Cross-Validation Results ({len(fold_results)} folds)")
    print(f"{'='*60}")
    for _, row in summary.iterrows():
        print(f"  {row['metric']:<20s}  {row['mean']:.4f}  ± {row['std']:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Save metrics to CSV
# ---------------------------------------------------------------------------
def save_metrics(
    fold_results: list[dict],
    fold_ids: list[str],
    model_name: str,
):
    """
    Save per-fold metrics + summary row to results/<model_name>/metrics.csv.

    Parameters
    ----------
    fold_results : list of metric dicts (one per fold)
    fold_ids     : list of participant IDs (one per fold)
    model_name   : e.g. "random_forest"
    """
    out_dir = get_model_results_dir(model_name)
    df = pd.DataFrame(fold_results)
    df.insert(0, "fold_participant", fold_ids)

    # Append summary row
    summary_row = {"fold_participant": "MEAN±STD"}
    for col in df.columns[1:]:
        summary_row[col] = f"{df[col].mean():.4f}±{df[col].std():.4f}"
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(path, index=False)
    print(f"[evaluation] Metrics saved to {path}")


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    model_name: str,
):
    """Save a confusion matrix heatmap for the aggregated predictions."""
    out_dir = get_model_results_dir(model_name)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Stressed", "Stressed"],
        yticklabels=["Not Stressed", "Stressed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix (LOSO)")
    fig.tight_layout()

    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Confusion matrix saved to {path}")


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------
def save_roc_curve(
    y_true_all: np.ndarray,
    y_proba_all: np.ndarray,
    model_name: str,
):
    """Save an ROC curve plot for the aggregated predictions."""
    out_dir = get_model_results_dir(model_name)

    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    auc = roc_auc_score(y_true_all, y_proba_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve (LOSO)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] ROC curve saved to {path}")


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------
def save_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str,
):
    """Save a horizontal bar chart of feature importances."""
    out_dir = get_model_results_dir(model_name)

    # Sort by importance
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Feature Importance")
    fig.tight_layout()

    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Feature importance saved to {path}")
