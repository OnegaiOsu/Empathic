"""Plot helpers: confusion matrices and model comparison bars."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


def plot_confusion(cm: np.ndarray, labels: Sequence[str], title: str, out_path: str) -> None:
    set_style()
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar_kws={"label": "row-normalised"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_comparison(metrics_per_model: Dict[str, Dict[str, float]], title: str, out_path: str) -> None:
    set_style()
    df = pd.DataFrame(metrics_per_model).T
    df = df.sort_values("macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["accuracy_mean"], width=0.4, label="accuracy", yerr=df.get("accuracy_std"), capsize=3)
    ax.bar(x + 0.2, df["macro_f1_mean"], width=0.4, label="macro F1", yerr=df.get("macro_f1_std"), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score (LOSO mean)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def plot_subject_f1(summary: pd.DataFrame, title: str, out_path: str) -> None:
    set_style()
    summary = summary.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.25 * len(summary))))
    sns.barplot(data=summary, x="macro_f1", y="subject", color="#4C72B0", ax=ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def ensure_results_dir(root: str, *parts: str) -> str:
    path = os.path.join(root, *parts)
    os.makedirs(path, exist_ok=True)
    return path
