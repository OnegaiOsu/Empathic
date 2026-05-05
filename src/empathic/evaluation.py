"""Metrics, classification reports and result persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    macro_recall: float
    weighted_f1: float
    cohen_kappa: float
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion: List[List[int]]
    labels: List[str]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> ClassificationMetrics:
    label_ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids).tolist()
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        per_class_f1={labels[i]: float(f[i]) for i in label_ids},
        per_class_support={labels[i]: int(s[i]) for i in label_ids},
        confusion=cm,
        labels=list(labels),
    )


def save_metrics(path: str, metrics: ClassificationMetrics, extra: Optional[Dict] = None) -> None:
    payload = asdict(metrics)
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def aggregate_fold_metrics(per_fold: List[ClassificationMetrics]) -> Dict[str, float]:
    if not per_fold:
        return {}
    return {
        "accuracy_mean": float(np.mean([m.accuracy for m in per_fold])),
        "accuracy_std": float(np.std([m.accuracy for m in per_fold])),
        "balanced_accuracy_mean": float(np.mean([m.balanced_accuracy for m in per_fold])),
        "macro_f1_mean": float(np.mean([m.macro_f1 for m in per_fold])),
        "macro_f1_std": float(np.std([m.macro_f1 for m in per_fold])),
        "macro_recall_mean": float(np.mean([m.macro_recall for m in per_fold])),
        "weighted_f1_mean": float(np.mean([m.weighted_f1 for m in per_fold])),
        "cohen_kappa_mean": float(np.mean([m.cohen_kappa for m in per_fold])),
    }


def build_summary_frame(per_fold: List[ClassificationMetrics], subjects: List[str]) -> pd.DataFrame:
    rows = []
    for subj, m in zip(subjects, per_fold):
        rows.append({
            "subject": subj,
            "accuracy": m.accuracy,
            "balanced_accuracy": m.balanced_accuracy,
            "macro_f1": m.macro_f1,
            "macro_recall": m.macro_recall,
            "weighted_f1": m.weighted_f1,
            "cohen_kappa": m.cohen_kappa,
        })
    return pd.DataFrame(rows)


def pool_session_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    session_key: np.ndarray,
) -> tuple:
    """Collapse per-window predictions into per-session predictions.

    For each unique session key we average the predicted class probabilities
    across all its windows and take the argmax. Because EmoSurv/WESAD labels
    are constant within a session this yields the per-session prediction that
    matches the actual granularity of the ground truth.

    Returns ``(y_true_session, y_pred_session, session_keys)``.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    keys = np.asarray(session_key)
    uniq = pd.unique(pd.Series(keys))
    y_t = np.empty(len(uniq), dtype=y_true.dtype)
    y_p = np.empty(len(uniq), dtype=y_true.dtype)
    for i, k in enumerate(uniq):
        mask = keys == k
        # Session label = modal label among its windows (constant in practice).
        vals, counts = np.unique(y_true[mask], return_counts=True)
        y_t[i] = vals[np.argmax(counts)]
        y_p[i] = int(np.argmax(probs[mask].mean(axis=0)))
    return y_t, y_p, uniq
