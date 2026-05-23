"""Protocol B: per-subject 70/30 within-subject split.

For each subject independently:
  1. Take their c-session windows only.
  2. Stratified 70/30 train/test split on the target.
  3. Train a fresh model (classical or deep) on the 70%.
  4. Score on the 30%.
Aggregate macro-F1 / accuracy across subjects.

This is the within-subject upper bound that DEAP/AMIGOS report. No cross-
subject training, full personalisation. Subjects with single-class targets
are skipped.

Outputs to ``results/emotion/emowork/within_subject/<target>/summary.csv``
to sit alongside the LOSO and calibrated LOSO numbers.

Usage::

    .venv\\Scripts\\python.exe runs\\emowork\\train_within_subject.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import replace  # noqa: F401  (kept for ad-hoc cfg overrides)
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from empathic.config import TrainDefaults
from empathic.data.unified import build_bundles, DatasetBundle
from empathic.models.classical import (
    make_baseline,
    make_random_forest,
    make_logistic_regression,
    make_xgboost,
)
from empathic.training import _label_list, _select_target, _train_deep_fold
from empathic.utils import align_feature_matrix


RESULTS_ROOT = os.path.join("results", "emotion", "emowork", "within_subject")
TARGETS = ("stress", "arousal", "valence", "quadrant")
TEST_SIZE = 0.30
SEED = 42

# Deep models to evaluate per-subject. DANN/TSTCC are excluded:
#   * DANN needs >1 source domain — meaningless within a single subject.
#   * TSTCC needs many unlabeled windows for contrastive pretraining.
DEEP_ARCHS = ("conformer", "bilstm", "cnn1d", "tiny_tcn")
# Per-subject training has ~10 windows, 1 batch/epoch. 60 epochs is cheap
# and lets cosine LR + early-stop converge.
DEEP_CFG = TrainDefaults(seed=SEED, deep_epochs=60, early_stop_patience=12,
                         deep_batch_size=64)


def _factories(seed: int):
    fs = {
        "Baseline": lambda: make_baseline(seed=seed),
        "RandomForest": lambda: make_random_forest(seed=seed, n_estimators=200),
        "LogisticRegression": lambda: make_logistic_regression(seed=seed),
    }
    xgb = make_xgboost(seed=seed, use_gpu=False, n_estimators=200, max_depth=4)
    if xgb is not None:
        fs["XGBoost"] = lambda: make_xgboost(seed=seed, use_gpu=False,
                                             n_estimators=200, max_depth=4)
    return fs


def _eval_subject(model_factory, X_tr, y_tr, X_te, y_te) -> Dict[str, float]:
    model = model_factory()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_te, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_te, pred)),
        "n_test": int(len(y_te)),
    }


def _eval_deep(arch: str, num_classes: int,
               seq_tr: np.ndarray, y_tr: np.ndarray,
               seq_te: np.ndarray, y_te: np.ndarray,
               feat_tr: Optional[np.ndarray], feat_te: Optional[np.ndarray]
               ) -> Dict[str, float]:
    # Class weights inverse-frequency per local fold (matches run_experiment).
    classes, counts = np.unique(y_tr, return_counts=True)
    weights_full = np.ones(num_classes, dtype=np.float32)
    if len(classes) == num_classes:
        inv = counts.max() / np.maximum(counts, 1)
        for c, w in zip(classes, inv):
            weights_full[int(c)] = float(w)
    preds, _, _, _ = _train_deep_fold(
        seq_tr.astype(np.float32),
        y_tr.astype(np.int64),
        seq_te.astype(np.float32),
        y_te.astype(np.int64),
        num_classes=num_classes,
        cfg=DEEP_CFG,
        arch=arch,
        mixup_alpha=0.0,
        class_weights=weights_full,
        feat_train=(feat_tr.astype(np.float32) if feat_tr is not None else None),
        feat_val=(feat_te.astype(np.float32) if feat_te is not None else None),
        verbose=False,
    )
    return {
        "accuracy": float(accuracy_score(y_te, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, preds)),
        "macro_f1": float(f1_score(y_te, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_te, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, preds, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_te, preds)),
        "n_test": int(len(y_te)),
    }


def _safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write *df* to *path* via a temp file to avoid PermissionError on locked CSVs."""
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".csv.tmp")
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic on same filesystem
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_target(bundle: DatasetBundle, target: str) -> pd.DataFrame:
    print(f"\n{'#'*70}\n# WITHIN-SUBJECT TARGET = {target}\n{'#'*70}", flush=True)
    labels = _label_list(target, bundle)
    num_classes = len(labels)
    y_all = _select_target(bundle, target)
    X_all = align_feature_matrix(bundle.samples, bundle.feature_cols)
    seq_all = bundle.sequences
    subj_all = bundle.subject_ids

    valid = y_all >= 0
    X_all = X_all[valid]
    seq_all = seq_all[valid]
    y_all = y_all[valid]
    subj_all = subj_all[valid]

    is_binary = num_classes == 2
    factories = _factories(SEED)
    rows: List[Dict] = []

    for subject in np.unique(subj_all):
        mask = subj_all == subject
        Xs = X_all[mask]
        Ss = seq_all[mask]
        ys = y_all[mask]
        if len(np.unique(ys)) < 2:
            print(f"  subject={subject}  SKIP (single-class)", flush=True)
            continue
        # Need >=2 of each class to do a stratified 70/30 split.
        _, counts = np.unique(ys, return_counts=True)
        if counts.min() < 2:
            print(f"  subject={subject}  SKIP (min class count={counts.min()})", flush=True)
            continue

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                          random_state=SEED)
        train_idx, test_idx = next(splitter.split(Xs, ys))
        X_tr, y_tr = Xs[train_idx], ys[train_idx]
        X_te, y_te = Xs[test_idx], ys[test_idx]
        S_tr, S_te = Ss[train_idx], Ss[test_idx]

        # Single-class test fold (rare with stratified split, but possible if
        # min class count is 2 and the one test sample is unique).
        if is_binary and len(np.unique(y_te)) < 2:
            print(f"  subject={subject}  SKIP (single-class test split)", flush=True)
            continue

        # ---- Classical ---------------------------------------------------
        for name, factory in factories.items():
            try:
                m = _eval_subject(factory, X_tr, y_tr, X_te, y_te)
            except Exception as exc:
                print(f"  subject={subject} {name}  ERROR {exc}", flush=True)
                continue
            m["subject"] = str(subject)
            m["model"] = name
            rows.append(m)
            print(f"  subject={subject}  {name:<22s}  acc={m['accuracy']:.3f}  "
                  f"f1={m['macro_f1']:.3f}  n_te={m['n_test']}", flush=True)

        # ---- Deep (sequence + tabular fusion) ---------------------------
        for arch in DEEP_ARCHS:
            try:
                m = _eval_deep(
                    arch, num_classes,
                    S_tr, y_tr, S_te, y_te,
                    feat_tr=X_tr, feat_te=X_te,
                )
            except Exception as exc:
                print(f"  subject={subject} {arch}  ERROR {exc}", flush=True)
                continue
            m["subject"] = str(subject)
            m["model"] = f"{arch}_fusion"
            rows.append(m)
            print(f"  subject={subject}  {arch+'_fusion':<22s}  acc={m['accuracy']:.3f}  "
                  f"f1={m['macro_f1']:.3f}  n_te={m['n_test']}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary = df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_recall_mean=("macro_recall", "mean"),
        weighted_f1_mean=("weighted_f1", "mean"),
        cohen_kappa_mean=("cohen_kappa", "mean"),
        n_subjects=("subject", "nunique"),
    ).reset_index()

    out_dir = os.path.join(RESULTS_ROOT, target)
    os.makedirs(out_dir, exist_ok=True)
    _safe_write_csv(df, os.path.join(out_dir, "per_subject.csv"))
    _safe_write_csv(summary, os.path.join(out_dir, "summary.csv"))

    print(f"\n[within] {target} summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    return df


def main() -> None:
    print("[within] loading bundle ...", flush=True)
    t0 = time.time()
    bundles = build_bundles(["emowork"], quick=False, verbose=False)
    bundle = bundles["emowork"]
    print(
        f"[within] bundle ready in {time.time()-t0:.1f}s  "
        f"samples={len(bundle.samples)}  subjects={bundle.samples['subject_id'].nunique()}",
        flush=True,
    )

    for target in TARGETS:
        _run_target(bundle, target)

    print("\n[within] done.", flush=True)


if __name__ == "__main__":
    main()
