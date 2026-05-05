"""Relaxed-protocol evaluation on WESAD to mirror common paper benchmarks.

We compare three protocols on the same v5 feature set + classical models:
  * LOSO              : leave-one-subject-out (subject-independent, our default)
  * Subject 5-fold    : 5-fold split on subjects (still subject-independent)
  * Window 10-fold    : stratified k-fold on windows -- WINDOWS FROM THE SAME
                        SUBJECT APPEAR IN BOTH TRAIN AND TEST (subject leakage).
                        This mirrors what most WESAD papers actually do.
  * Window 80/20      : single random stratified split (subject leakage).

We evaluate with cohen kappa, macro-f1 and accuracy on RandomForest /
LogisticRegression / XGBoost. Deep models are skipped to keep this fast
(deep gains were marginal anyway).
"""
from __future__ import annotations

import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.path.insert(0, "src")
from empathic.data.unified import build_bundles
from empathic.utils import align_feature_matrix
from empathic.models.classical import (
    make_random_forest, make_logistic_regression, make_xgboost,
)


_QUADRANT_TO_VA = {"HVHA": (1, 1), "HVLA": (1, 0), "LVHA": (0, 1), "LVLA": (0, 0)}


def evaluate(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def fit_predict(model, X_tr, y_tr, X_te):
    if model.supports_sample_weight:
        classes, counts = np.unique(y_tr, return_counts=True)
        w_per_class = {int(c): 1.0 / cnt for c, cnt in zip(classes, counts)}
        sw = np.array([w_per_class[int(y)] for y in y_tr])
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run(target, bundle):
    print(f"\n{'=' * 60}\nTARGET: {target}\n{'=' * 60}")
    samples = bundle.samples
    X = align_feature_matrix(samples, bundle.feature_cols)
    subj = np.asarray(bundle.subject_ids)
    q = bundle.quadrant_target
    qlabels = list(bundle.quadrant_labels)

    if target == "quadrant":
        y = q.copy()
    else:
        axis = 0 if target == "valence" else 1
        y = np.full(len(q), -1, dtype=np.int64)
        for i, name in enumerate(qlabels):
            bits = _QUADRANT_TO_VA.get(name)
            if bits is None:
                continue
            y[q == i] = bits[axis]

    valid = y >= 0
    X = X[valid]; y = y[valid]; subj = subj[valid]
    print(f"n_windows={len(y)}  n_subjects={len(np.unique(subj))}  classes={np.bincount(y)}")

    factories = [
        ("RandomForest", make_random_forest),
        ("LogisticRegression", make_logistic_regression),
        ("XGBoost", lambda: make_xgboost(use_gpu=False)),
    ]

    results = {}

    # 1) LOSO ---------------------------------------------------------------
    print("\n[Protocol] LOSO (subject-independent, strict)")
    for name, fac in factories:
        y_tr_all, y_pr_all = [], []
        for s in np.unique(subj):
            tr = subj != s; te = subj == s
            if len(np.unique(y[tr])) < 2: continue
            mdl = fac()
            yp = fit_predict(mdl, X[tr], y[tr], X[te])
            y_tr_all.append(y[te]); y_pr_all.append(yp)
        m = evaluate(np.concatenate(y_tr_all), np.concatenate(y_pr_all))
        print(f"  {name:20s}  acc={m['acc']:.3f}  f1={m['f1']:.3f}  kappa={m['kappa']:.3f}")
        results[("LOSO", name)] = m

    # 2) Subject 5-fold (still subject-independent, just fewer folds) ------
    print("\n[Protocol] Subject GroupKFold (k=5, subject-independent)")
    gkf = GroupKFold(n_splits=5)
    for name, fac in factories:
        y_tr_all, y_pr_all = [], []
        for tr, te in gkf.split(X, y, groups=subj):
            if len(np.unique(y[tr])) < 2: continue
            mdl = fac()
            yp = fit_predict(mdl, X[tr], y[tr], X[te])
            y_tr_all.append(y[te]); y_pr_all.append(yp)
        m = evaluate(np.concatenate(y_tr_all), np.concatenate(y_pr_all))
        print(f"  {name:20s}  acc={m['acc']:.3f}  f1={m['f1']:.3f}  kappa={m['kappa']:.3f}")
        results[("Subject5F", name)] = m

    # 3) Window 10-fold (subject leakage -- "papers' protocol") -----------
    print("\n[Protocol] Window StratifiedKFold (k=10, MIXED SUBJECTS - paper-style)")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for name, fac in factories:
        y_tr_all, y_pr_all = [], []
        for tr, te in skf.split(X, y):
            mdl = fac()
            yp = fit_predict(mdl, X[tr], y[tr], X[te])
            y_tr_all.append(y[te]); y_pr_all.append(yp)
        m = evaluate(np.concatenate(y_tr_all), np.concatenate(y_pr_all))
        print(f"  {name:20s}  acc={m['acc']:.3f}  f1={m['f1']:.3f}  kappa={m['kappa']:.3f}")
        results[("Window10F", name)] = m

    # 4) Single 80/20 random window split (subject leakage) ---------------
    print("\n[Protocol] Window 80/20 stratified split (MIXED SUBJECTS)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    for name, fac in factories:
        mdl = fac()
        yp = fit_predict(mdl, X_tr, y_tr, X_te)
        m = evaluate(y_te, yp)
        print(f"  {name:20s}  acc={m['acc']:.3f}  f1={m['f1']:.3f}  kappa={m['kappa']:.3f}")
        results[("Window80_20", name)] = m

    return results


if __name__ == "__main__":
    bundles = build_bundles(["wesad"], verbose=False)
    bundle = bundles["wesad"]
    all_results = {}
    for t in ["quadrant", "valence", "arousal"]:
        all_results[t] = run(t, bundle)

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY (kappa)")
    print("=" * 80)
    protos = ["LOSO", "Subject5F", "Window10F", "Window80_20"]
    models = ["RandomForest", "LogisticRegression", "XGBoost"]
    header = f"{'target':10s} {'model':20s} " + " ".join(f"{p:>12s}" for p in protos)
    print(header)
    print("-" * len(header))
    for t in ["quadrant", "valence", "arousal"]:
        for m in models:
            row = f"{t:10s} {m:20s} "
            for p in protos:
                k = all_results[t][(p, m)]["kappa"]
                row += f" {k:>12.3f}"
            print(row)
