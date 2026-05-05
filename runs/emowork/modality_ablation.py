"""Per-modality LOSO ablation for EmoWork.

For each target (stress, arousal, valence) and each tabular modality group
(ECG, BVP, HR, EDA, TEMP, ACC, EEG, plus all-physio = ECG+BVP+HR+EDA+TEMP+ACC,
all = full feature set), train RandomForest under LOSO and report macro-F1,
balanced accuracy, kappa.

This identifies which sensor groups carry the signal for each target.

Outputs results/emotion/emowork/ablations/modality_ablation.csv and a heatmap
PNG.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
)

from empathic.data.unified import build_bundles
from empathic.utils import align_feature_matrix
from empathic.models.classical import make_random_forest


_QUADRANT_TO_VA = {"HVHA": (1, 1), "HVLA": (1, 0), "LVHA": (0, 1), "LVLA": (0, 0)}

MODALITIES = {
    "ECG": ("ECG_",),
    "BVP": ("BVP_",),
    "HR": ("HR_",),
    "EDA": ("EDA_",),
    "TEMP": ("TEMP_",),
    "ACC": ("ACC_",),
    "EEG": ("EEG_",),
    "physio": ("ECG_", "BVP_", "HR_", "EDA_", "TEMP_", "ACC_"),
    "physio+EEG (all)": ("ECG_", "BVP_", "HR_", "EDA_", "TEMP_", "ACC_", "EEG_"),
}

TARGETS = ("stress", "arousal", "valence")


def select_target(target: str, bundle):
    samples = bundle.samples
    X_all = align_feature_matrix(samples, bundle.feature_cols)
    subj = np.asarray(bundle.subject_ids)
    q = bundle.quadrant_target
    qlabels = list(bundle.quadrant_labels)
    if target == "stress":
        if bundle.stress is None:
            raise ValueError("no stress")
        y = bundle.stress.astype(np.int64)
    else:
        axis = 0 if target == "valence" else 1
        y = np.full(len(q), -1, dtype=np.int64)
        for i, name in enumerate(qlabels):
            bits = _QUADRANT_TO_VA.get(name)
            if bits is None:
                continue
            y[q == i] = bits[axis]
    valid = y >= 0
    return X_all[valid], y[valid], subj[valid]


def loso_rf(X, y, subj, *, n_classes=2):
    y_true_all, y_pred_all = [], []
    for s in np.unique(subj):
        tr = subj != s
        te = subj == s
        if len(np.unique(y[tr])) < 2:
            continue
        if n_classes == 2 and len(np.unique(y[te])) < 2:
            continue
        mdl = make_random_forest()
        classes, counts = np.unique(y[tr], return_counts=True)
        w = {int(c): 1.0 / cnt for c, cnt in zip(classes, counts)}
        sw = np.array([w[int(v)] for v in y[tr]])
        if mdl.supports_sample_weight:
            mdl.fit(X[tr], y[tr], sample_weight=sw)
        else:
            mdl.fit(X[tr], y[tr])
        yp = mdl.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(yp)
    if not y_true_all:
        return None
    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    return {
        "n_windows": int(len(yt)),
        "acc": float(accuracy_score(yt, yp)),
        "bal_acc": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "kappa": float(cohen_kappa_score(yt, yp)),
    }


def column_subset(feature_cols, prefixes):
    keep = [i for i, c in enumerate(feature_cols)
            if any(c.startswith(p) for p in prefixes)]
    return np.array(keep, dtype=np.int64)


def main():
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
    out_dir = os.path.join("results", "emotion", "emowork", "ablations")
    os.makedirs(out_dir, exist_ok=True)

    print("[ablation] loading bundle ...", flush=True)
    bundle = build_bundles(["emowork"], verbose=False)["emowork"]
    feature_cols = bundle.feature_cols
    print(f"[ablation] {len(feature_cols)} features  {len(np.unique(bundle.subject_ids))} subjects", flush=True)

    rows = []
    for target in TARGETS:
        X_all, y, subj = select_target(target, bundle)
        n_classes = len(np.unique(y))
        print(f"\n=== TARGET: {target}  n={len(y)}  classes={np.bincount(y)}  ===", flush=True)
        for mod_name, prefixes in MODALITIES.items():
            cols = column_subset(feature_cols, prefixes)
            if len(cols) == 0:
                continue
            metrics = loso_rf(X_all[:, cols], y, subj, n_classes=n_classes)
            if metrics is None:
                continue
            row = {"target": target, "modality": mod_name, "n_features": len(cols), **metrics}
            rows.append(row)
            print(f"  {mod_name:18s}  feats={len(cols):3d}  "
                  f"f1={metrics['macro_f1']:.3f}  bal_acc={metrics['bal_acc']:.3f}  "
                  f"kappa={metrics['kappa']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "modality_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[ablation] wrote {csv_path}")

    # Heatmap of macro-F1.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"

    pivot_f1 = df.pivot(index="modality", columns="target", values="macro_f1")
    pivot_f1 = pivot_f1.reindex(list(MODALITIES.keys()))[list(TARGETS)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="viridis",
                vmin=0.3, vmax=0.75, cbar_kws={"label": "macro-F1"}, ax=ax)
    ax.set_title("EmoWork - Per-modality LOSO macro-F1 (RandomForest)")
    fig.savefig(os.path.join(out_dir, "modality_ablation_macro_f1.png"))
    plt.close(fig)

    pivot_k = df.pivot(index="modality", columns="target", values="kappa")
    pivot_k = pivot_k.reindex(list(MODALITIES.keys()))[list(TARGETS)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_k, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                vmin=-0.05, vmax=0.5, cbar_kws={"label": "kappa"}, ax=ax)
    ax.set_title("EmoWork - Per-modality LOSO kappa (RandomForest)")
    fig.savefig(os.path.join(out_dir, "modality_ablation_kappa.png"))
    plt.close(fig)
    print("[ablation] wrote heatmaps")


if __name__ == "__main__":
    main()
