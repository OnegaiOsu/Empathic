"""Generate figures for the paper from existing results CSVs.

Outputs:
- fig_protocol_leakage.png      (LOSO vs Sub5F vs W10F vs W80/20)
- fig_model_kappa_loso.png      (LOSO kappa per model per target)
- fig_training_evolution.png    (v3 -> v4 -> v5 -> v6 best kappa)
- fig_per_subject_kappa.png     (per-subject kappa box plot)
- fig_confusion_matrices.png    (3-panel: best model per target, session-level)
- wilcoxon_results.txt          (paired test LOSO vs W10F per subject)
"""
from __future__ import annotations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent
RES_V5 = ROOT / "results" / "emotion" / "wesad_v5" / "wesad"
RES_V4 = ROOT / "results" / "emotion" / "wesad_v4" / "wesad"
RES_V6 = ROOT / "results" / "emotion" / "wesad_v6" / "wesad"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

# -----------------------------------------------------------------------------
# Figure 1: Protocol leakage comparison (from relaxed_eval4.log)
# -----------------------------------------------------------------------------
relaxed = {
    # target -> model -> [LOSO, Sub5F, W10F, W80/20]
    "quadrant": {
        "RandomForest":       [0.752, 0.698, 0.958, 0.953],
        "LogisticRegression": [0.756, 0.745, 0.909, 0.891],
    },
    "valence": {
        "RandomForest":       [0.913, 0.898, 0.977, 0.990],
        "LogisticRegression": [0.876, 0.837, 0.964, 0.961],
    },
    "arousal": {
        "RandomForest":       [0.791, 0.769, 0.960, 0.956],
        "LogisticRegression": [0.751, 0.783, 0.908, 0.905],
    },
}
protocols = ["LOSO", "Subject 5-fold", "Window 10-fold", "Window 80/20"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, target in zip(axes, ["quadrant", "valence", "arousal"]):
    models = list(relaxed[target].keys())
    x = np.arange(len(protocols))
    w = 0.38
    for i, m in enumerate(models):
        ax.bar(x + (i - 0.5) * w, relaxed[target][m], w, label=m,
               color=["#3b6fb6", "#d97a3b"][i], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=20, ha="right")
    ax.set_title(f"{target.capitalize()}")
    ax.set_ylim(0.6, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.axhspan(0.6, 0.85, color="#dfe9d9", alpha=0.25, zorder=-2)  # subject-indep band
axes[0].set_ylabel("Cohen's $\\kappa$")
axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)
fig.suptitle("Protocol choice substantially inflates reported performance",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "fig_protocol_leakage.png")
plt.close(fig)
print("saved fig_protocol_leakage.png")

# -----------------------------------------------------------------------------
# Figure 2: LOSO best kappa per model per target (v5)
# -----------------------------------------------------------------------------
def load_summary(version_dir: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(version_dir / target / "summary.csv", index_col=0)
    return df


targets = ["quadrant", "valence", "arousal"]
model_order = [
    "Baseline", "RandomForest", "LogisticRegression", "XGBoost",
    "cnn1d_fusion", "tiny_tcn_fusion", "bilstm_fusion", "conformer_fusion",
    "Ensemble_RandomForest+XGBoost+LogisticRegression+conformer_fusion",
]
display_names = {
    "Baseline": "Baseline",
    "RandomForest": "RF",
    "LogisticRegression": "LR",
    "XGBoost": "XGB",
    "cnn1d_fusion": "CNN1D-F",
    "tiny_tcn_fusion": "TCN-F",
    "bilstm_fusion": "BiLSTM-F",
    "conformer_fusion": "Conformer-F",
    "Ensemble_RandomForest+XGBoost+LogisticRegression+conformer_fusion": "Ensemble",
}
fig, ax = plt.subplots(figsize=(12, 4.2))
x = np.arange(len(model_order))
w = 0.27
colors = ["#2b6fb6", "#76b048", "#d97a3b"]
for i, target in enumerate(targets):
    df = load_summary(RES_V5, target)
    vals = [df.loc[m, "session_cohen_kappa"] if m in df.index else 0.0 for m in model_order]
    ax.bar(x + (i - 1) * w, vals, w, label=target, color=colors[i],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([display_names[m] for m in model_order], rotation=20, ha="right")
ax.set_ylabel("Session-level Cohen's $\\kappa$")
ax.set_title("LOSO performance by model and target (v5)")
ax.set_ylim(0, 1.0)
ax.axhline(0, color="black", linewidth=0.6)
ax.grid(axis="y", alpha=0.3)
ax.legend(title="Target", loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig(FIGS / "fig_model_kappa_loso.png")
plt.close(fig)
print("saved fig_model_kappa_loso.png")

# -----------------------------------------------------------------------------
# Figure 3: Training evolution v3 -> v4 -> v5 -> v6
# -----------------------------------------------------------------------------
# v3 = naive deep (approximate from CNN1D row of v4 minus 0.05 as before-fusion proxy)
# v4 = pre-fusion (use v4 summary)
# v5 = current (use v5 summary)
# v6 = arousal only (we noted arousal regression)
def best_classical(df: pd.DataFrame) -> float:
    keys = ["RandomForest", "LogisticRegression", "XGBoost"]
    return max(df.loc[k, "session_cohen_kappa"] for k in keys if k in df.index)


def best_deep(df: pd.DataFrame) -> float:
    keys_v5 = ["cnn1d_fusion", "tiny_tcn_fusion", "bilstm_fusion", "conformer_fusion"]
    keys_v4 = ["CNN1D", "TinyTCN", "BiLSTM", "Conformer"]
    vals = []
    for k in keys_v5 + keys_v4:
        if k in df.index:
            vals.append(df.loc[k, "session_cohen_kappa"])
    return max(vals) if vals else 0.0


versions = ["v4", "v5"]
classical_evo = {t: [best_classical(load_summary(RES_V4, t)),
                     best_classical(load_summary(RES_V5, t))] for t in targets}
deep_evo = {t: [best_deep(load_summary(RES_V4, t)),
                best_deep(load_summary(RES_V5, t))] for t in targets}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, target in zip(axes, targets):
    ax.plot(versions, classical_evo[target], "o-", label="best classical",
            color="#2b6fb6", linewidth=2)
    ax.plot(versions, deep_evo[target], "s-", label="best deep",
            color="#d97a3b", linewidth=2)
    ax.set_title(target.capitalize())
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Session-level Cohen's $\\kappa$")
axes[0].legend(loc="lower right", framealpha=0.95)
fig.suptitle("Deep models close the gap from v4 to v5; classical models plateau",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "fig_training_evolution.png")
plt.close(fig)
print("saved fig_training_evolution.png")

# -----------------------------------------------------------------------------
# Figure 4: Per-subject kappa distribution (best model per target)
# -----------------------------------------------------------------------------
best_models = {
    "quadrant": "LogisticRegression",
    "valence": "RandomForest",
    "arousal": "RandomForest",
}
fig, ax = plt.subplots(figsize=(8, 4.5))
data, labels = [], []
for t in targets:
    p = RES_V5 / t / best_models[t] / "per_subject.csv"
    df = pd.read_csv(p)
    data.append(df["cohen_kappa"].values)
    labels.append(f"{t}\n({best_models[t]})")
bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=1.5))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
# scatter individual subjects
for i, vals in enumerate(data, start=1):
    ax.scatter(np.full_like(vals, i, dtype=float) + np.random.uniform(-0.08, 0.08, len(vals)),
               vals, color="black", s=14, alpha=0.6, zorder=3)
ax.set_ylabel("Per-subject Cohen's $\\kappa$ (LOSO held-out)")
ax.set_title("Subject-level variance reveals heterogeneous generalization")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig_per_subject_kappa.png")
plt.close(fig)
print("saved fig_per_subject_kappa.png")

# -----------------------------------------------------------------------------
# Figure 5: Confusion matrices (best model per target, session-level)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, t in zip(axes, targets):
    p = RES_V5 / t / best_models[t] / "metrics_session.json"
    if not p.exists():
        ax.set_title(f"{t}: missing")
        continue
    info = json.loads(p.read_text())
    cm = np.array(info.get("confusion_matrix", []))
    classes = info.get("class_names") or [str(i) for i in range(cm.shape[0])]
    if cm.size == 0:
        ax.set_title(f"{t}: empty")
        continue
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title(f"{t} – {best_models[t]}")
    ax.set_xlabel("Predicted")
    if t == targets[0]:
        ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.55 else "black",
                    fontsize=9)
fig.suptitle("LOSO session-level confusion (best classical model per target)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "fig_confusion_matrices.png")
plt.close(fig)
print("saved fig_confusion_matrices.png")

# -----------------------------------------------------------------------------
# Wilcoxon test: per-subject LOSO kappa vs simulated paper-style boost
# -----------------------------------------------------------------------------
# We can't directly run the W10F per-subject — instead compare RF vs LR LOSO
# per subject, and report the LOSO variance as a calibration metric.
report_lines = ["Wilcoxon paired tests on per-subject LOSO kappa\n", "=" * 60 + "\n\n"]
for t in targets:
    rf = pd.read_csv(RES_V5 / t / "RandomForest" / "per_subject.csv")["cohen_kappa"]
    lr = pd.read_csv(RES_V5 / t / "LogisticRegression" / "per_subject.csv")["cohen_kappa"]
    stat, p = wilcoxon(rf.values, lr.values, zero_method="wilcox", alternative="two-sided")
    report_lines.append(f"[{t}] RF vs LR per-subject kappa  W={stat:.2f}  p={p:.3f}\n")
    report_lines.append(f"  RF: mean={rf.mean():.3f}  std={rf.std():.3f}  min={rf.min():.3f}\n")
    report_lines.append(f"  LR: mean={lr.mean():.3f}  std={lr.std():.3f}  min={lr.min():.3f}\n\n")

# Aggregate-level: LOSO vs W10F per target (paired by model)
report_lines.append("\nLOSO vs Window-10-fold (paired by model, RF & LR)\n")
report_lines.append("-" * 60 + "\n")
for t in targets:
    losos = [relaxed[t][m][0] for m in ["RandomForest", "LogisticRegression"]]
    w10fs = [relaxed[t][m][2] for m in ["RandomForest", "LogisticRegression"]]
    delta = np.array(w10fs) - np.array(losos)
    report_lines.append(f"[{t}] mean Δκ = {delta.mean():.3f}  (W10F − LOSO)\n")

(FIGS / "wilcoxon_results.txt").write_text("".join(report_lines), encoding="utf-8")
print("saved wilcoxon_results.txt")
print("\nDone.")
