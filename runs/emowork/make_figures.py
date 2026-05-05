"""Cross-target comparison figures for the EmoWork sweep.

Reads each target's summary.csv from results/emotion/emowork/emowork/<target>/
and produces:
  - cross_target_macro_f1.png    (grouped bars: target x model)
  - cross_target_kappa.png       (grouped bars: window-level Cohen kappa)
  - cross_target_session_f1.png  (session-pooled macro-F1)
  - heatmap_macro_f1.png         (model x target heatmap)
  - best_per_target.png          (single bar per target with best model)

Outputs go to results/emotion/emowork/figures/.
"""
from __future__ import annotations

import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = os.path.join("results", "emotion", "emowork", "emowork")
OUT = os.path.join("results", "emotion", "emowork", "figures")
os.makedirs(OUT, exist_ok=True)

TARGETS = ["valence", "arousal", "stress", "quadrant"]
MODEL_ORDER = [
    "Baseline",
    "RandomForest", "LogisticRegression", "XGBoost",
    "conformer_fusion", "tiny_tcn_fusion", "bilstm_fusion",
    "cnn1d_fusion", "multistream_fusion",
    "DANN_Conformer", "TSTCC",
]


def load_target(target: str) -> pd.DataFrame:
    path = os.path.join(ROOT, target, "summary.csv")
    df = pd.read_csv(path, index_col=0)
    # Drop the (long-named) ensemble row from the per-model panels - keep separately.
    df["target"] = target
    df["model"] = df.index
    return df


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"

    frames = [load_target(t) for t in TARGETS]
    all_df = pd.concat(frames, ignore_index=True)

    # Strip the long ensemble name to just "Ensemble" for plotting.
    all_df["model_short"] = all_df["model"].apply(
        lambda s: "Ensemble" if s.startswith("Ensemble_") else s
    )

    plot_models = MODEL_ORDER + ["Ensemble"]
    df_plot = all_df[all_df["model_short"].isin(plot_models)].copy()
    df_plot["model_short"] = pd.Categorical(df_plot["model_short"], categories=plot_models, ordered=True)
    df_plot["target"] = pd.Categorical(df_plot["target"], categories=TARGETS, ordered=True)

    # 1. Window-level macro-F1 grouped bars.
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=df_plot, x="model_short", y="macro_f1_mean", hue="target",
        order=plot_models, hue_order=TARGETS, ax=ax,
    )
    # Add error bars from std.
    for i, tgt in enumerate(TARGETS):
        sub = df_plot[df_plot["target"] == tgt].set_index("model_short").reindex(plot_models)
        x_centers = np.arange(len(plot_models)) + (i - (len(TARGETS) - 1) / 2) * (0.8 / len(TARGETS))
        ax.errorbar(x_centers, sub["macro_f1_mean"], yerr=sub["macro_f1_std"],
                    fmt="none", ecolor="black", alpha=0.4, capsize=2)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("LOSO macro-F1 (mean ± std)")
    ax.set_xlabel("")
    ax.set_title("EmoWork - Window-level macro-F1 by target and model (LOSO)")
    ax.tick_params(axis="x", rotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(title="target", loc="upper right")
    fig.savefig(os.path.join(OUT, "cross_target_macro_f1.png"))
    plt.close(fig)

    # 2. Window-level Cohen kappa grouped bars.
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=df_plot, x="model_short", y="cohen_kappa_mean", hue="target",
        order=plot_models, hue_order=TARGETS, ax=ax,
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("LOSO Cohen's kappa (mean)")
    ax.set_xlabel("")
    ax.set_title("EmoWork - Window-level kappa by target and model (LOSO)")
    ax.tick_params(axis="x", rotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(title="target", loc="upper right")
    fig.savefig(os.path.join(OUT, "cross_target_kappa.png"))
    plt.close(fig)

    # 3. Session-pooled macro-F1.
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=df_plot, x="model_short", y="session_macro_f1", hue="target",
        order=plot_models, hue_order=TARGETS, ax=ax,
    )
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Session-pooled macro-F1")
    ax.set_xlabel("")
    ax.set_title("EmoWork - Session-pooled macro-F1 by target and model (LOSO)")
    ax.tick_params(axis="x", rotation=25)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(title="target", loc="upper right")
    fig.savefig(os.path.join(OUT, "cross_target_session_f1.png"))
    plt.close(fig)

    # 4. Heatmap (model x target) of macro-F1.
    pivot = df_plot.pivot_table(index="model_short", columns="target",
                                values="macro_f1_mean", aggfunc="first",
                                observed=True)
    pivot = pivot.reindex(plot_models)[TARGETS]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=0.75,
                cbar_kws={"label": "macro-F1"}, ax=ax)
    ax.set_title("EmoWork - LOSO macro-F1 heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(os.path.join(OUT, "heatmap_macro_f1.png"))
    plt.close(fig)

    # 5. Best-per-target summary.
    best_rows = []
    for t in TARGETS:
        sub = df_plot[(df_plot["target"] == t) & (df_plot["model_short"] != "Baseline")]
        idx = sub["macro_f1_mean"].idxmax()
        best_rows.append({
            "target": t,
            "model": sub.loc[idx, "model_short"],
            "macro_f1": sub.loc[idx, "macro_f1_mean"],
            "macro_f1_std": sub.loc[idx, "macro_f1_std"],
            "balanced_acc": sub.loc[idx, "balanced_accuracy_mean"],
            "kappa": sub.loc[idx, "cohen_kappa_mean"],
            "session_f1": sub.loc[idx, "session_macro_f1"],
        })
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(OUT, "best_per_target.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(best_df))
    width = 0.22
    ax.bar(x - 1.5 * width, best_df["macro_f1"], width=width, yerr=best_df["macro_f1_std"],
           capsize=3, label="macro-F1")
    ax.bar(x - 0.5 * width, best_df["balanced_acc"], width=width, label="balanced acc")
    ax.bar(x + 0.5 * width, best_df["session_f1"], width=width, label="session F1")
    ax.bar(x + 1.5 * width, best_df["kappa"], width=width, label="kappa")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['target']}\n({r['model']})" for _, r in best_df.iterrows()])
    ax.set_ylim(-0.05, 0.85)
    ax.set_title("EmoWork - Best LOSO model per target")
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(os.path.join(OUT, "best_per_target.png"))
    plt.close(fig)

    print(f"[figures] wrote 5 figures to {OUT}")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
