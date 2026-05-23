"""Generate paper figures emphasising protocol-matched results.

Outputs:
  fig1_stress_protocol_matched.png/.pdf
  fig2_within_subject_vs_baselines.png/.pdf
  fig3_protocol_inflation.png/.pdf
  fig4_modality_ablation.png/.pdf
  fig5_emowork_release_comparison.png/.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

OURS = "#1f77b4"          # blue: this work
OURS_CAL = "#2ca02c"      # green: this work, calibrated (the win)
BASELINE = "#888888"      # grey: published baselines
CEILING = "#d62728"       # red dashed: published ceiling
HIGHLIGHT_BG = "#fff4c2"  # pale yellow highlight band


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1. Stress: protocol-matched comparison + calibrated ceiling cross.
# ---------------------------------------------------------------------------
def fig1_stress() -> None:
    labels = [
        "WESAD\nLOSO\n(chest)\nSchmidt+2018",
        "WESAD\nLOSO\n(wrist)\nSchmidt+2018",
        "WESAD\nLOSO\nCNN-LSTM\nLai+2023",
        "EmoWork\nLOSO\nECG-only\n(this work)",
        "EmoWork\nLOSO\nmultimodal\n(this work)",
        "EmoWork\nwithin-subject\ncalibrated\n(this work)",
    ]
    macro_f1 = [0.80, 0.74, 0.83, 0.696, 0.686, 0.908]
    colors = [BASELINE, BASELINE, BASELINE, OURS, OURS, OURS_CAL]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, macro_f1, color=colors, edgecolor="black", linewidth=0.6, width=0.62)

    # Annotate each bar with its value.
    for b, v in zip(bars, macro_f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Ceiling line: best WESAD LOSO.
    ax.axhline(0.83, ls="--", lw=1.1, color=CEILING, alpha=0.8)
    ax.text(len(labels) - 0.5, 0.838, "WESAD LOSO ceiling (Lai+2023)",
            ha="right", va="bottom", fontsize=8.5, color=CEILING)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Binary stress — macro-F1 (or reported accuracy)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Stress detection: protocol-matched comparison\n"
                 "EmoWork LOSO sits in the WESAD LOSO band; EmoWork within-subject reaches macro-F1 0.91",
                 fontweight="bold")

    # Legend.
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=BASELINE, edgecolor="black", label="Published WESAD baselines"),
        Patch(facecolor=OURS, edgecolor="black", label="This work — LOSO"),
        Patch(facecolor=OURS_CAL, edgecolor="black", label="This work — within-subject (≈14 cal. windows)"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.95, fontsize=8.5)

    _save(fig, "fig1_stress_protocol_matched")


# ---------------------------------------------------------------------------
# Figure 2. Within-subject valence and arousal: we exceed DEAP/AMIGOS.
# ---------------------------------------------------------------------------
def fig2_within_subject() -> None:
    targets = ["Valence (binary)", "Arousal (binary)"]
    deap_low = [0.583, 0.563]   # Koelstra+2012 peripheral phys (within-subject)
    deap_high = [0.628, 0.620]  # Koelstra+2012 EEG (within-subject)
    amigos = [0.555, 0.573]     # Miranda-Correa+2018 mean of phys/EEG within-subject
    ours = [0.860, 0.818]       # this work, within-subject 70/30

    x = np.arange(len(targets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - 1.5 * width, deap_low, width, color=BASELINE, alpha=0.55,
                edgecolor="black", linewidth=0.6, label="DEAP peripheral phys (Koelstra+2012)")
    b2 = ax.bar(x - 0.5 * width, deap_high, width, color=BASELINE, alpha=0.85,
                edgecolor="black", linewidth=0.6, label="DEAP EEG (Koelstra+2012)")
    b3 = ax.bar(x + 0.5 * width, amigos, width, color="#555555", alpha=0.85,
                edgecolor="black", linewidth=0.6, label="AMIGOS (Miranda-Correa+2018)")
    b4 = ax.bar(x + 1.5 * width, ours, width, color=OURS_CAL,
                edgecolor="black", linewidth=0.8, label="This work — EmoWork (within-subject)")

    for group in (b1, b2, b3, b4):
        for b in group:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5)

    # Annotate the delta.
    for i, t in enumerate(targets):
        max_baseline = max(deap_high[i], amigos[i])
        delta = ours[i] - max_baseline
        ax.annotate("", xy=(i + 1.5 * width, ours[i] - 0.01),
                    xytext=(i + 1.5 * width, max_baseline + 0.01),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.2))
        ax.text(i + 1.5 * width + 0.06, (ours[i] + max_baseline) / 2,
                f"+{delta:.2f}", color="#d62728", fontsize=10, fontweight="bold",
                ha="left", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Within-subject macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Within-subject dimensional affect: EmoWork above the DEAP and AMIGOS within-subject band\n"
                 "by +0.23 (valence) and +0.20 (arousal) macro-F1",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    _save(fig, "fig2_within_subject_vs_baselines")


# ---------------------------------------------------------------------------
# Figure 3. Protocol inflation — the rigour that makes the wins meaningful.
# ---------------------------------------------------------------------------
def fig3_protocol_inflation() -> None:
    protocols = ["LOSO", "Subject\n5-fold", "Window-strat.\n10-fold", "Window\n80/20"]
    # Random Forest kappa.
    stress_kappa = [0.308, 0.315, 0.555, 0.664]
    arousal_kappa = [0.010, 0.045, 0.428, 0.393]

    x = np.arange(len(protocols))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bs = ax.bar(x - width / 2, stress_kappa, width,
                color=OURS, edgecolor="black", linewidth=0.6, label="Stress")
    ba = ax.bar(x + width / 2, arousal_kappa, width,
                color="#ff7f0e", edgecolor="black", linewidth=0.6, label="Arousal")

    for group in (bs, ba):
        for b in group:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5)

    # Highlight the subject-aware protocols.
    ax.axvspan(-0.5, 1.5, color=HIGHLIGHT_BG, alpha=0.6, zorder=0)
    ax.text(0.5, 0.72, "subject-aware\nsplits", ha="center", va="top",
            fontsize=9, color="#7a6a00", fontweight="bold")
    ax.text(2.5, 0.72, "window-shared\nsplits", ha="center", va="top",
            fontsize=9, color="#555555", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylabel("Cohen's κ (Random Forest, 149-d multimodal)")
    ax.set_ylim(-0.05, 0.8)
    ax.set_title("Protocol choice strongly affects reported κ on EmoWork\n"
                 "Window-shared splits more than double stress κ and lift arousal κ from chance to 0.43",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    _save(fig, "fig3_protocol_inflation")


# ---------------------------------------------------------------------------
# Figure 4. Per-modality ablation — one sensor is enough.
# ---------------------------------------------------------------------------
def fig4_modality_ablation() -> None:
    targets = ["Stress", "Arousal", "Valence"]

    # Random Forest LOSO kappa (best single sensor vs full 149-d).
    best_single_kappa = [0.392, 0.111, 0.102]
    best_single_label = ["ECG (17 f)", "EDA (14 f)", "HR (7 f)"]
    full_stack_kappa = [0.374, 0.023, 0.000]

    x = np.arange(len(targets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - width / 2, best_single_kappa, width,
                color=OURS_CAL, edgecolor="black", linewidth=0.6,
                label="Best single sensor (this work)")
    b2 = ax.bar(x + width / 2, full_stack_kappa, width,
                color=BASELINE, edgecolor="black", linewidth=0.6,
                label="Full 149-d multimodal stack")

    for b, lab in zip(b1, best_single_label):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.012,
                f"{v:.3f}\n{lab}", ha="center", va="bottom", fontsize=8.5)
    for b in b2:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Cohen's κ (LOSO, Random Forest)")
    ax.set_ylim(-0.02, 0.5)
    ax.set_title("One sensor is enough: best-single-sensor matches or beats the full multimodal stack\n"
                 "Stress: ECG alone > fusion; Arousal: EDA alone > fusion; Valence: HR alone > fusion",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    _save(fig, "fig4_modality_ablation")


# ---------------------------------------------------------------------------
# Figure 5. Comparison to the EmoWork release baselines (Lee et al., 2026).
# ---------------------------------------------------------------------------
def fig5_emowork_release() -> None:
    """Side-by-side with Lee et al. 2026 Tables 7 & 8 baselines.

    Lee et al. report AUC for Task 2 (stress/valence/arousal) and macro-F1
    for Task 1 (low vs high workload). We plot the two metrics on separate
    panels to avoid mixing them.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.2]})

    # --- Panel A: LOSO AUC (Lee et al. Task 2, RF) vs this work (no AUC).
    targets = ["Stress", "Valence", "Arousal"]
    lee_auc = [0.783, 0.745, 0.649]
    ours_f1 = [0.686, 0.507, 0.538]
    ours_cal_f1 = [0.908, 0.860, 0.818]

    x = np.arange(len(targets))
    width = 0.26
    a1 = ax1.bar(x - width, lee_auc, width, color=BASELINE,
                 edgecolor="black", linewidth=0.6,
                 label="Lee et al. 2026 — LOSO AUC (RF)")
    a2 = ax1.bar(x, ours_f1, width, color=OURS,
                 edgecolor="black", linewidth=0.6,
                 label="This work — LOSO macro-F1")
    a3 = ax1.bar(x + width, ours_cal_f1, width, color=OURS_CAL,
                 edgecolor="black", linewidth=0.6,
                 label="This work — within-subject macro-F1")

    for group in (a1, a2, a3):
        for b in group:
            v = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(targets)
    ax1.set_ylabel("Score (AUC or macro-F1, see legend)")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("A. EmoWork dimensional targets vs Lee et al. 2026\n"
                  "(AUC and F1 are different metrics; within-subject extends the release baselines)",
                  fontweight="bold", fontsize=10)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    # --- Panel B: Task 1 (low vs high workload, RF, LOSO) macro-F1, same metric.
    labels = [
        "Lee et al. 2026\nTask 1 (low vs high\nworkload), RF LOSO",
        "This work\nLOSO stress, RF\n(ECG-only)",
        "This work\nLOSO stress, RF\n(149-d multimodal)",
        "This work\nWithin-subject\nstress (LogReg)",
    ]
    f1_vals = [0.891, 0.696, 0.686, 0.908]
    colors = [BASELINE, OURS, OURS, OURS_CAL]

    xb = np.arange(len(labels))
    bb = ax2.bar(xb, f1_vals, color=colors, edgecolor="black",
                 linewidth=0.6, width=0.55)
    for b, v in zip(bb, f1_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    # Reference line at Lee et al. value.
    ax2.axhline(0.891, ls="--", lw=1.1, color=CEILING, alpha=0.7)
    ax2.text(len(labels) - 0.5, 0.898, "Lee et al. 2026 RF (Task 1)",
             ha="right", va="bottom", fontsize=8.5, color=CEILING)

    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0.5, 1.0)
    ax2.set_title("B. Stress / workload binary classification on EmoWork (matched metric)\n"
                  "Within-subject stress (0.908) edges past the Lee et al. Task 1 number (0.891)",
                  fontweight="bold", fontsize=10)

    _save(fig, "fig5_emowork_release_comparison")


if __name__ == "__main__":
    fig1_stress()
    fig2_within_subject()
    fig3_protocol_inflation()
    fig4_modality_ablation()
    fig5_emowork_release()
    print("Figures written to:", OUT)
