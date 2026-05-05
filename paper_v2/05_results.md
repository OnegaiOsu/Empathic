# 05 — LOSO results

We report leave-one-subject-out cross-validation results for eleven models
on four targets. All metrics are means across the 31 (or fewer, for
valence — see §2.4) subject folds.

## 5.1 Headline cross-target table

The table below reports window-level macro-F1 (with subject-level standard
deviation), Cohen's κ, and session-level macro-F1.

### Stress (binary, $n = 625$, classes $[309, 316]$)

| Model | Macro-F1 | std | Bal. acc | κ | Sess. F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.297 | 0.081 | 0.500 | 0.000 | 0.294 |
| **Random Forest** | **0.677** | 0.148 | **0.712** | **0.383** | **0.721** |
| XGBoost | 0.662 | 0.159 | 0.696 | 0.354 | 0.758 |
| Logistic Regression | 0.636 | 0.176 | 0.665 | 0.298 | 0.665 |
| BiLSTM fusion | 0.619 | 0.179 | 0.655 | 0.287 | 0.593 |
| DANN-Conformer | 0.582 | 0.160 | 0.637 | 0.231 | 0.654 |
| Conformer fusion | 0.567 | 0.162 | 0.620 | 0.216 | 0.583 |
| CNN1D fusion | 0.585 | 0.160 | 0.631 | 0.225 | 0.628 |
| TinyTCN fusion | 0.555 | 0.141 | 0.596 | 0.171 | 0.611 |
| Multi-stream fusion | 0.537 | 0.172 | 0.596 | 0.167 | 0.625 |
| TS-TCC | 0.557 | 0.205 | 0.589 | 0.166 | 0.592 |
| Ensemble (no DANN) | 0.655 | 0.177 | 0.692 | 0.350 | 0.686 |

### Arousal (binary, $n = 625$, classes $[255, 370]$)

| Model | Macro-F1 | std | Bal. acc | κ | Sess. F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.266 | 0.130 | 0.500 | 0.000 | 0.293 |
| **Conformer fusion** | **0.538** | 0.114 | **0.607** | **0.149** | **0.575** |
| TinyTCN fusion | 0.526 | 0.168 | 0.617 | 0.171 | 0.549 |
| Ensemble (no DANN) | 0.522 | 0.144 | 0.592 | 0.132 | 0.570 |
| DANN-Conformer | 0.520 | 0.125 | 0.606 | 0.136 | 0.580 |
| Random Forest | 0.519 | 0.139 | 0.600 | 0.124 | 0.544 |
| CNN1D fusion | 0.510 | 0.158 | 0.606 | 0.140 | 0.580 |
| TS-TCC | 0.507 | 0.179 | 0.580 | 0.114 | 0.579 |
| BiLSTM fusion | 0.505 | 0.147 | 0.561 | 0.102 | 0.526 |
| Logistic Regression | 0.491 | 0.126 | 0.548 | 0.064 | 0.546 |
| Multi-stream fusion | 0.473 | 0.150 | 0.539 | 0.084 | 0.530 |
| XGBoost | 0.471 | 0.103 | 0.539 | 0.024 | 0.483 |

### Valence (binary, $n = 625$, classes $[542, 83]$)

| Model | Macro-F1 | std | Bal. acc | κ | Sess. F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.430 | 0.092 | 0.500 | 0.000 | 0.439 |
| **BiLSTM fusion** | **0.507** | 0.136 | **0.567** | **0.095** | **0.520** |
| XGBoost | 0.496 | 0.149 | 0.565 | 0.097 | 0.434 |
| Random Forest | 0.491 | 0.188 | 0.555 | 0.109 | 0.439 |
| Conformer fusion | 0.475 | 0.176 | 0.519 | 0.032 | 0.495 |
| TS-TCC | 0.477 | 0.132 | 0.544 | 0.035 | 0.486 |
| DANN-Conformer | 0.469 | 0.119 | 0.531 | 0.037 | 0.486 |
| Logistic Regression | 0.461 | 0.116 | 0.553 | $-0.004$ | 0.555 |
| CNN1D fusion | 0.452 | 0.087 | 0.538 | 0.022 | 0.429 |
| TinyTCN fusion | 0.454 | 0.128 | 0.494 | 0.005 | 0.429 |
| Multi-stream fusion | 0.429 | 0.084 | 0.492 | $-0.002$ | 0.423 |
| Ensemble (no DANN) | 0.448 | 0.117 | 0.522 | 0.023 | 0.434 |

### Quadrant (4-class, $n = 625$, classes $[33, 50, 337, 205]$)

| Model | Macro-F1 | std | Bal. acc | κ | Sess. F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (majority) | 0.024 | 0.048 | 0.094 | 0.000 | 0.030 |
| **XGBoost** | **0.309** | 0.134 | 0.413 | **0.045** | 0.234 |
| Random Forest | 0.305 | 0.147 | 0.420 | 0.049 | 0.192 |
| DANN-Conformer | 0.257 | 0.111 | 0.376 | 0.065 | 0.266 |
| BiLSTM fusion | 0.255 | 0.141 | 0.357 | 0.053 | 0.246 |
| CNN1D fusion | 0.252 | 0.106 | 0.387 | 0.053 | 0.278 |
| Multi-stream fusion | 0.247 | 0.123 | 0.379 | 0.033 | 0.198 |
| TinyTCN fusion | 0.238 | 0.109 | 0.368 | 0.042 | 0.210 |
| Conformer fusion | 0.225 | 0.093 | 0.342 | 0.031 | 0.253 |
| TS-TCC | 0.216 | 0.091 | 0.314 | 0.026 | 0.373 |
| Logistic Regression | 0.195 | 0.086 | 0.312 | $-0.013$ | 0.204 |
| Ensemble (no DANN) | 0.283 | 0.142 | 0.401 | 0.055 | 0.202 |

## 5.2 Best-per-target

| Target | Best model | Macro-F1 | std | Bal. acc | κ | Sess. F1 |
|---|---|---:|---:|---:|---:|---:|
| Stress  | Random Forest      | 0.677 | 0.148 | 0.712 | **0.383** | 0.721 |
| Arousal | Conformer fusion   | 0.538 | 0.114 | 0.607 | 0.149 | 0.575 |
| Valence | BiLSTM fusion      | 0.507 | 0.136 | 0.567 | 0.095 | 0.520 |
| Quadrant | XGBoost           | 0.309 | 0.134 | 0.413 | 0.045 | 0.234 |

These are reported in [`figures/best_per_target.csv`](../results/emotion/emowork/figures/best_per_target.csv)
and visualised in [`best_per_target.png`](../results/emotion/emowork/figures/best_per_target.png).

## 5.3 Three observations from these tables

**(a) The best model per target is from a different family on every target.**
Random Forest wins stress, a Conformer wins arousal, BiLSTM wins valence,
XGBoost wins quadrant. There is no architectural prior that wins across the
board. This argues against deep architecture comparisons as the unit of
contribution; the dataset is too small for the model family to dominate the
random fold-to-fold variation.

**(b) Standard deviations are large.** The macro-F1 std on every model is
0.08–0.20 across the 31 LOSO folds. The Conformer's stress F1 has std
0.16; the difference between the *best* and *fourth-best* stress model
(0.677 vs 0.619) is well within one standard deviation. We therefore do
not claim a "winner" without subject-paired Wilcoxon testing — and even
those tests, when run, mostly fail to reject ties between adjacent
classical and fusion learners.

**(c) Deep models do not justify their compute on tabular features.**
On stress, the strongest deep model (BiLSTM fusion) trails Random Forest
by 0.058 macro-F1 and 0.096 κ. On valence, the strongest deep
model edges out the random forest by 0.016 macro-F1 — well inside the
noise. On arousal, the Conformer wins, but by 0.019 macro-F1 over
TinyTCN and 0.019 over Random Forest. Deep models earn their keep on
arousal and on quadrant (where DANN ties them); on stress and valence,
they are not yet better than a forest with hand-crafted HRV features.

## 5.4 What about the ensemble?

The soft-vote ensemble approximately tracks the best classical model on
each target (within $\pm 0.03$ macro-F1) but does not surpass it. In
plain language: the deep models contribute *redundant* probability mass
to the ensemble on most folds. This is consistent with diversity studies
that find deep models trained on similar feature stacks correlate
strongly in their errors. The DANN model is missing from the current
ensemble row due to a logging-key bug in
[`runs/emowork/train_all.py`](../runs/emowork/train_all.py); future
sweeps will include it. We report this as an artefact, not a finding.

## 5.5 Cross-target figures

Five summary figures are written to
[`results/emotion/emowork/figures/`](../results/emotion/emowork/figures/):

| File | Content |
|---|---|
| `cross_target_macro_f1.png` | Bar chart of macro-F1 per (target, model) |
| `cross_target_kappa.png`    | Bar chart of κ per (target, model)  |
| `cross_target_session_f1.png` | Session-level macro-F1 |
| `heatmap_macro_f1.png`      | (model × target) heatmap of macro-F1 |
| `best_per_target.png`       | Best-per-target headline figure |
