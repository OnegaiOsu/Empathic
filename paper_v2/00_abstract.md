# 00 — Abstract & Front Matter

## Title

**Picking a workplace stress detector by ablation: an honest
model × protocol × sensor × calibration evaluation on EmoWork.**

## Authors

*Empathic Project, 2026.*

## Abstract

We ask a single engineering question: *given a 31-subject multimodal
physiological corpus, which model–protocol–sensor–calibration
combination is the honest minimum for a deployable workplace stress
detector, and what is the achievable ceiling once a per-employee
calibration step is allowed?* We answer it on **EmoWork**, a 31-subject,
12-channel corpus combining cardiac (ECG, BVP, HR), electrodermal
(EDA), thermal (TEMP), inertial (ACC) and EEG (TP9, AF7, AF8, TP10)
recordings during simulated call-centre knowledge work, with binary
stress, binary arousal, binary valence and four-class affect-quadrant
labels. The published multimodal affect literature routinely reports
macro-F1 above 0.80 and Cohen's κ above 0.60; we show on EmoWork that
a substantial fraction of those numbers are protocol artefacts, that a
much smaller fraction of channels is actually doing the predictive
work, and that almost the entire residual gap to the literature can be
closed by per-subject calibration rather than by architectural change.

The study is structured as a four-axis ablation: **model × evaluation
protocol × sensor subset × calibration regime**. We benchmark eleven
models — three classical (Random Forest, Logistic Regression,
XGBoost), six deep fusion learners (1D-CNN, BiLSTM, TinyTCN,
Conformer, multi-stream, DANN-Conformer), one self-supervised
pre-trained encoder (TS-TCC), plus a soft-vote ensemble — under
leave-one-subject-out cross-validation (LOSO).
LOSO Cohen's κ peaks at $0.38$ for stress (Random Forest),
$0.15$ for arousal (Conformer fusion), $0.10$ for valence (BiLSTM fusion)
and $0.04$ for the four-class quadrant target (XGBoost). These numbers
are sober: they do not match the inflated claims that accompany many
multimodal affect benchmarks.

To explain the gap, we re-run the three classical models under three
additional protocols: subject-grouped 5-fold, window-stratified 10-fold and
a window-stratified 80/20 hold-out. Window-mixing protocols inflate stress
κ from 0.31 (LOSO) to **0.66** (window 80/20) — *more than a
doubling*. Arousal κ jumps from 0.01 to 0.43 under the same
manipulation. Subject-grouped 5-fold tracks LOSO closely, isolating the
inflation to *windows of the same subject crossing the train/test
boundary*.

We then run a per-modality ablation. The result reframes the multimodal
narrative on this corpus:

- **Stress is a cardiac problem.** ECG alone (17 features, 1 channel) reaches
  macro-F1 $= 0.696$, κ = 0.392 — *equal to or better than the full
  149-feature fusion stack*.
- **Adding EEG hurts arousal.** Physio-only features (89 dims) give
  κ = 0.076; adding EEG (60 extra dims) drops κ to $0.023$.
  Sixty EEG features overwhelm the random forest's feature selection at
  $n = 31$.
- **Valence is structurally weak under LOSO.** With 11 of 31 subjects
  single-class, HR-alone is the only modality that breaks κ = 0.10.

Finally, we add a calibration-regime axis. Two protocols capture the
deployment options that allow per-employee setup: rest-anchored LOSO
(one short rest recording per subject) and within-subject 70/30
($\approx 14$ labelled windows per subject). The rest-only variant is a
*defensible negative result*: the §5 pipeline already z-scores against
each subject's c-session statistics, so a rest-only reference is
strictly worse and every classical model loses 0.05–0.07 macro-F1.
The within-subject regime, by contrast, is dramatic:

- **Stress (LogReg): macro-F1 0.908, κ 0.822.** Crosses Schmidt
  et al.'s WESAD LOSO accuracy ceiling.
- **Arousal (BiLSTM): macro-F1 0.818.** Exceeds the DEAP and AMIGOS
  within-subject baselines (≈ 0.55–0.58) by $+0.25$.
- **Valence (BiLSTM): macro-F1 0.860.** Exceeds the same baselines by
  $+0.28$.
- **Quadrant (CNN1D): macro-F1 0.724.** Exceeds the §5 LOSO best
  (0.309) by $+0.42$.

The model-class winner itself flips between regimes: Random Forest
wins LOSO stress, but Logistic Regression and BiLSTM win the
within-subject targets — at $\approx 14$ training windows per subject,
the simplest classifier ties or beats every deep architecture on the
easier targets, and lightweight sequence models (BiLSTM, CNN1D,
TinyTCN) only pull ahead on quadrant. **No single model family
dominates across regimes.**

**Deployment recommendation.** A workplace stress detector built on
this corpus has two deployment-honest operating points:

1. **Generalisation floor (no per-employee setup).** A one-channel
   ECG sensor with 17 HRV-derived features under LOSO: macro-F1
   $\approx 0.70$, κ $\approx 0.39$.
2. **Personalisation ceiling (≤ 10 min per-employee calibration).**
   Full multimodal stack with $\approx 14$ labelled c-session windows
   per subject: macro-F1 $\approx 0.91$, κ $\approx 0.82$ on stress,
   and macro-F1 $\approx 0.82$/$0.86$ on arousal and valence.

The contribution is not a new architecture but a defensible four-axis
floor *and* ceiling: macro-F1 $\approx 0.70$ for the generalisation
problem, $\approx 0.91$ for the personalisation problem, with an
explicit accounting for how every $+0.05$ above either threshold in
the literature is most plausibly explained by protocol leakage or by
silent per-subject evaluation. Per-subject calibration, not model
architecture, is the dominant lever.

## Keywords

EmoWork, multimodal affect recognition, leave-one-subject-out, data
leakage, modality ablation, per-subject calibration, personalisation,
electrocardiography, electrodermal activity, electroencephalography,
stress detection, valence, arousal, Cohen's kappa.

## Repository

Code, configurations, full result tables, and per-subject CSVs accompany
this manuscript at the project repository. The headline experiments
each reproduce with a single command:

```bash
.venv\Scripts\python.exe runs\emowork\train_all.py             # §5 LOSO
.venv\Scripts\python.exe runs\emowork\make_figures.py          # §5 figures
.venv\Scripts\python.exe runs\emowork\relaxed_eval.py          # §6 protocols
.venv\Scripts\python.exe runs\emowork\modality_ablation.py     # §7 ablation
.venv\Scripts\python.exe runs\emowork\train_calibrated.py      # §6.6 Protocol C
.venv\Scripts\python.exe runs\emowork\train_within_subject.py  # §6.6 Protocol B
```
