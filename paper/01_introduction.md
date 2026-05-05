# 01 — Introduction

## 1.1 Workplace stress as the use case

Persistent occupational stress is one of the most pervasive — and most
expensive — health problems in modern knowledge work. The European Agency
for Safety and Health at Work estimates that work-related stress accounts
for half of all lost working days, and post-pandemic survey data indicate
that the prevalence of self-reported burnout has increased rather than
receded with the shift to remote and hybrid work (Hassard et al., 2018;
Gallup, 2023). Continuous, unobtrusive, and *personalized* monitoring of
physiological correlates of stress is therefore a natural target for
applied affective computing.

The promise of wearable stress detection is well rehearsed: a chest strap,
wrist band, or instrumented keyboard captures heart-rate variability (HRV),
electrodermal activity (EDA), respiration, electromyography (EMG), or skin
temperature, and a learned model maps short windows of those signals to a
discrete emotional state or a continuous valence-arousal coordinate. In a
workplace setting the deliverable is operationally simple: a per-user
indicator of "this person appears to be entering a sustained high-stress
state" that can drive nudges, breaks, or workload adjustment.

This paper is about whether the published numbers we use to justify
building such a system are honest.

## 1.2 The hidden cost of optimistic evaluation

The WESAD dataset (Schmidt et al., 2018) has become the *de facto* public
benchmark for chest-and-wrist stress detection. A casual reading of the
WESAD literature suggests that the problem is essentially solved: many
recent papers report binary stress accuracies above 95% and Cohen's
$\kappa > 0.95$ (Bobade & Vani, 2020; Gil-Martín et al., 2022).

Two facts complicate that picture.

1. WESAD contains only fifteen subjects. Even small amounts of leakage
   between training and evaluation can move reported metrics by tens of
   percentage points.
2. The same dataset is also reported with markedly lower numbers when
   evaluated under leave-one-subject-out (LOSO) cross-validation
   (Schmidt et al., 2018; Garg et al., 2021; Sah et al., 2022). The gap
   between "stratified k-fold over windows" and "LOSO" is rarely
   foregrounded, even though it is the only gap that matters for an
   employer who wants to deploy the model on a new employee.

This is not a hypothetical concern. Saeb et al. (2017) showed in a clinical
ML setting that subject-level leakage routinely turns a population-level
prediction problem into a subject-fingerprinting problem, with reported
accuracies that reflect identity recognition rather than the construct
under study. Tougui et al. (2021) document the same phenomenon broadly
across healthcare ML benchmarks. Our results below are consistent with
both: under LOSO our best models reach $\kappa \in [0.82, 0.96]$, but
under window-stratified k-fold the same models on the same data reach
$\kappa \in [0.91, 0.99]$ — an inflation of 0.06 to 0.21 in $\kappa$
attributable purely to protocol choice.

## 1.3 Contributions

This work makes four contributions.

1. **A model survey for workplace stress detection.** We benchmark seven
   architectures — three classical (Random Forest, Logistic Regression,
   XGBoost), three sequence baselines (CNN1D, TinyTCN, BiLSTM), and one
   modern attention-based model (Conformer) — on three targets relevant
   to a workplace dashboard: binary valence, binary arousal, and the
   joint quadrant. All models are exposed both as standalone classifiers
   and as members of a late-fusion / temperature-calibrated ensemble.
2. **A documented training-iteration trajectory (v1 → v5).** We report
   five iterations of preprocessing, augmentation, and fusion changes,
   including failed iterations (v6 channel dropout), so that future work
   can avoid retracing dead ends.
3. **A protocol-leakage audit.** We re-evaluate every classical model
   under four cross-validation protocols spanning the spectrum from
   "fully subject-independent" (LOSO) to "fully window-mixed"
   (stratified 80/20 split). We quantify the inflation that protocol
   choice introduces and interpret it in light of the deployment
   scenario.
4. **A workplace-deployment recommendation.** We argue that, given the
   data we have today, the production-ready stack for workplace stress
   detection is a Random Forest or Logistic Regression on tabular
   physiological features with per-subject baseline calibration —
   not a deep sequence model — and that the literature's headline
   accuracy figures should be discounted accordingly.

The remainder of the paper is organized as follows. Section 2 details the
WESAD dataset and its limitations. Section 3 motivates the model array
from a workplace-deployment perspective. Sections 4 and 5 cover
preprocessing and the iterative training story. Section 6 reports honest
LOSO results with per-class confusion analysis. Section 7 places those
numbers in the context of recently published WESAD work. Section 8 is the
methodological core: a re-evaluation of our v5 models under four
cross-validation protocols. Section 9 returns to the workplace-deployment
question and argues for a deliberately conservative production stack.
Section 10 concludes.
