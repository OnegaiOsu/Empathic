# 01 — Introduction

## 1.1 The credibility problem in multimodal affect recognition

Wearable affect recognition has matured into a rich publication area: in
the last five years dozens of papers have reported subject-independent
accuracy above 80% and Cohen's κ above 0.60 on canonical
benchmarks (WESAD, AMIGOS, K-EmoPhone, MAHNOB-HCI, EmoWork). At the same
time, deployments of these models to real users continue to underperform
their reported numbers by a wide margin (Schmidt et al., 2019; Smets et
al., 2018). Two methodological gaps account for most of the discrepancy:

1. **Cross-validation protocols leak subject identity.** Many studies
   either use random window-level splits or a grouped split that still
   permits same-subject windows to appear in train and test (e.g.
   stratified $k$-fold over windows ignoring subject id). With
   sub-second autocorrelated signals and 60–120 s windows, the model is
   not learning to recognise affect — it is learning to identify the
   subject.
2. **"Multimodal" is conflated with "more channels are better".** On
   small subject pools ($n \le 30$) the classical curse of
   dimensionality dominates: every additional sensor adds features
   that compete for the random forest's split budget, the SVM's
   regularisation budget, the deep model's capacity. We will show
   below that on EmoWork, *adding EEG to a physio-only stack reduces
   arousal κ by 0.05*.

This paper examines both gaps on the EmoWork corpus, a 31-subject
12-channel multimodal recording released for the explicit purpose of
modelling affect during knowledge work. EmoWork is a useful test bed
because it is large enough to make $k$-fold protocols look respectable
(625 windows) but small enough at the *subject* level that LOSO is
honestly hard.

## 1.2 Scope: this is an ablation study aimed at one product question

This paper is structured as an **ablation study, not a benchmark
race**. The product question we are scoping is concrete: *what is the
honest minimum specification — model, evaluation protocol, sensor
stack, calibration regime — for a deployable workplace stress detector
trained on a corpus of the size and shape of EmoWork?* Every experiment
in the paper exists because varying *one* axis of that question changes
the answer:

- **Model axis (§5).** Eleven model families covering classical,
  deep-fusion, self-supervised and ensemble approaches.
- **Protocol axis (§6).** Four cross-validation protocols differing
  only in how subjects and windows are partitioned, holding model and
  feature stack fixed.
- **Sensor axis (§7).** Nine modality subsets ranging from a single
  ECG channel to the full 12-channel / 149-feature multimodal stack,
  holding model (Random Forest) and protocol (LOSO) fixed.
- **Calibration axis (§6.6).** Three calibration regimes — LOSO with
  no per-subject information (the §5 baseline), rest-anchored LOSO
  (one rest recording per subject), and within-subject 70/30 ($\approx
  14$ labelled windows per subject). The first is the *generalisation*
  setting; the third is the *personalisation* ceiling that workplace
  deployments with employee-side calibration can reach.

We do not propose a new architecture. We propose that *all four
axes must be ablated* before a multimodal affect-recognition number is
credible.

## 1.3 Contributions

We make four contributions.

1. **An honest LOSO baseline.** We benchmark eleven models — three
   classical, six deep fusion architectures, one self-supervised
   pre-trained encoder, and a soft-vote ensemble — under LOSO on four
   targets (stress, arousal, valence, quadrant). We report window-level
   and session-level metrics with subject-level standard deviations.
   See §5.
2. **A protocol audit.** We re-run the three classical models under
   four protocols (LOSO, subject-grouped 5-fold, window-stratified
   10-fold, window 80/20) and quantify the inflation introduced by
   each. We show that subject-grouped folds track LOSO; only
   *window-level* leakage produces the order-of-0.4 κ
   inflation that explains the gap between literature and deployment.
   See §6.
3. **A per-modality ablation.** We run a sensor-level ablation under
   LOSO. The headline finding — *ECG alone equals the full 149-feature
   stack on stress* — argues that the appropriate honest baseline on
   this corpus is single-sensor RF, not multimodal fusion. We also
   identify a regime in which EEG actively harms downstream metrics
   under the curse of dimensionality. See §7.
4. **A calibration-regime ablation.** We add two personalisation
   protocols (rest-anchored LOSO and within-subject 70/30) to bound
   the achievable ceiling once per-employee setup is allowed. The
   rest-only variant is a *defensible negative result* (it loses to
   §5's c-session z-scoring); the within-subject variant lifts every
   target by $+0.23$ to $+0.42$ macro-F1 over the LOSO baseline,
   exceeding the DEAP/AMIGOS within-subject ceilings on dimensional
   affect by $\approx +0.25$. Per-subject calibration, not
   architecture, is the dominant lever on this corpus. See §6.6.

## 1.4 Scope and non-goals

We do not propose a new architecture. Every model in this paper is
either a sklearn estimator or a published architecture from the affect
recognition or self-supervised time-series literature. The contribution
is methodological: *what does the EmoWork corpus actually support, and
under which protocols?*

We do not claim that fusion never helps. We claim that, on this corpus,
under LOSO, a single ECG-derived feature set ties the full multimodal
stack on the most learnable target. Larger corpora may yield a
different verdict; that is the appropriate next experiment, not the
appropriate next architectural tweak.

We make no clinical or deployment claims. EmoWork is a research corpus
collected in a controlled office environment over a small population.
Effect sizes here are upper bounds for the cleanest case of the
phenomenon.
