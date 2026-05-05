# 00 — Abstract & Front Matter

## Title
**Honest evaluation of multimodal stress detection: protocol matters more than architecture.
A workplace-deployment-oriented benchmark on WESAD.**

## Authors
*Empathic Project, 2026.*

## Abstract

Workplace stress detection from wearable physiological sensors is a promising
application of affective computing, yet published benchmarks routinely report
near-perfect accuracy that does not survive deployment to new users.
We replicate and extend the WESAD chest-sensor benchmark with a survey of
seven model families (Random Forest, Logistic Regression, XGBoost, CNN1D,
TinyTCN, BiLSTM, and a Conformer late-fusion model), evaluated on three
tasks: arousal (binary), valence (binary), and the joint VA quadrant
(three-class, since the LVLA quadrant is structurally absent from WESAD).

Across five iterations of training improvements (v1–v5) we converge on a
late-fusion architecture combining hand-crafted physiological features with
sequence representations, temperature-calibrated ensembling, and per-subject
baseline correction. Under leave-one-subject-out cross-validation (LOSO) the
strongest models reach session-level Cohen's $\kappa = 0.86$ (quadrant,
Logistic Regression), $\kappa = 0.96$ (valence, Random Forest), and
$\kappa = 0.82$ (arousal, Logistic Regression). Deep architectures close the
gap relative to v4 by +0.09 to +0.15 $\kappa$ but do not exceed strong
classical baselines on tabular features.

We then re-evaluate the same models under three additional protocols
commonly used in the literature: subject-grouped 5-fold, window-stratified
10-fold, and a window-stratified 80/20 split. Window-mixing protocols
inflate $\kappa$ by **+0.06 to +0.21** relative to LOSO. The inflation is
largest on quadrant (+0.18) and arousal (+0.16), where subject identity
dominates the feature space. Subject-grouped 5-fold tracks LOSO closely,
indicating that the methodological harm comes specifically from leaking
windows of the *same subject* across train and test — not from $k$ itself.

We argue that for **workplace stress monitoring**, where deployment by
definition involves new users, LOSO or subject-grouped folds are the only
honest protocols. We discuss implications for system design: (i) deep models
do not justify their compute cost over Random Forest or Logistic Regression
on tabular features at this dataset scale; (ii) personalization
(few-shot baseline calibration) is likely the highest-leverage future work;
and (iii) reported "0.95+" benchmarks in the WESAD literature should be
interpreted as upper bounds on within-subject memorization, not on
generalization to a new employee.

## Keywords
WESAD, stress detection, affective computing, leave-one-subject-out,
cross-validation, data leakage, valence, arousal, workplace wellbeing,
physiological signals, late fusion, ensemble.

## Repository

Code, configuration, full result tables, and per-subject CSVs accompany this
manuscript at the project repository. Reproduction is single-command:

```bash
python -m empathic.train --dataset wesad --target quadrant --fusion
python runs/relaxed_eval.py
```

All deep training runs use leave-one-subject-out cross-validation by default.
Window-stratified protocols are available only through the explicit
`relaxed_eval.py` driver, which exists *because* of the methodological
contribution argued in Section 7.
