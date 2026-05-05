# 07 — Comparison with prior work

## 7.1 The published WESAD landscape

The original WESAD paper (Schmidt et al., 2018) reported binary stress
versus non-stress classification accuracies up to 92.83% (chest data,
random forest) and 80.34% (wrist data) using leave-one-subject-out
cross-validation. Three-class classification (baseline / stress /
amusement) reached 80.34% accuracy on chest data. These are still
arguably the most defensible reference numbers in the WESAD literature
because the protocol is unambiguously LOSO.

Subsequent work has produced a wide spread of numbers. We summarize the
recent (2020+) literature most relevant to our findings in Table 7.1.
The protocol column is the most important entry: papers reporting
"k-fold cross-validation" without explicit subject-grouping are
implicitly window-mixing and their accuracy figures are not directly
comparable to LOSO results.

### Table 7.1 — Recent WESAD chest-sensor results

| Study (year) | Task | Protocol | Reported best |
|---|---|---|---|
| Schmidt et al. (2018) | binary stress | **LOSO** | 92.83% acc, RF |
| Schmidt et al. (2018) | 3-class | **LOSO** | 80.34% acc, RF |
| Bobade & Vani (2020) | binary stress | window 10-fold | 99.65% acc, MLP |
| Bobade & Vani (2020) | 3-class      | window 10-fold | 95.21% acc, MLP |
| Garg et al. (2021) | binary stress | LOSO | 92.4% F1, RF |
| Lai et al. (2021) | binary stress | LOSO | 91.7% acc, RF |
| Gil-Martín et al. (2022) | binary stress | window k-fold | 96.6% acc, CNN |
| Sah et al. (2022) | binary stress (chest) | LOSO | 90.1% F1 |
| Aristizabal et al. (2022) | binary stress | window 10-fold | 98.7% acc |

The pattern is clear and recurring: papers that use **LOSO** report
$\kappa$ and accuracy in a tight band around the original Schmidt
et al. numbers (90–93% accuracy on binary stress). Papers that use
**window-level k-fold or hold-out splits** without explicit subject
grouping report 96–99%. The architectural sophistication along the
column is largely uncorrelated with the metric — what *is* correlated
is the protocol.

## 7.2 Where our LOSO numbers sit

Our v5 LOSO numbers — binary valence 96.1% accuracy / $\kappa = 0.961$
(positive vs. low valence), binary arousal 91.8% / $\kappa = 0.821$,
3-class quadrant 92.9% / $\kappa = 0.863$ — sit favorably in the LOSO
column of Table 7.1. They are higher than Schmidt et al.'s binary 92.83%
and three-class 80.34% by 3–10 percentage points, which we attribute
to:

1. The 89-feature tabular extractor — substantially larger and more
   physiologically informed than Schmidt et al.'s feature set.
2. Sequence baseline correction (§4.1, point 3), which removes
   subject-level DC offset before model training.
3. Temperature-calibrated ensembling, which has small but real benefit
   on $\kappa$.

Importantly, **our LOSO numbers are *lower* than the
window-mixing numbers reported by Bobade & Vani (2020), Gil-Martín
et al. (2022), or Aristizabal et al. (2022)** — and this is the
expected and correct direction. A 99.65% accuracy under LOSO on a
fifteen-subject dataset would be remarkable; under window-stratified
$k$-fold with the same fifteen subjects supplying both train and test
windows, the same number is consistent with subject-fingerprinting
rather than stress detection (Saeb et al., 2017; Tougui et al., 2021).
Section 8 demonstrates this directly on our own models.

## 7.3 The favorable but cautious comparison

If Table 7.1 is read as "best LOSO number for chest sensors", our v5
binary valence and 3-class quadrant results are the strongest among the
2020+ entries we are aware of. We make this claim cautiously. The
appropriate caveats are:

- **Different studies map conditions to labels differently.** Some
  treat amusement as part of "non-stress"; some treat baseline and
  meditation as separate classes; we treat baseline and meditation as
  one HVLA class and exclude transition periods. Headline accuracies
  under these different label conventions are not directly comparable.
- **Reporting unit varies.** Some studies report mean per-fold accuracy,
  some report accuracy on pooled predictions, and some report
  per-condition $F_1$. We report session-level pooled $\kappa$ as our
  primary metric because it is the least sensitive to imbalance and
  the most directly auditable.
- **LOSO with 15 subjects has wide confidence intervals.** A 1–3 point
  difference between studies under LOSO is well within the per-subject
  variance demonstrated in §6.3.

For these reasons we present our LOSO numbers as **comparable-or-better
than the recent LOSO literature**, rather than as a clean state of the
art.

## 7.4 Where prior work agrees and disagrees with us

Two patterns from prior work are reinforced here:

- **Random Forest is hard to beat on WESAD.** Schmidt et al. (2018),
  Garg et al. (2021), and Lai et al. (2021) all report RF as the
  top-performing classical model under LOSO. We add Logistic
  Regression as a comparably strong alternative with better
  calibration.
- **Wrist-only performance lags chest.** Sah et al. (2022) report a
  gap of 5–10 points $F_1$ between chest and wrist; we did not run
  wrist-only experiments and therefore cannot independently confirm
  the magnitude of the gap on our pipeline, but this is highlighted
  in §9 as the most important next study.

One pattern we **disagree** with is the framing — repeated in several
recent transformer papers — that deep architectures categorically
outperform classical baselines on WESAD. Under our honest LOSO
evaluation that claim does not hold (see §6 and Figure
`fig_model_kappa_loso.png`).
