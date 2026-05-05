# 07 — Per-modality ablation

If §6 was about *protocol* leakage, this section is about *modality*
budget. We ask: of the 149 tabular features and seven sensor groups,
which ones actually contribute to LOSO performance?

## 7.1 Procedure

We take the strongest tabular learner from §5 (Random Forest with
balanced class weights and inverse-frequency sample weights) and run it
under LOSO on subsets of features defined by sensor prefix. Folds in
which the training set has only one class are skipped; folds in which
the test set has only one class are skipped on binary targets. The
script is
[`runs/emowork/modality_ablation.py`](../runs/emowork/modality_ablation.py);
results are in
[`modality_ablation.csv`](../results/emotion/emowork/ablations/modality_ablation.csv).

The nine modality cells are:

| Set | Features |
|---|---:|
| ECG | 17 |
| BVP | 17 |
| HR  | 7 |
| EDA | 14 |
| TEMP | 9 |
| ACC | 25 |
| EEG | 60 |
| Physio (ECG ∪ BVP ∪ HR ∪ EDA ∪ TEMP ∪ ACC) | 89 |
| Physio + EEG (all) | 149 |

## 7.2 Results

### Stress (binary, $n = 625$)

| Set | Features | Macro-F1 | Bal. acc | κ |
|---|---:|---:|---:|---:|
| **ECG** | **17** | **0.696** | **0.697** | **0.392** |
| Physio + EEG (all) | 149 | 0.686 | 0.689 | 0.374 |
| Physio (no EEG) | 89 | 0.684 | 0.686 | 0.369 |
| HR | 7 | 0.609 | 0.611 | 0.220 |
| TEMP | 9 | 0.604 | 0.606 | 0.210 |
| EDA | 14 | 0.599 | 0.600 | 0.199 |
| EEG | 60 | 0.598 | 0.601 | 0.199 |
| BVP | 17 | 0.564 | 0.566 | 0.131 |
| ACC | 25 | 0.529 | 0.530 | 0.060 |

**Headline:** ECG-alone (17 features, one channel) delivers
κ = 0.392, beating the full 149-feature stack (κ = 0.374)
and the no-EEG physio stack (κ = 0.369). On stress, *every
sensor beyond ECG is at best neutral and sometimes harmful*.

### Arousal (binary, $n = 625$)

| Set | Features | Macro-F1 | Bal. acc | κ |
|---|---:|---:|---:|---:|
| **EDA** | **14** | **0.552** | **0.554** | **0.111** |
| TEMP | 9 | 0.535 | 0.535 | 0.071 |
| Physio (no EEG) | 89 | 0.525 | 0.535 | 0.076 |
| ECG | 17 | 0.521 | 0.533 | 0.072 |
| Physio + EEG (all) | 149 | 0.494 | 0.510 | 0.023 |
| EEG | 60 | 0.481 | 0.506 | 0.012 |
| HR | 7 | 0.488 | 0.491 | $-0.018$ |
| BVP | 17 | 0.470 | 0.482 | $-0.037$ |
| ACC | 25 | 0.458 | 0.478 | $-0.048$ |

**Headline:** EDA alone is the best single sensor, consistent with the
arousal-EDA literature. *Adding EEG to the 89-d physio stack reduces
κ from 0.076 to 0.023* — a 70% relative drop. The 60 EEG
features compete for the random forest's split budget against the 89
informative physio features and dilute the predictive signal.

### Valence (binary, $n = 625$, classes $[542, 83]$)

| Set | Features | Macro-F1 | Bal. acc | κ |
|---|---:|---:|---:|---:|
| **HR** | **7** | **0.522** | **0.536** | **0.102** |
| EDA | 14 | 0.497 | 0.524 | 0.071 |
| TEMP | 9 | 0.481 | 0.512 | 0.034 |
| ECG | 17 | 0.462 | 0.504 | 0.013 |
| BVP | 17 | 0.442 | 0.498 | $-0.005$ |
| ACC | 25 | 0.443 | 0.500 | 0.000 |
| EEG | 60 | 0.443 | 0.500 | 0.000 |
| Physio (no EEG) | 89 | 0.443 | 0.500 | 0.000 |
| Physio + EEG (all) | 149 | 0.443 | 0.500 | 0.000 |

**Headline:** Valence is structurally weak on this corpus — the
combined feature stacks collapse to majority, while *HR alone with
seven features* recovers κ = 0.10. This is consistent with the
class structure: 11 / 31 subjects single-class on valence, severe
imbalance, and a feature space in which subject baselines dominate.
The seven HR features are coarse enough to be largely subject-invariant
(rates are physiological universals to within $\pm 30\%$) and so
generalise across folds where richer features overfit.

## 7.3 What the ablation tells us

1. **Stress is a cardiac problem.** ECG alone reaches the best
   reported κ on the corpus (0.392). Multimodal fusion does not
   help; on this dataset, at this scale, additional sensors add more
   variance than signal.
2. **Arousal is an EDA problem.** EDA alone wins. EEG, on net, hurts:
   the curse of dimensionality at $n = 31$ subjects with 60 features
   per target dilutes the informative 89 physio features.
3. **Valence is structurally weak.** The richest feature sets collapse
   to the majority class. Only the smallest, most subject-invariant
   feature set (7 HR features) recovers any signal.
4. **ACC and BVP are uniformly weak.** ACC never breaks κ = 0.06
   on any target; BVP is the weakest cardiac modality despite
   nominally encoding the same information as ECG (lower SNR, more
   subject-specific morphology).

## 7.4 An honest claim against fusion at $n \approx 30$

The §5 fusion learners — multi-stream, Conformer, BiLSTM — are
ostensibly designed to combine modalities adaptively. Yet they do not
beat ECG-alone Random Forest on stress; they do not beat the
EDA-alone Random Forest on arousal by a margin that survives
subject-paired testing. The ablation suggests *why*: multimodal
features at this scale introduce noise faster than they introduce
signal, and the non-deep models are at least as good at exploiting
small modality-aware feature subsets as the deep ones.

The headline figures for this section are
[`modality_ablation_macro_f1.png`](../results/emotion/emowork/ablations/modality_ablation_macro_f1.png)
and
[`modality_ablation_kappa.png`](../results/emotion/emowork/ablations/modality_ablation_kappa.png).

## 7.5 Implication for system design

A workplace-deployment system targeting stress on this style of corpus
should default to a single ECG sensor and a 17-feature HRV-derived
representation. The marginal cost of adding EDA, TEMP, ACC and EEG
(four extra sensors, four extra calibration steps, four extra failure
modes) is not justified by a corresponding lift in held-out κ.

For arousal-targeted systems, the analogous statement holds for EDA.
For valence, no current configuration is deployment-ready on this
corpus.
