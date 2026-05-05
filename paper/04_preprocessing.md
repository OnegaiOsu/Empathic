# 04 — Preprocessing & feature extraction

## 4.1 Pipeline overview

The preprocessing pipeline is structured as four stages, applied
identically across all models so that protocol-level comparisons remain
clean:

1. **Raw segmentation.** 60-second windows at 700 Hz with 30-second
   stride; transition periods between conditions discarded.
2. **Per-channel standardization.** Each of the eight chest channels is
   z-scored within a subject (mean and standard deviation computed on
   the subject's own training-condition data, so no test-set statistics
   leak into normalization).
3. **Sequence baseline correction (v5).** For models that consume raw
   sequences, we subtract the per-subject mean of all baseline-condition
   windows from every window of that subject. This effectively removes
   the subject's resting physiological "DC offset" before the sequence
   model ever sees the signal. The motivation is that without this step,
   a deep model spends substantial capacity learning to identify the
   subject (their resting heart rate, baseline EDA, body temperature)
   rather than the affective state. Schmidt et al. (2018) and several
   subsequent WESAD studies (Lai et al., 2021; Sah et al., 2022) report
   improvements from analogous baseline-normalization tricks; ours
   differs in being applied uniformly to every input window during
   training rather than as a post-hoc per-condition baseline subtraction.
4. **Tabular feature extraction.** A separate parallel branch computes
   a fixed 89-dimensional feature vector per window for the classical
   models and for the late-fusion deep models.

## 4.2 The 89-feature tabular set

The tabular feature set is intentionally close to those reported in the
WESAD literature so that our results are commensurable with prior work.
Per window we compute, in summary form: ECG-derived heart rate, RR
intervals, time-domain HRV (SDNN, RMSSD, pNN50) and frequency-domain
HRV (LF, HF, LF/HF) statistics; EDA tonic level (SCL) and phasic
response (SCR) counts and amplitudes; respiration rate, depth, and
inspiration/expiration ratios; EMG mean, variance, and spectral energy
bands; skin temperature mean and slope; and per-axis accelerometer means,
variances, and signal magnitude vectors. Full per-feature definitions
and the exact extractor live in `src/empathic/features/wesad.py`; we
include only summary statistics here for brevity.

## 4.3 Augmentation

Sequence augmentation (applied only to the deep models, only at training
time, never on the validation fold) is composed of:

- **Jitter.** Additive Gaussian noise scaled to per-channel standard
  deviation.
- **Scaling.** Per-channel multiplicative scaling sampled around 1.0.
- **Time warping.** Smooth nonlinear time deformation via cubic spline
  (Um et al., 2017).
- **Channel dropout (v6 only, see §5.5).** Independent Bernoulli masking
  of entire channels, $p = 0.1$.

For the classical models, augmentation is omitted: tabular features
already absorb most of the variation that augmentation would induce.

For tabular and sequence labels alike, **balanced sample weights** are
used during fitting (`sklearn`'s `class_weight="balanced"` and equivalent
PyTorch weights), rather than upsampling. This avoids creating duplicate
windows that would inflate any kind of leakage in cross-validation.

## 4.4 What we deliberately do *not* do

Several common moves in WESAD studies are explicitly avoided here, for
reasons relevant to the workplace-deployment framing:

- **No global normalization.** All standardization is per-subject. Global
  z-scoring across all 1499 windows leaks the population mean into every
  fold and produces optimistic LOSO scores.
- **No SMOTE-style oversampling.** Synthetic minority oversampling
  duplicates information across train/test boundaries when applied
  before splitting; when applied after splitting it inflates training
  loss without adding generalization signal. We use sample weights
  instead.
- **No window-level subject merging.** We never compute features on a
  subject and then re-shuffle the resulting tabular rows across subjects
  before splitting — this is the most common cause of optimistic numbers
  in the WESAD literature, and Section 8 quantifies it directly.

The single most consequential preprocessing decision for honest
benchmarking is the third bullet above. Section 8 turns it into an
ablation.
