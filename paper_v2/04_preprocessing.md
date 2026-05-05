# 04 — Preprocessing & feature extraction

## 4.1 Per-sensor cleaning

Each sensor stream is cleaned with modality-appropriate filters before
windowing.

- **ECG (Polar, ~130 Hz).** 0.5–40 Hz band-pass (4th-order
  Butterworth), R-peak detection via Pan-Tompkins, RR-interval
  series with $200$ ms / $300$ ms ectopic-beat correction.
- **BVP (Empatica, 64 Hz).** 0.5–8 Hz band-pass; systolic peak
  detection; IBI series.
- **HR (Polar, 1 Hz).** Linear interpolation of dropped samples;
  upsampled to 32 Hz with hold-last semantics.
- **EDA (Empatica, 4 Hz).** Median filter, low-pass at 1 Hz,
  cvxEDA-style phasic / tonic decomposition.
- **TEMP (Empatica, 4 Hz).** Outlier clipping at $\pm 4$ SD;
  smoothing.
- **ACC (Empatica, 32 Hz).** Detrend per axis, magnitude derived as
  $\sqrt{x^2 + y^2 + z^2}$.
- **EEG (Muse, 256 Hz).** 1–45 Hz band-pass; 50 Hz notch; per-channel
  bad-segment masking ($> 100~\mu\text{V}$ amplitude).

After cleaning, all sensors are resampled onto a common 32 Hz grid
spanning the overlap of all sensor timestamp ranges for that
`(subject, session)` pair. Windows in which any required cardiac
channel is entirely missing are rejected; flat-signal windows
(constant-value across the entire window on every channel) are
rejected.

## 4.2 Windowing

A 60-second sliding window with 30-second stride yields the
$60 \times 32 = 1920$-sample raw segment. For deep models the segment
is downsampled to a 12 × 240 tensor. For classical models the segment
is summarised into a 149-dimensional feature vector (§4.3).

## 4.3 Tabular features (149 dims)

Features carry a sensor prefix so the §7 ablation can isolate them
directly.

- **ECG (17).** Heart-rate mean / median / std / min / max, RMSSD,
  SDNN, pNN50, pNN20, LF / HF / LF/HF, sample entropy, approximate
  entropy, mean / median / std absolute first differences.
- **BVP (17).** As ECG but on the systolic-peak IBI series, with two
  pulse-amplitude descriptors instead of approximate entropy.
- **HR (7).** Mean, median, std, min, max, slope, range over the window.
- **EDA (14).** Tonic mean / std / slope, phasic peak count / amplitude
  mean / amplitude std / area, full-window mean / std / first / last,
  rise-time mean / std, recovery-time mean.
- **TEMP (9).** Mean, std, min, max, slope, range, first, last,
  median.
- **ACC (25).** Per-axis (xyz) mean / std / range / first-difference
  std (12), magnitude mean / std / range / first-difference std (4),
  step-count proxy (1), entropy of magnitude histogram (1),
  high-frequency energy (1), low-frequency energy (1), signal-magnitude
  area (1), tilt-angle mean / std (2), zero-crossing rate (1),
  freezing-of-gait index (1).
- **EEG (60).** Per channel (4 channels: TP9, AF7, AF8, TP10): delta
  (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma
  (30–45 Hz) absolute band powers (5 × 4 = 20), relative band powers
  (5 × 4 = 20), spectral entropy per channel (4), Hjorth activity /
  mobility / complexity per channel (12), front-back asymmetry index
  per band (4 × 1 = 4).

## 4.4 Per-subject baseline correction

For each subject we compute the median of every tabular feature on
their `b1`/`b2`/`b3` baseline windows and subtract it from the
subject's call-session windows. Sequence tensors are *not*
baseline-corrected; the deep learners can absorb subject baselines
through their normalisation layers if they choose to.

## 4.5 Normalisation policy

We use **per-subject train-only $z$-scoring** for the classical
models. Per-fold normalisation parameters are computed from the
training subjects and applied to both the held-out subject's training
windows (none, in LOSO) and test windows. There is no normalisation
information leakage from test to train.

## 4.6 What this preprocessing does *not* do

- It does **not** apply data augmentation to the tabular features for
  classical training (mixup is restricted to sequence inputs of the
  deep models).
- It does **not** balance the class distribution by resampling. We
  rely on class-weighted losses and inverse-frequency sample weights.
- It does **not** drop subjects. Even subjects with single-class
  valence remain in the corpus; their LOSO folds are skipped at the
  training stage when binary classification is impossible (the only
  consequence is a slight reduction in the number of valid LOSO folds
  for the valence target — see §5).
