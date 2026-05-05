# Empathic Computing with EmoSurv + WESAD

> Multi-modal, subject-independent emotion recognition from keystroke dynamics
> and wearable physiology, with classical and cutting-edge deep models trained
> under a unified Russell's Circumplex label space.

**Version 2.0 (2026)** -- This document supersedes the SWELL-KW documentation
that now lives under [archive/DOCUMENTATION_swell.md](archive/DOCUMENTATION_swell.md).

---

## 1. Project Goal

Build an empathic-computing pipeline that predicts the emotional state of a
user from two very different sensor modalities:

1. **Keystroke dynamics** -- EmoSurv corpus (Yang & Qin, 2021).
2. **Wearable physiology** -- WESAD corpus (Schmidt et al., 2018).

The core requirements captured in the project brief were:

* **Not binary classification.** Discrete multi-class *or* dimensional
  affect, never a two-label stress-vs-no-stress setup.
* **At least three comparable models**, with at least one *deep-learning*
  model that is "cutting edge".
* **Careful preprocessing** plus an **augmentation strategy that makes the
  two datasets fit together** despite their very different sensors.
* **Leverage the local NVIDIA RTX 5070 Ti** when training deep models.
* **Academic-quality documentation** of reasoning, design choices and model
  descriptions.

This document details how each requirement is met.

---

## 2. Repository Layout

```
Empathic/
|-- train.py                     # command-line entry point
|-- DOCUMENTATION.md             # this file
|-- requirements.txt
|-- Dataset/
|   |-- EmoSurv/...              # 4 CSVs (fixed/free typing, frequency, info)
|   `-- WeSad/archive(2)/WESAD/  # S2..S17 subject folders with .pkl + quest csv
|-- src/empathic/                # importable package
|   |-- config.py                # paths, device, label maps, defaults
|   |-- utils.py                 # logging, normalisation, seeding
|   |-- augment.py               # tabular + sequence augmentations
|   |-- evaluation.py            # metric containers and reporters
|   |-- plotting.py              # figure helpers
|   |-- training.py              # LOSO harness
|   |-- data/
|   |   |-- emosurv.py           # keystroke loader and feature engineering
|   |   |-- wesad.py             # physio loader, windowing, questionnaire parsing
|   |   `-- unified.py           # cross-dataset label harmonisation
|   `-- models/
|       |-- classical.py         # RF, LogReg, XGBoost (GPU)
|       `-- conformer.py         # 1-D Conformer (cutting-edge deep model)
|-- results/emotion/<dataset>/<target>/<model>/  # auto-saved metrics + plots
`-- archive/                     # historical SWELL code and earlier iterations
```

---

## 3. Datasets

### 3.1 EmoSurv (keystroke dynamics)

* 123 participants typed prepared paragraphs ("fixed" corpus) and free-form
  responses ("free" corpus) while self-reporting one of five emotions.
* Each keystroke is logged with key code, press/release timestamps and seven
  derived inter-key intervals (`D1U1`, `D1U2`, `D1D2`, `U1D2`, `U1U2`,
  `D1U3`, `D1D3`).
* A supplementary **Frequency** dataset summarises backspace use, arrow-key
  use and total typing time per session.

Native labels: `H` (happy), `C` (calm), `N` (neutral), `A` (angry), `S` (sad).

### 3.2 WESAD (wearable physiology)

* 15 participants (S2..S17, no S12) completed a Trier Social Stress Test
  protocol in a lab setting with a RespiBAN chest band (700 Hz ECG, EDA,
  EMG, respiration, temperature, triaxial acceleration) and an Empatica E4
  wrist band.
* Four experimental conditions plus two reading "pre-condition" segments
  that we ignore. Native pickle labels: `1` baseline, `2` TSST stress,
  `3` amusement, `4` meditation.
* Each subject has a questionnaire CSV with SAM valence/arousal per stage,
  PANAS, STAI and SSSQ items.

We use the chest signals (higher sampling rate, richer dynamics) and read
SAM scores for dimensional analysis.

---

## 4. Why These Two Datasets Need Special Treatment

The two corpora measure *different channels of affect at different
timescales*. EmoSurv rows are individual keystrokes; WESAD rows are 700 Hz
multichannel biosignal samples. Naively concatenating their feature matrices
would produce mostly zeros. We therefore:

1. Keep **dataset-specific feature extractors** so each modality retains its
   most informative descriptors.
2. Harmonise **only the target** across datasets so model performance is
   comparable.
3. Apply **per-subject normalisation** to remove individual baselines that
   would otherwise dominate the features (heart rate, typing speed, etc.).
4. Use **subject-level augmentation** to expand each fold's training set
   without leaking identity cues across splits.

### 4.1 Known weaknesses of the datasets and how we mitigate each

An empirical audit on the full EmoSurv corpus (83 subjects, 2108 windows,
see `archive/inspection/` scripts and §14) exposed four structural
weaknesses that dominate raw accuracy numbers. We cannot fix the data, so
the pipeline must compensate.

| Weakness (measured) | Effect on naive models | Mitigation implemented |
|---------------------|------------------------|------------------------|
| **Severe class skew** -- HVLA = 72% of windows when Neutral is merged into Calm | RF / XGB collapse to the majority class (kappa ~= 0.04). | (a) `DummyClassifier` baseline always reported so collapse is visible. (b) `--emosurv-neutral drop` removes Neutral before windowing. (c) XGBoost now trained with inverse-frequency `sample_weight`; RF/LR already use `class_weight="balanced"`. (d) We quote **balanced accuracy** and **macro recall** alongside accuracy. |
| **Session-constant labels** -- every window inside an EmoSurv session shares the self-reported mood | Window-level accuracy over-counts independent predictions; a subject whose session-level guess is wrong can still look 60% accurate at window level. | **Session-pooled metrics** (`metrics_session.json`). We average window probabilities per `(subject, session)`, then argmax, and report the resulting per-session confusion matrix next to the window-level one. |
| **Tiny per-subject samples** -- median 19 windows/subject, 27/83 subjects carry only *one* quadrant | LOSO folds often test on a subject that shares zero minority-class examples with training. Trees then learn the subject's typing style rather than the emotion signal. | (a) Sequence jitter / scaling / time-warp on the training fold only. (b) **MixUp** (`--mixup-alpha > 0`) cross-subject sequence interpolation. (c) `--emosurv-neutral drop` reduces the number of subjects that look single-class. |
| **Large Conformer vs. small data** -- 1.6M parameters on ~1500 training windows | Easy to memorise training fold, generalise poorly. | New `--deep-arch tiny_tcn` option selects a ~150k-parameter dilated-TCN with heavy dropout, much better matched to the data budget. |
| **Tabular features discard rhythm** -- summary statistics collapse the temporal structure that actually encodes emotional typing. | Classical models hit a floor around kappa 0.17-0.20 even with balanced weights. | (a) `--emosurv-window` / `--emosurv-stride` flags expose sequence length; we default at 35 but run headline results at 70 events so the TCN receptive field (dilations 1/2/4/8) covers two sentences. (b) The deep path already consumes the raw event tensor instead of the summary stats. |
| **Subject identity dominates features** -- typing style / hand size / base speed are 10x larger than the emotion delta. LOSO therefore mostly measures "did the model memorise this person". | Per-subject z-score partially helps but still centres against an arbitrary mix of the subject's emotions. | `--emosurv-neutral baseline` uses each subject's Neutral windows as an explicit per-person calibration: we subtract the subject's Neutral-median feature vector (and per-channel Neutral-mean for sequences) from every non-Neutral window, then classify the **residual**. On the full corpus this lifted RF session-kappa from 0.17 to 0.28 and XGBoost from 0.10 to 0.31. |

These knobs are all controllable from the CLI (Section 12); no code edits
required.

### 4.2 Headline numbers after the mitigations

Full-corpus EmoSurv, 70 subjects with valid LOSO folds, **session-pooled**
metrics (windows aggregated per `(subject, session)` by mean-probability
voting). Each row is the best classical model for that target; the Baseline
column is the `DummyClassifier(most_frequent)` run under the same harness.

| Target               | Chance | Baseline acc / kappa | Best model | Session acc | Session macro-F1 | Session kappa |
|----------------------|:------:|:--------------------:|:-----------|:-----------:|:----------------:|:-------------:|
| 4-class quadrant (drop)     | 0.25 | 0.243 / 0.000 | RandomForest | 0.400 | 0.340 | 0.172 |
| 4-class quadrant (baseline) | 0.25 | 0.243 / 0.000 | **XGBoost**  | **0.486** | **0.495** | **0.311** |
| Binary valence (baseline)   | 0.50 | 0.486 / 0.000 | **RandomForest** | **0.657** | **0.650** | **0.309** |
| Binary arousal (baseline)   | 0.50 | 0.600 / 0.000 | **RandomForest** | **0.714** | **0.689** | **0.383** |

Binary arousal at kappa 0.38 is Landis-Koch "fair" agreement -- small but
real signal on a strict LOSO protocol. Binary valence and 4-class quadrant
sit in the "slight" band; LogisticRegression is uniformly indistinguishable
from chance on this data.

---

## 5. Unified Label Space: Russell's Circumplex

We project both label vocabularies onto the four quadrants of Russell's
(1980) Circumplex, which is the standard shared frame of affective
computing:

| Quadrant | Valence | Arousal | EmoSurv source | WESAD source |
|----------|---------|---------|-----------------|--------------|
| `HVHA`   | high    | high    | Happy           | Amusement    |
| `HVLA`   | high    | low     | Calm / Neutral  | Baseline / Meditation |
| `LVHA`   | low     | high    | Angry           | TSST stress  |
| `LVLA`   | low     | low     | Sad             | (absent)     |

This mapping is encoded as `EMOSURV_LABEL_TO_QUADRANT` and
`WESAD_LABEL_TO_QUADRANT` in [src/empathic/config.py](src/empathic/config.py).

Notes and trade-offs:

* WESAD lacks a negative-low-arousal condition (no "sad"); evaluating on
  WESAD alone therefore produces 3-class confusion matrices over HVHA/HVLA/
  LVHA. The missing quadrant is still reported in the per-class F1 table for
  transparency.
* EmoSurv "Neutral" is the default majority in every session (61% of
  events). Naively merging it into HVLA collapses the task into
  "majority-vs-rest" and destroys minority-class signal. We therefore expose
  a first-class `--emosurv-neutral` switch (see §5.1) so this decision is
  explicit in every experiment.
* Dataset-native labels remain available via `--target native` for direct
  comparison with prior work.

Each EmoSurv code is additionally given a reference (valence, arousal) pair
on the SAM 1..9 scale (`EMOSURV_LABEL_TO_VA`), which enables dimensional
analysis side-by-side with WESAD's self-reported SAM values.

### 5.1 Neutral-handling policy (EmoSurv only)

`--emosurv-neutral` accepts three values:

| Value       | Behaviour | When to use |
|-------------|-----------|-------------|
| `merge` (default, legacy) | Map `N` -> `HVLA` alongside `C`. | Reproducing the original v2 numbers. |
| `drop`      | Remove Neutral events before windowing; evaluate on four balanced-ish classes (H/C/A/S). | Headline 4-class results where we care about actually detecting emotion rather than detecting "nothing happening". |
| `separate`  | Keep Neutral as its own 5th quadrant `NEU` at the origin of the Circumplex. | When you want the model to be able to say "I don't know / no affect" explicitly. |
| `baseline`  | Use Neutral windows as per-subject calibration: compute each subject's median feature vector from their Neutral windows and subtract it from every *non-Neutral* window, then drop Neutral rows. Classify the **residual**. | Keystroke-biometrics-style residual modelling. Cancels inter-subject variance (typing style, hand size, baseline speed) so the classifier sees *how typing changes under emotion* instead of *who is typing*. |

The policy is stored in `bundle.extra["neutral_policy"]` so experiment logs
are self-describing.

---

## 6. Feature Engineering

### 6.1 EmoSurv (tabular + sequence)

We group the raw key events by `(user, split, emotion, session)` and slide a
35-event window with stride 20. For every window we compute:

* Keystroke category rates: backspace, space, alphabetic, digit, special,
  unique-key ratio.
* Typing throughput: keys-per-second derived from the down-timestamps.
* Interval statistics (mean / std / median / Q25 / Q75) for each of the
  seven inter-key intervals plus key-hold duration.
* Rhythm descriptors: coefficient of variation, IQR and first-difference
  std for `D1D2`, `hold_ms` and `U1D2`; pause-rate above 300 ms and
  negative-gap rate (evidence of overlapping keypresses).
* Session-level frequency features merged from `Frequency Dataset.csv`.

In parallel we keep a `(N, 35, 5)` tensor of raw per-event columns
(`hold_ms`, `D1D2`, `U1D2`, `U1U2`, `D1U3`) that feeds the Conformer.

### 6.2 WESAD (tabular + sequence)

We load the chest signals and slide 60 s windows with 30 s stride at 700 Hz.
For every window whose majority label is in `{1, 2, 3, 4}` we compute:

* Per-channel statistical descriptors (mean, std, min, max, median,
  peak-to-peak, first-difference std) for ACC x/y/z, ECG, EDA, EMG,
  respiration and temperature.
* A 240-step average-pooled downsampling of the same window feeds the
  Conformer as a `(N, 240, 8)` tensor.
* Questionnaire-aligned SAM scores per stage provide continuous valence and
  arousal values for dimensional experiments, with a fallback to the
  condition-level average when a window sits between labelled stages.

### 6.3 Sanitisation

The EmoSurv free-typing CSV occasionally stores timestamps that overflow to
`~1.58e12`. We clamp any inter-key interval outside `[-5000, 120000]` ms to
`NaN`, drop negative hold durations above 10 s, and log the number of
discarded values per file. For WESAD we discard windows whose majority label
is outside `{1..4}` (pre/post reading periods, unclassified transitions).

---

## 7. Preprocessing Pipeline

1. **Per-subject normalisation.** EmoSurv features use per-subject
   median/IQR scaling (robust to right-tailed timing artefacts). WESAD
   features use per-subject z-scoring. Both are clipped to `|z| <= 6` to
   stop a single outlier window from dominating gradient steps.
2. **Imputation.** `SimpleImputer(strategy="median")` lives *inside* every
   classical-model pipeline, so imputation statistics are re-fit per LOSO
   fold and never leak to the held-out subject.
3. **Scaling.** Logistic Regression adds a `StandardScaler` in-pipeline for
   numerical stability; trees do not scale.
4. **Per-subject grouping.** `subject_id` is preserved through every
   transform so the LOSO split operates on the original identifier.

---

## 8. Augmentation Strategy

Two families are implemented in [src/empathic/augment.py](src/empathic/augment.py):

### 8.1 Tabular (`augment_tabular`)

* `none` -- pass-through.
* `balance` -- oversample minority classes up to the majority class count.
* `full` -- balance, then add per-row Gaussian jitter (sigma = 0.02) plus
  random feature masking (p = 0.03) to simulate sensor dropout.

### 8.2 Sequence (`augment_sequences`)

Following Um et al. (2017):

* **Jitter** (sigma = 0.03) -- sensor noise simulation.
* **Scaling** (sigma = 0.10) per channel -- gain drift simulation.
* **Time-warping** with cubic-spline warped anchors (sigma = 0.20, 4 knots)
  -- compensates for inter-subject speed differences.

Augmentations are only applied to the training fold. The test fold is left
untouched so LOSO scores remain trustworthy.

### 8.3 MixUp (sequence deep models)

For the deep models we additionally expose `--mixup-alpha` to enable MixUp
(Zhang et al., 2018). Given two random training examples
$(x_i, y_i)$ and $(x_j, y_j)$ and $\lambda \sim \text{Beta}(\alpha, \alpha)$,
MixUp feeds a linear interpolation

$$\tilde x = \lambda x_i + (1-\lambda) x_j, \quad\quad \mathcal{L} = \lambda\,\text{CE}(\hat y, y_i) + (1-\lambda)\,\text{CE}(\hat y, y_j)$$

through the network. Because EmoSurv has a handful of subjects per minority
quadrant, MixUp generates *subject-crossing* virtual examples that push the
decision boundary away from subject idiosyncrasies without fabricating new
class labels. Implemented in `augment.mixup_batch` and wired through
`_train_deep_fold`.

### 8.3 Cross-dataset "fit together"

Because the raw features are incommensurable, we unify at the label level
(Section 5) and rely on per-subject normalisation and sequence warping to
make the two corpora behave like a single distribution from the model's
perspective. Feature-level concatenation was rejected after inspection:
EmoSurv and WESAD feature matrices have essentially zero shared columns and
combining them would require filling an overwhelming majority of each row
with zeros, which our experiments showed acts as destructive masking.

---

## 9. Models

All models target the same unified quadrant labels so their numbers are
directly comparable. The classical suite is trained on the tabular feature
matrix, the deep suite on the aligned per-window sequence tensors.

### 9.1 Baseline (majority-class `DummyClassifier`)

A reference model that always predicts the most-frequent training-fold
class. Cohen's kappa is 0 by construction. Any serious model must beat its
accuracy, its macro-F1 *and* its balanced accuracy. Having the baseline
run under the same LOSO harness makes the class-imbalance story explicit
in every comparison plot.

### 9.2 Random Forest

`sklearn.ensemble.RandomForestClassifier` with 400 trees, balanced class
weights, `min_samples_leaf=2`, `n_jobs=-1`. Non-linear tabular baseline.

### 9.3 Logistic Regression

Multinomial softmax regression (`solver="lbfgs"`) behind a
`StandardScaler` and `SimpleImputer`, `class_weight="balanced"`. Linear
sanity check for how much non-linear structure the trees actually add.

### 9.4 XGBoost (GPU-accelerated, class-weighted)

Gradient-boosted decision trees with `tree_method="hist"` and
`device="cuda"` when a CUDA device is available. 300 boosting rounds,
depth 6, learning rate 0.08, subsample 0.9, colsample 0.9. Unlike RF/LR,
XGBoost's scikit API does not accept `class_weight`; we therefore compute
per-sample weights

$$w_i = \frac{N}{K \cdot n_{y_i}}$$

(inverse class frequency, $K$ = number of classes) and pass them in via
the `clf__sample_weight` Pipeline kwarg inside every LOSO fold. This is
what lets XGBoost stop collapsing to the majority quadrant on EmoSurv.

### 9.5 Conformer (cutting-edge deep model)

The Conformer (Gulati et al., Interspeech 2020) alternates a multi-head
self-attention branch with a depthwise convolution module inside every
block:

* The **self-attention** branch captures long-range dependencies such as
  the slow EDA/ECG drift after the TSST stressor or the global pacing of a
  typing session.
* The **depthwise convolution** branch captures local motifs such as QRS
  complexes, respiratory cycles and keystroke bursts.

Our implementation is a compact 4-block network with `d_model=128`, 4
attention heads, kernel size 15, Swish activations, SiLU/GLU pointwise
projections, Batch Normalisation on the depthwise path and LayerNorm after
every sublayer. A stride-2 1-D-convolutional stem downsamples long WESAD
windows before the attention stack so compute stays manageable. Output
features are temporally averaged and projected to logits. ~1.6M parameters.

### 9.6 Tiny TCN (small, regularised deep alternative)

Enabled via `--deep-arch tiny_tcn`. A 4-block dilated-causal Temporal
Convolutional Network (Bai et al., 2018) with 64 channels per block,
kernel 5 and exponentially growing dilation (1, 2, 4, 8). With
`--emosurv-window 70` the receptive field covers the full two-sentence
EmoSurv window while keeping the model to ~150k parameters -- an order
of magnitude smaller than the Conformer and much better-matched to the
~400-window EmoSurv training budget. Heavy dropout (0.3 inside blocks
+ 0.3 at the head) compensates for the small data regime. It consumes
the same Neutral-residual sequence tensor as the baseline-residual
classical path, so any remaining signal is temporal rhythm rather than
per-subject identity. On a CPU-only machine the full 70-fold LOSO sweep
takes ~50 minutes; on CUDA it is a couple of minutes.

### 9.7 Shared deep-training settings

See [src/empathic/training.py](src/empathic/training.py):

* Loss: `CrossEntropyLoss` with label smoothing 0.05 and inverse-frequency
  class weights computed per fold.
* Optional MixUp on the training batch controlled by `--mixup-alpha`.
* Optimiser: AdamW (weight decay 1e-4).
* LR schedule: cosine annealing over `--epochs` (default 40).
* Gradient clipping at 1.0.
* Early stopping on training-loss plateau (patience 8).
* Data loaded with `pin_memory=True` when on CUDA.

---

## 10. Evaluation Protocol

We run **Leave-One-Subject-Out** cross-validation for every model on every
dataset. Each fold trains on $N-1$ subjects and tests on the held-out
subject, which is the strictest standard in affective computing and the
only one that reflects real deployment (a new user walks up, no fine-tuning).

Reported metrics per fold (see
[src/empathic/evaluation.py](src/empathic/evaluation.py)):

* Accuracy *and* **balanced accuracy** (mean per-class recall) -- the
  second is the honest number under imbalance.
* **Macro F1**, **macro recall**, weighted F1.
* **Cohen's kappa** (chance-adjusted agreement).
* Per-class F1 and support.
* Confusion matrix.

### 10.1 Window-level vs session-level metrics

Both corpora use **session-level** ground truth: every window inside an
EmoSurv typing session, or every window inside a WESAD TSST stage, shares
the same label. Window-level accuracy therefore over-counts how many
*independent* predictions the model has actually made.

For each model and dataset we additionally report **session-pooled**
metrics: for every unique `(subject_id, session_id)` we average the
predicted class probabilities across that session's windows and take the
argmax as the session prediction. The file layout is:

```
results/emotion/<dataset>/<target>/<model>/
    metrics_overall.json          # window-level
    metrics_session.json          # session-pooled (new)
    confusion_matrix.png          # window-level
    confusion_matrix_session.png  # session-pooled (new)
    per_subject.csv
```

Session-pooled numbers are what we should quote when comparing to the
EmoSurv / WESAD literature, which always scores per session.

### 10.2 Binary valence / arousal projection

With `--target valence` or `--target arousal` the quadrant label of every
window is collapsed onto one of the two circumplex axes (`HVHA/HVLA -> HV`,
`LVHA/LVLA -> LV`; analogous for arousal). Neutral rows under
`--emosurv-neutral baseline` are dropped after calibration. This gives a
two-class LOSO protocol directly comparable to published valence/arousal
dichotomy results -- and in practice it is where the residual signal is
cleanest (see §4.2). Binary chance is 0.5; the `Baseline` majority-class
dummy reports the class prior (~0.49 valence, ~0.60 arousal on EmoSurv).

Per-subject metrics are aggregated to mean / std (`aggregate_fold_metrics`)
and the concatenated-prediction confusion matrix is stored alongside the
model-comparison bar plot.

Outputs are written to `results/emotion/<dataset>/<target>/<model>/` and a
cross-model `summary.csv` plus `model_comparison.png` at the parent dir.

---

## 11. Hardware

The training harness reads `torch.cuda.is_available()` and routes the deep
model plus XGBoost to the CUDA device when present. On the RTX 5070 Ti
development box this delivered an order-of-magnitude speed-up on the
Conformer and kept the classical pipeline GPU-accelerated end-to-end.

The `--cpu` flag forces the pipeline onto CPU for testing on machines
without a CUDA-capable PyTorch build.

---

## 12. Usage

### Full training

```
python train.py --datasets emosurv wesad --target quadrant
```

Runs LOSO with Random Forest, Logistic Regression, XGBoost and Conformer on
both datasets. Results are saved under `results/emotion/`.

### Smoke test

```
python train.py --datasets emosurv --quick --epochs 5
```

Restricts each dataset to the first 10 (EmoSurv) / 6 (WESAD) subjects and
drops Conformer epochs to five so the whole suite finishes in a couple of
minutes on a laptop.

### Common switches

| Flag                   | Effect |
|------------------------|--------|
| `--datasets`           | `emosurv`, `wesad` or both. |
| `--target`             | `quadrant` (4-class), `native` (dataset-specific), or `valence` / `arousal` (binary axes derived from the quadrant). |
| `--deep-arch`          | `conformer` (default, ~1.6M params) or `tiny_tcn` (~150k params, heavier regularisation). |
| `--mixup-alpha`        | MixUp strength for deep models. `0` disables, `0.2` is a sensible start. |
| `--emosurv-neutral`    | `merge` (legacy) / `drop` / `separate` / `baseline` -- see §5.1. `baseline` is the current headline setting. |
| `--emosurv-window`     | Events per EmoSurv window (default 35; we ship headline numbers at 70). |
| `--emosurv-stride`     | Event stride between windows (default 20; use 35 with `--emosurv-window 70`). |
| `--augment-tabular`    | `none` / `balance` / `full` for classical models. |
| `--augment-sequences`  | Same options for the deep model. |
| `--emosurv-norm`       | `robust` (default), `zscore` or `none`. |
| `--wesad-norm`         | `zscore` (default) or `none`. |
| `--epochs`             | Deep-model training epochs. |
| `--batch-size`, `--lr` | Deep-model optimiser settings. |
| `--no-deep`            | Skip the deep model (faster classical-only sweeps). |
| `--cpu`                | Force CPU execution. |

### Recommended recipes

**Current headline recipe (baseline-residual, 4-class, 70-event windows):**

```
python train.py --datasets emosurv --emosurv-neutral baseline \
    --emosurv-window 70 --emosurv-stride 35 \
    --deep-arch tiny_tcn --mixup-alpha 0.2 --epochs 40 --target quadrant
```

**Binary valence / arousal (easier 2-class problem on the same residuals):**

```
python train.py --datasets emosurv --emosurv-neutral baseline \
    --emosurv-window 70 --emosurv-stride 35 --no-deep --target valence
python train.py --datasets emosurv --emosurv-neutral baseline \
    --emosurv-window 70 --emosurv-stride 35 --no-deep --target arousal
```

**Honest 4-class EmoSurv headline without Neutral calibration:**

```
python train.py --datasets emosurv --emosurv-neutral drop \
    --deep-arch tiny_tcn --mixup-alpha 0.2 --epochs 40
```

**5-class EmoSurv keeping Neutral as its own quadrant:**

```
python train.py --datasets emosurv --emosurv-neutral separate \
    --deep-arch conformer --epochs 40
```

**Both datasets, unified quadrants (for cross-dataset comparison):**

```
python train.py --datasets emosurv wesad --emosurv-neutral drop \
    --deep-arch conformer --epochs 40
```

### Programmatic API

```python
from empathic.data import build_bundles
from empathic.training import run_experiment

bundles = build_bundles(["emosurv", "wesad"], emosurv_neutral_policy="drop")
for name, bundle in bundles.items():
    run_experiment(bundle, target_kind="quadrant",
                   deep_arch="tiny_tcn", mixup_alpha=0.2)
```

---

## 13. References

* Bai, S., Kolter, J. Z. and Koltun, V. "An Empirical Evaluation of Generic
  Convolutional and Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.
* Gulati, A. et al. "Conformer: Convolution-augmented Transformer for
  Speech Recognition." *Interspeech 2020.*
* Russell, J. A. "A Circumplex Model of Affect." *Journal of Personality
  and Social Psychology*, 39(6), 1161-1178, 1980.
* Schmidt, P. et al. "Introducing WESAD, a Multimodal Dataset for Wearable
  Stress and Affect Detection." *ICMI 2018.*
* Um, T. T. et al. "Data Augmentation of Wearable Sensor Data for
  Parkinson's Disease Monitoring using Convolutional Neural Networks."
  *ICMI 2017.*
* Vaswani, A. et al. "Attention Is All You Need." *NeurIPS 2017.*
* Yang, J. and Qin, Y. "EmoSurv: A Database of Keystroke Dynamics for
  Emotion Recognition." Mendeley Data, 2021.
* Zhang, H., Cisse, M., Dauphin, Y. N. and Lopez-Paz, D. "mixup: Beyond
  Empirical Risk Minimization." *ICLR 2018.*

---

## 14. Reproducibility Notes

* Every fold seeds Python, NumPy and PyTorch (including CUDA) via
  `empathic.utils.set_seed`.
* Feature pipelines instantiate fresh estimators per fold (`factory`
  callbacks in [src/empathic/training.py](src/empathic/training.py)).
* Data augmentation uses per-fold-seeded `np.random.default_rng`.
* All preprocessing statistics are fit on the training fold only.
* Raw artefact thresholds, window sizes and default hyper-parameters are
  captured as frozen dataclasses in
  [src/empathic/config.py](src/empathic/config.py).
