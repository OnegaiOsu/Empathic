# 03 — Model array

## 3.1 Why a survey rather than a single architecture

A workplace stress monitor is a deployment, not a research artefact. The
relevant question is therefore not "what is the highest score
attainable?" but "what is the smallest, cheapest model that meets the
deployment constraint, and how confident are we that the score holds for
a new employee?". With that frame, a survey of model families is more
informative than a single tuned architecture. We chose seven members
across three categories.

### Tabular classical baselines (interpretable, cheap)

These three models are the operationally realistic candidates for a
production system. They train and infer in milliseconds on commodity
hardware, lend themselves to feature-attribution explanations, and have
extensive existing tuning experience in the WESAD literature.

- **Random Forest (RF).** A bagging ensemble of decision trees.
  Strong on tabular features with mixed scales. Robust to redundant
  features and to the heavy class imbalance present in WESAD.
- **Logistic Regression (LR).** Linear model with $\ell_2$ regularization
  and balanced class weights. Provides a calibrated decision boundary,
  which is useful for downstream thresholding (e.g. "alert when
  $P(\text{stress}) > 0.7$ for five consecutive minutes").
- **XGBoost (XGB).** Gradient-boosted trees. State of the art on most
  tabular benchmarks; included to test whether boosting captures
  interactions that bagging misses.

### Sequence baselines (representative inductive biases)

These three architectures span the canonical inductive-bias choices for
short physiological time series. They give us a controlled answer to the
question "how much can we gain by modelling the raw waveform rather
than its hand-crafted summary?".

- **CNN1D.** Stacked 1D convolutions over the eight-channel input. Local
  pattern detector; cannot natively model long-range dependencies.
- **TinyTCN.** A small temporal convolutional network with dilated
  causal convolutions (Bai et al., 2018). Provides large receptive fields
  cheaply; included as the modern successor to plain CNN1D.
- **BiLSTM.** Bidirectional long short-term memory. Recurrent inductive
  bias; expensive but historically dominant on physiological sequences.

### Modern attention model (current research frontier)

- **Conformer.** Convolution-augmented transformer (Gulati et al., 2020),
  originally proposed for ASR but increasingly adopted in physiological
  signal modelling because it interleaves local convolution with global
  attention. We include it as our representative of the
  "what does state-of-the-art architecture buy us?" hypothesis. The
  Conformer is also the deep model that participates in our
  late-fusion ensemble (Section 5), reflecting recent work that finds it
  to be the strongest single deep baseline on multiple physiological
  tasks (Lee et al., 2022).

### Late-fusion ensembling

Each deep architecture is exposed in two flavours: a pure sequence
classifier and a **late-fusion variant** that concatenates the deep
sequence embedding with the 89-dimensional tabular feature vector before
the final classification head. The late-fusion variants are denoted
`*_fusion` throughout (e.g. `conformer_fusion`).

The classification ensemble is a temperature-calibrated average over the
three classical models plus `conformer_fusion`. Calibration uses a
per-member temperature scalar fitted on a held-out validation slice via
NLL minimization on a 0.5–4.0 grid (Guo et al., 2017).

## 3.2 Design rationale: workplace deployment lens

This array reflects a deliberate hierarchy of *deployability*:

1. **Tier 1 (production today).** RF, LR, XGB. Fit in a few megabytes,
   inference in microseconds, easy to retrain on a per-employee basis.
2. **Tier 2 (research candidates).** CNN1D, TinyTCN, BiLSTM. Plausible
   for an edge device but with non-trivial integration cost; included
   so we can quantify whether they justify the engineering investment.
3. **Tier 3 (state-of-the-art ceiling).** Conformer. Establishes how much
   a maximally complex model recovers; included so a negative result
   (Section 6) is informative.

The same logic motivates the **survey-of-architectures** rather than
"survey-of-Transformers" framing common in current affective-computing
literature. From the perspective of an HR or occupational-health team
considering deploying such a system, the relevant comparison is not
"is Conformer better than BiLSTM?" but "is *any* deep model better than
a Random Forest on hand-crafted features by enough to justify a GPU?".
Section 6 answers that question empirically.
