# 03 — Models

We benchmark eleven learners under leave-one-subject-out cross-validation
on four targets. They fall into three families.

## 3.1 Classical (tabular features only)

The classical models consume the 149-dimensional tabular feature vector
described in §4. None of them sees the raw 12-channel sequence.

- **Random Forest** (Breiman, 2001). 500 trees, balanced class weights, default sklearn
  hyper-parameters otherwise. Inverse-frequency sample weights are used
  as a backup when class weights cannot fully address fold-level
  imbalance.
- **Logistic Regression.** $L_2$ penalty, balanced class weights,
  `lbfgs` solver, max 2000 iterations. Standardised via per-subject
  $z$-score on the train fold only.
- **XGBoost** (Chen & Guestrin, 2016). GPU histogram tree booster (`device="cuda"`,
  `tree_method="hist"`), `max_depth=6`, `n_estimators=500`,
  `learning_rate=0.05`, inverse-frequency sample weights. The
  re-evaluation script in §6 does *not* pass class weights to XGBoost
  and the model collapses to majority on binary targets there; this is
  documented as a script-level limitation.

## 3.2 Deep fusion architectures (sequence + tabular)

All deep models use a "fusion head" pattern: a sequence encoder produces
an embedding of the 12 × 240 input, which is concatenated with the
tabular feature vector and fed through a two-layer MLP classifier. They
are trained for 100 epochs with AdamW (lr 1e-3, weight decay 1e-2),
mixup ($\alpha = 0.2$) on the input, and class-balanced sampling. Early
stopping on within-train held-out subject loss.

- **CNN1D fusion.** Three 1D convolutional blocks
  (kernel 7 → 5 → 3, channels 64 → 128 → 128), global average pool, MLP
  head. ~250k params.
- **BiLSTM fusion** (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal,
  1997). Two-layer bidirectional LSTM, hidden 128, dropout 0.3,
  last-state read-out. ~600k params.
- **TinyTCN fusion** (Bai et al., 2018). Dilated causal TCN with
  dilations $1, 2, 4, 8$, channels 64, dropout 0.3. ~150k params.
- **Conformer fusion** (Gulati et al., 2020). Six conformer blocks
  (4 heads, 64 model dim, conv kernel 31), depthwise-separable
  convolution + relative multi-head attention. ~1.6 M params.
- **Multi-stream fusion.** One CNN1D branch per modality group (cardiac,
  EDA, TEMP, ACC, EEG), each producing a 64-d embedding; embeddings are
  attention-pooled and concatenated with tabular features.
- **DANN-Conformer.** Conformer fusion with a Domain Adversarial Neural
  Network head (Ganin et al., 2016) using subject id as the adversarial
  domain. Gradient reversal $\lambda$ ramped 0 → 0.5 over training.

## 3.3 Self-supervised pre-trained encoder

- **TS-TCC** (Eldele et al., 2021). Time-Series representations via
  Temporal-Contrastive Coding. The encoder is pre-trained
  unsupervised on the *training-fold* sequences only (no test-fold
  leakage), then fine-tuned with a linear classifier head on the same
  fold.

## 3.4 Soft-vote ensemble

A class-prior-aligned soft-vote ensemble averages the probability
outputs of (Random Forest, Logistic Regression, XGBoost,
Conformer fusion, TinyTCN fusion, multi-stream fusion, TS-TCC). The
DANN-Conformer is omitted from the current ensemble because its
result key was logged inconsistently with the lookup name; this is a
logging-key artefact in `runs/emowork/train_all.py` that has been
fixed for future runs but does not affect the standalone DANN row.
The fix is verbatim:

```python
ENSEMBLE = (
    "RandomForest", "LogisticRegression", "XGBoost",
    "conformer_fusion", "tiny_tcn_fusion", "multistream_fusion",
    "DANN_Conformer", "TSTCC",
)
```

## 3.5 Why this set of models

The classical trio represents the strongest non-deep baselines on
tabular physiology features in the affect literature. The deep family
covers the four canonical sequence priors (locality via CNN, recurrence
via LSTM, dilated causality via TCN, attention via Conformer), plus a
modality-aware multi-stream variant and a domain-adversarial variant
that should — in principle — be the right tool for subject-shift in
LOSO. TS-TCC adds a self-supervised pre-training arm to test whether
unlabelled within-subject signal helps generalise across subjects.

If multimodal fusion at $n = 31$ is going to outperform a
single-sensor classical baseline, *one of these eleven learners must do
it*. §5 reports whether that happens.
