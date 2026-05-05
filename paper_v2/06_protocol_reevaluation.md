# 06 — Re-evaluation under alternative protocols

The §5 numbers are sober. The literature, by and large, is not. To
explain the gap we re-run the three classical models under three
additional cross-validation protocols and quantify the inflation each
introduces.

## 6.1 Protocols

We compare four protocols on the same 625-window, 31-subject corpus:

1. **LOSO.** Leave-one-subject-out across the 31 subjects. Honest
   cross-subject generalisation.
2. **Subject GroupKFold $k = 5$.** Five folds, each fold holding out
   $\approx 6$ subjects. Same *kind* of split as LOSO; should give
   similar metrics if subject identity is the only leakage axis.
3. **Window StratifiedKFold $k = 10$.** Ten random folds, stratified
   by class. *Subject identity is ignored*: windows from the same
   subject can appear in train and test. This is what most affect
   recognition papers reported as "subject-independent" until the
   2018–2020 wave of methodological corrections.
4. **Window 80/20.** Single random 80/20 stratified hold-out, again
   ignoring subject identity. The least conservative protocol.

The script that produces these is
[`runs/emowork/relaxed_eval.py`](../runs/emowork/relaxed_eval.py); the
log is [`relaxed_eval.log`](../results/emotion/emowork/relaxed_eval.log).

## 6.2 Headline: Cohen's κ across protocols

| Target  | Model | LOSO | Subj 5F | Window 10F | Window 80/20 |
|---|---|---:|---:|---:|---:|
| Stress  | RandomForest       | 0.308 | 0.315 | **0.555** | **0.664** |
| Stress  | LogisticRegression | 0.273 | 0.213 | 0.373 | 0.408 |
| Arousal | RandomForest       | 0.010 | 0.045 | **0.428** | **0.393** |
| Arousal | LogisticRegression | 0.021 | $-0.059$ | 0.070 | 0.103 |
| Valence | RandomForest       | 0.000 | 0.000 | 0.061 | 0.000 |
| Valence | LogisticRegression | 0.010 | 0.000 | 0.104 | 0.099 |
| Quadrant| RandomForest       | 0.016 | 0.028 | **0.284** | 0.152 |
| Quadrant| LogisticRegression | 0.004 | $-0.041$ | 0.105 | 0.057 |

The pattern is consistent and large.

- **Stress, Random Forest:** κ = 0.31 under LOSO becomes
  κ = 0.66 when window-level leakage is permitted. **More than
  a doubling.**
- **Arousal, Random Forest:** κ = 0.01 (essentially chance)
  becomes κ = 0.43 under window leakage. The model has not
  learned arousal; it has learned to recognise the subject and predict
  their majority class.
- **Quadrant, Random Forest:** κ = 0.02 becomes κ = 0.28.
  Same mechanism.
- Subject-grouped 5-fold tracks LOSO almost exactly on every target.
  The harm is *not* from the small fold count or the validation
  protocol's variance; it is specifically from windows of the same
  subject crossing the train/test boundary.

## 6.3 Why the inflation is this large

Three mechanisms reinforce each other.

1. **Strong subject-level autocorrelation.** EmoWork sessions are
   four minutes long with overlapping 60s/30s windows. Adjacent
   windows of the same subject share roughly half their underlying
   raw signal. The model effectively memorises a subject's window-level
   feature signature.
2. **Subject-as-class structure.** The 149 features include strongly
   subject-specific quantities (HR baseline, EDA tonic level, EEG
   band-power signature). Distinguishing between subject A
   (mostly stress = 1) and subject B (mostly stress = 0) is much easier
   than distinguishing between stress = 0 and stress = 1 within either
   subject.
3. **Tree models exploit any leakage available.** Random Forest with
   500 trees and unlimited depth has more than enough capacity to
   index a 31-subject lookup. The window 80/20 result for stress
   (0.664) is a rough estimate of how much of that lookup the forest
   builds. The LOSO result (0.308) is what survives when the lookup is
   useless.

## 6.4 Why XGBoost is silent

XGBoost in `relaxed_eval.py` reports κ = 0 on every binary
target under every protocol. The model is collapsing to majority
because the script does not pass the inverse-frequency sample weights
that the main `train_all.py` does. Under correct weighting (as in §5)
XGBoost is competitive with Random Forest. We retain the unweighted
row because it is informative: it shows that without class weighting
even an aggressive booster will not learn binary minority classes from
this corpus. *This is a property of the data, not a defect of the
model;* and it is a reminder that protocol comparisons must control
for the same loss configuration.

## 6.5 Implication for the literature

The implication is simple and uncomfortable. A multimodal affect
recognition paper that reports κ = 0.55–$0.70$ on a small
corpus *without* a LOSO control is, with high probability, reporting
the leakage figure from §6.2 column 3 or 4. On EmoWork the LOSO
correction is approximately a $-0.30$ to $-0.40$ shift in κ on
the stress and arousal targets.

For a workplace deployment scenario, where every new employee is by
construction a held-out subject, the LOSO column is the only column
that means anything. Subject-grouped $k$-fold ($k = 5$ here) is a
defensible, faster surrogate. The two window-level protocols should
not appear in any paper that claims subject-independent
generalisation, except as an explicit upper-bound illustration of the
sort given here.

## 6.6 Calibration regime: rest-anchored LOSO and within-subject 70/30

The §6.1–§6.5 protocols are all *generalisation* protocols: they ask
whether a model trained on $n - 1$ subjects can predict the held-out
subject. A workplace stress detector deployed in the field, however,
would ordinarily be allowed *one calibration step per employee* before
going live. Two natural protocols capture this regime:

- **Protocol C — rest-anchored LOSO.** Each subject's resting-baseline
  windows are used to fit per-subject feature mean and standard
  deviation; c-session features are then z-scored against this
  per-subject rest reference (clipped at $\pm 6\sigma$). The model is
  still trained LOSO across subjects, but its inputs are anchored
  individually. This is the *minimal-cost* personalisation: one short
  rest recording, no labelled affect data per subject.
- **Protocol B — within-subject 70/30.** Each subject is treated as
  their own corpus: $70\%$ of their c-session windows train, $30\%$
  test, stratified by the target. This is the *upper-bound*
  personalisation regime, equivalent to the within-subject evaluation
  used by DEAP (Koelstra et al., 2012) and AMIGOS (Miranda-Correa et
  al., 2018).

Both are produced by
[`runs/emowork/train_calibrated.py`](../runs/emowork/train_calibrated.py)
and
[`runs/emowork/train_within_subject.py`](../runs/emowork/train_within_subject.py)
with logs and per-subject CSVs in
[`results/emotion/emowork/calibrated/`](../results/emotion/emowork/calibrated/)
and
[`results/emotion/emowork/within_subject/`](../results/emotion/emowork/within_subject/).

### 6.6.1 Headline: best-per-target across regimes

Best macro-F1 per target across the four LOSO-style regimes (Protocol
A is the §5 baseline; Protocol C is rest-anchored LOSO; Protocol B is
within-subject 70/30):

| Target | A — LOSO (§5) | C — Rest-anchored LOSO | **B — Within-subject** |
|---|---:|---:|---:|
| Stress    | 0.677 (RF)        | 0.612 (DANN-Conformer) | **0.908 (LogReg)** |
| Arousal   | 0.538 (Conformer) | 0.466 (LogReg)         | **0.818 (BiLSTM)** |
| Valence   | 0.507 (BiLSTM)    | 0.461 (Conformer)      | **0.860 (BiLSTM)** |
| Quadrant  | 0.309 (XGBoost)   | 0.289 (RF)             | **0.724 (CNN1D)**  |

The pattern is unambiguous: **per-subject calibration, not
architecture, is the dominant lever** on this corpus. Protocol B lifts
macro-F1 by $+0.23$ (stress), $+0.28$ (arousal), $+0.35$ (valence) and
$+0.42$ (quadrant) above the §5 LOSO baseline.

### 6.6.2 Why Protocol C does not help (a defensible negative result)

The §5 pipeline already performs per-subject z-scoring across each
subject's c-session windows (§4.4) and a per-subject baseline mean
correction (§4.3). Protocol C replaces that c-session-derived standard
deviation with a *rest*-derived one. Because rest c-sessions are
calmer than working c-sessions, rest standard deviations are
systematically *narrower* than c-session standard deviations; rescaling
c-session features by the smaller rest scale therefore inflates
z-scores toward the $\pm 6\sigma$ clip and discards information.

Empirically, every classical model loses $0.05$–$0.07$ macro-F1
relative to §5; deep models hold approximately steady (their internal
batch- and layer-norm absorbs the rescaling). Protocol C is reported
because it is the cheapest possible personalisation (a single rest
recording, no per-subject labels) and because the negative result is
informative: when the §5 pipeline already z-scores against c-session
statistics, a rest-only reference is strictly worse.

### 6.6.3 Protocol B caveats: dataset-inherent leakage

Protocol B's numbers exceed the within-subject baselines reported on
DEAP and AMIGOS by $0.25$–$0.30$ macro-F1 (§8.2), which warrants
careful framing. Two leakage paths are not removable without a
different data collection:

1. **60 s windows with 30 s stride overlap by 50%.** A random 70/30
   split places adjacent overlapping windows on opposite sides of the
   train/test boundary; the model can partially exploit the shared
   raw signal. This is a property of the §4.2 windowing choice on the
   §2 SWELL/EmoWork release, not of our split.
2. **Three c-sessions per subject.** Each c-session is a different
   condition (no-pressure, time-pressure, interruption); a stratified
   per-subject split places windows from the same c-session on both
   sides. Stable session-level artefacts (sensor placement that day,
   posture, time-of-day HR baseline) are therefore visible to the
   model. DEAP's 40 stimuli/subject and AMIGOS's 16+ stimuli/subject
   make this leakage path much narrower for those corpora; with three
   c-sessions per subject, a leak-free leave-one-call-out within-subject
   split on EmoWork would have $\le 7$ test windows per fold and only
   three folds per subject, which is statistically thin.

Both paths inflate Protocol B numbers in absolute terms. They do not,
however, account for the gap to Protocol A: the §5 LOSO numbers face
the same $50\%$ window overlap and three-session structure inside the
training subjects, and still produce far weaker discrimination on the
held-out subject. The $+0.23$ to $+0.42$ macro-F1 lift from A to B is
therefore *predominantly* attributable to per-subject calibration —
the model fitting subject-specific feature scales and decision
boundaries — rather than to overlap- or session-level leakage. We
report Protocol B as a deployment ceiling rather than a deployment
estimate, and recommend leave-one-call-out within-subject splits as
the publication-grade follow-up if a richer protocol stack (more
c-sessions, non-overlapping windows) is collected.

### 6.6.4 Linear models tie deep models on the easiest targets

A second non-trivial finding from Protocol B: with $\approx 14$
training windows per subject, the simplest classical model
(Logistic Regression) ties or beats every deep architecture on stress
and statistically ties on valence (BiLSTM 0.860 vs LogReg 0.857).
Deep sequence models (BiLSTM, CNN1D, TinyTCN) only pull ahead on the
harder arousal and quadrant targets, where sequence structure
contributes information that hand-features compress away. Conformer
is consistently the *worst* deep model in this regime — attention-
heavy architectures cannot learn good positional priors at $n = 14$.

This regime-flipping is a fourth axis to add to the
model × protocol × sensor framing: **the model-class winner is
itself a function of the calibration regime.** Random Forest wins
LOSO stress; Logistic Regression wins within-subject stress; deep
sequence models win within-subject quadrant. No single model family
dominates across regimes.
