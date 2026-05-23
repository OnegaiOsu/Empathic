# 09 — Workplace usability

This section returns to the question that motivated the rest of the
paper: given what we now know about model performance and protocol
sensitivity, *what should an organization actually deploy* if it wants
a passive stress-monitoring system for its employees?

## 9.1 What "usability" means here

A workplace stress detector is not consumed by an ML researcher. It is
consumed by an HR manager, an occupational-health clinician, or — at
best — a self-coaching application running on the employee's own
device. The deployment metric we care about is therefore not "best
$\kappa$ on a held-out fold" but a tuple of:

- **Generalization to a new employee.** The user buying or running the
  system is by definition someone the model has never seen.
- **Calibration of the alert.** A useful system needs to surface
  alerts at a controllable false-positive rate, not a maximum-accuracy
  argmax decision.
- **Latency and infrastructure footprint.** Real-time inference on a
  battery-powered wrist device, or near-real-time on a personal laptop.
- **Interpretability.** When the system produces an alert, an
  occupational-health professional needs a defensible explanation for
  why.
- **Privacy.** Continuous physiological sensing in the workplace raises
  serious data-protection concerns (Mantello et al., 2023). Any
  system needs to be defensible under those constraints.

Each of these criteria points away from the deep models in our survey
and toward the classical baselines.

## 9.2 Recommendations

### 1. Use LOSO (or subject-grouped $k$-fold) as the *only* internal
metric.

Window-level k-fold scores will look better and will mislead. The
appropriate internal sign-off metric for a workplace stress system is
LOSO $\kappa$, computed at session level on subjects who never
appeared in training. Anything else creates an internal expectation
that production deployment will not meet.

### 2. Report and use *per-subject* variance, not just LOSO mean.

Our per-subject LOSO $\kappa$ ranges from 0.06 to 0.97 on arousal
(§6.3). Reporting the mean alone (0.79) hides the fact that there
exists at least one subject for whom the model is essentially useless.
A workplace deployment needs to surface this variance — to the user,
to the operator, or both — and ideally needs to detect when a new
employee is likely to be in the low-$\kappa$ regime and behave
conservatively.

### 3. Deploy classical models on tabular features by default.

Across five iterations of training improvements we did not find a
deep architecture that justified its compute and engineering cost
relative to a Random Forest or Logistic Regression on the 89-feature
tabular set. Concretely: at this dataset scale, the default production
stack is a Logistic Regression with $\ell_2$ regularization, balanced
class weights, calibrated probability output, and a per-employee
baseline-correction offset. The model is small enough to run on the
wearable's companion phone, fast enough for sub-second inference, and
interpretable enough to defend in an occupational-health audit.

### 4. Include a personalization step.

The single most informative change we did *not* test in this paper
is per-user fine-tuning. The per-subject heterogeneity reported in
§6.3 strongly implies that even a small amount of personal calibration
(e.g. fine-tuning on the first day's data, or computing personal
baselines on a guided rest period when the employee enrolls) would
shrink the across-subject variance. This is the most concrete
follow-up we recommend. Recent work on few-shot adaptation for
physiological signals (Kim et al., 2023) suggests that 5–10 minutes of
personal data can produce 5–10 point gains in $F_1$ on cross-subject
tasks.

### 5. Treat absolute $\kappa$ numbers from the literature as upper
bounds.

Section 8 makes the case empirically. A practitioner reading the
literature should mentally discount window-mixing $\kappa$ values by
the empirical inflation factor (~0.06 on valence, ~0.16 on arousal,
~0.18 on quadrant). The discounted number is closer to what the
employee actually gets.

## 9.3 Difficulty of integrating non-physiological modalities

Our project also evaluated the EmoSurv keystroke-dynamics dataset
(Yıldırım & Tatar, 2022) as a complementary signal. We do not include
those results in the headline numbers because the integration with
WESAD-style physiological data is genuinely difficult, and the
difficulty is itself worth reporting.

The fundamental issue is **temporal alignment**. WESAD windows are 60
seconds at 700 Hz of densely-sampled, continuously-present
physiological signal. Keystroke dynamics is fundamentally event-
driven: a user types in bursts, and during long no-typing intervals
(reading, thinking, meetings) there is no signal at all. Aligning the
two for late fusion forced us to either:

- **Resample to a common slow rate.** Aggregating physiological
  features over the same 60-second window in which keystrokes
  occurred. This collapses the keystroke modality into a small handful
  of summary statistics (typing speed, dwell-time variance, etc.) and
  loses much of its event-level information.
- **Conditional inference.** Run the keystroke model only when typing
  is detected, and fall back to physiology-only inference otherwise.
  Operationally messy and required a context-aware gating model that
  was non-trivial to train at our data scale.

We did neither systematically. EmoSurv lives in our codebase as
parallel pipeline (`src/empathic/data/emosurv.py`,
`runs/emosurv_*.py`) but we do not have stable cross-modal numbers to
report. The methodological lesson is that any workplace deployment
that wants to combine continuous physiological sensors with
event-driven behavioural signals (keystrokes, mouse movement,
application focus) needs to take temporal alignment as a first-class
design problem, not an afterthought. Recent work on irregular-time-
series transformers (Zhang et al., 2023) is a plausible direction
that we did not evaluate.

In the meantime, our recommendation for workplace deployment is to
treat physiological and behavioural channels as **independent
detectors with independent thresholds**, surfacing each separately to
a human reviewer rather than trying to fuse them into a single score.

## 9.4 Privacy and consent: a non-technical caveat

Continuous physiological monitoring in an employer–employee
relationship is qualitatively different from consumer wellness
applications, even when the technology is identical. Mantello et al.
(2023) survey workplace-affective-computing deployments and document
substantial worker concerns about monitoring asymmetry, function
creep, and consent under economic pressure. A model with $\kappa = 0.86$
that is involuntarily applied to performance evaluation is much worse
ethically than a model with $\kappa = 0.50$ that the employee can
inspect, override, and switch off. Any technically-defensible
workplace stress system therefore has to start from the consent and
governance layer, not from the model. We make no contribution to that
layer in this paper but flag it as the constraint within which any of
our quantitative claims should be read.

## 9.5 Summary of recommendations

For practitioners:

- **Default model:** Logistic Regression on the 89-feature tabular set
  with per-subject baseline correction.
- **Default protocol:** LOSO on internal evaluation; subject-grouped
  $k$-fold acceptable for faster CI runs.
- **Default reporting:** Session-level Cohen's $\kappa$, plus per-
  subject standard deviation, plus a confusion matrix on the rare
  alert-class.
- **Default expectation:** Cross-subject $\kappa$ in the 0.75–0.85
  range on arousal/quadrant, 0.85–0.95 on valence. Anything reported
  above 0.95 on a similar dataset should be treated as protocol-
  inflated until proven otherwise.
- **Default next step:** Personalization. Five-to-ten minutes of guided
  per-user calibration is likely the highest-leverage improvement.

For the research community: report the protocol explicitly in the
abstract.
