# 02 — Dataset: WESAD

## 2.1 Acquisition protocol

WESAD (Schmidt et al., 2018) is a publicly released multimodal dataset
collected at the University of Siegen for the explicit purpose of
benchmarking wearable stress and affect detection. Fifteen healthy
volunteers (twelve male, three female; mean age 27.5 years, SD 2.4)
participated in a single ~36-minute laboratory session under four
controlled affective conditions:

- **Baseline (condition 1).** Participants sat at a desk reading neutral
  magazines. Roughly twenty minutes.
- **Stress (condition 2).** The Trier Social Stress Test (TSST), a
  validated psychosocial stressor consisting of public speaking and
  arithmetic in front of a panel. Roughly ten minutes.
- **Amusement (condition 3).** A set of funny video clips. Roughly six
  minutes.
- **Meditation (condition 4).** A guided breathing/meditation
  recovery period inserted between conditions. Roughly seven minutes,
  applied twice in the protocol.

Participants wore two synchronized devices: a **RespiBAN Professional**
chest strap sampling at 700 Hz with eight channels — a
single-lead ECG, EDA, EMG, respiration, body-surface temperature, and a
three-axis accelerometer — and an **Empatica E4** wrist band sampling
at lower rates (BVP at 64 Hz, EDA and temperature at 4 Hz, ACC at 32 Hz).

## 2.2 Scope decision: chest only

This work uses the **chest channels exclusively**. We make this scope
decision deliberately and acknowledge it as a limitation:

- The chest sensor's higher sampling rate and more standardized contact
  produces cleaner ECG morphology and EDA traces than the wrist band,
  which is methodologically helpful when isolating the effect of model
  architecture from the effect of signal quality.
- For workplace deployment, however, the wrist band is the more realistic
  form factor. A chest strap is unlikely to be worn through a working
  day. Section 9 returns to this trade-off and argues for a future
  benchmark that runs the same architectures on the wrist channels
  alone, both because of the deployment realism and because Sah et al.
  (2022) report a substantial accuracy gap between the two form
  factors.

## 2.3 From conditions to valence-arousal labels

The four condition labels are mapped onto the canonical circumplex of
affect (Russell, 1980) as follows:

| Condition  | Code | Quadrant | Valence | Arousal |
|------------|------|----------|---------|---------|
| Baseline   | 1    | HVLA     | High    | Low     |
| Stress     | 2    | LVHA     | Low     | High    |
| Amusement  | 3    | HVHA     | High    | High    |
| Meditation | 4    | HVLA     | High    | Low     |

The **LVLA quadrant (low valence, low arousal — sadness, depression,
boredom)** is structurally absent. This is a fundamental limitation of
WESAD: the four-quadrant claim that often appears in the affective
computing literature reduces to a *three-class* problem on this dataset.
Our quadrant target therefore has three classes, with HVLA as a
substantial majority because it absorbs both baseline and meditation.
The per-window class distribution is HVLA = 981 (65.4%), HVHA = 332
(22.1%), LVHA = 186 (12.4%), shown in Figure 1 of the dataset
inspection logs.

For binary targets we project onto the dimensional axes:

- **Valence.** Positive (HVHA ∪ HVLA) vs. negative (LVHA). Class
  distribution 1167 / 332.
- **Arousal.** High (LVHA ∪ HVHA) vs. low (HVLA). Class
  distribution 518 / 981.

Both are imbalanced. We address imbalance through sample-weighted
training rather than resampling.

## 2.4 Windowing and dataset size

We use a 60-second sliding window with 30-second stride at the native
700 Hz chest sampling rate. Transition periods between conditions and
short residual segments shorter than the window length are discarded.
This yields **1499 windows distributed across fifteen subjects**.

The size of this dataset is the dominant constraint on every methodological
decision in this paper. With ~100 windows per subject on average, even a
modest amount of subject-level leakage in the cross-validation procedure
will dominate the reported metric.

## 2.5 Demographics, ecological validity, and limitations

We close the dataset section with three caveats that drive the rest of the
paper.

1. **Demographic narrowness.** Fifteen mostly-male university-affiliated
   participants in a single laboratory cannot stand in for the full
   workplace population. WESAD performance numbers should be understood
   as upper bounds on what is achievable under near-ideal acquisition
   conditions.
2. **Lab-induced stress is not workplace stress.** The TSST evokes acute
   social-evaluative stress over minutes; sustained workplace stress
   accumulates over hours and weeks and is qualitatively different
   (Schneiderman et al., 2005). Models trained on WESAD will at best
   detect sharp acute stress events, which is still operationally useful
   but should not be marketed as "burnout detection".
3. **Class structure is uneven and incomplete.** The HVLA class absorbs
   both baseline and meditation; the LVLA class is absent. Reported
   accuracy is therefore not directly comparable across studies that
   merge or split these conditions differently.

These limitations make a *protocol-level* discussion (Section 8)
indispensable. Reporting an inflated metric on this small, structurally
imbalanced dataset produces an artifact, not a finding.
