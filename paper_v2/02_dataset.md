# 02 — Dataset: EmoWork

## 2.1 Acquisition protocol

EmoWork (Lee et al., 2026) is a multimodal physiological corpus collected
on **31 participants (P1–P31)** performing six approximately four-minute
sessions in a simulated Korean call-centre workplace. The sessions
alternate between rest and customer-service phone calls:

- `b1`, `b2`, `b3` — rest / break periods between calls.
- `c1`, `c2`, `c3` — customer-service phone calls with a scripted actor
  applying *mild*, *moderate*, and *severe* complaint pressure
  respectively.

Each subject is recorded simultaneously with multiple wrist- and head-form
devices sampling at very different native rates, plus continuous
self-reported affect labels at ~10 Hz on a Likert scale.

The label streams are continuous on:

- **arousal** $\in [1, 9]$
- **valence** $\in [1, 9]$
- **stress**  $\in [1, 20]$

We discretise at the natural midpoints (arousal/valence $> 5$;
stress $\ge 10$) following the dataset's own analysis conventions, and
derive a four-class **quadrant** label as the cross-product of binary
valence and binary arousal.

## 2.2 Sensor stack and channel set

Twelve channels are retained on a common 32 Hz grid:

| Group | Channel | Native rate | Source |
|---|---|---|---|
| Cardiac | ECG | ~130 Hz | Polar chest sensor |
| Cardiac | BVP | 64 Hz | Empatica E4 wrist |
| Cardiac | HR  | 1 Hz | Polar (upsampled) |
| Electrodermal | EDA | 4 Hz | Empatica E4 |
| Thermal | TEMP | 4 Hz | Empatica E4 |
| Inertial | ACC_x, ACC_y, ACC_z | 32 Hz | Empatica E4 |
| Cortical | EEG_TP9, EEG_AF7, EEG_AF8, EEG_TP10 | 256 Hz | Muse headband |

A Galaxy PPG channel was present in the raw release but consisted of
uniformly subnormal `float32` noise ($\approx 2.94 \times 10^{-39}$); we
drop it.

## 2.3 Windowing and dataset size

A 60-second sliding window with 30-second stride is applied to the
common 32 Hz grid, after resampling and inter-sensor alignment. After
rejecting flat-signal and missing-cardiac windows, the corpus contains:

- **625 windows** across **31 subjects**
- **149 tabular features** (HRV from ECG / BVP, EDA tonic / phasic
  decomposition, accelerometer activity descriptors, EEG band powers
  per channel) plus a 12 × 240 downsampled sequence tensor for the deep
  learners

Per-modality feature counts: ECG 17, BVP 17, EDA 14, TEMP 9, HR 7,
ACC 25, EEG 60. The EEG block is the largest by far — a fact that turns
out to matter materially in §7.

## 2.4 Class distributions and target structure

| Target | Type | Class counts | Note |
|---|---|---|---|
| Stress  | binary | $[309, 316]$ | Almost balanced |
| Arousal | binary | $[255, 370]$ | 41% / 59% |
| Valence | binary | $[542, 83]$ | **Severe imbalance** (87% high) |
| Quadrant | 4-class | $[33, 50, 337, 205]$ | LVLA = 33, LVHA = 50, HVLA = 337, HVHA = 205 |

Two facts about this label structure govern §5–§7.

1. Valence is dominated by the high-valence class, and *11 of 31
   subjects have all-one-class valence in the data they contributed*.
   Under LOSO this means single-class folds are common; the binary
   classifier cannot fit on those folds and they are skipped. Even on
   the 20 multi-class subjects, the per-fold class prior shifts
   dramatically, which inflates standard deviation and depresses
   κ.
2. Quadrant is highly imbalanced and small (33 LVLA windows shared
   across only a subset of subjects). It is the hardest target and
   should be read as a stress-test, not a primary metric.

## 2.5 Limitations

- **Single laboratory, single protocol.** Effect sizes here are upper
  bounds for one specific simulated workplace stressor.
- **31 subjects is small.** Per-subject standard deviations of all
  metrics are large; we report them everywhere.
- **Continuous-label thresholding is coarse.** Treating arousal $> 5$
  as binary discards the bulk of the information in the continuous
  scale. We retain the binary framing for direct comparability with
  the WESAD literature (§8) but acknowledge this as a modelling choice
  that drives valence's structural difficulty.
- **No demographics included with the release artefacts we used**;
  external generalisation claims should be conservative.

These constraints make the *protocol-level* discussion of §6
indispensable and the *modality-level* discussion of §7 more
interesting than another architectural shoot-out would have been.
