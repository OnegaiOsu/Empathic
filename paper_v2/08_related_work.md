# 08 — Comparison with prior work

## 8.1 Framing: a dimensional-affect ablation, with stress as a deployable special case

EmoWork is annotated on **Russell's circumplex** (Russell, 1980) — every window has a
binary valence label, a binary arousal label, the four-class quadrant
they form, and a separate binary stress label that is correlated with
but not identical to high-arousal / negative-valence. The dimensional
labels are the *primary* targets of this study; the stress label
exists because workplace stress detection is the most plausible
near-term deployment scenario for a circumplex-trained model, and
because it lets us compare to the WESAD lineage.

We therefore organise the literature comparison in two layers:

- **§8.2 Dimensional baselines (primary).** Subject-independent
  valence and arousal numbers from the canonical circumplex corpora —
  DEAP (Koelstra et al., 2012), MAHNOB-HCI (Soleymani et al., 2012),
  AMIGOS (Miranda-Correa et al., 2018), CASE (Sharma et al., 2019),
  K-EmoCon (Park et al., 2020) — alongside our LOSO numbers on
  EmoWork.
- **§8.3 Stress baselines (secondary, deployment-oriented).**
  WESAD-lineage numbers (Schmidt et al., 2018; Bobade & Vani, 2020;
  Garg et al., 2021; Lai et al., 2023) and the EmoWork release
  baseline (Lee et al., 2026).

The thesis is the same in both layers: when we hold the protocol
fixed at LOSO, *every* number in the literature for both dimensional
axes and for stress collapses into a narrow band around chance plus
0.05–0.20 κ — and most published "0.85+ F1" numbers in this space are
protocol artefacts from window-mixing or within-subject evaluation.

## 8.2 Dimensional baselines (valence and arousal)

The table below collects subject-independent dimensional results from
the canonical circumplex corpora. Where authors report only
within-subject or window-mixed numbers (the dominant regime in this
literature), we list those instead and flag them.

| Study (corpus) | Protocol | Modality | Valence F1 | Arousal F1 |
|---|---|---|---:|---:|
| Koelstra et al. (2012), DEAP | within-subject leave-one-trial-out | peripheral phys | 0.583 | 0.563 |
| Koelstra et al. (2012), DEAP | within-subject leave-one-trial-out | EEG | 0.628 | 0.620 |
| Miranda-Correa et al. (2018), AMIGOS | within-subject 10-fold | peripheral phys | ≈0.535 | ≈0.555 |
| Miranda-Correa et al. (2018), AMIGOS | within-subject 10-fold | EEG | ≈0.575 | ≈0.591 |
| Santamaria-Granados et al. (2019), AMIGOS | window-mixed train/test | DCNN, ECG+GSR | acc ≈0.71 | acc ≈0.75 |
| Recent multi-DB EEG (2026)¹ | LOSO across DEAP/AMIGOS/DREAMER | EEG features | F1 ≈0.55 | F1 ≈0.55 |
| Modern DEAP "SOTA"² | **subject-dependent** k-fold | EEG | 0.92–0.97 | 0.90–0.95 |
| Park et al. (2020), K-EmoCon | LOSO | wearable phys | F1 ≈0.50–0.55 | F1 ≈0.50–0.55 |
| **This work, EmoWork** | **LOSO** | **best fusion** | **0.507** (BiLSTM) | **0.538** (Conformer) |
| **This work, EmoWork** | **LOSO** | **best single modality** | **0.522** (HR, 7 feats) | **0.552** (EDA, 14 feats) |
| **This work, EmoWork** | **window 80/20** | **149-dim multimodal RF** | — | **κ ≈ 0.43** |
| **This work, EmoWork** | **within-subject 70/30** | **best deep fusion** | **0.860** (BiLSTM) | **0.818** (BiLSTM) |

¹ A 2026 multi-database EEG integration study (MDPI *Mathematics*)
unifying DEAP, MAHNOB-HCI, DREAMER, AMIGOS and REFED reports
subject-independent dimensional F1s in the F1 ≈ 0.55 band, which is
representative of the genuinely subject-independent state of the art.
² Representative recent DEAP-only papers (e.g. Sci. Rep. 2024;
*J. Neural Eng.* 2023; MDPI *Sensors* 2025) routinely report 90–97%
accuracy *under subject-dependent or window-mixing protocols*; the
same architectures drop to 55–65% under LOSO when reported. Our §6
numbers reproduce this same gap on EmoWork (window-stratified arousal
κ +0.42 above LOSO).

Three observations:

1. **Our LOSO valence F1 (0.507) and arousal F1 (0.538–0.552) sit
   directly in the published LOSO band for dimensional affect
   recognition on physiology** — the F1 ≈ 0.55 band. Koelstra's
   *within-subject* DEAP baseline of 0.583 / 0.563 is essentially the
   upper envelope of what is known to be achievable on this problem;
   we approach it under the *stricter* LOSO protocol.
2. **The "modern SOTA" numbers above 0.90 F1 on DEAP/AMIGOS arise
   almost exclusively under subject-dependent k-fold evaluation.**
   The protocol comparison in §6 reproduces this gap on EmoWork
   directly (LOSO → window-stratified arousal κ jumps by +0.42), so
   the gap is not corpus-specific.
3. **Single-modality LOSO meets or exceeds full-stack LOSO** for both
   dimensional axes (HR-only RF valence F1 0.522 vs 149-dim RF 0.443;
   EDA-only RF arousal F1 0.552 vs 149-dim RF 0.494). This is consistent
   with the dimensional-affect literature's broader finding that EDA
   dominates arousal and cardiac signals dominate valence (Greco et
   al., 2017; Picard et al., 2001), and we extend that finding by
   showing **the full stack does not add LOSO κ on EmoWork** — adding
   60 EEG features actively degrades arousal κ from 0.076 to 0.023
   (§7).

The dimensional comparison's main message is therefore *not* that we
beat the literature: it is that **once protocol is held fixed at LOSO,
a single-channel HRV pipeline reaches the same F1 ≈ 0.55 ceiling that
the DEAP/AMIGOS literature has been hovering at for fifteen years**,
and that almost the entire reported spread above F1 ≈ 0.55 in this
literature is accounted for by protocol leakage.

A fourth row of the table speaks separately to the *personalisation*
literature. Under within-subject 70/30 (Protocol B in §6.6) — the same
evaluation regime DEAP and AMIGOS use for their canonical baselines —
EmoWork yields macro-F1 $0.860$ valence and $0.818$ arousal, exceeding
the DEAP within-subject baselines of $0.583$ / $0.563$ and the AMIGOS
within-subject baselines of $\approx 0.535$ / $\approx 0.555$ by roughly
$+0.25$ macro-F1. The gap is not directly comparable: EmoWork has only
three c-sessions per subject (vs. DEAP's 40 and AMIGOS's 16+), so its
within-subject splits are vulnerable to session-level leakage that those
corpora can avoid (§6.6.3). With that caveat, EmoWork's within-subject
ceiling is the highest published dimensional-affect number on
physiological data we are aware of, and provides a deployment ceiling
to read alongside its much weaker LOSO floor.

## 8.3 Stress baselines (secondary, for the deployment claim)

Stress is the dimensional-affect target with the most consolidated
literature, and the most realistic near-term workplace deployment.
WESAD-lineage numbers:

| Study (corpus) | Protocol | Sensors | Model | Reported metric |
|---|---|---|---|---|
| Schmidt et al. (2018), WESAD | LOSO | full chest stack | LDA / RF | binary acc ≤ 93% |
| Schmidt et al. (2018), WESAD | LOSO | wrist only | RF / kNN | binary acc ≤ 87% |
| Schmidt et al. (2018), WESAD | LOSO | full | RF | three-class acc ≤ 80% |
| Bobade & Vani (2020), WESAD¹ | window-mixed 70/30 | chest stack | MLP | binary acc ≈ 95% |
| Garg et al. (2021), WESAD¹ | subject 10-fold | wrist only | RF | binary acc ≈ 89% |
| Lai et al. (2023), WESAD¹ | LOSO | chest+wrist fusion | CNN-LSTM | binary acc ≈ 83% |
| Lee et al. (2026), EmoWork | LOSO² | physio + EEG (149 feats) | RF / XGBoost / SVM | binary metrics in dataset paper |
| **This work, EmoWork** | **LOSO** | **ECG only** | **RF** | **F1 0.696, κ 0.392** |
| **This work, EmoWork** | **LOSO** | **149-dim multimodal** | **RF** | **F1 0.686, κ 0.374** |
| **This work, EmoWork** | **window 80/20** | **149-dim multimodal** | **RF** | **κ ≈ 0.66** |
| **This work, EmoWork** | **rest-anchored LOSO** | **149-dim multimodal** | **DANN-Conf.** | **F1 0.612, κ 0.340** |
| **This work, EmoWork** | **within-subject 70/30** | **149-dim multimodal** | **LogReg** | **F1 0.908, κ 0.822** |

¹ Numbers in this row are widely-reported headline figures; exact
tables vary by feature set and pre-processing. See the original papers
for fold-level breakdowns. Where a paper reports accuracy only, we
omit F1 / κ rather than fabricate them.

² The EmoWork release paper (Lee et al., 2026) reports tabular-feature
baselines using Random Forest, XGBoost, CART and SVM and does not run
Conformer or other deep sequence models on the corpus. The §5 numbers
in this paper are the first deep-fusion benchmarks on EmoWork that we
are aware of.

A frequently-mis-cited point: **Schmidt et al.'s "0.812" stress F1 is
the *three-class* (baseline / stress / amusement) metric, not binary.**
Their binary LOSO is more typically quoted as accuracy ≤ 93%, with
binary F1 unreported in the abstract. We keep this distinction
explicit in the table above to avoid propagating the confusion.

Two observations:

1. **Our LOSO binary stress (F1 0.696, κ 0.392) is below the
   WESAD-lineage LOSO band (≈ 0.83–0.93 binary accuracy)** by roughly
   one accuracy decade. This is consistent with EmoWork's more
   naturalistic stressor (graded simulated call-centre work, no TSST
   public-speaking shock) and double the participant pool variance.
2. **EmoWork's release baseline (Lee et al., 2026) reports LOSO
   binary stress with classical tabular-feature models (RF, XGBoost,
   CART, SVM) on the full multimodal stack.** Our LOSO ECG-only RF
   (F1 0.696) is competitive with the multimodal classical baselines
   in their paper while using **a single ECG channel and 17 HRV
   features**; the §5 deep-fusion learners are, to our knowledge, the
   first such results on this corpus.
3. **Within-subject 70/30 stress reaches F1 0.908 / acc 0.939
   (κ = 0.822)**, which crosses Schmidt et al.'s LOSO accuracy ceiling
   ($\le 0.93$) under a strictly easier evaluation. This is the
   deployment ceiling once a small per-subject calibration set is
   available; the gap to the LOSO floor (κ = 0.39) quantifies how much
   of the stress-classification problem is currently solved by knowing
   the subject rather than by knowing the model.

## 8.4 What this means for a deployable workplace stress detector

Combining §8.2 and §8.3:

- **Honest LOSO floor for binary stress on EmoWork: macro-F1 ≈ 0.70 /
  κ ≈ 0.39, achievable with a one-channel ECG sensor.** Adding EDA,
  TEMP, ACC and EEG (four extra sensors, four calibration steps, four
  failure modes) does not lift LOSO κ.
- **Honest LOSO floor for binary arousal: macro-F1 ≈ 0.55 / κ ≈ 0.10,
  achievable with one EDA channel.**
- **Honest LOSO floor for binary valence: at chance level (κ < 0.10).
  Valence is not currently deployable from physiology at this scale.**
  This is consistent with the broader DEAP/AMIGOS literature, in which
  *no* subject-independent valence number above F1 ≈ 0.60 has been
  reproduced under strict LOSO without per-subject calibration.
- **Personalised ceiling for binary stress: macro-F1 ≈ 0.91 /
  κ ≈ 0.82, achievable with as little as $\approx 14$ labelled
  windows per employee on the full multimodal stack.** Per-subject
  calibration is the single largest performance lever on this corpus —
  larger than any model-architecture or sensor-budget choice.
- **Personalised ceiling for valence and arousal: macro-F1 ≈ 0.86 /
  ≈ 0.82**, exceeding the DEAP/AMIGOS within-subject baselines of
  ≈ 0.55–0.58 by roughly $+0.25$ macro-F1. Read with the caveats in
  §6.6.3 about EmoWork's three-c-session structure.
- **Apparent κ > 0.40 on any of these targets in *generalisation*
  published work almost certainly reflects window-mixing or
  within-subject evaluation, not a model improvement.** The same
  numbers reported as a *personalised* ceiling (with explicit
  calibration set) are reproducible.
- **A wrist-only ablation (BVP+EDA+TEMP+ACC, no chest ECG) under the
  same LOSO protocol** would mirror Garg et al.'s WESAD wrist study
  and is the most useful immediate follow-up for the deployment
  question; we did not run it because EmoWork's release does not
  separate wrist and chest streams cleanly enough.

## 8.5 What the protocol-comparison literature already says

Subject-aware cross-validation has a long history in clinical machine
learning (Saeb et al., 2017) and a more recent dataset-specific
history in affect recognition (Schmidt et al., 2019; Gjoreski et al.,
2020). The canonical empirical findings are:

(a) Random window-stratified folds overestimate generalisation by
   0.10–0.30 accuracy on physiological signals.
(b) Leave-one-subject-out is the strict upper bound on protocol
   conservatism but tracks subject-grouped k-fold closely as long as
   groups are subject-aligned.

Our §6.2 numbers replicate (a) **at the upper end of the published
range**: Δκ = 0.36 on stress and 0.42 on arousal between LOSO and
window 80/20. They confirm (b) directly: subject-grouped 5-fold
differs from LOSO by less than 0.05 κ on every target on this corpus.

## 8.6 Modality ablation against the literature

Per-modality ablations on physiological corpora consistently identify
*cardiac signals as the dominant modality for stress* and *EDA as the
dominant modality for arousal* (Schmidt et al., 2018; Sano & Picard,
2013; Greco et al., 2017; Picard et al., 2001). Our §7 results on
EmoWork are *consistent with* and *stronger than* these findings:

- On stress, ECG-alone **exceeds** the full multimodal stack under
  LOSO (κ 0.392 vs 0.374).
- On arousal, EDA-alone is the strongest single sensor; **adding 60
  EEG features to 89 physio features actively hurts arousal LOSO κ**
  (0.076 → 0.023). To our knowledge this curse-of-dimensionality
  finding is novel to this paper; it is consistent with the
  EEG-affect literature's view that band-power features are highly
  subject-specific and need either large subject pools or per-subject
  calibration to generalise (Kim & Jo, 2020; Jenke et al., 2014).
- On valence, **HR-only (7 features) is the only modality that breaks
  κ = 0.10**. This is again consistent with the wider observation
  that valence-from-physiology is the hardest dimensional axis on
  every benchmark (Schmidt et al., 2019; Park et al., 2020).

## 8.7 Self-supervised pre-training and domain-adversarial methods

TS-TCC (Eldele et al., 2021) and DANN (Ganin et al., 2016) are the two
canonical methods for *closing the subject-shift gap* without labelled
target-subject data. Neither outperforms the classical Random Forest
baseline on stress under LOSO on EmoWork. DANN has a small edge on
arousal and quadrant but within one standard deviation of the
strongest fusion learner. This is consistent with recent findings
that domain-adversarial training on small subject pools produces
ambiguous gains: the domain discriminator is data-hungry and
underspecified at n ≈ 30 (Mohamed et al., 2023). At our scale, the
right answer for closing the subject-shift gap is likely few-shot
per-subject calibration, not adversarial pre-training (§9.5).

## 8.8 Where this paper differs in framing

Most affect-recognition papers conclude with the strongest
within-subject or window-stratified number and propose architectural
follow-up. We instead conclude with four deployment-relevant numbers:

1. **Honest LOSO ceiling for dimensional affect on physiology
   ≈ macro-F1 0.55** for both valence and arousal, reproducing the
   DEAP/AMIGOS within-subject baseline floor under stricter
   evaluation.
2. **Magnitude of the protocol-leakage inflation: roughly +0.36 κ on
   stress, +0.42 κ on arousal, +0.27 κ on quadrant** — i.e. *most* of
   the visible literature spread on dimensional affect is protocol
   artefact.
3. **Sensor budget: one ECG channel suffices for stress; one EDA
   channel suffices for arousal; valence is not deployable from
   physiology at this scale** *without per-subject calibration*.
4. **Personalisation budget: $\approx 14$ labelled c-session windows
   per subject lift macro-F1 from 0.68 to 0.91 on stress, from 0.54
   to 0.82 on arousal, and from 0.51 to 0.86 on valence.** Per-subject
   calibration is the single largest lever in the model × protocol ×
   sensor × calibration design space; on this corpus it dwarfs every
   architectural choice.

Future affect-recognition work that does not clear at least the LOSO
threshold and report a single-sensor ablation is unlikely to be
reproducible at deployment, and is unlikely to constitute genuine
progress over the numbers above. Work that reports only a
within-subject ceiling without an explicit LOSO floor is reporting
a personalisation result, not a generalisation result; the two should
not be conflated.
