# 09 — Conclusion

We have presented an honest evaluation of multimodal affect recognition
on the 31-subject EmoWork corpus, covering eleven model families, four
targets (stress, arousal, valence, quadrant), and four evaluation
regimes (LOSO, window-stratified, rest-anchored LOSO, within-subject
70/30). The headline numbers under leave-one-subject-out
cross-validation are:

- **Stress:** Random Forest, macro-F1 $0.677$, κ = 0.383.
- **Arousal:** Conformer fusion, macro-F1 $0.538$, κ = 0.149.
- **Valence:** BiLSTM fusion, macro-F1 $0.507$, κ = 0.095.
- **Quadrant:** XGBoost, macro-F1 $0.309$, κ = 0.045.

These are substantially below the $0.80$–$0.95$ macro-F1 band routinely
reported in the multimodal affect literature. *Three* findings of this
paper explain why, and bound what is reachable when the protocol is
relaxed in deployment-honest ways.

## 9.1 Protocol explains roughly $0.4$ of κ

Re-running Random Forest on stress under window-stratified 80/20
splits yields κ = 0.664 — a doubling of the LOSO κ of
$0.308$. The same pattern holds across arousal and quadrant. Subject
identity, not affect, is what window-leakage protocols learn. *LOSO is
the only protocol consistent with workplace deployment*, where every
new employee is by definition a held-out subject. Subject-grouped
5-fold is a defensible faster surrogate.

## 9.2 One sensor explains the rest

Per-modality ablation reveals that on this corpus a single ECG sensor
reaches κ = 0.392 for stress, *exceeding* the full
149-feature multimodal stack. EDA alone is the best modality for
arousal, and adding EEG to the physio stack actively *hurts* arousal
performance under LOSO at this sample size. The cost-effective
deployment for stress is a one-channel ECG with seventeen HRV-derived
features; everything else is at best neutral and sometimes harmful.

## 9.3 Per-subject calibration is the dominant lever

The within-subject 70/30 evaluation (Protocol B, §6.6) lifts every
target by an amount no architectural or sensor choice has matched:

- Stress: $0.677 \to 0.908$ macro-F1 (κ $0.38 \to 0.82$).
- Arousal: $0.538 \to 0.818$ macro-F1.
- Valence: $0.507 \to 0.860$ macro-F1.
- Quadrant: $0.309 \to 0.724$ macro-F1.

A workplace deployment that allows $\approx 14$ labelled windows of
per-employee calibration (under ten minutes of structured baseline
recording) more than doubles Cohen's κ on stress and lifts valence
from chance to deployable. The cheaper rest-anchored LOSO variant
(Protocol C) does *not* help — the §5 pipeline already z-scores
against c-session statistics, so a rest-only reference is strictly
worse. Personalisation, not a single rest recording, is what closes
the gap.

The model-class winner also flips with regime: Random Forest wins
LOSO stress, but Logistic Regression wins within-subject stress and
BiLSTM wins within-subject arousal and valence. The "best" model is
itself a function of the calibration regime; reporting it without
specifying the regime is meaningless.

## 9.4 Limitations

- **Small subject pool ($n = 31$).** Per-subject standard deviations
  of macro-F1 are 0.08–0.20 on every model. We do not claim
  significant differences between models that differ by less than one
  standard deviation; we report subject-paired Wilcoxon as a
  follow-up but do not lean on those results.
- **Single laboratory and protocol.** Effect sizes are upper bounds
  for the call-centre simulation; transfer to other workplaces will
  not match these numbers without adaptation.
- **Coarse continuous-label thresholding.** Treating arousal $> 5$ as
  binary discards most of the information in the continuous scale.
  Continuous-target regression is the appropriate next experiment for
  valence in particular.
- **Logging artefact in the ensemble.** The DANN-Conformer is
  excluded from the current soft-vote ensemble due to a key-mismatch
  bug in [`runs/emowork/train_all.py`](../runs/emowork/train_all.py)
  that has been fixed for future runs. This does not affect the
  standalone DANN results in §5.
- **XGBoost in `relaxed_eval.py`.** The protocol-comparison script
  does not pass class weights to XGBoost; we report XGBoost in §6 as
  κ = 0 for transparency, with the §5 weighted XGBoost
  (e.g. κ = 0.354 on stress) as the primary number.

## 9.5 Recommendations

For practitioners deploying affect recognition on small multimodal
corpora:

1. **Default to LOSO** for the *generalisation* claim. Subject-grouped
   $k$-fold is acceptable as a surrogate; window-level $k$-fold is not,
   except as an upper-bound illustration.
2. **Report a within-subject ceiling alongside the LOSO floor.** The
   gap between the two is the most informative single quantity in this
   design space; it tells the reader how much of their problem is
   solved by the model and how much by knowing the subject.
3. **Default to single-sensor RF as the LOSO baseline to beat.** Only
   adopt fusion (multimodal or deep) when it surpasses the
   single-sensor classical model by more than the LOSO standard
   deviation, on subject-paired tests.
4. **Treat EEG as a high-variance, high-feature-count modality.** On
   small subject pools the curse of dimensionality is real and EEG
   can actively hurt downstream LOSO performance.
5. **Report κ alongside macro-F1.** κ is harder to inflate by
   majority-class shortcuts; the difference between the two metrics is
   itself diagnostic (compare baseline rows in §5).

## 9.6 Future work

- **Continuous-target regression** for arousal, valence, and stress on
  the original 1–9 / 1–20 Likert scales, using session-level
  evaluation and per-subject calibration.
- **Leave-one-call-out within subject.** EmoWork's three c-sessions
  per subject permit a leak-free within-subject protocol (train on
  c1+c2, test on c3, etc.) at the cost of $\le 7$ test windows per
  fold. This would tighten the §6.6 ceiling against the dataset-
  inherent leakage paths discussed in §6.6.3.
- **Few-shot calibration cost curve.** Protocol B uses 70% of each
  subject's c-session windows ($\approx 14$ samples). The deployment-
  relevant question is the *minimum* calibration set size that
  recovers the ceiling — likely well under 14 windows for binary
  stress given the LR/RF margins.
- **Cross-corpus transfer.** Train on EmoWork ECG-only, test on WESAD
  ECG-only, and quantify the transfer drop. This would establish
  whether the cardiac-signal universality argued in §7 generalises
  beyond the EmoWork-specific call-centre stressor.
- **Larger subject pools.** None of the conclusions in this paper
  preclude multimodal fusion winning the *generalisation* race at
  $n \gg 31$. The right experiment is to scale subject count, not
  architecture.
