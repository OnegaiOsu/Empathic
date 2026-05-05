# 10 — Conclusion

We benchmarked a survey of seven model families — three classical, four
deep — on the WESAD chest-sensor dataset under leave-one-subject-out
cross-validation, with the workplace stress-monitoring use case in mind.
Our v5 pipeline, which combines hand-crafted physiological features with
late-fusion sequence representations, per-subject sequence baseline
correction, and temperature-calibrated ensembling, achieves session-level
Cohen's $\kappa$ of 0.86 (quadrant), 0.96 (binary valence), and 0.82
(binary arousal) on subjects who never appeared in training. These
numbers compare favorably to the recent LOSO-protocol literature
(§7) while remaining honest about the per-subject variance: the same
model produces $\kappa$ from below 0.10 to above 0.97 across our fifteen
subjects.

Three findings stand out.

**Classical baselines are the right default.** Across five training
iterations of representation, augmentation, fusion, and calibration
improvements, deep architectures closed but did not erase the gap to a
strong Random Forest or Logistic Regression on tabular features.
Workplace deployment of a stress monitor should default to the
calibrated linear model unless future work, ideally on a substantially
larger dataset than WESAD, demonstrates a clear deep-model advantage
under honest evaluation.

**Per-subject variance dominates per-architecture variance.** The gap
between the best and worst architecture on any target is smaller than
the gap between the best and worst subject on the best architecture.
The implication for deployment is that personalization — even in its
simplest form, per-user baseline calibration — is likely the
highest-leverage improvement available, larger than any architectural
change we tried.

**Protocol matters at least as much as architecture.** Substituting
window-stratified $k$-fold for LOSO inflates reported $\kappa$ by
+0.06 (valence) to +0.21 (quadrant) on the same models on the same
data. Subject-grouped $k$-fold tracks LOSO closely, demonstrating that
the methodological harm comes specifically from leaking same-subject
windows across train and test. Several recent WESAD papers reporting
$\kappa$ above 0.95 are most likely measuring this leakage, not
generalization. The community would benefit from making LOSO the
default reporting protocol, and from foregrounding the protocol in the
abstract of any new benchmark.

## 10.1 Limitations

We acknowledge several limitations.

- **Fifteen subjects.** WESAD's small subject pool means that even
  honest LOSO has wide confidence intervals. The conclusions of this
  paper would be substantially strengthened by replication on a larger
  dataset such as a future WESAD successor or an aggregated multi-
  cohort study.
- **Chest-only.** We did not evaluate the wrist-only configuration that
  is more realistic for daily workplace wear. Sah et al. (2022) report
  a substantial drop from chest to wrist; replicating our protocol-
  audit on the wrist channels is the most important follow-up.
- **No deep models in the protocol audit.** Section 8 ran the four-
  protocol comparison only on classical models. Adding the deep
  pipeline is operationally straightforward but multi-hour wall-clock;
  we expect the qualitative pattern of inflation to persist because
  the leakage mechanism is upstream of architecture.
- **Behavioural modalities not integrated.** Keystroke dynamics
  (EmoSurv) is included in our codebase but did not produce stable
  cross-modal numbers; §9.3 documents the temporal-alignment
  difficulty.
- **Lab stress, not workplace stress.** WESAD captures acute
  laboratory-induced stress; workplace stress is sustained and
  different in kind (Schneiderman et al., 2005). External validity
  to the deployment scenario is therefore qualitative, not
  quantitative.

## 10.2 Future work

The most informative follow-ups, in order of expected impact:

1. **Per-user personalization study.** Quantify how much a 5–10 minute
   personal calibration (or a few-shot fine-tune) closes the per-
   subject variance gap. Likely the largest available improvement.
2. **Wrist-only replication.** Run the same v5 pipeline and the same
   four-protocol audit on the wrist channels, both because the wrist
   is the deployable form factor and because Sah et al. (2022)'s gap
   warrants verification.
3. **Deep models in the protocol audit.** Extend `runs/relaxed_eval.py`
   to deep architectures, expected to confirm but possibly amplify the
   inflation pattern.
4. **Multi-cohort replication.** Apply the v5 pipeline to a recent
   larger physiological-affect dataset (e.g. K-EmoCon, MMSE-HR,
   StudentLife) and check whether the LOSO numbers transfer.
5. **Behavioural-physiological fusion at scale.** Investigate
   irregular-time-series transformers (Zhang et al., 2023) for proper
   cross-modal late fusion of keystrokes, mouse, and physiology.

## 10.3 Closing

The argument of this paper, briefly stated: workplace stress detection
on WESAD is a tractable problem that the field has reported on with
methodologically uneven evaluation. Under an evaluation protocol that
matches the deployment scenario — predicting on a new employee —
state-of-the-art performance is approximately $\kappa \in [0.82, 0.96]$
across the three target axes, achieved by a Random Forest or Logistic
Regression on hand-crafted physiological features. Deep models close
some of the gap but do not exceed it. Headline numbers above 0.95 in
the recent literature most likely reflect protocol inflation rather
than architectural progress. We hope this paper offers both a useful
honest baseline and a usable methodological argument for the field's
default protocol going forward.
