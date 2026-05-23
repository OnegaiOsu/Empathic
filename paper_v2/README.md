# Paper (v2) — EmoWork honest evaluation

This folder contains the academic-style write-up of the EmoWork
multimodal study. It is the EmoWork counterpart of
[`archive/wesad/paper/`](../archive/wesad/paper/), which covered WESAD.

## Sections

1. [00 — Abstract & front matter](00_abstract.md)
2. [01 — Introduction](01_introduction.md)
3. [02 — Dataset (EmoWork)](02_dataset.md)
4. [03 — Models](03_models.md)
5. [04 — Preprocessing & feature extraction](04_preprocessing.md)
6. [05 — LOSO results](05_results.md)
7. [06 — Re-evaluation under alternative protocols](06_protocol_reevaluation.md)
8. [07 — Per-modality ablation](07_modality_ablation.md)
9. [08 — Related work](08_related_work.md)
10. [09 — Conclusion](09_conclusion.md)
11. [References (APA)](references.md)

## Source artefacts

All numbers in this paper trace directly to files under
`results/emotion/emowork/`:

| Section | Source |
|---|---|
| §5 LOSO summary | [results/emotion/emowork/emowork/{quadrant,valence,arousal,stress}/summary.csv](../results/emotion/emowork/emowork/) |
| §5 cross-target figures | [results/emotion/emowork/figures/](../results/emotion/emowork/figures/) |
| §6 protocol comparison | [results/emotion/emowork/relaxed_eval.log](../results/emotion/emowork/relaxed_eval.log) |
| §7 modality ablation | [results/emotion/emowork/ablations/modality_ablation.csv](../results/emotion/emowork/ablations/modality_ablation.csv) |

## Reproducing the headline numbers

```bash
.venv\Scripts\python.exe runs\emowork\train_all.py
.venv\Scripts\python.exe runs\emowork\make_figures.py
.venv\Scripts\python.exe runs\emowork\relaxed_eval.py
.venv\Scripts\python.exe runs\emowork\modality_ablation.py
```

## Caveats carried in the text

- The DANN model is reported as a standalone learner. The current
  soft-vote ensemble was logged before the `DANN_Conformer` key was
  unified with the model name, so the ensemble row excludes DANN.
  This is a logging-key artefact, not a model defect, and is footnoted
  in §5.
- In `relaxed_eval.py`, XGBoost collapses to majority on all binary
  targets (no class weighting in that script). XGBoost's primary
  numbers come from `train_all`, which uses inverse-frequency sample
  weights and is unaffected.
