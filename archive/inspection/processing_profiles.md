# Emotion Processing Profiles (EmoSurv + WESAD)

## Why this exists
The current pipeline supports multiple preprocessing regimes so you can trade off deployment realism, score stability, and runtime.

## Core issues observed
- EmoSurv free typing includes corrupted sentinel-like timing values in interval features (around +-1.58e12), which can distort window statistics if not removed.
- EmoSurv frequency data previously risked leakage when joined with emotion labels; it is now merged by subject + split (fixed/free) and converted to rates.
- Subject-level sparsity is high in EmoSurv (many subjects have very few classes), which inflates fold variance in LOSO.
- Full synthetic augmentation increases non-deep training rows by ~7x and can heavily slow down GradientBoosting.

## Available preprocessing controls
- `--emosurv-normalization`: `zscore|robust` (robust is default for EmoSurv).
- `--clip-z`: clips per-subject z-scored features to [-clip_z, clip_z].
- `--min-subject-samples`: keeps only classification subjects with at least N rows.
- `--min-subject-classes`: keeps only classification subjects with at least K classes.
- `--min-reg-subject-samples`: keeps only regression subjects with at least N rows.
- `--classical-augment`: `none|balance|full` for sklearn classifiers.
- `--regression-augment`: `none|full` for sklearn regressors.

## Suggested profile A: Real-life robustness
Use this when you want stable, deployable behavior and realistic latency.

```bash
x:/dev/Empathic/.venv/Scripts/python.exe train_emotion.py \
  --dataset emosurv --task discrete \
  --class-models LogisticRegression RandomForest SVM GradientBoosting \
  --emosurv-normalization robust \
  --classical-augment none \
  --clip-z 5 \
  --min-subject-samples 20 \
  --min-subject-classes 2
```

## Suggested profile B: Balanced-score push
Use this when you want to push score with moderate extra compute.

```bash
x:/dev/Empathic/.venv/Scripts/python.exe train_emotion.py \
  --dataset emosurv --task discrete \
  --class-models LogisticRegression RandomForest SVM GradientBoosting \
  --emosurv-normalization robust \
  --classical-augment balance \
  --clip-z 5 \
  --min-subject-samples 15 \
  --min-subject-classes 2
```

## Suggested profile C: WESAD dimensional stability check
Use this to reduce highly sparse subject effects before regression LOSO.

```bash
x:/dev/Empathic/.venv/Scripts/python.exe train_emotion.py \
  --dataset wesad --task dimensional \
  --reg-models Ridge RandomForestRegressor SVR \
  --regression-augment none \
  --clip-z 5 \
  --min-reg-subject-samples 80
```

## Quick validation signal (from recent quick runs)
- Loader now reports dropped invalid timing values for EmoSurv.
- On a quick 10-subject EmoSurv logistic check:
  - baseline processing: macro-F1 ~0.290
  - robust profile (clip + subject filtering): macro-F1 ~0.303

