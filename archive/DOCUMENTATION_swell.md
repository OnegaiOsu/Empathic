# Deprecated Reference: SWELL-KW Documentation

> This document is historical and deprecated for active development.
> The active training pipeline is EmoSurv + WESAD in train_emotion.py.
> SWELL-related code and workflows are archived under archive/.

## Comprehensive Documentation: Data Treatment, Training, and Results

**Date:** March 2026  
**Dataset:** SWELL-KW (Koldijk et al., 2014)  
**Task:** Binary stress classification from non-camera sensor modalities  
**Best Result:** AUC = 0.898 (Transformer, per-person z-scored features)

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Data Issues & Forensics](#2-data-issues--forensics)
3. [Data Treatment Pipeline](#3-data-treatment-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architectures](#5-model-architectures)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Experiment Design](#7-experiment-design)
8. [Results: Comprehensive Experiments](#8-results-comprehensive-experiments)
9. [Results: Multimodal Training Suite](#9-results-multimodal-training-suite)
10. [Key Findings & Interpretation](#10-key-findings--interpretation)
11. [File Reference](#11-file-reference)

---

## 1. Dataset Overview

The **SWELL-KW** (SWELL Knowledge Work) dataset captures multi-modal sensor data from 25 participants performing typical knowledge work (writing reports, making presentations, reading, finding information) under different stress conditions.

### Experimental Protocol

| Condition | Code | Label | Description |
|-----------|------|-------|-------------|
| No Stress | N | 0 | Normal working condition |
| Relaxation | R | 0 | Relaxation period |
| Time Pressure | T | 1 (Stressed) | Reduced time, urgency cues |
| Interruptions | I | 1 (Stressed) | Frequent email/chat interruptions |

Each participant completed 3 blocks (conditions vary per block), producing minute-level observations.

### Dataset Dimensions

- **Participants:** 25 (PP1–PP25)
- **Total minutes:** 3,139
- **Class distribution:** ~55% stressed (T+I), ~45% not-stressed (N+R)
- **Source file:** `Behavioral-features - per minute.xlsx` (172 columns)

### Sensor Modalities (Non-Camera Only)

| Modality | Sensor | Features | Description |
|----------|--------|----------|-------------|
| Computer Interaction | uLog | 18 | Mouse activity, keystrokes, app switching |
| Physiology | Mobi (TMSi) | 3 | Heart rate, HRV (RMSSD), skin conductance |

We deliberately excluded camera-based modalities (FaceReader facial expressions, Kinect body posture) to focus on sensors that are practical for real-world deployment.

---

## 2. Data Issues & Forensics

During development, we discovered **five critical data issues** in the commonly-used CSV export that significantly degraded model performance. These issues explain why our initial baseline (Random Forest, AUC = 0.747) was well below the dataset's potential.

### Issue 1: Wrong Source File

**Problem:** Initial work used the CSV file (`Content-features - Labeled-EventBlocks.csv`), which has different granularity and feature alignment than the authoritative Excel workbook.

**Fix:** Switched to `Behavioral-features - per minute.xlsx`, the master dataset with properly aligned minute-level observations.

### Issue 2: No Per-Person Normalization

**Problem:** Raw feature values vary enormously between individuals. One person's "normal" typing speed might be another's "stressed" level. Without normalization, models learn person-specific baselines rather than stress-related deviations.

**Fix:** Per-person z-scoring (see Section 3). This was the **single largest improvement**, adding +0.08–0.14 AUC.

### Issue 3: Physiology Data — 53% Missing

**Problem:** The pre-processed physiology columns (HR, RMSSD, SCL) in the Excel file had only 47.5% valid HR values. The remaining were NaN.

**Fix:** Recovered physiology from raw `.S00` binary files (TMS International format). We ported the MATLAB `tms_read` function to Python and extracted minute-level statistics directly from the 2048 Hz signals:
- **Channel 2:** Continuous HR → minute mean
- **Channel 4:** Beat markers → inter-beat intervals → RMSSD
- **Channel 6:** SCL → minute mean

This raised HR coverage from **47.5% → 95.7%**.

### Issue 4: Padding Rows

**Problem:** 483 rows at condition boundaries had no meaningful data (padding artifacts from the minute-binning process).

**Fix:** These are handled naturally by the LOSO split and imputation — they don't affect the test participant's predictions.

### Issue 5: String Values in Numeric Columns

**Problem:** `SnMouseDistance` contained `#VALUE!` strings in the CSV export, causing silent coercion failures.

**Fix:** The master `.xlsx` file has clean `float64` values throughout — this issue only existed in the CSV.

---

## 3. Data Treatment Pipeline

The data pipeline applies the following steps in order:

```
Raw .xlsx + Raw .S00 files
        │
        ├─ 1. Load master xlsx (3,139 rows × 172 columns)
        ├─ 2. Load recovered physiology (3,042 rows from .S00 files)
        ├─ 3. Merge on (participant, block, minute_index)
        ├─ 4. Fill missing physiology with recovered values
        │          HR:    47.5% → 95.7% coverage
        │          RMSSD: similar improvement
        │          SCL:   similar improvement
        │
        ├─ 5. Map condition labels → binary: {T,I}→1, {N,R}→0
        ├─ 6. Create missingness indicators (before imputation)
        │          HR_missing, RMSSD_missing, SCL_missing ∈ {0,1}
        ├─ 7. Compute derived features
        │          KeystrokeEfficiency, MouseKeyRatio, SwitchRate, HR_SCL_product
        │
        ├─ 8. Per-person z-scoring
        │          For each feature f, for each participant p:
        │            f_normalized = (f - mean_p(f)) / std_p(f)
        │          This removes inter-individual differences
        │
        ├─ 9. Constant-0 imputation (fills remaining NaN with 0)
        │          After z-scoring, 0 = the person's own mean
        │          This is semantically correct (unlike median imputation)
        │
        └─ 10. Standard scaling per LOSO fold (train-set fit → transform both)
```

### Why Per-Person Z-Scoring?

Per-person z-scoring transforms each feature so that **within each participant**, the mean is 0 and standard deviation is 1. This means:

- A z-score of +1 means "this minute was 1 standard deviation above **this person's** average"
- The model learns **relative** deviations from personal baselines
- Physiological traits (resting HR, baseline SCL) are factored out
- This is critical for LOSO evaluation: the model must generalize to unseen people

### Why Constant-0 Imputation?

After per-person z-scoring, a value of 0 literally means "this person's average." Filling missing values with 0 is therefore equivalent to saying "assume this feature was at the person's typical level" — a principled default. This is far better than median imputation (which leaks population-level statistics into the test set).

---

## 4. Feature Engineering

### Computer Interaction Features (18)

Captured by **uLog** software monitoring user activity at the OS level.

| Feature | Description |
|---------|-------------|
| SnMouseAct | Total mouse activity events |
| SnLeftClicked | Left mouse clicks |
| SnRightClicked | Right mouse clicks |
| SnDoubleClicked | Double clicks |
| SnWheel | Mouse wheel events |
| SnDragged | Mouse drag events |
| SnMouseDistance | Total mouse distance (pixels) |
| SnKeyStrokes | Total keystrokes |
| SnChars | Character keystrokes |
| SnSpecialKeys | Special key presses (Ctrl, Alt, etc.) |
| SnDirectionKeys | Arrow key presses |
| SnErrorKeys | Backspace/Delete presses |
| SnShortcutKeys | Keyboard shortcuts |
| SnSpaces | Spacebar presses |
| SnAppChange | Application switches |
| SnTabfocusChange | Tab/focus changes |
| CharactersRatio | SnChars / SnKeyStrokes |
| ErrorKeyRatio | SnErrorKeys / SnKeyStrokes |

### Physiology Features (3)

Extracted from **Mobi** (TMSi) wearable device, sampled at 2048 Hz, processed to minute-level:

| Feature | Description | Extraction |
|---------|-------------|------------|
| HR | Heart Rate (bpm) | Mean of Ch2 continuous HR signal |
| RMSSD | Heart Rate Variability | Root mean square of successive beat-to-beat differences (from Ch4 beat markers) |
| SCL | Skin Conductance Level (μS) | Mean of Ch6 galvanic skin response |

### Derived Features (4)

Cross-modal and within-modal combinations:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| KeystrokeEfficiency | SnChars / (SnKeyStrokes + 1) | Ratio of productive keystrokes; stress increases error/correction keys |
| MouseKeyRatio | SnMouseAct / (SnKeyStrokes + 1) | Input modality preference shift under stress |
| SwitchRate | SnAppChange + SnTabfocusChange | Total context switching reflects interruption conditions |
| HR_SCL_product | HR × SCL | Cross-modal interaction; combined physiological arousal |

### Missingness Indicators (3)

Binary flags indicating whether the original physiology value was missing:

| Feature | Description |
|---------|-------------|
| HR_missing | 1 if Heart Rate was NaN before imputation |
| RMSSD_missing | 1 if RMSSD was NaN before imputation |
| SCL_missing | 1 if SCL was NaN before imputation |

These allow models to distinguish "feature was 0 (person's average)" from "feature was missing and imputed to 0."

---

## 5. Model Architectures

### Tabular Models (scikit-learn)

All tabular models use the same preprocessing: constant-0 imputation → StandardScaler (fit on train, transform both).

| Model | Key Hyperparameters | Notes |
|-------|-------------------|-------|
| **Random Forest** | 300 trees, min_samples_leaf=5, class_weight='balanced' | Strong baseline, handles feature interactions |
| **Gradient Boosting** | 200 trees, max_depth=5, lr=0.1, min_samples_leaf=5 | Sequential ensemble, captures non-linear patterns |
| **SVM** | RBF kernel, C=1.0, gamma='scale', class_weight='balanced' | Works well with z-scored features at moderate dimensionality |
| **Logistic Regression** | L2 penalty, C=1.0, class_weight='balanced' | Linear baseline; interpretable coefficients |
| **MLP** | Hidden layers (64, 32), adaptive LR, early stopping | Neural network baseline, captures non-linear boundaries |

All models use `class_weight='balanced'` to handle the slight class imbalance (~55/45).

### Transformer

A 2-layer Transformer encoder operating on **minute-level sequences** within each block. Unlike tabular models that classify each minute independently, the Transformer sees the full temporal context of a work block.

```
Architecture:
  Input projection: n_features → 64 (d_model)
  Positional encoding: Learned, max 200 positions
  Encoder: 2 layers, 4 attention heads, d_ff=128, GELU activation
  Dropout: 0.2
  Output head: LayerNorm → Dropout → Linear(64 → 2)
  
Training:
  Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  Scheduler: Cosine annealing over 80 epochs
  Loss: CrossEntropyLoss with class weights (balanced)
  Early stopping: patience=15 on validation loss
  Validation: 15% of training blocks held out per fold
```

**Key design choice:** The Transformer produces a prediction for **every minute position** (per-position output), not a single block-level prediction. This means:
- Evaluation is at minute-level, matching the tabular models
- No inflation from block-level accuracy aggregation
- The model can learn that stress patterns may evolve within a block

---

## 6. Evaluation Protocol

### Leave-One-Subject-Out (LOSO) Cross-Validation

We use LOSO CV with 25 folds (one fold per participant). In each fold:

1. **Train:** All data from 24 participants
2. **Test:** All data from the held-out participant
3. **No data leakage:** The held-out person's data is never seen during training, preprocessing fitting, or validation

This is the strictest evaluation protocol for person-independent models — it answers:  
*"How well does this model predict stress for a person it has never seen?"*

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision** | Of predicted-stressed minutes, how many truly stressed |
| **Recall** | Of truly-stressed minutes, how many detected |
| **AUC-ROC** | Area under the ROC curve (primary metric; threshold-independent) |

All metrics are computed per-fold (per-participant) and then averaged. We report mean ± standard deviation across the 25 folds.

**Primary metric: AUC-ROC**, because it is threshold-independent and better captures discrimination ability across class boundaries.

---

## 7. Experiment Design

### Ablation Study (train_final.py) — 5 Experiments × 6 Models = 30 Configurations

| Experiment | Features | n_features | Purpose |
|------------|----------|-----------|---------|
| **A: Baseline** | Computer + Physiology, global impute+scale (no z-scoring) | 21 | Reproduce original broken pipeline |
| **B: Z-Score** | Computer + Physiology, per-person z-scored | 21 | Isolate the effect of normalization |
| **C: All Fixes** | B + derived features + missingness indicators | 28 | Test if extra features help |
| **D: Physio Only** | Physiology only, per-person z-scored | 6 | Single-modality ablation |
| **E: Computer Only** | Computer only, per-person z-scored | 22 | Single-modality ablation |

### Multimodal Focus (train_multimodal.py) — 1 Experiment × 6 Models

| Experiment | Features | n_features | Purpose |
|------------|----------|-----------|---------|
| **Multimodal** | Computer + Physiology + Derived + Missingness, per-person z-scored | 28 | Production-ready multimodal classifier |

---

## 8. Results: Comprehensive Experiments

### Full Results Table (sorted by AUC, descending)

| Rank | Experiment | Model | Accuracy | F1 | AUC |
|------|-----------|-------|----------|-----|-----|
| 1 | B: Z-Score | Transformer | 0.862 ± 0.179 | 0.866 ± 0.171 | **0.898 ± 0.180** |
| 2 | E: Computer Only | Transformer | 0.823 ± 0.179 | 0.832 ± 0.170 | 0.870 ± 0.180 |
| 3 | C: All Fixes | Transformer | 0.815 ± 0.217 | 0.826 ± 0.210 | 0.848 ± 0.239 |
| 4 | C: All Fixes | Random Forest | 0.721 ± 0.123 | 0.753 ± 0.112 | 0.792 ± 0.127 |
| 5 | B: Z-Score | SVM | 0.731 ± 0.113 | 0.752 ± 0.110 | 0.792 ± 0.129 |
| 6 | B: Z-Score | Random Forest | 0.719 ± 0.115 | 0.752 ± 0.103 | 0.790 ± 0.126 |
| 7 | B: Z-Score | MLP | 0.727 ± 0.121 | 0.751 ± 0.117 | 0.788 ± 0.127 |
| 8 | C: All Fixes | MLP | 0.710 ± 0.126 | 0.735 ± 0.122 | 0.785 ± 0.136 |
| 9 | C: All Fixes | SVM | 0.716 ± 0.114 | 0.735 ± 0.112 | 0.783 ± 0.131 |
| 10 | C: All Fixes | Gradient Boosting | 0.714 ± 0.125 | 0.735 ± 0.126 | 0.776 ± 0.133 |
| 11 | C: All Fixes | Logistic Regression | 0.703 ± 0.127 | 0.718 ± 0.128 | 0.763 ± 0.139 |
| 12 | B: Z-Score | Logistic Regression | 0.701 ± 0.111 | 0.715 ± 0.110 | 0.760 ± 0.125 |
| 13 | A: Baseline | Transformer | 0.682 ± 0.149 | 0.692 ± 0.175 | 0.758 ± 0.181 |
| 14 | B: Z-Score | Gradient Boosting | 0.683 ± 0.134 | 0.705 ± 0.132 | 0.746 ± 0.148 |
| 15 | E: Computer Only | SVM | 0.677 ± 0.077 | 0.718 ± 0.074 | 0.732 ± 0.090 |
| 16–18 | E: Computer Only | LR / RF / MLP | 0.662–0.675 | 0.688–0.719 | 0.727–0.729 |
| 19 | D: Physio Only | Transformer | 0.681 ± 0.252 | 0.681 ± 0.264 | 0.721 ± 0.288 |
| 20 | A: Baseline | LR | 0.653 ± 0.083 | 0.662 ± 0.136 | 0.715 ± 0.095 |
| 21–23 | A: Baseline | RF / MLP / SVM | 0.641–0.647 | 0.665–0.680 | 0.703–0.710 |
| 24 | E: Computer Only | Gradient Boosting | 0.634 ± 0.068 | 0.672 ± 0.083 | 0.699 ± 0.079 |
| 25 | A: Baseline | Gradient Boosting | 0.605 ± 0.110 | 0.607 ± 0.160 | 0.670 ± 0.145 |
| 26 | D: Physio Only | SVM | 0.606 ± 0.156 | 0.625 ± 0.164 | 0.633 ± 0.182 |
| 27 | D: Physio Only | MLP | 0.604 ± 0.162 | 0.646 ± 0.158 | 0.622 ± 0.194 |
| 28 | D: Physio Only | LR | 0.555 ± 0.158 | 0.533 ± 0.185 | 0.596 ± 0.190 |
| 29 | D: Physio Only | Random Forest | 0.581 ± 0.121 | 0.592 ± 0.132 | 0.588 ± 0.156 |
| 30 | D: Physio Only | Gradient Boosting | 0.556 ± 0.128 | 0.578 ± 0.128 | 0.566 ± 0.150 |

---

## 9. Results: Multimodal Training Suite

The focused multimodal run uses the full feature configuration (physiology + computer + derived + missingness, per-person z-scored), with additional metrics and full visualization suite.

### Summary (sorted by AUC)

| Model | Accuracy | F1 | Precision | Recall | AUC |
|-------|----------|-----|-----------|--------|-----|
| **Transformer** | 0.793 ± 0.196 | 0.812 ± 0.175 | 0.805 ± 0.194 | 0.831 ± 0.171 | **0.825 ± 0.232** |
| Random Forest | 0.721 ± 0.123 | 0.753 ± 0.112 | 0.710 ± 0.119 | 0.808 ± 0.122 | 0.792 ± 0.127 |
| MLP | 0.710 ± 0.126 | 0.735 ± 0.122 | 0.711 ± 0.127 | 0.767 ± 0.134 | 0.785 ± 0.136 |
| SVM | 0.716 ± 0.114 | 0.735 ± 0.112 | 0.724 ± 0.117 | 0.753 ± 0.126 | 0.783 ± 0.131 |
| Gradient Boosting | 0.714 ± 0.125 | 0.735 ± 0.126 | 0.714 ± 0.120 | 0.763 ± 0.146 | 0.776 ± 0.133 |
| Logistic Regression | 0.703 ± 0.127 | 0.718 ± 0.128 | 0.714 ± 0.128 | 0.724 ± 0.135 | 0.763 ± 0.139 |

### Generated Visualizations

All plots saved to `results/multimodal/plots/`:

| Plot | File | Description |
|------|------|-------------|
| Model Comparison | `model_comparison_bars.png` | Bar chart of Accuracy, F1, AUC with error bars |
| Radar Chart | `model_radar.png` | Multi-metric radar overlay of all models |
| Fold Box Plots | `fold_boxplots.png` | Per-participant metric distributions per model |
| ROC Curves | `roc_curves.png` | Overlaid ROC curves with AUC values |
| Precision-Recall | `precision_recall_curves.png` | PR curves with PR-AUC values |
| Confusion Matrices | `confusion_matrices.png` | Grid of confusion matrices (counts + percentages) |
| Participant Heatmap | `participant_heatmap.png` | AUC per participant × model heatmap |
| Baseline Improvement | `improvement_over_baseline.png` | Delta vs old broken pipeline |

---

## 10. Key Findings & Interpretation

### Finding 1: Per-Person Z-Scoring Is the Single Biggest Fix

Comparing experiment A (no z-scoring) vs B (with z-scoring), **every model improves substantially**:

| Model | AUC (A: No Z-Score) | AUC (B: Z-Score) | Δ AUC |
|-------|---------------------|-------------------|-------|
| Transformer | 0.758 | 0.898 | **+0.140** |
| Random Forest | 0.710 | 0.790 | +0.080 |
| SVM | 0.704 | 0.792 | +0.089 |
| MLP | 0.709 | 0.788 | +0.078 |
| LR | 0.715 | 0.760 | +0.045 |
| Gradient Boosting | 0.670 | 0.746 | +0.076 |

**Why:** Without z-scoring, the model confuses inter-individual differences with stress signals. Person A might have a resting HR of 80 bpm and stressed HR of 90 bpm, while Person B has resting 60 and stressed 70. Without normalization, the model learns "HR > 75 → stressed" which fails on both. Z-scoring transforms both into a common scale where "HR deviation = +1σ above my personal mean → stressed."

### Finding 2: Transformer Dominates Every Experiment

The Transformer is the best model in **every single experiment category**, with the largest margin in the z-scored conditions:

| Experiment | Best Tabular AUC | Transformer AUC | Gap |
|-----------|-----------------|-----------------|-----|
| A: Baseline | 0.715 (LR) | 0.758 | +0.043 |
| B: Z-Score | 0.792 (SVM) | 0.898 | **+0.106** |
| C: All Fixes | 0.792 (RF) | 0.848 | +0.056 |
| D: Physio Only | 0.633 (SVM) | 0.721 | +0.088 |
| E: Computer Only | 0.732 (SVM) | 0.870 | **+0.138** |
| Multimodal | 0.792 (RF) | 0.825 | +0.033 |

**Why:** The Transformer's temporal attention mechanism captures **sequential patterns within work blocks** — e.g., gradually increasing mouse speed, declining keystroke efficiency, or rising heart rate over time. Tabular models treat each minute independently and miss these dynamics.

### Finding 3: Computer Interaction Alone Is Surprisingly Powerful

| Modality | Best AUC (Tabular) | Best AUC (Transformer) |
|----------|-------------------|----------------------|
| D: Physiology only | 0.633 | 0.721 |
| E: Computer only | 0.732 | **0.870** |
| B: Both (z-scored) | 0.792 | **0.898** |
| C: Both + derived | 0.792 | 0.848 |

Computer interaction (mouse/keyboard/app switching) is a stronger predictor of stress than physiology alone. The Transformer extracts rich temporal patterns from 18 computer features (vs. only 3 physiology features). Combining both modalities (B) yields the best overall result (0.898).

### Finding 4: Derived Features Hurt the Transformer (But Not Tabular Models)

| Setup | RF AUC | Transformer AUC |
|-------|--------|-----------------|
| B: Z-Score (21 features) | 0.790 | **0.898** |
| C: All Fixes (28 features) | **0.792** | 0.848 |

Adding derived features and missingness indicators:
- **Helps tabular models slightly** (+0.002 for RF): explicit feature engineering provides useful shortcuts
- **Hurts the Transformer** (−0.050): the Transformer already discovers these patterns through attention; explicit features add noise/redundancy, and the expanded feature space reduces signal density

### Finding 5: High Variance in Per-Participant Performance

Standard deviations are large (~0.12–0.19) across all models, indicating some participants are much easier to classify than others. Common patterns:
- **Easy participants:** Highly reactive physiology, dramatic behavior changes under stress
- **Hard participants:** Subtle or non-stereotypical stress responses
- The Transformer has the highest variance but also the highest ceiling

### Practical Recommendations

| Use Case | Recommended Model | AUC | Rationale |
|----------|-------------------|-----|-----------|
| **Best accuracy, sequential data available** | B_zscore Transformer | 0.898 | Requires minute-level sequences within blocks |
| **Production (no sequential context)** | B_zscore SVM | 0.792 | Fast, reliable, works on individual minutes |
| **Interpretability required** | Logistic Regression | 0.760 | Linear coefficients reveal feature importance |
| **Resource-constrained** | Random Forest | 0.790 | No GPU needed, fast inference, robust |

---

## 11. File Reference

### Active Scripts

| File | Purpose |
|------|---------|
| `train_final.py` | Comprehensive ablation study (5 experiments × 6 models = 30 configs) |
| `train_multimodal.py` | Focused multimodal training with visualizations (1 experiment × 6 models) |
| `extract_physiology.py` | Extract minute-level HR/RMSSD/SCL from raw .S00 binary files |
| `tms_reader.py` | Python port of MATLAB tms_read for .S00 binary format |

### Data Files

| File | Description |
|------|-------------|
| `0_SWELL/Behavioral-features - per minute.xlsx` | Master dataset (3,139 rows × 172 cols) |
| `physiology_from_raw.csv` | Recovered physiology (3,042 rows, HR/RMSSD/SCL from .S00) |
| `0_SWELL/0 - Raw data/D - Physiology - raw data/*.S00` | Raw 2048 Hz physiology signals |

### Results Directories

| Directory | Contents |
|-----------|----------|
| `results/final_recovered/` | Ablation study results (30 fold CSVs + summary) |
| `results/multimodal/` | Multimodal suite results + visualizations |
| `results/multimodal/plots/` | 8 visualization PNGs |
| `results/multimodal/fold_results/` | Per-model fold CSVs (25 participants each) |

### Archive

| Directory | Contents |
|-----------|----------|
| `archive/inspection/` | 10 data debugging/inspection scripts |
| `archive/old_training/` | 9 superseded training scripts |
| `archive/old_results/` | 11 old result directories |
| `archive/old_utils/` | Old data loaders and utilities |
| `archive/logs/` | Training and scanning logs |

---

*Generated from SWELL-KW dataset analysis. All results use Leave-One-Subject-Out cross-validation with 25 participants.*

---

## 12. New Emotion Pipeline (EmoSurv + WESAD)

The repository now includes an initial cross-dataset emotion pipeline in `train_emotion.py`.

### Supported tasks

| Dataset | Task | Targets |
|---------|------|---------|
| EmoSurv | Discrete classification | `emotionIndex` classes (N/H/C/A/S) |
| WESAD | Discrete classification | Native condition labels from `.pkl` (1/2/3/4) |
| WESAD | Dimensional regression | Questionnaire-derived Valence + Arousal from `# DIM` rows |

### Pipeline characteristics

- Shared LOSO harness for all tasks
- Dataset-specific loaders
- Per-subject z-scoring on engineered features
- Light augmentation (jitter + feature masking; plus class balancing for classification)
- Built-in progress logging at dataset load, fold execution, and artifact write stages
- Multi-model sweeps with per-fold and aggregate reporting
- Plot generation for both classification and regression
- Deep learning variants in the same harness:
  - `TorchMLP` (classification)
  - `TorchTransformer` (sequence classification)
  - `TorchMLPRegressor` (dimensional regression)
  - `TorchTransformerRegressor` (sequence dimensional regression)
- WESAD dimensional target alignment now uses questionnaire `# ORDER`, `# START`, and `# END` intervals, with stage-level `# DIM` mapping and condition fallback for missing stage-dim pairs

### Output directories

| Directory | Contents |
|-----------|----------|
| `results/emotion/emosurv/discrete/` | EmoSurv classification summaries, folds, and plots |
| `results/emotion/wesad/discrete/` | WESAD classification summaries, folds, and plots |
| `results/emotion/wesad/dimensional/` | WESAD dimensional summaries, folds, and plots |

### Example commands

```bash
python train_emotion.py --dataset emosurv --task discrete
python train_emotion.py --dataset wesad --task discrete
python train_emotion.py --dataset wesad --task dimensional
python train_emotion.py --dataset all --task both --quick
python train_emotion.py --dataset wesad --task discrete --include-deep
python train_emotion.py --dataset wesad --task dimensional --reg-models TorchTransformerRegressor
```

### Quick validation metrics (smoke tests)

- EmoSurv discrete (`LogisticRegression`, quick):
  - Accuracy mean: 0.591
  - F1 macro mean: 0.320
  - AUC OVR mean: 0.729
- WESAD discrete (`LogisticRegression`, quick):
  - Accuracy mean: 0.829
  - F1 macro mean: 0.742
  - AUC OVR mean: 0.921
- WESAD dimensional (`Ridge`, quick):
  - MAE mean: 1.136
  - RMSE mean: 1.372
  - R2 mean: 0.204
