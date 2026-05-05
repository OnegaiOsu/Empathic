"""
Final training pipeline — fixed version.

Data sources:
  1. Behavioral-features xlsx (3139 rows x 172 cols, all sensors merged)
  2. Recovered physiology from raw .S00 files (HR coverage 47.5% -> 95.7%)

Key fixes over previous version:
  - SimpleImputer strategy='constant', fill_value=0 (not median)
    After per-person z-scoring, 0 = person's mean -> much better than global median
  - Per-person z-scoring of ALL features (not just physiology)
  - Missingness indicators for HR/RMSSD/SCL (the model should know what's missing)
  - Proper derived feature engineering

Experiments:
  A. Baseline (global impute+scale, no z-scoring)
  B. Per-person z-scored (all features)
  C. All + derived features + missingness indicators
  D. Physiology-only (per-person z)
  E. Computer-only

Models: RF, GradientBoosting, SVM, LR, MLP, Transformer x LOSO CV

The Transformer operates at the minute level (per-position output from
block-length sequences) for fair comparison with tabular models. Early
stopping uses a held-out 15% of training blocks, not the test set.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Configuration ----------------------------------------------------------
XLSX_PATH = "0_SWELL/Behavioral-features - per minute.xlsx"
EXTRACTED_PHYSIO_PATH = "physiology_from_raw.csv"
RESULTS_DIR = "results/final_recovered"

# Computer interaction features (non-camera)
COMPUTER_FEATURES = [
    'SnMouseAct', 'SnLeftClicked', 'SnRightClicked', 'SnDoubleClicked',
    'SnWheel', 'SnDragged', 'SnMouseDistance', 'SnKeyStrokes',
    'SnChars', 'SnSpecialKeys', 'SnDirectionKeys', 'SnErrorKeys',
    'SnShortcutKeys', 'SnSpaces', 'SnAppChange', 'SnTabfocusChange',
    'CharactersRatio', 'ErrorKeyRatio',
]

# Physiology features
PHYSIO_FEATURES = ['HR', 'RMSSD', 'SCL']

# Derived features we'll engineer
DERIVED_FEATURES = [
    'KeystrokeEfficiency',  # SnChars / (SnKeyStrokes + 1)
    'MouseKeyRatio',        # SnMouseAct / (SnKeyStrokes + 1)
    'SwitchRate',           # SnAppChange + SnTabfocusChange
    'HR_SCL_product',       # HR * SCL (physio interaction)
]

# Binary label mapping: {T, I} = Stressed (1), {N, R} = Not Stressed (0)
STRESS_MAP = {'T': 1, 'I': 1, 'N': 0, 'R': 0}


# --- Data Loading -----------------------------------------------------------
def load_and_prepare_data():
    """Load master xlsx, merge recovered physiology, return clean DataFrame."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # 1. Load master dataset
    df = pd.read_excel(XLSX_PATH, sheet_name='SWELLdata')
    print(f"Master dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Load extracted physiology from .S00 files
    extracted = pd.read_csv(EXTRACTED_PHYSIO_PATH)
    print(f"Extracted physiology: {extracted.shape[0]} rows")

    # 3. Create PP number for merging
    df['PP_num'] = df['PP'].str.replace('PP', '').astype(int)

    # 4. Add minute index within each (PP, Blok) group for alignment
    df['minute_idx'] = df.groupby(['PP_num', 'Blok']).cumcount()
    extracted['minute_idx'] = extracted.groupby(['PP', 'block']).cumcount()

    # 5. Merge extracted physiology
    extracted_merge = extracted[['PP', 'block', 'minute_idx', 'HR', 'RMSSD', 'SCL']].copy()
    extracted_merge.columns = ['PP_num', 'Blok', 'minute_idx',
                               'HR_extracted', 'RMSSD_extracted', 'SCL_extracted']

    df = df.merge(extracted_merge, on=['PP_num', 'Blok', 'minute_idx'], how='left')

    # 6. Fill missing physiology with extracted values
    hr_before = df['HR'].notna().sum()
    rmssd_before = df['RMSSD'].notna().sum()
    scl_before = df['SCL'].notna().sum()

    df['HR'] = df['HR'].fillna(df['HR_extracted'])
    df['RMSSD'] = df['RMSSD'].fillna(df['RMSSD_extracted'])
    df['SCL'] = df['SCL'].fillna(df['SCL_extracted'])

    hr_after = df['HR'].notna().sum()
    rmssd_after = df['RMSSD'].notna().sum()
    scl_after = df['SCL'].notna().sum()

    print(f"\nPhysiology recovery:")
    print(f"  HR:    {hr_before} -> {hr_after} valid "
          f"({hr_before/len(df)*100:.1f}% -> {hr_after/len(df)*100:.1f}%)")
    print(f"  RMSSD: {rmssd_before} -> {rmssd_after} valid "
          f"({rmssd_before/len(df)*100:.1f}% -> {rmssd_after/len(df)*100:.1f}%)")
    print(f"  SCL:   {scl_before} -> {scl_after} valid "
          f"({scl_before/len(df)*100:.1f}% -> {scl_after/len(df)*100:.1f}%)")

    # 7. Coerce all feature columns to numeric (safety)
    for col in COMPUTER_FEATURES + PHYSIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 8. Create binary stress label
    df['stress_label'] = df['Condition'].map(STRESS_MAP)
    df = df.dropna(subset=['stress_label'])
    df['stress_label'] = df['stress_label'].astype(int)

    # 9. Report per-participant data quality
    print(f"\nPer-participant data quality:")
    print(f"  {'PP':<6s} {'Rows':>5s} {'HR%':>6s} {'SCL%':>6s} {'Comp%':>6s} "
          f"{'Stress%':>8s}")
    for pp in sorted(df['PP_num'].unique()):
        pp_data = df[df['PP_num'] == pp]
        n = len(pp_data)
        hr_pct = 100 * pp_data['HR'].notna().mean()
        scl_pct = 100 * pp_data['SCL'].notna().mean()
        comp_cols = [c for c in COMPUTER_FEATURES if c in df.columns]
        comp_pct = 100 * pp_data[comp_cols].notna().all(axis=1).mean()
        stress_pct = 100 * pp_data['stress_label'].mean()
        print(f"  PP{pp:<4d} {n:5d} {hr_pct:5.1f}% {scl_pct:5.1f}% "
              f"{comp_pct:5.1f}% {stress_pct:6.1f}%")

    print(f"\nLabel distribution: "
          f"Stressed={df['stress_label'].sum()} "
          f"Not-stressed={(df['stress_label']==0).sum()}")

    return df


# --- Feature Engineering ----------------------------------------------------
def add_missingness_indicators(df):
    """Add binary flags showing which physiology values were originally missing."""
    df = df.copy()
    for feat in PHYSIO_FEATURES:
        df[f'{feat}_missing'] = df[feat].isna().astype(int)
    return df


def per_person_zscore(df, features):
    """Z-score features within each participant.

    Critical because:
    - SCL varies from 65 (PP1) to 921 (PP9) across participants
    - HR baselines differ by 20+ bpm
    - The stress signal is in WITHIN-PERSON changes, not absolute values

    After z-scoring, remaining NaN (from all-NaN participants) -> 0.
    """
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        grouped = df.groupby('PP_num')[feat]
        pp_mean = grouped.transform('mean')
        pp_std = grouped.transform('std')
        pp_std = pp_std.replace(0, np.nan)  # avoid div by zero
        df[feat] = (df[feat] - pp_mean) / pp_std
        df[feat] = df[feat].fillna(0)  # zero-std or all-NaN -> 0
    return df


def add_derived_features(df):
    """Add interaction and ratio features."""
    df = df.copy()

    # Typing efficiency: how many characters per keystroke
    if 'SnKeyStrokes' in df.columns and 'SnChars' in df.columns:
        df['KeystrokeEfficiency'] = df['SnChars'] / (df['SnKeyStrokes'] + 1)

    # Mouse vs keyboard activity ratio
    if 'SnMouseAct' in df.columns and 'SnKeyStrokes' in df.columns:
        df['MouseKeyRatio'] = df['SnMouseAct'] / (df['SnKeyStrokes'] + 1)

    # App/tab switching rate (proxy for multitasking or distraction)
    if 'SnAppChange' in df.columns and 'SnTabfocusChange' in df.columns:
        df['SwitchRate'] = df['SnAppChange'] + df['SnTabfocusChange']

    # HR x SCL interaction (cross-modality physiology signal)
    if all(f in df.columns for f in ['HR', 'SCL']):
        df['HR_SCL_product'] = df['HR'] * df['SCL']

    return df


# --- Transformer Model (minute-level) ---------------------------------------
class StressTransformer(nn.Module):
    """Transformer encoder that outputs per-minute stress predictions.

    Unlike the old block-level approach (which gave 1 label per block and
    inflated accuracy), this produces a prediction for every minute position
    in the sequence, allowing fair comparison with tabular models.
    """
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.2, max_len=200):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x, src_key_padding_mask=None):
        """x: (batch, seq_len, n_features) -> (batch, seq_len, 2)"""
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(x)  # per-position logits


class BlockSequenceDataset(Dataset):
    """Each item is one block's minute-sequence + per-minute labels."""
    def __init__(self, sequences, label_seqs):
        self.sequences = sequences
        self.label_seqs = label_seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor(self.label_seqs[idx]),
        )


def collate_blocks(batch):
    """Pad block sequences to max length, create attention mask."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    n_features = seqs[0].shape[1]

    padded_x = torch.zeros(len(seqs), max_len, n_features)
    padded_y = torch.full((len(seqs), max_len), -100, dtype=torch.long)  # -100 = ignore
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)  # True = padded

    for i, (seq, lab, length) in enumerate(zip(seqs, labels, lengths)):
        padded_x[i, :length] = seq
        padded_y[i, :length] = lab
        mask[i, :length] = False

    return padded_x, mask, padded_y, lengths


def _build_block_sequences(df, features):
    """Build one sequence per (participant, block). Returns per-minute labels."""
    sequences, label_seqs, pp_ids = [], [], []
    available = [f for f in features if f in df.columns]

    for (pp, blok), group in df.groupby(['PP_num', 'Blok']):
        group = group.sort_index()
        seq = group[available].values.astype(float)
        labs = group['stress_label'].values.astype(int)
        # Replace NaN with 0 in sequences
        seq = np.nan_to_num(seq, nan=0.0)
        if len(seq) < 2:
            continue
        sequences.append(seq)
        label_seqs.append(labs)
        pp_ids.append(pp)

    return sequences, label_seqs, pp_ids


def run_transformer_loso(df, features, exp_name,
                         d_model=64, n_heads=4, n_layers=2, d_ff=128,
                         dropout=0.2, lr=1e-3, epochs=80, patience=15,
                         batch_size=16):
    """Run Transformer with LOSO CV, evaluating at the MINUTE level.

    For each fold:
      1. Build block-sequences for train/test
      2. Fit scaler on flattened train sequences
      3. Train Transformer with per-position CrossEntropy
      4. Collect per-minute predictions from test blocks
      5. Report minute-level accuracy/f1/auc

    Early stopping uses 15% of training blocks (not test data).
    """
    sequences, label_seqs, pp_ids = _build_block_sequences(df, features)
    participants = sorted(set(pp_ids))
    n_features = sequences[0].shape[1]

    fold_results = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for held_out in participants:
        # --- split ---
        train_seqs, train_labs = [], []
        test_seqs, test_labs = [], []
        for seq, lab, pp in zip(sequences, label_seqs, pp_ids):
            if pp == held_out:
                test_seqs.append(seq)
                test_labs.append(lab)
            else:
                train_seqs.append(seq)
                train_labs.append(lab)

        if len(test_seqs) == 0:
            continue

        # --- scale (fit on train) ---
        all_train_flat = np.vstack(train_seqs)
        scaler = StandardScaler()
        scaler.fit(all_train_flat)

        train_seqs = [scaler.transform(s) for s in train_seqs]
        test_seqs = [scaler.transform(s) for s in test_seqs]

        # --- validation split (15% of train blocks) ---
        n_val = max(1, int(len(train_seqs) * 0.15))
        rng = np.random.RandomState(42)
        val_idx = rng.choice(len(train_seqs), n_val, replace=False)
        tr_idx = [i for i in range(len(train_seqs)) if i not in val_idx]

        val_seqs = [train_seqs[i] for i in val_idx]
        val_labs = [train_labs[i] for i in val_idx]
        tr_seqs = [train_seqs[i] for i in tr_idx]
        tr_labs = [train_labs[i] for i in tr_idx]

        train_ds = BlockSequenceDataset(tr_seqs, tr_labs)
        val_ds = BlockSequenceDataset(val_seqs, val_labs)
        test_ds = BlockSequenceDataset(test_seqs, test_labs)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_blocks)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds),
                                shuffle=False, collate_fn=collate_blocks)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds),
                                 shuffle=False, collate_fn=collate_blocks)

        # --- class weights ---
        all_train_labels = np.concatenate(tr_labs)
        n_pos = (all_train_labels == 1).sum()
        n_neg = (all_train_labels == 0).sum()
        weight = torch.FloatTensor(
            [1.0, n_neg / max(n_pos, 1)]
        ).to(DEVICE)

        # --- model ---
        model = StressTransformer(
            n_features=n_features, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

        # --- training with early stopping on val ---
        best_val_loss = float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(epochs):
            model.train()
            for X_b, mask_b, y_b, _ in train_loader:
                X_b, mask_b, y_b = (
                    X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
                )
                optimizer.zero_grad()
                logits = model(X_b, src_key_padding_mask=mask_b)
                loss = criterion(
                    logits.reshape(-1, 2), y_b.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # val loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_b, mask_b, y_b, _ in val_loader:
                    X_b, mask_b, y_b = (
                        X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
                    )
                    logits = model(X_b, src_key_padding_mask=mask_b)
                    val_loss += criterion(
                        logits.reshape(-1, 2), y_b.reshape(-1)
                    ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        # --- evaluate on test (minute-level) ---
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        fold_y_true, fold_y_pred, fold_y_prob = [], [], []
        with torch.no_grad():
            for X_b, mask_b, y_b, lengths in test_loader:
                X_b, mask_b = X_b.to(DEVICE), mask_b.to(DEVICE)
                logits = model(X_b, src_key_padding_mask=mask_b)
                probs = torch.softmax(logits, dim=-1)

                for i, length in enumerate(lengths):
                    y_true_i = y_b[i, :length].numpy()
                    y_pred_i = logits[i, :length].argmax(dim=-1).cpu().numpy()
                    y_prob_i = probs[i, :length, 1].cpu().numpy()
                    fold_y_true.extend(y_true_i)
                    fold_y_pred.extend(y_pred_i)
                    fold_y_prob.extend(y_prob_i)

        fold_y_true = np.array(fold_y_true)
        fold_y_pred = np.array(fold_y_pred)
        fold_y_prob = np.array(fold_y_prob)

        if len(np.unique(fold_y_true)) < 2 or len(fold_y_true) < 5:
            continue

        acc = accuracy_score(fold_y_true, fold_y_pred)
        f1 = f1_score(fold_y_true, fold_y_pred)
        try:
            auc = roc_auc_score(fold_y_true, fold_y_prob)
        except ValueError:
            auc = np.nan

        fold_results.append({
            'participant': held_out,
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc,
            'n_test': len(fold_y_true),
            'n_stressed': int(fold_y_true.sum()),
        })

        all_y_true.extend(fold_y_true)
        all_y_pred.extend(fold_y_pred)
        all_y_prob.extend(fold_y_prob)

    results_df = pd.DataFrame(fold_results)
    model_name = f"{exp_name}_Transformer"

    summary = {
        'model': model_name,
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'f1_mean': results_df['f1_score'].mean(),
        'f1_std': results_df['f1_score'].std(),
        'auc_mean': results_df['auc'].mean(),
        'auc_std': results_df['auc'].std(),
        'n_folds': len(results_df),
        'total_test': len(all_y_true),
    }

    return summary, results_df, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


# --- LOSO Cross-Validation --------------------------------------------------
def run_loso_cv(df, features, model_fn, model_name):
    """Run Leave-One-Subject-Out cross-validation.

    Imputation: strategy='constant', fill_value=0
    After per-person z-scoring, 0 = person's mean -- much better than global median.
    For non-z-scored experiments, StandardScaler handles centering anyway.
    """
    participants = sorted(df['PP_num'].unique())

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for pp in participants:
        # Split
        test_mask = df['PP_num'] == pp
        train_mask = ~test_mask

        # Only use features that actually exist in the DataFrame
        available_feats = [f for f in features if f in df.columns]

        X_train = df.loc[train_mask, available_feats].values.astype(float)
        y_train = df.loc[train_mask, 'stress_label'].values
        X_test = df.loc[test_mask, available_feats].values.astype(float)
        y_test = df.loc[test_mask, 'stress_label'].values

        if len(np.unique(y_test)) < 2 or len(y_test) < 5:
            continue

        # Impute remaining NaN with 0 (after z-scoring 0 = person's mean)
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Scale (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        model = model_fn()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, 'predict_proba') else None)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        fold_results.append({
            'participant': pp,
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc,
            'n_test': len(y_test),
            'n_stressed': int(y_test.sum()),
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        if y_prob is not None:
            all_y_prob.extend(y_prob)

    # Aggregate
    results_df = pd.DataFrame(fold_results)

    summary = {
        'model': model_name,
        'accuracy_mean': results_df['accuracy'].mean(),
        'accuracy_std': results_df['accuracy'].std(),
        'f1_mean': results_df['f1_score'].mean(),
        'f1_std': results_df['f1_score'].std(),
        'auc_mean': results_df['auc'].mean(),
        'auc_std': results_df['auc'].std(),
        'n_folds': len(results_df),
        'total_test': len(all_y_true),
    }

    return summary, results_df, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)


# --- Models -----------------------------------------------------------------
def get_models():
    return {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        ),
        'SVM': lambda: SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42
        ),
        'LogisticRegression': lambda: LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'MLP': lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            learning_rate='adaptive', early_stopping=True, random_state=42
        ),
    }


# --- Main -------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- 1. Load and prepare data --
    df = load_and_prepare_data()

    # -- 2. Add missingness indicators (BEFORE z-scoring/imputation) --
    df = add_missingness_indicators(df)
    missingness_feats = [f'{f}_missing' for f in PHYSIO_FEATURES]

    # -- 3. Add derived features --
    df = add_derived_features(df)
    available_derived = [f for f in DERIVED_FEATURES if f in df.columns]

    # -- 4. Define base feature sets --
    base_features = COMPUTER_FEATURES + PHYSIO_FEATURES
    all_features = base_features + available_derived + missingness_feats

    print(f"\nFeature counts:")
    print(f"  Computer:    {len(COMPUTER_FEATURES)}")
    print(f"  Physiology:  {len(PHYSIO_FEATURES)}")
    print(f"  Derived:     {len(available_derived)}")
    print(f"  Missingness: {len(missingness_feats)}")
    print(f"  Total:       {len(all_features)}")

    # -- 5. Prepare experiment DataFrames --
    # Features to z-score: all numeric features (not missingness flags)
    zscore_targets = COMPUTER_FEATURES + PHYSIO_FEATURES + available_derived

    experiments = {
        'A_baseline': {
            'desc': 'Raw features, global impute+scale (no z-scoring)',
            'df': df.copy(),
            'features': base_features,
        },
        'B_zscore': {
            'desc': 'Per-person z-scored (all features)',
            'df': per_person_zscore(df.copy(), zscore_targets),
            'features': base_features,
        },
        'C_all_fixes': {
            'desc': 'Z-scored + derived + missingness indicators',
            'df': per_person_zscore(df.copy(), zscore_targets),
            'features': all_features,
        },
        'D_physio_only': {
            'desc': 'Physiology-only (per-person z-scored)',
            'df': per_person_zscore(df.copy(), PHYSIO_FEATURES),
            'features': PHYSIO_FEATURES + missingness_feats,
        },
        'E_computer_only': {
            'desc': 'Computer-only (per-person z-scored)',
            'df': per_person_zscore(df.copy(), COMPUTER_FEATURES + available_derived),
            'features': COMPUTER_FEATURES + available_derived,
        },
    }

    models = get_models()

    # -- 6. Run all experiments --
    all_summaries = []

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_name}: {exp_config['desc']}")
        print(f"  Features: {len(exp_config['features'])}")
        print(f"{'='*70}")

        # --- Sklearn models ---
        for model_name, model_fn in models.items():
            full_name = f"{exp_name}_{model_name}"
            print(f"\n  {model_name}...", end=" ", flush=True)

            try:
                summary, fold_df, y_true, y_pred, y_prob = run_loso_cv(
                    exp_config['df'], exp_config['features'], model_fn, full_name
                )

                print(f"acc={summary['accuracy_mean']:.3f}+-{summary['accuracy_std']:.3f}  "
                      f"f1={summary['f1_mean']:.3f}+-{summary['f1_std']:.3f}  "
                      f"auc={summary['auc_mean']:.3f}+-{summary['auc_std']:.3f}")

                summary['experiment'] = exp_name
                summary['description'] = exp_config['desc']
                summary['n_features'] = len(exp_config['features'])
                all_summaries.append(summary)

                fold_df.to_csv(
                    os.path.join(RESULTS_DIR, f"{full_name}_folds.csv"),
                    index=False
                )

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        # --- Transformer (minute-level sequence model) ---
        print(f"\n  Transformer...", end=" ", flush=True)
        try:
            t_summary, t_fold_df, t_yt, t_yp, t_yprob = run_transformer_loso(
                exp_config['df'], exp_config['features'], exp_name
            )

            print(f"acc={t_summary['accuracy_mean']:.3f}+-{t_summary['accuracy_std']:.3f}  "
                  f"f1={t_summary['f1_mean']:.3f}+-{t_summary['f1_std']:.3f}  "
                  f"auc={t_summary['auc_mean']:.3f}+-{t_summary['auc_std']:.3f}")

            t_summary['experiment'] = exp_name
            t_summary['description'] = exp_config['desc']
            t_summary['n_features'] = len(exp_config['features'])
            all_summaries.append(t_summary)

            t_fold_df.to_csv(
                os.path.join(RESULTS_DIR, f"{exp_name}_Transformer_folds.csv"),
                index=False
            )

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # -- 7. Summary comparison --
    print(f"\n{'='*70}")
    print("FINAL COMPARISON -- sorted by AUC")
    print(f"{'='*70}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df = summary_df.sort_values('auc_mean', ascending=False)

    print(f"\n{'Model':<40s} {'Acc':>7s} {'F1':>7s} {'AUC':>7s} {'Feats':>5s}")
    print("-" * 72)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<40s} {row['accuracy_mean']:6.3f}  "
              f"{row['f1_mean']:6.3f}  {row['auc_mean']:6.3f}  "
              f"{int(row['n_features']):5d}")

    # -- 8. Compare with old baselines --
    print(f"\n{'='*70}")
    print("IMPROVEMENT vs OLD BASELINES (47.5% HR, median-imputed)")
    print(f"{'='*70}")
    print(f"\nOld best (RF):  acc=0.684  f1=0.670  auc=0.747  (21 features)")

    best = summary_df.iloc[0]
    print(f"New best ({best['model']}):  "
          f"acc={best['accuracy_mean']:.3f}  "
          f"f1={best['f1_mean']:.3f}  "
          f"auc={best['auc_mean']:.3f}  "
          f"({int(best['n_features'])} features)")

    # -- 9. Per-experiment best model --
    print(f"\nBest model per experiment:")
    for exp in experiments:
        exp_rows = summary_df[summary_df['experiment'] == exp]
        if len(exp_rows) > 0:
            best_row = exp_rows.iloc[0]
            model_short = best_row['model'].replace(f"{exp}_", "")
            print(f"  {exp}: {model_short}  "
                  f"acc={best_row['accuracy_mean']:.3f}  "
                  f"auc={best_row['auc_mean']:.3f}")

    # -- 10. Save results --
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False
    )

    report = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Behavioral-features xlsx + raw .S00 physiology',
        'hr_coverage_before': '47.5%',
        'hr_coverage_after': '95.7%',
        'total_rows': len(df),
        'fixes_applied': [
            'Per-person z-scoring (all features)',
            'Constant-0 imputation (not median)',
            'Missingness indicators for physiology',
            'Derived interaction features',
            'Recovered physiology from raw .S00 files',
        ],
        'experiments': {name: cfg['desc'] for name, cfg in experiments.items()},
        'best_model': best['model'],
        'best_auc': float(best['auc_mean']),
        'best_accuracy': float(best['accuracy_mean']),
        'best_f1': float(best['f1_mean']),
    }
    with open(os.path.join(RESULTS_DIR, "report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
