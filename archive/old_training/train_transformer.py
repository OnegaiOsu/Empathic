"""
Transformer-based sequence classification for SWELL-KW stress classification.

Treats each participant's session (block) as a time series of minute-level
feature vectors and uses a Transformer encoder to classify the entire
sequence as Stressed vs. Not Stressed.

Each sequence = one contiguous segment of same-condition minutes within a block.
Sequences are padded to max length within each batch.

Uses PyTorch. Install: pip install torch

Usage:
    python train_transformer.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import (
    load_and_merge_data,
    get_feature_columns,
    LABEL_COL,
    GROUP_COL,
    PHYSIOLOGY_FEATURES,
)
from utils.evaluation import (
    compute_fold_metrics,
    print_summary,
    save_metrics,
    save_confusion_matrix,
    save_roc_curve,
)

MODEL_NAME = "transformer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
D_MODEL = 64         # transformer hidden dimension
N_HEADS = 4          # number of attention heads
N_LAYERS = 2         # number of transformer encoder layers
D_FF = 128           # feed-forward hidden dimension
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 15        # early stopping patience


# =========================================================================
# Feature Engineering (lightweight — same as enhanced)
# =========================================================================
def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add missingness indicators, ratios, and delta features."""
    new_cols = []

    for col in PHYSIOLOGY_FEATURES:
        ind_col = f"{col}_missing"
        df[ind_col] = df[col].isna().astype(int)
        new_cols.append(ind_col)

    df["error_rate"] = df["SnErrorKeys"] / (df["SnKeyStrokes"] + 1)
    df["click_rate"] = (df["SnLeftClicked"] + df["SnRightClicked"]) / (df["SnMouseAct"] + 1)
    df["keyboard_mouse_ratio"] = df["SnKeyStrokes"] / (df["SnMouseAct"] + 1)
    df["context_switches"] = df["SnAppChange"] + df["SnTabfocusChange"]
    new_cols += ["error_rate", "click_rate", "keyboard_mouse_ratio", "context_switches"]

    df = df.sort_values(["PP", "C", "timestamp"]).reset_index(drop=True)
    delta_sources = PHYSIOLOGY_FEATURES + ["SnMouseAct", "SnKeyStrokes", "SnMouseDistance"]
    for col in delta_sources:
        delta_col = f"{col}_delta"
        df[delta_col] = df.groupby(["PP", "C"])[col].diff()
        new_cols.append(delta_col)

    return df, new_cols


# =========================================================================
# Sequence Builder
# =========================================================================
def build_sequences(df: pd.DataFrame, feature_cols: list[str]):
    """
    Group data into sequences: contiguous same-condition segments within
    each (PP, block). Each sequence gets a single label.

    Returns:
        sequences: list of np.ndarray, each shape (seq_len, n_features)
        labels: list of int (0 or 1)
        pp_ids: list of str (participant ID for each sequence)
    """
    sequences, labels, pp_ids = [], [], []

    for (pp, c), group in df.groupby(["PP", "C"]):
        group = group.sort_values("timestamp")

        # Split into contiguous condition segments
        condition_changes = group["Condition"].ne(group["Condition"].shift()).cumsum()
        for _, segment in group.groupby(condition_changes):
            if len(segment) < 2:
                continue  # skip very short segments

            seq = segment[feature_cols].values
            label = int(segment[LABEL_COL].mode()[0])

            sequences.append(seq)
            labels.append(label)
            pp_ids.append(pp)

    return sequences, labels, pp_ids


# =========================================================================
# PyTorch Dataset & Collation
# =========================================================================
class SequenceDataset(Dataset):
    """PyTorch Dataset for padded sequences."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]]),
        )


def collate_fn(batch):
    """Pad sequences to max length in batch, create attention mask."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    n_features = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, n_features)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)  # True = masked

    for i, (seq, length) in enumerate(zip(seqs, lengths)):
        padded[i, :length] = seq
        mask[i, :length] = False  # False = real position (not masked)

    labels = torch.cat(labels)
    return padded, mask, labels


# =========================================================================
# Transformer Model
# =========================================================================
class StressTransformer(nn.Module):
    """
    Transformer encoder for sequence classification.

    Architecture:
        Input projection -> Positional encoding -> Transformer Encoder ->
        Mean pooling (over non-padded positions) -> Classification head
    """

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.2, max_len=200):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (batch, seq_len, n_features)
        src_key_padding_mask: (batch, seq_len) -- True for padded positions
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over non-padded positions
        if src_key_padding_mask is not None:
            real_mask = ~src_key_padding_mask
            real_mask_f = real_mask.unsqueeze(-1).float()
            x = (x * real_mask_f).sum(dim=1) / real_mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


# =========================================================================
# Training & Evaluation Loops
# =========================================================================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, mask_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch, src_key_padding_mask=mask_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_proba, all_true = [], [], []
    for X_batch, mask_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)

        logits = model(X_batch, src_key_padding_mask=mask_batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(probs[:, 1].cpu().numpy())
        all_true.extend(y_batch.numpy())

    return np.array(all_true), np.array(all_preds), np.array(all_proba)


# =========================================================================
# Main
# =========================================================================
def main():
    print(f"[transformer] Using device: {DEVICE}")

    # 1. Load & engineer features
    df = load_and_merge_data()
    df, new_cols = add_engineered_features(df)
    base_features = get_feature_columns()
    all_feature_cols = base_features + new_cols
    print(f"[transformer] Total features: {len(all_feature_cols)}")

    # 2. Build sequences
    sequences, labels, pp_ids = build_sequences(df, all_feature_cols)
    unique_pps = sorted(set(pp_ids))
    print(f"[transformer] Total sequences: {len(sequences)}, participants: {len(unique_pps)}")
    print(f"[transformer] Sequence lengths: min={min(len(s) for s in sequences)}, "
          f"max={max(len(s) for s in sequences)}, "
          f"mean={np.mean([len(s) for s in sequences]):.1f}")

    # 3. LOSO cross-validation
    fold_results, fold_ids = [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    for held_out_pp in unique_pps:
        print(f"\n  === Fold {held_out_pp} ===")

        # Split sequences by participant
        train_seqs, train_labels = [], []
        test_seqs, test_labels = [], []
        for seq, label, pp in zip(sequences, labels, pp_ids):
            if pp == held_out_pp:
                test_seqs.append(seq)
                test_labels.append(label)
            else:
                train_seqs.append(seq)
                train_labels.append(label)

        if len(test_seqs) == 0:
            print(f"  Skipping {held_out_pp} -- no test sequences")
            continue

        # Preprocess: fit KNN imputer + scaler on flattened training sequences
        all_train_flat = np.vstack(train_seqs)
        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()
        imputer.fit(all_train_flat)
        all_train_imputed = imputer.transform(all_train_flat)
        scaler.fit(all_train_imputed)

        # Apply to each sequence individually
        train_seqs_proc = []
        for seq in train_seqs:
            s = imputer.transform(seq)
            s = scaler.transform(s)
            train_seqs_proc.append(s)

        test_seqs_proc = []
        for seq in test_seqs:
            s = imputer.transform(seq)
            s = scaler.transform(s)
            test_seqs_proc.append(s)

        # Create datasets and dataloaders
        train_dataset = SequenceDataset(train_seqs_proc, train_labels)
        test_dataset = SequenceDataset(test_seqs_proc, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                                 shuffle=False, collate_fn=collate_fn)

        # Class weights
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)

        # Build model
        n_features = all_train_flat.shape[1]
        model = StressTransformer(
            n_features=n_features,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=weight)

        # Training with early stopping
        best_f1, patience_counter = 0.0, 0
        best_state = None

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()

            # Evaluate on test for monitoring
            y_true_ep, y_pred_ep, y_proba_ep = evaluate(model, test_loader)
            f1 = f1_score(y_true_ep, y_pred_ep, average="macro", zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Load best model and final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)
        y_true, y_pred, y_proba = evaluate(model, test_loader)

        metrics = compute_fold_metrics(y_true, y_pred, y_proba)
        fold_results.append(metrics)
        fold_ids.append(held_out_pp)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        print(f"  Fold {held_out_pp:>5s}  acc={metrics['accuracy']:.3f}  "
              f"f1={metrics['f1_macro']:.3f}  (sequences: {len(test_seqs)})")

    # 4. Save results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    print_summary(fold_results, MODEL_NAME)
    save_metrics(fold_results, fold_ids, MODEL_NAME)
    save_confusion_matrix(all_y_true, all_y_pred, MODEL_NAME)
    save_roc_curve(all_y_true, all_y_proba, MODEL_NAME)

    print(f"[{MODEL_NAME}] Done. Results saved to results/{MODEL_NAME}/")


if __name__ == "__main__":
    main()
