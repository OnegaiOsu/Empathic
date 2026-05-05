"""
Multimodal Stress Classification — Physiology + Computer Interaction
=====================================================================

Focused training suite that combines physiological signals (HR, RMSSD, SCL)
with computer interaction features (mouse, keyboard, app switching) for
binary stress classification on the SWELL-KW dataset.

Uses the best data pipeline:
  - Recovered physiology from raw .S00 files (95.7% HR coverage)
  - Per-person z-scoring (removes inter-individual variation)
  - Constant-0 imputation (after z-scoring, 0 = person's mean)
  - Missingness indicators for physiology channels
  - Derived cross-modal features (HR×SCL, keystroke efficiency, etc.)

Models:
  1. Random Forest (300 trees, balanced)
  2. Gradient Boosting (200 trees, depth 5)
  3. SVM (RBF kernel, balanced)
  4. Logistic Regression (L2, balanced)
  5. MLP (64→32, adaptive LR, early stopping)
  6. Transformer (2-layer encoder, per-minute predictions)

Evaluation: Leave-One-Subject-Out (LOSO) cross-validation, 25 folds.

Outputs:
  results/multimodal/
    fold_results/           — per-participant CSVs for each model
    plots/                  — all visualizations
    summary.csv             — aggregate metrics
    report.json             — machine-readable report
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc as sk_auc
)
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ------------------------------------------------------------------
XLSX_PATH = "0_SWELL/Behavioral-features - per minute.xlsx"
PHYSIO_RAW_PATH = "physiology_from_raw.csv"
RESULTS_DIR = "results/multimodal"
FOLDS_DIR = os.path.join(RESULTS_DIR, "fold_results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# --- Feature Definitions ----------------------------------------------------
COMPUTER_FEATURES = [
    'SnMouseAct', 'SnLeftClicked', 'SnRightClicked', 'SnDoubleClicked',
    'SnWheel', 'SnDragged', 'SnMouseDistance', 'SnKeyStrokes',
    'SnChars', 'SnSpecialKeys', 'SnDirectionKeys', 'SnErrorKeys',
    'SnShortcutKeys', 'SnSpaces', 'SnAppChange', 'SnTabfocusChange',
    'CharactersRatio', 'ErrorKeyRatio',
]

PHYSIO_FEATURES = ['HR', 'RMSSD', 'SCL']

DERIVED_FEATURES = [
    'KeystrokeEfficiency',
    'MouseKeyRatio',
    'SwitchRate',
    'HR_SCL_product',
]

STRESS_MAP = {'T': 1, 'I': 1, 'N': 0, 'R': 0}


# =============================================================================
#  DATA LOADING
# =============================================================================
def load_data():
    """Load xlsx + recovered physiology, merge, return DataFrame."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_excel(XLSX_PATH, sheet_name='SWELLdata')
    extracted = pd.read_csv(PHYSIO_RAW_PATH)
    print(f"  Master xlsx:  {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Raw physio:   {extracted.shape[0]} rows")

    # Merge keys
    df['PP_num'] = df['PP'].str.replace('PP', '').astype(int)
    df['minute_idx'] = df.groupby(['PP_num', 'Blok']).cumcount()
    extracted['minute_idx'] = extracted.groupby(['PP', 'block']).cumcount()

    ext = extracted[['PP', 'block', 'minute_idx', 'HR', 'RMSSD', 'SCL']].copy()
    ext.columns = ['PP_num', 'Blok', 'minute_idx',
                   'HR_ext', 'RMSSD_ext', 'SCL_ext']
    df = df.merge(ext, on=['PP_num', 'Blok', 'minute_idx'], how='left')

    # Fill gaps with recovered data
    for feat in PHYSIO_FEATURES:
        before = df[feat].notna().sum()
        df[feat] = df[feat].fillna(df[f'{feat}_ext'])
        after = df[feat].notna().sum()
        print(f"  {feat}: {before} -> {after} valid "
              f"({before/len(df)*100:.1f}% -> {after/len(df)*100:.1f}%)")

    # Numeric coercion
    for col in COMPUTER_FEATURES + PHYSIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Labels
    df['stress_label'] = df['Condition'].map(STRESS_MAP)
    df = df.dropna(subset=['stress_label'])
    df['stress_label'] = df['stress_label'].astype(int)

    n_stressed = df['stress_label'].sum()
    n_not = (df['stress_label'] == 0).sum()
    print(f"\n  Samples: {len(df)} total  |  Stressed: {n_stressed}  |  "
          f"Not-stressed: {n_not}  |  Ratio: {n_stressed/len(df):.1%}")
    print(f"  Participants: {df['PP_num'].nunique()}")

    return df


# =============================================================================
#  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Add missingness indicators, derived features, per-person z-score."""
    df = df.copy()

    # 1. Missingness indicators (before imputation)
    for feat in PHYSIO_FEATURES:
        df[f'{feat}_missing'] = df[feat].isna().astype(int)

    # 2. Derived features
    df['KeystrokeEfficiency'] = df['SnChars'] / (df['SnKeyStrokes'] + 1)
    df['MouseKeyRatio'] = df['SnMouseAct'] / (df['SnKeyStrokes'] + 1)
    df['SwitchRate'] = df['SnAppChange'] + df['SnTabfocusChange']
    df['HR_SCL_product'] = df['HR'] * df['SCL']

    # 3. Per-person z-scoring (all numeric features)
    zscore_cols = (COMPUTER_FEATURES + PHYSIO_FEATURES +
                   [f for f in DERIVED_FEATURES if f in df.columns])
    for feat in zscore_cols:
        if feat not in df.columns:
            continue
        pp_mean = df.groupby('PP_num')[feat].transform('mean')
        pp_std = df.groupby('PP_num')[feat].transform('std').replace(0, np.nan)
        df[feat] = (df[feat] - pp_mean) / pp_std
        df[feat] = df[feat].fillna(0)

    # Build feature list
    missingness = [f'{f}_missing' for f in PHYSIO_FEATURES]
    available_derived = [f for f in DERIVED_FEATURES if f in df.columns]
    all_features = COMPUTER_FEATURES + PHYSIO_FEATURES + available_derived + missingness

    print(f"\n  Feature breakdown:")
    print(f"    Computer interaction: {len(COMPUTER_FEATURES)}")
    print(f"    Physiology:          {len(PHYSIO_FEATURES)}")
    print(f"    Derived:             {len(available_derived)}")
    print(f"    Missingness flags:   {len(missingness)}")
    print(f"    Total:               {len(all_features)}")

    return df, all_features


# =============================================================================
#  TRANSFORMER MODEL
# =============================================================================
class StressTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.2, max_len=200):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.input_proj(x) + self.pos_enc[:, :T, :]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(x)


class SeqDataset(Dataset):
    def __init__(self, seqs, labs):
        self.seqs, self.labs = seqs, labs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        return torch.FloatTensor(self.seqs[i]), torch.LongTensor(self.labs[i])


def collate(batch):
    seqs, labs = zip(*batch)
    lens = [s.shape[0] for s in seqs]
    ml = max(lens)
    nf = seqs[0].shape[1]
    X = torch.zeros(len(seqs), ml, nf)
    Y = torch.full((len(seqs), ml), -100, dtype=torch.long)
    M = torch.ones(len(seqs), ml, dtype=torch.bool)
    for i, (s, l, n) in enumerate(zip(seqs, labs, lens)):
        X[i, :n] = s; Y[i, :n] = l; M[i, :n] = False
    return X, M, Y, lens


# =============================================================================
#  LOSO: TABULAR MODELS
# =============================================================================
def run_tabular_loso(df, features, model_fn, model_name):
    """LOSO CV for sklearn models. Returns summary dict + fold DataFrame + arrays."""
    participants = sorted(df['PP_num'].unique())
    available = [f for f in features if f in df.columns]
    folds, yt_all, yp_all, yprob_all = [], [], [], []

    for pp in participants:
        te = df['PP_num'] == pp
        tr = ~te
        X_tr = df.loc[tr, available].values.astype(float)
        y_tr = df.loc[tr, 'stress_label'].values
        X_te = df.loc[te, available].values.astype(float)
        y_te = df.loc[te, 'stress_label'].values

        if len(np.unique(y_te)) < 2 or len(y_te) < 5:
            continue

        X_tr = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_tr)
        sc = StandardScaler().fit(X_tr)
        X_tr = sc.transform(X_tr)
        X_te = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X_te)
        X_te = sc.transform(X_te)

        m = model_fn()
        m.fit(X_tr, y_tr)

        yp = m.predict(X_te)
        ypr = m.predict_proba(X_te)[:, 1] if hasattr(m, 'predict_proba') else None

        acc = accuracy_score(y_te, yp)
        f1 = f1_score(y_te, yp)
        prec = precision_score(y_te, yp, zero_division=0)
        rec = recall_score(y_te, yp, zero_division=0)
        auc_val = roc_auc_score(y_te, ypr) if ypr is not None else np.nan

        folds.append({
            'participant': pp, 'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'auc': auc_val,
            'n_test': len(y_te), 'n_stressed': int(y_te.sum()),
        })
        yt_all.extend(y_te); yp_all.extend(yp)
        if ypr is not None:
            yprob_all.extend(ypr)

    fold_df = pd.DataFrame(folds)
    summary = _make_summary(fold_df, model_name)
    return summary, fold_df, np.array(yt_all), np.array(yp_all), np.array(yprob_all)


# =============================================================================
#  LOSO: TRANSFORMER
# =============================================================================
def run_transformer_loso(df, features, epochs=80, patience=15, batch_size=16):
    """LOSO CV for Transformer (minute-level). Same return signature as tabular."""
    available = [f for f in features if f in df.columns]
    seqs, labs, pps = [], [], []
    for (pp, blk), g in df.groupby(['PP_num', 'Blok']):
        g = g.sort_index()
        s = np.nan_to_num(g[available].values.astype(float), nan=0.0)
        l = g['stress_label'].values.astype(int)
        if len(s) >= 2:
            seqs.append(s); labs.append(l); pps.append(pp)

    participants = sorted(set(pps))
    nf = seqs[0].shape[1]
    folds, yt_all, yp_all, yprob_all = [], [], [], []

    for held in participants:
        tr_s, tr_l, te_s, te_l = [], [], [], []
        for s, l, p in zip(seqs, labs, pps):
            (te_s if p == held else tr_s).append(s)
            (te_l if p == held else tr_l).append(l)
        if not te_s:
            continue

        # Scale
        flat = np.vstack(tr_s)
        sc = StandardScaler().fit(flat)
        tr_s = [sc.transform(s) for s in tr_s]
        te_s = [sc.transform(s) for s in te_s]

        # Val split (15%)
        nv = max(1, int(len(tr_s) * 0.15))
        rng = np.random.RandomState(42)
        vi = set(rng.choice(len(tr_s), nv, replace=False))
        ti = [i for i in range(len(tr_s)) if i not in vi]
        v_s = [tr_s[i] for i in vi]; v_l = [tr_l[i] for i in vi]
        t_s = [tr_s[i] for i in ti]; t_l = [tr_l[i] for i in ti]

        train_dl = DataLoader(SeqDataset(t_s, t_l), batch_size=batch_size,
                              shuffle=True, collate_fn=collate)
        val_dl = DataLoader(SeqDataset(v_s, v_l), batch_size=max(1, len(v_s)),
                            shuffle=False, collate_fn=collate)
        test_dl = DataLoader(SeqDataset(te_s, te_l), batch_size=max(1, len(te_s)),
                             shuffle=False, collate_fn=collate)

        # Class weights
        all_labs = np.concatenate(t_l)
        np1 = (all_labs == 1).sum(); nn1 = (all_labs == 0).sum()
        wt = torch.FloatTensor([1.0, nn1 / max(np1, 1)]).to(DEVICE)

        model = StressTransformer(n_features=nf).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss(weight=wt, ignore_index=-100)

        best_vl, best_st, pat = float('inf'), None, 0
        for ep in range(epochs):
            model.train()
            for xb, mb, yb, _ in train_dl:
                xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb, mb).reshape(-1, 2), yb.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = sum(
                    crit(model(xb.to(DEVICE), mb.to(DEVICE)).reshape(-1, 2),
                         yb.to(DEVICE).reshape(-1)).item()
                    for xb, mb, yb, _ in val_dl)
            if vl < best_vl:
                best_vl = vl
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    break

        if best_st:
            model.load_state_dict(best_st)
        model.eval()
        f_yt, f_yp, f_ypr = [], [], []
        with torch.no_grad():
            for xb, mb, yb, lens in test_dl:
                logits = model(xb.to(DEVICE), mb.to(DEVICE))
                probs = torch.softmax(logits, dim=-1)
                for i, n in enumerate(lens):
                    f_yt.extend(yb[i, :n].numpy())
                    f_yp.extend(logits[i, :n].argmax(-1).cpu().numpy())
                    f_ypr.extend(probs[i, :n, 1].cpu().numpy())

        f_yt, f_yp, f_ypr = np.array(f_yt), np.array(f_yp), np.array(f_ypr)
        if len(np.unique(f_yt)) < 2 or len(f_yt) < 5:
            continue

        acc = accuracy_score(f_yt, f_yp)
        f1v = f1_score(f_yt, f_yp)
        prec = precision_score(f_yt, f_yp, zero_division=0)
        rec = recall_score(f_yt, f_yp, zero_division=0)
        try:
            auc_val = roc_auc_score(f_yt, f_ypr)
        except ValueError:
            auc_val = np.nan
        folds.append({
            'participant': held, 'accuracy': acc, 'f1': f1v,
            'precision': prec, 'recall': rec, 'auc': auc_val,
            'n_test': len(f_yt), 'n_stressed': int(f_yt.sum()),
        })
        yt_all.extend(f_yt); yp_all.extend(f_yp); yprob_all.extend(f_ypr)

    fold_df = pd.DataFrame(folds)
    summary = _make_summary(fold_df, 'Transformer')
    return summary, fold_df, np.array(yt_all), np.array(yp_all), np.array(yprob_all)


def _make_summary(fold_df, name):
    return {
        'model': name,
        'accuracy_mean': fold_df['accuracy'].mean(),
        'accuracy_std': fold_df['accuracy'].std(),
        'f1_mean': fold_df['f1'].mean(),
        'f1_std': fold_df['f1'].std(),
        'precision_mean': fold_df['precision'].mean(),
        'precision_std': fold_df['precision'].std(),
        'recall_mean': fold_df['recall'].mean(),
        'recall_std': fold_df['recall'].std(),
        'auc_mean': fold_df['auc'].mean(),
        'auc_std': fold_df['auc'].std(),
        'n_folds': len(fold_df),
    }


# =============================================================================
#  MODEL DEFINITIONS
# =============================================================================
MODELS = {
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'GradientBoosting': lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42),
    'SVM': lambda: SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=42),
    'LogisticRegression': lambda: LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
    'MLP': lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500,
        learning_rate='adaptive', early_stopping=True, random_state=42),
}

MODEL_ORDER = ['RandomForest', 'GradientBoosting', 'SVM',
               'LogisticRegression', 'MLP', 'Transformer']


# =============================================================================
#  VISUALIZATIONS
# =============================================================================
def _set_style():
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'


def plot_metric_bars(summary_df, save_dir):
    """Bar chart comparing Accuracy, F1, AUC across all models."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [('accuracy', 'Accuracy'), ('f1', 'F1 Score'), ('auc', 'AUC-ROC')]
    palette = sns.color_palette('viridis', n_colors=len(summary_df))

    order = [m for m in MODEL_ORDER if m in summary_df['model'].values]
    sdf = summary_df.set_index('model').loc[order].reset_index()

    for ax, (metric, title) in zip(axes, metrics):
        bars = ax.bar(range(len(sdf)), sdf[f'{metric}_mean'], yerr=sdf[f'{metric}_std'],
                      color=palette, capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(sdf)))
        ax.set_xticklabels(sdf['model'], rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.05)
        # Value labels
        for bar, val in zip(bars, sdf[f'{metric}_mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Multimodal Stress Classification — Model Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'model_comparison_bars.png'))
    plt.close(fig)
    print(f"  Saved: model_comparison_bars.png")


def plot_radar(summary_df, save_dir):
    """Radar chart of all metrics per model."""
    _set_style()
    metrics = ['accuracy_mean', 'f1_mean', 'precision_mean', 'recall_mean', 'auc_mean']
    labels = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    order = [m for m in MODEL_ORDER if m in summary_df['model'].values]
    sdf = summary_df.set_index('model').loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette('husl', len(sdf))

    for i, (_, row) in enumerate(sdf.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, 'o-', linewidth=2, label=row['model'], color=colors[i])
        ax.fill(angles, vals, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title('Model Performance Radar', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.savefig(os.path.join(save_dir, 'model_radar.png'))
    plt.close(fig)
    print(f"  Saved: model_radar.png")


def plot_fold_boxplots(all_folds, save_dir):
    """Box plots of per-participant metrics for each model."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('accuracy', 'Accuracy'), ('f1', 'F1 Score'), ('auc', 'AUC-ROC')]
    order = [m for m in MODEL_ORDER if m in all_folds['model'].values]

    for ax, (metric, title) in zip(axes, metrics):
        sns.boxplot(data=all_folds, x='model', y=metric, order=order,
                    palette='viridis', ax=ax, showfliers=True, width=0.6)
        ax.set_xticklabels(order, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(f'Per-Participant {title} Distribution', fontweight='bold')
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fold_boxplots.png'))
    plt.close(fig)
    print(f"  Saved: fold_boxplots.png")


def plot_roc_curves(roc_data, save_dir):
    """Overlay ROC curves for all models."""
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette('husl', len(roc_data))

    for (name, yt, ypr), color in zip(roc_data, colors):
        fpr, tpr, _ = roc_curve(yt, ypr)
        auc_val = roc_auc_score(yt, ypr)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f'{name} (AUC={auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close(fig)
    print(f"  Saved: roc_curves.png")


def plot_precision_recall(roc_data, save_dir):
    """Precision-Recall curves for all models."""
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette('husl', len(roc_data))

    for (name, yt, ypr), color in zip(roc_data, colors):
        prec_arr, rec_arr, _ = precision_recall_curve(yt, ypr)
        pr_auc = sk_auc(rec_arr, prec_arr)
        ax.plot(rec_arr, prec_arr, linewidth=2, color=color,
                label=f'{name} (PR-AUC={pr_auc:.3f})')

    baseline = sum(roc_data[0][1]) / len(roc_data[0][1])
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — All Models', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    fig.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close(fig)
    print(f"  Saved: precision_recall_curves.png")


def plot_confusion_matrices(cm_data, save_dir):
    """Grid of confusion matrices for all models."""
    _set_style()
    n = len(cm_data)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (name, yt, yp) in enumerate(cm_data):
        cm = confusion_matrix(yt, yp)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = [[f'{cm[r][c]}\n({cm_pct[r][c]:.1f}%)' for c in range(2)] for r in range(2)]

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[i],
                    xticklabels=['Not Stressed', 'Stressed'],
                    yticklabels=['Not Stressed', 'Stressed'],
                    cbar=False, linewidths=1, linecolor='white')
        axes[i].set_title(name, fontweight='bold', fontsize=11)
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Confusion Matrices — All Models', fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close(fig)
    print(f"  Saved: confusion_matrices.png")


def plot_per_participant_heatmap(all_folds, save_dir):
    """Heatmap: participants × models, colored by AUC."""
    _set_style()
    order = [m for m in MODEL_ORDER if m in all_folds['model'].values]
    pivot = all_folds.pivot_table(
        index='participant', columns='model', values='auc'
    )[order]

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0.4, vmax=1.0,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'AUC'})
    ax.set_title('Per-Participant AUC by Model', fontweight='bold')
    ax.set_ylabel('Participant')
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'participant_heatmap.png'))
    plt.close(fig)
    print(f"  Saved: participant_heatmap.png")


def plot_improvement_over_baseline(summary_df, save_dir):
    """Bar chart showing improvement of each model vs old baseline."""
    _set_style()
    old_baseline = {'accuracy': 0.684, 'f1': 0.670, 'auc': 0.747}
    order = [m for m in MODEL_ORDER if m in summary_df['model'].values]
    sdf = summary_df.set_index('model').loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [('accuracy', 'Accuracy'), ('f1', 'F1 Score'), ('auc', 'AUC-ROC')]

    for ax, (metric, title) in zip(axes, metrics):
        deltas = sdf[f'{metric}_mean'] - old_baseline[metric]
        colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]
        bars = ax.bar(range(len(sdf)), deltas, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(sdf)))
        ax.set_xticklabels(sdf['model'], rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(f'Δ {title}')
        ax.set_title(f'{title} vs Old Baseline ({old_baseline[metric]:.3f})', fontweight='bold')
        for bar, d in zip(bars, deltas):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    y + (0.005 if y >= 0 else -0.015),
                    f'{d:+.3f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=8)

    fig.suptitle('Improvement Over Old Pipeline (RF, median-imputed, 47.5% HR)', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'improvement_over_baseline.png'))
    plt.close(fig)
    print(f"  Saved: improvement_over_baseline.png")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    for d in [RESULTS_DIR, FOLDS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load & engineer ──────────────────────────────────
    df = load_data()
    df, all_features = engineer_features(df)

    # ── 2. Train all models ─────────────────────────────────
    print(f"\n{'='*70}")
    print("TRAINING — Multimodal (Physiology + Computer Interaction)")
    print(f"{'='*70}")

    summaries = []
    all_folds = []
    roc_data = []      # [(name, y_true, y_prob), ...]
    cm_data = []       # [(name, y_true, y_pred), ...]

    # Tabular models
    for name, fn in MODELS.items():
        print(f"\n  {name}...", end=" ", flush=True)
        summary, fold_df, yt, yp, ypr = run_tabular_loso(df, all_features, fn, name)
        print(f"acc={summary['accuracy_mean']:.3f}  f1={summary['f1_mean']:.3f}  "
              f"auc={summary['auc_mean']:.3f}")
        summaries.append(summary)
        fold_df['model'] = name
        all_folds.append(fold_df)
        fold_df.to_csv(os.path.join(FOLDS_DIR, f'{name}_folds.csv'), index=False)
        if len(ypr) > 0:
            roc_data.append((name, yt, ypr))
        cm_data.append((name, yt, yp))

    # Transformer
    print(f"\n  Transformer...", end=" ", flush=True)
    summary, fold_df, yt, yp, ypr = run_transformer_loso(df, all_features)
    print(f"acc={summary['accuracy_mean']:.3f}  f1={summary['f1_mean']:.3f}  "
          f"auc={summary['auc_mean']:.3f}")
    summaries.append(summary)
    fold_df['model'] = 'Transformer'
    all_folds.append(fold_df)
    fold_df.to_csv(os.path.join(FOLDS_DIR, 'Transformer_folds.csv'), index=False)
    if len(ypr) > 0:
        roc_data.append(('Transformer', yt, ypr))
    cm_data.append(('Transformer', yt, yp))

    # ── 3. Aggregate results ────────────────────────────────
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values('auc_mean', ascending=False)
    all_folds_df = pd.concat(all_folds, ignore_index=True)

    # ── 4. Print summary table ──────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS — sorted by AUC-ROC")
    print(f"{'='*70}")
    print(f"\n{'Model':<22s} {'Acc':>10s} {'F1':>10s} {'Prec':>10s} "
          f"{'Rec':>10s} {'AUC':>10s}")
    print("-" * 78)
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<22s} "
              f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.2f} "
              f"{r['f1_mean']:.3f}±{r['f1_std']:.2f} "
              f"{r['precision_mean']:.3f}±{r['precision_std']:.2f} "
              f"{r['recall_mean']:.3f}±{r['recall_std']:.2f} "
              f"{r['auc_mean']:.3f}±{r['auc_std']:.2f}")

    print(f"\nOld baseline (RF, broken pipeline):  "
          f"acc=0.684  f1=0.670  auc=0.747")
    best = summary_df.iloc[0]
    print(f"New best ({best['model']}):  "
          f"acc={best['accuracy_mean']:.3f}  "
          f"f1={best['f1_mean']:.3f}  "
          f"auc={best['auc_mean']:.3f}")

    # ── 5. Generate visualizations ──────────────────────────
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")

    plot_metric_bars(summary_df, PLOTS_DIR)
    plot_radar(summary_df, PLOTS_DIR)
    plot_fold_boxplots(all_folds_df, PLOTS_DIR)
    plot_roc_curves(roc_data, PLOTS_DIR)
    plot_precision_recall(roc_data, PLOTS_DIR)
    plot_confusion_matrices(cm_data, PLOTS_DIR)
    plot_per_participant_heatmap(all_folds_df, PLOTS_DIR)
    plot_improvement_over_baseline(summary_df, PLOTS_DIR)

    # ── 6. Save results ────────────────────────────────────
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)
    all_folds_df.to_csv(os.path.join(RESULTS_DIR, 'all_folds.csv'), index=False)

    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'Binary stress classification (multimodal)',
        'dataset': 'SWELL-KW',
        'data_pipeline': {
            'source': 'Behavioral-features xlsx + recovered .S00 physiology',
            'hr_coverage': '95.7% (recovered from 47.5%)',
            'normalization': 'Per-person z-scoring',
            'imputation': 'Constant 0 (= person mean after z-scoring)',
            'features': {
                'computer_interaction': len(COMPUTER_FEATURES),
                'physiology': len(PHYSIO_FEATURES),
                'derived': len(DERIVED_FEATURES),
                'missingness': len(PHYSIO_FEATURES),
                'total': len(all_features),
            },
        },
        'evaluation': 'Leave-One-Subject-Out (25 folds)',
        'models': {r['model']: {
            'accuracy': f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}",
            'f1': f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}",
            'auc': f"{r['auc_mean']:.3f}±{r['auc_std']:.3f}",
        } for _, r in summary_df.iterrows()},
        'best_model': best['model'],
        'best_auc': float(best['auc_mean']),
    }
    with open(os.path.join(RESULTS_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nAll results and plots saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
