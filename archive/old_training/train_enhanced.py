"""
Enhanced binary classification model for SWELL-KW stress classification.

Incorporates improvements over the baseline models:
  1. KNN Imputation + missingness indicator features
  2. Feature engineering (ratios, deltas, temporal windows)
  3. Per-subject baseline normalization (deviation from Rest condition)
  4. Hyperparameter tuning (manual grid search, no nested parallelism)
  5. Ensemble voting (RF + XGBoost + LR -> soft vote)
  6. Threshold optimization (maximize F1 on validation split)
  7. Feature selection via mutual information

Usage:
    python train_enhanced.py
"""

import gc
import sys
import os
import warnings
import numpy as np
import pandas as pd
import time as _time
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import (
    load_and_merge_data,
    get_feature_columns,
    get_loso_splits,
    LABEL_COL,
    GROUP_COL,
    PHYSIOLOGY_FEATURES,
    COMPUTER_FEATURES,
)
from utils.preprocessing import compute_scale_pos_weight
from utils.evaluation import (
    compute_fold_metrics,
    print_summary,
    save_metrics,
    save_confusion_matrix,
    save_roc_curve,
    get_model_results_dir,
)

MODEL_NAME = "enhanced_ensemble"


# =========================================================================
# Feature Engineering  (applied per-fold to avoid data leakage)
# =========================================================================
def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add derived features.  Safe to call on any subset of the data since all
    operations are row-level or within-participant-within-condition groups.
    Returns updated df and list of new column names.
    """
    df = df.copy()
    new_cols = []

    # 1. Missingness indicators for physiology (binary — always valid)
    for col in PHYSIOLOGY_FEATURES:
        ind_col = f"{col}_missing"
        df[ind_col] = df[col].isna().astype(int)
        new_cols.append(ind_col)

    # 2. Ratio features (computer-interaction only — no NaN issues)
    df["error_rate"] = df["SnErrorKeys"] / (df["SnKeyStrokes"] + 1)
    df["click_rate"] = (df["SnLeftClicked"] + df["SnRightClicked"]) / (
        df["SnMouseAct"] + 1
    )
    df["keyboard_mouse_ratio"] = df["SnKeyStrokes"] / (df["SnMouseAct"] + 1)
    df["context_switches"] = df["SnAppChange"] + df["SnTabfocusChange"]
    new_cols += [
        "error_rate",
        "click_rate",
        "keyboard_mouse_ratio",
        "context_switches",
    ]

    # 3. Delta features (change from previous minute, within PP x C)
    df = df.sort_values(["PP", "C", "timestamp"]).reset_index(drop=True)
    delta_sources = PHYSIOLOGY_FEATURES + [
        "SnMouseAct",
        "SnKeyStrokes",
        "SnMouseDistance",
    ]
    for col in delta_sources:
        delta_col = f"{col}_delta"
        df[delta_col] = df.groupby(["PP", "C"])[col].diff().fillna(0)
        new_cols.append(delta_col)

    # 4. Rolling window features (3-minute window within PP x C)
    for col in PHYSIOLOGY_FEATURES:
        for stat, func in [("roll_mean", "mean"), ("roll_std", "std")]:
            roll_col = f"{col}_{stat}"
            df[roll_col] = df.groupby(["PP", "C"])[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).agg(func)
            )
            if stat == "roll_std":
                df[roll_col] = df[roll_col].fillna(0)
            new_cols.append(roll_col)

    return df, new_cols


# =========================================================================
# Per-Subject Baseline Normalization
# =========================================================================
def compute_rest_baselines(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Compute mean feature values during Rest (R) for each participant."""
    rest = df[df["Condition"] == "R"]
    if rest.empty:
        return pd.DataFrame()
    valid_cols = [c for c in feature_cols if c in rest.columns]
    return rest.groupby("PP")[valid_cols].mean()


def normalize_to_rest_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    rest_baselines: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Subtract each participant's Rest-condition mean from numeric feature
    columns.  Skips binary columns (missingness indicators) since subtracting
    a mean from a 0/1 flag is meaningless and causes dtype errors.
    """
    df = df.copy()
    if rest_baselines is None or rest_baselines.empty:
        return df

    # Only normalize continuous columns, not binary indicators
    skip_suffixes = ("_missing",)
    norm_cols = [
        c for c in feature_cols
        if c in rest_baselines.columns and not c.endswith(skip_suffixes)
    ]

    # Cast target columns to float first to avoid int64 -> float64 errors
    for col in norm_cols:
        df[col] = df[col].astype(float)

    for pp in df["PP"].unique():
        if pp not in rest_baselines.index:
            continue
        mask = df["PP"] == pp
        for col in norm_cols:
            bv = rest_baselines.loc[pp, col]
            if pd.notna(bv) and bv != 0:
                df.loc[mask, col] = df.loc[mask, col] - bv
    return df


# =========================================================================
# Enhanced Preprocessor
# =========================================================================
class EnhancedPreprocessor:
    """KNN imputation + scaling + optional mutual-info feature selection."""

    def __init__(self, n_features_to_select: int | None = None):
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()
        self.selector = None
        self.n_features_to_select = n_features_to_select

    def fit_transform(self, X_train, X_test, y_train):
        X_train = self.knn_imputer.fit_transform(X_train)
        X_test = self.knn_imputer.transform(X_test)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        if self.n_features_to_select is not None:
            k = min(self.n_features_to_select, X_train.shape[1])
            # ANOVA F-test is O(n) vs mutual_info_classif O(n^2)
            self.selector = SelectKBest(f_classif, k=k)
            X_train = self.selector.fit_transform(X_train, y_train)
            X_test = self.selector.transform(X_test)
        return X_train, X_test


# =========================================================================
# Threshold Optimization
# =========================================================================
def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 macro on given data."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


# =========================================================================
# Hyperparameter Tuning  (manual grid — avoids sklearn nested parallelism)
# =========================================================================
RF_GRID = [
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 5},
]

XGB_GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
]

LR_GRID = [{"C": 0.1}, {"C": 1.0}]


def _pick_best(model_cls, grid, X, y, fixed_params):
    """
    Evaluate each param dict in *grid* via 2-fold stratified CV and return
    the best estimator (fitted on full X, y).
    ALL fitting is sequential (n_jobs=1) to avoid Windows loky deadlocks.
    """
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    best_score, best_params = -1, grid[0]
    for params in grid:
        m = model_cls(**{**fixed_params, **params})
        scores = cross_val_score(m, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit best config on all training data
    best_model = model_cls(**{**fixed_params, **best_params})
    best_model.fit(X, y)
    return best_model, best_params


def build_tuned_ensemble(X_train, y_train):
    """
    Tune RF, XGBoost, LR individually via manual grid search, then combine
    in a soft-voting ensemble.

    Uses VotingClassifier instead of StackingClassifier:
      - Stacking internally runs cross_val_predict (re-trains each estimator
        cv times), which doubles computation and triggers nested loky spawning.
      - VotingClassifier just calls predict_proba on the already-fitted base
        models — no extra training, no extra processes.
    """
    spw = compute_scale_pos_weight(y_train)

    # --- Tune each base model (all sequential, n_jobs=1) ----------------
    rf, rf_params = _pick_best(
        RandomForestClassifier,
        RF_GRID,
        X_train,
        y_train,
        fixed_params={
            "class_weight": "balanced",
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,         # CRITICAL: n_jobs=1 to avoid loky deadlocks
        },
    )

    xgb, xgb_params = _pick_best(
        XGBClassifier,
        XGB_GRID,
        X_train,
        y_train,
        fixed_params={
            "scale_pos_weight": spw,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": 1,         # CRITICAL: n_jobs=1 to avoid loky deadlocks
            "verbosity": 0,
        },
    )

    lr, lr_params = _pick_best(
        LogisticRegression,
        LR_GRID,
        X_train,
        y_train,
        fixed_params={
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    )

    # --- Soft Voting (no internal CV, no extra process spawning) ---------
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
        voting="soft",
        n_jobs=1,
    )
    # VotingClassifier.fit() with pre-fitted estimators: we need to fit
    # again, but since models are already fitted this is fast.
    ensemble.fit(X_train, y_train)

    return ensemble, {
        "rf_best": str(rf_params),
        "xgb_best": str(xgb_params),
        "lr_best": str(lr_params),
    }


# =========================================================================
# Main
# =========================================================================
def main():
    t0_total = _time.time()

    # 1. Load raw data & engineer features ONCE
    #    (Feature engineering uses only within-PP-within-C operations so
    #     there is no cross-participant leakage.  Doing it once avoids
    #     repeating the expensive groupby/rolling 25 times.)
    df_all = load_and_merge_data()
    base_features = get_feature_columns()
    df_all, new_cols = add_engineered_features(df_all)
    all_feature_cols = base_features + new_cols
    print(f"[enhanced] Total features: {len(all_feature_cols)} ({len(new_cols)} engineered)")

    # 2. LOSO cross-validation
    fold_results, fold_ids = [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []
    all_best_params = []

    for train_idx, test_idx, pp_id in get_loso_splits(df_all):
        t0 = _time.time()
        print(f"\n  === Fold {pp_id} ===", flush=True)

        df_train = df_all.iloc[train_idx].copy()
        df_test = df_all.iloc[test_idx].copy()
        y_train = df_train[LABEL_COL].values
        y_test = df_test[LABEL_COL].values

        # (a) Baseline normalization on continuous feature columns
        rest_bl = compute_rest_baselines(df_train, all_feature_cols)
        df_train = normalize_to_rest_baseline(df_train, all_feature_cols, rest_bl)
        df_test = normalize_to_rest_baseline(df_test, all_feature_cols, rest_bl)
        print(f"    norm {_time.time()-t0:.1f}s", flush=True)

        # (b) Extract feature matrices
        X_train = df_train[all_feature_cols].values.astype(np.float64)
        X_test = df_test[all_feature_cols].values.astype(np.float64)

        # (c) KNN imputation + scaling + feature selection (top 25)
        preprocessor = EnhancedPreprocessor(n_features_to_select=25)
        X_train, X_test = preprocessor.fit_transform(X_train, X_test, y_train)
        print(f"    prep {_time.time()-t0:.1f}s", flush=True)

        # (d) Build tuned ensemble
        model, best_params = build_tuned_ensemble(X_train, y_train)
        all_best_params.append(best_params)
        print(f"    tune {_time.time()-t0:.1f}s", flush=True)

        # (e) Predict + threshold optimization
        y_proba = model.predict_proba(X_test)[:, 1]
        y_train_proba = model.predict_proba(X_train)[:, 1]
        optimal_thresh = optimize_threshold(y_train, y_train_proba)
        y_pred = (y_proba >= optimal_thresh).astype(int)

        metrics = compute_fold_metrics(y_test, y_pred, y_proba)
        fold_results.append(metrics)
        fold_ids.append(pp_id)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        print(
            f"  Fold {pp_id:>5s}  acc={metrics['accuracy']:.3f}"
            f"  f1={metrics['f1_macro']:.3f}  thresh={optimal_thresh:.2f}"
        )

        # Explicit cleanup to prevent memory buildup across 25 folds
        del model, X_train, X_test, df_train, df_test, preprocessor
        gc.collect()

    # 3. Save results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    print_summary(fold_results, MODEL_NAME)
    save_metrics(fold_results, fold_ids, MODEL_NAME)
    save_confusion_matrix(all_y_true, all_y_pred, MODEL_NAME)
    save_roc_curve(all_y_true, all_y_proba, MODEL_NAME)

    # Save hyperparameters log
    out_dir = get_model_results_dir(MODEL_NAME)
    params_df = pd.DataFrame(all_best_params)
    params_df.insert(0, "fold_participant", fold_ids)
    params_df.to_csv(os.path.join(out_dir, "best_hyperparameters.csv"), index=False)

    print(f"\n[{MODEL_NAME}] Done. Results saved to results/{MODEL_NAME}/")


if __name__ == "__main__":
    main()
