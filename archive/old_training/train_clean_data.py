"""
Clean data experiment for SWELL-KW stress classification.

Runs two experiments to diagnose the impact of missing physiology data:

  Experiment 1 — COMPUTER-ONLY:
      Use only the 16 computer interaction features (near-zero missing data).
      If performance matches the full 19-feature models, physiology features
      (after imputation) were just adding noise.

  Experiment 2 — COMPLETE ROWS ONLY:
      Use all 19 features but DROP rows where any physiology feature is NaN.
      Smaller dataset but every feature is real (no imputation).

Both experiments use Random Forest (the best baseline) with LOSO evaluation.

Usage:
    python train_clean_data.py
"""

import sys
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import (
    load_and_merge_data,
    get_loso_splits,
    LABEL_COL,
    GROUP_COL,
    PHYSIOLOGY_FEATURES,
    COMPUTER_FEATURES,
    ALL_FEATURES,
)
from utils.preprocessing import FoldPreprocessor
from utils.evaluation import (
    compute_fold_metrics,
    print_summary,
    save_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_feature_importance,
    get_model_results_dir,
)


def run_experiment(df, feature_cols, model_name, description):
    """Run LOSO Random Forest on given df & features, save results."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Dataset: {len(df)} rows, {len(feature_cols)} features")
    print(f"  Participants: {df['PP'].nunique()}")
    print(f"  Label dist: {dict(df[LABEL_COL].value_counts())}")
    print(f"  Missing values: {df[feature_cols].isna().sum().sum()}")
    print(f"{'='*70}\n")

    fold_results, fold_ids = [], []
    all_y_true, all_y_pred, all_y_proba = [], [], []
    all_importances = []
    skipped = 0

    for train_idx, test_idx, pp_id in get_loso_splits(df):
        X_train = df.iloc[train_idx][feature_cols].values
        X_test = df.iloc[test_idx][feature_cols].values
        y_train = df.iloc[train_idx][LABEL_COL].values
        y_test = df.iloc[test_idx][LABEL_COL].values

        # Skip folds where test set is empty (can happen in Experiment 2)
        if len(y_test) == 0:
            print(f"  Fold {pp_id:>5s}  SKIPPED (no test data after filtering)")
            skipped += 1
            continue

        # Skip folds where test set has only one class
        if len(set(y_test)) < 2:
            print(f"  Fold {pp_id:>5s}  SKIPPED (only one class in test, n={len(y_test)})")
            skipped += 1
            continue

        preprocessor = FoldPreprocessor()
        X_train, X_test = preprocessor.fit_transform(X_train, X_test)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_fold_metrics(y_test, y_pred, y_proba)
        fold_results.append(metrics)
        fold_ids.append(pp_id)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_importances.append(model.feature_importances_)

        print(
            f"  Fold {pp_id:>5s}  acc={metrics['accuracy']:.3f}"
            f"  f1={metrics['f1_macro']:.3f}  (n_test={len(y_test)})"
        )

    if not fold_results:
        print(f"\n  [!] No valid folds — cannot evaluate {model_name}\n")
        return

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    if skipped:
        print(f"\n  ({skipped} folds skipped due to missing/single-class test data)")

    print_summary(fold_results, model_name)
    save_metrics(fold_results, fold_ids, model_name)
    save_confusion_matrix(all_y_true, all_y_pred, model_name)
    save_roc_curve(all_y_true, all_y_proba, model_name)

    avg_importance = np.mean(all_importances, axis=0)
    save_feature_importance(avg_importance, feature_cols, model_name)

    print(f"  [{model_name}] Results saved to results/{model_name}/\n")


def main():
    df = load_and_merge_data()

    # =================================================================
    # Experiment 1: Computer-only features (no physiology)
    # =================================================================
    run_experiment(
        df=df,
        feature_cols=list(COMPUTER_FEATURES),
        model_name="rf_computer_only",
        description="EXPERIMENT 1 — Computer Interaction Features Only (16 features)",
    )

    # =================================================================
    # Experiment 2: All 19 features, but only complete rows
    # =================================================================
    df_complete = df.dropna(subset=PHYSIOLOGY_FEATURES).copy()
    run_experiment(
        df=df_complete,
        feature_cols=list(ALL_FEATURES),
        model_name="rf_complete_rows",
        description="EXPERIMENT 2 — All 19 Features, Complete Rows Only (no imputation)",
    )

    # =================================================================
    # Summary comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("  Baseline RF (19 features, imputed):  acc=0.684  f1=0.670  auc=0.747")
    print("  Compare with the two experiments above to identify the bottleneck.")
    print()
    print("  If Exp 1 ≈ Baseline  → physiology (after imputation) adds nothing")
    print("  If Exp 1 << Baseline → physiology matters, imputation is decent")
    print("  If Exp 2 >> Baseline → clean physiology helps; missing data was the problem")
    print("  If Exp 2 ≈ Baseline  → even real physiology doesn't help much")
    print("=" * 70)


if __name__ == "__main__":
    main()
