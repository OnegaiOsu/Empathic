"""
Train an MLP (Multi-Layer Perceptron) classifier on the SWELL-KW dataset.

Binary stress classification (Stressed vs. Not Stressed) using physiology
and computer interaction features with Leave-One-Subject-Out cross-validation.

Uses sklearn's MLPClassifier for simplicity and consistency with the rest
of the pipeline.

Usage:
    python train_mlp.py
"""

import sys
import os
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_and_merge_data, get_feature_columns, get_loso_splits, LABEL_COL
from utils.preprocessing import FoldPreprocessor
from utils.evaluation import (
    compute_fold_metrics,
    print_summary,
    save_metrics,
    save_confusion_matrix,
    save_roc_curve,
)

MODEL_NAME = "mlp"


def main():
    # Suppress convergence warnings (early_stopping handles this)
    warnings.filterwarnings("ignore", category=UserWarning)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_and_merge_data()
    feature_cols = get_feature_columns()

    # ------------------------------------------------------------------
    # 2. LOSO cross-validation
    # ------------------------------------------------------------------
    fold_results = []
    fold_ids = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for train_idx, test_idx, pp_id in get_loso_splits(df):
        X_train = df.iloc[train_idx][feature_cols].values
        X_test = df.iloc[test_idx][feature_cols].values
        y_train = df.iloc[train_idx][LABEL_COL].values
        y_test = df.iloc[test_idx][LABEL_COL].values

        # Preprocess
        preprocessor = FoldPreprocessor()
        X_train, X_test = preprocessor.fit_transform(X_train, X_test)

        # Train
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=500,
            batch_size="auto",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Collect
        metrics = compute_fold_metrics(y_test, y_pred, y_proba)
        fold_results.append(metrics)
        fold_ids.append(pp_id)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        print(f"  Fold {pp_id:>5s}  acc={metrics['accuracy']:.3f}  f1={metrics['f1_macro']:.3f}")

    # ------------------------------------------------------------------
    # 3. Aggregate and save results
    # ------------------------------------------------------------------
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    print_summary(fold_results, MODEL_NAME)
    save_metrics(fold_results, fold_ids, MODEL_NAME)
    save_confusion_matrix(all_y_true, all_y_pred, MODEL_NAME)
    save_roc_curve(all_y_true, all_y_proba, MODEL_NAME)

    # MLP has no direct feature importances — skip importance plot
    print(f"[{MODEL_NAME}] Note: MLP does not provide feature importances directly.")
    print(f"[{MODEL_NAME}] Done. Results saved to results/{MODEL_NAME}/")


if __name__ == "__main__":
    main()
