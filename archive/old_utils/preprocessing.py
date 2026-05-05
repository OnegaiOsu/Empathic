"""
Per-fold preprocessing for SWELL-KW stress classification.

Handles scaling, imputation, and optional feature engineering.
All transformations are fit on the training fold only, then applied to test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class FoldPreprocessor:
    """
    Stateful preprocessor that fits on training data and transforms both
    train and test data within a single LOSO fold.

    Steps:
        1. Median imputation of NaN values
        2. Standard scaling (zero mean, unit variance)
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit on X_train, transform both X_train and X_test.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
        X_test  : np.ndarray of shape (n_test, n_features)

        Returns
        -------
        X_train_proc : np.ndarray
        X_test_proc  : np.ndarray
        """
        # Step 1: Impute missing values
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)

        # Step 2: Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute XGBoost-style scale_pos_weight = n_negative / n_positive.

    Parameters
    ----------
    y_train : np.ndarray
        Binary labels (0 or 1).

    Returns
    -------
    float
        Ratio of negative to positive samples.
    """
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
