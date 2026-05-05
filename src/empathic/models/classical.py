"""Classical tabular classifiers.

We wrap scikit-learn / XGBoost estimators behind a very small common interface
so the training harness can treat them the same as our deep models.

Three classical models ship out of the box:

* ``RandomForest``     -- robust non-linear baseline that handles mixed feature
  ranges without much tuning.
* ``LogisticRegression`` -- interpretable linear reference; useful for
  checking how much non-linear structure the trees add.
* ``XGBoost``          -- gradient-boosted trees, GPU-accelerated when a CUDA
  device is available (the RTX 50-series target).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


@dataclass
class ClassicalModel:
    """Tiny wrapper giving all models the same ``.fit`` / ``.predict`` shape."""
    name: str
    estimator: object
    needs_dense: bool = True
    supports_sample_weight: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "ClassicalModel":
        if sample_weight is not None and self.supports_sample_weight:
            # sklearn Pipelines forward sample_weight via ``<step>__sample_weight``.
            if isinstance(self.estimator, Pipeline):
                final_step = self.estimator.steps[-1][0]
                self.estimator.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
            else:
                self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        return None

    @property
    def classes_(self):
        est = self.estimator
        if isinstance(est, Pipeline):
            est = est.steps[-1][1]
        return getattr(est, "classes_", None)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_baseline(seed: int = 42, strategy: str = "most_frequent") -> ClassicalModel:
    """Reference baseline (majority class or stratified random).

    Reports what any non-trivial model must beat. Included as a first-class
    model so per-fold metrics make the class-imbalance story explicit.
    """
    est = DummyClassifier(strategy=strategy, random_state=seed)
    return ClassicalModel("Baseline", est)


def make_random_forest(seed: int = 42, n_estimators: int = 400) -> ClassicalModel:
    est = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    return ClassicalModel("RandomForest", est)


def make_logistic_regression(seed: int = 42) -> ClassicalModel:
    est = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=seed,
        )),
    ])
    return ClassicalModel("LogisticRegression", est)


def make_xgboost(
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    use_gpu: bool = True,
) -> Optional[ClassicalModel]:
    if not _HAS_XGB:
        return None

    params: Dict[str, object] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    if use_gpu:
        # XGBoost >= 2.0 uses ``device="cuda"`` instead of gpu_hist.
        params["device"] = "cuda"

    clf = xgb.XGBClassifier(**params)
    est = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", clf),
    ])
    return ClassicalModel("XGBoost", est, supports_sample_weight=True)


def default_classical_models(seed: int, use_gpu: bool) -> Dict[str, ClassicalModel]:
    out: Dict[str, ClassicalModel] = {
        "Baseline": make_baseline(seed=seed),
        "RandomForest": make_random_forest(seed=seed),
        "LogisticRegression": make_logistic_regression(seed=seed),
    }
    xgb_model = make_xgboost(seed=seed, use_gpu=use_gpu)
    if xgb_model is not None:
        out["XGBoost"] = xgb_model
    return out
