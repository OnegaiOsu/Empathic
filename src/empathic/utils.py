"""Shared utilities: logging, seeding, directory helpers, subject normalization."""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Seed Python, numpy and torch (including CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: object) -> float:
    """Convert mixed numeric strings (possibly comma decimals) to float."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def zscore_by_subject(
    df: pd.DataFrame,
    features: Iterable[str],
    subject_col: str,
    clip_z: Optional[float] = None,
) -> pd.DataFrame:
    """Per-subject z-scoring.

    Motivation: physiological baselines and typing rhythms vary strongly across
    individuals. Normalising within each subject removes that nuisance variance
    so that the model learns condition-relative deviations rather than
    person identity.
    """
    out = df.copy()
    for feat in features:
        grp = out.groupby(subject_col)[feat]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        z = (out[feat] - mean) / std
        z = z.fillna(0.0)
        if clip_z is not None and clip_z > 0:
            z = z.clip(lower=-clip_z, upper=clip_z)
        out[feat] = z
    return out


def robust_scale_by_subject(
    df: pd.DataFrame,
    features: Iterable[str],
    subject_col: str,
    clip_val: Optional[float] = None,
) -> pd.DataFrame:
    """Per-subject median/IQR scaling; safer than z-score under heavy tails."""
    out = df.copy()
    for feat in features:
        grp = out.groupby(subject_col)[feat]
        med = grp.transform("median")
        q25 = grp.transform(lambda x: x.quantile(0.25))
        q75 = grp.transform(lambda x: x.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        val = (out[feat] - med) / iqr
        val = val.fillna(0.0)
        if clip_val is not None and clip_val > 0:
            val = val.clip(lower=-clip_val, upper=clip_val)
        out[feat] = val
    return out


def majority_label(values: np.ndarray) -> int:
    vals = values.astype(int)
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def align_feature_matrix(
    df: pd.DataFrame, features: List[str]
) -> np.ndarray:
    """Return a finite float matrix; fill NaN/Inf with 0 after scaling."""
    X = df[features].to_numpy(dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = 0.0
    return X
