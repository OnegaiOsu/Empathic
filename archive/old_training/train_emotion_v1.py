"""
Emotion Modeling Pipeline for EmoSurv and WESAD.

This script introduces a shared experiment harness with dataset-specific loaders:
- EmoSurv: discrete emotion classification from keystroke dynamics.
- WESAD: discrete condition classification and dimensional (valence, arousal) regression.

Key properties:
- Leave-One-Subject-Out (LOSO) evaluation.
- Per-subject z-scoring.
- Light data augmentation (jitter, masking, and optional class balancing).
- Multiple classical ML models for classification and regression.
- Consistent visualization outputs across EmoSurv and WESAD tasks.

Examples:
  python train_emotion.py --dataset emosurv --task discrete
  python train_emotion.py --dataset wesad --task discrete
  python train_emotion.py --dataset wesad --task dimensional
  python train_emotion.py --dataset all --task both --quick
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOSURV_DIR = os.path.join(BASE_DIR, "Dataset", "EmoSurv")
WESAD_DIR = os.path.join(BASE_DIR, "Dataset", "WeSad", "archive(2)", "WESAD")
RESULTS_ROOT = os.path.join(BASE_DIR, "results", "emotion")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DEEP_EPOCHS = 25
DEFAULT_DEEP_BATCH_SIZE = 128
DEFAULT_DEEP_LR = 1e-3
DEFAULT_SEQ_LEN = 32
DEFAULT_CATBOOST_ITERATIONS = 300
DEFAULT_CATBOOST_LEARNING_RATE = 0.05
DEFAULT_CATBOOST_DEPTH = 6

CATBOOST_CONFIG = {
    "iterations": DEFAULT_CATBOOST_ITERATIONS,
    "learning_rate": DEFAULT_CATBOOST_LEARNING_RATE,
    "depth": DEFAULT_CATBOOST_DEPTH,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {message}", flush=True)


def _safe_float(value: object) -> float:
    """Convert mixed numeric strings to float; returns np.nan on failure."""
    if value is None:
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _zscore_by_subject(
    df: pd.DataFrame,
    features: List[str],
    subject_col: str,
    clip_z: Optional[float] = None,
) -> pd.DataFrame:
    """Normalize each feature within each subject to reduce inter-individual bias."""
    out = df.copy()
    for feat in features:
        grp = out.groupby(subject_col)[feat]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[feat] = (out[feat] - mean) / std
        out[feat] = out[feat].fillna(0.0)
        if clip_z is not None and clip_z > 0:
            out[feat] = out[feat].clip(lower=-clip_z, upper=clip_z)
    return out


def _robust_scale_by_subject(
    df: pd.DataFrame,
    features: List[str],
    subject_col: str,
    clip_val: Optional[float] = None,
) -> pd.DataFrame:
    """Robustly normalize each feature per subject using median/IQR."""
    out = df.copy()
    for feat in features:
        grp = out.groupby(subject_col)[feat]
        med = grp.transform("median")
        q25 = grp.transform(lambda x: x.quantile(0.25))
        q75 = grp.transform(lambda x: x.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        out[feat] = (out[feat] - med) / iqr
        out[feat] = out[feat].fillna(0.0)
        if clip_val is not None and clip_val > 0:
            out[feat] = out[feat].clip(lower=-clip_val, upper=clip_val)
    return out


def _majority_label(values: np.ndarray) -> int:
    vals = values.astype(int)
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# EmoSurv loader
# ---------------------------------------------------------------------------
EMOSURV_EVENT_NUM_COLS = [
    "index",
    "keyDown",
    "keyUp",
    "D1U1",
    "D1U2",
    "D1D2",
    "U1D2",
    "U1U2",
    "D1U3",
    "D1D3",
]
EMOSURV_INTERVAL_COLS = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]
EMOSURV_SENTINEL_ABS = 1e10
EMOSURV_INTERVAL_MIN_MS = -5000.0
EMOSURV_INTERVAL_MAX_MS = 120000.0
EMOSURV_HOLD_MAX_MS = 10000.0


def _sanitize_emosurv_timing(df: pd.DataFrame) -> int:
    """Drop clearly invalid timing values from free typing sentinel artifacts."""
    invalid_total = 0
    for col in EMOSURV_INTERVAL_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        invalid = s.abs() > EMOSURV_SENTINEL_ABS
        invalid |= s < EMOSURV_INTERVAL_MIN_MS
        invalid |= s > EMOSURV_INTERVAL_MAX_MS
        invalid_total += int(invalid.sum())
        df.loc[invalid, col] = np.nan
    return invalid_total


def _load_emosurv_events(path: str, split_name: str) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(path, sep=";", dtype=str)
    # Normalize user id field names across files.
    if "userId" in df.columns:
        df["user_id"] = df["userId"]
    elif "userid" in df.columns:
        df["user_id"] = df["userid"]
    else:
        raise ValueError(f"No user id column found in {path}")

    df["split"] = split_name
    for col in EMOSURV_EVENT_NUM_COLS:
        if col in df.columns:
            df[col] = df[col].map(_safe_float)
        else:
            df[col] = np.nan

    invalid_timing = _sanitize_emosurv_timing(df)

    if "keyCode" not in df.columns:
        df["keyCode"] = ""

    # Free typing file has an explicit session key.
    if "_id" in df.columns:
        df["session_id"] = df["_id"].astype(str)
    else:
        df["session_id"] = f"{split_name}_default"

    # Key hold time is often informative for motor patterns.
    df["hold_ms"] = df["keyUp"] - df["keyDown"]
    hold_invalid = (df["hold_ms"] < 0) | (df["hold_ms"] > EMOSURV_HOLD_MAX_MS)
    invalid_timing += int(hold_invalid.sum())
    df.loc[hold_invalid, "hold_ms"] = np.nan

    # Keep rows with labels and user ids.
    df = df.dropna(subset=["user_id", "emotionIndex"]).copy()
    return df, invalid_timing


def _aggregate_emosurv_windows(events: pd.DataFrame, window_size: int = 35) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    timing_cols = [
        "D1U1",
        "D1U2",
        "D1D2",
        "U1D2",
        "U1U2",
        "D1U3",
        "D1D3",
        "hold_ms",
    ]

    for (user_id, split_name, emotion, session_id), grp in events.groupby(
        ["user_id", "split", "emotionIndex", "session_id"]
    ):
        grp = grp.sort_values("index")
        n = len(grp)
        if n < 5:
            continue

        # Slice keystrokes into fixed-size windows for augmentation by segmentation.
        for start in range(0, n, window_size):
            chunk = grp.iloc[start:start + window_size]
            if len(chunk) < 8:
                continue

            sample: Dict[str, float] = {
                "subject_id": str(user_id),
                "split": split_name,
                "emotion": str(emotion),
                "session_id": str(session_id),
                "n_keys": float(len(chunk)),
                "window_start": float(start),
            }

            key_series = chunk["keyCode"].fillna("").astype(str)
            backspace_count = key_series.isin(["\\b", "Backspace", "8"]).sum()
            space_count = key_series.isin([" ", "Space", "32"]).sum()
            alpha_count = key_series.str.match(r"^[A-Za-z]$", na=False).sum()
            digit_count = key_series.str.match(r"^[0-9]$", na=False).sum()
            special_count = (~key_series.str.match(r"^[A-Za-z0-9 ]$", na=False)).sum()
            sample["backspace_rate"] = float(backspace_count) / max(len(chunk), 1)
            sample["space_rate"] = float(space_count) / max(len(chunk), 1)
            sample["alpha_key_rate"] = float(alpha_count) / max(len(chunk), 1)
            sample["digit_key_rate"] = float(digit_count) / max(len(chunk), 1)
            sample["special_key_rate"] = float(special_count) / max(len(chunk), 1)
            sample["unique_key_ratio"] = float(key_series.nunique()) / max(len(chunk), 1)

            first_down = np.nanmin(chunk["keyDown"].values)
            last_down = np.nanmax(chunk["keyDown"].values)
            total_time_ms = last_down - first_down if np.isfinite(first_down) and np.isfinite(last_down) else np.nan
            if np.isfinite(total_time_ms) and total_time_ms > 0:
                sample["keys_per_sec"] = float(len(chunk)) / (total_time_ms / 1000.0)
            else:
                sample["keys_per_sec"] = np.nan

            for col in timing_cols:
                vals = chunk[col].dropna().values
                if len(vals) == 0:
                    sample[f"{col}_mean"] = np.nan
                    sample[f"{col}_std"] = np.nan
                    sample[f"{col}_median"] = np.nan
                    sample[f"{col}_q25"] = np.nan
                    sample[f"{col}_q75"] = np.nan
                else:
                    sample[f"{col}_mean"] = float(np.mean(vals))
                    sample[f"{col}_std"] = float(np.std(vals))
                    sample[f"{col}_median"] = float(np.median(vals))
                    sample[f"{col}_q25"] = float(np.percentile(vals, 25))
                    sample[f"{col}_q75"] = float(np.percentile(vals, 75))

            # Rhythm/jump descriptors are text-invariant and useful for stress transfer.
            d1d2_vals = chunk["D1D2"].dropna().values
            hold_vals = chunk["hold_ms"].dropna().values
            u1d2_vals = chunk["U1D2"].dropna().values

            if len(d1d2_vals) > 0:
                d1d2_mean_abs = float(np.mean(np.abs(d1d2_vals)))
                sample["d1d2_cv"] = float(np.std(d1d2_vals) / max(d1d2_mean_abs, 1e-6))
                sample["d1d2_iqr"] = float(np.percentile(d1d2_vals, 75) - np.percentile(d1d2_vals, 25))
                sample["d1d2_jump_std"] = (
                    float(np.std(np.diff(d1d2_vals))) if len(d1d2_vals) >= 2 else np.nan
                )
            else:
                sample["d1d2_cv"] = np.nan
                sample["d1d2_iqr"] = np.nan
                sample["d1d2_jump_std"] = np.nan

            if len(hold_vals) > 0:
                hold_mean_abs = float(np.mean(np.abs(hold_vals)))
                sample["hold_cv"] = float(np.std(hold_vals) / max(hold_mean_abs, 1e-6))
                sample["hold_iqr"] = float(np.percentile(hold_vals, 75) - np.percentile(hold_vals, 25))
                sample["hold_jump_std"] = (
                    float(np.std(np.diff(hold_vals))) if len(hold_vals) >= 2 else np.nan
                )
            else:
                sample["hold_cv"] = np.nan
                sample["hold_iqr"] = np.nan
                sample["hold_jump_std"] = np.nan

            if len(u1d2_vals) > 0:
                sample["pause_rate_300ms"] = float(np.mean(u1d2_vals > 300.0))
                sample["negative_gap_rate"] = float(np.mean(u1d2_vals < 0.0))
            else:
                sample["pause_rate_300ms"] = np.nan
                sample["negative_gap_rate"] = np.nan

            rows.append(sample)

    out = pd.DataFrame(rows)
    return out


def _load_emosurv_frequency_features() -> pd.DataFrame:
    path = os.path.join(EMOSURV_DIR, "Frequency Dataset.csv")
    freq = pd.read_csv(path, sep=";", dtype=str)
    if "User ID" in freq.columns:
        freq["subject_id"] = freq["User ID"].astype(str)
    else:
        freq["subject_id"] = ""

    if "textIndex" in freq.columns:
        text_idx = freq["textIndex"].astype(str).str.upper()
        freq["split"] = np.where(
            text_idx.str.startswith("FI"),
            "fixed",
            np.where(text_idx.str.startswith("FR"), "free", "unknown"),
        )
    else:
        freq["split"] = "unknown"

    for col in ["delFreq", "leftFreq", "TotTime"]:
        if col in freq.columns:
            freq[col] = freq[col].map(_safe_float)
        else:
            freq[col] = np.nan

    # Rates provide content-invariant frequency behavior with shorter latency.
    freq["tot_time_sec"] = freq["TotTime"] / 1000.0
    valid_time = freq["tot_time_sec"] > 0
    freq["del_per_sec"] = np.where(valid_time, freq["delFreq"] / freq["tot_time_sec"], np.nan)
    freq["left_per_sec"] = np.where(valid_time, freq["leftFreq"] / freq["tot_time_sec"], np.nan)

    agg = (
        freq.groupby(["subject_id", "split"]) 
        .agg(
            del_freq_mean=("delFreq", "mean"),
            left_freq_mean=("leftFreq", "mean"),
            total_time_mean=("TotTime", "mean"),
            del_per_sec_mean=("del_per_sec", "mean"),
            left_per_sec_mean=("left_per_sec", "mean"),
            freq_rows=("delFreq", "size"),
        )
        .reset_index()
    )
    return agg


def load_emosurv_dataset(
    window_size: int = 35,
    quick: bool = False,
    clip_z: Optional[float] = None,
    normalization: str = "robust",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    _log("[EmoSurv] Loading fixed/free typing event files...", verbose)
    fixed_path = os.path.join(EMOSURV_DIR, "Fixed Text Typing Dataset.csv")
    free_path = os.path.join(EMOSURV_DIR, "Free Text Typing Dataset.csv")

    fixed, fixed_invalid = _load_emosurv_events(fixed_path, split_name="fixed")
    free, free_invalid = _load_emosurv_events(free_path, split_name="free")
    _log(
        f"[EmoSurv] Raw events loaded: fixed={len(fixed)} free={len(free)}",
        verbose,
    )
    _log(
        f"[EmoSurv] Invalid timing values dropped: fixed={fixed_invalid} free={free_invalid}",
        verbose,
    )
    events = pd.concat([fixed, free], ignore_index=True)

    _log(f"[EmoSurv] Building windowed features (window_size={window_size})...", verbose)
    samples = _aggregate_emosurv_windows(events, window_size=window_size)
    freq = _load_emosurv_frequency_features()
    samples = samples.merge(freq, on=["subject_id", "split"], how="left")
    _log(f"[EmoSurv] Windowed samples: {len(samples)}", verbose)

    # Optional quick mode limits subjects for smoke testing.
    if quick:
        keep_subjects = sorted(samples["subject_id"].unique())[:10]
        samples = samples[samples["subject_id"].isin(keep_subjects)].copy()
        _log(f"[EmoSurv] Quick mode subject count: {len(keep_subjects)}", verbose)

    # Stable class ordering from observed labels.
    labels = sorted(samples["emotion"].dropna().unique())
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    samples["target"] = samples["emotion"].map(label_map).astype(int)

    meta_cols = {"subject_id", "split", "emotion", "session_id", "target", "window_start"}
    feature_cols = [c for c in samples.columns if c not in meta_cols]

    normalization = normalization.lower().strip()
    if normalization == "zscore":
        samples = _zscore_by_subject(
            samples,
            feature_cols,
            subject_col="subject_id",
            clip_z=clip_z,
        )
    elif normalization == "robust":
        samples = _robust_scale_by_subject(
            samples,
            feature_cols,
            subject_col="subject_id",
            clip_val=clip_z,
        )
    elif normalization == "none":
        _log("[EmoSurv] Skipping subject-level normalization", verbose)
    else:
        raise ValueError(f"Unknown EmoSurv normalization mode: {normalization}")

    _log(
        f"[EmoSurv] Final dataset: rows={len(samples)} subjects={samples['subject_id'].nunique()} features={len(feature_cols)} classes={label_map} normalization={normalization}",
        verbose,
    )
    return samples, feature_cols, label_map


# ---------------------------------------------------------------------------
# WESAD loader
# ---------------------------------------------------------------------------
WESAD_KEEP_LABELS = {1, 2, 3, 4}  # baseline, stress, amusement, meditation


def _mmss_to_seconds(value: str) -> Optional[float]:
    """Convert strings like '39.55' to seconds (39 min, 55 sec)."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if "." not in s:
        maybe = _safe_float(s)
        return maybe if np.isfinite(maybe) else None
    parts = s.split(".")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return float(minutes * 60 + seconds)
    except ValueError:
        maybe = _safe_float(s)
        return maybe if np.isfinite(maybe) else None


def _parse_wesad_questionnaire(
    path: str,
) -> Tuple[
    List[str],
    List[Tuple[str, float, float]],
    List[Tuple[float, float]],
    Dict[str, Tuple[float, float]],
]:
    """
    Parse questionnaire metadata.

    Returns:
      order: ordered stage names
      stage_intervals: list of (stage_name, start_sec, end_sec)
      dim_pairs: raw dimensional pairs from # DIM rows
      stage_to_dim: mapped stage -> (valence, arousal)

    The stage_to_dim mapping is intentionally tied to parsed intervals to make
    dimensional labels interval-aware rather than condition-only heuristics.
    """
    order: List[str] = []
    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    dim_pairs: List[Tuple[float, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                continue
            cells = [c.strip() for c in line.split(";")]
            tag = cells[0]
            vals = [v for v in cells[1:] if v != ""]

            if tag.startswith("# ORDER"):
                order = vals
            elif tag.startswith("# START"):
                starts = [_mmss_to_seconds(v) for v in vals]
            elif tag.startswith("# END"):
                ends = [_mmss_to_seconds(v) for v in vals]
            elif tag.startswith("# DIM") and len(vals) >= 2:
                valence = _safe_float(vals[0])
                arousal = _safe_float(vals[1])
                if np.isfinite(valence) and np.isfinite(arousal):
                    dim_pairs.append((float(valence), float(arousal)))

    stage_intervals: List[Tuple[str, float, float]] = []
    for stage, start, end in zip(order, starts, ends):
        if start is None or end is None:
            continue
        if np.isfinite(start) and np.isfinite(end) and end > start:
            stage_intervals.append((stage, float(start), float(end)))

    stage_to_dim: Dict[str, Tuple[float, float]] = {}
    # Most WESAD quest files expose 5 DIM rows that align with
    # Base, TSST, Medi1, Fun, Medi2 in the stage timeline.
    for idx, pair in enumerate(dim_pairs):
        if idx < len(stage_intervals):
            stage_name = stage_intervals[idx][0]
            stage_to_dim[stage_name] = pair

    return order, stage_intervals, dim_pairs, stage_to_dim


def _find_stage_at_time(stage_intervals: List[Tuple[str, float, float]], t_sec: float) -> Optional[str]:
    for stage_name, start, end in stage_intervals:
        if start <= t_sec <= end:
            return stage_name
    return None


def _condition_to_stage_dims(stage_to_dim: Dict[str, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    """
    Build condition-level dimensional targets from questionnaire stages.

    Mapping assumptions:
    - Label 1 baseline -> stage containing 'Base'
    - Label 2 stress -> stage containing 'TSST'
    - Label 3 amusement -> stage containing 'Fun'
    - Label 4 meditation -> average of stages containing 'Medi'
    """
    out: Dict[int, Tuple[float, float]] = {}

    def find_stage(part: str) -> Optional[Tuple[float, float]]:
        for stage, pair in stage_to_dim.items():
            if part.lower() in stage.lower():
                return pair
        return None

    base = find_stage("Base")
    tsst = find_stage("TSST")
    fun = find_stage("Fun")

    medi_pairs = [pair for stage, pair in stage_to_dim.items() if "medi" in stage.lower()]
    medi = None
    if medi_pairs:
        medi = (
            float(np.mean([p[0] for p in medi_pairs])),
            float(np.mean([p[1] for p in medi_pairs])),
        )

    if base is not None:
        out[1] = base
    if tsst is not None:
        out[2] = tsst
    if fun is not None:
        out[3] = fun
    if medi is not None:
        out[4] = medi

    return out


def _extract_chest_channels(signal_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    chest = signal_dict["chest"]
    channels: Dict[str, np.ndarray] = {}

    # ACC may be multichannel. Split it into axes.
    acc = np.asarray(chest["ACC"])
    if acc.ndim == 2 and acc.shape[1] >= 3:
        channels["ACC_x"] = acc[:, 0]
        channels["ACC_y"] = acc[:, 1]
        channels["ACC_z"] = acc[:, 2]
    else:
        channels["ACC"] = acc.reshape(-1)

    for name in ["ECG", "EDA", "EMG", "Resp", "Temp"]:
        arr = np.asarray(chest[name])
        channels[name] = arr.reshape(-1)

    return channels


def _window_channel_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_ptp": np.nan,
            f"{prefix}_diff_std": np.nan,
        }

    diffs = np.diff(values)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_ptp": float(np.ptp(values)),
        f"{prefix}_diff_std": float(np.std(diffs)) if len(diffs) > 0 else 0.0,
    }


def load_wesad_dataset(
    window_seconds: int = 60,
    stride_seconds: int = 30,
    sample_rate_hz: int = 700,
    quick: bool = False,
    clip_z: Optional[float] = None,
    normalization: str = "zscore",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Returns:
      discrete_df: classification dataframe with `target`.
      feature_cols: shared features.
      dim_df: regression dataframe with `valence`, `arousal` targets.
    """
    subject_dirs = sorted(glob.glob(os.path.join(WESAD_DIR, "S*")))
    if quick:
        subject_dirs = subject_dirs[:6]
    _log(
        f"[WESAD] Preparing dataset from {len(subject_dirs)} subjects (window={window_seconds}s stride={stride_seconds}s)",
        verbose,
    )

    rows: List[Dict[str, float]] = []

    win = window_seconds * sample_rate_hz
    stride = stride_seconds * sample_rate_hz

    for sdir in subject_dirs:
        sid = os.path.basename(sdir)
        pkl_path = os.path.join(sdir, f"{sid}.pkl")
        quest_path = os.path.join(sdir, f"{sid}_quest.csv")
        if not os.path.exists(pkl_path):
            continue

        data = pickle.load(open(pkl_path, "rb"), encoding="latin1")
        labels = np.asarray(data["label"]).reshape(-1)
        signals = _extract_chest_channels(data["signal"])

        stage_intervals: List[Tuple[str, float, float]] = []
        stage_to_dim: Dict[str, Tuple[float, float]] = {}
        dim_pairs: List[Tuple[float, float]] = []
        if os.path.exists(quest_path):
            _, stage_intervals, dim_pairs, stage_to_dim = _parse_wesad_questionnaire(quest_path)
        cond_to_dim = _condition_to_stage_dims(stage_to_dim)

        created = 0
        aligned = 0

        n = len(labels)
        for start in range(0, max(0, n - win + 1), stride):
            end = start + win
            y_win = labels[start:end]
            maj = _majority_label(y_win)
            if maj not in WESAD_KEEP_LABELS:
                continue

            sample: Dict[str, float] = {
                "subject_id": sid,
                "target": int(maj),
                "window_start_sec": float(start / sample_rate_hz),
            }

            for ch_name, ch_values in signals.items():
                if len(ch_values) < end:
                    continue
                stats = _window_channel_stats(ch_values[start:end], prefix=ch_name)
                sample.update(stats)

            center_sec = float((start + end) / 2.0 / sample_rate_hz)
            stage_name = _find_stage_at_time(stage_intervals, center_sec)
            sample["stage"] = stage_name if stage_name is not None else "unknown"

            dim_pair = None
            # First try strict interval alignment: window center inside stage bounds.
            if stage_name is not None and stage_name in stage_to_dim:
                dim_pair = stage_to_dim[stage_name]
                aligned += 1

            # Fallback to condition-level mapping where stage lacks explicit DIM value.
            if dim_pair is None:
                dim_pair = cond_to_dim.get(int(maj))

            if dim_pair is None:
                sample["valence"] = np.nan
                sample["arousal"] = np.nan
            else:
                sample["valence"] = float(dim_pair[0])
                sample["arousal"] = float(dim_pair[1])

            rows.append(sample)
            created += 1

        _log(
            f"[WESAD] {sid}: windows={created} interval_aligned={aligned} intervals={len(stage_intervals)} dim_rows={len(dim_pairs)}",
            verbose,
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No WESAD samples were generated. Check dataset path and parsing assumptions.")

    meta_cols = {"subject_id", "target", "valence", "arousal", "stage"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    normalization = normalization.lower().strip()
    if normalization == "zscore":
        # Subject-level normalization for stable LOSO transfer.
        df = _zscore_by_subject(
            df,
            feature_cols,
            subject_col="subject_id",
            clip_z=clip_z,
        )
    elif normalization == "none":
        _log("[WESAD] Skipping subject-level normalization", verbose)
    else:
        raise ValueError(f"Unknown WESAD normalization mode: {normalization}")

    discrete_df = df.dropna(subset=["target"]).copy()
    discrete_df["target"] = discrete_df["target"].astype(int)

    dim_df = df.dropna(subset=["valence", "arousal"]).copy()
    _log(
        f"[WESAD] Final datasets: discrete_rows={len(discrete_df)} dimensional_rows={len(dim_df)} subjects={discrete_df['subject_id'].nunique()} features={len(feature_cols)}",
        verbose,
    )
    return discrete_df, feature_cols, dim_df


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment_classification(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    noise_scale: float = 0.02,
    mask_prob: float = 0.03,
    mode: str = "full",
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment classification data.

    Modes:
      - none: return original data unchanged.
      - balance: class-balance only (oversampling minority classes).
      - full: jitter + masking + class balancing.
    """
    mode = mode.lower().strip()
    if mode not in {"none", "balance", "full"}:
        raise ValueError(f"Unknown classification augmentation mode: {mode}")
    if mode == "none":
        return X, y

    rng = np.random.default_rng(seed)
    if mode == "full":
        scale = np.std(X, axis=0, keepdims=True)
        scale[scale == 0] = 1.0

        jitter = X + rng.normal(0.0, noise_scale, size=X.shape) * scale
        masked = X.copy()
        mask = rng.random(size=X.shape) < mask_prob
        masked[mask] = 0.0

        X_aug = np.vstack([X, jitter, masked])
        y_aug = np.concatenate([y, y, y])
    else:
        X_aug = X
        y_aug = y

    classes, counts = np.unique(y_aug, return_counts=True)
    if len(classes) <= 1:
        return X_aug, y_aug

    max_count = int(np.max(counts))

    parts_x = []
    parts_y = []
    for cls, count in zip(classes, counts):
        idx = np.where(y_aug == cls)[0]
        if count < max_count:
            extra = rng.choice(idx, size=max_count - count, replace=True)
            idx = np.concatenate([idx, extra])
        parts_x.append(X_aug[idx])
        parts_y.append(y_aug[idx])

    X_bal = np.vstack(parts_x)
    y_bal = np.concatenate(parts_y)
    shuffle_idx = rng.permutation(len(y_bal))
    return X_bal[shuffle_idx], y_bal[shuffle_idx]


def augment_regression(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    noise_scale: float = 0.02,
    mask_prob: float = 0.03,
    mode: str = "full",
) -> Tuple[np.ndarray, np.ndarray]:
    mode = mode.lower().strip()
    if mode not in {"none", "full"}:
        raise ValueError(f"Unknown regression augmentation mode: {mode}")
    if mode == "none":
        return X, y

    rng = np.random.default_rng(seed)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale == 0] = 1.0

    jitter = X + rng.normal(0.0, noise_scale, size=X.shape) * scale
    masked = X.copy()
    mask = rng.random(size=X.shape) < mask_prob
    masked[mask] = 0.0

    X_aug = np.vstack([X, jitter, masked])
    y_aug = np.vstack([y, y, y])
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Deep models (Tabular MLP + Sequence Transformer)
# ---------------------------------------------------------------------------
class TorchTabularMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (256, 128), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchSequenceTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_len: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F) | mask: (B, T) with True for padded tokens
        b, t, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos[:, :t, :]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(x)


class TorchTemporalConvNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            dilation = 2 ** layer_idx
            padding = dilation * (kernel_size - 1) // 2
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F) | mask: (B, T) with True for padded tokens
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.input_proj(x)  # (B, T, H)
        h = h.transpose(1, 2)  # (B, H, T)
        for block in self.blocks:
            residual = h
            h = block(h)
            if h.shape == residual.shape:
                h = h + residual
        h = h.transpose(1, 2)  # (B, T, H)
        return self.head(h)


class TorchBiLSTMAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F) | mask: (B, T) with True for padded tokens
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=mask)
        h = h + attn_out
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        return self.head(h)


class _SeqClassificationDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], labels: List[np.ndarray]):
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.seqs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class _SeqRegressionDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], labels: List[np.ndarray]):
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.seqs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def _collate_seq_classification(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    seqs, ys = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    n_feat = seqs[0].shape[1]

    x_pad = torch.zeros(len(seqs), max_len, n_feat)
    y_pad = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)

    for i, (s, y, ln) in enumerate(zip(seqs, ys, lengths)):
        x_pad[i, :ln] = s
        y_pad[i, :ln] = y
        mask[i, :ln] = False

    return x_pad, y_pad, mask, lengths


def _collate_seq_regression(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    seqs, ys = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    n_feat = seqs[0].shape[1]
    out_dim = ys[0].shape[1]

    x_pad = torch.zeros(len(seqs), max_len, n_feat)
    y_pad = torch.zeros(len(seqs), max_len, out_dim)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)

    for i, (s, y, ln) in enumerate(zip(seqs, ys, lengths)):
        x_pad[i, :ln] = s
        y_pad[i, :ln] = y
        mask[i, :ln] = False

    return x_pad, y_pad, mask, lengths


def _infer_sequence_layout(df: pd.DataFrame, subject_col: str) -> Tuple[List[str], str]:
    if {"split", "emotion", "window_start"}.issubset(df.columns):
        return [subject_col, "split", "emotion"], "window_start"
    if "window_start_sec" in df.columns:
        return [subject_col], "window_start_sec"
    return [subject_col], "__order_idx"


def _build_classification_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    subject_col: str,
    seq_len: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    work = df.copy()
    group_cols, order_col = _infer_sequence_layout(work, subject_col)
    if order_col == "__order_idx":
        work[order_col] = np.arange(len(work))

    seqs: List[np.ndarray] = []
    labs: List[np.ndarray] = []

    for _, grp in work.groupby(group_cols):
        grp = grp.sort_values(order_col)
        x = grp[feature_cols].values.astype(np.float32)
        y = grp[target_col].values.astype(np.int64)
        for start in range(0, len(grp), seq_len):
            end = start + seq_len
            if end - start < 8 and len(grp) > 8:
                continue
            x_chunk = x[start:end]
            y_chunk = y[start:end]
            if len(x_chunk) < 2:
                continue
            seqs.append(x_chunk)
            labs.append(y_chunk)

    return seqs, labs


def augment_sequence_classification(
    seqs: List[np.ndarray],
    labels: List[np.ndarray],
    seed: int = 42,
    noise_scale: float = 0.02,
    mask_prob: float = 0.03,
    mode: str = "none",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Augment sequence classification data.

    Modes:
      - none: return original sequence dataset unchanged.
      - balance: oversample minority majority-label sequence classes.
      - full: jitter + masking per sequence, then class balancing.
    """
    mode = mode.lower().strip()
    if mode not in {"none", "balance", "full"}:
        raise ValueError(f"Unknown sequence classification augmentation mode: {mode}")
    if mode == "none" or len(seqs) == 0:
        return seqs, labels

    rng = np.random.default_rng(seed)
    aug_seqs = [s.copy() for s in seqs]
    aug_labels = [y.copy() for y in labels]

    if mode == "full":
        for s, y in zip(seqs, labels):
            scale = np.std(s, axis=0, keepdims=True)
            scale[scale == 0] = 1.0

            jitter = s + rng.normal(0.0, noise_scale, size=s.shape) * scale
            masked = s.copy()
            mask = rng.random(size=s.shape) < mask_prob
            masked[mask] = 0.0

            aug_seqs.append(jitter.astype(np.float32))
            aug_labels.append(y.copy())
            aug_seqs.append(masked.astype(np.float32))
            aug_labels.append(y.copy())

    major = np.array([_majority_label(y.astype(int)) for y in aug_labels], dtype=int)
    classes, counts = np.unique(major, return_counts=True)
    if len(classes) <= 1:
        return aug_seqs, aug_labels

    max_count = int(np.max(counts))
    all_indices = np.arange(len(aug_seqs))
    balanced_idx: List[int] = []

    for cls, count in zip(classes, counts):
        idx = all_indices[major == cls]
        if count < max_count:
            extra = rng.choice(idx, size=max_count - count, replace=True)
            idx = np.concatenate([idx, extra])
        balanced_idx.extend(idx.tolist())

    rng.shuffle(balanced_idx)
    out_seqs = [aug_seqs[i].copy() for i in balanced_idx]
    out_labels = [aug_labels[i].copy() for i in balanced_idx]
    return out_seqs, out_labels


def _build_regression_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Tuple[str, str],
    subject_col: str,
    seq_len: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    work = df.copy()
    group_cols, order_col = _infer_sequence_layout(work, subject_col)
    if order_col == "__order_idx":
        work[order_col] = np.arange(len(work))

    seqs: List[np.ndarray] = []
    labs: List[np.ndarray] = []

    for _, grp in work.groupby(group_cols):
        grp = grp.sort_values(order_col)
        x = grp[feature_cols].values.astype(np.float32)
        y = grp[list(target_cols)].values.astype(np.float32)
        for start in range(0, len(grp), seq_len):
            end = start + seq_len
            if end - start < 8 and len(grp) > 8:
                continue
            x_chunk = x[start:end]
            y_chunk = y[start:end]
            if len(x_chunk) < 2:
                continue
            seqs.append(x_chunk)
            labs.append(y_chunk)

    return seqs, labs


def _fit_predict_torch_mlp_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model = TorchTabularMLP(x_train.shape[1], num_classes).to(DEVICE)
    weights = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = np.where(weights > 0, weights.sum() / np.maximum(weights, 1), 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = np.argmax(probs, axis=1)

    return pred, probs


def _fit_predict_torch_transformer_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    subject_col: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    deep_augment_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _fit_predict_torch_sequence_classifier(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        subject_col=subject_col,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        deep_augment_mode=deep_augment_mode,
        architecture="transformer",
    )


def _fit_predict_torch_tcn_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    subject_col: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    deep_augment_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _fit_predict_torch_sequence_classifier(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        subject_col=subject_col,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        deep_augment_mode=deep_augment_mode,
        architecture="tcn",
    )


def _fit_predict_torch_bilstm_attention_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    subject_col: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    deep_augment_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _fit_predict_torch_sequence_classifier(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        subject_col=subject_col,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        deep_augment_mode=deep_augment_mode,
        architecture="bilstm_attention",
    )


def _fit_predict_torch_sequence_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    subject_col: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    deep_augment_mode: str,
    architecture: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_seqs, train_labs = _build_classification_sequences(
        train_df,
        feature_cols,
        target_col,
        subject_col,
        seq_len,
    )
    test_seqs, test_labs = _build_classification_sequences(
        test_df,
        feature_cols,
        target_col,
        subject_col,
        seq_len,
    )

    if len(train_seqs) == 0 or len(test_seqs) == 0:
        # Fallback path when sequence construction is too sparse.
        yt = test_df[target_col].values.astype(int)
        pred, probs = _fit_predict_torch_mlp_classifier(
            train_df[feature_cols].values.astype(np.float32),
            train_df[target_col].values.astype(int),
            test_df[feature_cols].values.astype(np.float32),
            num_classes,
            epochs,
            batch_size,
            lr,
        )
        return yt, pred, probs

    train_seqs, train_labs = augment_sequence_classification(
        train_seqs,
        train_labs,
        seed=42,
        mode=deep_augment_mode,
    )

    arch = architecture.lower().strip()
    if arch == "transformer":
        model = TorchSequenceTransformer(
            in_dim=len(feature_cols),
            out_dim=num_classes,
            max_len=max(seq_len, 32),
        ).to(DEVICE)
    elif arch == "tcn":
        model = TorchTemporalConvNet(
            in_dim=len(feature_cols),
            out_dim=num_classes,
        ).to(DEVICE)
    elif arch == "bilstm_attention":
        model = TorchBiLSTMAttention(
            in_dim=len(feature_cols),
            out_dim=num_classes,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown deep sequence classifier architecture: {architecture}")

    y_flat = np.concatenate(train_labs)
    weights = np.bincount(y_flat, minlength=num_classes).astype(np.float32)
    weights = np.where(weights > 0, weights.sum() / np.maximum(weights, 1), 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = _SeqClassificationDataset(train_seqs, train_labs)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=_collate_seq_classification,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb, mask, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            mask = mask.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb, mask=mask)
            loss = criterion(logits.reshape(-1, num_classes), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_ds = _SeqClassificationDataset(test_seqs, test_labs)
    test_loader = DataLoader(
        test_ds,
        batch_size=min(batch_size, len(test_ds)),
        shuffle=False,
        collate_fn=_collate_seq_classification,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for xb, yb, mask, lengths in test_loader:
            xb = xb.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(xb, mask=mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = np.argmax(probs, axis=-1)

            for i, ln in enumerate(lengths):
                y_true.extend(yb[i, :ln].numpy().tolist())
                y_pred.extend(pred[i, :ln].tolist())
                y_prob.extend(probs[i, :ln, :])

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def _fit_predict_torch_mlp_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    out_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> np.ndarray:
    model = TorchTabularMLP(x_train.shape[1], out_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return pred


def _fit_predict_torch_transformer_regressor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Tuple[str, str],
    subject_col: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_seqs, train_labs = _build_regression_sequences(
        train_df,
        feature_cols,
        target_cols,
        subject_col,
        seq_len,
    )
    test_seqs, test_labs = _build_regression_sequences(
        test_df,
        feature_cols,
        target_cols,
        subject_col,
        seq_len,
    )

    if len(train_seqs) == 0 or len(test_seqs) == 0:
        yt = test_df[list(target_cols)].values.astype(np.float32)
        pred = _fit_predict_torch_mlp_regressor(
            train_df[feature_cols].values.astype(np.float32),
            train_df[list(target_cols)].values.astype(np.float32),
            test_df[feature_cols].values.astype(np.float32),
            out_dim=2,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        return yt, pred

    model = TorchSequenceTransformer(
        in_dim=len(feature_cols),
        out_dim=2,
        max_len=max(seq_len, 32),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = _SeqRegressionDataset(train_seqs, train_labs)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=_collate_seq_regression,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb, mask, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            mask = mask.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, mask=mask)
            valid = (~mask).unsqueeze(-1).float()
            sq_err = (pred - yb) ** 2
            denom = valid.sum() * sq_err.shape[-1]
            loss = (sq_err * valid).sum() / torch.clamp(denom, min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_ds = _SeqRegressionDataset(test_seqs, test_labs)
    test_loader = DataLoader(
        test_ds,
        batch_size=min(batch_size, len(test_ds)),
        shuffle=False,
        collate_fn=_collate_seq_regression,
    )

    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for xb, yb, mask, lengths in test_loader:
            xb = xb.to(DEVICE)
            mask = mask.to(DEVICE)
            pred = model(xb, mask=mask).cpu().numpy()
            yb_np = yb.numpy()
            for i, ln in enumerate(lengths):
                y_true.append(yb_np[i, :ln, :])
                y_pred.append(pred[i, :ln, :])

    return np.vstack(y_true), np.vstack(y_pred)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DEEP_CLASSIFIER_NAMES = [
    "TorchTransformer",
    "TorchTemporalConvNet",
    "TorchBiLSTMAttention",
]
DEEP_REGRESSOR_NAMES = ["TorchMLPRegressor", "TorchTransformerRegressor"]

MODERN_A_CATBOOST_CLASS_MODELS = [
    "LogisticRegression",
    "RandomForest",
    "CatBoostClassifier",
    "TorchTransformer",
    "TorchTemporalConvNet",
    "TorchBiLSTMAttention",
]


def _make_catboost_classifier():
    try:
        import importlib

        catboost_module = importlib.import_module("catboost")
        CatBoostClassifier = getattr(catboost_module, "CatBoostClassifier")
    except Exception as exc:
        raise ImportError(
            "CatBoost is not installed. Install it with `pip install catboost` or remove CatBoostClassifier from --class-models."
        ) from exc

    return CatBoostClassifier(
        iterations=CATBOOST_CONFIG["iterations"],
        learning_rate=CATBOOST_CONFIG["learning_rate"],
        depth=CATBOOST_CONFIG["depth"],
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def _configure_catboost_runtime(
    iterations: int,
    learning_rate: float,
    depth: int,
    quick: bool,
) -> None:
    iters = max(20, int(iterations))
    if quick and iterations == DEFAULT_CATBOOST_ITERATIONS:
        iters = min(iters, 120)

    CATBOOST_CONFIG["iterations"] = iters
    CATBOOST_CONFIG["learning_rate"] = float(max(1e-4, learning_rate))
    CATBOOST_CONFIG["depth"] = int(max(2, depth))


CLASSIFIERS = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": lambda: LogisticRegression(
        C=1.0,
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    ),
    "CatBoostClassifier": lambda: _make_catboost_classifier(),
}


REGRESSORS = {
    "RandomForestRegressor": lambda: RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoostingRegressor": lambda: MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        )
    ),
    "SVR": lambda: MultiOutputRegressor(
        SVR(C=5.0, epsilon=0.1, gamma="scale")
    ),
    "Ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "MLPRegressor": lambda: MLPRegressor(
        hidden_layer_sizes=(128, 64),
        learning_rate="adaptive",
        max_iter=800,
        early_stopping=True,
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class ClassificationRun:
    summary_df: pd.DataFrame
    folds_df: pd.DataFrame
    y_true_by_model: Dict[str, np.ndarray]
    y_pred_by_model: Dict[str, np.ndarray]


@dataclass
class RegressionRun:
    summary_df: pd.DataFrame
    folds_df: pd.DataFrame
    y_true_by_model: Dict[str, np.ndarray]
    y_pred_by_model: Dict[str, np.ndarray]


def run_loso_classification(
    df: pd.DataFrame,
    feature_cols: List[str],
    subject_col: str,
    target_col: str,
    model_names: Optional[List[str]] = None,
    include_deep: bool = False,
    deep_epochs: int = DEFAULT_DEEP_EPOCHS,
    deep_batch_size: int = DEFAULT_DEEP_BATCH_SIZE,
    deep_lr: float = DEFAULT_DEEP_LR,
    classical_augment_mode: str = "none",
    deep_augment_mode: str = "full",
    seq_len: int = DEFAULT_SEQ_LEN,
    verbose: bool = True,
) -> ClassificationRun:
    subjects = sorted(df[subject_col].unique())
    model_keys = model_names if model_names else list(CLASSIFIERS.keys())
    if include_deep and model_names is None:
        model_keys = model_keys + DEEP_CLASSIFIER_NAMES

    summaries = []
    all_folds = []
    y_true_by_model: Dict[str, np.ndarray] = {}
    y_pred_by_model: Dict[str, np.ndarray] = {}

    _log(f"[Classification] Starting LOSO with {len(model_keys)} models over {len(subjects)} subjects", verbose)

    for model_name in model_keys:
        _log(f"[Classification] Model: {model_name}", verbose)
        model_fn = CLASSIFIERS.get(model_name)
        fold_rows = []
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        for subj in subjects:
            te_mask = df[subject_col] == subj
            tr_mask = ~te_mask

            X_train = df.loc[tr_mask, feature_cols].values.astype(float)
            y_train = df.loc[tr_mask, target_col].values.astype(int)
            X_test = df.loc[te_mask, feature_cols].values.astype(float)
            y_test = df.loc[te_mask, target_col].values.astype(int)

            if len(np.unique(y_test)) < 2:
                # Keep fold but mark AUC as nan when only one class appears in test.
                pass

            imp = SimpleImputer(strategy="constant", fill_value=0.0)
            X_train = imp.fit_transform(X_train)
            X_test = imp.transform(X_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            auc_val = np.nan
            y_prob = None

            if model_name in CLASSIFIERS:
                X_aug, y_aug = augment_classification(
                    X_train,
                    y_train,
                    seed=42,
                    mode=classical_augment_mode,
                )
                model = model_fn()
                model.fit(X_aug, y_aug)
                y_pred = np.asarray(model.predict(X_test)).reshape(-1)
                if y_pred.dtype.kind in {"f", "i", "u"}:
                    y_pred = y_pred.astype(int)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)

            elif model_name in ["TorchTransformer", "TorchTemporalConvNet", "TorchBiLSTMAttention"]:
                class_values = sorted(np.unique(y_train).tolist())
                class_to_idx = {c: i for i, c in enumerate(class_values)}
                idx_to_class = np.array(class_values)

                if not set(np.unique(y_test)).issubset(set(class_values)):
                    y_pred = np.full_like(y_test, fill_value=class_values[0])
                else:
                    train_seq_df = df.loc[tr_mask, [subject_col, target_col] + feature_cols].copy()
                    test_seq_df = df.loc[te_mask, [subject_col, target_col] + feature_cols].copy()
                    for col in ["split", "emotion", "window_start", "window_start_sec"]:
                        if col in df.columns:
                            train_seq_df[col] = df.loc[tr_mask, col].values
                            test_seq_df[col] = df.loc[te_mask, col].values
                    train_seq_df[feature_cols] = X_train
                    test_seq_df[feature_cols] = X_test
                    train_seq_df[target_col] = train_seq_df[target_col].map(class_to_idx).astype(int)
                    test_seq_df[target_col] = test_seq_df[target_col].map(class_to_idx).astype(int)

                    if model_name == "TorchTransformer":
                        yt_idx, yp_idx, y_prob = _fit_predict_torch_transformer_classifier(
                            train_df=train_seq_df,
                            test_df=test_seq_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            subject_col=subject_col,
                            num_classes=len(class_values),
                            epochs=deep_epochs,
                            batch_size=deep_batch_size,
                            lr=deep_lr,
                            seq_len=seq_len,
                            deep_augment_mode=deep_augment_mode,
                        )
                    elif model_name == "TorchTemporalConvNet":
                        yt_idx, yp_idx, y_prob = _fit_predict_torch_tcn_classifier(
                            train_df=train_seq_df,
                            test_df=test_seq_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            subject_col=subject_col,
                            num_classes=len(class_values),
                            epochs=deep_epochs,
                            batch_size=deep_batch_size,
                            lr=deep_lr,
                            seq_len=seq_len,
                            deep_augment_mode=deep_augment_mode,
                        )
                    else:
                        yt_idx, yp_idx, y_prob = _fit_predict_torch_bilstm_attention_classifier(
                            train_df=train_seq_df,
                            test_df=test_seq_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            subject_col=subject_col,
                            num_classes=len(class_values),
                            epochs=deep_epochs,
                            batch_size=deep_batch_size,
                            lr=deep_lr,
                            seq_len=seq_len,
                            deep_augment_mode=deep_augment_mode,
                        )

                    y_test = idx_to_class[yt_idx]
                    y_pred = idx_to_class[yp_idx]

            else:
                raise ValueError(f"Unknown classification model: {model_name}")

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

            if y_prob is not None:
                try:
                    if len(np.unique(y_test)) > 1:
                        auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except Exception:
                    auc_val = np.nan

            fold_rows.append(
                {
                    "participant": subj,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                    "auc_ovr": auc_val,
                    "n_test": len(y_test),
                }
            )
            _log(
                f"  [Classification:{model_name}] subject={subj} n_test={len(y_test)} acc={fold_rows[-1]['accuracy']:.3f} f1={fold_rows[-1]['f1_macro']:.3f}",
                verbose,
            )

        folds = pd.DataFrame(fold_rows)
        folds["model"] = model_name
        all_folds.append(folds)

        summary = {
            "model": model_name,
            "accuracy_mean": folds["accuracy"].mean(),
            "accuracy_std": folds["accuracy"].std(),
            "f1_macro_mean": folds["f1_macro"].mean(),
            "f1_macro_std": folds["f1_macro"].std(),
            "precision_macro_mean": folds["precision_macro"].mean(),
            "precision_macro_std": folds["precision_macro"].std(),
            "recall_macro_mean": folds["recall_macro"].mean(),
            "recall_macro_std": folds["recall_macro"].std(),
            "auc_ovr_mean": folds["auc_ovr"].mean(),
            "auc_ovr_std": folds["auc_ovr"].std(),
            "n_folds": len(folds),
        }
        summaries.append(summary)

        y_true_by_model[model_name] = np.array(y_true_all)
        y_pred_by_model[model_name] = np.array(y_pred_all)
        _log(
            f"[Classification] Finished {model_name}: acc={summary['accuracy_mean']:.3f} f1={summary['f1_macro_mean']:.3f} auc={summary['auc_ovr_mean']:.3f}",
            verbose,
        )

    summary_df = pd.DataFrame(summaries).sort_values("f1_macro_mean", ascending=False)
    folds_df = pd.concat(all_folds, ignore_index=True)
    return ClassificationRun(summary_df, folds_df, y_true_by_model, y_pred_by_model)


def filter_subjects_for_classification(
    df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    min_samples: int = 0,
    min_classes: int = 0,
    verbose: bool = True,
    context: str = "Classification",
) -> pd.DataFrame:
    if min_samples <= 0 and min_classes <= 0:
        return df

    stats = (
        df.groupby(subject_col)[target_col]
        .agg(n_samples="size", n_classes="nunique")
        .reset_index()
    )
    keep = stats[subject_col]
    if min_samples > 0:
        keep = keep[stats["n_samples"] >= min_samples]
        stats = stats[stats[subject_col].isin(keep)]
    if min_classes > 0:
        keep = keep[stats["n_classes"] >= min_classes]

    keep_set = set(keep.tolist())
    out = df[df[subject_col].isin(keep_set)].copy()
    _log(
        (
            f"[{context}] Subject quality filter -> kept {len(keep_set)}/{df[subject_col].nunique()} "
            f"subjects and {len(out)}/{len(df)} rows "
            f"(min_samples={min_samples}, min_classes={min_classes})"
        ),
        verbose,
    )
    return out


def filter_subjects_for_regression(
    df: pd.DataFrame,
    subject_col: str,
    min_samples: int = 0,
    verbose: bool = True,
    context: str = "Regression",
) -> pd.DataFrame:
    if min_samples <= 0:
        return df

    counts = df.groupby(subject_col).size().reset_index(name="n_samples")
    keep = set(counts.loc[counts["n_samples"] >= min_samples, subject_col].tolist())
    out = df[df[subject_col].isin(keep)].copy()
    _log(
        (
            f"[{context}] Subject quality filter -> kept {len(keep)}/{df[subject_col].nunique()} "
            f"subjects and {len(out)}/{len(df)} rows (min_samples={min_samples})"
        ),
        verbose,
    )
    return out


def run_loso_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    subject_col: str,
    target_cols: Tuple[str, str] = ("valence", "arousal"),
    model_names: Optional[List[str]] = None,
    include_deep: bool = False,
    deep_epochs: int = DEFAULT_DEEP_EPOCHS,
    deep_batch_size: int = DEFAULT_DEEP_BATCH_SIZE,
    deep_lr: float = DEFAULT_DEEP_LR,
    classical_augment_mode: str = "none",
    deep_augment_mode: str = "full",
    seq_len: int = DEFAULT_SEQ_LEN,
    verbose: bool = True,
) -> RegressionRun:
    subjects = sorted(df[subject_col].unique())
    model_keys = model_names if model_names else list(REGRESSORS.keys())
    if include_deep and model_names is None:
        model_keys = model_keys + DEEP_REGRESSOR_NAMES

    summaries = []
    all_folds = []
    y_true_by_model: Dict[str, np.ndarray] = {}
    y_pred_by_model: Dict[str, np.ndarray] = {}

    _log(f"[Regression] Starting LOSO with {len(model_keys)} models over {len(subjects)} subjects", verbose)

    for model_name in model_keys:
        _log(f"[Regression] Model: {model_name}", verbose)
        model_fn = REGRESSORS.get(model_name)
        fold_rows = []
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []

        for subj in subjects:
            te_mask = df[subject_col] == subj
            tr_mask = ~te_mask

            X_train = df.loc[tr_mask, feature_cols].values.astype(float)
            y_train = df.loc[tr_mask, list(target_cols)].values.astype(float)
            X_test = df.loc[te_mask, feature_cols].values.astype(float)
            y_test = df.loc[te_mask, list(target_cols)].values.astype(float)

            imp = SimpleImputer(strategy="constant", fill_value=0.0)
            X_train = imp.fit_transform(X_train)
            X_test = imp.transform(X_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if model_name in REGRESSORS:
                X_aug, y_aug = augment_regression(
                    X_train,
                    y_train,
                    seed=42,
                    mode=classical_augment_mode,
                )
                model = model_fn()
                model.fit(X_aug, y_aug)
                y_pred = model.predict(X_test)

            elif model_name == "TorchMLPRegressor":
                X_aug, y_aug = augment_regression(
                    X_train,
                    y_train,
                    seed=42,
                    mode=deep_augment_mode,
                )
                y_pred = _fit_predict_torch_mlp_regressor(
                    x_train=X_aug.astype(np.float32),
                    y_train=y_aug.astype(np.float32),
                    x_test=X_test.astype(np.float32),
                    out_dim=2,
                    epochs=deep_epochs,
                    batch_size=deep_batch_size,
                    lr=deep_lr,
                )

            elif model_name == "TorchTransformerRegressor":
                train_seq_df = df.loc[tr_mask, [subject_col] + list(target_cols) + feature_cols].copy()
                test_seq_df = df.loc[te_mask, [subject_col] + list(target_cols) + feature_cols].copy()
                for col in ["stage", "window_start", "window_start_sec"]:
                    if col in df.columns:
                        train_seq_df[col] = df.loc[tr_mask, col].values
                        test_seq_df[col] = df.loc[te_mask, col].values
                train_seq_df[feature_cols] = X_train
                test_seq_df[feature_cols] = X_test

                y_test_seq, y_pred_seq = _fit_predict_torch_transformer_regressor(
                    train_df=train_seq_df,
                    test_df=test_seq_df,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    subject_col=subject_col,
                    epochs=deep_epochs,
                    batch_size=deep_batch_size,
                    lr=deep_lr,
                    seq_len=seq_len,
                )
                y_test = y_test_seq
                y_pred = y_pred_seq

            else:
                raise ValueError(f"Unknown regression model: {model_name}")

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

            mae_v = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
            mae_a = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
            mse_v = mean_squared_error(y_test[:, 0], y_pred[:, 0])
            mse_a = mean_squared_error(y_test[:, 1], y_pred[:, 1])
            r2_v = r2_score(y_test[:, 0], y_pred[:, 0])
            r2_a = r2_score(y_test[:, 1], y_pred[:, 1])

            fold_rows.append(
                {
                    "participant": subj,
                    "mae_valence": mae_v,
                    "mae_arousal": mae_a,
                    "mae_mean": float((mae_v + mae_a) / 2.0),
                    "rmse_valence": float(np.sqrt(mse_v)),
                    "rmse_arousal": float(np.sqrt(mse_a)),
                    "rmse_mean": float((np.sqrt(mse_v) + np.sqrt(mse_a)) / 2.0),
                    "r2_valence": r2_v,
                    "r2_arousal": r2_a,
                    "r2_mean": float((r2_v + r2_a) / 2.0),
                    "n_test": len(y_test),
                }
            )
            _log(
                f"  [Regression:{model_name}] subject={subj} n_test={len(y_test)} mae={fold_rows[-1]['mae_mean']:.3f} r2={fold_rows[-1]['r2_mean']:.3f}",
                verbose,
            )

        folds = pd.DataFrame(fold_rows)
        folds["model"] = model_name
        all_folds.append(folds)

        summary = {
            "model": model_name,
            "mae_mean": folds["mae_mean"].mean(),
            "mae_std": folds["mae_mean"].std(),
            "rmse_mean": folds["rmse_mean"].mean(),
            "rmse_std": folds["rmse_mean"].std(),
            "r2_mean": folds["r2_mean"].mean(),
            "r2_std": folds["r2_mean"].std(),
            "n_folds": len(folds),
        }
        summaries.append(summary)

        y_true_by_model[model_name] = np.vstack(y_true_all)
        y_pred_by_model[model_name] = np.vstack(y_pred_all)
        _log(
            f"[Regression] Finished {model_name}: mae={summary['mae_mean']:.3f} rmse={summary['rmse_mean']:.3f} r2={summary['r2_mean']:.3f}",
            verbose,
        )

    summary_df = pd.DataFrame(summaries).sort_values("r2_mean", ascending=False)
    folds_df = pd.concat(all_folds, ignore_index=True)
    return RegressionRun(summary_df, folds_df, y_true_by_model, y_pred_by_model)


# ---------------------------------------------------------------------------
# Plots: classification
# ---------------------------------------------------------------------------
def plot_classification_bars(summary_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    order = summary_df["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("f1_macro_mean", "f1_macro_std", "F1 Macro"),
        ("auc_ovr_mean", "auc_ovr_std", "AUC OVR"),
    ]

    for ax, (m_col, s_col, title) in zip(axes, specs):
        vals = summary_df[m_col].values
        errs = summary_df[s_col].fillna(0).values
        bars = ax.bar(order, vals, yerr=errs, capsize=4, color=sns.color_palette("viridis", len(order)))
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Classification Model Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison_bars.png"))
    plt.close(fig)


def plot_classification_boxplots(folds_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    specs = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 Macro"),
        ("auc_ovr", "AUC OVR"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        sns.boxplot(data=folds_df, x="model", y=metric, ax=ax, palette="viridis")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fold_boxplots.png"))
    plt.close(fig)


def plot_confusion_for_best(
    summary_df: pd.DataFrame,
    y_true_by_model: Dict[str, np.ndarray],
    y_pred_by_model: Dict[str, np.ndarray],
    out_dir: str,
) -> None:
    _set_style()
    best = summary_df.iloc[0]["model"]
    yt = y_true_by_model[best]
    yp = y_pred_by_model[best]

    labels = sorted(np.unique(np.concatenate([yt, yp])))
    cm = confusion_matrix(yt, yp, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (Best Model: {best})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_best.png"))
    plt.close(fig)


def plot_classification_participant_heatmap(folds_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    pivot = folds_df.pivot_table(index="participant", columns="model", values="f1_macro")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Per-Participant F1 Macro")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "participant_heatmap.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots: regression
# ---------------------------------------------------------------------------
def plot_regression_bars(summary_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    order = summary_df["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("mae_mean", "mae_std", "MAE (lower better)", 0.0, None),
        ("rmse_mean", "rmse_std", "RMSE (lower better)", 0.0, None),
        ("r2_mean", "r2_std", "R2 (higher better)", -1.0, 1.0),
    ]

    for ax, (m_col, s_col, title, ymin, ymax) in zip(axes, specs):
        vals = summary_df[m_col].values
        errs = summary_df[s_col].fillna(0).values
        bars = ax.bar(order, vals, yerr=errs, capsize=4, color=sns.color_palette("mako", len(order)))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        if ymax is not None:
            ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Regression Model Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regression_model_bars.png"))
    plt.close(fig)


def plot_regression_boxplots(folds_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    specs = [
        ("mae_mean", "MAE"),
        ("rmse_mean", "RMSE"),
        ("r2_mean", "R2"),
    ]

    for ax, (metric, title) in zip(axes, specs):
        sns.boxplot(data=folds_df, x="model", y=metric, ax=ax, palette="mako")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regression_fold_boxplots.png"))
    plt.close(fig)


def plot_regression_scatter_for_best(
    summary_df: pd.DataFrame,
    y_true_by_model: Dict[str, np.ndarray],
    y_pred_by_model: Dict[str, np.ndarray],
    out_dir: str,
) -> None:
    _set_style()
    best = summary_df.iloc[0]["model"]
    yt = y_true_by_model[best]
    yp = y_pred_by_model[best]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ["Valence", "Arousal"]

    for idx, ax in enumerate(axes):
        ax.scatter(yt[:, idx], yp[:, idx], alpha=0.35, s=10)
        lo = min(float(np.min(yt[:, idx])), float(np.min(yp[:, idx])))
        hi = max(float(np.max(yt[:, idx])), float(np.max(yp[:, idx])))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{names[idx]}: Pred vs True")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")

    fig.suptitle(f"Regression Scatter (Best Model: {best})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regression_scatter_best.png"))
    plt.close(fig)


def plot_regression_participant_heatmap(folds_df: pd.DataFrame, out_dir: str) -> None:
    _set_style()
    pivot = folds_df.pivot_table(index="participant", columns="model", values="r2_mean")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1.0, vmax=1.0, ax=ax)
    ax.set_title("Per-Participant R2 Mean")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "participant_heatmap_regression.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _save_run_artifacts(
    dataset_name: str,
    task_name: str,
    summary_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    metadata: Dict[str, object],
    verbose: bool = True,
) -> str:
    run_dir = os.path.join(RESULTS_ROOT, dataset_name, task_name)
    fold_dir = os.path.join(run_dir, "fold_results")
    plot_dir = os.path.join(run_dir, "plots")

    _ensure_dir(run_dir)
    _ensure_dir(fold_dir)
    _ensure_dir(plot_dir)

    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    folds_df.to_csv(os.path.join(run_dir, "all_folds.csv"), index=False)

    for model_name, model_df in folds_df.groupby("model"):
        model_df.to_csv(os.path.join(fold_dir, f"{model_name}_folds.csv"), index=False)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "task": task_name,
        "n_models": int(summary_df.shape[0]),
        "best_model": str(summary_df.iloc[0]["model"]),
        "metadata": metadata,
    }
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _log(f"[Artifacts] Saved results to {run_dir}", verbose)

    return plot_dir


def run_emosurv_discrete(args: argparse.Namespace) -> None:
    _log("[Run] EmoSurv discrete task started", not args.quiet)
    data, feature_cols, label_map = load_emosurv_dataset(
        window_size=args.emosurv_window,
        quick=args.quick,
        clip_z=args.clip_z,
        normalization=args.emosurv_normalization,
        verbose=not args.quiet,
    )

    data = filter_subjects_for_classification(
        data,
        subject_col="subject_id",
        target_col="target",
        min_samples=args.min_subject_samples,
        min_classes=args.min_subject_classes,
        verbose=not args.quiet,
        context="EmoSurv",
    )
    if data["subject_id"].nunique() < 2:
        raise RuntimeError("Not enough subjects after filtering for LOSO evaluation.")

    run = run_loso_classification(
        df=data,
        feature_cols=feature_cols,
        subject_col="subject_id",
        target_col="target",
        model_names=args.class_models,
        include_deep=args.include_deep,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
        deep_lr=args.deep_lr,
        classical_augment_mode=args.classical_augment,
        deep_augment_mode=args.deep_augment,
        seq_len=args.seq_len,
        verbose=not args.quiet,
    )

    plot_dir = _save_run_artifacts(
        dataset_name="emosurv",
        task_name="discrete",
        summary_df=run.summary_df,
        folds_df=run.folds_df,
        metadata={
            "n_samples": int(len(data)),
            "n_subjects": int(data["subject_id"].nunique()),
            "n_features": int(len(feature_cols)),
            "label_map": label_map,
            "window_size": args.emosurv_window,
            "normalization": args.emosurv_normalization,
            "clip_z": args.clip_z,
            "min_subject_samples": args.min_subject_samples,
            "min_subject_classes": args.min_subject_classes,
        },
        verbose=not args.quiet,
    )

    plot_classification_bars(run.summary_df, plot_dir)
    plot_classification_boxplots(run.folds_df, plot_dir)
    plot_confusion_for_best(run.summary_df, run.y_true_by_model, run.y_pred_by_model, plot_dir)
    plot_classification_participant_heatmap(run.folds_df, plot_dir)

    print("\n[EmoSurv discrete] Completed")
    print(run.summary_df[["model", "accuracy_mean", "f1_macro_mean", "auc_ovr_mean"]].to_string(index=False))


def run_wesad_discrete(args: argparse.Namespace) -> None:
    _log("[Run] WESAD discrete task started", not args.quiet)
    discrete_df, feature_cols, _ = load_wesad_dataset(
        window_seconds=args.wesad_window_sec,
        stride_seconds=args.wesad_stride_sec,
        sample_rate_hz=args.wesad_sample_rate,
        quick=args.quick,
        clip_z=args.clip_z,
        normalization=args.wesad_normalization,
        verbose=not args.quiet,
    )

    discrete_df = filter_subjects_for_classification(
        discrete_df,
        subject_col="subject_id",
        target_col="target",
        min_samples=args.min_subject_samples,
        min_classes=args.min_subject_classes,
        verbose=not args.quiet,
        context="WESAD-discrete",
    )
    if discrete_df["subject_id"].nunique() < 2:
        raise RuntimeError("Not enough subjects after filtering for LOSO evaluation.")

    run = run_loso_classification(
        df=discrete_df,
        feature_cols=feature_cols,
        subject_col="subject_id",
        target_col="target",
        model_names=args.class_models,
        include_deep=args.include_deep,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
        deep_lr=args.deep_lr,
        classical_augment_mode=args.classical_augment,
        deep_augment_mode=args.deep_augment,
        seq_len=args.seq_len,
        verbose=not args.quiet,
    )

    plot_dir = _save_run_artifacts(
        dataset_name="wesad",
        task_name="discrete",
        summary_df=run.summary_df,
        folds_df=run.folds_df,
        metadata={
            "n_samples": int(len(discrete_df)),
            "n_subjects": int(discrete_df["subject_id"].nunique()),
            "n_features": int(len(feature_cols)),
            "labels_present": sorted(int(v) for v in discrete_df["target"].unique()),
            "window_seconds": args.wesad_window_sec,
            "stride_seconds": args.wesad_stride_sec,
            "sample_rate_hz": args.wesad_sample_rate,
            "normalization": args.wesad_normalization,
            "clip_z": args.clip_z,
            "min_subject_samples": args.min_subject_samples,
            "min_subject_classes": args.min_subject_classes,
        },
        verbose=not args.quiet,
    )

    plot_classification_bars(run.summary_df, plot_dir)
    plot_classification_boxplots(run.folds_df, plot_dir)
    plot_confusion_for_best(run.summary_df, run.y_true_by_model, run.y_pred_by_model, plot_dir)
    plot_classification_participant_heatmap(run.folds_df, plot_dir)

    print("\n[WESAD discrete] Completed")
    print(run.summary_df[["model", "accuracy_mean", "f1_macro_mean", "auc_ovr_mean"]].to_string(index=False))


def run_wesad_dimensional(args: argparse.Namespace) -> None:
    _log("[Run] WESAD dimensional task started", not args.quiet)
    _, feature_cols, dim_df = load_wesad_dataset(
        window_seconds=args.wesad_window_sec,
        stride_seconds=args.wesad_stride_sec,
        sample_rate_hz=args.wesad_sample_rate,
        quick=args.quick,
        clip_z=args.clip_z,
        normalization=args.wesad_normalization,
        verbose=not args.quiet,
    )

    if len(dim_df) == 0:
        raise RuntimeError("No WESAD dimensional samples found after questionnaire alignment.")

    dim_df = filter_subjects_for_regression(
        dim_df,
        subject_col="subject_id",
        min_samples=args.min_reg_subject_samples,
        verbose=not args.quiet,
        context="WESAD-dimensional",
    )
    if dim_df["subject_id"].nunique() < 2:
        raise RuntimeError("Not enough subjects after filtering for LOSO evaluation.")

    run = run_loso_regression(
        df=dim_df,
        feature_cols=feature_cols,
        subject_col="subject_id",
        target_cols=("valence", "arousal"),
        model_names=args.reg_models,
        include_deep=args.include_deep,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
        deep_lr=args.deep_lr,
        classical_augment_mode=args.regression_augment,
        deep_augment_mode=args.deep_regression_augment,
        seq_len=args.seq_len,
        verbose=not args.quiet,
    )

    plot_dir = _save_run_artifacts(
        dataset_name="wesad",
        task_name="dimensional",
        summary_df=run.summary_df,
        folds_df=run.folds_df,
        metadata={
            "n_samples": int(len(dim_df)),
            "n_subjects": int(dim_df["subject_id"].nunique()),
            "n_features": int(len(feature_cols)),
            "targets": ["valence", "arousal"],
            "window_seconds": args.wesad_window_sec,
            "stride_seconds": args.wesad_stride_sec,
            "sample_rate_hz": args.wesad_sample_rate,
            "normalization": args.wesad_normalization,
            "clip_z": args.clip_z,
            "min_reg_subject_samples": args.min_reg_subject_samples,
        },
        verbose=not args.quiet,
    )

    plot_regression_bars(run.summary_df, plot_dir)
    plot_regression_boxplots(run.folds_df, plot_dir)
    plot_regression_scatter_for_best(run.summary_df, run.y_true_by_model, run.y_pred_by_model, plot_dir)
    plot_regression_participant_heatmap(run.folds_df, plot_dir)

    print("\n[WESAD dimensional] Completed")
    print(run.summary_df[["model", "mae_mean", "rmse_mean", "r2_mean"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion modeling for EmoSurv and WESAD")
    p.add_argument("--dataset", choices=["emosurv", "wesad", "all"], default="all")
    p.add_argument("--task", choices=["discrete", "dimensional", "both"], default="both")
    p.add_argument("--quick", action="store_true", help="Use fewer subjects for smoke testing")
    p.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    p.add_argument("--include-deep", action="store_true", help="Include deep models in sweeps when using default model selection")
    p.add_argument(
        "--class-preset",
        choices=["all", "modern-a-catboost"],
        default="modern-a-catboost",
        help="Preset model family for classification when --class-models is not provided",
    )
    p.add_argument("--deep-epochs", type=int, default=DEFAULT_DEEP_EPOCHS)
    p.add_argument("--deep-batch-size", type=int, default=DEFAULT_DEEP_BATCH_SIZE)
    p.add_argument("--deep-lr", type=float, default=DEFAULT_DEEP_LR)
    p.add_argument(
        "--emosurv-normalization",
        choices=["zscore", "robust", "none"],
        default="robust",
        help="Normalization mode for EmoSurv features",
    )
    p.add_argument(
        "--wesad-normalization",
        choices=["zscore", "none"],
        default="zscore",
        help="Normalization mode for WESAD features",
    )
    p.add_argument(
        "--clip-z",
        type=float,
        default=None,
        help="Optional absolute clipping threshold after per-subject z-scoring",
    )
    p.add_argument(
        "--min-subject-samples",
        type=int,
        default=0,
        help="Drop classification subjects with fewer rows than this before LOSO",
    )
    p.add_argument(
        "--min-subject-classes",
        type=int,
        default=0,
        help="Drop classification subjects with fewer label classes than this",
    )
    p.add_argument(
        "--min-reg-subject-samples",
        type=int,
        default=0,
        help="Drop regression subjects with fewer rows than this before LOSO",
    )
    p.add_argument(
        "--classical-augment",
        choices=["none", "balance", "full"],
        default="none",
        help="Classification augmentation mode for non-deep sklearn models",
    )
    p.add_argument(
        "--deep-augment",
        choices=["none", "balance", "full"],
        default="balance",
        help="Classification augmentation mode for deep sequence classifiers",
    )
    p.add_argument(
        "--regression-augment",
        choices=["none", "full"],
        default="none",
        help="Regression augmentation mode for non-deep sklearn models",
    )
    p.add_argument(
        "--deep-regression-augment",
        choices=["none", "full"],
        default="full",
        help="Regression augmentation mode for deep Torch MLP regressor",
    )
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Token sequence length for transformer variants")

    p.add_argument("--emosurv-window", type=int, default=35, help="Keystrokes per EmoSurv window")
    p.add_argument("--wesad-window-sec", type=int, default=60)
    p.add_argument("--wesad-stride-sec", type=int, default=30)
    p.add_argument("--wesad-sample-rate", type=int, default=700)
    p.add_argument("--catboost-iterations", type=int, default=DEFAULT_CATBOOST_ITERATIONS)
    p.add_argument("--catboost-learning-rate", type=float, default=DEFAULT_CATBOOST_LEARNING_RATE)
    p.add_argument("--catboost-depth", type=int, default=DEFAULT_CATBOOST_DEPTH)

    p.add_argument(
        "--class-models",
        nargs="*",
        default=None,
        help=f"Subset of classification models: {list(CLASSIFIERS.keys()) + DEEP_CLASSIFIER_NAMES}",
    )
    p.add_argument(
        "--reg-models",
        nargs="*",
        default=None,
        help=f"Subset of regression models: {list(REGRESSORS.keys()) + DEEP_REGRESSOR_NAMES}",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    _configure_catboost_runtime(
        iterations=args.catboost_iterations,
        learning_rate=args.catboost_learning_rate,
        depth=args.catboost_depth,
        quick=args.quick,
    )

    _log(
        (
            f"Config: dataset={args.dataset} task={args.task} quick={args.quick} "
            f"include_deep={args.include_deep} cls_aug={args.classical_augment} "
            f"class_preset={args.class_preset} "
            f"emosurv_norm={args.emosurv_normalization} wesad_norm={args.wesad_normalization} "
            f"deep_aug={args.deep_augment} reg_aug={args.regression_augment} clip_z={args.clip_z} "
            f"min_samples={args.min_subject_samples} min_classes={args.min_subject_classes} "
            f"deep_reg_aug={args.deep_regression_augment} "
            f"catboost_iters={CATBOOST_CONFIG['iterations']} catboost_lr={CATBOOST_CONFIG['learning_rate']} catboost_depth={CATBOOST_CONFIG['depth']}"
        ),
        not args.quiet,
    )

    if args.class_models is None and args.class_preset == "modern-a-catboost":
        args.class_models = MODERN_A_CATBOOST_CLASS_MODELS.copy()
        args.include_deep = True

    if args.class_models is not None:
        allowed_class_models = set(CLASSIFIERS.keys()) | set(DEEP_CLASSIFIER_NAMES)
        unknown = [m for m in args.class_models if m not in allowed_class_models]
        if unknown:
            raise ValueError(f"Unknown classification models: {unknown}")

    if args.reg_models is not None:
        allowed_reg_models = set(REGRESSORS.keys()) | set(DEEP_REGRESSOR_NAMES)
        unknown = [m for m in args.reg_models if m not in allowed_reg_models]
        if unknown:
            raise ValueError(f"Unknown regression models: {unknown}")

    _ensure_dir(RESULTS_ROOT)

    if args.dataset in ["emosurv", "all"]:
        if args.task in ["discrete", "both"]:
            run_emosurv_discrete(args)
        if args.task == "dimensional":
            print("[EmoSurv] Dimensional task skipped (dataset does not expose valence/arousal targets).")

    if args.dataset in ["wesad", "all"]:
        if args.task in ["discrete", "both"]:
            run_wesad_discrete(args)
        if args.task in ["dimensional", "both"]:
            run_wesad_dimensional(args)

    print("\nAll requested runs finished.")


if __name__ == "__main__":
    main()
