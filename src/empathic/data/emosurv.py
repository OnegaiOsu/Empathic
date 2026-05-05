"""EmoSurv keystroke loader and feature engineering.

EmoSurv (Yang & Qin, 2021) contains two keystroke corpora collected while
participants self-reported one of five emotions (N, H, C, A, S):

* Fixed Text Typing Dataset   -- same prompt for every participant.
* Free Text Typing Dataset    -- free-form responses per session.

For each user/emotion/session we cut the event stream into overlapping windows
of ``W`` consecutive key events and compute statistical descriptors over the
common keystroke intervals (key hold, down-down, up-down, ...) together with
categorical rate features (backspace, space, alphabetic...). This yields
fixed-length tabular rows for classical models and a 1-D sequence representation
for the deep model.

The raw CSVs contain artefacts (scientific-notation timestamps that overflow to
~1.58e12 and negative intervals). We sanitise them here to keep the downstream
pipeline honest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import EMOSURV_DIR, DataDefaults
from ..utils import log, robust_scale_by_subject, safe_float, zscore_by_subject


EVENT_NUMERIC_COLS = [
    "index", "keyDown", "keyUp",
    "D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3",
]
INTERVAL_COLS = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]
SENTINEL_ABS = 1e10
INTERVAL_MIN_MS = -5_000.0
INTERVAL_MAX_MS = 120_000.0
HOLD_MAX_MS = 10_000.0

# Raw per-event columns used to build the deep model's sequence tensor.
SEQUENCE_CHANNELS = ["hold_ms", "D1D2", "U1D2", "U1U2", "D1U3"]

META_COLS = {"subject_id", "split", "emotion", "session_id", "window_start"}


@dataclass
class EmoSurvData:
    """Container for tabular features + aligned raw sequences."""
    samples: pd.DataFrame            # one row per window (tabular features)
    feature_cols: List[str]          # tabular feature column names
    sequences: np.ndarray            # (N, W, C) float32 sequences
    seq_channels: List[str]          # channel names for ``sequences``
    seq_length: int                  # W
    emotion_labels: List[str]        # ordered label vocabulary


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------
def _sanitize_timing(df: pd.DataFrame) -> int:
    dropped = 0
    for col in INTERVAL_COLS:
        if col not in df.columns:
            continue
        bad = (df[col].abs() > SENTINEL_ABS) | (df[col] < INTERVAL_MIN_MS) | (df[col] > INTERVAL_MAX_MS)
        dropped += int(bad.sum())
        df.loc[bad, col] = np.nan
    return dropped


def _read_events(path: str, split_name: str) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(path, sep=";", dtype=str)
    if "userId" in df.columns:
        df["user_id"] = df["userId"]
    elif "userid" in df.columns:
        df["user_id"] = df["userid"]
    else:
        raise ValueError(f"No user id column in {path}")

    df["split"] = split_name
    for col in EVENT_NUMERIC_COLS:
        df[col] = df[col].map(safe_float) if col in df.columns else np.nan

    dropped = _sanitize_timing(df)

    if "keyCode" not in df.columns:
        df["keyCode"] = ""
    df["session_id"] = df["_id"].astype(str) if "_id" in df.columns else f"{split_name}_session"

    df["hold_ms"] = df["keyUp"] - df["keyDown"]
    bad_hold = (df["hold_ms"] < 0) | (df["hold_ms"] > HOLD_MAX_MS)
    dropped += int(bad_hold.sum())
    df.loc[bad_hold, "hold_ms"] = np.nan

    df = df.dropna(subset=["user_id", "emotionIndex"]).copy()
    return df, dropped


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def _window_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_{s}": np.nan for s in ("mean", "std", "median", "q25", "q75")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.percentile(values, 25)),
        f"{prefix}_q75": float(np.percentile(values, 75)),
    }


def _key_category_rates(chunk: pd.DataFrame) -> Dict[str, float]:
    key = chunk["keyCode"].fillna("").astype(str)
    n = max(len(chunk), 1)
    backspace = key.isin(["\\b", "Backspace", "8"]).sum()
    space = key.isin([" ", "Space", "32"]).sum()
    alpha = key.str.match(r"^[A-Za-z]$", na=False).sum()
    digit = key.str.match(r"^[0-9]$", na=False).sum()
    special = (~key.str.match(r"^[A-Za-z0-9 ]$", na=False)).sum()
    return {
        "backspace_rate": float(backspace) / n,
        "space_rate": float(space) / n,
        "alpha_key_rate": float(alpha) / n,
        "digit_key_rate": float(digit) / n,
        "special_key_rate": float(special) / n,
        "unique_key_ratio": float(key.nunique()) / n,
    }


def _rhythm_features(chunk: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in ("D1D2", "hold_ms", "U1D2"):
        vals = chunk[col].dropna().to_numpy()
        if vals.size == 0:
            out[f"{col}_cv"] = np.nan
            out[f"{col}_iqr"] = np.nan
            out[f"{col}_jump_std"] = np.nan
        else:
            mean_abs = float(np.mean(np.abs(vals)))
            out[f"{col}_cv"] = float(np.std(vals) / max(mean_abs, 1e-6))
            out[f"{col}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            out[f"{col}_jump_std"] = float(np.std(np.diff(vals))) if vals.size >= 2 else np.nan
    u1d2 = chunk["U1D2"].dropna().to_numpy()
    out["pause_rate_300ms"] = float(np.mean(u1d2 > 300.0)) if u1d2.size else np.nan
    out["negative_gap_rate"] = float(np.mean(u1d2 < 0.0)) if u1d2.size else np.nan
    return out


def _build_windows(
    events: pd.DataFrame,
    window_size: int,
    stride: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return a tabular feature dataframe and a (N, window_size, C) tensor."""
    rows: List[Dict[str, float]] = []
    sequences: List[np.ndarray] = []
    timing_cols = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3", "hold_ms"]

    groupby_cols = ["user_id", "split", "emotionIndex", "session_id"]
    for (user_id, split_name, emotion, session_id), grp in events.groupby(groupby_cols):
        grp = grp.sort_values("index")
        n = len(grp)
        if n < 8:
            continue

        for start in range(0, max(n - window_size + 1, 1), stride):
            chunk = grp.iloc[start:start + window_size]
            if len(chunk) < 8:
                continue

            sample: Dict[str, float] = {
                "subject_id": str(user_id),
                "split": split_name,
                "emotion": str(emotion),
                "session_id": str(session_id),
                "window_start": float(start),
                "n_keys": float(len(chunk)),
            }
            sample.update(_key_category_rates(chunk))

            first_down = np.nanmin(chunk["keyDown"].to_numpy())
            last_down = np.nanmax(chunk["keyDown"].to_numpy())
            if np.isfinite(first_down) and np.isfinite(last_down) and last_down > first_down:
                sample["keys_per_sec"] = float(len(chunk)) / ((last_down - first_down) / 1000.0)
            else:
                sample["keys_per_sec"] = np.nan

            for col in timing_cols:
                sample.update(_window_stats(chunk[col].dropna().to_numpy(), col))
            sample.update(_rhythm_features(chunk))
            rows.append(sample)

            # Build a (window_size, C) sequence tensor for the deep model.
            seq = np.zeros((window_size, len(SEQUENCE_CHANNELS)), dtype=np.float32)
            for ci, ch in enumerate(SEQUENCE_CHANNELS):
                vals = chunk[ch].to_numpy(dtype=np.float32)
                # Pad short chunks with the column mean to preserve scale.
                if len(vals) < window_size:
                    pad = window_size - len(vals)
                    fill = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else 0.0
                    vals = np.concatenate([vals, np.full(pad, fill, dtype=np.float32)])
                vals = vals[:window_size]
                vals[~np.isfinite(vals)] = 0.0
                seq[:, ci] = vals
            sequences.append(seq)

    df = pd.DataFrame(rows)
    seq_tensor = np.stack(sequences) if sequences else np.zeros((0, window_size, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    return df, seq_tensor


def _load_frequency_features() -> pd.DataFrame:
    path = os.path.join(EMOSURV_DIR, "Frequency Dataset.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["subject_id", "split"])
    freq = pd.read_csv(path, sep=";", dtype=str)
    freq["subject_id"] = freq.get("User ID", "").astype(str)
    if "textIndex" in freq.columns:
        idx = freq["textIndex"].astype(str).str.upper()
        freq["split"] = np.where(
            idx.str.startswith("FI"), "fixed",
            np.where(idx.str.startswith("FR"), "free", "unknown"),
        )
    else:
        freq["split"] = "unknown"
    for col in ("delFreq", "leftFreq", "TotTime"):
        freq[col] = freq[col].map(safe_float) if col in freq.columns else np.nan
    freq["tot_time_sec"] = freq["TotTime"] / 1000.0
    ok = freq["tot_time_sec"] > 0
    freq["del_per_sec"] = np.where(ok, freq["delFreq"] / freq["tot_time_sec"], np.nan)
    freq["left_per_sec"] = np.where(ok, freq["leftFreq"] / freq["tot_time_sec"], np.nan)
    return (
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


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_emosurv(
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    quick: bool = False,
    normalization: str = "robust",
    clip_z: Optional[float] = None,
    neutral_policy: str = "merge",
    verbose: bool = True,
) -> EmoSurvData:
    window_size = window_size or DataDefaults().emosurv_window_keys
    stride = stride or DataDefaults().emosurv_window_stride
    clip_z = clip_z if clip_z is not None else DataDefaults().clip_z

    fixed_path = os.path.join(EMOSURV_DIR, "Fixed Text Typing Dataset.csv")
    free_path = os.path.join(EMOSURV_DIR, "Free Text Typing Dataset.csv")

    log("[EmoSurv] reading fixed and free typing CSVs", verbose)
    fixed, f_bad = _read_events(fixed_path, "fixed")
    free, r_bad = _read_events(free_path, "free")
    log(f"[EmoSurv] rows: fixed={len(fixed)}  free={len(free)}  dropped_timing_values={f_bad + r_bad}", verbose)
    events = pd.concat([fixed, free], ignore_index=True)

    if quick:
        keep = sorted(events["user_id"].astype(str).unique())[:10]
        events = events[events["user_id"].astype(str).isin(keep)].reset_index(drop=True)
        log(f"[EmoSurv] quick-mode events retained: {len(events)} (subjects={len(keep)})", verbose)

    policy = neutral_policy.lower().strip()
    if policy not in {"merge", "drop", "separate", "baseline"}:
        raise ValueError(f"unknown neutral_policy: {neutral_policy}")
    if policy == "drop":
        before = len(events)
        events = events[events["emotionIndex"].astype(str) != "N"].reset_index(drop=True)
        log(f"[EmoSurv] neutral_policy=drop: removed {before - len(events)} Neutral events", verbose)
    # For "baseline" policy we keep Neutral events through windowing; they are
    # consumed as per-subject calibration below and then dropped from the output.

    log(f"[EmoSurv] windowing (size={window_size}, stride={stride})", verbose)
    df, sequences = _build_windows(events, window_size, stride)

    freq = _load_frequency_features()
    if not freq.empty:
        df = df.merge(freq, on=["subject_id", "split"], how="left")

    if df.empty:
        raise RuntimeError("EmoSurv windowing produced no samples; check CSV format.")

    labels = sorted(df["emotion"].dropna().unique().tolist())
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # ----- Baseline-residual policy: subtract per-subject Neutral centroid ---
    if policy == "baseline":
        is_neutral = (df["emotion"].astype(str) == "N").to_numpy()
        subj = df["subject_id"].astype(str).to_numpy()
        n_subjects_with_neutral = 0
        n_subjects_total = int(pd.Series(subj).nunique())
        # Tabular residuals.
        feat_mat = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        for sid in np.unique(subj):
            mask_s = subj == sid
            mask_n = mask_s & is_neutral
            if mask_n.sum() == 0:
                continue
            n_subjects_with_neutral += 1
            centre = np.nanmedian(feat_mat[mask_n], axis=0)
            centre = np.where(np.isfinite(centre), centre, 0.0)
            feat_mat[mask_s] = feat_mat[mask_s] - centre
        df.loc[:, feature_cols] = feat_mat
        # Sequence residuals: subtract per-subject per-channel Neutral mean.
        if sequences.shape[0] == len(df):
            for sid in np.unique(subj):
                mask_s = subj == sid
                mask_n = mask_s & is_neutral
                if mask_n.sum() == 0:
                    continue
                # centre over Neutral windows: (C,) mean across time & windows
                seq_n = sequences[mask_n]
                centre_seq = np.nanmean(seq_n.reshape(-1, seq_n.shape[-1]), axis=0)
                centre_seq = np.where(np.isfinite(centre_seq), centre_seq, 0.0).astype(np.float32)
                sequences[mask_s] = sequences[mask_s] - centre_seq
        log(
            f"[EmoSurv] neutral_policy=baseline: centred {n_subjects_with_neutral}/{n_subjects_total} "
            f"subjects; dropping Neutral rows after calibration",
            verbose,
        )
        keep = ~is_neutral
        df = df.loc[keep].reset_index(drop=True)
        sequences = sequences[keep]
        labels = sorted(df["emotion"].dropna().unique().tolist())

    norm = normalization.lower().strip()
    if policy == "baseline" and norm != "none":
        # Per-subject centring has already been done against Neutral; a second
        # per-subject scaler would wipe the residual we care about. Apply only
        # a light global robust rescale via the per-subject IQR (no re-centring).
        if norm == "robust":
            df = robust_scale_by_subject(df, feature_cols, "subject_id", clip_val=clip_z)
        elif norm == "zscore":
            df = zscore_by_subject(df, feature_cols, "subject_id", clip_z=clip_z)
        else:
            raise ValueError(f"unknown normalization: {normalization}")
    elif norm == "zscore":
        df = zscore_by_subject(df, feature_cols, "subject_id", clip_z=clip_z)
    elif norm == "robust":
        df = robust_scale_by_subject(df, feature_cols, "subject_id", clip_val=clip_z)
    elif norm == "none":
        pass
    else:
        raise ValueError(f"unknown normalization: {normalization}")

    log(f"[EmoSurv] windows={len(df)}  features={len(feature_cols)}  subjects={df['subject_id'].nunique()}", verbose)
    return EmoSurvData(
        samples=df.reset_index(drop=True),
        feature_cols=feature_cols,
        sequences=sequences.astype(np.float32),
        seq_channels=list(SEQUENCE_CHANNELS),
        seq_length=window_size,
        emotion_labels=labels,
    )
