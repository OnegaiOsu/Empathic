"""WESAD chest-sensor loader.

Schmidt et al. (2018) recorded 700 Hz signals from a RespiBAN chest device
during four conditions (baseline, TSST stress, amusement, meditation) plus
self-reported SAM questionnaires. For each subject we:

1. Read ``Sx.pkl`` and extract the chest signals (ECG, EDA, EMG, Resp, Temp,
   ACC in three axes).
2. Slide a 60 s window with 30 s stride (standard protocol) over the signal
   array.
3. Keep only windows whose majority label is in ``WESAD_KEEP_LABELS``.
4. Compute per-channel statistical descriptors (mean, std, min, max, median,
   peak-to-peak, first-difference std) as tabular features.
5. Downsample the same window to a fixed length for the deep model input.
6. Parse ``Sx_quest.csv`` to pick up SAM (valence/arousal) per stage for
   dimensional regression.
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

from ..config import (
    DataDefaults,
    WESAD_DIR,
    WESAD_KEEP_LABELS,
)
from ..utils import log, majority_label, safe_float, zscore_by_subject


CHEST_CHANNELS = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EDA", "EMG", "Resp", "Temp"]
META_COLS = {"subject_id", "target", "valence", "arousal", "stage", "window_start_sec"}


@dataclass
class WESADData:
    samples: pd.DataFrame        # tabular features + target/valence/arousal
    feature_cols: List[str]
    sequences: np.ndarray        # (N, L, C) float32 downsampled window tensor
    seq_channels: List[str]
    seq_length: int


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _extract_chest_channels(signal_dict) -> Dict[str, np.ndarray]:
    chest = signal_dict["chest"]
    channels: Dict[str, np.ndarray] = {}
    acc = np.asarray(chest["ACC"])
    if acc.ndim == 2 and acc.shape[1] >= 3:
        channels["ACC_x"], channels["ACC_y"], channels["ACC_z"] = acc[:, 0], acc[:, 1], acc[:, 2]
    else:
        flat = acc.reshape(-1)
        channels["ACC_x"] = flat
        channels["ACC_y"] = np.zeros_like(flat)
        channels["ACC_z"] = np.zeros_like(flat)
    for name in ("ECG", "EDA", "EMG", "Resp", "Temp"):
        channels[name] = np.asarray(chest[name]).reshape(-1)
    return channels


def _channel_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        keys = ("mean", "std", "min", "max", "median", "ptp", "diff_std")
        return {f"{prefix}_{k}": np.nan for k in keys}
    diffs = np.diff(values)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_ptp": float(np.ptp(values)),
        f"{prefix}_diff_std": float(np.std(diffs)) if diffs.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Physiology-specific feature extractors (HRV, EDA phasic/tonic, Resp, EMG).
# These follow Schmidt 2018 / Bobade 2020 feature definitions and use only
# scipy primitives so we do not pull in neurokit2 / cvxEDA.
# ---------------------------------------------------------------------------
def _ecg_rpeaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Pan-Tompkins style R-peak detector. Returns peak indices."""
    if ecg.size < int(0.5 * fs):
        return np.array([], dtype=int)
    nyq = 0.5 * fs
    try:
        b, a = sps.butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
        filtered = sps.filtfilt(b, a, ecg)
    except Exception:
        return np.array([], dtype=int)
    sq = filtered ** 2
    win = max(1, int(0.150 * fs))
    mav = np.convolve(sq, np.ones(win, dtype=np.float32) / win, mode="same")
    if not np.isfinite(mav).any():
        return np.array([], dtype=int)
    height = np.percentile(mav, 75)
    distance = max(1, int(0.20 * fs))  # >=200 ms -> max 300 BPM
    peaks, _ = sps.find_peaks(mav, distance=distance, height=height)
    return peaks


def _hrv_features(ecg: np.ndarray, fs: int, prefix: str = "ECG") -> Dict[str, float]:
    keys = ("hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "pnn20", "lf", "hf", "lfhf", "tp")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    peaks = _ecg_rpeaks(ecg, fs)
    if peaks.size < 4:
        return out
    rr = np.diff(peaks) / float(fs)  # seconds
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # physiological bounds (30-200 BPM)
    if rr.size < 3:
        return out
    hr = 60.0 / rr
    out[f"{prefix}_hr_mean"] = float(np.mean(hr))
    out[f"{prefix}_hr_std"] = float(np.std(hr))
    out[f"{prefix}_sdnn"] = float(np.std(rr) * 1000.0)
    drr = np.diff(rr)
    out[f"{prefix}_rmssd"] = float(np.sqrt(np.mean(drr ** 2)) * 1000.0) if drr.size else 0.0
    out[f"{prefix}_pnn50"] = float(np.mean(np.abs(drr) > 0.050)) if drr.size else 0.0
    out[f"{prefix}_pnn20"] = float(np.mean(np.abs(drr) > 0.020)) if drr.size else 0.0
    # Frequency-domain: cubic-interpolate RR onto 4 Hz grid then Welch
    if rr.size >= 8:
        try:
            t_rr = np.cumsum(rr)
            grid = np.arange(t_rr[0], t_rr[-1], 0.25)
            if grid.size >= 16:
                rr_resamp = np.interp(grid, t_rr, rr)
                rr_resamp = rr_resamp - rr_resamp.mean()
                nperseg = min(256, rr_resamp.size)
                f, psd = sps.welch(rr_resamp, fs=4.0, nperseg=nperseg)
                lf = float(np.trapz(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
                hf = float(np.trapz(psd[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]))
                tp = float(np.trapz(psd[(f >= 0.04) & (f < 0.4)], f[(f >= 0.04) & (f < 0.4)]))
                out[f"{prefix}_lf"] = lf
                out[f"{prefix}_hf"] = hf
                out[f"{prefix}_lfhf"] = lf / hf if hf > 1e-9 else 0.0
                out[f"{prefix}_tp"] = tp
        except Exception:
            pass
    return out


def _eda_features(eda: np.ndarray, fs: int, prefix: str = "EDA") -> Dict[str, float]:
    """Decompose EDA into tonic (SCL) / phasic (SCR) and summarise SCRs."""
    keys = ("scl_mean", "scl_slope", "scr_count", "scr_mean_amp", "scr_max_amp", "scr_sum_amp", "phasic_std")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    if eda.size < int(2 * fs):
        return out
    nyq = 0.5 * fs
    try:
        b, a = sps.butter(2, 0.05 / nyq, btype="low")
        tonic = sps.filtfilt(b, a, eda)
    except Exception:
        tonic = np.full_like(eda, eda.mean())
    phasic = eda - tonic
    out[f"{prefix}_scl_mean"] = float(np.mean(tonic))
    if tonic.size >= 2:
        # Slope across the window in microsiemens / second
        t = np.arange(tonic.size) / float(fs)
        slope = np.polyfit(t, tonic, 1)[0]
        out[f"{prefix}_scl_slope"] = float(slope)
    out[f"{prefix}_phasic_std"] = float(np.std(phasic))
    distance = max(1, int(1.0 * fs))  # SCRs >= 1 s apart
    height = max(0.01, 0.05 * float(np.std(phasic) + 1e-6))
    try:
        peaks, props = sps.find_peaks(phasic, distance=distance, prominence=height)
    except Exception:
        peaks, props = np.array([], dtype=int), {}
    if peaks.size:
        amps = props.get("prominences", np.array([]))
        out[f"{prefix}_scr_count"] = float(peaks.size)
        if amps.size:
            out[f"{prefix}_scr_mean_amp"] = float(np.mean(amps))
            out[f"{prefix}_scr_max_amp"] = float(np.max(amps))
            out[f"{prefix}_scr_sum_amp"] = float(np.sum(amps))
    return out


def _resp_features(resp: np.ndarray, fs: int, prefix: str = "Resp") -> Dict[str, float]:
    keys = ("rate", "ibi_mean", "ibi_std", "amp_mean", "amp_std")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    if resp.size < int(2 * fs):
        return out
    nyq = 0.5 * fs
    try:
        b, a = sps.butter(2, [0.1 / nyq, 0.5 / nyq], btype="band")
        filt = sps.filtfilt(b, a, resp)
    except Exception:
        filt = resp - resp.mean()
    distance = max(1, int(1.5 * fs))  # >= 1.5 s between breaths -> max 40 bpm
    height = float(np.std(filt))
    try:
        peaks, _ = sps.find_peaks(filt, distance=distance, prominence=height * 0.3)
    except Exception:
        peaks = np.array([], dtype=int)
    if peaks.size >= 2:
        ibi = np.diff(peaks) / float(fs)
        out[f"{prefix}_rate"] = float(60.0 / np.mean(ibi)) if ibi.size else 0.0
        out[f"{prefix}_ibi_mean"] = float(np.mean(ibi))
        out[f"{prefix}_ibi_std"] = float(np.std(ibi))
        amps = filt[peaks]
        out[f"{prefix}_amp_mean"] = float(np.mean(amps))
        out[f"{prefix}_amp_std"] = float(np.std(amps))
    return out


def _emg_features(emg: np.ndarray, prefix: str = "EMG") -> Dict[str, float]:
    keys = ("mav", "rms", "zcr", "wamp", "peak_count")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    if emg.size < 4:
        return out
    centred = emg - np.mean(emg)
    out[f"{prefix}_mav"] = float(np.mean(np.abs(centred)))
    out[f"{prefix}_rms"] = float(np.sqrt(np.mean(centred ** 2)))
    signs = np.sign(centred)
    out[f"{prefix}_zcr"] = float(np.mean(np.diff(signs) != 0))
    thresh = 0.1 * np.std(centred)
    out[f"{prefix}_wamp"] = float(np.sum(np.abs(np.diff(centred)) > thresh))
    try:
        peaks, _ = sps.find_peaks(np.abs(centred), prominence=2 * np.std(centred) + 1e-6)
        out[f"{prefix}_peak_count"] = float(peaks.size)
    except Exception:
        pass
    return out


def _acc_features(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray, fs: int, prefix: str = "ACC") -> Dict[str, float]:
    keys = ("sma", "mag_mean", "mag_std", "peak_freq")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    if acc_x.size == 0:
        return out
    out[f"{prefix}_sma"] = float(np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)))
    mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    out[f"{prefix}_mag_mean"] = float(np.mean(mag))
    out[f"{prefix}_mag_std"] = float(np.std(mag))
    if mag.size >= 32:
        try:
            mag_c = mag - mag.mean()
            f, psd = sps.welch(mag_c, fs=fs, nperseg=min(512, mag_c.size))
            if psd.size and np.isfinite(psd).any():
                out[f"{prefix}_peak_freq"] = float(f[int(np.argmax(psd))])
        except Exception:
            pass
    return out


def _temp_features(temp: np.ndarray, fs: int, prefix: str = "Temp") -> Dict[str, float]:
    out = {f"{prefix}_slope": 0.0, f"{prefix}_range": 0.0}
    if temp.size >= 2:
        t = np.arange(temp.size) / float(fs)
        try:
            out[f"{prefix}_slope"] = float(np.polyfit(t, temp, 1)[0])
        except Exception:
            pass
        out[f"{prefix}_range"] = float(np.ptp(temp))
    return out


def _downsample(values: np.ndarray, target_len: int) -> np.ndarray:
    """Average-pool a 1-D array down to ``target_len`` samples.

    We average instead of naive subsampling so that we keep information about
    the full window instead of picking out ~240 individual samples.
    """
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    pool = values.size // target_len
    if pool <= 1:
        # Fallback: linear interpolation for short windows.
        x_old = np.linspace(0.0, 1.0, num=values.size)
        x_new = np.linspace(0.0, 1.0, num=target_len)
        return np.interp(x_new, x_old, values).astype(np.float32)
    trimmed = values[: pool * target_len]
    return trimmed.reshape(target_len, pool).mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Questionnaire parsing
# ---------------------------------------------------------------------------
def _mmss_to_seconds(value: str) -> Optional[float]:
    """Convert strings like '39.55' (39 minutes, 55 seconds) to seconds."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if "." not in s:
        v = safe_float(s)
        return v if np.isfinite(v) else None
    minutes_str, _, seconds_str = s.partition(".")
    try:
        return float(int(minutes_str) * 60 + int(seconds_str))
    except ValueError:
        v = safe_float(s)
        return v if np.isfinite(v) else None


def _parse_questionnaire(path: str):
    order: List[str] = []
    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    dim_pairs: List[Tuple[float, float]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
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
                val = safe_float(vals[0])
                ar = safe_float(vals[1])
                if np.isfinite(val) and np.isfinite(ar):
                    dim_pairs.append((float(val), float(ar)))

    stage_intervals: List[Tuple[str, float, float]] = []
    for stage, start, end in zip(order, starts, ends):
        if start is None or end is None:
            continue
        if end > start:
            stage_intervals.append((stage, float(start), float(end)))

    stage_to_dim: Dict[str, Tuple[float, float]] = {}
    for idx, pair in enumerate(dim_pairs):
        if idx < len(stage_intervals):
            stage_to_dim[stage_intervals[idx][0]] = pair

    return stage_intervals, stage_to_dim


def _find_stage(intervals: List[Tuple[str, float, float]], t_sec: float) -> Optional[str]:
    for name, start, end in intervals:
        if start <= t_sec <= end:
            return name
    return None


def _condition_dims(stage_to_dim: Dict[str, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    def find(substr: str) -> Optional[Tuple[float, float]]:
        for stage, pair in stage_to_dim.items():
            if substr.lower() in stage.lower():
                return pair
        return None

    out: Dict[int, Tuple[float, float]] = {}
    base = find("Base")
    tsst = find("TSST")
    fun = find("Fun")
    medi_pairs = [p for s, p in stage_to_dim.items() if "medi" in s.lower()]
    medi = (float(np.mean([p[0] for p in medi_pairs])), float(np.mean([p[1] for p in medi_pairs]))) if medi_pairs else None
    if base: out[1] = base
    if tsst: out[2] = tsst
    if fun:  out[3] = fun
    if medi: out[4] = medi
    return out


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def _zscore_sequences_by_subject(
    seqs: np.ndarray,
    subject_ids: np.ndarray,
    clip_z: Optional[float] = None,
) -> np.ndarray:
    """Per-subject, per-channel z-score for (N, L, C) sequences.

    Sensor scales differ by orders of magnitude (Temp ~34 vs ECG ~1e-3) and
    per-subject baselines (e.g. EDA, skin temperature) drift far more than
    condition-induced deltas. Without this, the first conv layer of any deep
    model is dominated by Temp and the LOSO subject shift is baked into the
    raw input. Robust to outlier glitches via optional clipping.
    """
    if seqs.size == 0:
        return seqs
    out = seqs.astype(np.float32, copy=True)
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        block = out[mask]                              # (n_s, L, C)
        mean = block.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
        std = block.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        z = (block - mean) / std
        if clip_z is not None and clip_z > 0:
            z = np.clip(z, -clip_z, clip_z)
        out[mask] = z.astype(np.float32)
    return out


def load_wesad(
    window_seconds: Optional[int] = None,
    stride_seconds: Optional[int] = None,
    sample_rate_hz: Optional[int] = None,
    seq_length: int = 240,
    quick: bool = False,
    normalization: str = "zscore",
    clip_z: Optional[float] = None,
    baseline_correct: bool = True,
    verbose: bool = True,
) -> WESADData:
    cfg = DataDefaults()
    window_seconds = window_seconds or cfg.wesad_window_seconds
    stride_seconds = stride_seconds or cfg.wesad_stride_seconds
    sample_rate_hz = sample_rate_hz or cfg.wesad_sample_rate
    clip_z = clip_z if clip_z is not None else cfg.clip_z

    subjects = sorted(glob.glob(os.path.join(WESAD_DIR, "S*")))
    subjects = [s for s in subjects if os.path.isdir(s)]
    if quick:
        subjects = subjects[:6]

    win = window_seconds * sample_rate_hz
    stride = stride_seconds * sample_rate_hz
    log(f"[WESAD] subjects={len(subjects)}  window={window_seconds}s stride={stride_seconds}s @ {sample_rate_hz}Hz", verbose)

    rows: List[Dict[str, float]] = []
    seq_list: List[np.ndarray] = []

    for sdir in subjects:
        sid = os.path.basename(sdir)
        pkl_path = os.path.join(sdir, f"{sid}.pkl")
        quest_path = os.path.join(sdir, f"{sid}_quest.csv")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        labels = np.asarray(data["label"]).reshape(-1)
        signals = _extract_chest_channels(data["signal"])

        stage_intervals: List[Tuple[str, float, float]] = []
        stage_to_dim: Dict[str, Tuple[float, float]] = {}
        if os.path.exists(quest_path):
            stage_intervals, stage_to_dim = _parse_questionnaire(quest_path)
        cond_to_dim = _condition_dims(stage_to_dim)

        n = len(labels)
        n_windows = 0
        for start in range(0, max(n - win + 1, 0), stride):
            end = start + win
            y_win = labels[start:end]
            maj = majority_label(y_win)
            if maj not in WESAD_KEEP_LABELS:
                continue

            sample: Dict[str, float] = {
                "subject_id": sid,
                "target": int(maj),
                "window_start_sec": float(start / sample_rate_hz),
            }
            seq = np.zeros((seq_length, len(CHEST_CHANNELS)), dtype=np.float32)
            window_signals: Dict[str, np.ndarray] = {}
            for ci, ch_name in enumerate(CHEST_CHANNELS):
                ch = signals.get(ch_name)
                if ch is None or len(ch) < end:
                    continue
                window_vals = ch[start:end]
                window_signals[ch_name] = window_vals
                sample.update(_channel_stats(window_vals, ch_name))
                seq[:, ci] = _downsample(window_vals, seq_length)

            # Domain-specific physiology features (Schmidt 2018 / Bobade 2020).
            if "ECG" in window_signals:
                sample.update(_hrv_features(window_signals["ECG"], sample_rate_hz))
            if "EDA" in window_signals:
                sample.update(_eda_features(window_signals["EDA"], sample_rate_hz))
            if "Resp" in window_signals:
                sample.update(_resp_features(window_signals["Resp"], sample_rate_hz))
            if "EMG" in window_signals:
                sample.update(_emg_features(window_signals["EMG"]))
            if "Temp" in window_signals:
                sample.update(_temp_features(window_signals["Temp"], sample_rate_hz))
            if all(k in window_signals for k in ("ACC_x", "ACC_y", "ACC_z")):
                sample.update(_acc_features(
                    window_signals["ACC_x"],
                    window_signals["ACC_y"],
                    window_signals["ACC_z"],
                    sample_rate_hz,
                ))

            center_sec = float((start + end) / 2.0 / sample_rate_hz)
            stage_name = _find_stage(stage_intervals, center_sec)
            sample["stage"] = stage_name or "unknown"

            dim_pair = stage_to_dim.get(stage_name) if stage_name else None
            if dim_pair is None:
                dim_pair = cond_to_dim.get(int(maj))
            sample["valence"] = float(dim_pair[0]) if dim_pair else np.nan
            sample["arousal"] = float(dim_pair[1]) if dim_pair else np.nan

            rows.append(sample)
            seq_list.append(seq)
            n_windows += 1

        log(f"[WESAD] {sid}: windows={n_windows}  intervals={len(stage_intervals)}", verbose)

    if not rows:
        raise RuntimeError("No WESAD windows produced; check dataset path.")
    df = pd.DataFrame(rows)
    sequences = np.stack(seq_list)

    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Per-subject baseline-condition correction (Schmidt 2018, Can et al. 2019):
    # subtract each subject's mean of features measured during the baseline
    # condition (label==1, mapped to "HVLA" quadrant) from all of that
    # subject's rows. This removes person-specific resting-state offsets
    # before any z-score, producing condition-relative deviations rather
    # than absolute values that are dominated by individual physiology.
    n_baseline_corrected = 0
    if baseline_correct:
        for sid, sub_df in df.groupby("subject_id"):
            base_rows = sub_df[sub_df["target"] == 1]
            if base_rows.empty:
                continue
            base_mean = base_rows[feature_cols].mean(axis=0)
            mask = df["subject_id"] == sid
            df.loc[mask, feature_cols] = (
                df.loc[mask, feature_cols].subtract(base_mean, axis=1)
            )
            n_baseline_corrected += int(mask.sum())

        # Sequence baseline correction: subtract per-subject mean baseline
        # window (target==1) from all of that subject's raw sequence windows
        # per channel. This is the sequence analogue of the tabular fix and
        # gives deep models condition-relative dynamics rather than absolute
        # subject-level offsets.
        subj_arr = df["subject_id"].to_numpy()
        tgt_arr = df["target"].to_numpy()
        for sid in np.unique(subj_arr):
            sid_mask = subj_arr == sid
            base_mask = sid_mask & (tgt_arr == 1)
            if not base_mask.any():
                continue
            # Mean over (n_baseline_windows, L) per channel -> (1, 1, C)
            base_mean = sequences[base_mask].mean(axis=(0, 1), keepdims=True)
            sequences[sid_mask] = sequences[sid_mask] - base_mean

    norm = normalization.lower().strip()
    if norm == "zscore":
        df = zscore_by_subject(df, feature_cols, "subject_id", clip_z=clip_z)
        sequences = _zscore_sequences_by_subject(
            sequences,
            np.asarray(df["subject_id"].values),
            clip_z=clip_z,
        )
    elif norm == "none":
        pass
    else:
        raise ValueError(f"unknown WESAD normalization: {normalization}")

    log(
        f"[WESAD] samples={len(df)}  features={len(feature_cols)}  subjects={df['subject_id'].nunique()}  "
        f"dim_with_va={int(df[['valence','arousal']].notna().all(axis=1).sum())}  "
        f"seq_norm={norm}  baseline_corrected={n_baseline_corrected}",
        verbose,
    )
    return WESADData(
        samples=df.reset_index(drop=True),
        feature_cols=feature_cols,
        sequences=sequences.astype(np.float32),
        seq_channels=list(CHEST_CHANNELS),
        seq_length=seq_length,
    )
