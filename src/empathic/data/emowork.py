"""EmoWork V2 multimodal sensor loader.

Lee et al. (2026, "EmoWork") collected 31 participants (P1..P31) performing
six ~4-minute sessions in a simulated Korean call-center workplace:

* ``b1``, ``b2``, ``b3`` — rest/break periods between calls.
* ``c1``, ``c2``, ``c3`` — customer-service phone calls with a scripted actor
  applying mild, moderate, and severe complaint pressure respectively.

Each session is recorded with multiple wrist-/head-form-factor devices that
sample at very different rates, plus continuous self-reported labels at
~10 Hz on a Likert scale.

Per the user's instructions we pool all six sessions per subject (continuous
labels drive the supervised signal; b-vs-c is treated as just another
context source) and predict four targets:

* ``arousal`` (1..9, threshold at 5)
* ``valence`` (1..9, threshold at 5)
* ``stress``  (1..20, threshold at 10)
* ``quadrant`` (Russell, derived from binary valence+arousal)

Pipeline mirrors :mod:`empathic.data.wesad`:

1. For each ``(subject, session)`` we resample every sensor onto a common
   32 Hz grid spanning the overlap of all sensor timestamp ranges.
2. Slide a 60 s / 30 s stride window over the grid.
3. For each window compute (a) tabular descriptors (HRV from ECG/BVP, EDA
   tonic/phasic, accelerometer activity, EEG band powers per channel, etc.)
   and (b) a downsampled (240-sample) sequence tensor for the deep models.
4. Continuous labels are averaged within the window and discretised at the
   threshold above.
5. Galaxy ``galaxy_ppg.csv`` is intentionally **dropped** — its values are
   uniformly subnormal float32 noise (~2.94e-39) and carry no information.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

from ..config import EMOWORK_DIR, PROJECT_ROOT
from ..utils import log, zscore_by_subject


# Cache directory for serialised pre-normalisation feature/sequence tensors
# so the (slow) ~1-2 min feature extraction does not re-run for every target
# / protocol / debugging cycle. Cache key is a hash of the extraction kwargs;
# normalisation and baseline correction happen *after* loading from cache so
# they can be tweaked without invalidating the heavy work. Prefer the SSD
# location when EMOWORK_DIR points to one; otherwise fall back to the repo.
_CACHE_DIR_SSD = r"C:\dev\datasets\.cache\emowork"
_CACHE_DIR_REPO = os.path.join(PROJECT_ROOT, ".cache", "emowork")
_CACHE_DIR = _CACHE_DIR_SSD if EMOWORK_DIR.lower().startswith(r"c:\dev\datasets") else _CACHE_DIR_REPO


# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------
SEQ_CHANNELS: List[str] = [
    "ECG",        # polar_ecg @ ~130 Hz
    "BVP",        # e4_bvp @ 64 Hz
    "EDA",        # e4_eda @ 4 Hz
    "TEMP",       # e4_temp @ 4 Hz
    "ACC_x",      # e4_acc @ 32 Hz
    "ACC_y",
    "ACC_z",
    "HR",         # polar_hr @ 1 Hz (upsampled)
    "EEG_TP9",    # muse @ 256 Hz
    "EEG_AF7",
    "EEG_AF8",
    "EEG_TP10",
]
COMMON_FS = 32       # Hz, target grid
SEQ_LENGTH = 240     # downsampled length for deep models (matches WESAD)

META_COLS = {
    "subject_id", "session", "window_start_sec",
    "arousal", "valence", "stress", "suppression",
    "arousal_cont", "valence_cont", "stress_cont", "suppression_cont",
    "quadrant", "quadrant_target", "native_target",
    "target", "stage",
}


@dataclass
class EmoWorkData:
    samples: pd.DataFrame
    feature_cols: List[str]
    sequences: np.ndarray
    seq_channels: List[str]
    seq_length: int
    # Per-subject rest-session (b1/b2/b3) windows surfaced for downstream
    # calibration experiments. Same feature columns as `samples`; sequences
    # share `seq_channels` and `seq_length`. None if no b-session windows
    # were produced.
    baseline_samples: Optional[pd.DataFrame] = None
    baseline_sequences: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _safe_read_csv(path: str, cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return None
    return df


def _resample_to_grid(ts_ms: np.ndarray, vals: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    """Linear-interpolate ``vals(ts_ms)`` onto ``grid_ms``.

    Both arrays are 1-D; values outside the source range are held at the
    nearest endpoint (``np.interp`` default).
    """
    if ts_ms.size == 0 or vals.size == 0:
        return np.full(grid_ms.size, np.nan, dtype=np.float32)
    order = np.argsort(ts_ms)
    ts_ms = ts_ms[order]
    vals = vals[order]
    # de-duplicate timestamps (np.interp requires strictly increasing xp)
    keep = np.concatenate([[True], np.diff(ts_ms) > 0])
    ts_ms = ts_ms[keep]
    vals = vals[keep]
    return np.interp(grid_ms, ts_ms, vals).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extractors (adapted from wesad.py for EmoWork sample rates)
# ---------------------------------------------------------------------------
def _ecg_rpeaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    if ecg.size < int(0.5 * fs):
        return np.array([], dtype=int)
    nyq = 0.5 * fs
    try:
        b, a = sps.butter(2, [5.0 / nyq, min(15.0, nyq * 0.95) / nyq], btype="band")
        filtered = sps.filtfilt(b, a, ecg)
    except Exception:
        return np.array([], dtype=int)
    sq = filtered ** 2
    win = max(1, int(0.150 * fs))
    mav = np.convolve(sq, np.ones(win, dtype=np.float32) / win, mode="same")
    if not np.isfinite(mav).any():
        return np.array([], dtype=int)
    height = np.percentile(mav, 75)
    distance = max(1, int(0.20 * fs))
    peaks, _ = sps.find_peaks(mav, distance=distance, height=height)
    return peaks


def _hrv_from_peaks(rr_sec: np.ndarray, prefix: str) -> Dict[str, float]:
    """Time- and frequency-domain HRV features from inter-beat intervals.

    Spectral features (LF/HF/TP) are reported as ``log1p`` because raw band
    powers span several orders of magnitude across subjects.
    """
    keys = ("hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "pnn20", "lf", "hf", "lfhf", "tp")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    rr = rr_sec[(rr_sec > 0.3) & (rr_sec < 2.0)]
    if rr.size < 3:
        return out
    hr = 60.0 / rr
    hr_mean = float(np.mean(hr))
    # Sanity: implausible HR (extreme tachy/brady) -> treat as artifact and
    # zero out HRV. 30-200 BPM is the conventional non-pathological range.
    if hr_mean < 30.0 or hr_mean > 200.0:
        return out
    out[f"{prefix}_hr_mean"] = hr_mean
    out[f"{prefix}_hr_std"] = float(np.std(hr))
    out[f"{prefix}_sdnn"] = float(np.std(rr) * 1000.0)
    drr = np.diff(rr)
    out[f"{prefix}_rmssd"] = float(np.sqrt(np.mean(drr ** 2)) * 1000.0) if drr.size else 0.0
    out[f"{prefix}_pnn50"] = float(np.mean(np.abs(drr) > 0.050)) if drr.size else 0.0
    out[f"{prefix}_pnn20"] = float(np.mean(np.abs(drr) > 0.020)) if drr.size else 0.0
    if rr.size >= 8:
        try:
            t_rr = np.cumsum(rr)
            grid = np.arange(t_rr[0], t_rr[-1], 0.25)
            if grid.size >= 16:
                rr_resamp = np.interp(grid, t_rr, rr)
                rr_resamp = rr_resamp - rr_resamp.mean()
                nperseg = min(256, rr_resamp.size)
                f, psd = sps.welch(rr_resamp, fs=4.0, nperseg=nperseg)
                lf = float(np.trapezoid(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
                hf = float(np.trapezoid(psd[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]))
                tp = float(np.trapezoid(psd[(f >= 0.04) & (f < 0.4)], f[(f >= 0.04) & (f < 0.4)]))
                out[f"{prefix}_lf"] = float(np.log1p(max(lf, 0.0)))
                out[f"{prefix}_hf"] = float(np.log1p(max(hf, 0.0)))
                out[f"{prefix}_lfhf"] = lf / hf if hf > 1e-9 else 0.0
                out[f"{prefix}_tp"] = float(np.log1p(max(tp, 0.0)))
        except Exception:
            pass
    return out


def _eda_features(eda: np.ndarray, fs: int, prefix: str = "EDA") -> Dict[str, float]:
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
        t = np.arange(tonic.size) / float(fs)
        slope = np.polyfit(t, tonic, 1)[0]
        out[f"{prefix}_scl_slope"] = float(slope)
    out[f"{prefix}_phasic_std"] = float(np.std(phasic))
    distance = max(1, int(1.0 * fs))
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


def _acc_features(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: int, prefix: str = "ACC") -> Dict[str, float]:
    keys = ("sma", "mag_mean", "mag_std", "peak_freq")
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    if ax.size == 0:
        return out
    out[f"{prefix}_sma"] = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
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


def _temp_features(temp: np.ndarray, fs: int, prefix: str = "TEMP") -> Dict[str, float]:
    out = {f"{prefix}_mean": 0.0, f"{prefix}_slope": 0.0, f"{prefix}_range": 0.0}
    if temp.size >= 2:
        out[f"{prefix}_mean"] = float(np.mean(temp))
        t = np.arange(temp.size) / float(fs)
        try:
            out[f"{prefix}_slope"] = float(np.polyfit(t, temp, 1)[0])
        except Exception:
            pass
        out[f"{prefix}_range"] = float(np.ptp(temp))
    return out


def _eeg_preprocess(sig: np.ndarray, fs: int) -> np.ndarray:
    """Standard EEG cleaning before spectral analysis.

    1. Clip extreme amplitudes to +-300 μV (Muse hardware range; anything
       larger is a blink, electrode pop, or movement artifact).
    2. Band-pass 1-45 Hz to remove DC drift and isolate the bands of
       interest.
    3. 60 Hz notch (Korea grid frequency) to suppress powerline noise.

    Returns a finite-valued array; a zero-vector if filtering fails.
    """
    if sig.size < int(2 * fs):
        return np.zeros_like(sig)
    x = np.clip(sig, -300.0, 300.0).astype(np.float64)
    x = x - np.mean(x)
    nyq = 0.5 * fs
    try:
        b_bp, a_bp = sps.butter(2, [1.0 / nyq, min(45.0, nyq * 0.95) / nyq], btype="band")
        x = sps.filtfilt(b_bp, a_bp, x)
    except Exception:
        return np.zeros_like(sig)
    if 60.0 < nyq:
        try:
            b_n, a_n = sps.iirnotch(60.0, 30.0, fs)
            x = sps.filtfilt(b_n, a_n, x)
        except Exception:
            pass
    if not np.isfinite(x).all():
        return np.zeros_like(sig)
    return x.astype(np.float32)


def _eeg_band_powers(sig: np.ndarray, fs: int, prefix: str) -> Dict[str, float]:
    """Welch-based delta/theta/alpha/beta/gamma log-powers + their relative
    fractions for a single frontal EEG channel.

    Power is reported as ``log1p(power)`` because raw band powers span 6+
    orders of magnitude across windows/subjects, which destabilises the
    per-subject z-score.
    """
    bands = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
             "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
    keys = list(bands.keys())
    out = {f"{prefix}_{k}": 0.0 for k in keys}
    out[f"{prefix}_total"] = 0.0
    out[f"{prefix}_alpha_rel"] = 0.0
    out[f"{prefix}_beta_rel"] = 0.0
    if sig.size < int(2 * fs):
        return out
    sig_c = _eeg_preprocess(sig, fs)
    if not np.isfinite(sig_c).all() or np.std(sig_c) < 1e-9:
        return out
    try:
        f, psd = sps.welch(sig_c, fs=fs, nperseg=min(int(fs * 2), sig_c.size))
    except Exception:
        return out
    if not np.isfinite(psd).any():
        return out
    powers: Dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        sel = (f >= lo) & (f < hi)
        powers[name] = float(np.trapezoid(psd[sel], f[sel])) if sel.any() else 0.0
    total = sum(powers.values()) + 1e-12
    for k, v in powers.items():
        out[f"{prefix}_{k}"] = float(np.log1p(max(v, 0.0)))
    out[f"{prefix}_total"] = float(np.log1p(max(total, 0.0)))
    out[f"{prefix}_alpha_rel"] = powers["alpha"] / total
    out[f"{prefix}_beta_rel"] = powers["beta"] / total
    return out


def _channel_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0 or not np.isfinite(values).any():
        keys = ("mean", "std", "min", "max", "median", "ptp", "diff_std")
        return {f"{prefix}_{k}": 0.0 for k in keys}
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


def _downsample(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    pool = values.size // target_len
    if pool <= 1:
        x_old = np.linspace(0.0, 1.0, num=values.size)
        x_new = np.linspace(0.0, 1.0, num=target_len)
        return np.interp(x_new, x_old, values).astype(np.float32)
    trimmed = values[: pool * target_len]
    return trimmed.reshape(target_len, pool).mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-session sensor loading and resampling
# ---------------------------------------------------------------------------
def _load_session_signals(sensor_dir: str, label_dir: Optional[str]) -> Optional[Dict[str, object]]:
    """Load every sensor + label CSV for one ``(subject, session)`` and
    resample onto a common 32 Hz grid spanning their overlap.

    ``label_dir`` may be ``None`` (or missing on disk) for rest/baseline
    sessions where no continuous self-report exists; in that case the
    returned ``labels`` dict is empty and the caller should treat the
    windows as baseline reference only (i.e. exclude them from training).

    Returns a dict with ``grid_ms`` (1-D), ``signals`` (channel -> 1-D float32),
    plus continuous label arrays. ``None`` if the session is unusable.
    """
    files = {
        "ecg":   os.path.join(sensor_dir, "polar_ecg.csv"),
        "hr":    os.path.join(sensor_dir, "polar_hr.csv"),
        "bvp":   os.path.join(sensor_dir, "e4_bvp.csv"),
        "eda":   os.path.join(sensor_dir, "e4_eda.csv"),
        "temp":  os.path.join(sensor_dir, "e4_temp.csv"),
        "acc":   os.path.join(sensor_dir, "e4_acc.csv"),
        "muse":  os.path.join(sensor_dir, "muse.csv"),
    }
    label_files = {
        "arousal":     os.path.join(label_dir, "arousal.csv") if label_dir else None,
        "valence":     os.path.join(label_dir, "valence.csv") if label_dir else None,
        "stress":      os.path.join(label_dir, "stress.csv") if label_dir else None,
        "suppression": os.path.join(label_dir, "suppression.csv") if label_dir else None,
    }

    raw_ts: Dict[str, np.ndarray] = {}
    raw_vals: Dict[str, np.ndarray] = {}

    def _read_single(name: str, key: str) -> bool:
        df = _safe_read_csv(files[name])
        if df is None:
            return False
        if "Timestamp" not in df.columns or key not in df.columns:
            return False
        ts = df["Timestamp"].to_numpy(dtype=np.float64)
        vv = df[key].to_numpy(dtype=np.float32)
        good = np.isfinite(ts) & np.isfinite(vv)
        if good.sum() < 8:
            return False
        raw_ts[name] = ts[good]
        raw_vals[name] = vv[good]
        return True

    have_ecg = _read_single("ecg", "ecg")
    have_hr  = _read_single("hr", "HR")
    have_bvp = _read_single("bvp", "bvp")
    have_eda = _read_single("eda", "eda")
    have_temp = _read_single("temp", "temp")

    # ACC has three columns
    df_acc = _safe_read_csv(files["acc"], cols=["Timestamp", "accX", "accY", "accZ"])
    have_acc = df_acc is not None and len(df_acc) >= 8
    if have_acc:
        ts = df_acc["Timestamp"].to_numpy(dtype=np.float64)
        good = np.isfinite(ts)
        raw_ts["acc"] = ts[good]
        raw_vals["acc_x"] = df_acc["accX"].to_numpy(dtype=np.float32)[good]
        raw_vals["acc_y"] = df_acc["accY"].to_numpy(dtype=np.float32)[good]
        raw_vals["acc_z"] = df_acc["accZ"].to_numpy(dtype=np.float32)[good]

    # Muse EEG: TP9, AF7, AF8, TP10
    df_muse = _safe_read_csv(files["muse"])
    have_muse = df_muse is not None and "Timestamp" in df_muse.columns and all(
        c in df_muse.columns for c in ("TP9", "AF7", "AF8", "TP10")
    )
    if have_muse:
        ts = df_muse["Timestamp"].to_numpy(dtype=np.float64)
        good = np.isfinite(ts)
        raw_ts["muse"] = ts[good]
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            raw_vals[f"eeg_{ch}"] = df_muse[ch].to_numpy(dtype=np.float32)[good]

    # Labels (continuous) — may be empty for rest/baseline sessions.
    label_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, path in label_files.items():
        if path is None:
            continue
        df = _safe_read_csv(path)
        if df is None or "Timestamp" not in df.columns or name not in df.columns:
            continue
        ts = df["Timestamp"].to_numpy(dtype=np.float64)
        vv = df[name].to_numpy(dtype=np.float32)
        good = np.isfinite(ts) & np.isfinite(vv)
        if good.sum() < 4:
            continue
        label_arrays[name] = (ts[good], vv[good])

    if not have_bvp and not have_ecg:
        return None  # no cardiac signal -> drop session

    # Establish overlap interval. Take the intersection of label timestamps
    # (when present) with each sensor's range; fall back to sensors only.
    t_starts = [ts.min() for ts in raw_ts.values()]
    t_ends = [ts.max() for ts in raw_ts.values()]
    if label_arrays:
        t_starts += [v[0].min() for v in label_arrays.values()]
        t_ends += [v[0].max() for v in label_arrays.values()]
    t0 = max(t_starts)
    t1 = min(t_ends)
    if t1 - t0 < 30_000:  # at least 30 s of overlap
        return None

    grid_ms = np.arange(t0, t1, 1000.0 / COMMON_FS, dtype=np.float64)
    if grid_ms.size < COMMON_FS * 30:
        return None

    signals: Dict[str, np.ndarray] = {}
    if have_ecg: signals["ECG"] = _resample_to_grid(raw_ts["ecg"], raw_vals["ecg"], grid_ms)
    if have_bvp: signals["BVP"] = _resample_to_grid(raw_ts["bvp"], raw_vals["bvp"], grid_ms)
    if have_eda: signals["EDA"] = _resample_to_grid(raw_ts["eda"], raw_vals["eda"], grid_ms)
    if have_temp: signals["TEMP"] = _resample_to_grid(raw_ts["temp"], raw_vals["temp"], grid_ms)
    if have_acc:
        signals["ACC_x"] = _resample_to_grid(raw_ts["acc"], raw_vals["acc_x"], grid_ms)
        signals["ACC_y"] = _resample_to_grid(raw_ts["acc"], raw_vals["acc_y"], grid_ms)
        signals["ACC_z"] = _resample_to_grid(raw_ts["acc"], raw_vals["acc_z"], grid_ms)
    if have_hr: signals["HR"] = _resample_to_grid(raw_ts["hr"], raw_vals["hr"], grid_ms)
    if have_muse:
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            signals[f"EEG_{ch}"] = _resample_to_grid(raw_ts["muse"], raw_vals[f"eeg_{ch}"], grid_ms)

    # Resample raw EEG (256 Hz) onto its own grid for spectral features.
    eeg_raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if have_muse:
        for ch in ("TP9", "AF7", "AF8", "TP10"):
            eeg_raw[ch] = (raw_ts["muse"], raw_vals[f"eeg_{ch}"])

    # Resample labels
    labels: Dict[str, np.ndarray] = {}
    for name, (ts, vv) in label_arrays.items():
        labels[name] = _resample_to_grid(ts, vv, grid_ms)

    return {
        "grid_ms": grid_ms,
        "signals": signals,
        "labels": labels,
        "eeg_raw": eeg_raw,           # for high-fidelity band powers
        "raw_ecg": (raw_ts["ecg"], raw_vals["ecg"]) if have_ecg else None,
        "raw_bvp": (raw_ts["bvp"], raw_vals["bvp"]) if have_bvp else None,
    }


# ---------------------------------------------------------------------------
# Window-level feature extraction
# ---------------------------------------------------------------------------
def _window_features(
    win_signals: Dict[str, np.ndarray],
    raw_ecg_window: Optional[Tuple[np.ndarray, np.ndarray, int]],
    raw_bvp_window: Optional[Tuple[np.ndarray, np.ndarray, int]],
    raw_eeg_window: Dict[str, Tuple[np.ndarray, int]],
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    for ch in SEQ_CHANNELS:
        feats.update(_channel_stats(win_signals.get(ch, np.array([], dtype=np.float32)), ch))

    # ECG HRV: prefer raw 130 Hz signal restricted to window.
    if raw_ecg_window is not None:
        rt, rv, fs = raw_ecg_window
        peaks = _ecg_rpeaks(rv, fs)
        if peaks.size >= 4:
            rr = np.diff(peaks) / float(fs)
            feats.update(_hrv_from_peaks(rr, "ECG"))
        else:
            feats.update(_hrv_from_peaks(np.array([]), "ECG"))
    else:
        feats.update(_hrv_from_peaks(np.array([]), "ECG"))

    # BVP HRV: detect PPG peaks.
    if raw_bvp_window is not None:
        rt, rv, fs = raw_bvp_window
        if rv.size >= fs:
            try:
                nyq = 0.5 * fs
                b, a = sps.butter(2, [0.5 / nyq, min(8.0, nyq * 0.95) / nyq], btype="band")
                filt = sps.filtfilt(b, a, rv)
                distance = max(1, int(0.4 * fs))
                peaks, _ = sps.find_peaks(filt, distance=distance, prominence=np.std(filt) * 0.3)
                if peaks.size >= 4:
                    rr = np.diff(peaks) / float(fs)
                    feats.update(_hrv_from_peaks(rr, "BVP"))
                else:
                    feats.update(_hrv_from_peaks(np.array([]), "BVP"))
            except Exception:
                feats.update(_hrv_from_peaks(np.array([]), "BVP"))
        else:
            feats.update(_hrv_from_peaks(np.array([]), "BVP"))
    else:
        feats.update(_hrv_from_peaks(np.array([]), "BVP"))

    # EDA on common grid (32 Hz is fine; native is 4 Hz)
    if "EDA" in win_signals:
        feats.update(_eda_features(win_signals["EDA"], COMMON_FS))
    else:
        feats.update(_eda_features(np.array([]), COMMON_FS))

    # ACC
    if all(k in win_signals for k in ("ACC_x", "ACC_y", "ACC_z")):
        feats.update(_acc_features(win_signals["ACC_x"], win_signals["ACC_y"], win_signals["ACC_z"], COMMON_FS))
    else:
        feats.update(_acc_features(np.array([]), np.array([]), np.array([]), COMMON_FS))

    # TEMP
    if "TEMP" in win_signals:
        feats.update(_temp_features(win_signals["TEMP"], COMMON_FS))
    else:
        feats.update(_temp_features(np.array([]), COMMON_FS))

    # EEG band powers from raw 256 Hz where available
    for ch in ("TP9", "AF7", "AF8", "TP10"):
        if ch in raw_eeg_window:
            sig, fs = raw_eeg_window[ch]
            feats.update(_eeg_band_powers(sig, fs, f"EEG_{ch}"))
        else:
            feats.update(_eeg_band_powers(np.array([]), 256, f"EEG_{ch}"))

    return feats


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def _zscore_sequences_by_subject(
    seqs: np.ndarray,
    subject_ids: np.ndarray,
    clip_z: Optional[float] = 6.0,
) -> np.ndarray:
    if seqs.size == 0:
        return seqs
    out = seqs.astype(np.float32, copy=True)
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        block = out[mask]
        mean = block.mean(axis=(0, 1), keepdims=True)
        std = block.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        z = (block - mean) / std
        if clip_z is not None and clip_z > 0:
            z = np.clip(z, -clip_z, clip_z)
        out[mask] = z.astype(np.float32)
    return out


def load_emowork(
    window_seconds: int = 60,
    stride_seconds: int = 30,
    seq_length: int = SEQ_LENGTH,
    quick: bool = False,
    normalization: str = "zscore",
    clip_z: float = 6.0,
    baseline_correct: bool = True,
    verbose: bool = True,
    use_cache: bool = True,
    force_reload: bool = False,
) -> EmoWorkData:
    # ---- disk cache short-circuit -----------------------------------------
    cache_key = hashlib.md5(
        repr({
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "seq_length": seq_length,
            "quick": quick,
            "normalization": normalization,
            "clip_z": clip_z,
            "baseline_correct": baseline_correct,
            "v": 3,  # bump if feature extraction logic changes
        }).encode()
    ).hexdigest()[:12]
    cache_path = os.path.join(_CACHE_DIR, f"emowork_{cache_key}.pkl")
    if use_cache and not force_reload and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cached: EmoWorkData = pickle.load(fh)
            log(f"[EmoWork] cache hit: {cache_path}  samples={len(cached.samples)}", verbose)
            return cached
        except Exception as exc:  # pragma: no cover - corrupt cache
            log(f"[EmoWork] cache read failed ({exc}); recomputing", verbose)

    csv_root = os.path.join(EMOWORK_DIR, "CSV")
    sensor_root = os.path.join(csv_root, "SENSORS_csv")
    label_root = os.path.join(csv_root, "LABEL_csv")
    if not os.path.isdir(sensor_root):
        raise RuntimeError(f"EmoWork sensor folder missing: {sensor_root}")

    sids = sorted(
        [d for d in os.listdir(sensor_root) if d.isdigit() and os.path.isdir(os.path.join(sensor_root, d))],
        key=int,
    )
    if quick:
        sids = sids[:4]
    sessions = ("b1", "b2", "b3", "c1", "c2", "c3")
    log(f"[EmoWork] subjects={len(sids)}  sessions={sessions}  window={window_seconds}s stride={stride_seconds}s grid={COMMON_FS}Hz", verbose)

    rows: List[Dict[str, float]] = []
    seq_list: List[np.ndarray] = []
    baseline_rows: List[Dict[str, float]] = []
    baseline_seq_list: List[np.ndarray] = []
    rej_no_label = 0
    rej_flat = 0
    rej_no_cardiac = 0

    win_n = window_seconds * COMMON_FS
    stride_n = stride_seconds * COMMON_FS

    for sid in sids:
        sid_windows = 0
        sid_baseline = 0
        for sess in sessions:
            sensor_dir = os.path.join(sensor_root, sid, sess)
            label_dir = os.path.join(label_root, sid, sess)
            if not os.path.isdir(sensor_dir):
                continue
            label_dir_arg = label_dir if os.path.isdir(label_dir) else None
            payload = _load_session_signals(sensor_dir, label_dir_arg)
            if payload is None:
                continue
            grid_ms: np.ndarray = payload["grid_ms"]
            signals: Dict[str, np.ndarray] = payload["signals"]
            labels: Dict[str, np.ndarray] = payload["labels"]
            eeg_raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = payload["eeg_raw"]
            raw_ecg = payload["raw_ecg"]
            raw_bvp = payload["raw_bvp"]

            n = grid_ms.size
            for start in range(0, max(n - win_n + 1, 0), stride_n):
                end = start + win_n
                t_lo = grid_ms[start]
                t_hi = grid_ms[end - 1]

                # window-level signals
                win_signals = {ch: arr[start:end] for ch, arr in signals.items()}

                # window-level raw signals (for HRV / EEG bands)
                raw_ecg_w = None
                if raw_ecg is not None:
                    rt, rv = raw_ecg
                    m = (rt >= t_lo) & (rt <= t_hi)
                    if m.sum() >= 4:
                        # estimate native sample rate from this window
                        ts_w = rt[m]
                        if ts_w.size >= 2:
                            fs_est = max(1, int(round(1000.0 / max(np.median(np.diff(ts_w)), 1.0))))
                            raw_ecg_w = (ts_w, rv[m], fs_est)

                raw_bvp_w = None
                if raw_bvp is not None:
                    rt, rv = raw_bvp
                    m = (rt >= t_lo) & (rt <= t_hi)
                    if m.sum() >= 4:
                        ts_w = rt[m]
                        fs_est = max(1, int(round(1000.0 / max(np.median(np.diff(ts_w)), 1.0))))
                        raw_bvp_w = (ts_w, rv[m], fs_est)

                raw_eeg_w: Dict[str, Tuple[np.ndarray, int]] = {}
                for ch, (rt, rv) in eeg_raw.items():
                    m = (rt >= t_lo) & (rt <= t_hi)
                    if m.sum() >= 16:
                        ts_w = rt[m]
                        fs_est = max(1, int(round(1000.0 / max(np.median(np.diff(ts_w)), 1.0))))
                        raw_eeg_w[ch] = (rv[m], fs_est)

                # labels: average over the window
                lab_means: Dict[str, float] = {}
                for name, arr in labels.items():
                    seg = arr[start:end]
                    lab_means[name] = float(np.nanmean(seg)) if seg.size else np.nan

                is_baseline = not labels  # b-session: no continuous labels
                if not is_baseline and (
                    not np.isfinite(lab_means.get("arousal", np.nan))
                    or not np.isfinite(lab_means.get("valence", np.nan))
                ):
                    rej_no_label += 1
                    continue

                # Cheap pre-feature QC: drop windows where every modality is
                # flat (sensor disconnect / extrapolation tail).
                eda_w = win_signals.get("EDA")
                acc_x = win_signals.get("ACC_x")
                ecg_grid = win_signals.get("ECG")
                bvp_grid = win_signals.get("BVP")
                def _flat(a):
                    return a is None or a.size == 0 or not np.isfinite(a).any() or float(np.nanstd(a)) < 1e-6
                if (_flat(eda_w) and _flat(acc_x)
                        and _flat(ecg_grid) and _flat(bvp_grid)):
                    rej_flat += 1
                    continue

                feats = _window_features(win_signals, raw_ecg_w, raw_bvp_w, raw_eeg_w)

                # Post-feature QC: at least one cardiac channel must yield a
                # plausible HR (HRV func returns 0 for implausible HR).
                if (not is_baseline and feats.get("ECG_hr_mean", 0.0) <= 0.0
                        and feats.get("BVP_hr_mean", 0.0) <= 0.0):
                    rej_no_cardiac += 1
                    continue

                if is_baseline:
                    base_row: Dict[str, float] = {
                        "subject_id": f"P{int(sid):02d}",
                        "session": sess,
                    }
                    base_row.update(feats)
                    baseline_rows.append(base_row)
                    seq = np.zeros((seq_length, len(SEQ_CHANNELS)), dtype=np.float32)
                    for ci, ch in enumerate(SEQ_CHANNELS):
                        arr = win_signals.get(ch)
                        if arr is None or arr.size == 0 or not np.isfinite(arr).any():
                            continue
                        arr = np.nan_to_num(arr, nan=0.0)
                        seq[:, ci] = _downsample(arr, seq_length)
                    baseline_seq_list.append(seq)
                    sid_baseline += 1
                    continue

                ar_cont = lab_means.get("arousal", np.nan)
                va_cont = lab_means.get("valence", np.nan)
                st_cont = lab_means.get("stress", np.nan)
                sup_cont = lab_means.get("suppression", np.nan)

                arousal_bin = int(ar_cont >= 5.0)
                valence_bin = int(va_cont >= 5.0)
                stress_bin = int(st_cont >= 10.0) if np.isfinite(st_cont) else -1
                suppr_bin = int(sup_cont >= 10.0) if np.isfinite(sup_cont) else -1

                # Quadrant index in QUADRANTS = ("HVHA","HVLA","LVHA","LVLA")
                if valence_bin == 1 and arousal_bin == 1:
                    quad = 0  # HVHA
                elif valence_bin == 1 and arousal_bin == 0:
                    quad = 1  # HVLA
                elif valence_bin == 0 and arousal_bin == 1:
                    quad = 2  # LVHA
                else:
                    quad = 3  # LVLA

                row: Dict[str, float] = {
                    "subject_id": f"P{int(sid):02d}",
                    "session": sess,
                    "stage": sess,
                    "window_start_sec": float(start / COMMON_FS),
                    "arousal_cont": ar_cont,
                    "valence_cont": va_cont,
                    "stress_cont": st_cont,
                    "suppression_cont": sup_cont,
                    "arousal": arousal_bin,
                    "valence": valence_bin,
                    "stress": stress_bin,
                    "suppression": suppr_bin,
                    "quadrant_target": quad,
                    "target": quad,  # for parity with WESAD bundle
                }
                row.update(feats)

                rows.append(row)

                # build sequence tensor (240, 12)
                seq = np.zeros((seq_length, len(SEQ_CHANNELS)), dtype=np.float32)
                for ci, ch in enumerate(SEQ_CHANNELS):
                    arr = win_signals.get(ch)
                    if arr is None or arr.size == 0 or not np.isfinite(arr).any():
                        continue
                    arr = np.nan_to_num(arr, nan=0.0)
                    seq[:, ci] = _downsample(arr, seq_length)
                seq_list.append(seq)
                sid_windows += 1

        log(f"[EmoWork] P{int(sid):02d}: windows={sid_windows} baseline_windows={sid_baseline}", verbose)

    if not rows:
        raise RuntimeError("No EmoWork windows produced; check dataset path / sensor coverage.")

    log(
        f"[EmoWork] rejected windows: no_label={rej_no_label}  flat_all={rej_flat}  no_cardiac={rej_no_cardiac}",
        verbose,
    )

    df = pd.DataFrame(rows)
    sequences = np.stack(seq_list)
    base_df = pd.DataFrame(baseline_rows) if baseline_rows else None
    base_sequences = np.stack(baseline_seq_list) if baseline_seq_list else None

    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Per-subject baseline correction using the rest (b-session) windows
    # collected separately. Subtract the per-subject mean over baseline
    # windows from every (c-session) row of that subject so we model
    # condition-relative deviations rather than absolute physiology.
    if baseline_correct and base_df is not None:
        n_corrected = 0
        for sid_, sub_b in base_df.groupby("subject_id"):
            base_mean = sub_b[feature_cols].mean(axis=0)
            mask = df["subject_id"] == sid_
            df.loc[mask, feature_cols] = df.loc[mask, feature_cols].subtract(base_mean, axis=1)
            n_corrected += int(mask.sum())
        # Sequence baseline correction
        if base_sequences is not None and base_sequences.size:
            subj_arr = df["subject_id"].to_numpy()
            base_subj_arr = base_df["subject_id"].to_numpy()
            for sid_ in np.unique(subj_arr):
                base_mask = base_subj_arr == sid_
                if not base_mask.any():
                    continue
                base_mean = base_sequences[base_mask].mean(axis=(0, 1), keepdims=True)
                sid_mask = subj_arr == sid_
                sequences[sid_mask] = sequences[sid_mask] - base_mean
        log(f"[EmoWork] baseline-corrected windows: {n_corrected} (using {len(base_df)} b-session windows)", verbose)

    norm = normalization.lower().strip()
    if norm == "zscore":
        df = zscore_by_subject(df, feature_cols, "subject_id", clip_z=clip_z)
        sequences = _zscore_sequences_by_subject(
            sequences, df["subject_id"].to_numpy(), clip_z=clip_z,
        )
    elif norm == "none":
        pass
    else:
        raise ValueError(f"unknown EmoWork normalization: {normalization}")

    log(
        f"[EmoWork] samples={len(df)}  features={len(feature_cols)}  subjects={df['subject_id'].nunique()}  "
        f"quadrant_distribution={df['quadrant_target'].value_counts().to_dict()}  "
        f"stress_mean={df['stress_cont'].mean():.2f}  norm={norm}",
        verbose,
    )

    base_samples_out: Optional[pd.DataFrame] = None
    base_sequences_out: Optional[np.ndarray] = None
    if base_df is not None and len(base_df) > 0:
        base_samples_out = base_df.reset_index(drop=True)
        if base_sequences is not None:
            base_sequences_out = base_sequences.astype(np.float32)
        log(
            f"[EmoWork] baseline windows surfaced: {len(base_samples_out)} "
            f"across {base_samples_out['subject_id'].nunique()} subjects",
            verbose,
        )

    bundle = EmoWorkData(
        samples=df.reset_index(drop=True),
        feature_cols=feature_cols,
        sequences=sequences.astype(np.float32),
        seq_channels=list(SEQ_CHANNELS),
        seq_length=seq_length,
        baseline_samples=base_samples_out,
        baseline_sequences=base_sequences_out,
    )

    if use_cache:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"[EmoWork] cache write: {cache_path}", verbose)
        except Exception as exc:  # pragma: no cover
            log(f"[EmoWork] cache write failed ({exc})", verbose)

    return bundle


__all__ = ["EmoWorkData", "load_emowork", "SEQ_CHANNELS", "SEQ_LENGTH", "COMMON_FS"]
