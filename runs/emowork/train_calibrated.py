"""Calibrated LOSO: per-subject rest-session z-score normalisation.

Protocol C from the calibration ablation: every subject's tabular features and
sequence channels are z-scored against the mean/std computed *only* on that
subject's rest sessions (b1, b2, b3). No labels from the held-out subject are
used; only the unlabelled rest baseline. This isolates the static
physiological-baseline component of subject variance and answers whether the
"generalised LOSO" floor of paper_v2/05_results.md is artefactually low because
of unaddressed inter-subject baseline shift.

Outputs go to ``results/emotion/emowork/calibrated/<target>/`` so the
uncalibrated LOSO numbers in ``emowork/<target>/`` stay intact for diff-style
comparison.

Usage::

    .venv\\Scripts\\python.exe runs\\emowork\\train_calibrated.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from empathic.data.emowork import load_emowork
from empathic.data.unified import _bundle_from_emowork
from empathic.training import run_experiment


RESULTS_ROOT = os.path.join("results", "emotion", "emowork", "calibrated")
ENSEMBLE = ("RandomForest", "LogisticRegression", "XGBoost",
            "conformer_fusion", "tiny_tcn_fusion", "multistream_fusion",
            "DANN_Conformer", "TSTCC")
TARGETS = ("stress", "arousal", "valence", "quadrant")
DEEP_ARCHS = ["conformer", "tiny_tcn", "bilstm", "cnn1d", "multistream",
              "dann", "tstcc"]

# Sessions used as the per-subject reference baseline.
REST_SESSIONS = ("b1", "b2", "b3")
EPS = 1e-6


def _calibrate_in_place(bundle) -> None:
    """Replace tabular feats and seq channels with per-subject z-scores
    against each subject's own rest sessions (b1/b2/b3).

    Uses the dedicated baseline arrays surfaced by ``load_emowork`` via
    ``bundle.extra['baseline_samples'] / 'baseline_sequences']``. Subjects
    with no rest windows fall back to using their full per-subject mean/std
    over the c-session windows (still subject-internal, just not rest-
    anchored). We log a count of such fallback subjects.
    """
    df = bundle.samples
    feature_cols = list(bundle.feature_cols)
    subj_col = df["subject_id"].astype(str).to_numpy()

    base_df = bundle.extra.get("baseline_samples")
    base_seq = bundle.extra.get("baseline_sequences")
    if base_df is None or len(base_df) == 0:
        raise RuntimeError(
            "bundle.extra['baseline_samples'] is empty. Did you load with "
            "baseline_correct=False so the loader exposes b-session windows?"
        )
    base_subj = base_df["subject_id"].astype(str).to_numpy()
    base_X = base_df[feature_cols].to_numpy(dtype=np.float64)
    base_seq_arr = (
        base_seq.astype(np.float64) if base_seq is not None else None
    )

    X = df[feature_cols].to_numpy(dtype=np.float64).copy()
    seq = bundle.sequences.astype(np.float64).copy()

    fallback_subjects: list = []
    used_full_subj = 0
    n_subjects = len(np.unique(subj_col))

    for subject in np.unique(subj_col):
        s_mask = subj_col == subject
        rest_mask = base_subj == subject
        n_rest = int(rest_mask.sum())

        if n_rest < 2:
            # Fall back to that subject's own c-session distribution.
            ref_X = X[s_mask]
            ref_seq = seq[s_mask]
            fallback_subjects.append(subject)
            used_full_subj += 1
        else:
            ref_X = base_X[rest_mask]
            ref_seq = base_seq_arr[rest_mask] if base_seq_arr is not None else None

        # Tabular features.
        mu = ref_X.mean(axis=0)
        sd = ref_X.std(axis=0)
        sd = np.where(sd < EPS, 1.0, sd)
        X[s_mask] = (X[s_mask] - mu) / sd

        # Sequence channels: per-channel mean/std across (windows * T).
        if ref_seq is not None and ref_seq.size:
            ch_mu = ref_seq.mean(axis=(0, 1))   # (C,)
            ch_sd = ref_seq.std(axis=(0, 1))
            ch_sd = np.where(ch_sd < EPS, 1.0, ch_sd)
            seq[s_mask] = (seq[s_mask] - ch_mu) / ch_sd

    # Clip extreme z-scores like the loader's default zscore_by_subject does
    # (clip_z=6) so a single noisy rest window can't blow up downstream.
    np.clip(X, -6.0, 6.0, out=X)
    np.clip(seq, -6.0, 6.0, out=seq)

    print(
        f"[calibrate] subjects fallback (no rest windows -> full c-session norm): "
        f"{used_full_subj}/{n_subjects}",
        flush=True,
    )
    if fallback_subjects:
        print(f"[calibrate]   fallback ids: {fallback_subjects}", flush=True)

    # Sanity stats.
    print(
        f"[calibrate] tabular post-norm mean|abs|={np.mean(np.abs(X)):.3f}  "
        f"std={X.std():.3f}",
        flush=True,
    )
    print(
        f"[calibrate] seq     post-norm mean|abs|={np.mean(np.abs(seq)):.3f}  "
        f"std={seq.std():.3f}",
        flush=True,
    )

    # Replace columns in-place. Keep dtype float32 for consistency with the
    # rest of the pipeline.
    df.loc[:, feature_cols] = X.astype(np.float32)
    bundle.sequences = seq.astype(np.float32)


def main() -> None:
    print("[emowork/train_calibrated] loading raw bundle (baseline_correct=False, normalization=none) ...", flush=True)
    t0 = time.time()
    data = load_emowork(
        quick=False,
        normalization="none",
        baseline_correct=False,
        verbose=True,
    )
    bundle = _bundle_from_emowork(data)
    n_base = 0 if data.baseline_samples is None else len(data.baseline_samples)
    print(
        f"[emowork/train_calibrated] bundle ready in {time.time()-t0:.1f}s "
        f"(c-samples={len(bundle.samples)}, "
        f"subjects={bundle.samples['subject_id'].nunique()}, "
        f"baseline_windows={n_base})",
        flush=True,
    )

    print("[emowork/train_calibrated] applying per-subject rest-session "
          "z-score calibration ...", flush=True)
    _calibrate_in_place(bundle)

    for target in TARGETS:
        print(f"\n{'#'*70}\n# CALIBRATED TARGET = {target}\n{'#'*70}", flush=True)
        run_experiment(
            bundle,
            target_kind=target,
            seed=42,
            use_gpu=True,
            include_classical=True,
            include_deep=True,
            deep_arch=DEEP_ARCHS,
            fusion=True,
            mixup_alpha=0.2,
            ensemble_members=ENSEMBLE,
            results_root=RESULTS_ROOT,
            verbose=True,
        )


if __name__ == "__main__":
    main()
