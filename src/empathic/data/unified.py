"""Unified dataset bundle and label harmonisation.

Design decision: EmoSurv and WESAD use different sensors, sampling rates and
label vocabularies, so we cannot align them at the raw-feature level. Instead
we harmonise them at the *label* level by projecting both label spaces onto
Russell's (1980) Circumplex quadrants, which is the de-facto common ground of
affective computing. The feature matrices remain dataset-specific so each
modality keeps its most informative signals, but every model reports metrics on
the same 4-class quadrant target (and on the original dataset-specific labels
for completeness).

Outputs of ``build_bundles`` are two :class:`DatasetBundle` objects that the
training harness consumes uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    EMOSURV_LABEL_TO_QUADRANT,
    EMOSURV_LABEL_TO_VA,
    QUADRANT_INDEX,
    QUADRANT_WITH_NEUTRAL_INDEX,
    QUADRANTS,
    QUADRANTS_WITH_NEUTRAL,
    WESAD_LABEL_TO_QUADRANT,
)
from .emosurv import EmoSurvData, load_emosurv
from .wesad import WESADData, load_wesad
from .emowork import EmoWorkData, load_emowork


@dataclass
class DatasetBundle:
    """Uniform view over a dataset for the training harness."""
    name: str                          # "emosurv" or "wesad"
    samples: pd.DataFrame              # tabular features + ``quadrant`` column
    feature_cols: List[str]            # tabular feature columns
    sequences: np.ndarray              # (N, L, C) float32 tensor
    seq_channels: List[str]
    seq_length: int
    native_labels: List[str]           # ordered dataset-specific label names
    native_target: np.ndarray          # int array (N,) in native space
    quadrant_target: np.ndarray        # int array (N,) in {0..3} (or {0..4})
    quadrant_labels: List[str]         # ordered quadrant vocabulary
    session_key: np.ndarray            # (N,) string key identifying contiguous session
    valence: Optional[np.ndarray] = None
    arousal: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def subject_ids(self) -> np.ndarray:
        return self.samples["subject_id"].to_numpy()


def unify_quadrant_label(dataset: str, native_label) -> int:
    """Map a native label to its circumplex quadrant index (0..3)."""
    dataset = dataset.lower()
    if dataset == "emosurv":
        q = EMOSURV_LABEL_TO_QUADRANT.get(str(native_label))
    elif dataset == "wesad":
        q = WESAD_LABEL_TO_QUADRANT.get(int(native_label))
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if q is None:
        raise ValueError(f"no quadrant mapping for {dataset} label {native_label!r}")
    return QUADRANT_INDEX[q]


def _bundle_from_emosurv(data: EmoSurvData, neutral_policy: str = "merge") -> DatasetBundle:
    df = data.samples.copy()
    policy = neutral_policy.lower().strip()
    if policy == "separate":
        label_to_q = dict(EMOSURV_LABEL_TO_QUADRANT)
        label_to_q["N"] = "NEU"
        q_vocab = list(QUADRANTS_WITH_NEUTRAL)
        q_index = dict(QUADRANT_WITH_NEUTRAL_INDEX)
    else:
        label_to_q = EMOSURV_LABEL_TO_QUADRANT
        q_vocab = list(QUADRANTS)
        q_index = dict(QUADRANT_INDEX)

    df["quadrant"] = df["emotion"].map(label_to_q)
    mask = df["quadrant"].notna().to_numpy()
    if not mask.all():
        df = df.loc[mask].reset_index(drop=True)
        sequences = data.sequences[mask]
    else:
        sequences = data.sequences
    df["quadrant_target"] = df["quadrant"].map(q_index).astype(int)

    # EmoSurv has no ground-truth SAM scores, but we project each emotion code
    # onto its literature-motivated default valence/arousal so the model can
    # be evaluated in the dimensional frame too.
    va = df["emotion"].map(EMOSURV_LABEL_TO_VA)
    df["valence"] = va.map(lambda pair: float(pair[0]) if pair else np.nan)
    df["arousal"] = va.map(lambda pair: float(pair[1]) if pair else np.nan)

    native_labels = sorted(df["emotion"].unique().tolist())
    name_to_idx = {name: i for i, name in enumerate(native_labels)}
    df["native_target"] = df["emotion"].map(name_to_idx).astype(int)

    session_key = (
        df["subject_id"].astype(str) + "|" + df["session_id"].astype(str)
    ).to_numpy()

    return DatasetBundle(
        name="emosurv",
        samples=df,
        feature_cols=data.feature_cols,
        sequences=sequences.astype(np.float32),
        seq_channels=list(data.seq_channels),
        seq_length=data.seq_length,
        native_labels=native_labels,
        native_target=df["native_target"].to_numpy(),
        quadrant_target=df["quadrant_target"].to_numpy(),
        quadrant_labels=q_vocab,
        session_key=session_key,
        valence=df["valence"].to_numpy(dtype=np.float32),
        arousal=df["arousal"].to_numpy(dtype=np.float32),
        extra={"neutral_policy": policy},
    )


def _bundle_from_wesad(data: WESADData) -> DatasetBundle:
    df = data.samples.copy()
    df["quadrant"] = df["target"].map(WESAD_LABEL_TO_QUADRANT)
    mask = df["quadrant"].notna().to_numpy()
    if not mask.all():
        df = df.loc[mask].reset_index(drop=True)
        sequences = data.sequences[mask]
    else:
        sequences = data.sequences
    df["quadrant_target"] = df["quadrant"].map(QUADRANT_INDEX).astype(int)

    native_labels_int = sorted(df["target"].unique().tolist())
    native_labels = [str(v) for v in native_labels_int]
    int_to_idx = {v: i for i, v in enumerate(native_labels_int)}
    df["native_target"] = df["target"].map(int_to_idx).astype(int)

    stage_col = df["stage"].astype(str) if "stage" in df.columns else df["target"].astype(str)
    session_key = (df["subject_id"].astype(str) + "|" + stage_col).to_numpy()

    return DatasetBundle(
        name="wesad",
        samples=df,
        feature_cols=data.feature_cols,
        sequences=sequences.astype(np.float32),
        seq_channels=list(data.seq_channels),
        seq_length=data.seq_length,
        native_labels=native_labels,
        native_target=df["native_target"].to_numpy(),
        quadrant_target=df["quadrant_target"].to_numpy(),
        quadrant_labels=list(QUADRANTS),
        session_key=session_key,
        valence=df["valence"].to_numpy(dtype=np.float32),
        arousal=df["arousal"].to_numpy(dtype=np.float32),
    )


def _bundle_from_emowork(data: "EmoWorkData") -> DatasetBundle:
    """Bundle EmoWork V2.

    Quadrant labels are derived from the binary discretisations of the
    self-reported continuous valence/arousal channels (threshold = 5 on the
    1..9 Likert), so all four quadrants are present (unlike WESAD where the
    LVLA quadrant has no condition mapping).
    """
    df = data.samples.copy()
    # Native target = quadrant for parity with WESAD's "stage"-style key.
    df["native_target"] = df["quadrant_target"].astype(int)
    native_labels = list(QUADRANTS)

    session_key = (
        df["subject_id"].astype(str) + "|" + df["session"].astype(str)
    ).to_numpy()

    return DatasetBundle(
        name="emowork",
        samples=df,
        feature_cols=data.feature_cols,
        sequences=data.sequences.astype(np.float32),
        seq_channels=list(data.seq_channels),
        seq_length=data.seq_length,
        native_labels=native_labels,
        native_target=df["native_target"].to_numpy(),
        quadrant_target=df["quadrant_target"].to_numpy(),
        quadrant_labels=list(QUADRANTS),
        session_key=session_key,
        valence=df["valence_cont"].to_numpy(dtype=np.float32),
        arousal=df["arousal_cont"].to_numpy(dtype=np.float32),
        stress=df["stress"].to_numpy(dtype=np.int64),
        extra={
            "stress_cont": df["stress_cont"].to_numpy(dtype=np.float32),
            "suppression": df["suppression"].to_numpy(dtype=np.int64),
            "suppression_cont": df["suppression_cont"].to_numpy(dtype=np.float32),
            # Rest-session (b1/b2/b3) windows for per-subject calibration
            # experiments. May be None if loader did not surface them.
            "baseline_samples": data.baseline_samples,
            "baseline_sequences": data.baseline_sequences,
        },
    )


def build_bundles(
    datasets: List[str],
    *,
    quick: bool = False,
    emosurv_normalization: str = "robust",
    wesad_normalization: str = "zscore",
    emosurv_neutral_policy: str = "merge",
    emosurv_window_size: Optional[int] = None,
    emosurv_stride: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, DatasetBundle]:
    """Load requested datasets and wrap them in :class:`DatasetBundle` objects."""
    bundles: Dict[str, DatasetBundle] = {}
    for name in datasets:
        name = name.lower().strip()
        if name == "emosurv":
            data = load_emosurv(
                window_size=emosurv_window_size,
                stride=emosurv_stride,
                quick=quick,
                normalization=emosurv_normalization,
                neutral_policy=emosurv_neutral_policy,
                verbose=verbose,
            )
            bundles["emosurv"] = _bundle_from_emosurv(data, neutral_policy=emosurv_neutral_policy)
        elif name == "wesad":
            data = load_wesad(quick=quick, normalization=wesad_normalization, verbose=verbose)
            bundles["wesad"] = _bundle_from_wesad(data)
        elif name == "emowork":
            data = load_emowork(quick=quick, normalization=wesad_normalization, verbose=verbose)
            bundles["emowork"] = _bundle_from_emowork(data)
        else:
            raise ValueError(f"unknown dataset: {name}")
    return bundles


__all__ = [
    "DatasetBundle",
    "build_bundles",
    "unify_quadrant_label",
    "QUADRANTS",
]
