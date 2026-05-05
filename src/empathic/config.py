"""Global configuration: paths, device, defaults.

Centralising these avoids magic strings across loaders and trainers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
EMOSURV_DIR = os.path.join(DATASET_DIR, "EmoSurv")
WESAD_DIR = os.path.join(DATASET_DIR, "WeSad", "archive(2)", "WESAD")

# EmoWork is large (~3.6 GB / 5.6k files) and the in-repo HDD copy is slow.
# Prefer the SSD mirror at C:\dev\datasets when present; fall back to the
# in-repo location. Override with EMPATHIC_EMOWORK_DIR for custom layouts.
_EMOWORK_SSD = r"C:\dev\datasets\EmoWork_v2\EmoWorker_v2"
_EMOWORK_REPO = os.path.join(DATASET_DIR, "EmoWork_v2", "EmoWorker_v2")
EMOWORK_DIR = os.environ.get(
    "EMPATHIC_EMOWORK_DIR",
    _EMOWORK_SSD if os.path.isdir(_EMOWORK_SSD) else _EMOWORK_REPO,
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "emotion")

# ---------------------------------------------------------------------------
# Device: prefer CUDA (RTX 50-series) when present.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Unified label space: Russell's Circumplex quadrants.
#
# We project both corpora onto the same four emotional quadrants so that a
# single classifier head can be compared between datasets. The reference
# coordinates follow Russell (1980) and the SAM scale used by WESAD.
#
#   HVHA = high valence + high arousal  (e.g. happy, amused)
#   HVLA = high valence + low arousal   (e.g. calm, meditation)
#   LVHA = low valence  + high arousal  (e.g. angry, stressed)
#   LVLA = low valence  + low arousal   (e.g. sad)
# ---------------------------------------------------------------------------
QUADRANTS: Tuple[str, str, str, str] = ("HVHA", "HVLA", "LVHA", "LVLA")
QUADRANT_INDEX: Dict[str, int] = {q: i for i, q in enumerate(QUADRANTS)}

# Extended label space used when EmoSurv Neutral is kept as its own class
# (see ``--emosurv-neutral separate``). ``NEU`` sits at the circumplex origin.
QUADRANTS_WITH_NEUTRAL: Tuple[str, str, str, str, str] = ("HVHA", "HVLA", "LVHA", "LVLA", "NEU")
QUADRANT_WITH_NEUTRAL_INDEX: Dict[str, int] = {q: i for i, q in enumerate(QUADRANTS_WITH_NEUTRAL)}

# EmoSurv single-letter codes -> quadrant.
EMOSURV_LABEL_TO_QUADRANT: Dict[str, str] = {
    "H": "HVHA",   # Happy
    "C": "HVLA",   # Calm
    "N": "HVLA",   # Neutral is treated as calm/low-activation reference.
    "A": "LVHA",   # Angry
    "S": "LVLA",   # Sad
}

# EmoSurv default SAM coordinates (1..9 scale) for dimensional regression.
EMOSURV_LABEL_TO_VA: Dict[str, Tuple[float, float]] = {
    "H": (8.0, 7.0),
    "C": (7.0, 3.0),
    "N": (5.0, 5.0),
    "A": (2.0, 7.5),
    "S": (2.5, 3.0),
}

# WESAD integer labels from the pickle (see wesad_readme.pdf).
# 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement,
# 4 = meditation, 5/6/7 = pre/post periods that we ignore.
WESAD_LABEL_TO_QUADRANT: Dict[int, str] = {
    1: "HVLA",   # Baseline
    2: "LVHA",   # TSST stress
    3: "HVHA",   # Amusement (fun clips)
    4: "HVLA",   # Meditation
}
WESAD_KEEP_LABELS = tuple(sorted(WESAD_LABEL_TO_QUADRANT.keys()))

# WESAD questionnaire stage names (as written in Sx_quest.csv) mapped to
# condition labels used in the pickle. Used to align self-reported SAM scores
# with windowed signal segments.
WESAD_STAGE_TO_LABEL: Dict[str, int] = {
    "Base": 1,
    "TSST": 2,
    "Fun": 3,
    "Medi 1": 4,
    "Medi 2": 4,
}


# ---------------------------------------------------------------------------
# Windowing and training defaults.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataDefaults:
    # EmoSurv keystroke windows (number of consecutive key events).
    emosurv_window_keys: int = 35
    emosurv_window_stride: int = 20
    # WESAD chest-sensor windows in seconds; 60 s with 30 s stride is the
    # Schmidt et al. (2018) protocol.
    wesad_window_seconds: int = 60
    wesad_stride_seconds: int = 30
    wesad_sample_rate: int = 700
    # Per-subject robust clipping to suppress extreme artefacts.
    clip_z: float = 6.0


@dataclass(frozen=True)
class TrainDefaults:
    seed: int = 42
    deep_epochs: int = 40
    deep_batch_size: int = 128
    deep_lr: float = 1e-3
    deep_weight_decay: float = 1e-4
    early_stop_patience: int = 8
    seq_len_keystroke: int = 35
    seq_len_physio: int = 240  # downsampled from 60 s @ 4 Hz effective rate
