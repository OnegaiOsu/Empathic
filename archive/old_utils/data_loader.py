"""
Data loading and merging utilities for SWELL-KW stress classification.

Loads physiology (HR, RMSSD, SCL) and computer interaction (mouse, keyboard,
app switching) feature CSVs, merges them, and encodes binary stress labels.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(
    _BASE_DIR,
    "0_SWELL",
    "3 - Feature dataset",
    "per sensor",
)

PHYSIOLOGY_CSV = os.path.join(
    _DATA_DIR,
    "D - Physiology features (HR_HRV_SCL - final).csv",
)
COMPUTER_CSV = os.path.join(
    _DATA_DIR,
    "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
)

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
PHYSIOLOGY_FEATURES = ["HR", "RMSSD", "SCL"]

COMPUTER_FEATURES = [
    "SnMouseAct",
    "SnLeftClicked",
    "SnRightClicked",
    "SnDoubleClicked",
    "SnWheel",
    "SnDragged",
    "SnMouseDistance",
    "SnKeyStrokes",
    "SnChars",
    "SnSpecialKeys",
    "SnDirectionKeys",
    "SnErrorKeys",
    "SnShortcutKeys",
    "SnSpaces",
    "SnAppChange",
    "SnTabfocusChange",
]

ALL_FEATURES = PHYSIOLOGY_FEATURES + COMPUTER_FEATURES

LABEL_COL = "label"
GROUP_COL = "PP"


def get_feature_columns() -> list[str]:
    """Return the list of all 19 feature column names."""
    return list(ALL_FEATURES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_physiology() -> pd.DataFrame:
    """Load physiology CSV and clean extra trailing columns."""
    df = pd.read_csv(PHYSIOLOGY_CSV)
    # Drop unnamed trailing columns (artefact of extra commas in CSV)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Keep only the columns we need
    df = df[["PP", "C", "Condition", "timestamp"] + PHYSIOLOGY_FEATURES].copy()
    # Replace 999 sentinel with NaN
    for col in PHYSIOLOGY_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 999, col] = np.nan
    return df


def _load_computer_interaction() -> pd.DataFrame:
    """Load computer interaction CSV and clean extra trailing columns."""
    df = pd.read_csv(COMPUTER_CSV)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # Rename 'Blok' -> 'C' for consistency
    df = df.rename(columns={"Blok": "C"})
    df = df[["PP", "C", "Condition", "timestamp"] + COMPUTER_FEATURES].copy()
    for col in COMPUTER_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_merge_data() -> pd.DataFrame:
    """
    Load both CSVs, merge on (PP, C, Condition, timestamp), encode binary
    stress label, and return a clean DataFrame.

    Label encoding:
        - Condition in {T, I} -> 1  (Stressed)
        - Condition in {N, R} -> 0  (Not Stressed)

    Returns
    -------
    pd.DataFrame
        Columns: PP, C, Condition, timestamp, <19 features>, label
    """
    physio = _load_physiology()
    computer = _load_computer_interaction()

    # Merge on shared keys
    df = pd.merge(
        physio,
        computer,
        on=["PP", "C", "Condition", "timestamp"],
        how="inner",
    )

    # Binary label: stressed (T or I) = 1, not stressed (N or R) = 0
    df[LABEL_COL] = df["Condition"].map({"T": 1, "I": 1, "N": 0, "R": 0})

    # Drop rows where the condition didn't map (shouldn't happen)
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"[data_loader] Merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[data_loader] Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    print(f"[data_loader] Participants: {df['PP'].nunique()}")
    missing = df[ALL_FEATURES].isna().sum()
    if missing.any():
        print(f"[data_loader] Missing values per feature:\n{missing[missing > 0].to_string()}")

    return df


# ---------------------------------------------------------------------------
# LOSO splitting
# ---------------------------------------------------------------------------
def get_loso_splits(df: pd.DataFrame):
    """
    Yield (train_idx, test_idx, participant_id) for Leave-One-Subject-Out CV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'PP' (participant identifier).

    Yields
    ------
    train_idx : np.ndarray
    test_idx : np.ndarray
    pp_id : str
        Participant ID for the held-out fold.
    """
    logo = LeaveOneGroupOut()
    groups = df[GROUP_COL].values
    X_dummy = np.zeros(len(df))  # placeholder, split only needs groups
    y_dummy = np.zeros(len(df))

    for train_idx, test_idx in logo.split(X_dummy, y_dummy, groups):
        pp_id = df.iloc[test_idx][GROUP_COL].iloc[0]
        yield train_idx, test_idx, pp_id
