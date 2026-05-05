"""
Improved data loading using the master Behavioral Features file and proper normalization.

Key fixes over original data_loader.py:
1. Uses the master 'Behavioral-features - per minute.xlsx' as data source
2. Applies per-person z-scoring (physiology varies 14x across participants)
3. Adds self-reported stress/NASA-TLX as optional label targets
4. Adds missingness indicators instead of masking with median imputation
5. Properly handles 999 sentinel AND native NaN values
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "0_SWELL")

BEHAVIORAL_XLSX = os.path.join(_DATA_DIR, "Behavioral-features - per minute.xlsx")

# Fallback CSV paths (if xlsx fails)
_SENSOR_DIR = os.path.join(_DATA_DIR, "3 - Feature dataset", "per sensor")
PHYSIOLOGY_CSV = os.path.join(_SENSOR_DIR, "D - Physiology features (HR_HRV_SCL - final).csv")
COMPUTER_CSV_SHEET3 = os.path.join(
    _SENSOR_DIR,
    "A - Computer interaction features (Ulog - All Features per minute)-Sheet_3.csv",
)

# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------
PHYSIOLOGY_FEATURES = ["HR", "RMSSD", "SCL"]

COMPUTER_FEATURES = [
    "SnMouseAct", "SnLeftClicked", "SnRightClicked", "SnDoubleClicked",
    "SnWheel", "SnDragged", "SnMouseDistance", "SnKeyStrokes",
    "SnChars", "SnSpecialKeys", "SnDirectionKeys", "SnErrorKeys",
    "SnShortcutKeys", "SnSpaces", "SnAppChange", "SnTabfocusChange",
]

# Derived features we'll engineer
DERIVED_FEATURES = [
    "CharactersRatio",   # SnChars / SnKeyStrokes
    "ErrorKeyRatio",     # SnErrorKeys / SnKeyStrokes
]

# Self-reported questionnaire features (per block, repeated for each minute)
QUESTIONNAIRE_FEATURES = [
    "Stress", "MentalEffort", "Arousal_rc", "Valence_rc",
    "Frustration", "NasaTLX",
]

ALL_FEATURES = PHYSIOLOGY_FEATURES + COMPUTER_FEATURES
LABEL_COL = "label"
GROUP_COL = "PP"


def get_feature_columns(include_derived=True, include_questionnaire=False):
    """Return the list of feature column names."""
    cols = list(ALL_FEATURES)
    if include_derived:
        cols.extend(DERIVED_FEATURES)
    if include_questionnaire:
        cols.extend(QUESTIONNAIRE_FEATURES)
    return cols


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_from_behavioral_xlsx():
    """
    Load the master Behavioral Features file.
    172 columns, minute-level data with all sensors merged.
    """
    df = pd.read_excel(BEHAVIORAL_XLSX, sheet_name="SWELLdata")

    # Keep relevant columns
    keep_cols = ["PP", "Blok", "Condition", "timestamp"]
    keep_cols += PHYSIOLOGY_FEATURES + COMPUTER_FEATURES
    keep_cols += DERIVED_FEATURES + QUESTIONNAIRE_FEATURES

    # Only keep columns that exist
    available = [c for c in keep_cols if c in df.columns]
    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        print(f"[data_loader_v2] Columns not in xlsx: {missing_cols}")

    df = df[available].copy()
    df = df.rename(columns={"Blok": "C"})

    # Clean sentinel values in physiology
    for col in PHYSIOLOGY_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 999, col] = np.nan

    # Clean computer interaction features
    for col in COMPUTER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _load_from_csvs():
    """Fallback: load from separate CSV files (using Sheet 3 for computer data)."""
    # Physiology
    physio = pd.read_csv(PHYSIOLOGY_CSV)
    physio = physio.loc[:, ~physio.columns.str.startswith("Unnamed")]
    physio = physio[["PP", "C", "Condition", "timestamp"] + PHYSIOLOGY_FEATURES].copy()
    for col in PHYSIOLOGY_FEATURES:
        physio[col] = pd.to_numeric(physio[col], errors="coerce")
        physio.loc[physio[col] == 999, col] = np.nan

    # Computer interaction (Sheet 3 — clean numeric SnMouseDistance)
    comp = pd.read_csv(COMPUTER_CSV_SHEET3)
    comp = comp.loc[:, ~comp.columns.str.startswith("Unnamed")]
    comp = comp.rename(columns={"Blok": "C"})
    # Drop the duplicate SnMouseDistance.1 column if present
    if "SnMouseDistance.1" in comp.columns:
        comp = comp.drop(columns=["SnMouseDistance.1"])
    comp = comp[["PP", "C", "Condition", "timestamp"] + COMPUTER_FEATURES].copy()
    for col in COMPUTER_FEATURES:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")

    # Merge
    df = pd.merge(physio, comp, on=["PP", "C", "Condition", "timestamp"], how="inner")
    return df


def _add_missingness_indicators(df):
    """
    Add binary indicators for missing physiology data.
    These encode WHICH sensor failed, which is informative
    (e.g., PP8 has no heart data at all — the model should know this).
    """
    for col in PHYSIOLOGY_FEATURES:
        indicator_col = f"{col}_missing"
        df[indicator_col] = df[col].isna().astype(int)
    return df


def _per_person_normalize(df, features):
    """
    Z-score each feature within each participant.

    This is critical because:
    - SCL varies from 65 (PP1) to 921 (PP9) across participants
    - HR baselines differ by 20+ bpm across participants
    - The stress signal is in WITHIN-PERSON changes, not absolute values

    For features with no variance within a participant (all same value
    or all missing → imputed to same value), sets to 0.
    """
    df = df.copy()
    for col in features:
        if col not in df.columns:
            continue
        # Group by participant and z-score
        grouped = df.groupby("PP")[col]
        pp_mean = grouped.transform("mean")
        pp_std = grouped.transform("std")
        # Avoid division by zero (constant feature for this participant)
        pp_std = pp_std.replace(0, np.nan)
        df[col] = (df[col] - pp_mean) / pp_std
        # Fill NaN from zero-std participants with 0
        df[col] = df[col].fillna(0)
    return df


def _add_derived_features(df):
    """Compute ratio features if not already present."""
    if "CharactersRatio" not in df.columns:
        total_keys = df["SnKeyStrokes"].replace(0, np.nan)
        df["CharactersRatio"] = df["SnChars"] / total_keys
        df["CharactersRatio"] = df["CharactersRatio"].fillna(0)

    if "ErrorKeyRatio" not in df.columns:
        total_keys = df["SnKeyStrokes"].replace(0, np.nan)
        df["ErrorKeyRatio"] = df["SnErrorKeys"] / total_keys
        df["ErrorKeyRatio"] = df["ErrorKeyRatio"].fillna(0)

    return df


def load_and_merge_data(
    per_person_zscore=True,
    add_missingness=True,
    add_derived=True,
    use_behavioral_xlsx=True,
    drop_fully_missing_physio=False,
):
    """
    Load the SWELL-KW dataset with proper handling.

    Parameters
    ----------
    per_person_zscore : bool
        Z-score features within each participant (recommended).
    add_missingness : bool
        Add binary missingness indicators for physiology features.
    add_derived : bool
        Add derived ratio features (CharactersRatio, ErrorKeyRatio).
    use_behavioral_xlsx : bool
        Use the master Behavioral Features xlsx (recommended).
    drop_fully_missing_physio : bool
        Drop participants with 100% missing physiology.

    Returns
    -------
    pd.DataFrame with columns: PP, C, Condition, timestamp, features, label
    """
    # Load data
    if use_behavioral_xlsx and os.path.exists(BEHAVIORAL_XLSX):
        print("[data_loader_v2] Loading from Behavioral Features xlsx (master file)")
        df = _load_from_behavioral_xlsx()
    else:
        print("[data_loader_v2] Loading from separate CSVs (Sheet 3)")
        df = _load_from_csvs()

    # Binary label: stressed (T or I) = 1, not stressed (N or R) = 0
    df[LABEL_COL] = df["Condition"].map({"T": 1, "I": 1, "N": 0, "R": 0})
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Drop participants with 100% missing physiology if requested
    if drop_fully_missing_physio:
        before = df["PP"].nunique()
        pp_physio_valid = (
            df.groupby("PP")[PHYSIOLOGY_FEATURES]
            .apply(lambda g: g.notna().any().any())
        )
        valid_pp = pp_physio_valid[pp_physio_valid].index
        df = df[df["PP"].isin(valid_pp)]
        after = df["PP"].nunique()
        if before != after:
            print(f"[data_loader_v2] Dropped {before - after} participants with 100% missing physiology")

    # Add missingness indicators BEFORE imputation
    if add_missingness:
        df = _add_missingness_indicators(df)

    # Add derived features
    if add_derived:
        df = _add_derived_features(df)

    # Per-person z-scoring (on non-missing values)
    if per_person_zscore:
        normalize_cols = list(PHYSIOLOGY_FEATURES) + list(COMPUTER_FEATURES)
        if add_derived:
            normalize_cols += list(DERIVED_FEATURES)
        df = _per_person_normalize(df, normalize_cols)

    # Report
    all_feat = list(ALL_FEATURES)
    if add_derived:
        all_feat += list(DERIVED_FEATURES)
    if add_missingness:
        all_feat += [f"{c}_missing" for c in PHYSIOLOGY_FEATURES]

    print(f"[data_loader_v2] Dataset: {df.shape[0]} rows, {len(all_feat)} features")
    print(f"[data_loader_v2] Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")
    print(f"[data_loader_v2] Participants: {df['PP'].nunique()}")

    missing = df[all_feat].isna().sum()
    if missing.any():
        print(f"[data_loader_v2] Missing values:\n{missing[missing > 0].to_string()}")

    return df


# ---------------------------------------------------------------------------
# LOSO splitting
# ---------------------------------------------------------------------------
def get_loso_splits(df):
    """Yield (train_idx, test_idx, participant_id) for LOSO CV."""
    logo = LeaveOneGroupOut()
    groups = df[GROUP_COL].values
    X_dummy = np.zeros(len(df))
    y_dummy = np.zeros(len(df))

    for train_idx, test_idx in logo.split(X_dummy, y_dummy, groups):
        pp_id = df.iloc[test_idx][GROUP_COL].iloc[0]
        yield train_idx, test_idx, pp_id
