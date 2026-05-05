"""Quick smoke-test of the EmoWork loader.

Loads only the first 4 subjects with normalisation off and prints a few
sanity-check statistics. Usage::

    .venv\\Scripts\\python.exe runs\\emowork\\_smoke_loader.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from empathic.data.emowork import load_emowork


def main() -> None:
    data = load_emowork(quick=True, normalization="zscore", verbose=True)
    df = data.samples
    print()
    print(f"sequences: shape={data.sequences.shape}  dtype={data.sequences.dtype}")
    print(f"non-finite cells in sequences: {(~np.isfinite(data.sequences)).sum()}")
    print(f"feature_cols ({len(data.feature_cols)}): {data.feature_cols[:10]} ... {data.feature_cols[-5:]}")
    print(f"subjects: {sorted(df['subject_id'].unique().tolist())}")
    print(f"sessions per subject:")
    print(df.groupby('subject_id')['session'].value_counts().to_string())
    print()
    print("quadrant counts:")
    print(df['quadrant_target'].value_counts().to_string())
    print()
    print("stress / suppression class counts (-1 = missing):")
    print(df['stress'].value_counts().to_string())
    print(df['suppression'].value_counts().to_string())
    print()
    print(df[["arousal_cont", "valence_cont", "stress_cont", "suppression_cont"]].describe().to_string())


if __name__ == "__main__":
    main()
