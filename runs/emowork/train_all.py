"""Train all four EmoWork V2 targets with the full classical+deep+fusion stack.

Mirrors the WESAD v5 protocol: classical (RF/LR/XGB) + deep (Conformer/TinyTCN/
BiLSTM/CNN1D) with late-fusion variants and a temperature-calibrated ensemble,
all under leave-one-subject-out (LOSO).

Usage::

    .venv\\Scripts\\python.exe runs\\emowork\\train_all.py

Each target writes to ``results/emotion/emowork/emowork/<target>/`` with
summary.csv, per-subject metrics, confusion matrices and the comparison plot.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from empathic.data.unified import build_bundles
from empathic.training import run_experiment


RESULTS_ROOT = os.path.join("results", "emotion", "emowork")
ENSEMBLE = ("RandomForest", "LogisticRegression", "XGBoost",
            "conformer_fusion", "tiny_tcn_fusion", "multistream_fusion",
            "DANN_Conformer", "TSTCC")
# quadrant already completed in prior run; resume on remaining targets.
# Restore ("quadrant", "valence", "arousal", "stress") to run all.
TARGETS = ("valence", "arousal", "stress")
DEEP_ARCHS = ["conformer", "tiny_tcn", "bilstm", "cnn1d", "multistream",
              "dann", "tstcc"]


def main() -> None:
    print("[emowork/train_all] loading bundle ...", flush=True)
    t0 = time.time()
    bundles = build_bundles(["emowork"], quick=False, verbose=True)
    bundle = bundles["emowork"]
    print(f"[emowork/train_all] bundle ready in {time.time()-t0:.1f}s "
          f"(samples={len(bundle.samples)}, subjects={bundle.samples['subject_id'].nunique()})", flush=True)

    for target in TARGETS:
        print(f"\n{'#'*70}\n# TARGET = {target}\n{'#'*70}", flush=True)
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
