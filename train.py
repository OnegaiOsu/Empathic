"""Command-line entry point for the Empathic emotion pipeline.

Examples
--------

    python train.py --datasets emosurv wesad --target quadrant
    python train.py --datasets wesad --target native --quick
    python train.py --datasets emosurv --no-deep
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Allow running without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from empathic.config import DEVICE, RESULTS_DIR, TrainDefaults  # noqa: E402
from empathic.data import build_bundles  # noqa: E402
from empathic.training import run_experiment  # noqa: E402
from empathic.utils import ensure_dir, log  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOSO emotion recognition over EmoSurv + WESAD.")
    p.add_argument("--datasets", nargs="+", choices=["emosurv", "wesad"], default=["emosurv", "wesad"])
    p.add_argument("--target", choices=["quadrant", "native", "valence", "arousal"], default="quadrant",
                   help="Unified circumplex quadrants, dataset-native labels, or a binary "
                        "valence/arousal axis collapsed from the quadrants.")
    p.add_argument("--quick", action="store_true", help="Load a subset of subjects for smoke tests.")
    p.add_argument("--no-deep", dest="include_deep", action="store_false", help="Skip the deep model.")
    p.add_argument("--no-classical", dest="include_classical", action="store_false",
                   help="Skip the classical models (useful when adding new deep archs to an existing summary).")
    p.add_argument("--deep-arch", nargs="+",
                   choices=["conformer", "tiny_tcn", "bilstm", "cnn1d", "tstcc", "dann"],
                   default=["conformer"],
                   help="One or more deep architectures to evaluate in the same sweep "
                        "(e.g. --deep-arch conformer tiny_tcn bilstm cnn1d tstcc dann).")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="MixUp Beta(alpha,alpha) strength for deep training (0 disables).")
    p.add_argument("--fusion", action="store_true",
                   help="Late-fuse hand-engineered tabular features into the deep model head "
                        "(applies to conformer/tiny_tcn/bilstm/cnn1d).")
    p.add_argument("--ensemble", nargs="+", default=None,
                   help="Names of models to soft-vote into an Ensemble run "
                        "(e.g. --ensemble RandomForest XGBoost Conformer). "
                        "Uses fold-aligned probabilities; matched case-insensitively.")
    p.add_argument("--augment-tabular", choices=["none", "balance", "full"], default="balance")
    p.add_argument("--augment-sequences", choices=["none", "balance", "full"], default="full")
    p.add_argument("--emosurv-norm", choices=["none", "zscore", "robust"], default="robust")
    p.add_argument("--wesad-norm", choices=["none", "zscore"], default="zscore")
    p.add_argument("--emosurv-neutral", choices=["merge", "drop", "separate", "baseline"], default="merge",
                   help="How to treat EmoSurv's Neutral class: merge into HVLA (default), "
                        "drop entirely, keep as its own 5th quadrant, or use Neutral windows "
                        "as per-subject calibration and classify residuals.")
    p.add_argument("--emosurv-window", type=int, default=None,
                   help="Number of consecutive key events per EmoSurv window (default 35).")
    p.add_argument("--emosurv-stride", type=int, default=None,
                   help="Stride in events between consecutive EmoSurv windows (default 20).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=TrainDefaults.deep_epochs)
    p.add_argument("--batch-size", type=int, default=TrainDefaults.deep_batch_size)
    p.add_argument("--lr", type=float, default=TrainDefaults.deep_lr)
    p.add_argument("--cpu", action="store_true", help="Disable GPU (XGBoost and torch).")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    use_gpu = (not args.cpu) and (DEVICE.type == "cuda")
    log(f"device={DEVICE} (GPU used by deep model: {use_gpu})")
    ensure_dir(args.results_dir)

    bundles = build_bundles(
        args.datasets,
        quick=args.quick,
        emosurv_normalization=args.emosurv_norm,
        wesad_normalization=args.wesad_norm,
        emosurv_neutral_policy=args.emosurv_neutral,
        emosurv_window_size=args.emosurv_window,
        emosurv_stride=args.emosurv_stride,
    )

    cfg = TrainDefaults(
        seed=args.seed,
        deep_epochs=args.epochs,
        deep_batch_size=args.batch_size,
        deep_lr=args.lr,
    )

    for name, bundle in bundles.items():
        run_experiment(
            bundle,
            target_kind=args.target,
            seed=args.seed,
            use_gpu=use_gpu,
            augment_tabular_mode=args.augment_tabular,
            augment_sequences_mode=args.augment_sequences,
            include_deep=args.include_deep,
            include_classical=args.include_classical,
            deep_arch=args.deep_arch,
            mixup_alpha=args.mixup_alpha,
            fusion=args.fusion,
            ensemble_members=args.ensemble,
            train_cfg=cfg,
            results_root=args.results_dir,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
