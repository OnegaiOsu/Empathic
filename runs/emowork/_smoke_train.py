"""Quick LOSO smoke test for multistream + fusion path."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from empathic.data.unified import build_bundles
from empathic.training import run_experiment

bundle = build_bundles(["emowork"], quick=True, verbose=False)["emowork"]
print(f"bundle samples={len(bundle.samples)} subjects={bundle.samples['subject_id'].nunique()}")

# Tiny config: just multistream + 1 LOSO fold (4 subjects -> 4 folds; small)
results = run_experiment(
    bundle,
    target_kind="valence",
    seed=42,
    use_gpu=True,
    include_classical=False,
    include_deep=True,
    deep_arch=["multistream"],
    fusion=True,
    mixup_alpha=0.0,
    ensemble_members=(),
    results_root=os.path.join("results", "emotion", "emowork", "_smoke"),
    verbose=True,
)
for n, r in results.items():
    print(f"{n}: macro_f1={r.overall.macro_f1:.3f} acc={r.overall.accuracy:.3f}")
