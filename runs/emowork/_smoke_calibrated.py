"""Smoke-test the calibration helper without running training."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from empathic.data.emowork import load_emowork
from empathic.data.unified import _bundle_from_emowork
from train_calibrated import _calibrate_in_place

t0 = time.time()
print("[smoke] loading raw EmoWork (baseline_correct=False, norm=none) ...", flush=True)
data = load_emowork(quick=False, normalization="none", baseline_correct=False, verbose=False)
b = _bundle_from_emowork(data)
n_base = 0 if data.baseline_samples is None else len(data.baseline_samples)
print(
    f"[smoke] ready in {time.time()-t0:.1f}s  c-samples={len(b.samples)}  "
    f"subjects={b.samples['subject_id'].nunique()}  baseline_windows={n_base}",
    flush=True,
)

pre_mean = float(b.samples[b.feature_cols].abs().mean().mean())
pre_seq_mean = float(abs(b.sequences).mean())
print(f"[smoke] pre   |feat|={pre_mean:.3f}  |seq|={pre_seq_mean:.3f}", flush=True)

_calibrate_in_place(b)

post_mean = float(b.samples[b.feature_cols].abs().mean().mean())
post_seq_mean = float(abs(b.sequences).mean())
print(f"[smoke] post  |feat|={post_mean:.3f}  |seq|={post_seq_mean:.3f}", flush=True)

df = b.samples
X = df[b.feature_cols].to_numpy()
print(
    f"[smoke] c-session post-norm: mean={X.mean():+.4f}  std={X.std():.4f}  "
    f"(std need not be 1 since reference is rest, not c-sessions)",
    flush=True,
)
