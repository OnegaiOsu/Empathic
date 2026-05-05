"""Smoke-test that the unified DatasetBundle wrapper handles EmoWork."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from empathic.data.unified import build_bundles

bundles = build_bundles(["emowork"], quick=True, verbose=True)
b = bundles["emowork"]
print()
print(f"name={b.name}")
print(f"samples={len(b.samples)}  feature_cols={len(b.feature_cols)}")
print(f"sequences shape={b.sequences.shape}")
print(f"native_labels={b.native_labels}")
print(f"quadrant_labels={b.quadrant_labels}")
print(f"quadrant_target counts: {dict(zip(*[x.tolist() for x in __import__('numpy').unique(b.quadrant_target, return_counts=True)]))}")
print(f"stress shape={None if b.stress is None else b.stress.shape}; sample={None if b.stress is None else b.stress[:5].tolist()}")
print(f"session_key sample={b.session_key[:5].tolist()}")
print(f"subjects={sorted(set(b.subject_ids))}")
