"""Quick smoke test for the multistream model + new arch dispatch."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch

from empathic.data.unified import build_bundles
from empathic.models import MultiStream, MultiStreamConfig, count_parameters

bundle = build_bundles(["emowork"], quick=True, verbose=False)["emowork"]
print(f"bundle: samples={len(bundle.samples)}  seq={bundle.sequences.shape}")

# Direct forward
cfg = MultiStreamConfig(in_channels=bundle.sequences.shape[-1], num_classes=4)
m = MultiStream(cfg)
print(f"multistream params: {count_parameters(m):,}")
print(f"  active groups: {list(m.active_groups.keys())}")
print(f"  total embed_dim: {m.embed_dim}")
x = torch.from_numpy(bundle.sequences[:8]).float()
y = m(x)
print(f"  forward: {x.shape} -> {y.shape}")

# Confirm DANN and TSTCC dispatch via training.py
from empathic.training import _build_deep_model
for a in ("conformer", "tiny_tcn", "bilstm", "cnn1d", "multistream"):
    mdl = _build_deep_model(a, in_channels=12, num_classes=4, seq_len=240)
    print(f"  {a}: params={count_parameters(mdl):,}  embed_dim={getattr(mdl,'embed_dim','-')}")
print("OK")
