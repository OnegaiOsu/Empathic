import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from empathic.training import _build_deep_model

x = torch.randn(2, 240, 8)
for arch in ('conformer', 'tiny_tcn', 'bilstm', 'cnn1d'):
    m = _build_deep_model(arch, 8, 4, 240)
    n = sum(p.numel() for p in m.parameters())
    out = m(x)
    print(f"{arch:10s} params={n:>8d}  out={tuple(out.shape)}")
