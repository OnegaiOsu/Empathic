import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from empathic.data.wesad import load_wesad

d = load_wesad(verbose=False)
s = d.sequences
print("shape:", s.shape, "dtype:", s.dtype)
print(f"global mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
print()
print(f"{'channel':8s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}")
for i, c in enumerate(d.seq_channels):
    x = s[:, :, i]
    print(f"{c:8s} {x.mean():>12.4f} {x.std():>12.4f} {x.min():>12.4f} {x.max():>12.4f}")

# Per-subject per-channel stats spread (subject shift on raw signals)
print("\nPer-subject channel means (showing range across subjects):")
df = d.samples
subjects = df["subject_id"].values
for i, c in enumerate(d.seq_channels):
    means = []
    for sid in np.unique(subjects):
        mask = subjects == sid
        means.append(s[mask, :, i].mean())
    means = np.array(means)
    print(f"{c:8s} subj-mean range=[{means.min():>10.3f}, {means.max():>10.3f}]  spread={means.max()-means.min():>10.3f}")
