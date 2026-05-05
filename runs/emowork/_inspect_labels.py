import os
import pandas as pd

sids = [1, 5, 10, 15, 20, 25, 30]
sess = ["b1", "c1", "c2", "c3"]
root = "Dataset/EmoWork_v2/EmoWorker_v2/CSV/LABEL_csv"
for sid in sids:
    for s in sess:
        p = f"{root}/{sid}/{s}/arousal.csv"
        if not os.path.exists(p):
            continue
        a = pd.read_csv(p)["arousal"]
        v = pd.read_csv(f"{root}/{sid}/{s}/valence.csv")["valence"]
        st = pd.read_csv(f"{root}/{sid}/{s}/stress.csv")["stress"]
        sup = pd.read_csv(f"{root}/{sid}/{s}/suppression.csv")["suppression"]
        print(
            f"sid={sid:>2} {s} "
            f"a[{a.min():.1f},{a.max():.1f}] m={a.mean():.2f} u={a.nunique()}  "
            f"v[{v.min():.1f},{v.max():.1f}] m={v.mean():.2f} u={v.nunique()}  "
            f"s[{st.min():.1f},{st.max():.1f}] m={st.mean():.2f} u={st.nunique()}  "
            f"sup u={sup.nunique()}  n={len(a)}"
        )
