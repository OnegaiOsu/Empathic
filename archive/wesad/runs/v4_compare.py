import pandas as pd

for tgt in ['quadrant', 'valence', 'arousal']:
    v3 = pd.read_csv(f'results/emotion/wesad_full/wesad/{tgt}/summary.csv')
    v4 = pd.read_csv(f'results/emotion/wesad_v4/wesad/{tgt}/summary.csv')
    v3['model'] = v3.iloc[:, 0]
    v4['model'] = v4.iloc[:, 0]
    m = v3.merge(v4, on='model', suffixes=('_v3', '_v4'), how='outer')
    print(f'=== {tgt} (session_cohen_kappa | session_macro_f1) ===')
    for _, r in m.iterrows():
        k3 = r.get('session_cohen_kappa_v3')
        k4 = r.get('session_cohen_kappa_v4')
        f3 = r.get('session_macro_f1_v3')
        f4 = r.get('session_macro_f1_v4')
        s = lambda v: f'{v:.3f}' if pd.notna(v) else '  -  '
        d = lambda a, b: f'{(b - a):+.3f}' if (pd.notna(a) and pd.notna(b)) else '     '
        print(f'  {r.model:34s}  k:{s(k3)}->{s(k4)} ({d(k3,k4)})  f1:{s(f3)}->{s(f4)} ({d(f3,f4)})')
    print()
