import pandas as pd

for tgt in ['quadrant', 'valence', 'arousal']:
    try:
        v4 = pd.read_csv(f'results/emotion/wesad_v4/wesad/{tgt}/summary.csv')
        v5 = pd.read_csv(f'results/emotion/wesad_v5/wesad/{tgt}/summary.csv')
    except FileNotFoundError as e:
        print(f'=== {tgt}: missing summary ({e})')
        continue
    v4['model'] = v4.iloc[:, 0]
    v5['model'] = v5.iloc[:, 0]
    # Map v5 _fusion variants back to base names for direct comparison.
    v5['model_base'] = v5['model'].str.replace('_fusion', '', regex=False)
    v4['model_base'] = v4['model']
    m = v4.merge(v5, on='model_base', suffixes=('_v4', '_v5'), how='outer')
    print(f'=== {tgt} (session_cohen_kappa | session_macro_f1) ===')
    for _, r in m.iterrows():
        k4 = r.get('session_cohen_kappa_v4')
        k5 = r.get('session_cohen_kappa_v5')
        f4 = r.get('session_macro_f1_v4')
        f5 = r.get('session_macro_f1_v5')
        s = lambda v: f'{v:.3f}' if pd.notna(v) else '  -  '
        d = lambda a, b: f'{(b - a):+.3f}' if (pd.notna(a) and pd.notna(b)) else '     '
        name = r.get('model_v5') if pd.notna(r.get('model_v5')) else r.get('model_v4')
        print(f'  {name:38s}  k:{s(k4)}->{s(k5)} ({d(k4,k5)})  f1:{s(f4)}->{s(f5)} ({d(f4,f5)})')
    print()
