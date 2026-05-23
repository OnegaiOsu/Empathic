import pandas as pd
for t in ['quadrant','valence','arousal']:
    df = pd.read_csv(f'results/emotion/wesad_full/wesad/{t}/summary.csv')
    print('===',t,'===')
    df = df.rename(columns={'Unnamed: 0':'model'})
    print(df[['model','session_cohen_kappa','session_macro_f1','session_balanced_accuracy','session_accuracy']].sort_values('session_cohen_kappa',ascending=False).to_string(index=False))
