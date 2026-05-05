import pandas as pd

df = pd.read_excel('0_SWELL/Behavioral-features - per minute.xlsx', sheet_name='SWELLdata', nrows=20)
print('First 20 columns:', df.columns.tolist()[:20])

cond_cols = [c for c in df.columns if 'condit' in c.lower() or 'cond' in c.lower() or c == 'Conditie' or c == 'C']
print('Condition columns:', cond_cols)

if cond_cols:
    print(df[['PP'] + cond_cols].head(20))

# Also check unique PP and condition combos
df_full = pd.read_excel('0_SWELL/Behavioral-features - per minute.xlsx', sheet_name='SWELLdata')
if cond_cols:
    mapping = df_full.groupby('PP')[cond_cols[0]].unique()
    print('\nPP -> Conditions:')
    for pp, conds in mapping.items():
        print(f'  PP{pp}: {conds}')
