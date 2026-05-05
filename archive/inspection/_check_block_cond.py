import pandas as pd

df = pd.read_excel('0_SWELL/Behavioral-features - per minute.xlsx', sheet_name='SWELLdata')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()[:10]}')
print(f'\nPP dtype: {df["PP"].dtype}, sample: {df["PP"].unique()[:5]}')
print(f'Blok dtype: {df["Blok"].dtype}, unique: {sorted(df["Blok"].unique())}')
print(f'Condition dtype: {df["Condition"].dtype}, unique: {sorted(df["Condition"].unique())}')

# Block-Condition mapping per participant
print('\nBlock-Condition mapping:')
for pp in sorted(df['PP'].unique(), key=lambda x: int(x.replace('PP', ''))):
    pp_data = df[df['PP'] == pp]
    blok_cond = pp_data.groupby('Blok')['Condition'].unique()
    mapping = {b: c.tolist() for b, c in blok_cond.items()}
    counts = pp_data.groupby(['Blok', 'Condition']).size()
    detail = ', '.join([f'B{b}:{c}({n}min)' for (b, c), n in counts.items()])
    print(f'  {pp}: {detail}')

# How many minutes per condition per PP
print('\nMinutes per condition:')
summary = df.groupby(['PP', 'Condition']).size().unstack(fill_value=0)
print(summary)
