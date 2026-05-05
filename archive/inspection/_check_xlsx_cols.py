import pandas as pd
import numpy as np

df = pd.read_excel('0_SWELL/Behavioral-features - per minute.xlsx', sheet_name='SWELLdata')
print(f'Shape: {df.shape}')
print(f'\nAll columns ({len(df.columns)}):')
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    n_valid = df[col].notna().sum()
    n_unique = df[col].nunique()
    print(f'  {i:3d}. {col:<40s}  {str(dtype):<10s}  valid={n_valid}/{len(df)}  unique={n_unique}')
