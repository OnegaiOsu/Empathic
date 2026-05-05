"""Check questionnaire data + fix MatLab physio reading + Sheet 3."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Questionnaire data (subjective stress scores) ─────────────
print("=" * 60)
print("1. QUESTIONNAIRE BLOK DATA (self-reported stress)")
print("=" * 60)
quest_path = os.path.join(
    BASE, "0_SWELL", "0 - Raw data",
    "0 - Subjective experience - questionnaire data",
    "Questionnaire Blok Results_final - performance added.xlsx"
)
df_q = pd.read_excel(quest_path)
print(f"Shape: {df_q.shape}")
print(f"Columns ({len(df_q.columns)}):\n{list(df_q.columns)}")
print(f"\nFirst 3 rows:\n{df_q.head(3).to_string()}")
print(f"\ndtypes:\n{df_q.dtypes}")

# Check for stress-related columns
stress_cols = [c for c in df_q.columns if any(x in c.lower() for x in ['stress', 'nasa', 'arousal', 'valence', 'effort', 'frustrat', 'mental', 'demand', 'performance'])]
print(f"\nStress-related columns: {stress_cols}")
if stress_cols:
    print(f"\nStress columns describe:\n{df_q[stress_cols].describe().to_string()}")

# ── 2. MatLab physiology - read properly ──────────────────────────
print("\n" + "=" * 60)
print("2. MATLAB PHYSIOLOGY FILE - PROPER READING")
print("=" * 60)
physio_minute_path = os.path.join(
    BASE, "0_SWELL", "2 - Minute data", "D - Physiology - minute data",
    "Physiology - All available data per minute (MatLabTable - SCL complete).txt"
)
# Read with header
with open(physio_minute_path, 'r') as f:
    first_lines = [f.readline() for _ in range(10)]
print("First 10 lines:")
for i, line in enumerate(first_lines):
    print(f"  Line {i}: {line.rstrip()}")

# Try reading properly
df_mat = pd.read_csv(physio_minute_path, sep='\t')
print(f"\nShape: {df_mat.shape}")
print(f"Columns: {list(df_mat.columns)}")
print(f"dtypes:\n{df_mat.dtypes}")
# Check for header rows repeated
print(f"\nRows where PP='PP': {(df_mat['PP'] == 'PP').sum()}")
# Remove header rows
df_mat_clean = df_mat[df_mat['PP'] != 'PP'].copy()
for col in ['HR', 'RMSSD', 'SCL']:
    df_mat_clean[col] = pd.to_numeric(df_mat_clean[col], errors='coerce')
df_mat_clean['C'] = pd.to_numeric(df_mat_clean['C'], errors='coerce')
print(f"\nClean shape: {df_mat_clean.shape}")
print(f"HR non-null: {df_mat_clean['HR'].notna().sum()}")
print(f"RMSSD non-null: {df_mat_clean['RMSSD'].notna().sum()}")
print(f"SCL non-null: {df_mat_clean['SCL'].notna().sum()}")
print(f"PP unique: {sorted(df_mat_clean['PP'].unique())}")
# This file does NOT have condition column, so we'd need to map it

# ── 3. Check Sheet 3 vs Sheet 1 of computer interaction ──────────
print("\n" + "=" * 60)
print("3. SHEET 3 vs SHEET 1 COMPARISON")
print("=" * 60)
sheet1_path = os.path.join(BASE, "0_SWELL", "3 - Feature dataset", "per sensor",
    "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv")
sheet3_path = os.path.join(BASE, "0_SWELL", "3 - Feature dataset", "per sensor",
    "A - Computer interaction features (Ulog - All Features per minute)-Sheet_3.csv")
s1 = pd.read_csv(sheet1_path)
s3 = pd.read_csv(sheet3_path)
print(f"Sheet 1: {s1.shape}, Cols: {list(s1.columns)}")
print(f"Sheet 3: {s3.shape}, Cols: {list(s3.columns)}")

# Compare SnMouseDistance
s1_md = pd.to_numeric(s1['SnMouseDistance'], errors='coerce')
s3_md = s3['SnMouseDistance']
print(f"\nSheet 1 SnMouseDistance: dtype={s1['SnMouseDistance'].dtype}, NaN={s1_md.isna().sum()}")
print(f"Sheet 3 SnMouseDistance: dtype={s3['SnMouseDistance'].dtype}, NaN={s3_md.isna().sum()}")
print(f"Sheet 3 SnMouseDistance.1 values (unique): {s3['SnMouseDistance.1'].nunique()}")
print(f"Sheet 3 SnMouseDistance.1 sample: {s3['SnMouseDistance.1'].head(20).tolist()}")

# Are the values the same where both are valid?
merged = pd.merge(s1[['PP', 'Blok', 'timestamp']], s3[['PP', 'Blok', 'timestamp', 'SnMouseDistance']], 
                  on=['PP', 'Blok', 'timestamp'], suffixes=('_s1', '_s3'))
# Sheet 1 has extra row, sheet 3 has extra row
print(f"\nSheet 1 rows: {len(s1)}")
print(f"Sheet 3 rows: {len(s3)}")
# Sheet 3 has Condition?
if 'Condition' in s3.columns:
    print(f"Sheet 3 has Condition: {sorted(s3['Condition'].unique())}")
else:
    print("Sheet 3 does NOT have Condition column")

# ── 4. Behavioral features file ──────────────────────────────────
print("\n" + "=" * 60)
print("4. BEHAVIORAL FEATURES FILE")
print("=" * 60)
behav_path = os.path.join(BASE, "0_SWELL", "Behavioral-features - per minute.xlsx")
if os.path.exists(behav_path):
    try:
        df_b = pd.read_excel(behav_path, nrows=5)
        print(f"Shape (first sheet): {df_b.shape}")
        print(f"Columns ({len(df_b.columns)}):\n{list(df_b.columns)}")
        print(f"\nFirst 3 rows:\n{df_b.head(3).to_string()}")
    except Exception as e:
        print(f"Error: {e}")
    try:
        xl = pd.ExcelFile(behav_path)
        print(f"\nSheet names: {xl.sheet_names}")
    except Exception as e:
        print(f"Error getting sheets: {e}")

# ── 5. SWELL-KW overview ─────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SWELL-KW OVERVIEW FILE")
print("=" * 60)
overview_path = os.path.join(BASE, "0_SWELL", "SWELL-KW - overview available data.xlsx")
if os.path.exists(overview_path):
    try:
        xl = pd.ExcelFile(overview_path)
        print(f"Sheet names: {xl.sheet_names}")
        for sheet in xl.sheet_names[:3]:
            df_o = pd.read_excel(overview_path, sheet_name=sheet, nrows=5)
            print(f"\n--- Sheet: {sheet} ---")
            print(f"Shape: {df_o.shape}")
            print(f"Columns: {list(df_o.columns)}")
            print(f"\n{df_o.head(3).to_string()}")
    except Exception as e:
        print(f"Error: {e}")
