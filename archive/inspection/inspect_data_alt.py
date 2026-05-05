"""Check alternative data sources we might be missing."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Physiology minute data (MatLab Table) ──────────────────────
print("=" * 60)
print("1. PHYSIOLOGY MINUTE DATA (MatLab Table)")
print("=" * 60)
physio_minute_path = os.path.join(
    BASE, "0_SWELL", "2 - Minute data", "D - Physiology - minute data",
    "Physiology - All available data per minute (MatLabTable - SCL complete).txt"
)
if os.path.exists(physio_minute_path):
    # Try different delimiters
    for sep in ['\t', ',', ';', ' ']:
        try:
            df = pd.read_csv(physio_minute_path, sep=sep, nrows=5)
            if len(df.columns) > 1:
                print(f"Delimiter: {repr(sep)}")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"\nFirst 5 rows:\n{df.head().to_string()}")
                print(f"\ndtypes:\n{df.dtypes}")
                # Now load full
                df_full = pd.read_csv(physio_minute_path, sep=sep)
                print(f"\nFull shape: {df_full.shape}")
                print(f"\ndescribe:\n{df_full.describe().to_string()}")
                break
        except:
            pass

# ── 2. Computer interaction Sheet 3 ──────────────────────────────
print("\n" + "=" * 60)
print("2. COMPUTER INTERACTION SHEET 3 (may have different features)")
print("=" * 60)
sheet3_path = os.path.join(
    BASE, "0_SWELL", "3 - Feature dataset", "per sensor",
    "A - Computer interaction features (Ulog - All Features per minute)-Sheet_3.csv"
)
if os.path.exists(sheet3_path):
    df3 = pd.read_csv(sheet3_path)
    print(f"Shape: {df3.shape}")
    print(f"Columns: {list(df3.columns)}")
    print(f"\nFirst 3 rows:\n{df3.head(3).to_string()}")

# ── 3. Questionnaire data ─────────────────────────────────────────
print("\n" + "=" * 60)
print("3. QUESTIONNAIRE (SUBJECTIVE) DATA")
print("=" * 60)
quest_dir = os.path.join(BASE, "0_SWELL", "0 - Raw data", "0 - Subjective experience - questionnaire data")
if os.path.exists(quest_dir):
    for f in os.listdir(quest_dir):
        fpath = os.path.join(quest_dir, f)
        print(f"\n--- {f} ({os.path.getsize(fpath):,} bytes) ---")
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(fpath)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"\nFirst 5 rows:\n{df.head().to_string()}")
            except:
                try:
                    df = pd.read_csv(fpath, sep=';')
                    print(f"Shape (sep=;): {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
                except Exception as e:
                    print(f"Error: {e}")
        elif f.endswith('.xlsx') or f.endswith('.ods'):
            try:
                df = pd.read_excel(fpath)
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
            except Exception as e:
                print(f"Error: {e}")

# ── 4. Minute data computer interaction txt (detailed per-minute) ─
print("\n" + "=" * 60)
print("4. MINUTE DATA - COMPUTER INTERACTION (sample file)")
print("=" * 60)
minute_comp_dir = os.path.join(BASE, "0_SWELL", "2 - Minute data", "A - Computer interaction - minute data uLog no Content-data")
if os.path.exists(minute_comp_dir):
    sample_file = sorted(os.listdir(minute_comp_dir))[0]
    fpath = os.path.join(minute_comp_dir, sample_file)
    print(f"File: {sample_file}")
    with open(fpath, 'r') as fp:
        content = fp.read()
    print(f"Content (first 2000 chars):\n{content[:2000]}")

# ── 5. Check if physiology CSV has additional data when 
#    we read the original MatLab source ─────────────────────────────
print("\n" + "=" * 60)
print("5. COMPARISON: Feature CSV vs Minute data physiology")
print("=" * 60)
physio_feature = pd.read_csv(os.path.join(
    BASE, "0_SWELL", "3 - Feature dataset", "per sensor",
    "D - Physiology features (HR_HRV_SCL - final).csv"
))


# How many rows per participant have ALL physiology as 999?
for pp in sorted(physio_feature['PP'].unique()):
    sub = physio_feature[physio_feature['PP'] == pp]
    all_999 = ((sub['HR'] == 999) & (sub['RMSSD'] == 999) & (sub['SCL'] == 999)).sum()
    some_999 = ((sub['HR'] == 999) | (sub['RMSSD'] == 999) | (sub['SCL'] == 999)).sum()
    none_999 = ((sub['HR'] != 999) & (sub['RMSSD'] != 999) & (sub['SCL'] != 999)).sum()
    total = len(sub)
    print(f"  {pp}: total={total}, all_999={all_999} ({all_999/total*100:.0f}%), some_999={some_999} ({some_999/total*100:.0f}%), clean={none_999} ({none_999/total*100:.0f}%)")

# ── 6. Check SCL within-person variability by condition ───────────
print("\n" + "=" * 60)
print("6. SCL BY CONDITION (per participant, excluding 999)")
print("=" * 60)
p = physio_feature.copy()
p.loc[p['HR'] == 999, 'HR'] = np.nan
p.loc[p['RMSSD'] == 999, 'RMSSD'] = np.nan
p.loc[p['SCL'] == 999, 'SCL'] = np.nan

for pp in ['PP1', 'PP2', 'PP16', 'PP17']:  # Participants with good data
    sub = p[p['PP'] == pp]
    print(f"\n  {pp}:")
    for cond in ['R', 'N', 'T', 'I']:
        c_sub = sub[sub['Condition'] == cond]['SCL'].dropna()
        if len(c_sub) > 0:
            print(f"    {cond}: mean={c_sub.mean():.1f} ± {c_sub.std():.1f}, n={len(c_sub)}")

print("\nHR BY CONDITION (same participants):")
for pp in ['PP1', 'PP2', 'PP16', 'PP17']:
    sub = p[p['PP'] == pp]
    print(f"\n  {pp}:")
    for cond in ['R', 'N', 'T', 'I']:
        c_sub = sub[sub['Condition'] == cond]['HR'].dropna()
        if len(c_sub) > 0:
            print(f"    {cond}: mean={c_sub.mean():.1f} ± {c_sub.std():.1f}, n={len(c_sub)}")
