"""Deep-dive into data anomalies found in first inspection."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "0_SWELL", "3 - Feature dataset", "per sensor")

# ── 1. SnMouseDistance string issue ────────────────────────────────
comp_path = os.path.join(DATA_DIR, "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv")
c = pd.read_csv(comp_path)
print("=" * 60)
print("1. SnMouseDistance STRING ANALYSIS")
print("=" * 60)
print(f"dtype: {c['SnMouseDistance'].dtype}")
# Show some sample values
print(f"Sample values:\n{c['SnMouseDistance'].head(20).tolist()}")
# Try converting
converted = pd.to_numeric(c['SnMouseDistance'], errors='coerce')
print(f"NaN after to_numeric: {converted.isna().sum()}")
# Values that couldn't be converted
bad_mask = converted.isna() & c['SnMouseDistance'].notna()
if bad_mask.any():
    print(f"Non-numeric SnMouseDistance values:")
    print(c.loc[bad_mask, ['PP', 'Blok', 'Condition', 'timestamp', 'SnMouseDistance']].to_string())
else:
    print("All SnMouseDistance values are numeric after conversion")
print(f"Converted describe:\n{converted.describe()}")

# ── 2. SCL value distribution — is 999 the only sentinel? ─────────
physio_path = os.path.join(DATA_DIR, "D - Physiology features (HR_HRV_SCL - final).csv")
p = pd.read_csv(physio_path)
print("\n" + "=" * 60)
print("2. SCL DETAILED ANALYSIS")
print("=" * 60)
scl = p['SCL'].copy()
scl_clean = scl[scl != 999]
print(f"SCL (excluding 999) describe:\n{scl_clean.describe()}")
print(f"\nSCL percentiles (excluding 999):")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {pct}th: {scl_clean.quantile(pct/100):.4f}")
print(f"\nSCL values > 500 (excluding 999): {(scl_clean > 500).sum()}")
print(f"SCL values > 800: {(scl_clean > 800).sum()}")
# Show distribution by PP
print(f"\nSCL mean by PP (excluding 999):")
for pp in sorted(p['PP'].unique()):
    sub = p[p['PP'] == pp]['SCL']
    sub_clean = sub[sub != 999]
    if len(sub_clean) > 0:
        print(f"  {pp}: mean={sub_clean.mean():.1f}, std={sub_clean.std():.1f}, n={len(sub_clean)}")

# ── 3. HR: Check if 999 is the ONLY sentinel ──────────────────────
print("\n" + "=" * 60)
print("3. HR DETAILED ANALYSIS")
print("=" * 60)
hr = p['HR'].copy()
hr_clean = hr[hr != 999]
print(f"HR (excluding 999) describe:\n{hr_clean.describe()}")
# any HR=0 ?
print(f"HR == 0: {(hr == 0).sum()}")
# Histogram bins
print(f"\nHR distribution (excluding 999):")
bins = [50, 60, 70, 80, 90, 100, 110, 120]
for i in range(len(bins)-1):
    count = ((hr_clean >= bins[i]) & (hr_clean < bins[i+1])).sum()
    print(f"  [{bins[i]}, {bins[i+1]}): {count}")

# ── 4. RMSSD check for 999 as sentinel ────────────────────────────
print("\n" + "=" * 60)
print("4. RMSSD DETAILED ANALYSIS")
print("=" * 60)
rmssd = p['RMSSD'].copy()
rmssd_clean = rmssd[rmssd != 999]
print(f"RMSSD (excluding 999) describe:\n{rmssd_clean.describe()}")
print(f"RMSSD > 0.5: {(rmssd_clean > 0.5).sum()}")

# ── 5. Check if HR and RMSSD are ALWAYS missing together ─────────
print("\n" + "=" * 60)
print("5. MISSING DATA PATTERNS")
print("=" * 60)
hr_miss = p['HR'] == 999
rmssd_miss = p['RMSSD'] == 999
scl_miss = p['SCL'] == 999
print(f"HR=999: {hr_miss.sum()}")
print(f"RMSSD=999: {rmssd_miss.sum()}")
print(f"SCL=999: {scl_miss.sum()}")
print(f"Both HR and RMSSD = 999: {(hr_miss & rmssd_miss).sum()}")
print(f"HR=999 but RMSSD!=999: {(hr_miss & ~rmssd_miss).sum()}")
print(f"RMSSD=999 but HR!=999: {(~hr_miss & rmssd_miss).sum()}")
print(f"All three = 999: {(hr_miss & rmssd_miss & scl_miss).sum()}")
print(f"SCL=999 but HR!=999: {(scl_miss & ~hr_miss).sum()}")

# Per-condition missingness
print(f"\nMissingness by Condition:")
for cond in ['R', 'N', 'T', 'I']:
    sub = p[p['Condition'] == cond]
    print(f"  {cond}: total={len(sub)}, HR_miss={int((sub['HR']==999).sum())} ({(sub['HR']==999).mean()*100:.1f}%), SCL_miss={int((sub['SCL']==999).sum())} ({(sub['SCL']==999).mean()*100:.1f}%)")

# ── 6. Check the "Unnamed" columns — they might have annotations ──
print("\n" + "=" * 60)
print("6. UNNAMED COLUMNS IN PHYSIOLOGY CSV")
print("=" * 60)
unnamed_cols = [c for c in p.columns if c.startswith("Unnamed")]
for col in unnamed_cols:
    non_null = p[col].dropna()
    if len(non_null) > 0:
        print(f"\n{col}: {len(non_null)} non-null values")
        print(f"  Unique: {non_null.unique()}")
        # Show context
        print(f"  Rows with this value:")
        print(p[p[col].notna()][['PP', 'C', 'Condition', 'timestamp', 'HR', 'SCL', col]].head(10).to_string())

# ── 7. Content features file ──────────────────────────────────────
print("\n" + "=" * 60)
print("7. CONTENT-FEATURES CSV")
print("=" * 60)
content_path = os.path.join(BASE, "0_SWELL", "Content-features - Labeled-EventBlocks.csv")
try:
    cf = pd.read_csv(content_path, sep=';')  # try semicolon
    print(f"Shape: {cf.shape}")
    print(f"Columns: {list(cf.columns)}")
    print(f"\nFirst 5 rows:\n{cf.head().to_string()}")
except:
    try:
        cf = pd.read_csv(content_path, on_bad_lines='skip')
        print(f"Shape (skip bad lines): {cf.shape}")
        print(f"Columns: {list(cf.columns)}")
        print(f"\nFirst 5 rows:\n{cf.head().to_string()}")
    except Exception as e:
        print(f"Error: {e}")

# ── 8. List ALL available data files ──────────────────────────────
print("\n" + "=" * 60)
print("8. ALL AVAILABLE DATA FILES")
print("=" * 60)
swell_dir = os.path.join(BASE, "0_SWELL")
for root, dirs, files in os.walk(swell_dir):
    # Skip raw data subdirectories with many small files
    rel = os.path.relpath(root, swell_dir)
    if any(x in rel for x in ['Raw data', 'Sequential data']):
        if files:
            print(f"  {rel}/ [{len(files)} files]")
        continue
    for f in files:
        fpath = os.path.join(root, f)
        fsize = os.path.getsize(fpath)
        print(f"  {os.path.relpath(fpath, swell_dir)} ({fsize:,} bytes)")

# ── 9. Check timestamp alignment between CSVs ─────────────────────
print("\n" + "=" * 60)
print("9. TIMESTAMP FORMAT & ALIGNMENT CHECK")
print("=" * 60)
# Check if timestamp format is consistent
p_ts = p[p['PP'] == 'PP1']['timestamp'].tolist()[:15]
c2 = c.rename(columns={"Blok": "C"})
c_ts = c2[c2['PP'] == 'PP1']['timestamp'].tolist()[:15]
print(f"Physio timestamps (PP1, first 15):  {p_ts}")
print(f"Computer timestamps (PP1, first 15): {c_ts}")

# Check for gap in timestamps (should be 1 minute = T+100000?)
p_ts_pp1 = sorted(p[p['PP'] == 'PP1']['timestamp'].values)
gaps = []
for i in range(1, len(p_ts_pp1)):
    t1 = int(p_ts_pp1[i-1].replace('T', ''))
    t2 = int(p_ts_pp1[i].replace('T', ''))
    gaps.append(t2 - t1)
print(f"\nTimestamp gaps for PP1: unique values = {sorted(set(gaps))}")
print(f"  Most common gap: {max(set(gaps), key=gaps.count)}")
# Condition transitions
pp1_data = p[p['PP'] == 'PP1'][['timestamp', 'Condition', 'C']].reset_index(drop=True)
for i in range(1, len(pp1_data)):
    if pp1_data.loc[i, 'Condition'] != pp1_data.loc[i-1, 'Condition']:
        print(f"  Condition change: {pp1_data.loc[i-1, 'Condition']} -> {pp1_data.loc[i, 'Condition']} at row {i}, ts={pp1_data.loc[i, 'timestamp']}")
