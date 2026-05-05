"""Inspect raw data files to check for reading issues."""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "0_SWELL", "3 - Feature dataset", "per sensor")

# ── 1. Physiology CSV ──────────────────────────────────────────────
physio_path = os.path.join(DATA_DIR, "D - Physiology features (HR_HRV_SCL - final).csv")
p = pd.read_csv(physio_path)
print("=" * 60)
print("PHYSIOLOGY CSV")
print("=" * 60)
print(f"Shape: {p.shape}")
print(f"Columns: {list(p.columns)}")
print(f"\ndtypes:\n{p.dtypes}")
print(f"\nFirst 8 rows:\n{p.head(8).to_string()}")
print(f"\nLast 5 rows:\n{p.tail(5).to_string()}")
print(f"\nUnique PP: {sorted(p['PP'].unique())}")
print(f"Unique Condition: {sorted(p['Condition'].unique())}")
print(f"Unique C: {sorted(p['C'].unique())}")
print(f"\nTimestamp sample: {p['timestamp'].head(10).tolist()}")
print(f"Timestamp dtype: {p['timestamp'].dtype}")

for col in ["HR", "RMSSD", "SCL"]:
    print(f"\n--- {col} ---")
    print(f"dtype: {p[col].dtype}")
    print(f"describe:\n{p[col].describe()}")
    # Check for sentinel / suspicious values
    vals = pd.to_numeric(p[col], errors="coerce")
    print(f"NaN after coerce: {vals.isna().sum()}")
    print(f"Value == 999: {(vals == 999).sum()}")
    print(f"Value == 0: {(vals == 0).sum()}")
    print(f"Value < 0: {(vals < 0).sum()}")
    # Sample non-NaN non-999 values
    clean = vals[(vals != 999) & vals.notna()]
    if len(clean) > 0:
        print(f"Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
        print(f"Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")

# ── 2. Computer interaction CSV ────────────────────────────────────
comp_path = os.path.join(DATA_DIR, "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv")
c = pd.read_csv(comp_path)
print("\n" + "=" * 60)
print("COMPUTER INTERACTION CSV")
print("=" * 60)
print(f"Shape: {c.shape}")
print(f"Columns: {list(c.columns)}")
print(f"\ndtypes:\n{c.dtypes}")
print(f"\nFirst 8 rows:\n{c.head(8).to_string()}")
print(f"\nUnique PP: {sorted(c['PP'].unique())}")
print(f"Unique Condition: {sorted(c['Condition'].unique())}")
if "Blok" in c.columns:
    print(f"Unique Blok: {sorted(c['Blok'].unique())}")
elif "C" in c.columns:
    print(f"Unique C: {sorted(c['C'].unique())}")
print(f"\nTimestamp sample: {c['timestamp'].head(10).tolist()}")
print(f"Timestamp dtype: {c['timestamp'].dtype}")

# ── 3. Merge check ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MERGE DIAGNOSTICS")
print("=" * 60)
c2 = c.rename(columns={"Blok": "C"}) if "Blok" in c.columns else c.copy()
print(f"Physiology rows: {len(p)}")
print(f"Computer rows: {len(c2)}")

# Check key intersection
p_keys = set(zip(p["PP"], p["C"], p["Condition"], p["timestamp"]))
c_keys = set(zip(c2["PP"], c2["C"], c2["Condition"], c2["timestamp"]))
print(f"\nPhysiology unique keys: {len(p_keys)}")
print(f"Computer unique keys: {len(c_keys)}")
print(f"Intersection: {len(p_keys & c_keys)}")
print(f"Only in physiology: {len(p_keys - c_keys)}")
print(f"Only in computer: {len(c_keys - p_keys)}")

# inner merge
merged = pd.merge(p, c2, on=["PP", "C", "Condition", "timestamp"], how="inner")
print(f"\nInner merge rows: {len(merged)}")

# outer merge
merged_outer = pd.merge(p, c2, on=["PP", "C", "Condition", "timestamp"], how="outer")
print(f"Outer merge rows: {len(merged_outer)}")

# left/right
merged_left = pd.merge(p, c2, on=["PP", "C", "Condition", "timestamp"], how="left")
merged_right = pd.merge(p, c2, on=["PP", "C", "Condition", "timestamp"], how="right")
print(f"Left merge rows (physio left): {len(merged_left)}")
print(f"Right merge rows (computer left): {len(merged_right)}")

# What are we losing?
if len(c_keys - p_keys) > 0:
    lost = c2[~c2.set_index(["PP", "C", "Condition", "timestamp"]).index.isin(
        p.set_index(["PP", "C", "Condition", "timestamp"]).index)]
    lost_by_pp_cond = lost.groupby(["PP", "Condition"]).size().reset_index(name="count")
    print(f"\nComputer rows lost in inner merge by (PP, Condition):")
    print(lost_by_pp_cond.to_string())

if len(p_keys - c_keys) > 0:
    lost = p[~p.set_index(["PP", "C", "Condition", "timestamp"]).index.isin(
        c2.set_index(["PP", "C", "Condition", "timestamp"]).index)]
    lost_by_pp_cond = lost.groupby(["PP", "Condition"]).size().reset_index(name="count")
    print(f"\nPhysiology rows lost in inner merge by (PP, Condition):")
    print(lost_by_pp_cond.to_string())

# ── 4. Check per-participant data completeness after merge ─────────
print("\n" + "=" * 60)
print("PER-PARTICIPANT DATA AFTER INNER MERGE")
print("=" * 60)
merged["label"] = merged["Condition"].map({"T": 1, "I": 1, "N": 0, "R": 0})
for pp in sorted(merged["PP"].unique()):
    sub = merged[merged["PP"] == pp]
    n_rows = len(sub)
    n_conds = sub["Condition"].nunique()
    conds = sorted(sub["Condition"].unique())
    hr_miss = sub["HR"].isna().sum() + (sub["HR"] == 999).sum()
    scl_miss = sub["SCL"].isna().sum() + (sub["SCL"] == 999).sum()
    label_dist = sub["label"].value_counts().to_dict()
    print(f"  {pp}: {n_rows:3d} rows, conditions={conds}, labels={label_dist}, HR_miss={hr_miss}, SCL_miss={scl_miss}")

# ── 5. Check the content features CSV ─────────────────────────────
content_path = os.path.join(BASE, "0_SWELL", "Content-features - Labeled-EventBlocks.csv")
if os.path.exists(content_path):
    print("\n" + "=" * 60)
    print("CONTENT-FEATURES CSV (Event Blocks)")
    print("=" * 60)
    cf = pd.read_csv(content_path)
    print(f"Shape: {cf.shape}")
    print(f"Columns: {list(cf.columns)}")
    print(f"\nFirst 5 rows:\n{cf.head().to_string()}")

# ── 6. Check other data directories ───────────────────────────────
print("\n" + "=" * 60)
print("ALL FILES IN FEATURE DATASET DIR")
print("=" * 60)
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        fpath = os.path.join(root, f)
        fsize = os.path.getsize(fpath)
        print(f"  {os.path.relpath(fpath, DATA_DIR)} ({fsize:,} bytes)")

# Also check minute data dir
minute_dir = os.path.join(BASE, "0_SWELL", "2 - Minute data")
if os.path.exists(minute_dir):
    print(f"\nALL FILES IN MINUTE DATA DIR:")
    for root, dirs, files in os.walk(minute_dir):
        for f in files:
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, minute_dir)} ({fsize:,} bytes)")
