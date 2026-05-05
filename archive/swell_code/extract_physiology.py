"""
Validate extracted .S00 data against known minute-level physiology data.
Then build a complete physiology dataset from all .S00 files.
"""
import os
import numpy as np
import pandas as pd
from tms_reader import tms_read

RAW_DIR = os.path.join(
    "0_SWELL", "0 - Raw data", "D - Physiology - raw data",
    "Mobi signals (raw and filtered)"
)
MINUTE_FILE = os.path.join(
    "0_SWELL", "2 - Minute data", "D - Physiology - minute data",
    "Physiology - All available data per minute (MatLabTable - SCL complete).txt"
)
OUTPUT_FILE = "physiology_from_raw.csv"


def extract_minute_features(sig):
    """
    Extract per-minute HR, RMSSD, SCL from a loaded .S00 signal.
    
    Channel mapping (from analysis):
      Ch0/Ch1: Raw ECG leads (mV)
      Ch2: Continuous HR estimate (BPM)
      Ch3: Beat-by-beat HR (discrete values)
      Ch4: Beat markers (binary 0/1 pulses)
      Ch5: Raw physiological signal (uV)
      Ch6: Skin conductance level (mV) = portiSkinRaw
      Ch7: Heart preprocessed (uV) = portiHeartPre
    """
    fs = sig['fs']
    samples_per_min = fs * 60
    total_samples = sig['total_samples']
    n_minutes = total_samples // samples_per_min
    
    results = []
    
    # Get channel data
    data_hr_cont = sig['data'][2]   # Continuous HR estimate
    data_beats = sig['data'][4]     # Beat markers
    data_scl = sig['data'][6]       # Skin conductance level
    
    # Detect all beats globally first
    beat_indices = None
    if data_beats is not None:
        above = data_beats > 0.5
        rising = np.diff(above.astype(int)) > 0
        beat_indices = np.where(rising)[0]
    
    for m in range(n_minutes):
        start = m * samples_per_min
        end = (m + 1) * samples_per_min
        
        minute_data = {'minute': m + 1}
        
        # ── SCL: simple mean of Ch6 ──
        if data_scl is not None:
            scl_chunk = data_scl[start:end]
            minute_data['SCL'] = np.mean(scl_chunk)
        else:
            minute_data['SCL'] = np.nan
        
        # ── HR from continuous estimate (Ch2) ──
        if data_hr_cont is not None:
            hr_chunk = data_hr_cont[start:end]
            # Filter out zeros and extreme values
            valid_hr = hr_chunk[(hr_chunk > 30) & (hr_chunk < 200)]
            minute_data['HR_ch2'] = np.mean(valid_hr) if len(valid_hr) > fs * 10 else np.nan
        else:
            minute_data['HR_ch2'] = np.nan
        
        # ── HR and RMSSD from beat markers (Ch4) ──
        minute_data['HR_beats'] = np.nan
        minute_data['RMSSD'] = np.nan
        minute_data['n_beats'] = 0
        
        if beat_indices is not None and len(beat_indices) > 0:
            mask = (beat_indices >= start) & (beat_indices < end)
            minute_beats = beat_indices[mask]
            minute_data['n_beats'] = len(minute_beats)
            
            if len(minute_beats) >= 5:  # Need at least 5 beats for reliable estimate
                ibis = np.diff(minute_beats) / fs  # Inter-beat intervals in seconds
                
                # Filter plausible IBIs (30-200 BPM -> 0.3-2.0 sec)
                plausible = ibis[(ibis > 0.3) & (ibis < 2.0)]
                
                if len(plausible) >= 4:
                    # Additional IBI cleaning: remove outliers (>2 SD from median)
                    median_ibi = np.median(plausible)
                    mad = np.median(np.abs(plausible - median_ibi))  # median absolute deviation
                    if mad > 0:
                        clean_mask = np.abs(plausible - median_ibi) < 3 * mad
                        clean_ibis = plausible[clean_mask]
                    else:
                        clean_ibis = plausible
                    
                    if len(clean_ibis) >= 3:
                        minute_data['HR_beats'] = 60.0 / np.mean(clean_ibis)
                        
                        # RMSSD from successive differences
                        ibi_diffs = np.diff(clean_ibis * 1000)  # Convert to ms
                        minute_data['RMSSD'] = np.sqrt(np.mean(ibi_diffs ** 2))
        
        results.append(minute_data)
    
    return results


def parse_pp_and_block(filename):
    """Extract participant number and block from .S00 filename."""
    # Examples: pp1_18-9-2012_c1.S00, pp10_8-10-2012-c1.S00
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    pp_num = int(parts[0].replace('pp', ''))
    
    # Find the block number (c1, c2, c3)
    block = None
    for part in parts:
        for sep in ['-c', '_c']:
            pass
        if part.startswith('c') and len(part) >= 2 and part[1:].isdigit():
            block = int(part[1:])
            break
    
    # Some filenames have the block attached to the date: pp10_8-10-2012-c1.S00
    if block is None:
        for part in parts:
            if '-c' in part:
                idx = part.index('-c')
                block_str = part[idx+2:]
                if block_str.isdigit():
                    block = int(block_str)
                    break
    
    # Special case: pp23_1-11-2012_c2_cont.S00
    if block is None and 'cont' in base:
        # This is a continuation file, skip or mark specially
        block = -1
    
    return pp_num, block


def load_existing_minute_data():
    """Load existing physiology minute data for validation."""
    df = pd.read_csv(MINUTE_FILE, sep='\t')
    # Remove duplicated header rows
    df = df[df['PP'] != 'PP'].copy()
    
    # Convert numeric columns (but keep PP as string)
    for col in ['C', 'HR', 'RMSSD', 'SCL']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract PP number
    df['PP_num'] = df['PP'].str.replace('PP', '').astype(int)
    
    return df


def main():
    # ── 1. Load existing data for validation ────────────────
    print("Loading existing minute data for validation...")
    existing = load_existing_minute_data()
    print(f"  {len(existing)} rows, {existing['HR'].notna().sum()} with valid HR")
    
    # ── 2. Extract features from ALL .S00 files ─────────────
    print("\nExtracting features from all .S00 files...")
    
    s00_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.S00')])
    
    all_rows = []
    for fname in s00_files:
        pp_num, block = parse_pp_and_block(fname)
        
        if block == -1:
            # Skip continuation files for now
            print(f"  Skipping continuation file: {fname}")
            continue
        
        if block is None:
            print(f"  WARNING: Could not parse block from {fname}")
            continue
        
        fpath = os.path.join(RAW_DIR, fname)
        try:
            sig = tms_read(fpath)
            minute_features = extract_minute_features(sig)
            
            for mf in minute_features:
                mf['PP'] = pp_num
                mf['block'] = block
                mf['source_file'] = fname
                all_rows.append(mf)
            
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
    
    df_extracted = pd.DataFrame(all_rows)
    print(f"\nExtracted: {len(df_extracted)} minute records from {len(s00_files)} files")
    print(f"  HR (Ch2) valid:  {df_extracted['HR_ch2'].notna().sum()} ({100*df_extracted['HR_ch2'].notna().mean():.1f}%)")
    print(f"  HR (beats) valid: {df_extracted['HR_beats'].notna().sum()} ({100*df_extracted['HR_beats'].notna().mean():.1f}%)")
    print(f"  RMSSD valid:     {df_extracted['RMSSD'].notna().sum()} ({100*df_extracted['RMSSD'].notna().mean():.1f}%)")
    print(f"  SCL valid:       {df_extracted['SCL'].notna().sum()} ({100*df_extracted['SCL'].notna().mean():.1f}%)")
    
    # ── 3. Validate against existing data ────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION: Compare extracted vs existing data")
    print("=" * 70)
    
    # For each participant/block in existing data, compare minute-by-minute
    hr_diffs = []
    scl_diffs = []
    rmssd_diffs = []
    
    for pp in sorted(existing['PP_num'].unique()):
        for block in sorted(existing[existing['PP_num'] == pp]['C'].dropna().unique()):
            block = int(block)
            
            ex = existing[(existing['PP_num'] == pp) & (existing['C'] == block)].reset_index(drop=True)
            ext = df_extracted[(df_extracted['PP'] == pp) & (df_extracted['block'] == block)].reset_index(drop=True)
            
            if len(ex) == 0 or len(ext) == 0:
                continue
            
            n = min(len(ex), len(ext))
            
            # Compare SCL (should match well since it's a direct measurement)
            for i in range(n):
                ex_scl = ex.iloc[i]['SCL']
                ext_scl = ext.iloc[i]['SCL']
                if not np.isnan(ex_scl) and not np.isnan(ext_scl) and ex_scl > 0:
                    scl_diffs.append({
                        'PP': pp, 'block': block, 'minute': i+1,
                        'existing': ex_scl, 'extracted': ext_scl,
                        'ratio': ext_scl / ex_scl,
                        'diff_pct': 100 * (ext_scl - ex_scl) / ex_scl
                    })
            
            # Compare HR
            for i in range(n):
                ex_hr = ex.iloc[i]['HR']
                ext_hr = ext.iloc[i]['HR_beats']
                if not np.isnan(ex_hr) and not np.isnan(ext_hr) and ex_hr > 0:
                    hr_diffs.append({
                        'PP': pp, 'block': block, 'minute': i+1,
                        'existing': ex_hr, 'extracted': ext_hr,
                        'diff': ext_hr - ex_hr
                    })
            
            # Compare RMSSD
            for i in range(n):
                ex_rmssd = ex.iloc[i]['RMSSD']
                ext_rmssd = ext.iloc[i]['RMSSD']
                if not np.isnan(ex_rmssd) and not np.isnan(ext_rmssd) and ex_rmssd > 0:
                    rmssd_diffs.append({
                        'PP': pp, 'block': block, 'minute': i+1,
                        'existing': ex_rmssd, 'extracted': ext_rmssd,
                        'ratio': ext_rmssd / (ex_rmssd * 1000)  # existing is in seconds, extracted in ms
                    })
    
    # SCL validation
    if scl_diffs:
        scl_df = pd.DataFrame(scl_diffs)
        print(f"\nSCL Validation ({len(scl_df)} matched minutes):")
        print(f"  Mean ratio (extracted/existing): {scl_df['ratio'].mean():.4f}")
        print(f"  Median ratio: {scl_df['ratio'].median():.4f}")
        print(f"  Mean diff %: {scl_df['diff_pct'].mean():.2f}%")
        print(f"  Std diff %: {scl_df['diff_pct'].std():.2f}%")
        
        # Check if there's a systematic offset (timing misalignment)
        # Try shifting by -1 minute
        scl_shift = []
        for pp in existing['PP_num'].unique():
            for block in existing[existing['PP_num'] == pp]['C'].dropna().unique():
                block = int(block)
                ex = existing[(existing['PP_num'] == pp) & (existing['C'] == block)].reset_index(drop=True)
                ext = df_extracted[(df_extracted['PP'] == pp) & (df_extracted['block'] == block)].reset_index(drop=True)
                if len(ex) < 2 or len(ext) < 2:
                    continue
                n = min(len(ex), len(ext) - 1)
                for i in range(n):
                    ex_scl = ex.iloc[i]['SCL']
                    ext_scl = ext.iloc[i + 1]['SCL']  # shifted by +1
                    if not np.isnan(ex_scl) and not np.isnan(ext_scl) and ex_scl > 0:
                        scl_shift.append(ext_scl / ex_scl)
        
        if scl_shift:
            print(f"  With +1 minute shift: mean ratio = {np.mean(scl_shift):.4f}")
    
    # HR validation
    if hr_diffs:
        hr_df = pd.DataFrame(hr_diffs)
        print(f"\nHR Validation ({len(hr_df)} matched minutes):")
        print(f"  Mean diff (extracted - existing): {hr_df['diff'].mean():.2f} BPM")
        print(f"  Median diff: {hr_df['diff'].median():.2f} BPM")
        print(f"  Std diff: {hr_df['diff'].std():.2f} BPM")
        print(f"  Mean abs diff: {hr_df['diff'].abs().mean():.2f} BPM")
        print(f"  Correlation: {hr_df['existing'].corr(hr_df['extracted']):.4f}")
        
        # Show by participant
        print(f"\n  Per-participant HR diff (mean):")
        for pp in sorted(hr_df['PP'].unique()):
            pp_data = hr_df[hr_df['PP'] == pp]
            print(f"    PP{pp}: diff={pp_data['diff'].mean():+.1f} BPM  "
                  f"(n={len(pp_data)}, corr={pp_data['existing'].corr(pp_data['extracted']):.3f})")
    
    # RMSSD validation
    if rmssd_diffs:
        rmssd_df = pd.DataFrame(rmssd_diffs)
        print(f"\nRMSSD Validation ({len(rmssd_df)} matched minutes):")
        print(f"  Existing unit appears to be seconds, extracted is ms")
        print(f"  Mean ratio (ms/sec): {rmssd_df['ratio'].mean():.4f}")
        print(f"  Median ratio: {rmssd_df['ratio'].median():.4f}")
    
    # ── 4. Decide on best HR source ─────────────────────────
    print("\n" + "=" * 70)
    print("HR SOURCE COMPARISON")
    print("=" * 70)
    
    # Compare Ch2 (continuous estimate) vs Ch4 (beat-derived)
    both_valid = df_extracted[df_extracted['HR_ch2'].notna() & df_extracted['HR_beats'].notna()]
    if len(both_valid) > 0:
        corr = both_valid['HR_ch2'].corr(both_valid['HR_beats'])
        diff = (both_valid['HR_ch2'] - both_valid['HR_beats']).mean()
        print(f"  Ch2 vs beats HR correlation: {corr:.4f}")
        print(f"  Mean diff (Ch2 - beats): {diff:.2f} BPM")
        print(f"  Ch2 mean: {both_valid['HR_ch2'].mean():.1f}, Beats mean: {both_valid['HR_beats'].mean():.1f}")
    
    # ── 5. Save extracted data ──────────────────────────────
    # Use best available HR: beats first, Ch2 as fallback
    df_extracted['HR'] = df_extracted['HR_beats'].fillna(df_extracted['HR_ch2'])
    
    # Select final columns
    output = df_extracted[['PP', 'block', 'minute', 'HR', 'RMSSD', 'SCL', 
                           'HR_ch2', 'HR_beats', 'n_beats', 'source_file']].copy()
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved extracted physiology to {OUTPUT_FILE}")
    print(f"  {len(output)} rows")
    print(f"  HR valid: {output['HR'].notna().sum()} ({100*output['HR'].notna().mean():.1f}%)")
    print(f"  Per-participant HR coverage:")
    for pp in sorted(output['PP'].unique()):
        pp_data = output[output['PP'] == pp]
        hr_pct = 100 * pp_data['HR'].notna().mean()
        print(f"    PP{pp}: {pp_data['HR'].notna().sum()}/{len(pp_data)} minutes ({hr_pct:.0f}%)")


if __name__ == "__main__":
    main()
