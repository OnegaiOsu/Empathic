"""
Detailed inspection of .S00 channel data to properly identify signals
and validate against known minute-level data.
"""
import os
import sys
import numpy as np
import pandas as pd
from tms_reader import tms_read

RAW_DIR = os.path.join(
    "0_SWELL", "0 - Raw data", "D - Physiology - raw data",
    "Mobi signals (raw and filtered)"
)
MINUTE_DIR = os.path.join(
    "0_SWELL", "2 - Minute data", "D - Physiology - minute data"
)


def inspect_channel_data(sig, channel_idx, duration_sec=10, start_sec=60):
    """Look at a small window of channel data to understand its nature."""
    fs = sig['fs']
    start = int(start_sec * fs)
    end = int((start_sec + duration_sec) * fs)
    data = sig['data'][channel_idx]
    if data is None:
        return None
    chunk = data[start:end]
    
    # Count unique values (for binary-like channels)
    unique = np.unique(chunk)
    
    # Look for periodicity (for HR-like channels)
    diff = np.diff(chunk)
    zero_crossings = np.sum(np.diff(np.sign(chunk - np.mean(chunk))) != 0)
    
    return {
        'min': chunk.min(),
        'max': chunk.max(),
        'mean': chunk.mean(),
        'std': chunk.std(),
        'n_unique': len(unique),
        'zero_crossings': zero_crossings,
        'unique_sample': unique[:10] if len(unique) < 20 else None,
    }


def compute_minute_features(sig, channel_idx, fs):
    """Compute per-minute mean for a given channel."""
    data = sig['data'][channel_idx]
    if data is None:
        return []
    samples_per_min = fs * 60
    n_minutes = len(data) // samples_per_min
    
    means = []
    for m in range(n_minutes):
        chunk = data[m * samples_per_min : (m + 1) * samples_per_min]
        means.append(np.mean(chunk))
    return means


def compute_hr_from_beats(sig, beat_channel_idx, fs):
    """
    Try to extract HR from the beat marker channel.
    The beat channel likely has pulses at each detected heartbeat.
    """
    data = sig['data'][beat_channel_idx]
    if data is None:
        return [], []
    
    # Find rising edges (transitions from 0 to 1)
    threshold = 0.5
    above = data > threshold
    rising = np.diff(above.astype(int)) > 0
    beat_indices = np.where(rising)[0]
    
    if len(beat_indices) < 2:
        return [], []
    
    # Compute IBI (inter-beat intervals) in seconds
    ibis = np.diff(beat_indices) / fs
    
    # Filter physiologically plausible IBIs (30-200 BPM → 0.3-2.0 sec)
    valid_mask = (ibis > 0.3) & (ibis < 2.0)
    
    # Per-minute HR and RMSSD
    samples_per_min = fs * 60
    n_minutes = len(data) // samples_per_min
    
    hr_per_min = []
    rmssd_per_min = []
    
    for m in range(n_minutes):
        # Find beats in this minute
        min_start = m * samples_per_min
        min_end = (m + 1) * samples_per_min
        
        # Beats in this minute
        mask = (beat_indices >= min_start) & (beat_indices < min_end)
        minute_beats = beat_indices[mask]
        
        if len(minute_beats) < 3:
            hr_per_min.append(np.nan)
            rmssd_per_min.append(np.nan)
            continue
        
        # IBIs for this minute
        minute_ibis = np.diff(minute_beats) / fs
        # Filter plausible
        plausible = minute_ibis[(minute_ibis > 0.3) & (minute_ibis < 2.0)]
        
        if len(plausible) < 2:
            hr_per_min.append(np.nan)
            rmssd_per_min.append(np.nan)
            continue
        
        hr = 60.0 / np.mean(plausible)
        rmssd = np.sqrt(np.mean(np.diff(plausible * 1000) ** 2))  # in ms
        
        hr_per_min.append(hr)
        rmssd_per_min.append(rmssd)
    
    return hr_per_min, rmssd_per_min


def load_existing_minute_data():
    """Load the existing physiology minute data for comparison."""
    minute_file = os.path.join(
        MINUTE_DIR,
        "Physiology - All available data per minute (MatLabTable - SCL complete).txt"
    )
    if not os.path.exists(minute_file):
        print(f"Minute data file not found: {minute_file}")
        return None
    
    df = pd.read_csv(minute_file, sep='\t')
    # Clean: remove duplicated header rows
    df = df[df['PP'] != 'PP'].copy()
    
    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def main():
    # ── 1. Detailed channel inspection on PP1 c1 ────────────
    print("=" * 70)
    print("PART 1: Detailed channel inspection (PP1 c1)")
    print("=" * 70)
    
    fpath = os.path.join(RAW_DIR, "pp1_18-9-2012_c1.S00")
    sig = tms_read(fpath)
    
    for ch_idx in range(sig['num_channels']):
        desc = sig['descriptions'][ch_idx * 2]
        print(f"\nChannel {ch_idx}: '{desc['name']}' [{desc['unit']}]")
        
        # Look at data properties at a few time points
        for start_sec in [60, 300, 600]:
            info = inspect_channel_data(sig, ch_idx, duration_sec=5, start_sec=start_sec)
            if info:
                print(f"  @{start_sec}s: mean={info['mean']:.3f} std={info['std']:.3f} "
                      f"range=[{info['min']:.3f}, {info['max']:.3f}] "
                      f"unique={info['n_unique']} zero_cross={info['zero_crossings']}")
                if info['unique_sample'] is not None:
                    print(f"    unique values: {info['unique_sample']}")
    
    # ── 2. Extract per-minute SCL from Ch6, compare with known data ──
    print("\n" + "=" * 70)
    print("PART 2: Compare extracted SCL with existing minute data")
    print("=" * 70)
    
    existing = load_existing_minute_data()
    
    # PP1, block 1
    fs = sig['fs']
    scl_per_min = compute_minute_features(sig, 6, fs)  # Ch6 = skin conductance
    hr_ch2_per_min = compute_minute_features(sig, 2, fs)  # Ch2 = HR estimate
    
    if existing is not None:
        # PP column is string like 'PP1', C is condition block (1,2,3)
        pp1_c1 = existing[(existing['PP'] == 'PP1') & (existing['C'] == 1)]
        
        print(f"\nPP1 Block 1:")
        print(f"  Extracted SCL (Ch6): {len(scl_per_min)} minutes, "
              f"mean={np.mean(scl_per_min):.2f}")
        if 'SCL' in pp1_c1.columns:
            print(f"  Existing SCL:        {len(pp1_c1)} minutes, "
                  f"mean={pp1_c1['SCL'].mean():.2f}")
        
        # Compare first few minute values
        n_compare = min(10, len(scl_per_min), len(pp1_c1))
        print(f"\n  Minute-by-minute SCL comparison (first {n_compare}):")
        print(f"  {'Min':>4s}  {'Extracted':>10s}  {'Existing':>10s}  {'Ratio':>8s}")
        for i in range(n_compare):
            ext_val = scl_per_min[i]
            exist_val = pp1_c1.iloc[i]['SCL'] if 'SCL' in pp1_c1.columns else np.nan
            ratio = ext_val / exist_val if exist_val != 0 and not np.isnan(exist_val) else np.nan
            print(f"  {i+1:4d}  {ext_val:10.2f}  {exist_val:10.2f}  {ratio:8.4f}")
        
        # Also compare HR
        print(f"\n  Extracted HR (Ch2):  {len(hr_ch2_per_min)} minutes, "
              f"mean={np.mean(hr_ch2_per_min):.2f}")
        if 'HR' in pp1_c1.columns:
            pp1_c1_hr = pp1_c1['HR'].dropna()
            print(f"  Existing HR:         {len(pp1_c1_hr)} valid minutes, "
                  f"mean={pp1_c1_hr.mean():.2f}")
    
    # ── 3. Try beat-based HR extraction ─────────────────────
    print("\n" + "=" * 70)
    print("PART 3: Beat-based HR extraction from Ch4")
    print("=" * 70)
    
    hr_beats, rmssd_beats = compute_hr_from_beats(sig, 4, fs)
    print(f"\nPP1 c1 beat-based extraction:")
    print(f"  HR:    {len(hr_beats)} minutes, valid={sum(1 for h in hr_beats if not np.isnan(h))}")
    if hr_beats:
        valid_hr = [h for h in hr_beats if not np.isnan(h)]
        if valid_hr:
            print(f"         mean={np.mean(valid_hr):.1f} BPM  (range {np.min(valid_hr):.1f}-{np.max(valid_hr):.1f})")
    print(f"  RMSSD: {len(rmssd_beats)} minutes, valid={sum(1 for r in rmssd_beats if not np.isnan(r))}")
    if rmssd_beats:
        valid_rmssd = [r for r in rmssd_beats if not np.isnan(r)]
        if valid_rmssd:
            print(f"         mean={np.mean(valid_rmssd):.1f} ms  (range {np.min(valid_rmssd):.1f}-{np.max(valid_rmssd):.1f})")
    
    # Compare with existing data
    if existing is not None and 'HR' in existing.columns:
        pp1_c1 = existing[(existing['PP'] == 'PP1') & (existing['C'] == 1)]
        pp1_hr = pp1_c1['HR'].dropna()
        pp1_rmssd = pp1_c1['RMSSD'].dropna() if 'RMSSD' in pp1_c1.columns else pd.Series()
        
        n_compare = min(5, len(hr_beats), len(pp1_c1))
        print(f"\n  Minute-by-minute HR comparison (first {n_compare}):")
        print(f"  {'Min':>4s}  {'Ch2_mean':>10s}  {'Beat_HR':>10s}  {'Existing':>10s}")
        for i in range(n_compare):
            ch2_val = hr_ch2_per_min[i] if i < len(hr_ch2_per_min) else np.nan
            beat_val = hr_beats[i] if i < len(hr_beats) else np.nan
            exist_val = pp1_c1.iloc[i]['HR'] if i < len(pp1_c1) else np.nan
            print(f"  {i+1:4d}  {ch2_val:10.2f}  {beat_val:10.2f}  {exist_val:10.2f}")
    
    # ── 4. Check problematic participants ────────────────────
    print("\n" + "=" * 70)
    print("PART 4: Beat extraction for problematic participants (PP8, PP11, PP23)")
    print("=" * 70)
    
    problem_files = {
        'PP8 c1': "pp8_4-10-2012_c1.S00",
        'PP11 c1': "pp11_9-10-2012_c1.S00",
        'PP23 c1': "pp23_1-11-2012_c1.S00",
    }
    
    for label, fname in problem_files.items():
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n{label}: file not found")
            continue
        
        sig = tms_read(fpath)
        print(f"\n{label} ({sig['measurement_duration']})")
        
        # Beat detection
        hr_beats, rmssd_beats = compute_hr_from_beats(sig, 4, sig['fs'])
        valid_hr = [h for h in hr_beats if not np.isnan(h)]
        valid_rmssd = [r for r in rmssd_beats if not np.isnan(r)]
        
        print(f"  Ch2 (HR est):  mean={np.mean(compute_minute_features(sig, 2, sig['fs'])):.1f}")
        print(f"  Ch4 beat HR:   {len(valid_hr)}/{len(hr_beats)} valid minutes")
        if valid_hr:
            print(f"                 mean={np.mean(valid_hr):.1f} BPM")
        print(f"  Ch4 RMSSD:     {len(valid_rmssd)}/{len(rmssd_beats)} valid minutes")
        if valid_rmssd:
            print(f"                 mean={np.mean(valid_rmssd):.1f} ms")
        
        # Ch6 SCL
        scl = compute_minute_features(sig, 6, sig['fs'])
        print(f"  Ch6 SCL:       {len(scl)} minutes, mean={np.mean(scl):.1f}")
        
        # Count beats in Ch4
        data4 = sig['data'][4]
        if data4 is not None:
            above = data4 > 0.5
            rising = np.diff(above.astype(int)) > 0
            n_beats = np.sum(rising)
            duration_min = len(data4) / sig['fs'] / 60
            print(f"  Total beats detected: {n_beats} in {duration_min:.1f} min "
                  f"({n_beats/duration_min:.0f}/min avg)" if n_beats > 0 else "  NO BEATS DETECTED")
    
    # ── 5. Scan ALL .S00 files for data quality ─────────────
    print("\n" + "=" * 70)
    print("PART 5: Data quality scan across ALL .S00 files")
    print("=" * 70)
    
    s00_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.S00')])
    
    print(f"\n{'File':<35s}  {'Dur':>6s}  {'Beats':>6s}  {'HR_valid':>8s}  {'SCL_mean':>8s}  {'HR_mean':>8s}")
    print("-" * 85)
    
    for fname in s00_files:
        fpath = os.path.join(RAW_DIR, fname)
        try:
            sig = tms_read(fpath)
            duration = sig['measurement_duration']
            
            # Beat count
            data4 = sig['data'][4]
            if data4 is not None:
                above = data4 > 0.5
                rising = np.diff(above.astype(int)) > 0
                n_beats = int(np.sum(rising))
            else:
                n_beats = 0
            
            # Minute-level
            n_mins = sig['total_samples'] // (sig['fs'] * 60)
            hr_beats, _ = compute_hr_from_beats(sig, 4, sig['fs'])
            valid_hr = [h for h in hr_beats if not np.isnan(h)]
            scl = compute_minute_features(sig, 6, sig['fs'])
            
            hr_str = f"{len(valid_hr)}/{n_mins}"
            hr_mean = f"{np.mean(valid_hr):.0f}" if valid_hr else "N/A"
            scl_mean = f"{np.mean(scl):.1f}" if scl else "N/A"
            
            print(f"{fname:<35s}  {duration:>6s}  {n_beats:6d}  {hr_str:>8s}  {scl_mean:>8s}  {hr_mean:>8s}")
        except Exception as e:
            print(f"{fname:<35s}  ERROR: {e}")


if __name__ == "__main__":
    main()
