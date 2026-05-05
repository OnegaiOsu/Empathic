"""Quick scan of ALL .S00 files - data quality summary only."""
import os
import numpy as np
from tms_reader import tms_read

RAW_DIR = os.path.join(
    "0_SWELL", "0 - Raw data", "D - Physiology - raw data",
    "Mobi signals (raw and filtered)"
)

def compute_hr_from_beats(sig, beat_channel_idx, fs):
    data = sig['data'][beat_channel_idx]
    if data is None:
        return [], []
    above = data > 0.5
    rising = np.diff(above.astype(int)) > 0
    beat_indices = np.where(rising)[0]
    if len(beat_indices) < 2:
        return [], []
    
    samples_per_min = fs * 60
    n_minutes = len(data) // samples_per_min
    hr_per_min = []
    rmssd_per_min = []
    
    for m in range(n_minutes):
        min_start = m * samples_per_min
        min_end = (m + 1) * samples_per_min
        mask = (beat_indices >= min_start) & (beat_indices < min_end)
        minute_beats = beat_indices[mask]
        
        if len(minute_beats) < 3:
            hr_per_min.append(np.nan)
            rmssd_per_min.append(np.nan)
            continue
        
        minute_ibis = np.diff(minute_beats) / fs
        plausible = minute_ibis[(minute_ibis > 0.3) & (minute_ibis < 2.0)]
        
        if len(plausible) < 2:
            hr_per_min.append(np.nan)
            rmssd_per_min.append(np.nan)
            continue
        
        hr_per_min.append(60.0 / np.mean(plausible))
        rmssd_per_min.append(np.sqrt(np.mean(np.diff(plausible * 1000) ** 2)))
    
    return hr_per_min, rmssd_per_min

s00_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.S00')])

print(f"{'File':<35s}  {'Dur':>6s}  {'Beats':>6s}  {'HR_ok':>8s}  {'SCL_mean':>8s}  {'HR_mean':>8s}  {'RMSSD':>8s}")
print("-" * 95)

total_hr_valid = 0
total_minutes = 0
participants_with_hr = set()

for fname in s00_files:
    fpath = os.path.join(RAW_DIR, fname)
    try:
        sig = tms_read(fpath)
        duration = sig['measurement_duration']
        fs = sig['fs']
        
        # Beat count
        data4 = sig['data'][4]
        if data4 is not None:
            above = data4 > 0.5
            rising = np.diff(above.astype(int)) > 0
            n_beats = int(np.sum(rising))
        else:
            n_beats = 0
        
        n_mins = sig['total_samples'] // (fs * 60)
        hr_beats, rmssd_beats = compute_hr_from_beats(sig, 4, fs)
        valid_hr = [h for h in hr_beats if not np.isnan(h)]
        valid_rmssd = [r for r in rmssd_beats if not np.isnan(r)]
        
        # SCL from Ch6
        data6 = sig['data'][6]
        if data6 is not None:
            samples_per_min = fs * 60
            scl_mins = [np.mean(data6[m*samples_per_min:(m+1)*samples_per_min]) for m in range(n_mins)]
            scl_mean = f"{np.mean(scl_mins):.1f}" if scl_mins else "N/A"
        else:
            scl_mean = "N/A"
        
        hr_str = f"{len(valid_hr)}/{n_mins}"
        hr_mean = f"{np.mean(valid_hr):.0f}" if valid_hr else "N/A"
        rmssd_mean = f"{np.mean(valid_rmssd):.0f}" if valid_rmssd else "N/A"
        
        total_hr_valid += len(valid_hr)
        total_minutes += n_mins
        if valid_hr:
            # Extract PP number from filename
            pp = fname.split('_')[0].replace('pp', 'PP')
            participants_with_hr.add(pp)
        
        print(f"{fname:<35s}  {duration:>6s}  {n_beats:6d}  {hr_str:>8s}  {scl_mean:>8s}  {hr_mean:>8s}  {rmssd_mean:>8s}")
    except Exception as e:
        print(f"{fname:<35s}  ERROR: {e}")

print(f"\n{'='*95}")
print(f"SUMMARY:")
print(f"  Total files: {len(s00_files)}")
print(f"  Total minutes: {total_minutes}")
print(f"  Total minutes with valid HR: {total_hr_valid} ({100*total_hr_valid/total_minutes:.1f}%)")
print(f"  Participants with any HR: {len(participants_with_hr)}/25")
print(f"  Participants with HR: {sorted(participants_with_hr)}")
