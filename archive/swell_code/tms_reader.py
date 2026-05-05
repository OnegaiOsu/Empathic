"""
Port of tms_read.m — reads Poly5/TMS32 (.S00) physiology files from the Mobi device.

Based on the MATLAB tms_read.m from the SWELL-KW dataset.
The .S00 files contain raw physiological signals:
  - Skin conductance (raw and filtered)
  - Heart rate signals (raw and filtered/preprocessed)
  - Heart beats
"""

import struct
import numpy as np
import os


def tms_read(filepath):
    """
    Read a Poly5/TMS32 (.S00) file and return signal data.

    Returns
    -------
    dict with keys:
        'fname': str, filename without extension
        'fs': int, sampling frequency in Hz
        'measurement_date': str
        'measurement_time': str
        'measurement_duration': str
        'num_signals': int, number of signal channels (32-bit pairs)
        'descriptions': list of dicts with channel info
        'data': list of numpy arrays, one per channel
    """
    result = {}
    result['fname'] = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'rb') as fid:
        # ── Read file header ────────────────────────────────
        # Determine version: peek at byte 31
        fid.seek(31)
        version = struct.unpack('<h', fid.read(2))[0]

        fid.seek(0)
        if version == 203:
            fid_bytes = fid.read(31)  # FID
            version_num = struct.unpack('<h', fid.read(2))[0]
            header_base = 217
        else:  # version 204
            fid_bytes = fid.read(32)  # FID
            version_num = struct.unpack('<h', fid.read(2))[0]
            header_base = 218

        result['version'] = version_num

        meas_name_raw = fid.read(81)
        # First byte is length, then name chars
        name_len = meas_name_raw[0]
        result['measurement_name'] = meas_name_raw[1:1+name_len].decode('ascii', errors='replace')

        fs = struct.unpack('<h', fid.read(2))[0]
        result['fs'] = fs

        storage_rate = struct.unpack('<h', fid.read(2))[0]
        storage_type = struct.unpack('<B', fid.read(1))[0]
        num_signals = struct.unpack('<h', fid.read(2))[0]
        num_sample_periods = struct.unpack('<i', fid.read(4))[0]
        empty_bytes = fid.read(4)
        start_measurement = fid.read(14)
        num_sample_blocks = struct.unpack('<i', fid.read(4))[0]
        samples_per_block = struct.unpack('<H', fid.read(2))[0]
        size_signal_data_block = struct.unpack('<H', fid.read(2))[0]
        delta_compression = struct.unpack('<h', fid.read(2))[0]
        trailing_zeros = fid.read(64)

        result['num_signals_raw'] = num_signals
        num_channels = num_signals // 2  # 32-bit channels from 16-bit pairs
        result['num_channels'] = num_channels

        # ── Read signal descriptions ────────────────────────
        descriptions = []
        for g in range(num_signals):
            desc = {}
            sig_name_raw = fid.read(41)
            name_len = sig_name_raw[0]
            desc['name'] = sig_name_raw[1:1+name_len].decode('ascii', errors='replace')

            desc['reserved'] = fid.read(4)

            unit_name_raw = fid.read(11)
            unit_len = unit_name_raw[0]
            desc['unit'] = unit_name_raw[1:1+unit_len].decode('ascii', errors='replace')

            desc['unit_low'] = struct.unpack('<f', fid.read(4))[0]
            desc['unit_high'] = struct.unpack('<f', fid.read(4))[0]
            desc['adc_low'] = struct.unpack('<f', fid.read(4))[0]
            desc['adc_high'] = struct.unpack('<f', fid.read(4))[0]
            desc['index_signal_list'] = struct.unpack('<h', fid.read(2))[0]
            desc['cache_offset'] = struct.unpack('<h', fid.read(2))[0]
            desc['reserved2'] = fid.read(60)

            descriptions.append(desc)

        result['descriptions'] = descriptions

        # ── Read data blocks ────────────────────────────────
        NB = num_sample_blocks
        SD = size_signal_data_block
        NS = num_signals
        NS_32bit = NS // 2

        all_data = []
        for g in range(NB):
            # Jump to right position
            pos = header_base + NS * 136 + g * (86 + SD)
            fid.seek(pos)

            # Block metadata
            pi = struct.unpack('<i', fid.read(4))[0]  # period index
            fid.read(4)  # reserved
            bt = struct.unpack('<7h', fid.read(14))  # dostime
            fid.read(64)  # reserved

            # Read signal data
            num_floats = SD // 4
            data = np.frombuffer(fid.read(num_floats * 4), dtype=np.float32)
            data = data.reshape(NS_32bit, -1, order='F')
            all_data.append(data)

            if g == 0:
                # Extract date/time from first block
                result['measurement_date'] = f"{bt[2]:02d}-{bt[1]:02d}-{bt[0]:02d}"
                result['measurement_time'] = f"{bt[4]:02d}:{bt[5]:02d}:{bt[6]:02d}"

        # Concatenate all blocks
        all_data = np.concatenate(all_data, axis=1).astype(np.float64)

        # Convert to calibrated units (uV)
        channel_data = []
        active_channels = []
        for g in range(NS_32bit):
            # Use even-indexed descriptions (the 32-bit channel descriptions)
            desc = descriptions[g * 2]
            adc_low = desc['adc_low']
            adc_high = desc['adc_high']
            unit_low = desc['unit_low']
            unit_high = desc['unit_high']

            calibrated = (all_data[g] - adc_low) / (adc_high - adc_low) * (unit_high - unit_low) + unit_low

            # Check if channel has data (not all zeros)
            if np.mean(calibrated) != 0 or np.std(calibrated) != 0:
                channel_data.append(calibrated)
                active_channels.append(g)
            else:
                channel_data.append(None)

        result['data'] = channel_data
        result['active_channels'] = active_channels

        # Compute duration
        total_samples = all_data.shape[1]
        ts = total_samples / fs
        th = int(ts // 3600)
        tm = int((ts - th * 3600) // 60)
        tss = int(ts - th * 3600 - tm * 60)
        result['measurement_duration'] = f"{th:02d}:{tm:02d}:{tss:02d}"
        result['total_samples'] = total_samples

    return result


def print_signal_info(signal):
    """Print summary of loaded signal."""
    print(f"  File: {signal['fname']}")
    print(f"  Sample rate: {signal['fs']} Hz")
    print(f"  Date: {signal['measurement_date']}")
    print(f"  Time: {signal['measurement_time']}")
    print(f"  Duration: {signal['measurement_duration']}")
    print(f"  Total samples: {signal['total_samples']}")
    print(f"  Channels: {signal['num_channels']} (32-bit)")
    print(f"  Active channels: {signal['active_channels']}")
    for i, desc in enumerate(signal['descriptions']):
        if i % 2 == 0:  # Only show 32-bit channel descriptions
            ch_idx = i // 2
            has_data = signal['data'][ch_idx] is not None
            if has_data:
                d = signal['data'][ch_idx]
                print(f"    Ch{ch_idx}: '{desc['name']}' [{desc['unit']}] "
                      f"range=[{d.min():.2f}, {d.max():.2f}] mean={d.mean():.2f}")
            else:
                print(f"    Ch{ch_idx}: '{desc['name']}' [{desc['unit']}] — NO DATA")


if __name__ == "__main__":
    import sys

    # Test with a few sample files
    raw_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "0_SWELL", "0 - Raw data", "D - Physiology - raw data",
        "Mobi signals (raw and filtered)"
    )

    # Test with PP1 (good data), PP8 (100% missing in CSV), PP11 (no c2)
    test_files = [
        "pp1_18-9-2012_c1.S00",
        "pp8_4-10-2012_c1.S00",
        "pp11_9-10-2012_c1.S00",
        "pp23_1-11-2012_c1.S00",
    ]

    for fname in test_files:
        fpath = os.path.join(raw_dir, fname)
        if os.path.exists(fpath):
            print(f"\n{'='*60}")
            print(f"Reading: {fname}")
            print(f"{'='*60}")
            try:
                sig = tms_read(fpath)
                print_signal_info(sig)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  File not found: {fname}")
