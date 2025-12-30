#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis
====================================
Analyzes collected Iridium data for noise, SINR, sample rates, and other statistics

Usage:
    python analyze_data_quality_simple.py
    
This script analyzes the experiment defined in src/config/setup.py
"""

import pickle
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator
from datetime import datetime, timedelta

# Import configuration from setup.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config.setup import WORKING_DIR, EXP_NAME

print(f"Using configuration from src/config/setup.py:")
print(f"  EXP_NAME = {EXP_NAME}")
print(f"  WORKING_DIR = {WORKING_DIR}\n")

# File paths
DATA_PATH = WORKING_DIR + f"/{EXP_NAME}/"
FRAME_FILE = DATA_PATH + "decoded.txt"
NAV_DATA_FILE = DATA_PATH + "saved_nav_data.pickle"

# NavData array indices
class IDX:
    t = 0           # time
    f = 1           # frequency
    fb = 2          # base frequency
    sat_id = 3      # satellite ID
    x = 4           # position x
    y = 5           # position y
    z = 6           # position z
    vx = 7          # velocity x
    vy = 8          # velocity y
    vz = 9          # velocity z


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_section(title):
    """Print section header"""
    print(f"\n{title}")
    print("-" * 80)


def load_data():
    """Load navigation data and frames"""
    print("Loading data...")
    
    # Load processed navigation data
    with open(NAV_DATA_FILE, "rb") as file:
        nav_data = pickle.load(file)
    print(f"  ✓ Loaded {len(nav_data):,} navigation measurements")
    
    # Load raw frames
    with open(FRAME_FILE, "r") as file:
        frames = file.readlines()
    print(f"  ✓ Loaded {len(frames):,} raw frames")
    
    return nav_data, frames


def analyze_basic_stats(nav_data):
    """Analyze basic dataset statistics"""
    print_header(f"BASIC STATISTICS: {EXP_NAME}")
    
    # Try to load start time
    try:
        with open(DATA_PATH + "start_time.txt", "r") as file:
            start_time = file.readlines()[0].strip()
        print(f"{'Start Time:':<30} {start_time}")
    except:
        print(f"{'Start Time:':<30} Not found")
    
    # Duration
    duration = nav_data[-1, IDX.t] - nav_data[0, IDX.t]
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"{'Total Measurements:':<30} {len(nav_data):,}")
    print(f"{'Duration:':<30} {hours:02d}h {minutes:02d}m {seconds:02d}s ({duration:.1f} seconds)")
    
    # Satellites
    unique_sats = np.unique(nav_data[:, IDX.sat_id])
    print(f"{'Number of Satellites:':<30} {len(unique_sats)}")
    print(f"{'Satellite IDs:':<30} {', '.join(str(int(s)) for s in unique_sats)}")


def analyze_hardware_config():
    """Print hardware configuration info"""
    print_header("HARDWARE CONFIGURATION")
    
    #print(f"{'SDR Type:':<30} PlutoSDR (ADALM-PLUTO)")
    #print(f"{'Sample Rate:':<30} 5,000,000 samples/sec (5 MHz)")
    #print(f"{'Center Frequency:':<30} 1,626.2708 MHz")
    #print(f"{'Bandwidth:':<30} Up to 20 MHz")
    #print(f"{'Reference Clock:':<30} 40 MHz (±25 ppm accuracy)")


def analyze_timing_stats(nav_data):
    """Analyze timing and frame rate"""
    print_header("TIMING & FRAME RATE ANALYSIS")
    
    duration = nav_data[-1, IDX.t] - nav_data[0, IDX.t]
    frame_rate = len(nav_data) / duration
    
    print(f"\n{'Effective Frame Rate:'}")
    print(f"  {'Frames per second:':<28} {frame_rate:.2f} Hz")
    print(f"  {'Average time between frames:':<28} {1000/frame_rate:.1f} ms")
    
    # Time gaps analysis
    time_diffs = np.diff(nav_data[:, IDX.t])
    
    print(f"\n{'Frame Timing Statistics:'}")
    print(f"  {'Min time gap:':<28} {np.min(time_diffs):.3f} s")
    print(f"  {'Mean time gap:':<28} {np.mean(time_diffs):.3f} s")
    print(f"  {'Max time gap:':<28} {np.max(time_diffs):.3f} s")
    print(f"  {'Std dev of time gaps:':<28} {np.std(time_diffs):.3f} s")
    
    # Large gaps
    large_gaps = time_diffs > 60
    if np.any(large_gaps):
        print(f"  {'Large gaps (>60s):':<28} {np.sum(large_gaps)} gaps found")


def parse_snr_from_frames(frames, min_confidence=80):
    """Parse SNR data from IRA frames (Signal|Noise|SNR format)"""
    snr_data = []
    
    for frame in frames:
        if not frame.startswith("IRA"):
            continue
            
        try:
            parts = frame.split()
            if len(parts) < 6:
                continue
            
            # Extract confidence
            confidence = None
            for part in parts:
                if '%' in part:
                    confidence = int(part.replace('%', ''))
                    break
            
            if confidence is None or confidence < min_confidence:
                continue
            
            # Extract Signal|Noise|SNR (should be field 5)
            signal_noise_snr = parts[5]
            if '|' not in signal_noise_snr:
                continue
            
            try:
                signal_db, noise_db, snr_db = signal_noise_snr.split('|')
                snr_data.append({
                    'signal_db': float(signal_db),
                    'noise_db': float(noise_db),
                    'snr_db': float(snr_db),
                    'confidence': confidence
                })
            except (ValueError, IndexError):
                continue
                
        except:
            continue
    
    return snr_data


def analyze_noise_and_quality(frames):
    """Analyze noise, signal quality, and true SNR from frames"""
    print_header("SIGNAL QUALITY, NOISE & SNR ANALYSIS")
    
    # Count frame types
    ira_frames = [f for f in frames if f.startswith("IRA")]
    ibc_frames = [f for f in frames if f.startswith("IBC")]
    other_frames = len(frames) - len(ira_frames) - len(ibc_frames)
    
    print(f"\n{'Frame Counts:'}")
    print(f"  {'Total frames:':<28} {len(frames):,}")
    print(f"  {'IRA frames (Ring Alert):':<28} {len(ira_frames):,} ({100*len(ira_frames)/len(frames):.1f}%)")
    print(f"  {'IBC frames (Broadcast):':<28} {len(ibc_frames):,} ({100*len(ibc_frames)/len(frames):.1f}%)")
    print(f"  {'Other frames:':<28} {other_frames:,}")
    
    if len(ira_frames) == 0:
        print("\n  ⚠ No IRA frames found for detailed analysis")
        return None
    
    # Parse SNR data from frames
    snr_data = parse_snr_from_frames(frames, min_confidence=80)
    
    if len(snr_data) == 0:
        print("\n  ⚠ No SNR data found in frames")
        return None
    
    print(f"\n  ✓ Extracted SNR data from {len(snr_data)} high-quality frames (≥80% confidence)")
    
    # Extract arrays
    confidences = [d['confidence'] for d in snr_data]
    signal_levels = [d['signal_db'] for d in snr_data]
    noise_levels = [d['noise_db'] for d in snr_data]
    snr_values = [d['snr_db'] for d in snr_data]
    
    # === TRUE SNR STATISTICS ===
    print(f"\n{'═'*50}")
    print(f"{'TRUE SNR STATISTICS (from decoder)'}")
    print(f"{'═'*50}")
    
    print(f"\n{'SNR (Signal-to-Noise Ratio):'}")
    print(f"  {'Mean SNR:':<28} {np.mean(snr_values):.2f} dB")
    print(f"  {'Median SNR:':<28} {np.median(snr_values):.2f} dB")
    print(f"  {'Std Dev:':<28} {np.std(snr_values):.2f} dB")
    print(f"  {'Min SNR:':<28} {np.min(snr_values):.2f} dB")
    print(f"  {'Max SNR:':<28} {np.max(snr_values):.2f} dB")
    
    print(f"\n{'Signal Strength (dBFS):'}")
    print(f"  {'Mean:':<28} {np.mean(signal_levels):.2f} dBFS")
    print(f"  {'Std Dev:':<28} {np.std(signal_levels):.2f} dBFS")
    print(f"  {'Min:':<28} {np.min(signal_levels):.2f} dBFS")
    print(f"  {'Max:':<28} {np.max(signal_levels):.2f} dBFS")
    
    print(f"\n{'Noise Floor (dBFS):'}")
    print(f"  {'Mean:':<28} {np.mean(noise_levels):.2f} dBFS")
    print(f"  {'Std Dev:':<28} {np.std(noise_levels):.2f} dBFS")
    print(f"  {'Min:':<28} {np.min(noise_levels):.2f} dBFS")
    print(f"  {'Max:':<28} {np.max(noise_levels):.2f} dBFS")
    
    # SNR Quality Distribution
    print(f"\n{'SNR Quality Distribution:'}")
    excellent_snr = np.sum(np.array(snr_values) >= 20)
    good_snr = np.sum((np.array(snr_values) >= 15) & (np.array(snr_values) < 20))
    fair_snr = np.sum((np.array(snr_values) >= 10) & (np.array(snr_values) < 15))
    poor_snr = np.sum(np.array(snr_values) < 10)
    total = len(snr_values)
    
    print(f"  {'Excellent (≥20 dB):':<28} {excellent_snr:5d} frames ({100*excellent_snr/total:5.1f}%)")
    print(f"  {'Good (15-20 dB):':<28} {good_snr:5d} frames ({100*good_snr/total:5.1f}%)")
    print(f"  {'Fair (10-15 dB):':<28} {fair_snr:5d} frames ({100*fair_snr/total:5.1f}%)")
    print(f"  {'Poor (<10 dB):':<28} {poor_snr:5d} frames ({100*poor_snr/total:5.1f}%)")
    
    # Confidence statistics
    print(f"\n{'Decoder Confidence Statistics:'}")
    print(f"  {'Mean confidence:':<28} {np.mean(confidences):.1f}%")
    print(f"  {'Min confidence:':<28} {np.min(confidences):.1f}%")
    print(f"  {'Max confidence:':<28} {np.max(confidences):.1f}%")
    print(f"  {'Std dev:':<28} {np.std(confidences):.1f}%")
    
    # Interpretation
    mean_snr = np.mean(snr_values)
    print(f"\n{'Signal Quality Assessment:'}")
    if mean_snr >= 20:
        print(f"  ✓ Excellent SNR ({mean_snr:.1f} dB) - Very strong signals")
    elif mean_snr >= 15:
        print(f"  ✓ Good SNR ({mean_snr:.1f} dB) - Strong signals")
    elif mean_snr >= 10:
        print(f"  ⚠ Fair SNR ({mean_snr:.1f} dB) - Acceptable but could be improved")
    else:
        print(f"  ✗ Poor SNR ({mean_snr:.1f} dB) - Weak signals, consider:")
        print(f"      - Better antenna (higher gain)")
        print(f"      - Add LNA (Low Noise Amplifier)")
        print(f"      - Reduce RF interference")
        print(f"      - Improve antenna positioning")
    
    return snr_data


def analyze_per_satellite(nav_data):
    """Analyze statistics per satellite"""
    print_header("PER-SATELLITE STATISTICS")
    
    unique_sats = np.unique(nav_data[:, IDX.sat_id])
    
    print(f"\n{'Sat ID':<10} {'Frames':<10} {'Duration':<12} {'Frame Rate':<12} {'Doppler Range':<15}")
    print("-" * 80)
    
    for sat_id in unique_sats:
        mask = nav_data[:, IDX.sat_id] == sat_id
        sat_data = nav_data[mask]
        
        n_frames = len(sat_data)
        duration = sat_data[-1, IDX.t] - sat_data[0, IDX.t]
        frame_rate = n_frames / duration if duration > 0 else 0
        
        doppler = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
        dopp_range = np.max(doppler) - np.min(doppler)
        
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        dur_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print(f"{int(sat_id):<10} {n_frames:<10} {dur_str:<12} {frame_rate:>8.2f} Hz  {dopp_range/1000:>10.2f} kHz")


def analyze_doppler_stats(nav_data):
    """Analyze Doppler shift statistics"""
    print_header("DOPPLER SHIFT STATISTICS")
    
    doppler = nav_data[:, IDX.f] - nav_data[:, IDX.fb]
    
    print(f"\n{'Doppler Shift Range:':<30} {np.min(doppler)/1000:.2f} to {np.max(doppler)/1000:.2f} kHz")
    print(f"{'Mean Doppler:':<30} {np.mean(doppler)/1000:.2f} kHz")
    print(f"{'Std Dev Doppler:':<30} {np.std(doppler)/1000:.2f} kHz")
    
    # Doppler rate (per satellite)
    print(f"\n{'Doppler Rate of Change:'}")
    
    doppler_rates = []
    unique_sats = np.unique(nav_data[:, IDX.sat_id])
    
    for sat_id in unique_sats:
        mask = nav_data[:, IDX.sat_id] == sat_id
        sat_data = nav_data[mask]
        
        if len(sat_data) > 2:
            sat_doppler = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
            sat_time = sat_data[:, IDX.t]
            
            d_dopp = np.diff(sat_doppler)
            d_time = np.diff(sat_time)
            
            # Only valid time gaps
            valid_mask = (d_time > 0) & (d_time < 60)
            if np.any(valid_mask):
                rates = d_dopp[valid_mask] / d_time[valid_mask]
                doppler_rates.extend(rates[np.abs(rates) < 1000])
    
    if len(doppler_rates) > 0:
        print(f"  {'Mean rate:':<28} {np.mean(doppler_rates):.2f} Hz/s")
        print(f"  {'Std dev:':<28} {np.std(doppler_rates):.2f} Hz/s")
        print(f"  {'Max rate:':<28} {np.max(np.abs(doppler_rates)):.2f} Hz/s")


def plot_sample_collection_timeline(nav_data, frames, trim_empty_start=True):
    """Plot each sample as a point on a timeline, colored by confidence
    
    Args:
        nav_data: Navigation data array
        frames: Raw frame data
        trim_empty_start: If True, trim leading empty period from plot
    """
    print_header("GENERATING SAMPLE COLLECTION TIMELINE")
    
    # Get sample times relative to start
    t_start = nav_data[0, IDX.t]
    t_end = nav_data[-1, IDX.t]
    duration = t_end - t_start
    
    # Get relative times for each sample
    rel_times = nav_data[:, IDX.t] - t_start
    
    # Analyze initial data collection period
    print(f"\n  Initial Data Collection Analysis:")
    print(f"    {'First sample at:':<30} {rel_times[0]:.1f} seconds from nominal start")
    
    # Find first significant gap after start
    time_diffs = np.diff(rel_times)
    gap_threshold = 60  # seconds - gaps longer than this are significant
    large_gap_threshold = 600  # 10 minutes - very large gaps
    
    first_gap_idx = None
    first_large_gap_idx = None
    
    for i, gap in enumerate(time_diffs):
        if gap > gap_threshold and first_gap_idx is None:
            first_gap_idx = i
        if gap > large_gap_threshold and first_large_gap_idx is None:
            first_large_gap_idx = i
            break
    
    # Analyze initial burst
    if first_gap_idx is not None:
        initial_burst_duration = rel_times[first_gap_idx] - rel_times[0]
        initial_burst_samples = first_gap_idx + 1
        first_gap_duration = time_diffs[first_gap_idx]
        
        print(f"    {'Initial burst samples:':<30} {initial_burst_samples}")
        print(f"    {'Initial burst duration:':<30} {initial_burst_duration:.1f} seconds ({initial_burst_duration/60:.2f} minutes)")
        print(f"    {'First gap duration:':<30} {first_gap_duration:.1f} seconds ({first_gap_duration/60:.2f} minutes)")
        print(f"    {'Data resumes at:':<30} {rel_times[first_gap_idx + 1]:.1f} seconds ({rel_times[first_gap_idx + 1]/60:.1f} minutes)")
    
    # Check for very large initial gap (>10 minutes)
    if first_large_gap_idx is not None and first_large_gap_idx < 10:
        # Large gap very early in data - likely a few test samples then real collection
        print(f"\n  ⚠️  Large gap detected early in data:")
        print(f"    {'Gap after sample:':<30} {first_large_gap_idx + 1}")
        print(f"    {'Gap duration:':<30} {time_diffs[first_large_gap_idx]:.1f} seconds ({time_diffs[first_large_gap_idx]/60:.1f} minutes)")
        print(f"    {'Real data starts at:':<30} {rel_times[first_large_gap_idx + 1]:.1f} seconds ({rel_times[first_large_gap_idx + 1]/60:.1f} minutes)")
        
        # Analyze continuous data after the gap
        remaining_samples = len(nav_data) - (first_large_gap_idx + 1)
        remaining_duration = rel_times[-1] - rel_times[first_large_gap_idx + 1]
        print(f"\n  Continuous Data Collection (after gap):")
        print(f"    {'Samples collected:':<30} {remaining_samples}")
        print(f"    {'Duration:':<30} {remaining_duration:.1f} seconds ({remaining_duration/60:.1f} minutes)")
        print(f"    {'Average rate:':<30} {remaining_samples/remaining_duration:.3f} Hz")
    
    # Determine plot start time
    samples_trimmed = 0
    gap_trimmed = 0
    
    if trim_empty_start:
        if first_large_gap_idx is not None and first_large_gap_idx < 10:
            # Skip early scattered samples before large gap
            plot_start_idx = first_large_gap_idx + 1
            plot_start = rel_times[plot_start_idx] - 60  # Start 60s before real data
            plot_start = max(0, plot_start)
            samples_trimmed = plot_start_idx
            gap_trimmed = time_diffs[first_large_gap_idx]
            print(f"\n  📊 Trimming plot: Skipping {samples_trimmed} initial samples before large gap")
            print(f"      Gap duration: {gap_trimmed:.1f} seconds ({gap_trimmed/60:.1f} minutes)")
            print(f"      Plot will start at {plot_start:.1f} seconds from nominal start")
            print(f"\n  📊 Color coding: Samples colored by noise floor")
            print(f"      Green = Low noise (better), Red = High noise (worse)")
        elif rel_times[0] > 30:
            # Trim if there's a significant delay before first sample
            plot_start = rel_times[0] - 60  # Start 60s before first sample
            plot_start = max(0, plot_start)
            print(f"\n  📊 Trimming plot to start at {plot_start:.1f} seconds (removing empty leading period)")
        else:
            plot_start = 0
    else:
        plot_start = 0
    
    # Filter times for plotting
    plot_mask = rel_times >= plot_start
    rel_times_plot = rel_times[plot_mask] - plot_start
    
    # Always print color coding info
    if not (samples_trimmed > 0):
        print(f"\n  📊 Color coding: Samples colored by noise floor")
        print(f"      Green = Low noise (better), Red = High noise (worse)")
    
    # Extract confidence and noise for each sample from frames
    # Parse IRA frames to match with nav_data samples
    ira_frames = [f for f in frames if f.startswith("IRA")]
    
    # Create a dictionary of sat_id + time -> (confidence, noise)
    frame_data_map = {}
    for frame in ira_frames:
        try:
            parts = frame.split()
            # Extract sat_id, time, confidence, and noise
            sat_str = [p for p in parts if p.startswith('sat=')]
            time_str = [p for p in parts if p.startswith('time=')]
            conf_str = [p for p in parts if '%' in p]
            
            if sat_str and time_str and conf_str and len(parts) >= 6:
                sat_id = int(sat_str[0].split('=')[1])
                time_val = float(time_str[0].split('=')[1])
                conf = int(conf_str[0].replace('%', ''))
                
                # Extract noise from Signal|Noise|SNR field
                signal_noise_snr = parts[5]
                noise_db = None
                if '|' in signal_noise_snr:
                    try:
                        _, noise_db_str, _ = signal_noise_snr.split('|')
                        noise_db = float(noise_db_str)
                    except:
                        pass
                
                # Use sat_id + time as key
                key = (sat_id, time_val)
                frame_data_map[key] = {'confidence': conf, 'noise': noise_db}
        except:
            continue
    
    # Match nav_data with confidence and noise
    confidences = []
    noise_values = []
    matched_count = 0
    
    for i in range(len(nav_data)):
        sat_id = int(nav_data[i, IDX.sat_id])
        time_val = nav_data[i, IDX.t]
        
        # Try to find matching frame (within larger time window for better matching)
        conf = None
        noise = None
        best_match_diff = float('inf')
        
        for (map_sat, map_time), data in frame_data_map.items():
            if map_sat == sat_id:
                time_diff = abs(map_time - time_val)
                if time_diff < 10.0 and time_diff < best_match_diff:  # Increased window to 10 seconds
                    conf = data['confidence']
                    noise = data['noise']
                    best_match_diff = time_diff
        
        if conf is not None:
            matched_count += 1
        else:
            conf = 85  # Default confidence if not found
        
        confidences.append(conf)
        noise_values.append(noise if noise is not None else -95.0)  # Default noise
    
    confidences = np.array(confidences)
    noise_values = np.array(noise_values)
    
    # Debug: Print matching and noise value statistics
    print(f"\n  Frame Matching Statistics:")
    print(f"    {'IRA frames available:':<30} {len(ira_frames)}")
    print(f"    {'Frames with noise data:':<30} {len(frame_data_map)}")
    print(f"    {'Nav samples matched:':<30} {matched_count} of {len(nav_data)}")
    
    print(f"\n  Noise Floor Statistics:")
    print(f"    {'Samples with noise data:':<30} {np.sum(noise_values != -95.0)}")
    print(f"    {'Min noise:':<30} {np.min(noise_values):.2f} dBFS")
    print(f"    {'Max noise:':<30} {np.max(noise_values):.2f} dBFS")
    print(f"    {'Mean noise:':<30} {np.mean(noise_values):.2f} dBFS")
    print(f"    {'Std dev:':<30} {np.std(noise_values):.2f} dBFS")
    print(f"    {'Noise range:':<30} {np.max(noise_values) - np.min(noise_values):.2f} dB")
    
    # Apply plot mask to confidences and noise
    confidences_plot = confidences[plot_mask]
    noise_values_plot = noise_values[plot_mask]
    
    # Get satellite IDs for optional display
    sat_ids_plot = nav_data[plot_mask, IDX.sat_id]
    unique_sats_plot = np.unique(sat_ids_plot)
    
    # Create the main plot - 2 subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 8), 
                                     gridspec_kw={'height_ratios': [1, 1.5]}, 
                                     sharex=True)
    
    # Plot 1: Timeline of samples - color-coded by noise power
    # High noise (closer to 0) = red, Low noise (more negative) = green
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # Check if there's variation in noise values
    noise_range = np.max(noise_values_plot) - np.min(noise_values_plot)
    
    if noise_range > 1.0:  # Sufficient variation for color coding
        # Normalize noise values for color mapping
        # Since noise is in dBFS (negative values), more negative = better (green)
        # Less negative (closer to 0) = worse (red)
        norm = Normalize(vmin=np.min(noise_values_plot), vmax=np.max(noise_values_plot))
        cmap = cm.get_cmap('RdYlGn_r')  # Red-Yellow-Green reversed
        
        # Use seconds for X-axis
        x_times = rel_times_plot
        y_positions = np.ones(len(rel_times_plot))  # All at same y position
        
        # Create scatter plot with noise-based coloring
        scatter = ax1.scatter(x_times, y_positions, c=noise_values_plot, cmap='RdYlGn_r',
                              s=120, alpha=0.8, edgecolors='black', linewidths=1.5,
                              vmin=np.min(noise_values_plot), vmax=np.max(noise_values_plot))
        
        # Add colorbar for noise levels
        cbar = plt.colorbar(scatter, ax=ax1, orientation='vertical', pad=0.01)
        cbar.set_label('Noise Floor (dBFS)\n← Better (More Negative)', 
                       rotation=270, labelpad=20, fontweight='bold')
        
        color_coding_enabled = True
    else:
        # Not enough variation - use single color
        print(f"    ⚠️ Insufficient noise variation ({noise_range:.2f} dB) for color coding")
        print(f"       Using single color instead")
        
        x_times = rel_times_plot
        y_positions = np.ones(len(rel_times_plot))
        
        scatter = ax1.scatter(x_times, y_positions, c='steelblue', s=120, alpha=0.8, 
                              edgecolors='black', linewidths=1.5)
        color_coding_enabled = False
    
    # Formatting
    ax1.set_ylabel('Samples', fontsize=12, fontweight='bold')
    
    #title_str = f'Sample Collection Timeline - {EXP_NAME}\n'
    title_str = f'Sample Collection Timeline - Rover\n'

    #if color_coding_enabled:
    #    title_str += f'Showing {len(rel_times_plot)} of {len(nav_data)} samples (colored by Noise Floor)'
    #else:
    #    title_str += f'Showing {len(rel_times_plot)} of {len(nav_data)} samples'
    
    #if trim_empty_start and (samples_trimmed > 0 or plot_start > 0):
    #    if samples_trimmed > 0:
    #        title_str += f'\n(trimmed {samples_trimmed} initial samples + {gap_trimmed/60:.1f} min gap)'
    #    else:
    #        title_str += f'\n(trimmed {plot_start/60:.1f} min empty period)'
    
    ax1.set_title(title_str, fontsize=14, fontweight='bold')
    ax1.set_ylim([0.5, 1.5])
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add vertical lines for major time intervals
    plot_duration = rel_times_plot[-1] - rel_times_plot[0]
    if plot_duration > 3600:  # More than 1 hour
        interval = 1800  # 30 min intervals
        for t in np.arange(0, plot_duration, interval):
            ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    elif plot_duration > 600:  # More than 10 minutes
        interval = 300  # 5 min intervals
        for t in np.arange(0, plot_duration, interval):
            ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Plot 2: Burst size vs time - shows data collection patterns
    # Calculate bursts from plotted data only (same time range as plot 1)
    time_diffs_all = np.diff(rel_times)
    bursts_all = []
    current_burst = [0]
    burst_threshold = 5.0  # seconds
    
    for i in range(1, len(rel_times)):
        if time_diffs_all[i-1] <= burst_threshold:
            current_burst.append(i)
        else:
            if len(current_burst) >= 1:
                bursts_all.append(current_burst)
            current_burst = [i]
    if len(current_burst) >= 1:
        bursts_all.append(current_burst)
    
    # Filter bursts to only include those in the plotted range
    burst_start_times_sec = []
    burst_durations = []
    
    for burst in bursts_all:
        # Check if burst overlaps with plot range
        burst_start_time_abs = rel_times[burst[0]]
        burst_end_time_abs = rel_times[burst[-1]]
        
        if burst_start_time_abs >= plot_start:  # Burst is in plotted range
            start_time_plot = burst_start_time_abs - plot_start  # Relative to plot start
            duration = burst_end_time_abs - burst_start_time_abs if len(burst) > 1 else 0
            
            burst_start_times_sec.append(start_time_plot)
            burst_durations.append(duration)
    
    # Plot burst durations over time
    ax3.bar(burst_start_times_sec, burst_durations, width=2.0, alpha=0.7, 
            color='coral', edgecolor='black', linewidth=0.5)
    
    ax3.set_ylabel('Burst Duration (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Data Collection Bursts Over Time', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics text
    avg_burst_dur = np.mean([d for d in burst_durations if d > 0])
    max_burst_dur = np.max(burst_durations)
    stats_text = f'Bursts: {len(burst_durations)} | Avg: {avg_burst_dur:.1f}s | Max: {max_burst_dur:.1f}s'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add dual x-axis labels (seconds and minutes)
    # Bottom axis: seconds
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    # Top axis: minutes
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    # Convert seconds to minutes for top axis
    ax1_top.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold', color='blue')
    ax1_top.tick_params(axis='x', labelcolor='blue')
    
    # Function to convert seconds to minutes
    def sec_to_min(x):
        return x / 60
    def min_to_sec(x):
        return x * 60
    
    # Set the top axis ticks
    secax_xlim = ax1.get_xlim()
    min_ticks = np.arange(0, sec_to_min(secax_xlim[1]) + 1, 5)  # Every 5 minutes
    ax1_top.set_xticks(min_to_sec(min_ticks))
    ax1_top.set_xticklabels([f'{int(m)}' for m in min_ticks])
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{DATA_PATH}sample_timeline_with_gaps.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Timeline plot saved to: {output_file}")
    
    # Analyze bursts (use original data, not filtered)
    print(f"\n  Detailed Burst Analysis (threshold: 5s between samples):")
    burst_threshold = 5.0  # seconds - samples within this are considered a burst
    
    bursts = []
    current_burst = [0]
    
    for i in range(1, len(rel_times)):
        if time_diffs[i-1] <= burst_threshold:
            current_burst.append(i)
        else:
            if len(current_burst) >= 1:  # Include single-sample "bursts"
                bursts.append(current_burst)
            current_burst = [i]
    
    # Don't forget last burst
    if len(current_burst) >= 1:
        bursts.append(current_burst)
    
    print(f"    {'Number of bursts:':<30} {len(bursts)}")
    if len(bursts) > 0:
        burst_sizes = [len(b) for b in bursts]
        burst_durations = [rel_times[b[-1]] - rel_times[b[0]] if len(b) > 1 else 0 for b in bursts]
        
        print(f"    {'Samples per burst (mean):':<30} {np.mean(burst_sizes):.1f}")
        print(f"    {'Samples per burst (max):':<30} {np.max(burst_sizes)}")
        print(f"    {'Burst duration (mean):':<30} {np.mean(burst_durations):.1f} s")
        print(f"    {'Burst duration (max):':<30} {np.max(burst_durations):.1f} s")
        
        # Show first few bursts in detail
        print(f"\n    First 100 bursts detail:")
        for i, burst in enumerate(bursts[:100]):
            burst_start = rel_times[burst[0]]
            burst_end = rel_times[burst[-1]]
            burst_dur = burst_end - burst_start
            print(f"      Burst {i+1}: {len(burst)} samples, "
                  f"{burst_start:.1f}s - {burst_end:.1f}s ({burst_dur:.1f}s duration)")
    
    # Gap statistics
    large_gaps = time_diffs[time_diffs > 60]
    print(f"\n  Gap Statistics (>60s):")
    print(f"    {'Total gaps > 60s:':<30} {len(large_gaps)}")
    if len(large_gaps) > 0:
        print(f"    {'Mean large gap:':<30} {np.mean(large_gaps):.1f} s ({np.mean(large_gaps)/60:.1f} min)")
        print(f"    {'Max gap:':<30} {np.max(time_diffs):.1f} s ({np.max(time_diffs)/60:.1f} min)")
        print(f"    {'Min large gap:':<30} {np.min(large_gaps):.1f} s")
    
    # Show the plot
    print(f"\n  Opening plot...")
    plt.show()
    
    return rel_times, confidences, noise_values


def plot_snr_analysis(snr_data):
    """Create SNR analysis plots"""
    if snr_data is None or len(snr_data) == 0:
        return
    
    print_header("GENERATING SNR PLOTS")
    
    signal_levels = [d['signal_db'] for d in snr_data]
    noise_levels = [d['noise_db'] for d in snr_data]
    snr_values = [d['snr_db'] for d in snr_data]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: SNR Histogram
    ax1.hist(snr_values, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(snr_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(snr_values):.2f} dB')
    ax1.axvline(np.median(snr_values), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(snr_values):.2f} dB')
    ax1.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('SNR Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal Strength Histogram
    ax2.hist(signal_levels, bins=40, color='orange', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(signal_levels), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(signal_levels):.2f} dBFS')
    ax2.set_xlabel('Signal Strength (dBFS)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signal Strength Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Noise Floor Histogram
    ax3.hist(noise_levels, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(noise_levels), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(noise_levels):.2f} dBFS')
    ax3.set_xlabel('Noise Floor (dBFS)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Noise Floor Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f"{DATA_PATH}snr_histograms.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ SNR histograms saved to: {output_file}")
    
    # Create SNR vs Signal scatter plot
    fig2, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(signal_levels, snr_values, c=noise_levels, s=20, alpha=0.6, cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Noise Floor (dBFS)', rotation=270, labelpad=20, fontweight='bold')
    ax.set_xlabel('Signal Strength (dBFS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SNR vs Signal Strength (colored by Noise Floor)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f'Mean SNR: {np.mean(snr_values):.1f} dB\n'
                  f'Mean Signal: {np.mean(signal_levels):.1f} dBFS\n'
                  f'Mean Noise: {np.mean(noise_levels):.1f} dBFS\n'
                  f'Data points: {len(snr_data)}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    output_file2 = f"{DATA_PATH}snr_vs_signal.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"  ✓ SNR scatter plot saved to: {output_file2}")
    
    print(f"\n  Opening SNR plots...")
    plt.show()


def main():
    """Main analysis function"""
    try:
        # Load data
        nav_data, frames = load_data()
        
        # Run all analyses
        analyze_basic_stats(nav_data)
        analyze_hardware_config()
        analyze_timing_stats(nav_data)
        snr_data = analyze_noise_and_quality(frames)  # Returns SNR data
        analyze_per_satellite(nav_data)
        analyze_doppler_stats(nav_data)
        
        # Generate plots
        plot_sample_collection_timeline(nav_data, frames)
        plot_snr_analysis(snr_data)
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find data files")
        print(f"  {e}")
        print(f"\nMake sure EXP_NAME is set correctly at the top of this script")
        print(f"Current settings:")
        print(f"  WORKING_DIR = {WORKING_DIR}")
        print(f"  EXP_NAME = {EXP_NAME}")
        return 1
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

