#!/usr/bin/env python3
"""
Simple time alignment tool using correlation-based matching.
Finds optimal time offset between measured and simulated Doppler curves.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# Mapping provided by user
MAPPING = {
    'sat_25': 155,
    'sat_67': 167,
    'sat_71': 126,
    'sat_89': 133,
    'sat_74': 144,
    'sat_73': 100,
    'sat_46': 156,
    'sat_17': 112,
    'sat_18': 158,
    'sat_110': 104
}

# Data paths
DATA_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_28th_night')
OUTPUT_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Comparison_matlab_vs_measured/comp')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_correlation_with_offset(meas_times, meas_doppler, sim_times, sim_doppler, offset_sec):
    """Calculate Pearson correlation after applying time offset to simulated data."""
    sim_times_shifted = sim_times + offset_sec
    
    # Find overlap
    time_start = max(meas_times.min(), sim_times_shifted.min())
    time_end = min(meas_times.max(), sim_times_shifted.max())
    
    if time_start >= time_end:
        return 0, 0
    
    # Filter to overlap
    meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
    sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
    
    if meas_mask.sum() < 3 or sim_mask.sum() < 3:
        return 0, 0
    
    meas_times_overlap = meas_times[meas_mask]
    meas_doppler_overlap = meas_doppler[meas_mask]
    sim_times_overlap = sim_times_shifted[sim_mask]
    sim_doppler_overlap = sim_doppler[sim_mask]
    
    # Interpolate simulated to measured time points
    try:
        interp_func = interp1d(sim_times_overlap, sim_doppler_overlap,
                              kind='linear', bounds_error=False, fill_value=np.nan)
        sim_interp = interp_func(meas_times_overlap)
        
        valid = ~np.isnan(sim_interp)
        if valid.sum() < 3:
            return 0, 0
        
        corr, _ = pearsonr(meas_doppler_overlap[valid], sim_interp[valid])
        return corr if not np.isnan(corr) else 0, valid.sum()
    except:
        return 0, 0


print("="*80)
print("SIMPLE TIME ALIGNMENT TOOL")
print("="*80)

# Load simulated data
print("\nLoading simulated data...")
sim_file = DATA_DIR / 'ue1_doppler_data.csv'
sim_data = pd.read_csv(sim_file, parse_dates=['Time_UTC'])
sim_data['unix_time'] = sim_data['Time_UTC'].apply(lambda x: x.timestamp())
print(f"✓ Loaded {sim_file.name}")

# Get simulated satellites
sim_satellites = [col for col in sim_data.columns if col not in ['Time_UTC', 'unix_time']]

# Load measured data
print("\nLoading measured data...")
measured_data = {}
for sat_name in MAPPING.keys():
    meas_file = DATA_DIR / f'{sat_name}_ira_doppler.csv'
    if meas_file.exists():
        df = pd.read_csv(meas_file, comment='#')
        measured_data[sat_name] = df
        print(f"✓ {sat_name}: {len(df)} points")

print(f"\nTotal: {len(measured_data)} satellites")

# Analyze each satellite pair
results = []

for sat_name, tle_id in MAPPING.items():
    if sat_name not in measured_data:
        continue
    
    sim_sat_name = f'IRIDIUM{tle_id}'
    if sim_sat_name not in sim_satellites:
        continue
    
    print(f"\n{'='*80}")
    print(f"{sat_name} → {sim_sat_name}")
    print(f"{'='*80}")
    
    # Get data
    meas_df = measured_data[sat_name]
    meas_times = meas_df['UNIX_Timestamp'].values
    meas_doppler = meas_df['Doppler_Frequency_Hz'].values
    
    sim_doppler_raw = sim_data[sim_sat_name].values
    sim_doppler = -1 * sim_doppler_raw  # Invert
    sim_times = sim_data['unix_time'].values
    
    # Filter NaN
    valid_mask = ~np.isnan(sim_doppler)
    sim_times = sim_times[valid_mask]
    sim_doppler = sim_doppler[valid_mask]
    
    # Calculate initial offset
    meas_center = (meas_times.min() + meas_times.max()) / 2
    sim_center = (sim_times.min() + sim_times.max()) / 2
    initial_offset_hours = (meas_center - sim_center) / 3600
    
    print(f"Initial offset: {initial_offset_hours:.3f} hours")
    
    # Search for best offset (±3 hours around initial guess, every 30 seconds)
    search_range_hours = 3
    step_seconds = 30
    
    base_offset = initial_offset_hours * 3600  # Convert to seconds
    search_offsets = np.arange(
        base_offset - search_range_hours * 3600,
        base_offset + search_range_hours * 3600 + step_seconds,
        step_seconds
    )
    
    print(f"Searching {len(search_offsets)} offsets...")
    
    correlations = []
    for offset in search_offsets:
        corr, n_pts = calculate_correlation_with_offset(
            meas_times, meas_doppler, sim_times, sim_doppler, offset
        )
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # Find best
    best_idx = np.argmax(correlations)
    best_offset = search_offsets[best_idx]
    best_corr = correlations[best_idx]
    
    print(f"✓ Best offset: {best_offset/3600:.3f} hours")
    print(f"✓ Best correlation: {best_corr:.4f}")
    
    # Calculate metrics at best offset
    sim_times_shifted = sim_times + best_offset
    time_start = max(meas_times.min(), sim_times_shifted.min())
    time_end = min(meas_times.max(), sim_times_shifted.max())
    
    meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
    sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
    
    meas_times_overlap = meas_times[meas_mask]
    meas_doppler_overlap = meas_doppler[meas_mask]
    sim_times_overlap = sim_times_shifted[sim_mask]
    sim_doppler_overlap = sim_doppler[sim_mask]
    
    interp_func = interp1d(sim_times_overlap, sim_doppler_overlap,
                          kind='linear', bounds_error=False, fill_value=np.nan)
    sim_interp = interp_func(meas_times_overlap)
    valid = ~np.isnan(sim_interp)
    
    rmse = np.sqrt(np.mean((meas_doppler_overlap[valid] - sim_interp[valid])**2))
    print(f"✓ RMSE: {rmse:.2f} Hz")
    print(f"✓ Overlap points: {valid.sum()}")
    
    # Store results
    results.append({
        'Measured': sat_name,
        'Simulated': sim_sat_name,
        'TLE_ID': tle_id,
        'Initial_Offset_h': initial_offset_hours,
        'Best_Offset_h': best_offset / 3600,
        'Correction_h': (best_offset / 3600) - initial_offset_hours,
        'Correlation': best_corr,
        'RMSE_Hz': rmse,
        'N_Points': valid.sum()
    })
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Correlation vs offset
    ax = axes[0]
    ax.plot((search_offsets - base_offset) / 3600, correlations, 'b-', linewidth=2)
    ax.axvline((best_offset - base_offset) / 3600, color='r', linestyle='--', linewidth=2,
              label=f'Best: {(best_offset-base_offset)/3600:.3f}h')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Offset from Initial Guess (hours)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'Correlation vs Time Offset', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Before alignment
    ax = axes[1]
    meas_utc = pd.to_datetime(meas_times, unit='s', utc=True)
    sim_utc = pd.to_datetime(sim_times, unit='s', utc=True)
    ax.plot(meas_utc, meas_doppler, 'ro-', label=sat_name, linewidth=2, markersize=3)
    ax.plot(sim_utc, sim_doppler, 'bs-', label=sim_sat_name, linewidth=1.5, markersize=2, alpha=0.6)
    ax.set_xlabel('UTC Time', fontsize=12)
    ax.set_ylabel('Doppler (Hz)', fontsize=12)
    ax.set_title(f'BEFORE Alignment (offset: {initial_offset_hours:.3f}h)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: After alignment
    ax = axes[2]
    sim_utc_shifted = pd.to_datetime(sim_times_shifted, unit='s', utc=True)
    ax.plot(meas_utc, meas_doppler, 'ro-', label=sat_name, linewidth=2, markersize=3)
    ax.plot(sim_utc_shifted, sim_doppler, 'bs-', label=f'{sim_sat_name} (aligned)', 
           linewidth=2, markersize=2.5, alpha=0.7)
    ax.set_xlabel('UTC Time', fontsize=12)
    ax.set_ylabel('Doppler (Hz)', fontsize=12)
    ax.set_title(f'AFTER Alignment (offset: {best_offset/3600:.3f}h, corr: {best_corr:.4f})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(f'{sat_name} ↔ {sim_sat_name} (Time Alignment)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f'{sat_name}_to_IRIDIUM{tle_id}_aligned.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file.name}")

# Save summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

summary_file = OUTPUT_DIR / 'time_alignment_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved: {summary_file.name}")

avg_offset = summary_df['Best_Offset_h'].mean()
avg_corr = summary_df['Correlation'].mean()
print(f"\n📊 Average time offset: {avg_offset:.3f} hours ({avg_offset*60:.1f} minutes)")
print(f"📊 Average correlation: {avg_corr:.4f}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)



