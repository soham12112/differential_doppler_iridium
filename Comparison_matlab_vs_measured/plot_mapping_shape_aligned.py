import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import traceback

# Mapping provided by user
MAPPING = {
    'sat_23': 165, #overlapping 
    'sat_96': 166,  #overlapping
    'sat_94': 154,
    'sat_3': 163,
    'sat_26': 114,
    'sat_110': 104,
    'sat_16': 111,
    'sat_112': 102,
    'sat_17': 112,
    'sat_9': 110
}

# Data paths
DATA_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/paper_compare_tle_measured')
OUTPUT_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/paper_compare_tle_measured/comp')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

def align_by_shape(meas_times, meas_doppler, sim_times, sim_doppler, debug=False):
    """
    Align simulation data to measured data based on doppler curve shape.
    Uses time-scaling and sliding window to find best overlap.
    Returns: aligned_sim_times, aligned_sim_doppler, time_offset, correlation
    """
    # Remove NaN from simulation
    valid_mask = ~np.isnan(sim_doppler)
    if valid_mask.sum() < 3:
        if debug:
            print(f"    Not enough valid sim points: {valid_mask.sum()}")
        return None, None, None, 0
    
    sim_times_valid = sim_times[valid_mask]
    sim_doppler_valid = sim_doppler[valid_mask]
    
    try:
        # Remove duplicates in measured data
        meas_unique_idx = np.unique(meas_times, return_index=True)[1]
        meas_times_unique = meas_times[meas_unique_idx]
        meas_doppler_unique = meas_doppler[meas_unique_idx]
        
        # Remove duplicates in sim data (already filtered for valid)
        sim_unique_idx = np.unique(sim_times_valid, return_index=True)[1]
        sim_times_unique = sim_times_valid[sim_unique_idx]
        sim_doppler_unique = sim_doppler_valid[sim_unique_idx]
        
        if len(meas_times_unique) < 3 or len(sim_times_unique) < 3:
            if debug:
                print(f"    Not enough unique points after deduplication")
            return None, None, None, 0
        
        # Step 1: Scale sim duration to match measured duration
        meas_duration = meas_times_unique[-1] - meas_times_unique[0]
        sim_duration = sim_times_unique[-1] - sim_times_unique[0]
        time_scale = meas_duration / sim_duration
        
        # Rescale sim times (but keep them starting at their original offset)
        sim_times_scaled = (sim_times_unique - sim_times_unique[0]) * time_scale + sim_times_unique[0]
        
        # Step 2: Find best time offset using sliding correlation
        # Create interpolation function for sim data
        sim_interp_func = interp1d(sim_times_scaled, sim_doppler_unique,
                                   kind='linear', bounds_error=False, fill_value=np.nan)
        
        # Try different time offsets
        meas_center = (meas_times_unique[0] + meas_times_unique[-1]) / 2
        sim_center = (sim_times_scaled[0] + sim_times_scaled[-1]) / 2
        
        # Search range: +/- 2x the duration around center alignment
        search_range = 2 * max(meas_duration, sim_duration * time_scale)
        initial_offset = meas_center - sim_center
        
        # Try offsets in steps
        n_steps = 100
        offsets = np.linspace(initial_offset - search_range, initial_offset + search_range, n_steps)
        correlations = []
        
        for offset in offsets:
            # Shift sim times
            sim_times_shifted = sim_times_scaled + offset
            
            # Interpolate sim to meas time points
            sim_at_meas = sim_interp_func(sim_times_shifted)
            
            # Only use overlapping region
            valid_overlap = ~np.isnan(sim_at_meas)
            if valid_overlap.sum() < 3:
                correlations.append(-1)
                continue
            
            # Calculate correlation
            try:
                corr, _ = pearsonr(meas_doppler_unique[valid_overlap], sim_at_meas[valid_overlap])
                correlations.append(corr if np.isfinite(corr) else -1)
            except:
                correlations.append(-1)
        
        # Find best offset
        correlations = np.array(correlations)
        if correlations.max() <= -1:
            if debug:
                print(f"    No valid correlations found")
            return None, None, None, 0
        
        best_idx = np.argmax(correlations)
        best_offset = offsets[best_idx]
        best_corr = correlations[best_idx]
        
        # Apply best offset
        sim_times_aligned = sim_times_scaled + best_offset
        
        return sim_times_aligned, sim_doppler_unique, best_offset, best_corr
        
    except Exception as e:
        if debug:
            print(f"    Alignment error: {e}")
            traceback.print_exc()
        return None, None, None, 0


# Load simulated data
print("\nLoading simulated data...")
try:
    sim_file = DATA_DIR / 'ue1_doppler_data.csv'
    sim_data = pd.read_csv(sim_file, parse_dates=['Time_UTC'])
    sim_data['unix_time'] = sim_data['Time_UTC'].apply(lambda x: x.timestamp())
    print(f"✓ Loaded simulated data from {sim_file.name}")
    print(f"  Time range: {sim_data['Time_UTC'].min()} to {sim_data['Time_UTC'].max()}")
except Exception as e:
    print(f"ERROR loading simulated data: {e}")
    traceback.print_exc()
    exit(1)

# Get list of simulated satellites
sim_satellites = [col for col in sim_data.columns if col not in ['Time_UTC', 'unix_time']]
print(f"  Available simulated satellites: {len(sim_satellites)}")

# Load measured data
print("\nLoading measured data...")
measured_data = {}
for sat_name in MAPPING.keys():
    meas_file = DATA_DIR / f'{sat_name}_ira_doppler.csv'
    if meas_file.exists():
        df = pd.read_csv(meas_file, comment='#')
        measured_data[sat_name] = df
        start_time = datetime.fromtimestamp(df['UNIX_Timestamp'].iloc[0])
        end_time = datetime.fromtimestamp(df['UNIX_Timestamp'].iloc[-1])
        print(f"✓ {sat_name}: {start_time} to {end_time} ({len(df)} points)")
    else:
        print(f"⚠️  {sat_name}: File not found")

print(f"\nTotal measured satellites loaded: {len(measured_data)}/{len(MAPPING)}")

# Store alignment results
alignment_results = []

# Create individual plots for each mapping
print("\n" + "="*80)
print("CREATING SHAPE-ALIGNED DOPPLER CURVE PLOTS")
print("="*80)

for sat_name, tle_id in MAPPING.items():
    try:
        if sat_name not in measured_data:
            print(f"Skipping {sat_name} (no data)")
            continue
        
        sim_sat_name = f'IRIDIUM{tle_id}'
        
        # Check if simulated satellite exists
        if sim_sat_name not in sim_satellites:
            print(f"⚠️  Warning: {sim_sat_name} not found in simulated data for {sat_name}")
            continue
        
        print(f"\nProcessing {sat_name} → {sim_sat_name}...")
        
        # Get measured data
        meas_df = measured_data[sat_name]
        meas_times = meas_df['UNIX_Timestamp'].values
        meas_doppler = meas_df['Doppler_Frequency_Hz'].values
        
        # Get simulated data
        sim_times = sim_data['unix_time'].values
        sim_doppler_raw = sim_data[sim_sat_name].values
        
        # Try both normal and inverted
        best_corr = -1
        best_alignment = None
        best_inverted = False
        
        for inverted in [False, True]:
            sim_doppler = -1 * sim_doppler_raw if inverted else sim_doppler_raw
            
            sim_times_aligned, sim_doppler_aligned, time_offset, corr = align_by_shape(
                meas_times, meas_doppler, sim_times, sim_doppler, debug=True
            )
            
            if sim_times_aligned is not None and corr > best_corr:
                best_corr = corr
                best_alignment = (sim_times_aligned, sim_doppler_aligned, time_offset)
                best_inverted = inverted
        
        if best_alignment is None:
            print(f"  ⚠️  Could not align {sat_name}")
            continue
        
        sim_times_aligned, sim_doppler_aligned, time_offset = best_alignment
        
        print(f"  ✓ Best alignment: inverted={best_inverted}, time_offset={time_offset:.1f}s, corr={best_corr:.4f}")
        
        # Store results
        alignment_results.append({
            'measured_sat': sat_name,
            'sim_sat': sim_sat_name,
            'inverted': best_inverted,
            'time_offset_sec': time_offset,
            'correlation': best_corr
        })
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Convert times to datetime for plotting
        meas_times_dt = pd.to_datetime(meas_times, unit='s', utc=True)
        sim_times_aligned_dt = pd.to_datetime(sim_times_aligned, unit='s', utc=True)
        
        # Plot 1: Aligned curves
        ax1.plot(meas_times_dt, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
        ax1.plot(sim_times_aligned_dt, sim_doppler_aligned, 'bs--', 
                label=f'{sim_sat_name} (sim, {"inverted" if best_inverted else "normal"})', 
                linewidth=2, markersize=3, alpha=0.7)
        
        ax1.set_xlabel('Time (Aligned)', fontsize=12)
        ax1.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
        ax1.set_title(f'Shape-Aligned Doppler Comparison: {sat_name} → {sim_sat_name}\n' +
                     f'Time Offset: {time_offset:.1f}s, Correlation: {best_corr:.4f}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Normalized time (starting from 0)
        meas_times_norm = (meas_times - meas_times[0]) / 60  # minutes
        sim_times_norm = (sim_times_aligned - meas_times[0]) / 60  # minutes
        
        ax2.plot(meas_times_norm, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
        ax2.plot(sim_times_norm, sim_doppler_aligned, 'bs--', 
                label=f'{sim_sat_name} (sim, {"inverted" if best_inverted else "normal"})', 
                linewidth=2, markersize=3, alpha=0.7)
        
        ax2.set_xlabel('Time from Start (minutes)', fontsize=12)
        ax2.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
        ax2.set_title(f'Normalized Time View', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Calculate RMSE
        sim_interp = interp1d(sim_times_aligned, sim_doppler_aligned,
                             kind='linear', bounds_error=False, fill_value=np.nan)
        sim_at_meas = sim_interp(meas_times)
        valid = ~np.isnan(sim_at_meas)
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((meas_doppler[valid] - sim_at_meas[valid])**2))
            
            # Add statistics text box
            textstr = f'Correlation: {best_corr:.4f}\nRMSE: {rmse:.2f} Hz\n'
            textstr += f'Time Offset: {time_offset:.1f} s\nInverted: {best_inverted}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f'{sat_name}_to_{sim_sat_name}_shape_aligned.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_file.name}")
    
    except Exception as e:
        print(f"  ERROR plotting {sat_name}: {e}")
        traceback.print_exc()
        plt.close('all')
        continue

# Save alignment results
if alignment_results:
    results_df = pd.DataFrame(alignment_results)
    results_file = OUTPUT_DIR / 'shape_alignment_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Alignment results saved: {results_file.name}")
    print("\nAlignment Summary:")
    print(results_df.to_string(index=False))

# Create summary plot with all mappings
print("\n" + "="*80)
print("CREATING SUMMARY PLOT (ALL SHAPE-ALIGNED MAPPINGS)")
print("="*80)

n_plots = len(alignment_results)
if n_plots > 0:
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, result in enumerate(alignment_results):
        ax = axes[idx]
        
        sat_name = result['measured_sat']
        sim_sat_name = result['sim_sat']
        inverted = result['inverted']
        time_offset = result['time_offset_sec']
        corr = result['correlation']
        
        # Get measured data
        meas_df = measured_data[sat_name]
        meas_times = meas_df['UNIX_Timestamp'].values
        meas_doppler = meas_df['Doppler_Frequency_Hz'].values
        
        # Get and align simulated data
        sim_times = sim_data['unix_time'].values
        sim_doppler_raw = sim_data[sim_sat_name].values
        sim_doppler = -1 * sim_doppler_raw if inverted else sim_doppler_raw
        
        # Apply alignment
        sim_times_aligned, sim_doppler_aligned, _, _ = align_by_shape(
            meas_times, meas_doppler, sim_times, sim_doppler
        )
        
        if sim_times_aligned is not None:
            # Normalize time to minutes from start
            meas_times_norm = (meas_times - meas_times[0]) / 60
            sim_times_norm = (sim_times_aligned - meas_times[0]) / 60
            
            ax.plot(meas_times_norm, meas_doppler, 'ro-', 
                   label=f'{sat_name} (meas)', linewidth=2, markersize=3, zorder=5)
            ax.plot(sim_times_norm, sim_doppler_aligned, 'bs--', 
                   label=f'{sim_sat_name} (sim)', linewidth=1.5, markersize=2, alpha=0.7)
            
            ax.set_xlabel('Time from Start (min)', fontsize=9)
            ax.set_ylabel('Doppler (Hz)', fontsize=9)
            ax.set_title(f'{sat_name} → {sim_sat_name}\nCorr: {corr:.3f}, Offset: {time_offset:.0f}s', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Shape-Aligned Doppler Curve Comparison (All Mappings)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    summary_file = OUTPUT_DIR / 'all_mappings_shape_aligned_summary.png'
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Summary plot saved: {summary_file.name}")

print("\n" + "="*80)
print("SHAPE-ALIGNED ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print(f"✓ Individual plots: {len(alignment_results)} files")
print(f"✓ Summary plot: all_mappings_shape_aligned_summary.png")
print(f"✓ Alignment results: shape_alignment_results.csv")

