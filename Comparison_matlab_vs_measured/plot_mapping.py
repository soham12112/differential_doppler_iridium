import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import traceback

# Mapping provided by user
MAPPING = {
    'sat_96': 166,
    'sat_94': 154,
    'sat_23': 165,
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

# Create individual plots for each mapping
print("\n" + "="*80)
print("CREATING INDIVIDUAL DOPPLER CURVE PLOTS")
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
        
        print(f"\nPlotting {sat_name} → {sim_sat_name}...")
        
        # Get measured data
        meas_df = measured_data[sat_name]
        meas_times_utc = pd.to_datetime(meas_df['UNIX_Timestamp'], unit='s', utc=True)
        meas_doppler = meas_df['Doppler_Frequency_Hz'].values
        
        # Get simulated data (INVERTED as per FINAL_SUMMARY.md)
        sim_doppler = -1 * sim_data[sim_sat_name].values  # Apply inversion
        sim_times_utc = sim_data['Time_UTC']
        sim_unix_time = sim_data['unix_time'].values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(sim_doppler)
        sim_times_plot = sim_times_utc[valid_mask]
        sim_doppler_plot = sim_doppler[valid_mask]
        sim_unix_plot = sim_unix_time[valid_mask]
        
        # Find overlapping time range
        if len(sim_unix_plot) > 0:
            meas_times_unix = meas_df['UNIX_Timestamp'].values
            time_start = max(meas_times_unix.min(), sim_unix_plot.min())
            time_end = min(meas_times_unix.max(), sim_unix_plot.max())
            
            # Filter to overlapping region
            meas_overlap_mask = (meas_times_unix >= time_start) & (meas_times_unix <= time_end)
            sim_overlap_mask = (sim_unix_plot >= time_start) & (sim_unix_plot <= time_end)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Full time range
            ax1.plot(meas_times_utc, meas_doppler, 'ro-', 
                    label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
            
            if sim_overlap_mask.sum() > 0:
                ax1.plot(sim_times_plot[sim_overlap_mask], sim_doppler_plot[sim_overlap_mask], 
                        'bs--', label=f'{sim_sat_name} (simulated, inverted)', 
                        linewidth=2, markersize=3, alpha=0.7)
            
            ax1.set_xlabel('UTC Time', fontsize=12)
            ax1.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
            ax1.set_title(f'Doppler Comparison: {sat_name} → {sim_sat_name}\nFull Time Range',
                         fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Overlapping region only (zoomed)
            if meas_overlap_mask.sum() > 0 and sim_overlap_mask.sum() > 0:
                ax2.plot(meas_times_utc[meas_overlap_mask], meas_doppler[meas_overlap_mask], 
                        'ro-', label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
                ax2.plot(sim_times_plot[sim_overlap_mask], sim_doppler_plot[sim_overlap_mask], 
                        'bs--', label=f'{sim_sat_name} (simulated, inverted)', 
                        linewidth=2, markersize=3, alpha=0.7)
                
                ax2.set_xlabel('UTC Time', fontsize=12)
                ax2.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
                ax2.set_title(f'Overlapping Time Region (Zoomed)',
                             fontsize=14, fontweight='bold')
                ax2.legend(fontsize=11, loc='best')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax2.tick_params(axis='x', rotation=45)
                
                # Calculate correlation for overlapping region
                from scipy.interpolate import interp1d
                from scipy.stats import pearsonr
                
                try:
                    # Interpolate simulated to measured time points
                    interp_func = interp1d(sim_unix_plot[sim_overlap_mask], 
                                          sim_doppler_plot[sim_overlap_mask],
                                          kind='linear', bounds_error=False, fill_value=np.nan)
                    sim_interp = interp_func(meas_times_unix[meas_overlap_mask])
                    
                    # Remove NaN
                    valid = ~np.isnan(sim_interp)
                    if valid.sum() > 3:
                        corr, _ = pearsonr(meas_doppler[meas_overlap_mask][valid], sim_interp[valid])
                        rmse = np.sqrt(np.mean((meas_doppler[meas_overlap_mask][valid] - sim_interp[valid])**2))
                        
                        # Add statistics text box
                        textstr = f'Correlation: {corr:.4f}\nRMSE: {rmse:.2f} Hz\nPoints: {valid.sum()}'
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                                verticalalignment='top', bbox=props)
                except Exception as e:
                    print(f"  Warning: Could not calculate correlation: {e}")
            
            plt.tight_layout()
            output_file = OUTPUT_DIR / f'{sat_name}_to_IRIDIUM{tle_id}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {output_file.name}")
    
    except Exception as e:
        print(f"  ERROR plotting {sat_name}: {e}")
        traceback.print_exc()
        plt.close('all')  # Clean up any open figures
        continue

# Create summary plot with all mappings
print("\n" + "="*80)
print("CREATING SUMMARY PLOT (ALL MAPPINGS)")
print("="*80)

# Calculate grid dimensions
n_plots = len([s for s in MAPPING.keys() if s in measured_data])
n_cols = 2
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
axes = axes.flatten() if n_plots > 1 else [axes]

plot_idx = 0
for sat_name, tle_id in MAPPING.items():
    if sat_name not in measured_data:
        continue
    
    sim_sat_name = f'IRIDIUM{tle_id}'
    
    if sim_sat_name not in sim_satellites:
        continue
    
    ax = axes[plot_idx]
    
    # Get measured data
    meas_df = measured_data[sat_name]
    meas_times_utc = pd.to_datetime(meas_df['UNIX_Timestamp'], unit='s', utc=True)
    meas_doppler = meas_df['Doppler_Frequency_Hz'].values
    
    # Get simulated data (INVERTED)
    sim_doppler = -1 * sim_data[sim_sat_name].values
    sim_times_utc = sim_data['Time_UTC']
    sim_unix_time = sim_data['unix_time'].values
    
    # Filter out NaN
    valid_mask = ~np.isnan(sim_doppler)
    sim_times_plot = sim_times_utc[valid_mask]
    sim_doppler_plot = sim_doppler[valid_mask]
    sim_unix_plot = sim_unix_time[valid_mask]
    
    # Plot
    ax.plot(meas_times_utc, meas_doppler, 'ro-', 
           label=f'{sat_name} (meas)', linewidth=2, markersize=3, zorder=5)
    
    if len(sim_doppler_plot) > 0:
        # Find overlapping region
        meas_times_unix = meas_df['UNIX_Timestamp'].values
        time_start = max(meas_times_unix.min(), sim_unix_plot.min())
        time_end = min(meas_times_unix.max(), sim_unix_plot.max())
        
        sim_overlap_mask = (sim_unix_plot >= time_start) & (sim_unix_plot <= time_end)
        
        if sim_overlap_mask.sum() > 0:
            ax.plot(sim_times_plot[sim_overlap_mask], sim_doppler_plot[sim_overlap_mask], 
                   'bs--', label=f'{sim_sat_name} (sim, inv)', 
                   linewidth=1.5, markersize=2, alpha=0.7)
    
    ax.set_xlabel('UTC Time', fontsize=9)
    ax.set_ylabel('Doppler (Hz)', fontsize=9)
    ax.set_title(f'{sat_name} → {sim_sat_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    plot_idx += 1

# Hide unused subplots
for idx in range(plot_idx, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Doppler Curve Comparison: Measured vs Simulated (All Mappings)', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()

summary_file = OUTPUT_DIR / 'all_mappings_summary.png'
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Summary plot saved: {summary_file.name}")

# Create mapping table CSV
print("\n" + "="*80)
print("CREATING MAPPING TABLE")
print("="*80)

mapping_df = pd.DataFrame([
    {'Measured_Satellite': sat, 'TLE_ID': tle, 'Simulated_Satellite': f'IRIDIUM{tle}'}
    for sat, tle in MAPPING.items()
])

mapping_file = OUTPUT_DIR / 'satellite_mapping.csv'
mapping_df.to_csv(mapping_file, index=False)
print(f"✓ Mapping table saved: {mapping_file.name}")
print("\nMapping Table:")
print(mapping_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print(f"✓ Individual plots: {n_plots} files")
print(f"✓ Summary plot: all_mappings_summary.png")
print(f"✓ Mapping table: satellite_mapping.csv")

