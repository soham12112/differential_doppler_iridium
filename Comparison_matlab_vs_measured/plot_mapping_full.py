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
print("CREATING INDIVIDUAL DOPPLER CURVE PLOTS (FULL DATA)")
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
        sim_doppler_raw = sim_data[sim_sat_name].values
        sim_doppler = -1 * sim_doppler_raw  # Apply inversion
        sim_times_utc = sim_data['Time_UTC']
        
        # Filter out NaN values from simulated data
        valid_mask = ~np.isnan(sim_doppler)
        sim_times_plot = sim_times_utc[valid_mask]
        sim_doppler_plot = sim_doppler[valid_mask]
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Both on UTC time axis (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(meas_times_utc, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5, alpha=0.8)
        
        if len(sim_doppler_plot) > 0:
            ax1.plot(sim_times_plot, sim_doppler_plot, 
                    'bs-', label=f'{sim_sat_name} (simulated, inverted)', 
                    linewidth=1.5, markersize=2, alpha=0.6)
        
        ax1.set_xlabel('UTC Time', fontsize=12)
        ax1.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
        ax1.set_title(f'Doppler Comparison (UTC Time): {sat_name} → {sim_sat_name}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Measured data on relative time (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        meas_rel_time = (meas_df['UNIX_Timestamp'] - meas_df['UNIX_Timestamp'].iloc[0]) / 60.0
        ax2.plot(meas_rel_time, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4)
        ax2.set_xlabel('Relative Time (minutes from start)', fontsize=11)
        ax2.set_ylabel('Doppler Frequency (Hz)', fontsize=11)
        ax2.set_title(f'Measured: {sat_name}', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add stats
        textstr = f'Duration: {meas_rel_time.max():.2f} min\nPoints: {len(meas_doppler)}\nRange: {meas_doppler.min():.1f} to {meas_doppler.max():.1f} Hz'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Plot 3: Simulated data on relative time (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        if len(sim_doppler_plot) > 0:
            sim_unix = sim_data['unix_time'].values[valid_mask]
            sim_rel_time = (sim_unix - sim_unix[0]) / 60.0
            ax3.plot(sim_rel_time, sim_doppler_plot, 'bs-', 
                    label=f'{sim_sat_name} (sim, inv)', linewidth=2, markersize=2)
            ax3.set_xlabel('Relative Time (minutes from start)', fontsize=11)
            ax3.set_ylabel('Doppler Frequency (Hz)', fontsize=11)
            ax3.set_title(f'Simulated: {sim_sat_name} (inverted)', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Add stats
            textstr = f'Duration: {sim_rel_time.max():.2f} min\nPoints: {len(sim_doppler_plot)}\nRange: {sim_doppler_plot.min():.1f} to {sim_doppler_plot.max():.1f} Hz'
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
        
        # Plot 4: Normalized comparison (bottom, full width)
        ax4 = fig.add_subplot(gs[2, :])
        
        # Normalize both to 0-1 time scale
        meas_norm_time = (meas_rel_time - meas_rel_time.min()) / (meas_rel_time.max() - meas_rel_time.min()) if meas_rel_time.max() > meas_rel_time.min() else meas_rel_time
        ax4.plot(meas_norm_time, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
        
        if len(sim_doppler_plot) > 0 and sim_rel_time.max() > sim_rel_time.min():
            sim_norm_time = (sim_rel_time - sim_rel_time.min()) / (sim_rel_time.max() - sim_rel_time.min())
            ax4.plot(sim_norm_time, sim_doppler_plot, 'bs--', 
                    label=f'{sim_sat_name} (simulated, inverted)', linewidth=2, markersize=3, alpha=0.7)
        
        ax4.set_xlabel('Normalized Time (0 = start, 1 = end)', fontsize=12)
        ax4.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
        ax4.set_title(f'Normalized Time Comparison (Shape Comparison)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{sat_name} ↔ {sim_sat_name} (TLE ID: {tle_id})', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        output_file = OUTPUT_DIR / f'{sat_name}_to_IRIDIUM{tle_id}_full.png'
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
print("CREATING SUMMARY PLOT (ALL MAPPINGS - NORMALIZED)")
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
    
    try:
        # Get measured data
        meas_df = measured_data[sat_name]
        meas_doppler = meas_df['Doppler_Frequency_Hz'].values
        meas_rel_time = (meas_df['UNIX_Timestamp'] - meas_df['UNIX_Timestamp'].iloc[0]) / 60.0
        
        # Normalize measured time
        if meas_rel_time.max() > meas_rel_time.min():
            meas_norm_time = (meas_rel_time - meas_rel_time.min()) / (meas_rel_time.max() - meas_rel_time.min())
        else:
            meas_norm_time = meas_rel_time
        
        # Get simulated data (INVERTED)
        sim_doppler = -1 * sim_data[sim_sat_name].values
        
        # Filter out NaN
        valid_mask = ~np.isnan(sim_doppler)
        sim_doppler_plot = sim_doppler[valid_mask]
        sim_unix_plot = sim_data['unix_time'].values[valid_mask]
        
        # Plot measured
        ax.plot(meas_norm_time, meas_doppler, 'ro-', 
               label=f'{sat_name} (meas)', linewidth=2, markersize=3, zorder=5)
        
        # Plot simulated (normalized)
        if len(sim_doppler_plot) > 0:
            sim_rel_time = (sim_unix_plot - sim_unix_plot[0]) / 60.0
            if sim_rel_time.max() > sim_rel_time.min():
                sim_norm_time = (sim_rel_time - sim_rel_time.min()) / (sim_rel_time.max() - sim_rel_time.min())
            else:
                sim_norm_time = sim_rel_time
            
            ax.plot(sim_norm_time, sim_doppler_plot, 'bs--', 
                   label=f'{sim_sat_name} (sim, inv)', 
                   linewidth=1.5, markersize=2, alpha=0.7)
        
        ax.set_xlabel('Normalized Time', fontsize=9)
        ax.set_ylabel('Doppler (Hz)', fontsize=9)
        ax.set_title(f'{sat_name} → {sim_sat_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
    except Exception as e:
        print(f"  Warning: Could not plot {sat_name}: {e}")
    
    plot_idx += 1

# Hide unused subplots
for idx in range(plot_idx, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Doppler Curve Comparison (Normalized Time): Measured vs Simulated', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()

summary_file = OUTPUT_DIR / 'all_mappings_summary_full.png'
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Summary plot saved: {summary_file.name}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print(f"✓ Individual plots: {n_plots} files")
print(f"✓ Summary plot: all_mappings_summary_full.png")
print("\nNote: Simulated data is inverted (multiplied by -1)")
print("Plots show full datasets without time alignment restrictions")


