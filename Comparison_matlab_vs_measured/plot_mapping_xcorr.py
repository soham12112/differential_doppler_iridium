import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
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
# DATA_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_')
DATA_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/paper_compare_tle_measured')
OUTPUT_DIR = Path('/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/paper_compare_tle_measured/comp')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

def find_best_time_offset_brute_force(meas_times, meas_doppler, sim_times, sim_doppler, 
                                      max_offset_hours=6, step_minutes=1):
    """
    Use brute-force search to find the best time offset between measured and simulated data.
    More memory efficient than full cross-correlation.
    
    Parameters:
    - meas_times: Unix timestamps for measured data
    - meas_doppler: Measured Doppler values
    - sim_times: Unix timestamps for simulated data  
    - sim_doppler: Simulated Doppler values (should already be inverted if needed)
    - max_offset_hours: Maximum time offset to search (in hours)
    - step_minutes: Step size for search (in minutes)
    
    Returns:
    - best_offset: Best time offset in seconds (to add to sim_times)
    - max_corr: Maximum correlation coefficient
    - offsets: Array of all tested offsets
    - corr_values: Correlation values for each offset
    """
    
    # Create array of offsets to test (in seconds)
    step_sec = step_minutes * 60
    max_offset_sec = max_offset_hours * 3600
    offsets = np.arange(-max_offset_sec, max_offset_sec + step_sec, step_sec)
    
    corr_values = []
    
    print(f"  Testing {len(offsets)} time offsets from {-max_offset_hours:.1f}h to {max_offset_hours:.1f}h...")
    
    for i, offset in enumerate(offsets):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(offsets)} ({100*i/len(offsets):.1f}%)", end='\r')
        
        # Apply offset to simulated times
        sim_times_shifted = sim_times + offset
        
        # Find overlapping region
        time_start = max(meas_times.min(), sim_times_shifted.min())
        time_end = min(meas_times.max(), sim_times_shifted.max())
        
        if time_start >= time_end:
            corr_values.append(0)
            continue
        
        # Filter to overlapping region
        meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
        sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
        
        if meas_mask.sum() < 3 or sim_mask.sum() < 3:
            corr_values.append(0)
            continue
        
        meas_times_overlap = meas_times[meas_mask]
        meas_doppler_overlap = meas_doppler[meas_mask]
        sim_times_overlap = sim_times_shifted[sim_mask]
        sim_doppler_overlap = sim_doppler[sim_mask]
        
        # Interpolate simulated to measured time points
        try:
            interp_func = interp1d(sim_times_overlap, sim_doppler_overlap,
                                  kind='linear', bounds_error=False, fill_value=np.nan)
            sim_interpolated = interp_func(meas_times_overlap)
            
            # Remove NaN
            valid = ~np.isnan(sim_interpolated)
            if valid.sum() < 3:
                corr_values.append(0)
                continue
            
            meas_final = meas_doppler_overlap[valid]
            sim_final = sim_interpolated[valid]
            
            # Calculate correlation
            correlation, _ = pearsonr(meas_final, sim_final)
            corr_values.append(correlation if not np.isnan(correlation) else 0)
            
        except Exception:
            corr_values.append(0)
    
    print()  # New line after progress
    
    corr_values = np.array(corr_values)
    
    # Find the offset with maximum correlation
    max_idx = np.argmax(corr_values)
    best_offset = offsets[max_idx]
    max_corr = corr_values[max_idx]
    
    return best_offset, max_corr, offsets, corr_values


def calculate_metrics_with_offset(meas_times, meas_doppler, sim_times, sim_doppler, time_offset):
    """
    Calculate correlation and RMSE after applying time offset.
    
    Parameters:
    - time_offset: Time offset to add to sim_times (in seconds)
    """
    # Apply offset to simulated times
    sim_times_shifted = sim_times + time_offset
    
    # Find overlapping region
    time_start = max(meas_times.min(), sim_times_shifted.min())
    time_end = min(meas_times.max(), sim_times_shifted.max())
    
    if time_start >= time_end:
        return None, None, None, 0
    
    # Filter to overlapping region
    meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
    sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
    
    if meas_mask.sum() < 3 or sim_mask.sum() < 3:
        return None, None, None, 0
    
    meas_times_overlap = meas_times[meas_mask]
    meas_doppler_overlap = meas_doppler[meas_mask]
    sim_times_overlap = sim_times_shifted[sim_mask]
    sim_doppler_overlap = sim_doppler[sim_mask]
    
    # Interpolate simulated to measured time points
    try:
        interp_func = interp1d(sim_times_overlap, sim_doppler_overlap,
                              kind='linear', bounds_error=False, fill_value=np.nan)
        sim_interpolated = interp_func(meas_times_overlap)
        
        # Remove NaN
        valid = ~np.isnan(sim_interpolated)
        if valid.sum() < 3:
            return None, None, None, 0
        
        meas_final = meas_doppler_overlap[valid]
        sim_final = sim_interpolated[valid]
        
        # Calculate metrics
        correlation, _ = pearsonr(meas_final, sim_final)
        rmse = np.sqrt(np.mean((meas_final - sim_final)**2))
        mae = np.mean(np.abs(meas_final - sim_final))
        
        return correlation, rmse, mae, len(meas_final)
    
    except Exception as e:
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

# Analyze and plot with cross-correlation time alignment
print("\n" + "="*80)
print("CROSS-CORRELATION TIME ALIGNMENT ANALYSIS")
print("="*80)

results = {}

for sat_name, tle_id in MAPPING.items():
    try:
        if sat_name not in measured_data:
            print(f"\nSkipping {sat_name} (no data)")
            continue
        
        sim_sat_name = f'IRIDIUM{tle_id}'
        
        if sim_sat_name not in sim_satellites:
            print(f"\n⚠️  Warning: {sim_sat_name} not found in simulated data for {sat_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing {sat_name} → {sim_sat_name}")
        print(f"{'='*80}")
        
        # Get measured data
        meas_df = measured_data[sat_name]
        meas_times = meas_df['UNIX_Timestamp'].values
        meas_doppler = meas_df['Doppler_Frequency_Hz'].values
        
        # Get simulated data (INVERTED)
        sim_doppler = -1 * sim_data[sim_sat_name].values
        sim_times = sim_data['unix_time'].values
        
        # Filter out NaN from simulated
        valid_mask = ~np.isnan(sim_doppler)
        sim_times_valid = sim_times[valid_mask]
        sim_doppler_valid = sim_doppler[valid_mask]
        
        if len(sim_doppler_valid) < 3:
            print(f"  ⚠️  Insufficient simulated data")
            continue
        
        # Calculate initial time difference
        meas_center_time = (meas_times.min() + meas_times.max()) / 2
        sim_center_time = (sim_times_valid.min() + sim_times_valid.max()) / 2
        initial_offset = (meas_center_time - sim_center_time) / 3600  # in hours
        
        print(f"\nInitial time offset (center): {initial_offset:.2f} hours")
        print(f"  Measured center: {datetime.fromtimestamp(meas_center_time)}")
        print(f"  Simulated center: {datetime.fromtimestamp(sim_center_time)}")
        
        # Find best offset using brute-force search
        print("\nPerforming time offset search...")
        best_offset, max_corr, offsets, corr_values = find_best_time_offset_brute_force(
            meas_times, meas_doppler, sim_times_valid, sim_doppler_valid,
            max_offset_hours=6, step_minutes=0.5
        )
        
        print(f"  ✓ Best time offset found: {best_offset:.1f} seconds ({best_offset/3600:.3f} hours)")
        print(f"  ✓ Cross-correlation peak: {max_corr:.2e}")
        
        # Calculate metrics with the best offset
        correlation, rmse, mae, n_points = calculate_metrics_with_offset(
            meas_times, meas_doppler, sim_times_valid, sim_doppler_valid, best_offset
        )
        
        if correlation is not None:
            print(f"\nMetrics after time alignment:")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  RMSE: {rmse:.2f} Hz")
            print(f"  MAE: {mae:.2f} Hz")
            print(f"  Overlap points: {n_points}")
        else:
            print(f"  ⚠️  Could not calculate metrics (insufficient overlap)")
        
        # Store results
        results[sat_name] = {
            'sim_sat_name': sim_sat_name,
            'tle_id': tle_id,
            'best_offset_sec': best_offset,
            'best_offset_hours': best_offset / 3600,
            'initial_offset_hours': initial_offset,
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae,
            'n_points': n_points,
            'max_corr_value': max_corr
        }
        
        # Create detailed plot
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
        
        # Plot 1: Cross-correlation function (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        # Only plot reasonable range
        offset_hours = offsets / 3600
        plot_range = np.abs(offset_hours) <= 6
        ax1.plot(offset_hours[plot_range], corr_values[plot_range], 'b-', linewidth=1.5)
        ax1.axvline(best_offset/3600, color='r', linestyle='--', linewidth=2, 
                   label=f'Best offset: {best_offset/3600:.3f} h')
        ax1.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Time Offset (hours)', fontsize=11)
        ax1.set_ylabel('Normalized Cross-Correlation', fontsize=11)
        ax1.set_title('Cross-Correlation Analysis', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Offset info (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        info_text = f"""
CROSS-CORRELATION RESULTS

Initial Time Offset: {initial_offset:.3f} hours
Best Time Offset: {best_offset/3600:.3f} hours
Correction: {(best_offset/3600 - initial_offset):.3f} hours

After Alignment:
  • Correlation: {correlation:.4f if correlation else 'N/A'}
  • RMSE: {rmse:.2f if rmse else 'N/A'} Hz
  • MAE: {mae:.2f if mae else 'N/A'} Hz
  • Overlap Points: {n_points if n_points else 0}

Measured: {sat_name}
Simulated: {sim_sat_name} (TLE {tle_id})
        """
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Before alignment - UTC time (middle, full width)
        ax3 = fig.add_subplot(gs[1, :])
        meas_times_utc = pd.to_datetime(meas_times, unit='s', utc=True)
        sim_times_utc = pd.to_datetime(sim_times_valid, unit='s', utc=True)
        
        ax3.plot(meas_times_utc, meas_doppler, 'ro-', 
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
        ax3.plot(sim_times_utc, sim_doppler_valid, 'bs-',
                label=f'{sim_sat_name} (simulated, inverted)', 
                linewidth=1.5, markersize=2, alpha=0.6)
        ax3.set_xlabel('UTC Time', fontsize=11)
        ax3.set_ylabel('Doppler Frequency (Hz)', fontsize=11)
        ax3.set_title('BEFORE Time Alignment (Original UTC Times)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: After alignment - with offset applied (third row, full width)
        ax4 = fig.add_subplot(gs[2, :])
        sim_times_shifted = sim_times_valid + best_offset
        sim_times_shifted_utc = pd.to_datetime(sim_times_shifted, unit='s', utc=True)
        
        ax4.plot(meas_times_utc, meas_doppler, 'ro-',
                label=f'{sat_name} (measured)', linewidth=2.5, markersize=4, zorder=5)
        ax4.plot(sim_times_shifted_utc, sim_doppler_valid, 'bs-',
                label=f'{sim_sat_name} (time-aligned, inverted)',
                linewidth=2, markersize=3, alpha=0.7)
        ax4.set_xlabel('UTC Time', fontsize=11)
        ax4.set_ylabel('Doppler Frequency (Hz)', fontsize=11)
        ax4.set_title(f'AFTER Time Alignment (Offset: {best_offset/3600:.3f} hours)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Zoomed overlap region (bottom left)
        ax5 = fig.add_subplot(gs[3, 0])
        time_start = max(meas_times.min(), sim_times_shifted.min())
        time_end = min(meas_times.max(), sim_times_shifted.max())
        
        if time_start < time_end:
            meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
            sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
            
            meas_times_zoom = pd.to_datetime(meas_times[meas_mask], unit='s', utc=True)
            sim_times_zoom = pd.to_datetime(sim_times_shifted[sim_mask], unit='s', utc=True)
            
            ax5.plot(meas_times_zoom, meas_doppler[meas_mask], 'ro-',
                    label=f'{sat_name}', linewidth=2.5, markersize=4, zorder=5)
            ax5.plot(sim_times_zoom, sim_doppler_valid[sim_mask], 'bs-',
                    label=f'{sim_sat_name}', linewidth=2, markersize=3, alpha=0.7)
            ax5.set_xlabel('UTC Time', fontsize=10)
            ax5.set_ylabel('Doppler (Hz)', fontsize=10)
            ax5.set_title('Overlapping Region (Zoomed)', fontsize=11, fontweight='bold')
            ax5.legend(fontsize=9, loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax5.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Plot 6: Residuals (bottom right)
        ax6 = fig.add_subplot(gs[3, 1])
        if correlation is not None and n_points > 0:
            # Calculate residuals
            time_start = max(meas_times.min(), sim_times_shifted.min())
            time_end = min(meas_times.max(), sim_times_shifted.max())
            meas_mask = (meas_times >= time_start) & (meas_times <= time_end)
            sim_mask = (sim_times_shifted >= time_start) & (sim_times_shifted <= time_end)
            
            meas_times_overlap = meas_times[meas_mask]
            meas_doppler_overlap = meas_doppler[meas_mask]
            sim_times_overlap = sim_times_shifted[sim_mask]
            sim_doppler_overlap = sim_doppler_valid[sim_mask]
            
            # Interpolate
            interp_func = interp1d(sim_times_overlap, sim_doppler_overlap,
                                  kind='linear', bounds_error=False, fill_value=np.nan)
            sim_interp = interp_func(meas_times_overlap)
            valid = ~np.isnan(sim_interp)
            
            residuals = meas_doppler_overlap[valid] - sim_interp[valid]
            times_residuals = pd.to_datetime(meas_times_overlap[valid], unit='s', utc=True)
            
            ax6.plot(times_residuals, residuals, 'go-', linewidth=1.5, markersize=3)
            ax6.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
            ax6.fill_between(times_residuals, residuals, 0, alpha=0.3, color='green')
            ax6.set_xlabel('UTC Time', fontsize=10)
            ax6.set_ylabel('Residual (Hz)', fontsize=10)
            ax6.set_title(f'Residuals (Measured - Simulated)', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax6.tick_params(axis='x', rotation=45, labelsize=9)
            
            # Add statistics
            textstr = f'Mean: {np.mean(residuals):.2f} Hz\nStd: {np.std(residuals):.2f} Hz'
            ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'{sat_name} ↔ {sim_sat_name} (Cross-Correlation Time Alignment)', 
                     fontsize=16, fontweight='bold', y=0.997)
        
        output_file = OUTPUT_DIR / f'{sat_name}_to_IRIDIUM{tle_id}_xcorr.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Plot saved: {output_file.name}")
        
    except Exception as e:
        print(f"\n  ERROR analyzing {sat_name}: {e}")
        traceback.print_exc()
        plt.close('all')
        continue

# Create summary table
print("\n" + "="*80)
print("SUMMARY: TIME OFFSET ANALYSIS")
print("="*80)

if results:
    summary_df = pd.DataFrame([
        {
            'Measured': sat_name,
            'Simulated': res['sim_sat_name'],
            'TLE_ID': res['tle_id'],
            'Initial_Offset_h': res['initial_offset_hours'],
            'Best_Offset_h': res['best_offset_hours'],
            'Correction_h': res['best_offset_hours'] - res['initial_offset_hours'],
            'Correlation': res['correlation'] if res['correlation'] else np.nan,
            'RMSE_Hz': res['rmse'] if res['rmse'] else np.nan,
            'MAE_Hz': res['mae'] if res['mae'] else np.nan,
            'N_Points': res['n_points']
        }
        for sat_name, res in results.items()
    ])
    
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_file = OUTPUT_DIR / 'xcorr_time_alignment_results.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved: {summary_file.name}")
    
    # Print average offset
    avg_offset = summary_df['Best_Offset_h'].mean()
    print(f"\n📊 Average time offset: {avg_offset:.3f} hours ({avg_offset*60:.1f} minutes)")
    print(f"📊 Average correlation after alignment: {summary_df['Correlation'].mean():.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print(f"✓ Individual plots with cross-correlation analysis")
print(f"✓ Time offset summary table")

