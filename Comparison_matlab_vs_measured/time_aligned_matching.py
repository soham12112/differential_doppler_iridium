import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import correlate
from math import ceil

# ==========================================
# UTILITY: ALIGN SIGNALS
# ==========================================
def align_and_score(t_meas, y_meas, t_sim, y_sim):
    """
    1. Resamples to uniform grid
    2. Cross-correlates to find time lag
    3. Aligns time
    4. Calculates Frequency Offset
    5. Calculates RMSE on centered data
    """
    # Remove NaNs
    mask_meas = ~np.isnan(y_meas)
    mask_sim = ~np.isnan(y_sim)
    if mask_meas.sum() < 10 or mask_sim.sum() < 10: return None
    
    t_m, y_m = t_meas[mask_meas], y_meas[mask_meas]
    t_s, y_s = t_sim[mask_sim], y_sim[mask_sim]
    
    # 1. Uniform Resampling (10Hz) for Correlation
    dt = 0.1 
    duration = min(t_m.max() - t_m.min(), t_s.max() - t_s.min())
    if duration <= 0: return None
    
    # Create common time axis relative to start
    num_points = int(duration / dt)
    if num_points < 10: return None
    t_grid = np.linspace(0, duration, num_points)
    
    # Interpolate both to grid (relative to their own start times)
    f_m = interp1d(t_m - t_m[0], y_m, bounds_error=False, fill_value=0)
    f_s = interp1d(t_s - t_s[0], y_s, bounds_error=False, fill_value=0)
    
    y_m_grid = f_m(t_grid)
    y_s_grid = f_s(t_grid)
    
    # 2. Cross Correlation on Centered Data (Shape Matching)
    y_m_centered = y_m_grid - np.mean(y_m_grid)
    y_s_centered = y_s_grid - np.mean(y_s_grid)
    
    corr = correlate(y_m_centered, y_s_centered, mode='full')
    lags = np.arange(-(len(t_grid)-1), len(t_grid))
    best_lag_idx = np.argmax(corr)
    best_lag = lags[best_lag_idx]
    time_shift = best_lag * dt # Time shift in seconds
    
    # Pearson Correlation Coefficient at best lag
    # Re-extract overlapping segments
    if best_lag >= 0:
        y_m_overlap = y_m_centered[best_lag:]
        y_s_overlap = y_s_centered[:len(y_m_overlap)]
    else:
        y_s_overlap = y_s_centered[-best_lag:]
        y_m_overlap = y_m_centered[:len(y_s_overlap)]
        
    if len(y_m_overlap) < 10: return None
    pearson = np.corrcoef(y_m_overlap, y_s_overlap)[0, 1]
    
    # 3. Calculate Physical Offsets
    # Real Time Offset = (t_meas_start - t_sim_start) + time_shift
    abs_time_diff = (t_m[0] - t_s[0]) - time_shift 
    
    # Frequency Offset (Difference in means of overlapping regions)
    f_s_aligned = interp1d(t_s + abs_time_diff, y_s, bounds_error=False, fill_value=np.nan)
    y_s_aligned = f_s_aligned(t_m)
    
    valid_mask = ~np.isnan(y_s_aligned)
    if valid_mask.sum() < 10: return None
    
    diff = y_m[valid_mask] - y_s_aligned[valid_mask]
    freq_offset = np.mean(diff)
    rmse = np.sqrt(np.mean((diff - freq_offset)**2)) # RMSE after removing constant offset
    
    return {
        'pearson': pearson,
        'rmse': rmse,
        'freq_offset': freq_offset,
        'time_offset': abs_time_diff,
        'aligned_sim_doppler': y_s_aligned
    }

# ==========================================
# MAIN SCRIPT
# ==========================================
def load_data():
    # Load Sim
    ues = {}
    for ue in ['ue1', 'ue2']:
        files = list(Path(".").glob(f"{ue}_doppler_data*.csv"))
        if files:
            df = pd.read_csv(files[0], parse_dates=['Time_UTC'])
            df['unix_time'] = df['Time_UTC'].apply(lambda x: x.timestamp())
            ues[ue] = df
            print(f"Loaded {ue}")
            
    # Load Meas
    meas = {}
    for p in Path(".").glob("sat_*_ira_doppler.csv"):
        if "corrected" not in p.stem:
            meas[p.stem.replace("_ira_doppler", "")] = pd.read_csv(p, comment='#')
            
    return ues, meas

ues, measured_data = load_data()
if not ues or not measured_data: raise RuntimeError("Data missing")

print("\n" + "="*80)
print("TIME & FREQUENCY ALIGNED ANALYSIS")
print("="*80)

best_matches = {}
results_list = []

for m_name, m_df in measured_data.items():
    t_meas = m_df['UNIX_Timestamp'].values
    y_meas = m_df['Doppler_Frequency_Hz'].values
    
    candidates = []
    
    for ue_name, ue_df in ues.items():
        sim_sats = [c for c in ue_df.columns if c not in ['Time_UTC', 'unix_time']]
        t_sim = ue_df['unix_time'].values
        
        for sat in sim_sats:
            y_sim = ue_df[sat].values
            
            # Check Normal
            res_norm = align_and_score(t_meas, y_meas, t_sim, y_sim)
            if res_norm:
                candidates.append({**res_norm, 'ue': ue_name, 'sat': sat, 'inv': False})
                
            # Check Inverted
            res_inv = align_and_score(t_meas, y_meas, t_sim, -1 * y_sim)
            if res_inv:
                candidates.append({**res_inv, 'ue': ue_name, 'sat': sat, 'inv': True})

    # Sort candidates by Correlation
    if candidates:
        candidates.sort(key=lambda x: x['pearson'], reverse=True)
        best = candidates[0]
        best_matches[m_name] = best
        
        # Save for CSV
        results_list.append({
            'measured_sat': m_name,
            'identified_sat': best['sat'],
            'ue': best['ue'],
            'inverted': best['inv'],
            'confidence_score': best['pearson'],
            'rmse_hz': best['rmse'],
            'freq_offset_hz': best['freq_offset'],
            'time_lag_sec': best['time_offset']
        })
        
        inv_flag = "YES" if best['inv'] else "NO"
        print(f"{m_name:<10} → {best['sat']:<15} (Conf: {best['pearson']:.2f})")

# SAVE RESULTS TO CSV FOR RUN_MATCHING.PY
results_df = pd.DataFrame(results_list)
results_df.to_csv('time_aligned_matches.csv', index=False)
print("\nResults saved to time_aligned_matches.csv")

# ==========================================
# VISUALIZATION
# ==========================================
if best_matches:
    print("Generating aligned plots...")
    cols = 2
    rows = ceil(len(best_matches) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if len(best_matches) == 1: axes = [axes]
    else: axes = axes.flatten()

    for idx, (m_name, match) in enumerate(best_matches.items()):
        ax = axes[idx]
        
        # Plot Measured (Raw)
        m_df = measured_data[m_name]
        t_utc = pd.to_datetime(m_df['UNIX_Timestamp'], unit='s')
        ax.plot(t_utc, m_df['Doppler_Frequency_Hz'], 'k.', label='Measured', markersize=3, zorder=5)
        
        # Plot Sim (Corrected)
        ue_df = ues[match['ue']]
        t_sim = ue_df['unix_time'].values
        y_sim = ue_df[match['sat']].values
        if match['inv']: y_sim = -1 * y_sim
        
        # Apply Calculated Offsets for Visualization
        t_sim_corrected = t_sim + match['time_offset']
        y_sim_corrected = y_sim + match['freq_offset']
        
        # Convert to datetime for plotting
        t_sim_dt = pd.to_datetime(t_sim_corrected, unit='s')
        
        # Plot Sim
        ax.plot(t_sim_dt, y_sim_corrected, 'r-', alpha=0.7, linewidth=1.5,
                label=f"Sim {match['sat']}")
                
        # Formatting
        ax.set_title(f"{m_name} → {match['sat']} (Conf: {match['pearson']:.2f})\nRMSE: {match['rmse']:.1f}Hz")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('time_aligned_matching_results.png')
    print("Visualization saved.")