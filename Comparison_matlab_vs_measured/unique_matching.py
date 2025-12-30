import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
from scipy.optimize import linear_sum_assignment

# ==========================================
# FEATURE EXTRACTION HELPER
# ==========================================
def extract_doppler_features(times, doppler_values):
    """Extract key features from a Doppler signature"""
    if len(doppler_values) < 3:
        return None
    
    valid_mask = ~np.isnan(doppler_values)
    if valid_mask.sum() < 3:
        return None
    
    times_valid = times[valid_mask]
    doppler_valid = doppler_values[valid_mask]
    
    # Duration in minutes
    duration = (times_valid[-1] - times_valid[0]) / 60 if len(times_valid) > 1 else 0
    
    # Basic Stats
    doppler_min = np.min(doppler_valid)
    doppler_max = np.max(doppler_valid)
    doppler_range = doppler_max - doppler_min
    doppler_mean = np.mean(doppler_valid)
    
    # Rate (Hz/sec)
    if len(times_valid) > 1:
        time_diffs = np.diff(times_valid)
        # Filter out zero or very small time differences to avoid division issues
        valid_time_mask = time_diffs > 1e-6
        if valid_time_mask.sum() > 0:
            doppler_rates = np.diff(doppler_valid)[valid_time_mask] / time_diffs[valid_time_mask]
            # Filter out inf and nan values
            valid_rate_mask = np.isfinite(doppler_rates)
            if valid_rate_mask.sum() > 0:
                avg_doppler_rate = np.mean(doppler_rates[valid_rate_mask])
                max_doppler_rate = np.max(np.abs(doppler_rates[valid_rate_mask]))
            else:
                avg_doppler_rate = 0
                max_doppler_rate = 0
        else:
            avg_doppler_rate = 0
            max_doppler_rate = 0
    else:
        avg_doppler_rate = 0
        max_doppler_rate = 0
    
    # Peak location (Normalized 0.0 to 1.0)
    peak_idx = np.argmax(doppler_valid)
    peak_location_norm = peak_idx / (len(doppler_valid) - 1) if len(doppler_valid) > 1 else 0.5
    
    # Trend (End - Start)
    if len(doppler_valid) >= 3:
        first_third = doppler_valid[:len(doppler_valid)//3]
        last_third = doppler_valid[-len(doppler_valid)//3:]
        trend = np.mean(last_third) - np.mean(first_third)
    else:
        trend = doppler_valid[-1] - doppler_valid[0]
    
    return {
        'duration': duration,
        'doppler_min': doppler_min,
        'doppler_max': doppler_max,
        'doppler_range': doppler_range,
        'doppler_mean': doppler_mean,
        'avg_doppler_rate': avg_doppler_rate,
        'max_doppler_rate': max_doppler_rate,
        'peak_location_norm': peak_location_norm,
        'trend': trend
    }

def compare_signatures(feat1, feat2):
    """
    Compare two Doppler signatures.
    Returns: score (0-1)
    """
    if feat1 is None or feat2 is None:
        return 0
    
    # Check for invalid values in features
    for feat in [feat1, feat2]:
        for key, val in feat.items():
            if not np.isfinite(val):
                return 0
    
    weights = {
        'doppler_range': 0.30,      # Range is critical
        'avg_doppler_rate': 0.30,   # Rate is critical
        'trend': 0.20,              # Direction must match
        'peak_location': 0.10,
        'duration': 0.10
    }
    
    scores = {}
    
    # Range Score
    range_diff = abs(feat1['doppler_range'] - feat2['doppler_range'])
    max_range = max(feat1['doppler_range'], feat2['doppler_range'])
    scores['doppler_range'] = 1 - min(range_diff / max_range, 1) if max_range > 0 else 0
    
    # Rate Score
    rate_diff = abs(feat1['avg_doppler_rate'] - feat2['avg_doppler_rate'])
    max_rate = max(abs(feat1['avg_doppler_rate']), abs(feat2['avg_doppler_rate']))
    scores['avg_doppler_rate'] = 1 - min(rate_diff / max_rate, 1) if max_rate > 0 else 0
    
    # Trend Score (Must match sign)
    if np.sign(feat1['trend']) == np.sign(feat2['trend']):
        trend_diff = abs(feat1['trend'] - feat2['trend'])
        max_trend = max(abs(feat1['trend']), abs(feat2['trend']))
        scores['trend'] = 1 - min(trend_diff / max_trend, 1) if max_trend > 0 else 0.5
    else:
        scores['trend'] = 0 # Penalty for wrong direction
    
    # Peak Location Score
    peak_diff = abs(feat1['peak_location_norm'] - feat2['peak_location_norm'])
    scores['peak_location'] = 1 - min(peak_diff, 1)
    
    # Duration Score
    dur_diff = abs(feat1['duration'] - feat2['duration'])
    max_dur = max(feat1['duration'], feat2['duration'])
    scores['duration'] = 1 - min(dur_diff / max_dur, 1) if max_dur > 0 else 0
    
    total_score = sum(scores[k] * weights[k] for k in weights)
    return total_score

# ==========================================
# DATA LOADING
# ==========================================
def load_simulated(prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_doppler_data*.csv"
    candidates = sorted(Path(".").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern}")
    df = pd.read_csv(candidates[0], parse_dates=['Time_UTC'])
    print(f"Loaded {candidates[0].name}")
    return df

print("Loading data...")
ue_data = {}
try:
    ue1 = load_simulated("ue1")
    ue1['unix_time'] = ue1['Time_UTC'].apply(lambda x: x.timestamp())
    ue_data['ue1'] = ue1
except FileNotFoundError:
    pass

try:
    ue2 = load_simulated("ue2")
    ue2['unix_time'] = ue2['Time_UTC'].apply(lambda x: x.timestamp())
    ue_data['ue2'] = ue2
except FileNotFoundError:
    pass

if not ue_data:
    raise RuntimeError("No UE data files found.")

# Load Measured
measured_data = {}
meas_paths = sorted(p for p in Path(".").glob("sat_*_ira_doppler.csv") if "corrected" not in p.stem)
if not meas_paths:
    raise RuntimeError("No measured files found.")

for p in meas_paths:
    measured_data[p.stem.replace("_ira_doppler", "")] = pd.read_csv(p, comment='#')

# ==========================================
# 1. FEATURE EXTRACTION
# ==========================================
print("\n" + "="*80)
print("OPTIMAL MATCHING (HUNGARIAN ALGORITHM)")
print("="*80)

meas_features = {}
for name, df in measured_data.items():
    meas_features[name] = extract_doppler_features(df['UNIX_Timestamp'].values, df['Doppler_Frequency_Hz'].values)

sim_features = {}
# Extract features for every sim sat (Normal AND Inverted)
for ue_name, df in ue_data.items():
    sim_sats = [c for c in df.columns if c not in ['Time_UTC', 'unix_time']]
    times = df['unix_time'].values
    
    for sat in sim_sats:
        raw_doppler = df[sat].values
        
        # Store Normal
        feat_norm = extract_doppler_features(times, raw_doppler)
        if feat_norm:
            sim_features[f"{ue_name}:{sat}:NORM"] = {'ue': ue_name, 'sat': sat, 'inv': False, 'feat': feat_norm}
            
        # Store Inverted
        feat_inv = extract_doppler_features(times, -1 * raw_doppler)
        if feat_inv:
            sim_features[f"{ue_name}:{sat}:INV"] = {'ue': ue_name, 'sat': sat, 'inv': True, 'feat': feat_inv}

# ==========================================
# 2. BUILD COST MATRIX
# ==========================================
meas_keys = sorted(meas_features.keys())
sim_unique_keys = [] # Unique (UE, Sat) pairs

# We need to map (UE, Sat) -> Best Orientation (Normal/Inv) first
# This prevents the algorithm from matching the same satellite twice (once normal, once inverted)
pre_processed_sims = {} # Key: (ue, sat), Value: {best_score_vs_meas_X, orientation}

# Identify all unique physical satellites
physical_sats = set()
for key in sim_features:
    parts = key.split(':') # ue, sat, orientation
    physical_sats.add(f"{parts[0]}:{parts[1]}")
sim_keys_ordered = sorted(list(physical_sats))

# Matrix: Rows = Measured, Cols = Simulated Physical Satellites
cost_matrix = np.ones((len(meas_keys), len(sim_keys_ordered))) # Init with 1.0 (Worst cost)
orientation_map = {} # Stores which orientation was best for (Meas, Sim) pair

print(f"Computing scores for {len(meas_keys)} measured signals vs {len(sim_keys_ordered)} simulated satellites...")

for i, m_key in enumerate(meas_keys):
    for j, s_key in enumerate(sim_keys_ordered):
        m_feat = meas_features[m_key]
        if m_feat is None: continue
        
        # Check both Normal and Inverted for this physical satellite
        ue, sat = s_key.split(':')
        
        score_norm = 0
        if f"{s_key}:NORM" in sim_features:
            score_norm = compare_signatures(m_feat, sim_features[f"{s_key}:NORM"]['feat'])
            
        score_inv = 0
        if f"{s_key}:INV" in sim_features:
            score_inv = compare_signatures(m_feat, sim_features[f"{s_key}:INV"]['feat'])
            
        # Pick best orientation
        best_score = max(score_norm, score_inv)
        is_inverted = (score_inv > score_norm)
        
        # Store orientation choice
        orientation_map[(i, j)] = is_inverted
        
        # Cost = 1.0 - Score (Optimization minimizes cost)
        cost_matrix[i, j] = 1.0 - best_score

# ==========================================
# 3. SOLVE ASSIGNMENT
# ==========================================
# Replace any remaining NaN or inf values with maximum cost (1.0)
cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1.0)

row_ind, col_ind = linear_sum_assignment(cost_matrix)

# ==========================================
# 4. OUTPUT RESULTS
# ==========================================
print(f"\n{'Measured':<15} {'→':<3} {'Simulated':<20} {'UE':<6} {'Inv?':<6} {'Score':<8}")
print("-" * 75)

matches = []
for r, c in zip(row_ind, col_ind):
    meas_name = meas_keys[r]
    sim_key = sim_keys_ordered[c]
    ue, sat = sim_key.split(':')
    inverted = orientation_map[(r, c)]
    score = 1.0 - cost_matrix[r, c]
    
    inv_str = "YES" if inverted else "NO"
    print(f"{meas_name:<15} → {sat:<20} {ue:<6} {inv_str:<6} {score:.4f}")
    
    matches.append({
        'satellite': meas_name,
        'sim_sat': sat,
        'ue': ue,
        'inverted': inverted,
        'score': score
    })

# Save results
out_df = pd.DataFrame(matches)
out_df.to_csv('matching_results.csv', index=False)
print("\nMatches saved to matching_results.csv")

# Visualization - Filter for specific satellites
PLOT_ONLY = ['sat_25', 'sat_67', 'sat_73', 'sat_17']
matches_to_plot = [m for m in matches if m['satellite'] in PLOT_ONLY]
print(f"\nPlotting only: {PLOT_ONLY}")
print(f"Matches to plot: {len(matches_to_plot)}/{len(matches)}")

num_sats = len(matches_to_plot)
if num_sats > 0:
    cols = 2
    rows = ceil(num_sats / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if num_sats == 1: axes = [axes]
    else: axes = axes.flatten()

    for idx, match in enumerate(matches_to_plot):
        ax = axes[idx]
        meas_df = measured_data[match['satellite']]
        
        # Normalize time to start at 0
        t_meas = (meas_df['UNIX_Timestamp'] - meas_df['UNIX_Timestamp'].min())/60
        y_meas = meas_df['Doppler_Frequency_Hz']
        
        ax.plot(t_meas, y_meas, 'r.-', label='Measured')
        
        # Get Sim Data
        df_sim = ue_data[match['ue']]
        t_sim_full = (df_sim['unix_time'] - df_sim['unix_time'].min())/60
        y_sim_full = df_sim[match['sim_sat']].values
        if match['inverted']:
            y_sim_full = -1 * y_sim_full
            
        # Scale time for visual overlay (rough)
        # Note: This is just for visual confirmation, time_aligned_matching.py does the real sync
        valid = ~np.isnan(y_sim_full)
        if valid.sum() > 0:
            t_valid = t_sim_full[valid]
            y_valid = y_sim_full[valid]
            # Normalize sim time to 0-max_duration of measured
            t_norm = (t_valid - t_valid.min()) / (t_valid.max() - t_valid.min())
            t_scaled = t_norm * t_meas.max()
            ax.plot(t_scaled, y_valid, 'b--', alpha=0.6, label='Simulated')
            
        ax.set_title(f"{match['satellite']} -> {match['sim_sat']} (Score: {match['score']:.2f})")
        ax.legend()
    
    # Hide unused subplots
    for idx in range(num_sats, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig('unique_matching_results.png')
    print("Visualization saved.")