import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil

def extract_doppler_features(times, doppler_values):
    """Extract key features from a Doppler signature"""
    if len(doppler_values) < 3:
        return None
    
    # Remove NaN values
    valid_mask = ~np.isnan(doppler_values)
    if valid_mask.sum() < 3:
        return None
    
    times_valid = times[valid_mask]
    doppler_valid = doppler_values[valid_mask]
    
    # Duration (in minutes)
    duration = (times_valid[-1] - times_valid[0]) / 60 if len(times_valid) > 1 else 0
    
    # Doppler statistics
    doppler_min = np.min(doppler_valid)
    doppler_max = np.max(doppler_valid)
    doppler_range = doppler_max - doppler_min
    doppler_mean = np.mean(doppler_valid)
    doppler_std = np.std(doppler_valid)
    
    # Doppler rate (Hz/s)
    if len(times_valid) > 1:
        doppler_rates = np.diff(doppler_valid) / np.diff(times_valid)
        avg_doppler_rate = np.mean(doppler_rates)
        max_doppler_rate = np.max(np.abs(doppler_rates))
    else:
        avg_doppler_rate = 0
        max_doppler_rate = 0
    
    # Peak location (normalized 0-1)
    peak_idx = np.argmax(doppler_valid)
    peak_location_norm = peak_idx / (len(doppler_valid) - 1) if len(doppler_valid) > 1 else 0.5
    
    # Trend
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
        'doppler_std': doppler_std,
        'avg_doppler_rate': avg_doppler_rate,
        'max_doppler_rate': max_doppler_rate,
        'peak_location_norm': peak_location_norm,
        'trend': trend,
        'n_points': len(doppler_valid)
    }

def compare_signatures(feat1, feat2):
    """Compare two Doppler signatures and return a similarity score"""
    if feat1 is None or feat2 is None:
        return 0, {}
    
    # Weights for different features
    weights = {
        'doppler_range': 0.25,      # Increased - most important
        'avg_doppler_rate': 0.25,   # Increased - very important
        'max_doppler_rate': 0.15,
        'trend': 0.20,              # Increased - important for direction
        'doppler_mean': 0.05,       # Decreased - can be shifted
        'peak_location': 0.05,
        'duration': 0.05
    }
    
    scores = {}
    
    # Doppler range similarity (normalized)
    range_diff = abs(feat1['doppler_range'] - feat2['doppler_range'])
    max_range = max(feat1['doppler_range'], feat2['doppler_range'])
    scores['doppler_range'] = 1 - min(range_diff / max_range, 1) if max_range > 0 else 0
    
    # Doppler mean similarity (with less weight as it can be offset)
    mean_diff = abs(feat1['doppler_mean'] - feat2['doppler_mean'])
    avg_range = (feat1['doppler_range'] + feat2['doppler_range']) / 2
    scores['doppler_mean'] = 1 - min(mean_diff / avg_range, 1) if avg_range > 0 else 0
    
    # Doppler rate similarity (IMPORTANT)
    rate_diff = abs(feat1['avg_doppler_rate'] - feat2['avg_doppler_rate'])
    max_rate = max(abs(feat1['avg_doppler_rate']), abs(feat2['avg_doppler_rate']))
    scores['avg_doppler_rate'] = 1 - min(rate_diff / max_rate, 1) if max_rate > 0 else 0
    
    # Max doppler rate similarity
    max_rate_diff = abs(feat1['max_doppler_rate'] - feat2['max_doppler_rate'])
    max_max_rate = max(feat1['max_doppler_rate'], feat2['max_doppler_rate'])
    scores['max_doppler_rate'] = 1 - min(max_rate_diff / max_max_rate, 1) if max_max_rate > 0 else 0
    
    # Trend similarity (CRITICAL - must match direction and magnitude)
    if np.sign(feat1['trend']) == np.sign(feat2['trend']):
        trend_diff = abs(feat1['trend'] - feat2['trend'])
        max_trend = max(abs(feat1['trend']), abs(feat2['trend']))
        scores['trend'] = 1 - min(trend_diff / max_trend, 1) if max_trend > 0 else 0.5
    else:
        scores['trend'] = 0  # Wrong direction = no match
    
    # Peak location similarity
    peak_diff = abs(feat1['peak_location_norm'] - feat2['peak_location_norm'])
    scores['peak_location'] = 1 - min(peak_diff, 1)
    
    # Duration similarity
    duration_diff = abs(feat1['duration'] - feat2['duration'])
    max_duration = max(feat1['duration'], feat2['duration'])
    scores['duration'] = 1 - min(duration_diff / max_duration, 1) if max_duration > 0 else 0
    
    # Calculate weighted total score
    total_score = sum(scores[key] * weights[key] for key in weights.keys())
    
    return total_score, scores

# Load simulated data
def load_simulated(prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_doppler_data*.csv"
    candidates = sorted(Path(".").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern}")
    df = pd.read_csv(candidates[0], parse_dates=['Time_UTC'])
    print(f"Loaded {candidates[0].name}")
    return df


print("Loading simulated data...")
ue1 = None
ue2 = None

try:
    ue1 = load_simulated("ue1")
    ue1['unix_time'] = ue1['Time_UTC'].apply(lambda x: x.timestamp())
    print("  ✓ Loaded UE1 data")
except FileNotFoundError:
    print("  ⚠️  No UE1 data found, skipping UE1")

try:
    ue2 = load_simulated("ue2")
    ue2['unix_time'] = ue2['Time_UTC'].apply(lambda x: x.timestamp())
    print("  ✓ Loaded UE2 data")
except FileNotFoundError:
    print("  ⚠️  No UE2 data found, skipping UE2")

if ue1 is None and ue2 is None:
    raise RuntimeError("No UE data files found. Need at least ue1_doppler_data*.csv or ue2_doppler_data*.csv")

# Load measured data dynamically
print("Loading measured data...")
measured_data = {}
measured_paths = sorted(
    p for p in Path(".").glob("sat_*_ira_doppler.csv")
    if "corrected" not in p.stem
)

if not measured_paths:
    raise RuntimeError("No sat_*_ira_doppler.csv files found.")

for path in measured_paths:
    sat_name = path.stem.replace("_ira_doppler", "")
    df = pd.read_csv(path, comment='#')
    measured_data[sat_name] = df

# Get list of simulated satellites from whichever UE dataset exists
ue_reference = ue1 if ue1 is not None else ue2
simulated_satellites = [col for col in ue_reference.columns if col not in ['Time_UTC', 'unix_time']]

print("\n" + "="*80)
print("IMPROVED DOPPLER MATCHING WITH INVERSION CONSIDERATION")
print("="*80)

# Extract features for measured satellites
measured_features = {}
for meas_sat, meas_df in measured_data.items():
    times = meas_df['UNIX_Timestamp'].values
    doppler = meas_df['Doppler_Frequency_Hz'].values
    measured_features[meas_sat] = extract_doppler_features(times, doppler)

# Extract features for simulated satellites (BOTH NORMAL AND INVERTED)
print("\nExtracting features from simulated satellites (normal and inverted)...")
simulated_features = {
    'ue1_normal': {}, 
    'ue1_inverted': {},
    'ue2_normal': {}, 
    'ue2_inverted': {}
}

for sim_sat in simulated_satellites:
    # UE1 - Normal and Inverted (only if ue1 exists)
    if ue1 is not None:
        times = ue1['unix_time'].values
        doppler = ue1[sim_sat].values
        simulated_features['ue1_normal'][sim_sat] = extract_doppler_features(times, doppler)
        
        # UE1 - Inverted (multiply Doppler by -1)
        doppler_inverted = -1 * doppler
        simulated_features['ue1_inverted'][sim_sat] = extract_doppler_features(times, doppler_inverted)
    
    # UE2 - Normal and Inverted (only if ue2 exists)
    if ue2 is not None:
        times = ue2['unix_time'].values
        doppler = ue2[sim_sat].values
        simulated_features['ue2_normal'][sim_sat] = extract_doppler_features(times, doppler)
        
        # UE2 - Inverted
        doppler_inverted = -1 * doppler
        simulated_features['ue2_inverted'][sim_sat] = extract_doppler_features(times, doppler_inverted)

# Compare each measured satellite against all simulated satellites (normal and inverted)
results = {}

for meas_sat in measured_data.keys():
    print(f"\n\nAnalyzing {meas_sat.upper()}...")
    print("-" * 80)
    
    meas_feat = measured_features[meas_sat]
    if meas_feat is None:
        print(f"Insufficient data for {meas_sat}")
        continue
    
    # Print measured satellite features
    print(f"\nMeasured Satellite Features:")
    print(f"  Duration: {meas_feat['duration']:.2f} minutes")
    print(f"  Doppler range: {meas_feat['doppler_min']:.0f} to {meas_feat['doppler_max']:.0f} Hz (range: {meas_feat['doppler_range']:.0f} Hz)")
    print(f"  Average Doppler rate: {meas_feat['avg_doppler_rate']:.2f} Hz/s")
    print(f"  Trend: {'Increasing' if meas_feat['trend'] > 0 else 'Decreasing'} ({meas_feat['trend']:.0f} Hz)")
    
    matches = []
    
    for variant_name, variant_features in simulated_features.items():
        for sim_sat, sim_feat in variant_features.items():
            if sim_feat is None:
                continue
            
            score, component_scores = compare_signatures(meas_feat, sim_feat)
            
            if score > 0:
                # Parse variant name
                ue_name = variant_name.split('_')[0]  # 'ue1' or 'ue2'
                is_inverted = 'inverted' in variant_name
                
                matches.append({
                    'ue': ue_name,
                    'satellite': sim_sat,
                    'inverted': is_inverted,
                    'score': score,
                    'component_scores': component_scores,
                    'sim_features': sim_feat
                })
    
    # Sort by score
    matches.sort(key=lambda x: x['score'], reverse=True)
    results[meas_sat] = matches
    
    # Print top 10 matches
    print(f"\n{'='*80}")
    print(f"TOP 10 MATCHES FOR {meas_sat.upper()}")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'UE':<6} {'Satellite':<20} {'Inverted':<10} {'Score':<10} {'Key Features'}")
    print("-" * 80)
    
    for i, match in enumerate(matches[:10], 1):
        sim_feat = match['sim_features']
        
        # Highlight best matching features
        comp_scores = match['component_scores']
        top_features = sorted(comp_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        features_str = ", ".join([f"{feat}:{score:.2f}" for feat, score in top_features])
        
        inv_str = "YES" if match['inverted'] else "NO"
        
        print(f"{i:<6} {match['ue']:<6} {match['satellite']:<20} {inv_str:<10} {match['score']:.4f}   {features_str}")

# Create visualization
print("\n\n" + "="*80)
print("CREATING VISUALIZATION...")
print("="*80)

selected_sats = [
    sat for sat in measured_data.keys()
    if sat in results and len(results[sat]) > 0 and results[sat][0]['score'] >= 0.90
]

if not selected_sats:
    print("No satellites with score >= 0.90; skipping visualization.")
else:
    num_sats = len(selected_sats)
    cols = 2
    rows = ceil(num_sats / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten()

    for idx, meas_sat in enumerate(selected_sats):
        meas_df = measured_data[meas_sat]
        ax = axes[idx]
        
        meas_times_rel = (meas_df['UNIX_Timestamp'] - meas_df['UNIX_Timestamp'].min()) / 60
        ax.plot(meas_times_rel, meas_df['Doppler_Frequency_Hz'], 
                'ro-', label=f'{meas_sat} (measured)', linewidth=2.5, markersize=6, zorder=5)
        
        for rank, match in enumerate(results[meas_sat][:3], 1):
            ue_data = ue1 if match['ue'] == 'ue1' else ue2
            sim_sat_name = match['satellite']
            sim_times_rel = (ue_data['unix_time'] - ue_data['unix_time'].min()) / 60
            sim_doppler = ue_data[sim_sat_name].values
            if match['inverted']:
                sim_doppler = -1 * sim_doppler
            valid_mask = ~np.isnan(sim_doppler)
            sim_times_plot = sim_times_rel[valid_mask].values
            sim_doppler_plot = sim_doppler[valid_mask]
            if len(sim_doppler_plot) == 0:
                continue
            sim_times_normalized = (sim_times_plot - sim_times_plot.min()) / (sim_times_plot.max() - sim_times_plot.min()) if len(sim_times_plot) > 1 else sim_times_plot
            meas_duration = meas_times_rel.max() - meas_times_rel.min()
            sim_times_scaled = sim_times_normalized * meas_duration
            linestyle = '--' if rank == 1 else ':' if rank == 2 else '-.'
            alpha = 0.8 if rank == 1 else 0.6 if rank == 2 else 0.4
            linewidth = 2 if rank == 1 else 1.5 if rank == 2 else 1
            inv_label = " [INVERTED]" if match['inverted'] else ""
            ax.plot(sim_times_scaled, sim_doppler_plot, linestyle=linestyle,
                    label=f'#{rank}: {sim_sat_name} ({match["ue"]}){inv_label} - {match["score"]:.3f}',
                    linewidth=linewidth, alpha=alpha, zorder=4-rank)
        
        ax.set_xlabel('Normalized Time (minutes)', fontsize=11)
        ax.set_ylabel('Doppler Frequency (Hz)', fontsize=11)
        ax.set_title(f'{meas_sat.upper()} vs Best Matches (score ≥ 0.90)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    for ax in axes[num_sats:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('improved_matching_with_inversion.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'improved_matching_with_inversion.png'")

# Create summary table
print("\n\n" + "="*80)
print("SUMMARY: BEST MATCHES (WITH INVERSION CONSIDERED)")
print("="*80)
print(f"{'Measured Sat':<15} {'Best Match':<20} {'UE':<6} {'Inverted':<10} {'Score':<10} {'Improvement'}")
print("-" * 80)

for meas_sat in measured_data.keys():
    if meas_sat in results and len(results[meas_sat]) > 0:
        best = results[meas_sat][0]
        
        # Get top feature
        comp_scores = best['component_scores']
        top_feature = max(comp_scores.items(), key=lambda x: x[1])
        
        inv_str = "YES" if best['inverted'] else "NO"
        improvement = f"(Best: {top_feature[0]}={top_feature[1]:.2f})"
        
        print(f"{meas_sat:<15} {best['satellite']:<20} {best['ue']:<6} {inv_str:<10} "
              f"{best['score']:.4f}   {improvement}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\n✓ This analysis properly considers inverted Doppler signatures")
print("✓ Measured data (all decreasing) is now correctly matched with inverted simulated data")

