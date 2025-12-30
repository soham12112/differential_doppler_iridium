import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
    print("  ✓ Loaded UE1 data")
except FileNotFoundError:
    print("  ⚠️  No UE1 data found, skipping UE1")

try:
    ue2 = load_simulated("ue2")
    print("  ✓ Loaded UE2 data")
except FileNotFoundError:
    print("  ⚠️  No UE2 data found, skipping UE2")

if ue1 is None and ue2 is None:
    raise RuntimeError("No UE data files found. Need at least ue1_doppler_data*.csv or ue2_doppler_data*.csv")

matches_path = Path('matching_results.csv')
if not matches_path.exists():
    raise FileNotFoundError("matching_results.csv not found. Run unique_matching.py first.")

matches_df = pd.read_csv(matches_path)

print("\n" + "="*80)
print("FREQUENCY OFFSET ANALYSIS")
print("="*80)

results = []

for _, row in matches_df.iterrows():
    meas_name = row['satellite']
    sim_sat = row['sim_sat']
    ue_name = row['ue']
    score = row['score']
    meas_file = Path(f"{meas_name}_ira_doppler.csv")
    if not meas_file.exists():
        print(f"Skipping {meas_name} (file {meas_file} not found)")
        continue
    
    meas_df = pd.read_csv(meas_file, comment='#')
    meas_doppler = meas_df['Doppler_Frequency_Hz'].values
    if len(meas_doppler) == 0:
        print(f"Skipping {meas_name} (no Doppler samples)")
        continue
    
    ue_data = ue1 if ue_name == 'ue1' else ue2
    if ue_data is None:
        print(f"Skipping {meas_name} ({ue_name} data not available)")
        continue
    if sim_sat not in ue_data.columns:
        print(f"Skipping {meas_name} (sim satellite {sim_sat} missing)")
        continue
    sim_doppler = -1 * ue_data[sim_sat].values
    valid_mask = ~np.isnan(sim_doppler)
    if valid_mask.sum() == 0:
        print(f"Skipping {meas_name} (sim Doppler empty)")
        continue
    sim_clean = sim_doppler[valid_mask]
    
    meas_mean = np.mean(meas_doppler)
    meas_median = np.median(meas_doppler)
    meas_min = np.min(meas_doppler)
    meas_max = np.max(meas_doppler)
    
    sim_mean = np.mean(sim_clean)
    sim_median = np.median(sim_clean)
    sim_min = np.min(sim_clean)
    sim_max = np.max(sim_clean)
    
    offset_mean = meas_mean - sim_mean
    offset_median = meas_median - sim_median
    offset_min = meas_min - sim_min
    offset_max = meas_max - sim_max
    
    print(f"\n{meas_name.upper()} → {sim_sat} ({ue_name})  score={score:.4f}")
    print("-" * 80)
    print(f"  Measured mean: {meas_mean:>10.1f} Hz")
    print(f"  Sim mean:      {sim_mean:>10.1f} Hz")
    print(f"  Mean offset:   {offset_mean:>10.1f} Hz")
    
    results.append({
        'measured': meas_name,
        'simulated': sim_sat,
        'ue': ue_name,
        'score': score,
        'offset_mean': offset_mean,
        'offset_median': offset_median,
        'offset_min': offset_min,
        'offset_max': offset_max
    })

if not results:
    raise RuntimeError("No offsets computed. Check input files.")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Measured':<12} {'Simulated':<15} {'Mean Offset (Hz)':>16}")
print("-" * 50)
for r in results:
    print(f"{r['measured']:<12} {r['simulated']:<15} {r['offset_mean']:>16.1f}")

mean_offsets = np.array([r['offset_mean'] for r in results])
median_offsets = np.array([r['offset_median'] for r in results])
min_offsets = np.array([r['offset_min'] for r in results])
max_offsets = np.array([r['offset_max'] for r in results])
measured_names = [r['measured'] for r in results]

avg_offset = np.mean(mean_offsets)
std_offset = np.std(mean_offsets)
print("\nAverage mean offset: {:.1f} ± {:.1f} Hz".format(avg_offset, std_offset))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Mean offsets
axes[0].bar(measured_names, mean_offsets, color='steelblue')
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Mean Offset by Satellite')
axes[0].set_ylabel('Offset (Hz)')
axes[0].grid(True, axis='y', alpha=0.3)

# Offset components
x = np.arange(len(measured_names))
width = 0.2
axes[1].bar(x - 1.5*width, min_offsets, width, label='Min')
axes[1].bar(x - 0.5*width, mean_offsets, width, label='Mean')
axes[1].bar(x + 0.5*width, max_offsets, width, label='Max')
axes[1].bar(x + 1.5*width, median_offsets, width, label='Median')
axes[1].set_xticks(x)
axes[1].set_xticklabels(measured_names)
axes[1].set_title('Offset Components')
axes[1].set_ylabel('Offset (Hz)')
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].legend()
axes[1].grid(True, axis='y', alpha=0.3)

# Distribution histogram
axes[2].hist(mean_offsets, bins=min(10, len(mean_offsets)), color='coral', edgecolor='black')
axes[2].axvline(avg_offset, color='red', linestyle='--', label=f'Mean={avg_offset:.1f}')
axes[2].set_title('Distribution of Mean Offsets')
axes[2].set_xlabel('Offset (Hz)')
axes[2].set_ylabel('Count')
axes[2].legend()

# Summary text
axes[3].axis('off')
summary = f"Total satellites: {len(results)}\n"
summary += f"Average offset: {avg_offset:.1f} Hz\n"
summary += f"Std deviation: {std_offset:.1f} Hz\n"
summary += "\nOffsets per satellite:\n"
for r in results:
    summary += f"  {r['measured']:>6}: {r['offset_mean']:>8.1f} Hz\n"
axes[3].text(0.02, 0.98, summary, va='top', ha='left', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.6))

plt.tight_layout()
plt.savefig('frequency_offset_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'frequency_offset_analysis.png'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
