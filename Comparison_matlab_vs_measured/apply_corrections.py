import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil


def load_simulated(prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_doppler_data*.csv"
    candidates = sorted(Path(".").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching {pattern}")
    df = pd.read_csv(candidates[0], parse_dates=['Time_UTC'])
    print(f"Loaded {candidates[0].name}")
    return df


print("Loading simulated data...")
ue1 = load_simulated("ue1")
ue2 = load_simulated("ue2")
ue1['unix_time'] = ue1['Time_UTC'].apply(lambda x: x.timestamp())
ue2['unix_time'] = ue2['Time_UTC'].apply(lambda x: x.timestamp())

matches_path = Path('matching_results.csv')
if not matches_path.exists():
    raise FileNotFoundError("matching_results.csv not found. Run unique_matching.py first.")

matches_df = pd.read_csv(matches_path)

print("\n" + "="*80)
print("APPLYING FREQUENCY CORRECTIONS")
print("="*80)

results = []

for _, row in matches_df.iterrows():
    meas_name = row['satellite']
    sim_sat = row['sim_sat']
    ue_name = row['ue']
    meas_file = Path(f"{meas_name}_ira_doppler.csv")
    if not meas_file.exists():
        print(f"Skipping {meas_name} (file {meas_file} missing)")
        continue
    
    meas_df = pd.read_csv(meas_file, comment='#')
    meas_times = meas_df['UNIX_Timestamp'].values
    meas_doppler = meas_df['Doppler_Frequency_Hz'].values
    if len(meas_doppler) == 0:
        print(f"Skipping {meas_name} (empty Doppler)")
        continue
    
    ue_data = ue1 if ue_name == 'ue1' else ue2
    if sim_sat not in ue_data.columns:
        print(f"Skipping {meas_name} (sim {sim_sat} missing)")
        continue
    sim_doppler = -1 * ue_data[sim_sat].values
    valid_mask = ~np.isnan(sim_doppler)
    if valid_mask.sum() == 0:
        print(f"Skipping {meas_name} (sim data empty)")
        continue
    sim_clean = sim_doppler[valid_mask]
    sim_times = (ue_data['unix_time'].values)[valid_mask]
    
    meas_mean = np.mean(meas_doppler)
    sim_mean = np.mean(sim_clean)
    correction = meas_mean - sim_mean
    meas_corrected = meas_doppler - correction
    residual = np.mean(meas_corrected) - sim_mean
    
    print(f"\n{meas_name.upper()}:")
    print("-" * 80)
    print(f"  Correction applied: {correction:.1f} Hz")
    print(f"  Residual mean offset: {residual:.1f} Hz")
    
    # Save corrected CSV
    meas_df['Doppler_Frequency_Hz_Original'] = meas_df['Doppler_Frequency_Hz']
    meas_df['Frequency_Correction_Applied_Hz'] = correction
    meas_df['Doppler_Frequency_Hz'] = meas_corrected
    out_path = meas_file.with_name(meas_file.stem + '_corrected.csv')
    meas_df.to_csv(out_path, index=False)
    print(f"  Saved corrected data to {out_path.name}")
    
    results.append({
        'measured': meas_name,
        'simulated': sim_sat,
        'ue': ue_name,
        'correction': correction,
        'residual': residual,
        'meas_times': meas_times,
        'meas_original': meas_doppler,
        'meas_corrected': meas_corrected,
        'sim_times': sim_times,
        'sim_values': sim_clean
    })

if not results:
    raise RuntimeError("No corrections applied.")

# Plot comparison (original vs corrected)
num_sats = len(results)
cols = 2
rows = ceil(num_sats / cols)
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten()

for idx, res in enumerate(results):
    ax = axes[idx]
    # Normalise time axis for plotting
    t_rel = (res['meas_times'] - res['meas_times'].min()) / 60
    ax.plot(t_rel, res['meas_original'], 'r-', label='Original', linewidth=1.5)
    ax.plot(t_rel, res['meas_corrected'], 'g-', label='Corrected', linewidth=1.5)
    
    sim_times = (res['sim_times'] - res['sim_times'].min())
    if sim_times.max() > 0:
        sim_norm = (sim_times / sim_times.max()) * (t_rel.max() - t_rel.min())
    else:
        sim_norm = sim_times
    ax.plot(sim_norm, res['sim_values'], 'b--', label='Simulated [INV]', linewidth=1.2, alpha=0.7)
    
    ax.set_title(f"{res['measured'].upper()} (corr={res['correction']:.0f} Hz)\nresidual={res['residual']:.1f} Hz")
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Doppler (Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

for ax in axes[num_sats:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('corrected_doppler_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'corrected_doppler_comparison.png'")

print("\n" + "="*80)
print("CORRECTION SUMMARY")
print("="*80)
print(f"{'Measured':<10} {'Simulated':<15} {'Correction (Hz)':>15} {'Residual (Hz)':>15}")
print("-" * 60)
for res in results:
    print(f"{res['measured']:<10} {res['simulated']:<15} {res['correction']:>15.1f} {res['residual']:>15.1f}")

avg_residual = np.mean([r['residual'] for r in results])
std_residual = np.std([r['residual'] for r in results])
print("\nAverage residual offset: {:.1f} ± {:.1f} Hz".format(avg_residual, std_residual))
print("\n✅ All corrected CSV files have been generated." )
