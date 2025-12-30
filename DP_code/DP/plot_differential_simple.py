#!/usr/bin/env python3
"""
Simplified plotting script for differential results - avoids font issues
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Simplified rcParams without font specification
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

# Your data from the example
data = {
    'scenarios': ['Base\nStation\n(Office)', 'Rover\nStandard\n(Home)', 'Rover\nDifferential\n(Home)'],
    'true_lat': [37.419008, 37.361979, 37.361979],
    'true_lon': [-122.096184, -122.028127, -122.028127],
    'est_lat': [37.417, 37.401, 37.382492],
    'est_lon': [-122.101, -122.056, -122.043330],
    'error_m': [50.8, 497.3, 264.5]
}

# === PLOT 1: Spatial Error Map (Improved - Zoomed) ===
print("Generating spatial error map...")
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Collect all coordinates to determine optimal zoom
all_lats = [data['true_lat'][0], data['est_lat'][0], 
           data['true_lat'][1], data['est_lat'][1], data['est_lat'][2]]
all_lons = [data['true_lon'][0], data['est_lon'][0],
           data['true_lon'][1], data['est_lon'][1], data['est_lon'][2]]

# Calculate center and range
center_lat = np.mean(all_lats)
center_lon = np.mean(all_lons)
lat_range = max(all_lats) - min(all_lats)
lon_range = max(all_lons) - min(all_lons)

# Add reasonable margins (20% of range, but at least 0.01 degrees)
lat_margin = max(lat_range * 0.2, 0.01)
lon_margin = max(lon_range * 0.2, 0.01)

# Base station
ax1.scatter(data['true_lon'][0], data['true_lat'][0], 
           color='darkgreen', marker='*', s=500, label='Base True', zorder=5, 
           edgecolors='black', linewidths=2)
ax1.scatter(data['est_lon'][0], data['est_lat'][0], 
           color='lightgreen', marker='o', s=250, 
           label=f'Base Est ({data["error_m"][0]:.0f} m)', zorder=4,
           edgecolors='darkgreen', linewidths=2)
ax1.annotate('', xy=(data['est_lon'][0], data['est_lat'][0]),
            xytext=(data['true_lon'][0], data['true_lat'][0]),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen', 
                          linestyle='--', alpha=0.7))

# Rover true
ax1.scatter(data['true_lon'][1], data['true_lat'][1],
           color='darkblue', marker='*', s=500, label='Rover True', zorder=5,
           edgecolors='black', linewidths=2)

# Rover standard
ax1.scatter(data['est_lon'][1], data['est_lat'][1],
           color='red', marker='x', s=300, linewidths=3.5,
           label=f'Rover Std ({data["error_m"][1]:.0f} m)', zorder=4)
ax1.annotate('', xy=(data['est_lon'][1], data['est_lat'][1]),
            xytext=(data['true_lon'][1], data['true_lat'][1]),
            arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                          linestyle='--', alpha=0.7))

# Rover differential
ax1.scatter(data['est_lon'][2], data['est_lat'][2],
           color='dodgerblue', marker='o', s=250,
           label=f'Rover Diff ({data["error_m"][2]:.0f} m)', zorder=4,
           edgecolors='darkblue', linewidths=2)
ax1.annotate('', xy=(data['est_lon'][2], data['est_lat'][2]),
            xytext=(data['true_lon'][2], data['true_lat'][2]),
            arrowprops=dict(arrowstyle='->', lw=3, color='blue', 
                          linestyle='--', alpha=0.7))

# Add error distance labels on arrows (positioned to the right to avoid overlap)
# Standard error label
mid_lon_std = (data['true_lon'][1] + data['est_lon'][1]) / 2
mid_lat_std = (data['true_lat'][1] + data['est_lat'][1]) / 2
# Offset to the right
label_offset_lon = (max(all_lons) - min(all_lons)) * 0.08  # 8% of longitude range
ax1.text(mid_lon_std + label_offset_lon, mid_lat_std, f'{data["error_m"][1]:.1f} m', 
        color='red', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor='red', alpha=0.9, linewidth=1.5),
        ha='left', va='center')

# Differential error label
mid_lon_diff = (data['true_lon'][2] + data['est_lon'][2]) / 2
mid_lat_diff = (data['true_lat'][2] + data['est_lat'][2]) / 2
# Offset to the right
ax1.text(mid_lon_diff + label_offset_lon, mid_lat_diff, f'{data["error_m"][2]:.1f} m', 
        color='blue', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor='blue', alpha=0.9, linewidth=1.5),
        ha='left', va='center')

# Add improvement text box (positioned near rover data on the right)
improvement = data['error_m'][1] - data['error_m'][2]
improvement_pct = (improvement / data['error_m'][1]) * 100
ax1.text(0.98, 0.55, 
        f'Improvement:\n{improvement:.0f} m ({improvement_pct:.1f}%)',
        transform=ax1.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='center', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                 edgecolor='black', alpha=0.95, linewidth=2))

# Set axis limits to zoom in on the data
ax1.set_xlim(min(all_lons) - lon_margin, max(all_lons) + lon_margin)
ax1.set_ylim(min(all_lats) - lat_margin, max(all_lats) + lat_margin)

ax1.set_xlabel('Longitude (deg)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latitude (deg)', fontsize=12, fontweight='bold')
ax1.set_title('Spatial Error Map: Standard vs Differential', 
             fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.95, edgecolor='black')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('Plots/position_map.png', dpi=150, bbox_inches='tight')
print("✓ Saved: Plots/position_map.png")
plt.close()

# === PLOT 2: Error Bar Chart ===
print("Generating error comparison bar...")
fig2, ax2 = plt.subplots(figsize=(10, 7))

colors = ['#2E7D32', '#C62828', '#1565C0']
bars = ax2.bar(data['scenarios'], data['error_m'], color=colors, 
              edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bar, error in zip(bars, data['error_m']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
            f'{int(error)} m',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement text
mid_y = (data['error_m'][1] + data['error_m'][2]) / 2
ax2.text(1.5, mid_y,
        f'Improvement:\n{improvement:.0f} m\n({improvement_pct:.1f}%)',
        fontsize=11, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2.set_ylabel('Distance Error (meters)', fontsize=12, fontweight='bold')
ax2.set_title('Positioning Error Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_ylim(0, max(data['error_m']) * 1.1)

plt.tight_layout()
plt.savefig('Plots/error_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: Plots/error_comparison.png")
plt.close()

# === PLOT 3: Satellite Quality (RMS Comparison Only) ===
print("Generating satellite quality comparison...")
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Sample satellite data - REPLACE WITH YOUR ACTUAL DATA
satellites = ['Iridium 135', 'Iridium 168', 'Iridium 173', 'Iridium 108']
base_rms = [6.8, 6.3, 12.6, 0]  # 0 means no data
rover_rms = [87.4, 104.7, 79.5, 135.6]
common_view = [True, True, False, False]

# RMS comparison
x = np.arange(len(satellites))
width = 0.35

# Filter out zero values for base
base_rms_plot = [rms if rms > 0 else 0 for rms in base_rms]

bars1 = ax3.bar(x - width/2, base_rms_plot, width, 
                label='Monitoring Station', color='#2E7D32', alpha=0.85,
                edgecolor='black', linewidth=1.2)
bars2 = ax3.bar(x + width/2, rover_rms, width,
                label='Rover', color='#C62828', alpha=0.85,
                edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (b, r) in enumerate(zip(base_rms_plot, rover_rms)):
    if b > 0:
        ax3.text(x[i] - width/2, b + 3, f'{b:.1f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax3.text(x[i] - width/2, 3, 'N/A', 
                 ha='center', va='bottom', fontsize=8, style='italic')
    ax3.text(x[i] + width/2, r + 3, f'{r:.1f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add common view indicators (checkmarks/crosses on x-axis labels)
tick_labels = []
for i, (sat, cv) in enumerate(zip(satellites, common_view)):
    marker = '✓' if cv else '✗'
    color_code = 'green' if cv else 'red'
    tick_labels.append(f'{sat}\n{marker}')

ax3.set_xlabel('Satellite', fontsize=11, fontweight='bold')
ax3.set_ylabel('RMS Residual (Hz)', fontsize=12, fontweight='bold')
ax3.set_title('Doppler Measurement Quality Comparison', fontsize=13, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(satellites, fontsize=10)
ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black')
ax3.grid(axis='y', linestyle='--', alpha=0.4)
ax3.set_axisbelow(True)

# Add average lines with labels
base_avg = np.mean([r for r in base_rms if r > 0])
rover_avg = np.mean(rover_rms)
ax3.axhline(base_avg, color='#2E7D32', linestyle='--', linewidth=2, alpha=0.6)
ax3.axhline(rover_avg, color='#C62828', linestyle='--', linewidth=2, alpha=0.6)

# Add average annotations
ax3.text(len(satellites) - 0.3, base_avg + 5, f'Base Avg: {base_avg:.1f} Hz',
         color='#2E7D32', fontweight='bold', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax3.text(len(satellites) - 0.3, rover_avg + 5, f'Rover Avg: {rover_avg:.1f} Hz',
         color='#C62828', fontweight='bold', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add quality ratio tex

plt.tight_layout()
plt.savefig('Plots/satellite_quality.png', dpi=150, bbox_inches='tight')
print("✓ Saved: Plots/satellite_quality.png")
plt.close()

# === Print Summary Tables ===
print("\n" + "="*80)
print("TABLE 1: POSITIONING PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Scenario':<30} {'Error (m)':<12} {'Improvement':<15}")
print("-"*80)
print(f"{'Monitoring Station':<30} {data['error_m'][0]:<12.0f} {'-':<15}")
print(f"{'Rover - Standard':<30} {data['error_m'][1]:<12.0f} {'Reference':<15}")
print(f"{'Rover - Differential Corrected':<30} {data['error_m'][2]:<12.0f} {improvement_pct:.1f}%")
print("="*80)

print("\n" + "="*80)
print("TABLE 2: SATELLITE MEASUREMENT QUALITY")
print("="*80)
print(f"{'Satellite':<15} {'Base RMS':<12} {'Rover RMS':<12} {'Common View':<15}")
print("-"*80)
for i, sat in enumerate(['Iridium 165', 'Iridium 166', 'Iridium 154', 'Iridium 108']):
    base_str = f"{base_rms[i]:.1f} Hz" if base_rms[i] > 0 else "N/A"
    rover_str = f"{rover_rms[i]:.1f} Hz"
    cv_str = "Yes" if common_view[i] else "No"
    print(f"{sat:<15} {base_str:<12} {rover_str:<12} {cv_str:<15}")
print("-"*80)
print(f"{'Average':<15} {base_avg:.1f} Hz{'':<6} {rover_avg:.1f} Hz")
print("="*80)

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"1. POSITIONING IMPROVEMENT: {improvement:.0f} m ({improvement_pct:.1f}%)")
print(f"2. MEASUREMENT QUALITY: Base={base_avg:.1f} Hz, Rover={rover_avg:.1f} Hz")
print(f"   Quality ratio: {rover_avg/base_avg:.1f}× worse at rover")
print(f"3. COMMON VIEW: {sum(common_view)}/{len(common_view)} satellites overlapping")
print("="*80)

print("\nAll plots generated successfully!")
print("Plots saved in: Plots/")
print("  - position_map.png")
print("  - error_comparison.png")
print("  - satellite_quality.png")

