"""
Analyze residuals for ALL satellites to identify ghosts
Shows detailed statistics for each satellite
"""

import pickle
import numpy as np
from astropy.coordinates import EarthLocation

from src.config.locations import LOCATIONS
from src.config.setup import *
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve, C
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import NavDataArrayIndices as IDX

LON_HOME, LAT_HOME, ALT_HOME = LOCATIONS["HOME"][0], LOCATIONS["HOME"][1], LOCATIONS["HOME"][2]


def calculate_residuals_per_satellite(nav_data, lat, lon, alt, off, dft, satellites):
    """
    Calculate residuals for each satellite
    """
    residuals_by_sat = {}
    
    for sat_id in np.unique(nav_data[:, IDX.sat_id]):
        mask = nav_data[:, IDX.sat_id] == sat_id
        sat_data = nav_data[mask]
        
        # Get satellite info
        try:
            sat = satellites[str(int(sat_id))]
            sat_name = sat.name
        except KeyError:
            sat_name = f"UNKNOWN-{int(sat_id)}"
            continue
        
        # Calculate predicted doppler
        measured_freq = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
        times = sat_data[:, IDX.t]
        
        r_sat = np.column_stack((sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]))
        v_sat = np.column_stack((sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]))
        
        # User position
        r_user = (EarthLocation.from_geodetic(lon, lat, alt)
                 .get_itrs().cartesian.without_differentials())
        r_user = np.array([r_user.x.value, r_user.y.value, r_user.z.value]) * 1000  # to meters
        
        # Calculate range rate
        r_sat_m = r_sat.T * 1000  # to meters
        v_sat_m = v_sat.T * 1000
        r_user_rep = r_user[:, np.newaxis] * np.ones(len(times))
        
        rel_vel = np.sum(v_sat_m * (r_sat_m - r_user_rep) / np.linalg.norm(r_sat_m - r_user_rep, axis=0), axis=0)
        f_b = sat_data[:, IDX.fb]
        f_d = -1 * rel_vel * f_b / C
        
        # Calculate drift
        f_drift = (times - np.min(times)) * dft
        
        # Predicted frequency
        predicted = f_d + off + f_drift
        
        # Residuals
        residuals = measured_freq - predicted
        
        # Statistics
        rms = np.sqrt(np.mean(residuals**2))
        mean = np.mean(residuals)
        std = np.std(residuals)
        max_abs = np.max(np.abs(residuals))
        
        residuals_by_sat[sat_id] = {
            'name': sat_name,
            'rms': rms,
            'mean': mean,
            'std': std,
            'max': max_abs,
            'n_points': len(residuals)
        }
    
    return residuals_by_sat


# Load data
print("Loading data...")
with open(DATA_PATH + SAVED_DATA_FILE, "rb") as file:
    saved_nav_data = pickle.load(file)

print(f"Total measurements: {len(saved_nav_data)}")

# Load satellites
satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")["Iridium"]

# Run initial solve to get position estimate
print("\nRunning initial position solve...")
parameters = CurveFitMethodParameters()
TRUE_LOCATION = (LAT_HOME, LON_HOME, ALT_HOME)

try:
    lat, lon, alt, off, dft = solve(saved_nav_data, satellites, parameters, true_location=TRUE_LOCATION)
    print(f"Initial solution: lat={lat:.3f}°, lon={lon:.3f}°, alt={alt:.0f}m")
except Exception as e:
    print(f"Solve failed: {e}")
    print("Using approximate location for analysis...")
    lat, lon, alt = LAT_HOME, LON_HOME, ALT_HOME
    off, dft = 0, 0

# Calculate residuals for all satellites
print("\nCalculating residuals for all satellites...")
residuals = calculate_residuals_per_satellite(saved_nav_data, lat, lon, alt, off, dft, satellites)

# Print results sorted by RMS
print("\n" + "="*90)
print("SATELLITE RESIDUAL ANALYSIS - ALL SATELLITES")
print("="*90)
print(f"{'Sat ID':<8} {'Name':<25} {'RMS [Hz]':<12} {'Mean [Hz]':<12} {'Std [Hz]':<12} {'Max [Hz]':<12} {'Points':<8} {'Status'}")
print("-"*90)

sorted_sats = sorted(residuals.items(), key=lambda x: x[1]['rms'])

for sat_id, stats in sorted_sats:
    rms = stats['rms']
    mean = stats['mean']
    std = stats['std']
    max_val = stats['max']
    n = stats['n_points']
    name = stats['name']
    
    # Determine status
    if rms > 2000:
        status = "🔴 GHOST?"
    elif rms > 1000:
        status = "🟡 SUSPECT"
    elif rms > 500:
        status = "🟢 MARGINAL"
    else:
        status = "✅ GOOD"
    
    print(f"{int(sat_id):<8} {name:<25} {rms:>10.1f}   {mean:>10.1f}   {std:>10.1f}   {max_val:>10.1f}   {n:>6}   {status}")

print("="*90)

# Recommend satellites to exclude
print("\n" + "="*90)
print("RECOMMENDATIONS")
print("="*90)

bad_sats = [int(sat_id) for sat_id, stats in residuals.items() if stats['rms'] > 1000]
suspect_sats = [int(sat_id) for sat_id, stats in residuals.items() if 500 < stats['rms'] <= 1000]

if bad_sats:
    print(f"\n🔴 DEFINITELY EXCLUDE (RMS > 1000 Hz):")
    print(f"   GHOST_SATELLITES = {bad_sats}")
    
if suspect_sats:
    print(f"\n🟡 CONSIDER EXCLUDING (RMS 500-1000 Hz):")
    print(f"   Additional satellites: {suspect_sats}")

good_sats = [int(sat_id) for sat_id, stats in residuals.items() if stats['rms'] <= 500]
print(f"\n✅ GOOD SATELLITES (RMS < 500 Hz): {len(good_sats)} satellites")

print("\nTo exclude satellites, update your script:")
print(f"GHOST_SATELLITES = {bad_sats + [124]}")  # Include 124 which was already identified

print("\n" + "="*90)

