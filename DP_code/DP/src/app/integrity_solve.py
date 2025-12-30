"""
    Integrity Solver (RAIM) for High-Noise Environments
    
    Iteratively solves position and removes the "worst" satellite
    until all remaining satellites agree (low residuals).
    This fixes large position errors caused by multipath/reflections.
    
    Key improvements over basic approach:
    - Statistical outlier detection (not just fixed threshold)
    - Detailed residual tracking and visualization
    - Protection against over-filtering
    - Saves filtered dataset for reuse
"""

import pickle
import numpy as np
from astropy.coordinates import EarthLocation
from pathlib import Path
from src.config.locations import LOCATIONS
from src.navigation.calculations import latlon_distance
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve, C
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import find_curves, NavDataArrayIndices as IDX


# --- CONFIGURATION ---
# Use your "Home" dataset here
EXPERIMENT_NAME = "b200_28th_night"  # Update if folder name is different
DATA_PATH = f"/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/{EXPERIMENT_NAME}/"
FILE_NAME = "saved_nav_data.pickle"

# Target Location (for verification)
TRUE_LOC_NAME = "HOME"
TRUE_LON, TRUE_LAT, TRUE_ALT = LOCATIONS[TRUE_LOC_NAME]

# Filtering Parameters
MAX_RESIDUAL_ALLOWED = 50.0  # Hz - Target residual level
MIN_SATELLITES_REQUIRED = 4  # Don't drop below this
USE_STATISTICAL_OUTLIER = False  # Use median + 2*MAD instead of fixed threshold


def calculate_residuals(curve_array, lat, lon, alt, off, dft, satellites):
    """
    Calculate RMS residual for each satellite given a position solution
    
    Returns:
        residuals_map: dict {sat_id: rms_residual}
        detailed_residuals: dict {sat_id: array of per-point residuals}
    """
    residuals_map = {}
    detailed_residuals = {}
    
    r_user = (EarthLocation.from_geodetic(lon, lat, alt)
              .get_itrs().cartesian.without_differentials())
    
    # User position vector in meters
    ru_vec = np.array([r_user.x.to("m").value, 
                      r_user.y.to("m").value, 
                      r_user.z.to("m").value])

    for sat_id in np.unique(curve_array[:, IDX.sat_id]):
        mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[mask]
        
        # Satellite State (convert km to m)
        r_sat = np.column_stack((sat_data[:, IDX.x], 
                                sat_data[:, IDX.y], 
                                sat_data[:, IDX.z])) * 1000
        v_sat = np.column_stack((sat_data[:, IDX.vx], 
                                sat_data[:, IDX.vy], 
                                sat_data[:, IDX.vz])) * 1000
        
        # Range Rate
        diff = r_sat - ru_vec[np.newaxis, :]
        dist = np.linalg.norm(diff, axis=1)
        u_vec = diff / dist[:, np.newaxis]
        rel_vel = np.sum(v_sat * u_vec, axis=1)
        
        # Expected Frequency
        f_doppler = -1 * rel_vel * sat_data[:, IDX.fb] / C
        
        # Clock Model (use global start time)
        t_rel = sat_data[:, IDX.t] - np.min(curve_array[:, IDX.t])
        f_clock = off + (dft * t_rel)
        
        f_pred = f_doppler + f_clock
        
        # Measured Frequency
        f_meas = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
        
        # Residuals
        resid = f_meas - f_pred
        rms = np.sqrt(np.mean(resid**2))
        
        residuals_map[sat_id] = rms
        detailed_residuals[sat_id] = resid
        
    return residuals_map, detailed_residuals


def detect_outlier(resid_map, satellites):
    """
    Detect the worst satellite using statistical methods
    
    Returns:
        worst_sat_id, worst_rms, is_outlier, threshold_used
    """
    if len(resid_map) == 0:
        return None, None, False, None
        
    # Find worst
    worst_sat = max(resid_map, key=resid_map.get)
    worst_rms = resid_map[worst_sat]
    
    # Statistical test: is it an outlier?
    if USE_STATISTICAL_OUTLIER and len(resid_map) > 3:
        values = np.array(list(resid_map.values()))
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        # Modified Z-score threshold (robust outlier detection)
        threshold = median + 3.5 * 1.4826 * mad  # 1.4826 makes MAD consistent with std
        is_outlier = worst_rms > threshold
    else:
        threshold = MAX_RESIDUAL_ALLOWED
        is_outlier = worst_rms > threshold
        
    return worst_sat, worst_rms, is_outlier, threshold


def print_residual_summary(resid_map, satellites):
    """Print a nice table of satellite residuals"""
    print("\n  Satellite Residuals:")
    print("  " + "-" * 50)
    
    # Sort by residual (worst first)
    sorted_sats = sorted(resid_map.items(), key=lambda x: x[1], reverse=True)
    
    for sat_id, rms in sorted_sats:
        sat_name = satellites[str(int(sat_id))].name
        status = "⚠️  HIGH" if rms > MAX_RESIDUAL_ALLOWED else "✓  OK"
        print(f"  {status}  {sat_name:20s} (ID {int(sat_id):3d}): {rms:6.1f} Hz")
    
    # Statistics
    values = list(resid_map.values())
    print("  " + "-" * 50)
    print(f"  Statistics: Mean={np.mean(values):.1f} Hz, "
          f"Median={np.median(values):.1f} Hz, "
          f"Max={np.max(values):.1f} Hz")


def run_integrity_solve():
    """Main RAIM solver loop"""
    
    print("=" * 60)
    print("INTEGRITY SOLVER (RAIM) - Multipath/Outlier Rejection")
    print("=" * 60)
    
    # Load data
    data_file = Path(DATA_PATH) / FILE_NAME
    print(f"\nLOADING DATA: {data_file}")
    
    if not data_file.exists():
        print(f"ERROR: File not found: {data_file}")
        print(f"Please update EXPERIMENT_NAME and DATA_PATH in the script.")
        return
        
    with open(data_file, "rb") as f:
        nav_data = pickle.load(f)
    
    print(f"  Loaded {len(nav_data)} measurements")
    print(f"  Satellites present: {len(np.unique(nav_data[:, IDX.sat_id]))}")
    
    # Initial Cleanup (known ghost satellites)
    if 124 in nav_data[:, IDX.sat_id]:
        nav_data = nav_data[nav_data[:, IDX.sat_id] != 124]
        print(f"  Removed ghost satellite 124")
    
    # Load TLEs
    print("\nLoading TLEs...")
    satellites = download_tles(constellations=["Iridium"], 
                              offline_dir="tmp/download/")["Iridium"]
    
    # Setup Solver (2D mode - fixed altitude)
    params = CurveFitMethodParameters()
    params.iteration.alt = type(params.iteration.alt)(0, 0, 0, 0)
    
    print(f"\nTrue Location: {TRUE_LAT:.6f}, {TRUE_LON:.6f} ({TRUE_LOC_NAME})")
    print(f"Solver Mode: 2D (altitude fixed at 0m)")
    print(f"Residual Threshold: {MAX_RESIDUAL_ALLOWED} Hz")
    print(f"Minimum Satellites: {MIN_SATELLITES_REQUIRED}")
    
    # --- ITERATIVE SOLVER LOOP ---
    active_data = nav_data.copy()
    iteration = 0
    history = []
    
    while True:
        iteration += 1
        num_sats = len(np.unique(active_data[:, IDX.sat_id]))
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} - {num_sats} Satellites")
        print(f"{'='*60}")
        
        if num_sats < MIN_SATELLITES_REQUIRED:
            print("⚠️  WARNING: Too few satellites remaining. Stopping.")
            break

        # 1. Solve with current set
        try:
            res = solve(active_data, satellites, params, 
                       true_location=(TRUE_LAT, TRUE_LON, TRUE_ALT))
            lat, lon, alt, off, dft = res
        except Exception as e:
            print(f"❌ Solver failed: {e}")
            break
            
        # 2. Check Accuracy
        dist_error = latlon_distance(TRUE_LAT, lat, TRUE_LON, lon)
        print(f"\n  Position: {lat:.6f}°, {lon:.6f}°")
        print(f"  Error:    {dist_error:.0f} m")
        print(f"  Clock:    Offset={off:.1f} Hz, Drift={dft*1e6:.3f} Hz/s")
        
        # 3. Calculate Residuals
        resid_map, detailed_resid = calculate_residuals(
            active_data, lat, lon, alt, off, dft, satellites)
        
        # 4. Display residual summary
        print_residual_summary(resid_map, satellites)
        
        # 5. Detect outlier
        worst_sat, worst_rms, is_outlier, threshold = detect_outlier(
            resid_map, satellites)
        
        if worst_sat is None:
            break
            
        sat_name = satellites[str(int(worst_sat))].name
        
        # 6. Record history
        history.append({
            'iteration': iteration,
            'num_sats': num_sats,
            'error_m': dist_error,
            'lat': lat,
            'lon': lon,
            'worst_sat': int(worst_sat),
            'worst_sat_name': sat_name,
            'worst_rms': worst_rms,
            'all_residuals': resid_map.copy()
        })
        
        # 7. Decide: Drop or Stop?
        if is_outlier:
            print(f"\n  🗑️  ACTION: DROP {sat_name} (ID {int(worst_sat)})")
            print(f"     RMS {worst_rms:.1f} Hz > Threshold {threshold:.1f} Hz")
            active_data = active_data[active_data[:, IDX.sat_id] != worst_sat]
        else:
            print(f"\n  ✅ SUCCESS: All residuals acceptable")
            print(f"     Worst satellite {sat_name}: {worst_rms:.1f} Hz ≤ {threshold:.1f} Hz")
            break
            
    # --- FINAL RESULTS ---
    print(f"\n{'='*60}")
    print(f"FINAL RESULT (Integrity Filtered)")
    print(f"{'='*60}")
    
    final_sats = np.unique(active_data[:, IDX.sat_id])
    print(f"\n  Position:     {lat:.6f}°, {lon:.6f}°")
    print(f"  True Pos:     {TRUE_LAT:.6f}°, {TRUE_LON:.6f}°")
    print(f"  Error:        {dist_error:.0f} m")
    print(f"  Satellites:   {len(final_sats)} used")
    
    print(f"\n  Satellites Used:")
    for sat_id in final_sats:
        sat_name = satellites[str(int(sat_id))].name
        rms = resid_map[sat_id]
        print(f"    - {sat_name:20s} (ID {int(sat_id):3d}): {rms:.1f} Hz")
    
    # Show improvement
    if len(history) > 1:
        initial_error = history[0]['error_m']
        improvement = initial_error - dist_error
        print(f"\n  Improvement:  {improvement:.0f} m ({improvement/initial_error*100:.1f}% reduction)")
    
    # Save filtered dataset
    output_file = Path(DATA_PATH) / "saved_nav_data_filtered.pickle"
    with open(output_file, "wb") as f:
        pickle.dump(active_data, f)
    print(f"\n  💾 Filtered dataset saved: {output_file}")
    
    # Save history
    history_file = Path(DATA_PATH) / "integrity_solve_history.pickle"
    with open(history_file, "wb") as f:
        pickle.dump(history, f)
    print(f"  💾 Iteration history saved: {history_file}")
    
    print(f"\n{'='*60}\n")
    
    return active_data, history


if __name__ == "__main__":
    run_integrity_solve()

