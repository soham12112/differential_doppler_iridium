"""
    Differential Doppler Positioning
    
    Uses base station data (Office) to improve the accuracy of rover data (Home)
    by calculating and applying error corrections from the known base location.
"""

import pickle
import numpy as np
from scipy.interpolate import interp1d
from astropy.coordinates import EarthLocation

from src.config.locations import LOCATIONS
from src.navigation.calculations import latlon_distance
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve, C
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import find_curves
from src.navigation.data_processing import NavDataArrayIndices as IDX


# --- CONFIGURATION ---
# Update these paths to point to your pickle files
WORKING_DIR = "/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data"
BASE_EXPERIMENT = "b200_30_office"  # Office data (known location)
ROVER_EXPERIMENT = "ant_30_home"    # Home data (unknown location)

BASE_FILE = f"{WORKING_DIR}/{BASE_EXPERIMENT}/saved_nav_data.pickle"
ROVER_FILE = f"{WORKING_DIR}/{ROVER_EXPERIMENT}/saved_nav_data.pickle"

# Locations (LOCATIONS stores as: lon, lat, alt)
BASE_LON, BASE_LAT, BASE_ALT = LOCATIONS["FEL"]    # Office (known)
ROVER_LON_TRUE, ROVER_LAT_TRUE, ROVER_ALT = LOCATIONS["HOME"]  # Home (to verify)

# TLE path
TMP_PATH = "/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/DP_code/DP/tmp/"

# Parameters
GHOST_SATELLITES = [124]  # Filter out ghost satellites
RMS_THRESHOLD = 2000  # Loose filter for Rover


def get_theoretical_doppler(curve_array, lat, lon, alt, satellites):
    """
    Calculates what the Doppler SHOULD be for a known location.
    Returns the theoretical frequency array.
    
    :param curve_array: Navigation data array
    :param lat: Latitude in degrees
    :param lon: Longitude in degrees
    :param alt: Altitude in meters
    :param satellites: Satellite TLE dictionary
    :return: Theoretical Doppler shift array (Hz)
    """
    # User position vector (Fixed Earth)
    r_user_arr = (EarthLocation.from_geodetic(lon, lat, alt)
                  .get_itrs().cartesian.without_differentials())
    
    # Prepare arrays
    r_sat = np.column_stack((curve_array[:, IDX.x], curve_array[:, IDX.y], curve_array[:, IDX.z])) * 1000
    v_sat = np.column_stack((curve_array[:, IDX.vx], curve_array[:, IDX.vy], curve_array[:, IDX.vz])) * 1000
    r_user = np.array([r_user_arr.x.to("km").value, r_user_arr.y.to("km").value, r_user_arr.z.to("km").value]) * 1000
    
    # Calculate Range Rate (Relative Velocity)
    # v_rel = dot(v_sat, unit_vector(r_sat - r_user))
    diff_vec = r_sat - r_user  # Shape (N, 3)
    dist = np.linalg.norm(diff_vec, axis=1)
    unit_vec = diff_vec / dist[:, None]
    
    rel_vel = np.sum(v_sat * unit_vec, axis=1)
    
    # Expected Doppler Shift
    f_d = -1 * rel_vel * curve_array[:, IDX.fb] / C
    
    return f_d


def run_differential():
    print(f"{'='*80}")
    print("DIFFERENTIAL DOPPLER POSITIONING")
    print(f"{'='*80}")
    
    # 1. Load Data
    print(f"\nLoading data...")
    print(f"  Base:  {BASE_FILE}")
    print(f"  Rover: {ROVER_FILE}")
    
    try:
        with open(BASE_FILE, "rb") as f:
            base_data = pickle.load(f)
        with open(ROVER_FILE, "rb") as f:
            rover_data = pickle.load(f)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find data file: {e}")
        print("\nPlease update the paths in the script:")
        print("  BASE_FILE and ROVER_FILE variables")
        return
        
    print(f"  Base:  {len(base_data)} measurements")
    print(f"  Rover: {len(rover_data)} measurements")
    
    # Filter Ghost Satellites
    base_data = base_data[~np.isin(base_data[:, IDX.sat_id], GHOST_SATELLITES)]
    rover_data = rover_data[~np.isin(rover_data[:, IDX.sat_id], GHOST_SATELLITES)]
    
    print(f"\nFiltered ghost satellites: {GHOST_SATELLITES}")
    print(f"  Base:  {len(base_data)} measurements (after filtering)")
    print(f"  Rover: {len(rover_data)} measurements (after filtering)")

    # Load TLEs
    constellations = ["Iridium"]
    satellites = download_tles(constellations=constellations, offline_dir=TMP_PATH + "download/")["Iridium"]
    
    params = CurveFitMethodParameters()
    # Fix altitude for both base and rover
    params.iteration.alt = type(params.iteration.alt)(0, 0, BASE_ALT, BASE_ALT)
    params.grid_search.alt.steps = [0]

    # --- STEP 2: CALCULATE CORRECTIONS FROM BASE ---
    print("\n" + "="*80)
    print("[Base Station] Calculating Error Corrections (Residuals)...")
    print("="*80)
    
    # Solve the base first to get its clock offset/drift
    # This aligns the measurements to the physics model
    res_base = solve(base_data, satellites, params, true_location=(BASE_LAT, BASE_LON, BASE_ALT))
    _, _, _, base_clock_offset, base_clock_drift = res_base
    
    print(f"\nBase Station Solution:")
    print(f"  Clock offset: {base_clock_offset:7.1f} Hz")
    print(f"  Clock drift:  {base_clock_drift:7.3f} Hz/s")
    
    corrections = {}  # Store interpolators
    
    base_ids = np.unique(base_data[:, IDX.sat_id])
    
    print(f"\nCalculating corrections for {len(base_ids)} satellites...")
    
    for sat_id in base_ids:
        mask = base_data[:, IDX.sat_id] == sat_id
        sat_data = base_data[mask]
        
        # 1. Calculate Geometric Doppler (Perfect World)
        f_theo = get_theoretical_doppler(sat_data, BASE_LAT, BASE_LON, BASE_ALT, satellites)
        
        # 2. Calculate Measured Frequency (minus Base Clock errors)
        # We remove base clock so the correction ONLY contains TLE Error + Atmos
        t_norm = sat_data[:, IDX.t] - np.min(base_data[:, IDX.t])
        f_clock_corr = base_clock_offset + (base_clock_drift * t_norm)
        f_measured_clean = (sat_data[:, IDX.f] - sat_data[:, IDX.fb]) - f_clock_corr
        
        # 3. The Residual (The Correction) = Measured - Theoretical
        # This represents: TLE Error + Atmospheric Noise
        residual = f_measured_clean - f_theo
        
        # Create interpolator for time
        # We use bounds_error=False to drop Rover points that don't overlap with Base
        corrections[sat_id] = interp1d(sat_data[:, IDX.t], residual, 
                                       kind='linear', fill_value=np.nan, bounds_error=False)
        
        sat_name = satellites[str(int(sat_id))].name
        rms_residual = np.sqrt(np.mean(residual**2))
        print(f"  Sat {int(sat_id):3d} ({sat_name:20s}): RMS={rms_residual:6.1f} Hz, N={len(sat_data):4d} points")

    # --- STEP 3: APPLY CORRECTIONS TO ROVER ---
    print("\n" + "="*80)
    print("[Rover] Applying Differential Corrections...")
    print("="*80)
    
    corrected_rover_data = []
    
    rover_ids = np.unique(rover_data[:, IDX.sat_id])
    
    overlap_sats = []
    no_overlap_sats = []
    
    for sat_id in rover_ids:
        sat_name = satellites[str(int(sat_id))].name
        
        if sat_id not in corrections:
            no_overlap_sats.append((int(sat_id), sat_name))
            continue
            
        mask = rover_data[:, IDX.sat_id] == sat_id
        sat_data = rover_data[mask].copy()
        
        # Get correction for these timestamps
        corr_vals = corrections[sat_id](sat_data[:, IDX.t])
        
        # Filter out points where we have no overlap (NaNs)
        valid_mask = ~np.isnan(corr_vals)
        if np.sum(valid_mask) < 10:
            no_overlap_sats.append((int(sat_id), sat_name))
            continue
            
        sat_data = sat_data[valid_mask]
        corr_vals = corr_vals[valid_mask]
        
        # Apply Correction
        # We SUBTRACT the error observed at the base
        # Original: Measured = RangeRate + Clock + Error
        # New:      Measured - Error = RangeRate + Clock
        sat_data[:, IDX.f] = sat_data[:, IDX.f] - corr_vals
        
        corrected_rover_data.append(sat_data)
        overlap_sats.append((int(sat_id), sat_name, len(sat_data)))

    print(f"\nOverlapping satellites: {len(overlap_sats)}")
    for sat_id, sat_name, n_points in overlap_sats:
        print(f"  ✓ Sat {sat_id:3d} ({sat_name:20s}): Corrected {n_points:4d} points")
    
    if no_overlap_sats:
        print(f"\nSkipped satellites (no time overlap): {len(no_overlap_sats)}")
        for sat_id, sat_name in no_overlap_sats:
            print(f"  ✗ Sat {sat_id:3d} ({sat_name:20s}): No overlap with Base")

    if not corrected_rover_data:
        print("\nERROR: No overlapping data found!")
        print("Base and Rover measurements don't overlap in time or satellites.")
        return

    final_data = np.vstack(corrected_rover_data)
    
    print(f"\nFinal corrected dataset: {len(final_data)} measurements")

    # --- STEP 4: SOLVE ROVER POSITION ---
    print("\n" + "="*80)
    print("[Rover] Solving Position with Differential Data...")
    print("="*80)
    
    # We allow the solver to find the Rover's unique clock bias
    # The differential process removed the Base's clock and the Satellite errors,
    # leaving only the Rover's clock and the geometric truth.
    
    # Keep altitude fixed for rover too
    params.iteration.alt = type(params.iteration.alt)(0, 0, ROVER_ALT, ROVER_ALT)
    
    res = solve(final_data, satellites, params, true_location=(ROVER_LAT_TRUE, ROVER_LON_TRUE, ROVER_ALT))
    lat, lon, alt, off, dft = res
    
    dist_err = latlon_distance(ROVER_LAT_TRUE, lat, ROVER_LON_TRUE, lon)
    
    print(f"\n{'='*80}")
    print(f"DIFFERENTIAL POSITIONING RESULT:")
    print(f"{'='*80}")
    print(f"  Distance error: {dist_err:7.0f} m")
    print(f"  Latitude:       {lat:10.6f}° (True: {ROVER_LAT_TRUE:10.6f}°)")
    print(f"  Longitude:      {lon:10.6f}° (True: {ROVER_LON_TRUE:10.6f}°)")
    print(f"  Altitude:       {alt:7.0f} m (fixed)")
    print(f"  Clock offset:   {off:7.0f} Hz")
    print(f"  Clock drift:    {dft:7.3f} Hz/s")
    print(f"{'='*80}\n")
    
    # Compare with standard (non-differential) solution
    print("="*80)
    print("[Comparison] Standard Solution (No Differential Correction)")
    print("="*80)
    
    res_std = solve(rover_data, satellites, params, true_location=(ROVER_LAT_TRUE, ROVER_LON_TRUE, ROVER_ALT))
    lat_std, lon_std, alt_std, off_std, dft_std = res_std
    
    dist_err_std = latlon_distance(ROVER_LAT_TRUE, lat_std, ROVER_LON_TRUE, lon_std)
    
    print(f"\nStandard Solution (without differential correction):")
    print(f"  Distance error: {dist_err_std:7.0f} m")
    print(f"  Latitude:       {lat_std:10.6f}°")
    print(f"  Longitude:      {lon_std:10.6f}°")
    
    print(f"\n{'='*80}")
    print(f"IMPROVEMENT:")
    print(f"{'='*80}")
    improvement = dist_err_std - dist_err
    improvement_pct = (improvement / dist_err_std) * 100 if dist_err_std > 0 else 0
    print(f"  Absolute:  {improvement:7.0f} m")
    print(f"  Relative:  {improvement_pct:6.1f}%")
    print(f"  Final error: {dist_err:7.0f} m (was {dist_err_std:7.0f} m)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_differential()

