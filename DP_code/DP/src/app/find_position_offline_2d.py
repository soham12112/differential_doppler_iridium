"""
    Find 2D position from processed data (fixed altitude)
    Filters satellites based on residuals to use only the most certain ones
"""

import pickle
import numpy as np
from astropy.coordinates import EarthLocation
from matplotlib import pyplot as plt

from src.config.locations import LOCATIONS
from src.config.setup import *
from src.navigation.calculations import latlon_distance
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve, C
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import nav_data_to_array, find_curves
from src.navigation.data_processing import NavDataArrayIndices as IDX


# Use location from setup.py configuration
LON_HOME, LAT_HOME, ALT_HOME = LOCATIONS[LOCATION][0], LOCATIONS[LOCATION][1], LOCATIONS[LOCATION][2]
print(f"Using location: {LOCATION} -> Lat: {LAT_HOME:.6f}°, Lon: {LON_HOME:.6f}°, Alt: {ALT_HOME} m")


def calculate_residuals_per_satellite(curve_array, lat, lon, alt, off, dft, satellites):
    """
    Calculate RMS residuals for each satellite
    Returns dict: {sat_id: (rms, num_points)}
    """
    residuals_by_sat = {}
    
    for sat_id in np.unique(curve_array[:, IDX.sat_id]):
        mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[mask]
        
        # Calculate predicted doppler for this satellite
        measured_curve = np.column_stack((sat_data[:, IDX.t],
                                         sat_data[:, IDX.f] - sat_data[:, IDX.fb],
                                         sat_data[:, IDX.fb]))
        
        r_sat_arr = np.column_stack((sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]))
        v_sat_arr = np.column_stack((sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]))
        
        # Calculate predicted doppler
        curve_len = measured_curve.shape[0]
        r_user_arr = (EarthLocation.from_geodetic(lon, lat, alt)
                     .get_itrs().cartesian.without_differentials())
        
        vs, rs, ru = v_sat_arr.T * 1000, r_sat_arr.T * 1000, r_user_arr * np.ones(curve_len)
        ru = np.array([ru.x.to("km").value, ru.y.to("km").value, ru.z.to("km").value]) * 1000
        f_b = measured_curve[:, 2]
        
        # Calculate range rate
        rel_vel = np.sum(vs * (rs - ru) / np.linalg.norm(rs - ru, axis=0), axis=0)
        f_d = -1 * rel_vel * f_b / C
        
        # Calculate drift
        f_drift = (measured_curve[:, 0] - np.min(measured_curve[:, 0])) * dft
        
        # Predicted frequency
        predicted = f_d + off + f_drift
        
        # Residuals
        residuals = measured_curve[:, 1] - predicted
        rms = np.sqrt(np.mean(residuals**2))
        
        sat_name = satellites[str(int(sat_id))].name
        residuals_by_sat[sat_id] = (sat_name, rms, len(residuals))
    
    return residuals_by_sat


def filter_satellites_by_residuals(nav_data_array, satellites, params, rms_threshold=1000):
    """
    Run initial solve and filter satellites based on RMS residuals
    
    :param nav_data_array: numpy array of navigation data
    :param satellites: satellite TLEs
    :param params: parameters
    :param rms_threshold: RMS threshold in Hz
    :return: filtered numpy array
    """
    print("\n" + "="*80)
    print("PASS 1: Initial solve to calculate residuals")
    print("="*80)
    
    # Initial solve with all satellites
    detected_curves = find_curves(nav_data_array, max_time_gap=params.max_time_gap, 
                                  min_curve_length=params.min_curve_length)
    
    if len(detected_curves) == 0:
        print("ERROR: No curves detected!")
        return nav_data_array
    
    # Get initial estimate
    lat, lon, alt, off, dft = solve(nav_data_array, satellites, params, 
                                    true_location=(LAT_HOME, LON_HOME, ALT_HOME))
    
    # Calculate residuals
    curve_array = np.vstack(detected_curves)
    residuals = calculate_residuals_per_satellite(curve_array, lat, lon, alt, off, dft, satellites)
    
    # Print and filter
    print("\n" + "="*80)
    print(f"FILTERING: Keeping satellites with RMS < {rms_threshold} Hz")
    print("="*80)
    
    good_sats = []
    bad_sats = []
    
    for sat_id, (sat_name, rms, n_points) in sorted(residuals.items(), key=lambda x: x[1][1]):
        if rms < rms_threshold:
            good_sats.append(sat_id)
            print(f"✓ KEEP: {sat_name:20s} RMS={rms:7.1f} Hz, N={n_points:3d} points")
        else:
            bad_sats.append(sat_id)
            print(f"✗ DROP: {sat_name:20s} RMS={rms:7.1f} Hz, N={n_points:3d} points (ghost)")
    
    if len(good_sats) < 3:
        print(f"\nWARNING: Only {len(good_sats)} good satellites found (need at least 3)!")
        if rms_threshold < 10000:  # Prevent infinite recursion
            print("Trying with more lenient threshold...")
            # Try with more lenient threshold - but use original nav_data_array
            return filter_satellites_by_residuals(nav_data_array, satellites, params, rms_threshold * 2)
        else:
            print("ERROR: Cannot find enough good satellites even with lenient threshold.")
            print("Continuing with all satellites...")
            return nav_data_array
    
    # Filter nav_data_array to keep only good satellites
    mask = np.isin(nav_data_array[:, IDX.sat_id], good_sats)
    filtered_data = nav_data_array[mask]
    
    print(f"\nFiltered data: {len(good_sats)} satellites, {len(filtered_data)} measurements")
    print("="*80 + "\n")
    
    return filtered_data


def slice_data_into_time_chunks(data, chunk_time):
    if chunk_time is None:
        return [data]
    
    time_chunks = list()
    start_time, end_time = data[0, 0], data[-1, 0]
    i = 0
    
    while start_time + chunk_time * i < end_time:
        mask = (data[:, 0] >= start_time + chunk_time * i) & (data[:, 0] < start_time + chunk_time * (i + 1))
        time_chunks.append(data[mask])
        i += 1
    
    return time_chunks


# Load data
with open(DATA_PATH + SAVED_DATA_FILE, "rb") as file:
    saved_nav_data = pickle.load(file)

# --- ADD THIS PATCH ---
# If your optimizer said "Optimal offset: +2.05 seconds"
# and you haven't regenerated the pickle file yet:
MANUAL_OFFSET = 0 
print(f"Applying manual time correction: {MANUAL_OFFSET} seconds")
saved_nav_data[:, IDX.t] += MANUAL_OFFSET

# Filter out ghost satellite 124
GHOST_SATELLITES = [124]  # Add any other ghost satellite IDs here
print(f"Filtering out ghost satellites: {GHOST_SATELLITES}")
mask = ~np.isin(saved_nav_data[:, IDX.sat_id], GHOST_SATELLITES)
saved_nav_data = saved_nav_data[mask]
print(f"Data after filtering: {len(saved_nav_data)} measurements\n")

#satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=DATA_PATH)["Iridium"]
satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")["Iridium"]

# Configure for 2D positioning (fixed altitude)
parameters = CurveFitMethodParameters()
# Fix altitude by setting step to 0 and bounds tight
parameters.iteration.alt = type(parameters.iteration.alt)(0, 0, ALT_HOME, ALT_HOME)
parameters.grid_search.alt.steps = [0]

print(f"\n{'='*80}")
print(f"2D POSITIONING MODE - Altitude fixed at {ALT_HOME} m")
print(f"{'='*80}\n")

# True location for comparison
TRUE_LOCATION = (LAT_HOME, LON_HOME, ALT_HOME)

# Choose initialization method: 'zero_doppler', 'centroid', or 'coarse_grid'
INIT_METHOD = 'centroid'  # Try 'coarse_grid' for best results, 'centroid' for faster but less accurate
print(f"Using initialization method: {INIT_METHOD}")

# Filter satellites based on residuals
RMS_THRESHOLD = 1700  # Hz - adjust this to be more/less strict
filtered_data = filter_satellites_by_residuals(saved_nav_data, satellites, parameters, RMS_THRESHOLD)

# Solve with filtered data
print("\n" + "="*80)
print("PASS 2: Final solve with filtered satellites (2D mode)")
print("="*80 + "\n")

est_state = None
for data in slice_data_into_time_chunks(filtered_data, None):
    res = solve(data, satellites, parameters, init_state=est_state, true_location=TRUE_LOCATION)
    lat, lon, alt, off, dft = res
    # est_state = res
    
    dist = latlon_distance(LAT_HOME, lat, LON_HOME, lon)
    print(f"\n{'='*80}")
    print(f"FINAL RESULT:")
    print(f"  Distance error: {dist:7.0f} m")
    print(f"  Latitude:       {lat:7.3f}° (true: {LAT_HOME:.3f}°)")
    print(f"  Longitude:      {lon:7.3f}° (true: {LON_HOME:.3f}°)")
    print(f"  Altitude:       {alt:7.0f} m (fixed)")
    print(f"  Clock offset:   {off:7.0f} Hz")
    print(f"  Clock drift:    {dft:7.3f} Hz/s")
    print(f"{'='*80}\n")

plt.show()

