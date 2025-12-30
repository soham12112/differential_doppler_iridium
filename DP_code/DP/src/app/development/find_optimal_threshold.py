"""
Find optimal RMS threshold by testing multiple values
Shows tradeoff between satellite quality and quantity/geometry
"""

import pickle
import numpy as np
from astropy.coordinates import EarthLocation

from src.config.locations import LOCATIONS
from src.config.setup import *
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve, C
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import NavDataArrayIndices as IDX, find_curves
from src.navigation.calculations import latlon_distance

LON_HOME, LAT_HOME, ALT_HOME = LOCATIONS["HOME"][0], LOCATIONS["HOME"][1], LOCATIONS["HOME"][2]


def calculate_residuals_per_satellite(curve_array, lat, lon, alt, off, dft, satellites):
    """Calculate RMS residuals for each satellite"""
    residuals_by_sat = {}
    
    for sat_id in np.unique(curve_array[:, IDX.sat_id]):
        mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[mask]
        
        measured_curve = np.column_stack((sat_data[:, IDX.t],
                                         sat_data[:, IDX.f] - sat_data[:, IDX.fb],
                                         sat_data[:, IDX.fb]))
        
        r_sat_arr = np.column_stack((sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]))
        v_sat_arr = np.column_stack((sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]))
        
        curve_len = measured_curve.shape[0]
        r_user_arr = (EarthLocation.from_geodetic(lon, lat, alt)
                     .get_itrs().cartesian.without_differentials())
        
        vs, rs, ru = v_sat_arr.T * 1000, r_sat_arr.T * 1000, r_user_arr * np.ones(curve_len)
        ru = np.array([ru.x.to("km").value, ru.y.to("km").value, ru.z.to("km").value]) * 1000
        f_b = measured_curve[:, 2]
        
        rel_vel = np.sum(vs * (rs - ru) / np.linalg.norm(rs - ru, axis=0), axis=0)
        f_d = -1 * rel_vel * f_b / C
        f_drift = (measured_curve[:, 0] - np.min(measured_curve[:, 0])) * dft
        predicted = f_d + off + f_drift
        residuals = measured_curve[:, 1] - predicted
        rms = np.sqrt(np.mean(residuals**2))
        
        try:
            sat_name = satellites[str(int(sat_id))].name
        except:
            sat_name = f"SAT-{int(sat_id)}"
        
        residuals_by_sat[sat_id] = (sat_name, rms, len(residuals))
    
    return residuals_by_sat


def test_threshold(nav_data_array, satellites, parameters, threshold, true_location):
    """Test a specific RMS threshold"""
    
    # Initial solve
    detected_curves = find_curves(nav_data_array, 
                                  max_time_gap=parameters.max_time_gap,
                                  min_curve_length=parameters.min_curve_length)
    
    if len(detected_curves) == 0:
        return None
    
    try:
        lat, lon, alt, off, dft = solve(nav_data_array, satellites, parameters, 
                                       true_location=true_location)
    except:
        return None
    
    # Calculate residuals
    curve_array = np.vstack(detected_curves)
    residuals = calculate_residuals_per_satellite(curve_array, lat, lon, alt, off, dft, satellites)
    
    # Filter satellites
    good_sats = [sat_id for sat_id, (_, rms, _) in residuals.items() if rms < threshold]
    
    if len(good_sats) < 3:
        return None
    
    # Solve with filtered data
    mask = np.isin(nav_data_array[:, IDX.sat_id], good_sats)
    filtered_data = nav_data_array[mask]
    
    try:
        lat2, lon2, alt2, off2, dft2 = solve(filtered_data, satellites, parameters,
                                            true_location=true_location)
        dist_error = latlon_distance(true_location[0], lat2, true_location[1], lon2)
        
        return {
            'n_sats': len(good_sats),
            'n_measurements': len(filtered_data),
            'n_dropped': len(residuals) - len(good_sats),
            'dist_error': dist_error,
            'lat': lat2,
            'lon': lon2,
            'alt': alt2,
            'good_sats': good_sats,
            'residuals': residuals
        }
    except:
        return None


# Load data
print("Loading data...")
with open(DATA_PATH + SAVED_DATA_FILE, "rb") as file:
    saved_nav_data = pickle.load(file)

# Remove known ghost satellite 124
KNOWN_GHOSTS = [124]
mask = ~np.isin(saved_nav_data[:, IDX.sat_id], KNOWN_GHOSTS)
saved_nav_data = saved_nav_data[mask]

satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")["Iridium"]

# Configure for 2D
parameters = CurveFitMethodParameters()
parameters.iteration.alt = type(parameters.iteration.alt)(0, 0, ALT_HOME, ALT_HOME)
parameters.grid_search.alt.steps = [0]

TRUE_LOCATION = (LAT_HOME, LON_HOME, ALT_HOME)

print(f"Total measurements: {len(saved_nav_data)}")
print(f"Testing different RMS thresholds...\n")

# Test multiple thresholds
thresholds = [300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 1700, 2000, 2500, 3000]
results = []

print("="*95)
print(f"{'Threshold':<12} {'#Sats':<8} {'#Dropped':<10} {'#Points':<10} {'Error (m)':<12} {'Status'}")
print("="*95)

for threshold in thresholds:
    result = test_threshold(saved_nav_data, satellites, parameters, threshold, TRUE_LOCATION)
    
    if result is None:
        print(f"{threshold:<12} {'--':<8} {'--':<10} {'--':<10} {'FAILED':<12} ❌")
    else:
        results.append((threshold, result))
        error = result['dist_error']
        n_sats = result['n_sats']
        n_dropped = result['n_dropped']
        n_points = result['n_measurements']
        
        # Status indicator
        if error < 5000:
            status = "🏆 EXCELLENT"
        elif error < 10000:
            status = "✅ GOOD"
        elif error < 15000:
            status = "🟢 OK"
        elif error < 25000:
            status = "🟡 FAIR"
        else:
            status = "🔴 POOR"
        
        print(f"{threshold:<12} {n_sats:<8} {n_dropped:<10} {n_points:<10} {error:>10.0f}   {status}")

print("="*95)

# Find optimal threshold
if results:
    best_threshold, best_result = min(results, key=lambda x: x[1]['dist_error'])
    
    print("\n" + "="*95)
    print("OPTIMAL THRESHOLD FOUND")
    print("="*95)
    print(f"Best RMS threshold: {best_threshold} Hz")
    print(f"Position error:     {best_result['dist_error']:.0f} m")
    print(f"Satellites used:    {best_result['n_sats']}")
    print(f"Satellites dropped: {best_result['n_dropped']}")
    print(f"Measurements:       {best_result['n_measurements']}")
    print(f"\nEstimated position:")
    print(f"  Latitude:  {best_result['lat']:.3f}° (true: {LAT_HOME:.3f}°)")
    print(f"  Longitude: {best_result['lon']:.3f}° (true: {LON_HOME:.3f}°)")
    print(f"  Altitude:  {best_result['alt']:.0f} m (true: {ALT_HOME:.0f} m)")
    
    print(f"\n🎯 UPDATE YOUR SCRIPT WITH:")
    print(f"   RMS_THRESHOLD = {best_threshold}  # Hz")
    
    # Show which satellites would be dropped
    print(f"\nSatellites that would be DROPPED at optimal threshold:")
    for sat_id, (name, rms, n_pts) in best_result['residuals'].items():
        if rms >= best_threshold:
            print(f"  - {name} (ID {int(sat_id)}): RMS = {rms:.0f} Hz, {n_pts} points")
    
    print("="*95)
else:
    print("\n❌ No valid solutions found at any threshold!")




