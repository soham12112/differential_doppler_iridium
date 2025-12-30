"""
Comprehensive diagnostic script to identify the source of positioning errors
Tests frequency offsets, time offsets, and ghost satellites
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
from src.navigation.calculations import latlon_distance

LON_HOME, LAT_HOME, ALT_HOME = LOCATIONS["HOME"][0], LOCATIONS["HOME"][1], LOCATIONS["HOME"][2]


def calculate_satellite_residuals(nav_data, lat, lon, alt, off, dft, satellites):
    """Calculate RMS residuals for each satellite"""
    residuals_by_sat = {}
    
    for sat_id in np.unique(nav_data[:, IDX.sat_id]):
        mask = nav_data[:, IDX.sat_id] == sat_id
        sat_data = nav_data[mask]
        
        try:
            sat = satellites[str(int(sat_id))]
            sat_name = sat.name
        except KeyError:
            continue
        
        # Calculate predicted doppler
        measured_freq = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
        times = sat_data[:, IDX.t]
        
        r_sat = np.column_stack((sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]))
        v_sat = np.column_stack((sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]))
        
        # User position
        r_user = (EarthLocation.from_geodetic(lon, lat, alt)
                 .get_itrs().cartesian.without_differentials())
        r_user = np.array([r_user.x.value, r_user.y.value, r_user.z.value]) * 1000
        
        # Calculate range rate
        r_sat_m = r_sat.T * 1000
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
        rms = np.sqrt(np.mean(residuals**2))
        
        residuals_by_sat[sat_id] = {
            'name': sat_name,
            'rms': rms,
            'n_points': len(residuals),
            'mean': np.mean(residuals),
            'std': np.std(residuals)
        }
    
    return residuals_by_sat


def test_with_corrections(nav_data, satellites, time_offset=0, freq_offset=0, 
                          ghost_sats=None, mode="2D"):
    """Test positioning with corrections applied"""
    if ghost_sats is None:
        ghost_sats = [124]
    
    # Apply corrections
    adjusted_data = nav_data.copy()
    adjusted_data[:, IDX.t] += time_offset
    adjusted_data[:, IDX.f] += freq_offset
    
    # Filter ghost satellites
    mask = ~np.isin(adjusted_data[:, IDX.sat_id], ghost_sats)
    adjusted_data = adjusted_data[mask]
    
    # Setup parameters
    parameters = CurveFitMethodParameters()
    if mode == "2D":
        parameters.iteration.alt = type(parameters.iteration.alt)(0, 0, ALT_HOME, ALT_HOME)
        parameters.grid_search.alt.steps = [0]
    
    # Solve
    try:
        lat, lon, alt, off, dft = solve(adjusted_data, satellites, parameters, 
                                        true_location=(LAT_HOME, LON_HOME, ALT_HOME))
        error = latlon_distance(LAT_HOME, lat, LON_HOME, lon)
        
        # Calculate residuals
        residuals = calculate_satellite_residuals(adjusted_data, lat, lon, alt, off, dft, satellites)
        avg_rms = np.mean([r['rms'] for r in residuals.values()])
        max_rms = np.max([r['rms'] for r in residuals.values()])
        
        return {
            'success': True,
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'offset': off,
            'drift': dft,
            'error': error,
            'n_sats': len(residuals),
            'avg_rms': avg_rms,
            'max_rms': max_rms,
            'residuals': residuals
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


print("="*100)
print("COMPREHENSIVE POSITION ERROR DIAGNOSIS")
print("="*100)
print(f"True position: lat {LAT_HOME:.4f}°, lon {LON_HOME:.4f}°, alt {ALT_HOME} m")
print(f"Data path: {DATA_PATH}")
print(f"Experiment: {EXP_NAME}")
print("="*100 + "\n")

# Load data
print("Loading data...")
with open(DATA_PATH + SAVED_DATA_FILE, "rb") as file:
    nav_data = pickle.load(file)
print(f"Loaded {len(nav_data)} measurements")

# Load satellites
all_satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")
satellites = all_satellites["Iridium"]
print(f"Loaded {len(satellites)} satellite TLEs\n")

# ============================================================================
# TEST 1: BASELINE (current configuration)
# ============================================================================
print("\n" + "="*100)
print("TEST 1: BASELINE - Current configuration")
print("="*100)
result = test_with_corrections(nav_data, satellites, mode="2D")
if result['success']:
    print(f"✓ Position error: {result['error']:,.0f} m ({result['error']/1000:.1f} km)")
    print(f"  Estimated: lat {result['lat']:.4f}°, lon {result['lon']:.4f}°")
    print(f"  Clock offset: {result['offset']:,.0f} Hz")
    print(f"  Clock drift: {result['drift']:.4f} Hz/s")
    print(f"  Satellites used: {result['n_sats']}")
    print(f"  Average RMS residual: {result['avg_rms']:.1f} Hz")
    print(f"  Max RMS residual: {result['max_rms']:.1f} Hz")
    baseline_error = result['error']
else:
    print(f"✗ FAILED: {result['error']}")
    baseline_error = float('inf')

# ============================================================================
# TEST 2: IDENTIFY GHOST SATELLITES
# ============================================================================
print("\n" + "="*100)
print("TEST 2: GHOST SATELLITE IDENTIFICATION")
print("="*100)

if result['success'] and result['residuals']:
    print(f"\n{'Sat ID':<8} {'Name':<25} {'RMS [Hz]':<12} {'Points':<8} Status")
    print("-"*70)
    
    sorted_sats = sorted(result['residuals'].items(), key=lambda x: x[1]['rms'])
    potential_ghosts = [124]  # Already known
    
    for sat_id, stats in sorted_sats:
        rms = stats['rms']
        n = stats['n_points']
        name = stats['name']
        
        if rms > 1500:
            status = "🔴 GHOST"
            if int(sat_id) not in potential_ghosts:
                potential_ghosts.append(int(sat_id))
        elif rms > 800:
            status = "🟡 SUSPECT"
        elif rms > 400:
            status = "🟢 OK"
        else:
            status = "✅ GOOD"
        
        print(f"{int(sat_id):<8} {name:<25} {rms:>10.1f}   {n:>6}   {status}")
    
    print(f"\nIdentified ghost/suspect satellites: {potential_ghosts}")

# ============================================================================
# TEST 3: TEST WITH GHOST FILTERING
# ============================================================================
if len(potential_ghosts) > 1:
    print("\n" + "="*100)
    print("TEST 3: WITH GHOST SATELLITE FILTERING")
    print("="*100)
    result_filtered = test_with_corrections(nav_data, satellites, 
                                           ghost_sats=potential_ghosts, mode="2D")
    if result_filtered['success']:
        improvement = baseline_error - result_filtered['error']
        print(f"✓ Position error: {result_filtered['error']:,.0f} m ({result_filtered['error']/1000:.1f} km)")
        print(f"  Improvement: {improvement:,.0f} m ({'+' if improvement > 0 else ''}{improvement/1000:.1f} km)")
        print(f"  Satellites used: {result_filtered['n_sats']}")
        print(f"  Average RMS: {result_filtered['avg_rms']:.1f} Hz")
        
        if improvement > 1000:
            print(f"\n✅ GHOST FILTERING HELPS! Add to your code:")
            print(f"   GHOST_SATELLITES = {potential_ghosts}")
            baseline_error = result_filtered['error']  # Update baseline

# ============================================================================
# TEST 4: SWEEP TIME OFFSETS
# ============================================================================
print("\n" + "="*100)
print("TEST 4: TIME OFFSET SWEEP")
print("="*100)
print(f"Testing time corrections from -10 to +10 seconds...")
print(f"\n{'Time Offset [s]':<18} {'Error [m]':<15} {'Error [km]':<12} Improvement   Status")
print("-"*80)

time_offsets = [-10, -5, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5, 10]
best_time_result = None
best_time_error = baseline_error

for time_offset in time_offsets:
    result_time = test_with_corrections(nav_data, satellites, 
                                       time_offset=time_offset, 
                                       ghost_sats=potential_ghosts if len(potential_ghosts) > 1 else [124],
                                       mode="2D")
    if result_time['success']:
        error = result_time['error']
        improvement = baseline_error - error
        
        if error < best_time_error:
            best_time_error = error
            best_time_result = (time_offset, result_time)
        
        status = "🏆" if error < 1000 else "✅" if error < 5000 else "🟢" if error < 10000 else "🟡"
        print(f"{time_offset:>+8.2f}          {error:>12,.0f}   {error/1000:>8.1f}     {improvement:>+7.0f} m   {status}")

if best_time_result and abs(best_time_result[0]) > 0.1:
    print(f"\n✅ TIME CORRECTION HELPS!")
    print(f"   Best offset: {best_time_result[0]:+.2f} seconds")
    print(f"   Error improvement: {baseline_error - best_time_error:,.0f} m")
    print(f"\n   Add to your code (line 163 in find_position_offline_2d.py):")
    print(f"   MANUAL_OFFSET = {best_time_result[0]:.2f}")
    baseline_error = best_time_error

# ============================================================================
# TEST 5: SWEEP FREQUENCY OFFSETS
# ============================================================================
print("\n" + "="*100)
print("TEST 5: FREQUENCY OFFSET SWEEP")
print("="*100)
print(f"Testing frequency corrections (SDR calibration error)...")
print(f"\n{'Freq Offset [Hz]':<18} {'Error [m]':<15} {'Error [km]':<12} Improvement   Status")
print("-"*80)

freq_offsets = [-5000, -2500, -1000, -500, -250, 0, 250, 500, 1000, 2500, 5000]
best_freq_result = None
best_freq_error = baseline_error

for freq_offset in freq_offsets:
    result_freq = test_with_corrections(nav_data, satellites, 
                                       freq_offset=freq_offset,
                                       ghost_sats=potential_ghosts if len(potential_ghosts) > 1 else [124],
                                       mode="2D")
    if result_freq['success']:
        error = result_freq['error']
        improvement = baseline_error - error
        
        if error < best_freq_error:
            best_freq_error = error
            best_freq_result = (freq_offset, result_freq)
        
        status = "🏆" if error < 1000 else "✅" if error < 5000 else "🟢" if error < 10000 else "🟡"
        print(f"{freq_offset:>+8}          {error:>12,.0f}   {error/1000:>8.1f}     {improvement:>+7.0f} m   {status}")

if best_freq_result and abs(best_freq_result[0]) > 100:
    print(f"\n✅ FREQUENCY CORRECTION HELPS!")
    print(f"   Best offset: {best_freq_result[0]:+} Hz")
    print(f"   Error improvement: {baseline_error - best_freq_error:,.0f} m")
    print(f"\n   This indicates SDR frequency calibration error!")
    print(f"   Add to your data processing code:")
    print(f"   FREQ_CORRECTION = {best_freq_result[0]}")
    baseline_error = best_freq_error

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSIS SUMMARY")
print("="*100)

print(f"\nOriginal error:     {result['error'] if result['success'] else 'N/A':>12,.0f} m")
print(f"Best achievable:    {min(best_time_error, best_freq_error):>12,.0f} m")
print(f"Total improvement:  {(result['error'] if result['success'] else 0) - min(best_time_error, best_freq_error):>12,.0f} m")

print("\n" + "="*100)
print("RECOMMENDATIONS:")
print("="*100)

issues_found = []

if len(potential_ghosts) > 1:
    issues_found.append("ghost_sats")
    print(f"\n1. 🔴 GHOST SATELLITES DETECTED")
    print(f"   Add to your scripts: GHOST_SATELLITES = {potential_ghosts}")

if best_time_result and abs(best_time_result[0]) > 0.25:
    issues_found.append("time_offset")
    print(f"\n2. 🔴 START TIME ERROR DETECTED")
    print(f"   Your start time is off by ~{best_time_result[0]:+.2f} seconds")
    print(f"   Add to line 163 in find_position_offline_2d.py:")
    print(f"   MANUAL_OFFSET = {best_time_result[0]:.2f}")

if best_freq_result and abs(best_freq_result[0]) > 250:
    issues_found.append("freq_offset")
    print(f"\n3. 🔴 SDR FREQUENCY OFFSET DETECTED")
    print(f"   Your SDR has a {best_freq_result[0]:+} Hz calibration error")
    print(f"   You need to apply this correction in your data processing")

if not issues_found:
    print("\n⚠️  No obvious systematic errors found.")
    print("   Possible issues:")
    print("   - TLE inaccuracy (but unlikely to cause 15 km error)")
    print("   - Incorrect satellite ID mapping in iridium_channels.py")
    print("   - Data quality issues")
    print("   - Need more satellites or longer observation time")

print("\n" + "="*100)
print("\nABOUT YOUR QUESTION: Will a 2nd SDR help with TLE errors?")
print("="*100)
print("❌ NO - A 2nd SDR at the SAME location will NOT help with TLE errors!")
print("   Both SDRs would use the same TLEs → same position errors")
print("\n✅ What WOULD help:")
print("   • 2 SDRs at DIFFERENT known locations (differential positioning)")
print("   • Better ephemeris data (not publicly available for Iridium)")
print("   • Fix systematic errors first (time/frequency offsets)")
print("\n" + "="*100)

