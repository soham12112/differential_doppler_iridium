"""
Optimize by sweeping frequency offset and time offset
Tests if a systematic bias is causing the position error
"""

import pickle
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta
import astropy.units as unit

from src.config.locations import LOCATIONS
from src.config.setup import *
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import NavDataArrayIndices as IDX
from src.navigation.calculations import latlon_distance

LON_HOME, LAT_HOME, ALT_HOME = LOCATIONS["HOME"][0], LOCATIONS["HOME"][1], LOCATIONS["HOME"][2]


def adjust_frequencies(nav_data, freq_offset_hz):
    """Add a systematic frequency offset to all measurements"""
    adjusted = nav_data.copy()
    adjusted[:, IDX.f] += freq_offset_hz
    return adjusted


def adjust_times(nav_data, time_offset_s):
    """Add a systematic time offset to all measurements"""
    adjusted = nav_data.copy()
    adjusted[:, IDX.t] += time_offset_s
    return adjusted


# Load data
print("Loading data...")
with open(DATA_PATH + SAVED_DATA_FILE, "rb") as file:
    saved_nav_data = pickle.load(file)

# Filter out known ghost satellites
GHOST_SATELLITES = [124]  # Add any others you've identified
print(f"Filtering out ghost satellites: {GHOST_SATELLITES}")
mask = ~np.isin(saved_nav_data[:, IDX.sat_id], GHOST_SATELLITES)
saved_nav_data = saved_nav_data[mask]
print(f"Using {len(saved_nav_data)} measurements (after filtering)\n")

satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")["Iridium"]

# Test both 2D and 3D
for mode in ["2D", "3D"]:
    print(f"\n{'='*100}")
    print(f"TESTING {mode} MODE")
    print(f"{'='*100}\n")
    
    parameters = CurveFitMethodParameters()
    
    if mode == "2D":
        # Fix altitude
        parameters.iteration.alt = type(parameters.iteration.alt)(0, 0, ALT_HOME, ALT_HOME)
        parameters.grid_search.alt.steps = [0]
    
    TRUE_LOCATION = (LAT_HOME, LON_HOME, ALT_HOME)
    
    # Test 1: Sweep frequency offsets
    print("Testing systematic frequency offsets...")
    print("-"*100)
    print(f"{'Freq Offset [Hz]':<20} {'Error [m]':<15} {'Lat':<12} {'Lon':<12} {'Alt [m]':<10} Status")
    print("-"*100)
    
    freq_offsets = [-5000, -2500, -1000, -500, -250, -100, 0, 100, 250, 500, 1000, 2500, 5000]
    best_freq_result = None
    best_freq_error = float('inf')
    
    for freq_offset in freq_offsets:
        try:
            adjusted_data = adjust_frequencies(saved_nav_data, freq_offset)
            lat, lon, alt, off, dft = solve(adjusted_data, satellites, parameters, true_location=TRUE_LOCATION)
            error = latlon_distance(LAT_HOME, lat, LON_HOME, lon)
            
            if error < best_freq_error:
                best_freq_error = error
                best_freq_result = (freq_offset, lat, lon, alt, off, dft, error)
            
            status = "🏆" if error < 5000 else "✅" if error < 10000 else "🟢" if error < 15000 else "🟡"
            print(f"{freq_offset:>+8} Hz          {error:>10.0f}     {lat:>8.3f}   {lon:>9.3f}   {alt:>6.0f}     {status}")
        except Exception as e:
            print(f"{freq_offset:>+8} Hz          FAILED: {str(e)[:50]}")
    
    if best_freq_result:
        print(f"\n{'='*100}")
        print(f"BEST FREQUENCY OFFSET FOR {mode} MODE:")
        print(f"  Offset:     {best_freq_result[0]:+} Hz")
        print(f"  Error:      {best_freq_result[6]:.0f} m")
        print(f"  Position:   lat {best_freq_result[1]:.3f}°, lon {best_freq_result[2]:.3f}°, alt {best_freq_result[3]:.0f} m")
        print(f"{'='*100}\n")
    
    # Test 2: Sweep time offsets
    print("\nTesting systematic time offsets...")
    print("-"*100)
    print(f"{'Time Offset [s]':<20} {'Error [m]':<15} {'Lat':<12} {'Lon':<12} {'Alt [m]':<10} Status")
    print("-"*100)
    
    time_offsets = [-5.0, -2.5, -1.0, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    best_time_result = None
    best_time_error = float('inf')
    
    for time_offset in time_offsets:
        try:
            adjusted_data = adjust_times(saved_nav_data, time_offset)
            lat, lon, alt, off, dft = solve(adjusted_data, satellites, parameters, true_location=TRUE_LOCATION)
            error = latlon_distance(LAT_HOME, lat, LON_HOME, lon)
            
            if error < best_time_error:
                best_time_error = error
                best_time_result = (time_offset, lat, lon, alt, off, dft, error)
            
            status = "🏆" if error < 5000 else "✅" if error < 10000 else "🟢" if error < 15000 else "🟡"
            print(f"{time_offset:>+8.2f} s          {error:>10.0f}     {lat:>8.3f}   {lon:>9.3f}   {alt:>6.0f}     {status}")
        except Exception as e:
            print(f"{time_offset:>+8.2f} s          FAILED: {str(e)[:50]}")
    
    if best_time_result:
        print(f"\n{'='*100}")
        print(f"BEST TIME OFFSET FOR {mode} MODE:")
        print(f"  Offset:     {best_time_result[0]:+.2f} s")
        print(f"  Error:      {best_time_result[6]:.0f} m")
        print(f"  Position:   lat {best_time_result[1]:.3f}°, lon {best_time_result[2]:.3f}°, alt {best_time_result[3]:.0f} m")
        print(f"{'='*100}\n")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("\nIf systematic offsets improved results significantly:")
print("1. Frequency offset → Your SDR has a calibration error")
print("2. Time offset → Your start_time.txt needs adjustment")
print("3. Neither helps → Issue might be satellite mapping or TLEs")
print("="*100)

