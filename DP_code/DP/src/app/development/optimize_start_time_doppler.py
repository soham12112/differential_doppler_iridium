"""
Optimize start time by minimizing Doppler positioning errors
This is the CORRECT way to optimize - uses actual positioning accuracy
"""

from astropy.time import Time, TimeDelta
import astropy.units as unit
import numpy as np
import pickle

from src.config.setup import *
from src.config.locations import LOCATIONS
from src.satellites.download_tle import download_tles
from src.navigation.curve_fit_method import solve
from src.config.parameters import CurveFitMethodParameters
from src.navigation.data_processing import process_received_frames
from src.radio.iridium_offline_radio import IridiumOfflineRadio
from src.navigation.calculations import latlon_distance


def evaluate_position_error(start_time_offset, base_start_time, frames_array, satellites, 
                            true_location, time_corr_factor=1.0):
    """
    Evaluate positioning error for a given start time offset
    
    :param start_time_offset: Offset in seconds to add to base_start_time
    :param base_start_time: Base start time (ISO string)
    :param frames_array: Processed frames array
    :param satellites: Satellite TLEs
    :param true_location: (lat, lon, alt) tuple
    :param time_corr_factor: Time correction factor for clock drift
    :return: Position error in meters
    """
    # Adjust start time
    adjusted_start_time = (Time(base_start_time, format="iso", scale="utc") + 
                          TimeDelta(start_time_offset * unit.s)).iso
    
    # Process frames with adjusted time
    try:
        nav_data = process_received_frames(frames_array, adjusted_start_time, satellites["Iridium"],
                                          time_correction_factor=time_corr_factor)
        
        # Skip if not enough data
        if len(nav_data) < 50:
            return 1e9  # Return huge error
        
        # Configure for 2D positioning if altitude is provided
        parameters = CurveFitMethodParameters()
        if true_location[2] is not None:
            alt_fixed = true_location[2]
            parameters.iteration.alt = type(parameters.iteration.alt)(0, 0, alt_fixed, alt_fixed)
            parameters.grid_search.alt.steps = [0]
        
        # Solve position
        lat, lon, alt, off, dft = solve(nav_data, satellites["Iridium"], parameters, 
                                        true_location=true_location)
        
        # Calculate error
        error = latlon_distance(true_location[0], lat, true_location[1], lon)
        return error
        
    except Exception as e:
        print(f"  Error at offset {start_time_offset:+.1f}s: {e}")
        return 1e9


def optimize_start_time_doppler(initial_start_time, frames, satellites, true_location,
                                coarse_range=30, coarse_step=2.0, 
                                fine_range=5, fine_step=0.25):
    """
    Optimize start time by minimizing Doppler positioning error
    
    :param initial_start_time: Initial guess for start time (ISO format string)
    :param frames: List of decoded frames (text)
    :param satellites: Satellite TLEs dict
    :param true_location: (lat, lon, alt) tuple for the true location
    :param coarse_range: Coarse search range in seconds (±)
    :param coarse_step: Coarse search step size in seconds
    :param fine_range: Fine search range in seconds (±)
    :param fine_step: Fine search step size in seconds
    :return: Optimized start time offset in seconds, minimum position error
    """
    
    # Process frames once (doesn't depend on start time)
    print("\nProcessing frames...")
    radio = IridiumOfflineRadio(frames, file_is_parsed=True, non_ira_frames=False, drop_frequencies=True)
    frames_array = np.array(radio.get_frames())
    print(f"Loaded {len(frames_array)} IRA frames")
    
    if len(frames_array) < 100:
        print("ERROR: Not enough frames for optimization")
        return 0, None
    
    # Coarse search
    print(f"\n{'='*80}")
    print(f"COARSE SEARCH: ±{coarse_range}s in {coarse_step}s steps")
    print(f"{'='*80}")
    
    coarse_offsets = np.arange(-coarse_range, coarse_range + coarse_step, coarse_step)
    coarse_errors = []
    
    for offset in coarse_offsets:
        error = evaluate_position_error(offset, initial_start_time, frames_array, 
                                       satellites, true_location)
        coarse_errors.append(error)
        print(f"Offset {offset:+6.1f}s: position error = {error/1000:.2f} km")
    
    # Find best coarse offset
    best_coarse_idx = np.argmin(coarse_errors)
    best_coarse_offset = coarse_offsets[best_coarse_idx]
    best_coarse_error = coarse_errors[best_coarse_idx]
    
    print(f"\nBest coarse offset: {best_coarse_offset:+.1f}s (error: {best_coarse_error/1000:.2f} km)")
    
    # Fine search
    print(f"\n{'='*80}")
    print(f"FINE SEARCH: {best_coarse_offset-fine_range:+.1f}s to {best_coarse_offset+fine_range:+.1f}s in {fine_step}s steps")
    print(f"{'='*80}")
    
    fine_offsets = np.arange(best_coarse_offset - fine_range,
                            best_coarse_offset + fine_range + fine_step/2, 
                            fine_step)
    fine_errors = []
    
    for offset in fine_offsets:
        error = evaluate_position_error(offset, initial_start_time, frames_array,
                                       satellites, true_location)
        fine_errors.append(error)
        print(f"Offset {offset:+6.2f}s: position error = {error/1000:.3f} km")
    
    # Find best fine offset
    best_fine_idx = np.argmin(fine_errors)
    best_offset = fine_offsets[best_fine_idx]
    best_error = fine_errors[best_fine_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Initial start time:   {initial_start_time}")
    print(f"Optimal offset:       {best_offset:+.2f} seconds")
    optimized_time = Time(initial_start_time, format='iso', scale='utc') + TimeDelta(best_offset * unit.s)
    print(f"Optimized start time: {optimized_time.iso}")
    print(f"Position error:       {best_error:.0f} m ({best_error/1000:.3f} km)")
    print(f"{'='*80}\n")
    
    return best_offset, best_error


if __name__ == "__main__":
    # Get true location from setup
    true_lat, true_lon, true_alt = LOCATIONS[LOCATION]
    TRUE_LOCATION = (true_lat, true_lon, true_alt)
    
    print(f"{'='*80}")
    print(f"DOPPLER-BASED START TIME OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Dataset:       {EXP_NAME}")
    print(f"Location:      {LOCATION} -> Lat: {true_lat:.6f}°, Lon: {true_lon:.6f}°, Alt: {true_alt} m")
    print(f"Data path:     {DATA_PATH}")
    print(f"Frame file:    {FRAME_FILE}")
    
    # Check if START_TIME is set
    if START_TIME is None:
        print("\nERROR: You must set an initial START_TIME guess in setup.py")
        print("Even a rough estimate (±1 minute) is fine for optimization")
        exit(1)
    
    print(f"Initial guess: {START_TIME}\n")
    
    # Load data
    satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")
    
    with open(DATA_PATH + FRAME_FILE, "r") as file:
        frames = file.readlines()
    
    # Optimize
    best_offset, error = optimize_start_time_doppler(
        START_TIME, frames, satellites, TRUE_LOCATION,
        coarse_range=30,   # Search ±30 seconds
        coarse_step=2.0,   # 2 second steps
        fine_range=5,      # Fine search ±5 seconds
        fine_step=0.25     # 0.25 second steps
    )
    
    # Calculate and save optimized time
    optimized_time = Time(START_TIME, format="iso", scale="utc") + TimeDelta(best_offset * unit.s)
    
    output_file = DATA_PATH + "start_time_optimized.txt"
    with open(output_file, "w") as f:
        f.write(f"{optimized_time.iso}\n")
    
    print(f"Optimized start time saved to: {output_file}")
    print(f"\nTo use this optimized time:")
    print(f"1. Copy {output_file} to start_time.txt")
    print(f"2. Or update setup.py: START_TIME = \"{optimized_time.iso}\"")
    print(f"3. Then rerun process_offline_data.py to regenerate saved_nav_data.pickle")



