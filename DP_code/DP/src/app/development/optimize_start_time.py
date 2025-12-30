"""
Optimize start time by minimizing IRA frame position errors
This script requires IRA frames and an approximate start time guess

WARNING: This method is DEPRECATED and often gives poor results!
Use optimize_start_time_doppler.py instead, which optimizes based on
actual Doppler positioning accuracy rather than IRA position matching.
"""

from astropy.time import Time, TimeDelta
import astropy.units as unit
from sgp4.api import SatrecArray
import numpy as np

from src.config.setup import *
from src.radio.iridium_frame_operations import decompose_ira_frame
from src.satellites.download_tle import download_tles, unpack
from src.satellites.predictions import find_closest_tle_id_to_ira_id_by_position


def optimize_start_time_with_ira_frames(initial_start_time, frames, satellites, 
                                         search_range=60, fine_search_range=10):
    """
    Optimize start time by finding offset that minimizes IRA position errors
    
    :param initial_start_time: Initial guess for start time (ISO format string)
    :param frames: List of decoded frames
    :param satellites: Satellite TLEs
    :param search_range: Coarse search range in seconds (±)
    :param fine_search_range: Fine search range in seconds (±)
    :return: Optimized start time offset in seconds, median distance error
    """
    sat_list = unpack(satellites)
    
    # Collect IRA frame data
    print("Parsing IRA frames...")
    ira_data = []
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            print(f"\r{100 * i / len(frames):.1f}%", end="")
        
        if not frame.startswith("IRA"):
            continue
            
        sat_id, rel_time, freq, frame_data = decompose_ira_frame(frame)
        beam_id, lat, lon, alt, x, y, z = frame_data
        
        # Filter valid frames
        if not (sat_id and rel_time and lat is not False and lon is not False 
                and alt is not False and 700 < alt < 900):
            continue
        
        ira_data.append([rel_time / 1000, lat, lon, alt])  # Convert to seconds
    
    print(f"\rFound {len(ira_data)} valid IRA frames")
    
    if len(ira_data) < 10:
        print("ERROR: Not enough valid IRA frames for optimization")
        return 0, None
    
    ira_array = np.array(ira_data)
    
    # Coarse search: -60 to +60 seconds in 5 second steps
    print(f"\nCoarse search: testing offsets from -{search_range} to +{search_range} seconds")
    coarse_offsets = np.arange(-search_range, search_range + 1, 5)
    coarse_errors = []
    
    for offset in coarse_offsets:
        distances = []
        base_time = Time(initial_start_time, format="iso", scale="utc")
        
        # Sample frames (use every 10th to speed up)
        for rel_time, lat, lon, alt in ira_array[::10]:
            time = base_time + TimeDelta((rel_time + offset) * unit.s)
            satrecs = SatrecArray([sat.satrec for sat in sat_list])
            
            _, dist, _, _, _ = find_closest_tle_id_to_ira_id_by_position(
                time, satrecs, sat_list, lat, lon, alt)
            distances.append(dist)
        
        median_error = np.median(distances)
        coarse_errors.append(median_error)
        print(f"Offset {offset:+4.0f}s: median error = {median_error/1000:.1f} km")
    
    # Find best coarse offset
    best_coarse_idx = np.argmin(coarse_errors)
    best_coarse_offset = coarse_offsets[best_coarse_idx]
    
    print(f"\nBest coarse offset: {best_coarse_offset:+.0f}s (error: {coarse_errors[best_coarse_idx]/1000:.1f} km)")
    
    # Fine search: ±10 seconds around best coarse in 0.5 second steps
    print(f"\nFine search: testing offsets from {best_coarse_offset-fine_search_range} to {best_coarse_offset+fine_search_range} seconds")
    fine_offsets = np.arange(best_coarse_offset - fine_search_range, 
                             best_coarse_offset + fine_search_range + 0.1, 0.5)
    fine_errors = []
    
    for offset in fine_offsets:
        distances = []
        base_time = Time(initial_start_time, format="iso", scale="utc")
        
        # Use all frames for fine search
        for rel_time, lat, lon, alt in ira_array:
            time = base_time + TimeDelta((rel_time + offset) * unit.s)
            satrecs = SatrecArray([sat.satrec for sat in sat_list])
            
            _, dist, _, _, _ = find_closest_tle_id_to_ira_id_by_position(
                time, satrecs, sat_list, lat, lon, alt)
            distances.append(dist)
        
        median_error = np.median(distances)
        fine_errors.append(median_error)
        print(f"Offset {offset:+6.1f}s: median error = {median_error/1000:.1f} km")
    
    # Find best fine offset
    best_fine_idx = np.argmin(fine_errors)
    best_offset = fine_offsets[best_fine_idx]
    best_error = fine_errors[best_fine_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Initial start time:  {initial_start_time}")
    print(f"Optimal offset:      {best_offset:+.1f} seconds")
    print(f"Optimized start time: {(Time(initial_start_time, format='iso', scale='utc') + TimeDelta(best_offset * unit.s)).iso}")
    print(f"Median position error: {best_error/1000:.1f} km")
    print(f"{'='*80}\n")
    
    return best_offset, best_error


if __name__ == "__main__":
    # Check if START_TIME is set
    if START_TIME is None:
        print("ERROR: You must set an initial START_TIME guess in setup.py")
        print("Even a rough estimate (±1 minute) is fine for optimization")
        exit(1)
    
    print(f"Initial start time guess: {START_TIME}")
    print(f"Data path: {DATA_PATH}")
    print(f"Frame file: {FRAME_FILE}\n")
    
    # Load data - TLEs are in TMP_PATH
    satellites = download_tles(constellations=CONSTELLATIONS, offline_dir=TMP_PATH + "download/")
    
    with open(DATA_PATH + FRAME_FILE, "r") as file:
        frames = file.readlines()
    
    # Optimize
    best_offset, error = optimize_start_time_with_ira_frames(
        START_TIME, frames, satellites, 
        search_range=80,  # Search ±60 seconds
        fine_search_range=10  # Fine search ±10 seconds around best
    )
    
    # Calculate optimized time
    optimized_time = Time(START_TIME, format="iso", scale="utc") + TimeDelta(best_offset * unit.s)
    
    # Save to file
    output_file = DATA_PATH + "start_time.txt"
    with open(output_file, "w") as f:
        f.write(f"{optimized_time.iso}\n")
    
    print(f"Optimized start time saved to: {output_file}")
    print(f"\nYou can now update setup.py with:")
    print(f'START_TIME = "{optimized_time.iso}"')

