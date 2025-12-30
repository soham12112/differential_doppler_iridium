#!/usr/bin/env python3
"""
Helper script to extract satellite statistics for plotting

This shows how to calculate RMS residuals and other quality metrics
from your navigation data for use in the publication plots.
"""

import numpy as np
from src.utils.data import IDX


def calculate_doppler_rms(sat_data, satellite, params):
    """
    Calculate RMS of Doppler residuals for a single satellite.
    
    This measures the quality of the Doppler measurements by comparing
    observed vs predicted values.
    
    Parameters:
    -----------
    sat_data : ndarray
        Navigation data for a single satellite (filtered from main nav_data array)
        Expected columns: [time, lat, lon, alt, sat_id, doppler, ...]
    
    satellite : Satellite object
        Satellite object from your satellite list
    
    params : Parameters object
        System parameters (from config)
    
    Returns:
    --------
    rms : float
        RMS of Doppler residuals in Hz
    """
    
    from src.navigation.data_processing import compute_residuals
    
    # Extract data
    times = sat_data[:, IDX.time]
    measured_doppler = sat_data[:, IDX.f]
    
    # Get satellite positions at measurement times
    sat_positions = []
    sat_velocities = []
    for t in times:
        pos, vel = satellite.get_position_and_velocity(t)
        sat_positions.append(pos)
        sat_velocities.append(vel)
    
    sat_positions = np.array(sat_positions)
    sat_velocities = np.array(sat_velocities)
    
    # Compute predicted Doppler (simplified - use your actual method)
    # This is a placeholder - adapt to your actual calculation
    residuals = compute_residuals(
        measured_doppler,
        sat_positions,
        sat_velocities,
        params
    )
    
    # Calculate RMS
    rms = np.sqrt(np.mean(residuals**2))
    
    return rms


def analyze_satellite_quality(nav_data, satellites, params):
    """
    Analyze measurement quality for all satellites.
    
    Returns:
    --------
    stats : list of dict
        [{
            'sat_id': int,
            'sat_name': str,
            'norad_id': int,
            'rms_hz': float,
            'n_measurements': int,
            'time_span_s': float,
            'doppler_range_hz': float,
        }, ...]
    """
    
    stats = []
    
    unique_sat_ids = np.unique(nav_data[:, IDX.sat_id])
    
    for sat_id in unique_sat_ids:
        # Filter data for this satellite
        sat_mask = nav_data[:, IDX.sat_id] == sat_id
        sat_data = nav_data[sat_mask]
        
        # Find corresponding satellite object
        satellite = None
        for sat in satellites:
            if sat.id == sat_id or sat.norad_id == sat_id:
                satellite = sat
                break
        
        if satellite is None:
            print(f"Warning: Satellite {sat_id} not found in satellite list")
            continue
        
        # Calculate statistics
        times = sat_data[:, IDX.time]
        doppler = sat_data[:, IDX.f]
        
        # Calculate RMS (if you have residuals available)
        # Otherwise, use standard deviation as a proxy
        try:
            rms = calculate_doppler_rms(sat_data, satellite, params)
        except:
            # Fallback: use standard deviation
            rms = np.std(doppler)
        
        stats.append({
            'sat_id': int(sat_id),
            'sat_name': satellite.name if hasattr(satellite, 'name') else f'Satellite {sat_id}',
            'norad_id': satellite.norad_id if hasattr(satellite, 'norad_id') else int(sat_id),
            'rms_hz': float(rms),
            'n_measurements': len(sat_data),
            'time_span_s': float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            'doppler_range_hz': float(np.max(doppler) - np.min(doppler)),
            'mean_doppler_hz': float(np.mean(doppler)),
            'doppler_std_hz': float(np.std(doppler))
        })
    
    return stats


def compare_base_rover_quality(base_data, rover_data, satellites, params):
    """
    Compare measurement quality between base and rover stations.
    
    Returns data structure suitable for plotting.
    """
    
    base_stats = analyze_satellite_quality(base_data, satellites, params)
    rover_stats = analyze_satellite_quality(rover_data, satellites, params)
    
    # Create lookup dictionaries
    base_dict = {s['sat_id']: s for s in base_stats}
    rover_dict = {s['sat_id']: s for s in rover_stats}
    
    # Find common satellites
    base_sat_ids = set(base_dict.keys())
    rover_sat_ids = set(rover_dict.keys())
    
    common_sats = base_sat_ids & rover_sat_ids
    base_only = base_sat_ids - rover_sat_ids
    rover_only = rover_sat_ids - base_sat_ids
    
    # Prepare comparison data
    comparison = []
    
    # Common satellites
    for sat_id in sorted(common_sats):
        base_s = base_dict[sat_id]
        rover_s = rover_dict[sat_id]
        
        # Check if there's time overlap
        base_times = base_data[base_data[:, IDX.sat_id] == sat_id, IDX.time]
        rover_times = rover_data[rover_data[:, IDX.sat_id] == sat_id, IDX.time]
        
        base_t_start, base_t_end = base_times[0], base_times[-1]
        rover_t_start, rover_t_end = rover_times[0], rover_times[-1]
        
        has_overlap = not (base_t_end < rover_t_start or rover_t_end < base_t_start)
        
        comparison.append({
            'sat_id': sat_id,
            'sat_name': base_s['sat_name'],
            'norad_id': base_s['norad_id'],
            'base_rms': base_s['rms_hz'],
            'rover_rms': rover_s['rms_hz'],
            'base_n': base_s['n_measurements'],
            'rover_n': rover_s['n_measurements'],
            'common_view': has_overlap,
            'common_view_str': 'Yes' if has_overlap else 'No (Time skew)',
            'quality_ratio': rover_s['rms_hz'] / base_s['rms_hz'] if base_s['rms_hz'] > 0 else np.inf
        })
    
    # Base-only satellites
    for sat_id in sorted(base_only):
        base_s = base_dict[sat_id]
        comparison.append({
            'sat_id': sat_id,
            'sat_name': base_s['sat_name'],
            'norad_id': base_s['norad_id'],
            'base_rms': base_s['rms_hz'],
            'rover_rms': np.nan,
            'base_n': base_s['n_measurements'],
            'rover_n': 0,
            'common_view': False,
            'common_view_str': 'No (Rover blind)',
            'quality_ratio': np.nan
        })
    
    # Rover-only satellites
    for sat_id in sorted(rover_only):
        rover_s = rover_dict[sat_id]
        comparison.append({
            'sat_id': sat_id,
            'sat_name': rover_s['sat_name'],
            'norad_id': rover_s['norad_id'],
            'base_rms': np.nan,
            'rover_rms': rover_s['rms_hz'],
            'base_n': 0,
            'rover_n': rover_s['n_measurements'],
            'common_view': False,
            'common_view_str': 'No (Base blind)',
            'quality_ratio': np.nan
        })
    
    # Print summary
    print("\n" + "="*80)
    print("SATELLITE QUALITY COMPARISON")
    print("="*80)
    print(f"{'Satellite':<20} {'Base RMS':<12} {'Rover RMS':<12} {'Quality Ratio':<15} {'Common View':<15}")
    print("-"*80)
    
    for c in comparison:
        base_str = f"{c['base_rms']:.1f} Hz" if not np.isnan(c['base_rms']) else "N/A"
        rover_str = f"{c['rover_rms']:.1f} Hz" if not np.isnan(c['rover_rms']) else "N/A"
        ratio_str = f"{c['quality_ratio']:.1f}x" if not np.isnan(c['quality_ratio']) and np.isfinite(c['quality_ratio']) else "-"
        
        print(f"{c['sat_name']:<20} {base_str:<12} {rover_str:<12} {ratio_str:<15} {c['common_view_str']:<15}")
    
    # Overall statistics
    base_avg = np.nanmean([c['base_rms'] for c in comparison])
    rover_avg = np.nanmean([c['rover_rms'] for c in comparison])
    common_count = sum([c['common_view'] for c in comparison])
    
    print("-"*80)
    print(f"{'Average':<20} {base_avg:.1f} Hz{'':<6} {rover_avg:.1f} Hz{'':<6} {rover_avg/base_avg:.1f}x{'':<10} {common_count}/{len(comparison)}")
    print("="*80)
    
    return comparison


def format_for_plotting(comparison):
    """
    Convert comparison data to format expected by plot_differential_simple.py
    """
    
    print("\n" + "="*80)
    print("PLOTTING DATA (Copy to plot_differential_simple.py)")
    print("="*80)
    
    print("\n# Satellite data for plotting:")
    print(f"satellites = {[c['sat_name'].replace('Iridium', 'Iridium\\n') for c in comparison]}")
    print(f"base_rms = {[c['base_rms'] if not np.isnan(c['base_rms']) else 0 for c in comparison]}")
    print(f"rover_rms = {[c['rover_rms'] for c in comparison]}")
    print(f"common_view = {[c['common_view'] for c in comparison]}")
    print(f"base_n = {[c['base_n'] for c in comparison]}")
    print(f"rover_n = {[c['rover_n'] for c in comparison]}")
    
    print("\n" + "="*80 + "\n")


# Example usage in differential_solve.py:
"""
Add this after loading base_data and rover_data:

from extract_sat_stats import compare_base_rover_quality, format_for_plotting

# Analyze satellite quality
comparison = compare_base_rover_quality(base_data, rover_data, satellites, params)

# Format for plotting
format_for_plotting(comparison)

# Use this data in your plots
sat_quality_data = []
for c in comparison:
    sat_quality_data.append({
        'sat_name': c['sat_name'],
        'norad_id': c['norad_id'],
        'base_rms': c['base_rms'],
        'rover_rms': c['rover_rms'],
        'common_view': c['common_view'],
        'base_n': c['base_n'],
        'rover_n': c['rover_n']
    })
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nThis module provides helper functions to extract satellite statistics.")
    print("Import it in differential_solve.py to analyze your data.")
    print("\nExample usage:")
    print("  from extract_sat_stats import compare_base_rover_quality")
    print("  comparison = compare_base_rover_quality(base_data, rover_data, satellites, params)")



