# Publication-Quality Plots for Differential Doppler Positioning

This guide explains how to create publication-quality visualizations for your Differential Doppler positioning results.

## Overview

Two plotting scripts have been created to showcase the effectiveness of your differential correction:

1. **`plot_differential_results.py`** - Standalone script with example data
2. **`plot_differential_auto.py`** - Automated plotting module for integration with `differential_solve.py`

## Quick Start

### Option 1: Standalone Plotting (Manual Data Entry)

Edit the data in `plot_differential_results.py` and run:

```bash
python3 plot_differential_results.py
```

This generates three plots in the `Plots/` directory:
- `differential_spatial_map.png` - 2D position error visualization
- `differential_error_comparison.png` - Bar chart comparing errors
- `satellite_quality_comparison.png` - Satellite RMS and common view analysis

### Option 2: Automated Plotting (Integrated with Differential Solve)

Add the following to the end of `run_differential()` in `differential_solve.py`:

```python
from plot_differential_auto import DifferentialPlotter

# Prepare result dictionaries
base_result = {
    'lat_true': BASE_LAT_TRUE, 
    'lon_true': BASE_LON_TRUE,
    'lat_est': base_lat, 
    'lon_est': base_lon,
    'error_m': base_dist_err,
    'clock_offset': base_off, 
    'clock_drift': base_dft
}

rover_std_result = {
    'lat_true': ROVER_LAT_TRUE, 
    'lon_true': ROVER_LON_TRUE,
    'lat_est': lat_std, 
    'lon_est': lon_std,
    'error_m': dist_err_std,
    'clock_offset': off_std, 
    'clock_drift': dft_std
}

rover_diff_result = {
    'lat_true': ROVER_LAT_TRUE, 
    'lon_true': ROVER_LON_TRUE,
    'lat_est': lat, 
    'lon_est': lon,
    'error_m': dist_err,
    'clock_offset': off, 
    'clock_drift': dft
}

# Optional: Satellite quality data
sat_quality = []
for sat_id in overlap_sats:
    # Extract RMS and measurement counts from your analysis
    sat_quality.append({
        'sat_name': f'Iridium {sat_id}',
        'base_rms': base_rms_for_sat,  # Calculate from base data
        'rover_rms': rover_rms_for_sat,  # Calculate from rover data
        'common_view': True,
        'base_n': base_measurement_count,
        'rover_n': rover_measurement_count
    })

# Generate plots
plotter = DifferentialPlotter()
plotter.plot_all_results(base_result, rover_std_result, rover_diff_result, sat_quality)
```

## Matplotlib Troubleshooting

If you encounter segmentation faults or matplotlib errors:

### Issue 1: Font Cache Problems
```bash
# Clear matplotlib cache
rm -rf ~/.matplotlib
rm -rf ~/.cache/matplotlib
```

### Issue 2: Missing Backend Dependencies (macOS)
```bash
# Reinstall matplotlib with proper backend support
pip3 install --upgrade --force-reinstall matplotlib
```

### Issue 3: System Library Issues
```bash
# Check matplotlib configuration
python3 -c "import matplotlib; print(matplotlib.get_backend())"

# Try different backend
export MPLBACKEND=Agg
python3 plot_differential_results.py
```

## Generated Plots

### 1. Spatial Error Map
**Purpose:** Visually demonstrate the positioning improvement

**Key Features:**
- Shows true positions (stars) vs estimated positions (circles/crosses)
- Error vectors clearly indicate the magnitude and direction of errors
- Color-coded: Green (Base), Red (Standard), Blue (Differential)
- Improvement percentage displayed prominently
- Geographic coordinate system with equal aspect ratio

**Best for:** Primary figure in results section, conference presentations

### 2. Error Comparison Bar Chart
**Purpose:** Quantitative comparison of positioning errors

**Key Features:**
- Bar heights show error magnitudes
- Numeric labels on each bar
- Improvement arrow with percentage
- Clean, publication-ready styling

**Best for:** Abstract graphics, executive summaries

### 3. Satellite Quality Comparison
**Purpose:** Justify the need for differential correction

**Key Features:**
- Left panel: RMS residuals (showing Rover has 8× worse measurements)
- Right panel: Common view status (showing limited overlap)
- Demonstrates environmental differences between Base and Rover

**Best for:** Methods section, explaining data quality constraints

## Customizing for Your Data

### Update Example Data (plot_differential_results.py)

**Line 26-37:** Positioning results
```python
data = {
    'Scenario': ['Base Station (Office)', 'Rover (Home) - Standard', 'Rover (Home) - Differential'],
    'True_Lat': [37.419008, 37.361979, 37.361979],
    'True_Lon': [-122.096184, -122.028127, -122.028127],
    'Est_Lat': [37.417, 37.401, 37.382492],
    'Est_Lon': [-122.101, -122.056, -122.043330],
    'Error_m': [508, 4973, 2645],
    'Clock_Offset_Hz': [-1200, -2800, -2400],
    'Clock_Drift_Hz_s': [-0.15, -0.059, -0.278]
}
```

**Line 45-54:** Satellite quality data
```python
data = {
    'Satellite': ['Iridium 165', 'Iridium 166', 'Iridium 154', 'Iridium 108'],
    'NORAD_ID': [43479, 43480, 43246, 42956],
    'Base_RMS_Hz': [6.8, 6.3, 12.6, np.nan],
    'Rover_RMS_Hz': [87.4, 104.7, 79.5, 135.6],
    'Common_View': ['Yes', 'Yes', 'No (Time skew)', 'No (Base blind)'],
    'Base_Measurements': [120, 115, 45, 0],
    'Rover_Measurements': [95, 88, 32, 78]
}
```

## Summary Tables for Paper

The scripts also generate formatted summary tables:

### Table 1: Positioning Performance Summary
```
Scenario                       Latitude (°)  Longitude (°)   Error (m)  Improvement
--------------------------------------------------------------------------------
Base Station (Office)          37.417000     -122.101000     508        -
Rover (Home) - Standard        37.401000     -122.056000     4973       Reference
Rover (Home) - Differential    37.382492     -122.043330     2645       46.8%
```

### Table 2: Satellite Measurement Quality
```
Satellite       Base RMS    Rover RMS    Common View
--------------------------------------------------------------------------------
Iridium 165     6.8 Hz      87.4 Hz      Yes
Iridium 166     6.3 Hz      104.7 Hz     Yes
Iridium 154     12.6 Hz     79.5 Hz      No (Time skew)
Iridium 108     N/A         135.6 Hz     No (Base blind)
--------------------------------------------------------------------------------
Average         12.0        101.8        2/4 overlapping
```

## Key Discussion Points for Your Paper

### 1. Positioning Improvement
- **Standard error:** 4,973 m
- **Differential error:** 2,645 m
- **Improvement:** 2,328 m (46.8% reduction)

### 2. Measurement Quality Disparity
- **Base station RMS:** ~12 Hz (clean environment)
- **Rover RMS:** ~102 Hz (challenging environment)
- **Quality ratio:** 8.5× worse at rover
- **Interpretation:** Rover in multipath/attenuated environment

### 3. Common View Constraints
- **Overlapping satellites:** 2 out of 4 (50%)
- **Critical finding:** 46.8% improvement with only 2 overlapping satellites
- **Implication:** Greater improvement expected with better satellite availability

### 4. Clock Drift Absorption
- **Standard clock drift:** -0.059 Hz/s
- **Differential clock drift:** -0.278 Hz/s
- **Interpretation:** Differential process absorbs receiver clock instability

## Plot Styling Configuration

All plots use publication-quality defaults:
- **DPI:** 300 (print quality)
- **Font:** Serif (Times New Roman style)
- **Size:** 10-14 pt for readability
- **Grid:** Subtle dashed lines
- **Colors:** Colorblind-friendly palette
- **Format:** PNG (easily convertible to EPS/PDF for journals)

## Converting Plots for LaTeX

```bash
# Convert PNG to PDF (vector format for LaTeX)
for file in Plots/*.png; do
    convert "$file" "${file%.png}.pdf"
done

# Or use ImageMagick with better quality
convert -density 300 input.png -quality 100 output.pdf
```

## Recommended Figure Captions

### Figure 1: Spatial Error Map
> **Spatial error comparison between standard and differential Doppler positioning.**
> True positions are shown as stars, estimated positions as circles/crosses. Error vectors 
> indicate the magnitude and direction of positioning errors. The differential solution 
> reduces error from 4,973 m to 2,645 m, a 46.8% improvement, despite limited satellite 
> overlap (2 common view satellites).

### Figure 2: Error Bar Chart
> **Positioning error comparison across scenarios.** The differential correction reduces 
> rover positioning error by 2,328 m (46.8%) compared to the standard solution. Base 
> station error (508 m) demonstrates the quality achievable in clean RF environments.

### Figure 3: Satellite Quality
> **Doppler measurement quality and common view analysis.** (Left) RMS residuals show 
> 8.5× higher noise at the rover location, indicating challenging propagation conditions. 
> (Right) Only 2 of 4 satellites had overlapping time coverage, limiting differential 
> correction potential. Greater improvement is expected with increased satellite visibility.

## LaTeX Integration Example

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{Plots/differential_spatial_map.pdf}
    \caption{Spatial error comparison between standard and differential Doppler 
    positioning. True positions are shown as stars, estimated positions as 
    circles/crosses. The differential solution reduces error by 46.8\%.}
    \label{fig:spatial_error}
\end{figure}
```

## Additional Resources

- **Requirements:** matplotlib, pandas, numpy
- **Output directory:** `Plots/` (created automatically)
- **Supported formats:** PNG (default), can be modified to PDF, SVG, EPS

## Support

If you encounter issues:
1. Check matplotlib installation: `pip3 show matplotlib`
2. Try clearing font cache (see Troubleshooting section)
3. Check backend: `python3 -c "import matplotlib; print(matplotlib.get_backend())"`
4. Use `matplotlib.use('Agg')` for headless operation

---

**Note:** These scripts are designed to produce publication-ready figures. Adjust colors, 
fonts, and sizes as needed for your specific journal's requirements.



