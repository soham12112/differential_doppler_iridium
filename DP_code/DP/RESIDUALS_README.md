# Residual Analysis for Ghost Satellite Detection

## Overview

This feature helps identify "ghost satellites" (satellites with incorrect IDs) by analyzing the residuals between measured and predicted Doppler frequencies.

## What Are Residuals?

Residuals = Measured Frequency - Predicted Frequency

For each satellite, after the position solver estimates the user's position, we compare:
- The actual measured Doppler shift from the satellite
- The predicted Doppler shift based on the estimated position

## How to Use

### 1. Enable Residual Plotting

In `src/config/setup.py`, set:

```python
class DEBUG:
    plot_residuals = True  # Already enabled by default
```

### 2. Run Your Position Estimation

Run your normal position finding scripts (e.g., `find_position_offline.py`). The residual plot will be automatically generated after the solver completes.

### 3. Analyze the Output

The system will generate:
- **A plot file**: `residuals_per_satellite.png` (saved via `get_fig_filename()`)
- **Console output**: Statistical analysis for each satellite

## Interpreting the Results

### The Plot

The output contains two subplots:

#### Top Plot: Residuals vs Time
- **X-axis**: Unix timestamp
- **Y-axis**: Residuals in Hz (Measured - Predicted)
- Each satellite is plotted in a different color
- A horizontal line at y=0 represents perfect fit

#### Bottom Plot: Residual Distribution
- Histogram showing the distribution of residuals for each satellite

### Console Output

Example output:
```
================================================================================
RESIDUAL ANALYSIS - Ghost Satellite Detection
================================================================================
Satellite                   Mean [Hz]      Std [Hz]      RMS [Hz]      Max [Hz]   N Points
--------------------------------------------------------------------------------
IRIDIUM 120 (24869)           2500.5        1800.3        3100.2        5200.0        150  *** GHOST? ***
IRIDIUM 98 (24795)             120.3          85.2         147.5         280.0        200
IRIDIUM 102 (24841)            -15.2          92.1          93.3         250.0        180
================================================================================
```

### What to Look For

#### ✅ Good Fit (Correct Satellite ID)
- **RMS residuals**: < 500 Hz (ideally < 200 Hz)
- **Pattern**: Random scatter around 0 Hz
- **Mean**: Close to 0 Hz
- **No systematic trends**: Points should look like random noise

#### ❌ Bad Fit (Ghost Satellite - Wrong ID)
- **RMS residuals**: > 1000 Hz (often > 5000 Hz)
- **Pattern**: Clear systematic bias, linear trends, or consistent offset
- **Mean**: Large positive or negative value
- **The script flags these with**: `*** GHOST? ***`

### The 2000 km Clue

When a satellite has the wrong ID:
- The predicted Doppler is calculated for the wrong satellite's position
- This satellite might be ~2000 km away from the actual satellite
- This causes massive residuals (often 10-20 kHz or more)
- The error is systematic (not random) because it's consistently predicting the wrong satellite

## Example Scenario

Suppose your output shows:

```
Satellite                   Mean [Hz]      Std [Hz]      RMS [Hz]      Max [Hz]   N Points
IRIDIUM 120 (24869)          18000.0        2300.0       18145.6       22000.0        150  *** GHOST? ***
IRIDIUM 98 (24795)             -35.2          98.1         104.3         280.0        200
IRIDIUM 102 (24841)             42.1          75.8          87.7         190.0        180
```

**Interpretation**: 
- IRIDIUM 120 is likely a ghost satellite (wrong ID)
- Remove it from your satellite list
- Re-run the position solver
- The position estimate should improve significantly

## Taking Action

### If You Find a Ghost Satellite

1. **Note the satellite ID**: e.g., 24869
2. **Remove it from your data**: Edit your data processing to exclude this ID
3. **Re-run the solver**: Position accuracy should improve
4. **Investigate**: Try to identify the correct satellite ID by:
   - Checking which satellites were actually visible at that time
   - Looking at TLE data
   - Comparing against known good passes

### Example Code to Remove a Satellite

In your data processing:

```python
# Filter out ghost satellite
ghost_sat_id = 24869
nav_data_filtered = [d for d in nav_data if d[4] != ghost_sat_id]  # Adjust index as needed
```

## Technical Details

### Calculation Method

1. **Extract data per satellite**: Group measurements by satellite ID
2. **Calculate predicted Doppler**: 
   - Use final estimated position (lat, lon, alt)
   - Calculate range rate between user and satellite
   - Apply Doppler formula: f_d = -v_rel * f_b / c
   - Add offset and drift corrections
3. **Compute residuals**: Measured - Predicted for each timestamp
4. **Calculate statistics**: RMS, mean, std, max for each satellite

### Why 1 kHz Threshold?

The script flags satellites with RMS > 1000 Hz as potential ghosts because:
- Typical measurement noise: ~50-200 Hz
- Modeling errors: ~100-500 Hz
- Ghost satellites: Often > 5000 Hz (sometimes > 20 kHz)
- 1 kHz provides a safe margin to catch ghosts while avoiding false positives

## Troubleshooting

### All Satellites Show Large Residuals

If all satellites have large residuals (> 1 kHz):
- Your position estimate might be poor
- Check if the solver converged properly
- Verify your TLE data is current
- Check start time accuracy

### No Plot Generated

Ensure:
- `DEBUG.plot_residuals = True` in `setup.py`
- The solver completed successfully
- You have matplotlib and required dependencies installed

### Import Errors

If you see import errors, ensure you have:
```bash
pip install numpy matplotlib astropy
```

## Files Modified

This feature was added to:
1. `src/utils/plots.py`: New function `plot_residuals_per_satellite()`
2. `src/navigation/curve_fit_method.py`: Calls the plotting function in `solve()`
3. `src/config/setup.py`: Added `plot_residuals` flag to DEBUG class

## Further Reading

- Ghost satellites occur due to ID mismatches in the Iridium decoding process
- The ~2000 km offset is related to the typical spacing between satellites in the Iridium constellation
- Residual analysis is a standard technique in navigation and geodesy for identifying outliers




