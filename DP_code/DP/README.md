# Differential Doppler Positioning

## Overview

Differential Doppler positioning uses a **base station** at a known location to improve the accuracy of a **rover** at an unknown location. This technique corrects for systematic errors in the satellite TLE (Two-Line Element) predictions, which are identical for both receivers tracking the same satellites.

## How It Works

### The Problem
TLE-based navigation typically has ~5km errors because satellite orbit predictions are imprecise. However, when two receivers track the same satellites, they experience **identical TLE errors**.

### The Solution
1. **Base Station (Office):** We know the exact location. We calculate what the Doppler *should be* versus what we *measured*. The difference is the **Error Correction** (TLE Error + Atmospheric Delay + Base Clock).

2. **Rover (Home):** We subtract the **Error Correction** from the rover measurements. This removes the TLE error, leaving us with a much cleaner signal for positioning.

### Key Requirements
- Both receivers must track **common satellites** during **overlapping time periods**
- Base station location must be **precisely known**
- Data must be time-synchronized (same reference frame)

## Implementation

### Files Created
- `differential_solve.py` - Main differential positioning script
- `run_differential.sh` - Convenient execution wrapper
- `DIFFERENTIAL_README.md` - This documentation

### Data Configuration

The script is pre-configured to use:
- **Base Station:** Office data (`b200_30_office`)
  - Location: FEL (37.4190082°N, 122.0961837°W)
  - Known precise location
  
- **Rover:** Home data (`ant_30_home`)
  - Location: HOME (37.3619787°N, 122.0281274°W)
  - ~5km baseline from base station

### Running the Script

#### Option 1: Using the wrapper script
```bash
./run_differential.sh
```

#### Option 2: Direct execution
```bash
python3 differential_solve.py
```

#### Option 3: Using the existing run.sh
```bash
./run.sh differential_solve.py
```

## Expected Output

The script will produce output in several stages:

### 1. Data Loading
```
DIFFERENTIAL DOPPLER POSITIONING
================================================================================
Loading data...
  Base:  /path/to/b200_30_office/saved_nav_data.pickle
  Rover: /path/to/ant_30_home/saved_nav_data.pickle
```

### 2. Base Station Analysis
```
[Base Station] Calculating Error Corrections (Residuals)...
================================================================================
SOLVING POSITION
...
Base Station Solution:
  Clock offset:    XXXX Hz
  Clock drift:     X.XXX Hz/s
  
Calculating corrections for N satellites...
  Sat XXX (Iridium XXX):    RMS=XX.X Hz, N=XXXX points
```

### 3. Differential Correction
```
[Rover] Applying Differential Corrections...
================================================================================
Overlapping satellites: N
  ✓ Sat XXX (Iridium XXX): Corrected XXXX points
```

### 4. Final Results
```
DIFFERENTIAL POSITIONING RESULT:
================================================================================
  Distance error:    XXXX m
  Latitude:       XX.XXXXXX° (True: XX.XXXXXX°)
  Longitude:      XX.XXXXXX° (True: XX.XXXXXX°)
  
IMPROVEMENT:
================================================================================
  Absolute:  XXXX m
  Relative:   XX.X%
  Final error: XXXX m (was XXXX m)
```

## Expected Performance

Based on your data:

| Metric | Before (Standard) | After (Differential) | Expected Improvement |
|--------|------------------|---------------------|---------------------|
| Position Error | ~5,000m | ~500-1,000m | 80-90% |
| RMS Residuals | ~80-100 Hz | ~10-20 Hz | 75-85% |

### Common Satellites
Your datasets share these satellites (excellent overlap):
- Iridium 166
- Iridium 154
- Iridium 165

These satellites provide the differential corrections.

## Customization

### Using Different Datasets

Edit the configuration section in `differential_solve.py`:

```python
# --- CONFIGURATION ---
BASE_EXPERIMENT = "your_base_experiment"     # Known location
ROVER_EXPERIMENT = "your_rover_experiment"   # Unknown location

# Update locations in src/config/locations.py if needed
BASE_LAT, BASE_LON, BASE_ALT = LOCATIONS["YOUR_BASE"]
ROVER_LAT_TRUE, ROVER_LON_TRUE, ROVER_ALT = LOCATIONS["YOUR_ROVER"]
```

### Adjusting Parameters

```python
GHOST_SATELLITES = [124]  # Add problematic satellite IDs
RMS_THRESHOLD = 2000      # Increase for noisier data
```

## Troubleshooting

### Error: "No overlapping data found"
**Cause:** Base and rover don't share common satellites or time periods.

**Solution:**
- Check that both datasets were collected at similar times
- Verify satellite IDs are present in both datasets
- Increase the time window or collect new data

### Error: "Could not find data file"
**Cause:** Pickle files don't exist at the specified paths.

**Solution:**
- Run `process_offline_data.py` first to generate pickle files
- Update `BASE_FILE` and `ROVER_FILE` paths in the script

### Poor Results (< 50% improvement)
**Possible causes:**
- Insufficient satellite overlap
- Large baseline between base and rover (> 50km)
- Time synchronization issues
- Atmospheric variations between locations

**Solutions:**
- Filter satellites more strictly (lower RMS_THRESHOLD)
- Verify time stamps are in the same reference frame
- Collect data closer to the base station

## Technical Details

### Algorithm Steps

1. **Load Data:** Both base and rover navigation data (time, frequency, satellite positions)

2. **Base Station Solution:**
   - Solve for base station clock offset and drift
   - Calculate theoretical Doppler for known location
   - Compute residuals: `error = measured - theoretical`
   - Create time-interpolated error correction functions

3. **Apply Corrections:**
   - For each rover measurement, look up base correction at same time
   - Subtract correction: `corrected = measured - correction`
   - This removes TLE error and common-mode atmospheric delays

4. **Rover Solution:**
   - Solve for rover position using corrected measurements
   - Rover has its own clock offset/drift (different receiver)
   - Final solution should be much more accurate

### Mathematical Model

**Standard Doppler equation:**
```
f_measured = f_doppler + f_clock + f_error
```

Where:
- `f_doppler` = true geometric Doppler shift
- `f_clock` = receiver clock bias + drift
- `f_error` = TLE error + atmospheric error

**At base (known location):**
```
f_error_base = f_measured_base - f_doppler_theoretical - f_clock_base
```

**At rover (corrected):**
```
f_corrected_rover = f_measured_rover - f_error_base
                  = f_doppler_rover + f_clock_rover
```

Now the measurement only contains geometric information and the rover's clock!

## Performance Notes

- Processing time: ~30-60 seconds (depends on data size)
- Memory usage: ~100MB (typical)
- Requires `scipy` for interpolation functions

## References

- Differential GPS (DGPS) uses the same principle with pseudorange
- This implementation uses Doppler shift instead of pseudorange
- Effective baseline: < 50km for best results

## Support

For issues or questions:
1. Check that `scipy` is installed: `pip install scipy`
2. Verify pickle files exist and contain valid data
3. Check satellite overlap with standard position solve first
4. Review residual plots to identify problematic satellites

## Future Enhancements

Possible improvements:
- [ ] Real-time differential correction (streaming mode)
- [ ] Network RTK (multiple base stations)
- [ ] Atmospheric model (remove tropospheric delay)
- [ ] Automatic satellite selection (quality metrics)
- [ ] Time-varying error interpolation (spline instead of linear)


