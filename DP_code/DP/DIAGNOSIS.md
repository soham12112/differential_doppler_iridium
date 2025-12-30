# 15 km Position Error - Diagnosis and Solutions

## Your Question About Clock Offset and Drift

**YES, clock offset and drift ARE being applied correctly!**

Looking at your code in `curve_fit_method.py` lines 77-106:

```python
# Line 92: Calculate predicted Doppler
f_d = -1 * rel_vel * f_b / C

# Line 95: Calculate drift over time
f_drift = (trial_curve[:, 0] - np.min(trial_curve[:, 0])) * dft

# Line 98: Apply both offset and drift
trial_curve[:, 1] = f_d + off + f_drift

# Line 101: Compare with measured
sum_of_squares = np.sum((measured_curve[:, 1] - trial_curve[:, 1]) ** 2)
```

The clock offset (Hz) and drift (Hz/s) are correctly applied DURING the optimization process, not before.

---

## Understanding Your 15 km Error

### Likely Causes (in order of probability):

1. **START TIME ERROR** (Most likely!)
   - If start time is off by even 1-2 seconds, satellite positions will be wrong
   - This causes systematic position errors
   - Your satellites move ~7.5 km/s, so 2 seconds = 15 km error!

2. **SDR FREQUENCY OFFSET**
   - Systematic frequency calibration error in your SDR
   - Can cause position biases

3. **GHOST SATELLITES**
   - Satellites with incorrect ID mapping
   - Already filtered out 124, but there may be more

4. **TLE INACCURACY**
   - TLEs can have ~1 km position errors
   - But usually not 15 km unless very old

---

## Will a 2nd SDR Receiver Help?

**Short answer: NO, not for TLE errors!**

### Why it WON'T help with TLE inaccuracy:
- TLE errors affect satellite position **prediction**
- Both receivers at the same location would use the same TLEs
- Both would have the same TLE-induced errors
- You'd just get two measurements with the same systematic error

### What WOULD help:
1. **2 receivers at DIFFERENT known locations** (Differential positioning)
   - Errors cancel out in the differential measurements
   - This is how DGPS works
   - But requires known reference station

2. **Better ephemeris data**
   - Use precise ephemeris instead of TLEs
   - Iridium doesn't publish precise ephemeris publicly

3. **Fix your systematic errors first** (SDR offset, start time)
   - These are likely causing your 15 km error, not TLE inaccuracy

---

## Diagnostic Steps - Run These Scripts

### Step 1: Check for Ghost Satellites
```bash
./run.sh src/app/development/analyze_all_satellites.py
```
This identifies satellites with high residuals that should be filtered.

### Step 2: Test for Systematic Frequency Offset
```bash
./run.sh src/app/development/optimize_frequency_offset.py
```
This sweeps through frequency offsets to find if your SDR has a calibration error.

### Step 3: Optimize Start Time (if you have IRA frames)
```bash
./run.sh src/app/development/optimize_start_time.py
```
This uses IRA position broadcast frames to find the correct start time.

---

## Quick Test: Manual Corrections

Try adding these corrections to `find_position_offline_2d.py`:

```python
# After loading data (around line 164), try:

# Test 1: Add time correction
TIME_CORRECTION = 2.0  # seconds - try different values: -5, -2, 0, +2, +5
saved_nav_data[:, IDX.t] += TIME_CORRECTION

# Test 2: Add frequency correction  
FREQ_CORRECTION = 0  # Hz - try: -1000, -500, 0, +500, +1000
saved_nav_data[:, IDX.f] += FREQ_CORRECTION
```

Run your positioning with different values and see which gives the smallest error.

---

## Expected Accuracy with Good Data

From the original DP paper (Voskavoj):
- **2D positioning**: ~500 m - 2 km typical error
- **3D positioning**: ~1-3 km typical error
- **With good TLEs and calibration**: < 1 km is achievable

Your 15 km error suggests a systematic problem, not just TLE inaccuracy.

---

## My Recommendation

1. **First priority**: Run `optimize_frequency_offset.py` to test for systematic offsets
2. **Second priority**: Check if you have accurate start time
3. **Third priority**: Filter more ghost satellites using `analyze_all_satellites.py`
4. **Last resort**: If still bad, check your satellite ID mapping in `iridium_channels.py`

The clock offset parameter you're estimating (-4027 Hz) is very large, which also suggests there may be a systematic frequency or time error in your data.

Let me know what you find!

