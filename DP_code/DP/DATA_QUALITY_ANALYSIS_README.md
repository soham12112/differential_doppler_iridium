# Data Quality Analysis Tools

This directory contains tools to analyze the quality of your collected Iridium satellite data, including noise statistics, SINR (Signal-to-Interference-plus-Noise Ratio), sample rates, and other important metrics.

## Quick Start

### Analyze Your Data

Run the comprehensive analysis script:

```bash
python analyze_data_quality_simple.py
```

This will analyze the experiment defined by `EXP_NAME` at the top of the script (default: `b200_28th_night`).

### Sample Output

The script provides:

1. **Basic Statistics**
   - Total measurements collected
   - Duration of collection
   - Number of satellites observed
   - Satellite IDs

2. **Hardware Configuration**
   - SDR sample rate: **5,000,000 samples/sec (5 MHz)**
   - Center frequency: 1,626.2708 MHz
   - Device: PlutoSDR (ADALM-PLUTO)

3. **Timing & Frame Rate**
   - Effective frame rate (how many frames/second)
   - Time gaps between frames
   - Detection of large gaps in data

4. **Signal Quality & Noise**
   - Frame counts (IRA, IBC, other)
   - Confidence statistics from decoder
   - **Noise levels** (overall and per-channel)
   - **SINR** (Signal-to-Interference+Noise Ratio) in dB
   - Signal quality distribution

5. **Per-Satellite Statistics**
   - Frames captured per satellite
   - Duration of satellite visibility
   - Frame rate per satellite
   - Doppler shift range

6. **Doppler Statistics**
   - Doppler shift range and mean
   - Doppler rate of change (Hz/s)

## Understanding the Results

### Sample Rate

Your SDR is configured to sample at **5 MHz** (5,000,000 samples per second). This is the hardware sampling rate for the RF signal.

### Effective Frame Rate

The "effective frame rate" is how many Iridium frames you're successfully decoding per second. This will be much lower than the sample rate because:
- Iridium signals are intermittent (not continuous)
- Satellites may not always be in view
- Some frames may fail to decode

Example: 0.03 Hz = ~1 frame every 36 seconds

### SINR Interpretation

**SINR (Signal-to-Interference-plus-Noise Ratio)** indicates signal quality:

| SINR Range | Quality | Navigation Performance |
|------------|---------|----------------------|
| > 20 dB    | Excellent | Very accurate positioning |
| 10-20 dB   | Good     | Good positioning |
| 5-10 dB    | Fair     | Marginal positioning |
| < 5 dB     | Poor     | Unreliable measurements |

**Note:** Low SINR may indicate:
- Weak signal (distant satellite, poor antenna)
- High noise floor (interference, hardware issues)
- Need for signal amplification

### Noise Statistics

Noise values are typically reported in arbitrary units from the decoder. Lower (more negative) values generally indicate cleaner channels.

### Confidence

Decoder confidence (percentage) indicates how certain the decoder is about the frame:
- **> 90%**: Excellent, highly reliable
- **70-90%**: Good
- **50-70%**: Marginal
- **< 50%**: Unreliable (usually filtered out)

## Customization

### Analyze Different Experiments

Edit the top of `analyze_data_quality_simple.py`:

```python
# Configuration
WORKING_DIR = "/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data"
EXP_NAME = "your_experiment_name"  # Change this
```

### Available Experiments

Based on your setup, you have these validation datasets:
- `b200_28th_night`
- `b200_30_office`
- `ant_30_home`
- `val06`, `val07`, `val08`, `val09`, `val10`

## Advanced Analysis Tools

### 1. Residual Analysis (Existing Tool)

Analyze residuals for all satellites to identify "ghost" satellites:

```bash
cd /Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/DP_code/DP
python -m src.app.development.analyze_all_satellites
```

This shows which satellites have good vs. poor Doppler fit and helps identify false detections.

### 2. Data Statistics (Existing Tool)

Comprehensive statistics across multiple datasets:

```bash
python -m src.app.validation.data_statistics
```

### 3. Noise Analysis (Existing Tool)

Test how your solution performs with added noise:

```bash
python -m src.app.development.noise_analysis
```

## Sample Output Interpretation

### Example: b200_28th_night

```
Total Measurements:            827
Duration:                      08h 20m 30s
Number of Satellites:          33
Effective Frame Rate:          0.03 Hz

Mean confidence:               90.0%
Overall mean noise:            -42.03
Mean SINR:                     2.78 dB
```

**Analysis:**
- ✓ Good: 827 measurements over 8+ hours
- ✓ Good: 33 satellites observed
- ✓ Good: 90% average confidence
- ⚠ Low: Frame rate is low (sparse data)
- ⚠ Low: SINR is only 2.78 dB (poor signal quality)

**Recommendations:**
1. Signal quality is poor - consider:
   - Better antenna (higher gain)
   - LNA (Low Noise Amplifier)
   - Reduce local RF interference
   - Better antenna placement

2. Frame rate is low but acceptable for navigation
3. Confidence is high, so decoded frames are reliable

## Improving Data Quality

### To Increase Sample Rate

Your SDR is already sampling at 5 MHz. To capture more frames:
- Increase collection duration
- Use better antenna positioning
- Optimize signal thresholds in decoder

### To Improve SINR

1. **Hardware:**
   - Use a higher gain antenna
   - Add a Low Noise Amplifier (LNA)
   - Reduce cable losses (shorter cables)

2. **Environment:**
   - Move away from RF interference sources
   - Better antenna location (clear sky view)
   - Check for local noise sources

3. **Software:**
   - Adjust decoder sensitivity thresholds
   - Filter out low-confidence frames
   - Optimize frequency offset correction

## Files

- `analyze_data_quality_simple.py` - Main analysis script (standalone, no complex imports)
- `src/app/data_analysis/comprehensive_data_stats.py` - Module version (may have import issues)
- `src/app/development/analyze_all_satellites.py` - Per-satellite residual analysis
- `src/app/development/noise_analysis.py` - Synthetic noise testing
- `src/app/validation/data_statistics.py` - Multi-dataset statistics

## Troubleshooting

### FileNotFoundError

Make sure:
1. `WORKING_DIR` points to your Data directory
2. `EXP_NAME` matches an actual experiment folder
3. The folder contains `decoded.txt` and `saved_nav_data.pickle`

### Import Errors

Use `analyze_data_quality_simple.py` instead of the module version - it has no external imports beyond numpy and pickle.

### No IRA Frames

If you see "No IRA frames found", your `decoded.txt` may contain only other frame types. The analysis will still work but SINR estimation will be limited.

## Questions?

Common questions:

**Q: What's a good sample rate?**
A: 5 MHz is excellent for Iridium signals (1616-1626 MHz band).

**Q: Why is my frame rate so low?**
A: Iridium Ring Alert (IRA) frames are transmitted intermittently. 0.01-0.1 Hz is normal.

**Q: How can I improve SINR?**
A: Use a better antenna, add an LNA, reduce interference, or improve antenna positioning.

**Q: What does "Poor (<5 dB)" SINR mean?**
A: Your signal is barely above the noise floor. Positioning will work but with reduced accuracy.

**Q: How many measurements do I need?**
A: For positioning, 20-50 measurements from 3-4 satellites over 5-10 minutes is minimum.

## Summary

You now have comprehensive tools to:
- ✓ Understand your data collection setup (5 MHz sample rate)
- ✓ Analyze signal quality (SINR, noise, confidence)
- ✓ Evaluate per-satellite performance
- ✓ Identify data quality issues
- ✓ Make informed decisions about hardware improvements

For more details on the navigation algorithms, see `DIFFERENTIAL_README.md`.



