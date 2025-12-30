# Differential Doppler - Quick Start

## What is this?

Use your **Office data** (known location) to improve your **Home data** (unknown location) by correcting satellite orbit errors.

## One Command to Run

```bash
./run_differential.sh
```

Or:

```bash
python3 differential_solve.py
```

## What to Expect

**Current Results (without differential):**
- Office: ~500m error
- Home: ~5,000m error

**Expected Results (with differential):**
- Home: ~500-1,000m error (80-90% improvement!)

## How it Works

1. **Office (Base):** Calculate satellite orbit errors using known location
2. **Apply to Home (Rover):** Subtract those errors from Home measurements
3. **Result:** Much better position accuracy!

## Common Satellites (Your Data)

Your Office and Home data share these satellites:
- ✓ Iridium 166
- ✓ Iridium 154  
- ✓ Iridium 165

This is **excellent overlap** for differential correction!

## Customization

To use different datasets, edit `differential_solve.py`:

```python
BASE_EXPERIMENT = "b200_30_office"  # Change this
ROVER_EXPERIMENT = "ant_30_home"    # Change this
```

## Troubleshooting

**"No overlapping data found"**
- Check that both datasets were collected around the same time
- Make sure pickle files exist (run `process_offline_data.py` first)

**Poor improvement (< 50%)**
- Check satellite overlap: need at least 3 common satellites
- Verify time stamps are synchronized
- Increase data collection time

## Files

- `differential_solve.py` - Main script
- `run_differential.sh` - Run wrapper  
- `DIFFERENTIAL_README.md` - Full documentation
- `DIFFERENTIAL_QUICKSTART.md` - This file

## Full Documentation

See `DIFFERENTIAL_README.md` for complete technical details.



