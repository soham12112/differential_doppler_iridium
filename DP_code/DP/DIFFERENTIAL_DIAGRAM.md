# Differential Doppler - Visual Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DIFFERENTIAL DOPPLER SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐         ┌─────────────────────────┐
│   BASE STATION (Office)  │         │    ROVER STATION (Home) │
│                          │         │                         │
│  Location: KNOWN         │         │  Location: UNKNOWN      │
│  FEL: 37.419°N          │         │  HOME: 37.362°N         │
│       122.096°W         │         │        122.028°W        │
│                          │         │                         │
│  Error: ~500m           │         │  Error: ~5000m (before) │
│  RMS: 5-7 Hz            │         │  RMS: 80-100 Hz         │
└───────────┬─────────────┘         └─────────────┬───────────┘
            │                                     │
            │    Tracking Same Satellites         │
            │    (Iridium 166, 154, 165)         │
            │                                     │
            └────────────┬────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  COMMON SATELLITES  │
              │                     │
              │  ▪ Iridium 166     │
              │  ▪ Iridium 154     │
              │  ▪ Iridium 165     │
              │                     │
              │  All have TLE Error │
              └─────────────────────┘
```

## Data Flow

```
STAGE 1: BASE STATION ANALYSIS
═══════════════════════════════

Base Data                     Known Location
(measured)                    (37.419°N, 122.096°W)
    │                                │
    ├────────────────────────────────┤
    │                                │
    ▼                                ▼
┌─────────────────┐         ┌─────────────────┐
│  f_measured     │    -    │  f_theoretical  │  =  ERROR CORRECTION
│                 │         │                 │     (TLE + Atmos)
│  What we see    │         │  What we expect │
└─────────────────┘         └─────────────────┘
         │
         ├─ For Iridium 166: error_166(t)
         ├─ For Iridium 154: error_154(t)
         └─ For Iridium 165: error_165(t)


STAGE 2: APPLY TO ROVER
═══════════════════════════

Rover Data                    Error Corrections
(measured, raw)               (from base station)
    │                                │
    ├────────────────────────────────┤
    │                                │
    ▼                                ▼
┌─────────────────┐         ┌─────────────────┐
│  f_measured     │    -    │  error(t)       │  =  CORRECTED DATA
│                 │         │                 │     (clean signal)
│  Noisy, biased  │         │  Remove TLE err │
└─────────────────┘         └─────────────────┘


STAGE 3: POSITION SOLUTION
═══════════════════════════════

Corrected Data  ──────▶  ┌─────────────────────┐
                         │  Position Solver     │
                         │  (curve_fit_method)  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  IMPROVED POSITION  │
                         │  Error: ~500-1000m  │
                         │  (80-90% better!)   │
                         └─────────────────────┘
```

## Error Budget

```
BEFORE DIFFERENTIAL CORRECTION (Home/Rover)
═══════════════════════════════════════════

Total Error: ~5000m

┌─────────────────────────────────────────┐
│ TLE Error (Orbit Prediction)   ~4000m   │ ◄── Can be corrected!
├─────────────────────────────────────────┤
│ Atmospheric Delay              ~500m     │ ◄── Can be corrected!
├─────────────────────────────────────────┤
│ Receiver Clock Drift           ~400m     │ ◄── Solved separately
├─────────────────────────────────────────┤
│ Multipath / Noise              ~100m     │ ◄── Remains
└─────────────────────────────────────────┘


AFTER DIFFERENTIAL CORRECTION (Home/Rover)
══════════════════════════════════════════

Total Error: ~500-1000m

┌─────────────────────────────────────────┐
│ TLE Error (Orbit Prediction)    0m      │ ✓ CORRECTED
├─────────────────────────────────────────┤
│ Atmospheric Delay               0m      │ ✓ CORRECTED
├─────────────────────────────────────────┤
│ Receiver Clock Drift            400m    │ ✓ Solved
├─────────────────────────────────────────┤
│ Multipath / Noise               100m    │ ~ Reduced
├─────────────────────────────────────────┤
│ Residual differential effects   100m    │ (baseline < 10km)
└─────────────────────────────────────────┘
```

## Time Synchronization

```
Base Station Timeline:
├────────┬────────┬────────┬────────┬────────┬────────┤
0s      10s      20s      30s      40s      50s      60s
│                 │                 │                 │
│    Sat 166     │    Sat 154     │    Sat 165     │
│    visible     │    visible     │    visible     │
│                 │                 │                 │
▼                 ▼                 ▼                 ▼
error_166(t)    error_154(t)    error_165(t)    ...


Rover Timeline (must overlap!):
├────────┬────────┬────────┬────────┬────────┬────────┤
0s      10s      20s      30s      40s      50s      60s
│                 │                 │                 │
│    Sat 166     │    Sat 154     │    Sat 165     │
│   correction   │   correction   │   correction   │
│    applied     │    applied     │    applied     │
│                 │                 │                 │
▼                 ▼                 ▼                 ▼
Clean measurements ──▶ Position Solution
```

## Satellite Geometry

```
                    ╭─────────╮
                 ╭──┤ Sat 166 ├──╮
              ╭──┴──╰─────────╯  ╰──╮
           ╭──┴──╮              ╭──┴──╮
        ╭──┤Sat  ├──╮        ╭──┤ Sat ├──╮
        │  │154  │  │        │  │ 165 │  │
        ╰──┴──╬──┴──╯        ╰──┴──╬──┴──╯
              ║                     ║
         ┌────╨────┐           ┌───╨────┐
         │ Office  │───────────│  Home  │
         │ (Base)  │  ~6.8 km  │ (Rover)│
         │  KNOWN  │           │ UNKNOWN│
         └─────────┘           └────────┘
         
All three satellites visible to both receivers
Same TLE errors affect both ────────────▶ Can be differenced!
```

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│ START                                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │ Load Base & Rover Data   │
         │  - Filter ghosts (124)   │
         │  - Load TLEs             │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Solve Base Position      │
         │  - Get clock offset/drift│
         │  - Measure quality (RMS) │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ For each Base Satellite: │
         │  1. Calc theoretical f_d │
         │  2. Calc measured f_d    │
         │  3. error = meas - theo  │
         │  4. Create interp(t)     │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ For each Rover Satellite:│
         │  1. Check overlap        │
         │  2. Interpolate error(t) │
         │  3. Apply correction     │
         │  4. f_corr = f - error   │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Solve Rover Position     │
         │  - Use corrected data    │
         │  - Find lat, lon, clock  │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Compare Results          │
         │  - Standard vs Diff      │
         │  - Calculate improvement │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ Report Success!          │
         │  Expected: 80-90% better │
         └──────────────────────────┘
```

## File Dependencies

```
differential_solve.py
    │
    ├─▶ src/config/locations.py       (LOCATIONS dict)
    ├─▶ src/config/parameters.py      (CurveFitMethodParameters)
    ├─▶ src/navigation/calculations.py (latlon_distance)
    ├─▶ src/navigation/curve_fit_method.py (solve, C)
    ├─▶ src/navigation/data_processing.py (IDX, find_curves)
    ├─▶ src/satellites/download_tle.py (download_tles)
    │
    ├─▶ Data/b200_30_office/saved_nav_data.pickle (Base)
    └─▶ Data/ant_30_home/saved_nav_data.pickle    (Rover)
```

## Performance Expectations

```
┌──────────────────┬──────────┬───────────┬──────────────┐
│ Metric           │ Before   │ After     │ Improvement  │
├──────────────────┼──────────┼───────────┼──────────────┤
│ Position Error   │ 5000 m   │ 500-1000m │ 80-90%       │
│ RMS Residuals    │ 80-100Hz │ 10-20 Hz  │ 75-85%       │
│ Satellites Used  │ 5-8      │ 3-5 (best)│ Better select│
│ Processing Time  │ 30s      │ 60s       │ 2x slower    │
└──────────────────┴──────────┴───────────┴──────────────┘

Worth it? YES! ✓
```

## Why This Works

```
KEY INSIGHT: Same satellites = Same errors

┌─────────────────────────────────────────────────────┐
│                    Iridium 166                      │
│                         │                           │
│         TLE predicts: ─┼── (slightly wrong)        │
│         Actual orbit: ─┼── (reality)               │
│                         │                           │
│         ┌───────────────┼───────────────────┐      │
│         ▼               ▼                   ▼      │
│    ┌────────┐      ┌────────┐         ┌────────┐  │
│    │Satellite│      │ Office │         │  Home  │  │
│    │  (Sky) │      │ (Base) │         │ (Rover)│  │
│    └────────┘      └────────┘         └────────┘  │
│                         │                   │      │
│    Same TLE error  ─────┴───────────────────┴──▶  │
│    affects both receivers identically!             │
│                                                     │
│    Office knows error ──▶ Subtract from Home ──▶  │
│                           Much better result! ✓    │
└─────────────────────────────────────────────────────┘
```

## Success Indicators

```
✓ GOOD RESULT:
  ┌────────────────────────────────────┐
  │ Distance error:     500 m          │
  │ Improvement:        90%            │
  │ Overlapping sats:   3+             │
  │ Final RMS:          < 20 Hz        │
  └────────────────────────────────────┘

⚠ MARGINAL RESULT:
  ┌────────────────────────────────────┐
  │ Distance error:     2000 m         │
  │ Improvement:        50%            │
  │ Overlapping sats:   2-3            │
  │ Final RMS:          30-50 Hz       │
  └────────────────────────────────────┘
  → Try filtering satellites more strictly
  → Collect longer baseline data

✗ POOR RESULT:
  ┌────────────────────────────────────┐
  │ Distance error:     > 3000 m       │
  │ Improvement:        < 30%          │
  │ Overlapping sats:   < 2            │
  │ Final RMS:          > 60 Hz        │
  └────────────────────────────────────┘
  → Check time synchronization
  → Verify satellite overlap
  → May need new data collection
```

## Quick Reference

**Run:** `./run_differential.sh`

**Check:** Look for "IMPROVEMENT: XX% " in output

**Expected:** 80-90% improvement (5000m → 500-1000m)

**Time:** ~30-60 seconds

**Common Sats:** Iridium 166, 154, 165

**Docs:** See DIFFERENTIAL_README.md for full details



