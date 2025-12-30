# Final Satellite Matching & Corrections (Updated)

**Date:** 2025-11-29  
**Sim data:** 28 Nov 11:55–18:35 UTC (dense sampling)  
**Measured sats on disk:** sat_2, sat_3, sat_13, sat_18, sat_23, sat_25, sat_46, sat_49, sat_78, sat_90

---

## ✅ Unique Signature Matches (inversion applied)

| Measured | → | Simulated | Dataset | Score |
|----------|---|-----------|---------|-------|
| `sat_2`  | → | `IRIDIUM181` | UE2 | **0.9582** |
| `sat_3`  | → | `IRIDIUM109` | UE1 | **0.9128** |
| `sat_13` | → | `IRIDIUM134` | UE2 | **0.9338** |
| `sat_18` | → | `IRIDIUM160` | UE1 | **0.8664** |
| `sat_23` | → | `IRIDIUM163` | UE1 | **0.6616** |
| `sat_25` | → | `IRIDIUM144` | UE1 | **0.9220** |
| `sat_46` | → | `IRIDIUM104` | UE1 | **0.9481** |
| `sat_49` | → | `IRIDIUM119` | UE1 | **0.8508** |
| `sat_78` | → | `IRIDIUM107` | UE2 | **0.8283** |
| `sat_90` | → | `IRIDIUM137` | UE1 | **0.8891** |

*(All simulated Doppler traces must be multiplied by **-1** before comparison.)*

---

## 📊 Frequency Offsets & Corrections

Offsets are computed as `mean(measured) - mean(simulated)` per satellite. Applying the opposite value (subtract the offset from the measured Doppler) removes the bias.

| Measured | Match | Offset (Hz) | Correction Applied |
|----------|-------|------------|--------------------|
| sat_2  | IRIDIUM181 | +5 129.7  | −5 129.7 |
| sat_3  | IRIDIUM109 | +2 503.1  | −2 503.1 |
| sat_13 | IRIDIUM134 | −12 136.8 | +12 136.8 |
| sat_18 | IRIDIUM160 | +14 491.8 | −14 491.8 |
| sat_23 | IRIDIUM163 | −1 535.1  | +1 535.1 |
| sat_25 | IRIDIUM144 | +10 728.9 | −10 728.9 |
| sat_46 | IRIDIUM104 | +8 720.4  | −8 720.4 |
| sat_49 | IRIDIUM119 | +8 085.9  | −8 085.9 |
| sat_78 | IRIDIUM107 | −8 323.1  | +8 323.1 |
| sat_90 | IRIDIUM137 | +9 121.3  | −9 121.3 |

After corrections the residual mean offset for every satellite is ≈0 Hz.

---

## 💾 Corrected Measurement Files

For every `sat_X_ira_doppler.csv` there is now a companion file `sat_X_ira_doppler_corrected.csv` containing:

- `Doppler_Frequency_Hz_Original`
- `Frequency_Correction_Applied_Hz`
- `Doppler_Frequency_Hz` (corrected – use this column)

Scripts automatically regenerate these files when simulation data changes.

---

## 🔁 Re-running the pipeline

1. `python improved_matching_with_inversion.py`
2. `python unique_matching.py`  → writes `matching_results.csv`
3. `python analyze_frequency_offset.py`
4. `python apply_corrections.py`

Artifacts created:
- `improved_matching_with_inversion.png`
- `unique_matching_results.png`
- `frequency_offset_analysis.png`
- `corrected_doppler_comparison.png`
- `matching_results.csv`
- `*_corrected.csv` files

---

## 📝 Notes

- The new satellite set introduces both positive and negative offsets (some passes start below the sim curve). Corrections now handle both cases automatically.
- The large differences in offset magnitude are driven by different observation geometries; once corrected, the Doppler shapes align (see `corrected_doppler_comparison.png`).
- If you extend the measurement set again, simply drop the new `sat_*` CSVs into the folder and rerun the four scripts above.

---

**Status:** ✅ Analysis, matching, frequency offset estimation, and correction files updated for all 10 measured satellites.



