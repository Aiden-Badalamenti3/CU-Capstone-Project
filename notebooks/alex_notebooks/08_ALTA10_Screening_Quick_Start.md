# 08 - ALTA10 farm-wide SCADA screening

## Run

Save either `08_ALTA10_Farmwide_SCADA_Screening.ipynb` or the `.py` version inside
`CU-Capstone-Project/notebooks/alex_notebooks/`.
Open the notebook and run from top to bottom. Alternatively, in a notebook in that folder:

```python
%run "08_ALTA10_Farmwide_SCADA_Screening.py"
```

No need to rerun 06 or 07. This file reads the original processed SCADA/mast tables,
turbine coordinates, terrain raster, and (optionally) reference power curve.
It never reads the mast inspection workbook or any LiDAR measurements.

## Required local inputs

- `data/processed/scada_data.parquet`
- `data/processed/metmasts_data.parquet`
- `turbine_coordinate_ALTA10.csv` somewhere under `data/`
- `elevation.tif` somewhere under `data/`
- Optional: `GE_2_85MW_103mRD_PC.csv` under `data/`

Dependencies: pandas, NumPy, matplotlib, a Parquet reader such as pyarrow, pyproj,
and rasterio. These overlap the existing 07 environment. The code never installs packages.

## What this version does

1. Reuses the existing Jensen assumptions at all coordinate-listed turbines.
2. Picks three fixed nearby SCADA peers, excluding the target.
3. Requires simultaneous generating-proxy and low-predicted-wake eligibility for all four.
4. Uses the first 60% of available calendar days (from May 1, 2026 onward by default)
   as an early reference period and the later 40% for retrospective evaluation.
5. Learns an early median target-minus-peer difference within mast-direction and speed bins.
6. Flags sustained later departures and near-flat wind signals with changing peers/power.
7. Retains unadjusted peer differences so a persistent offset is not completely hidden
   by subtracting an early baseline.

No claim is made that the early period is healthy or that later dates have never been inspected.
No training labels are taken from the six LiDAR sites. WTG159 need not be flagged by this screen.

## First-pass research settings (not validated tolerances)

- Mast channel 3, speed 7-10 m/s.
- Both target and fixed peers: <2% modeled deficit at -10, -5, 0, +5, +10 degree shifts.
- Three closest peers within 3 km and 75 m ground-elevation difference.
- Power 100-2850 kW and reported wind 3-25 m/s; these are proxies, not confirmed normal operation.
- Baseline cells: 30-degree direction sectors and 1 m/s mast-speed bins; 30 records/5 days minimum.
- Adequate evaluation: 200 peer records/10 days, 50% record coverage; baseline-dependent checks
  also require 200 scored records/10 days and 50% condition support.
- Departure: same-sign deviation beyond max(1 m/s, 3 robust baseline standard deviations)
  for at least three consecutive 10-minute labels.
- Near-flat: six consecutive eligible labels, target range <=0.02 m/s, peer range >=1 m/s,
  and target power range >=200 kW.
- Persistent raw peer discrepancy: >=70% of eligible later records differ in the same direction
  by more than 1 m/s. This yields MONITOR, not a fault conclusion.
- Priority: adequate data plus departure episodes on >=3 days and >=5% of scored records,
  or near-flat candidates on >=3 days.

Freeze the settings before inspecting results. Later sensitivity runs should be documented,
not selected because they produce a desired result. Each run saves a new output folder.

## Outputs

Under `data/processed/08_scada_screening/run_<timestamp>/`:

- `08_turbine_review_table.csv` - labels, reasons, peer IDs and evidence coverage.
- `08_data_coverage.csv` - missing values, available records, and screening attrition.
- `08_fixed_peers.csv` - distances and ground-elevation differences of each chosen peer.
- `08_flagged_episodes.csv` - dates and durations of qualifying episodes.
- `08_early_condition_baselines.csv` - fitted early-period summaries and support.
- `intervals/` - auditable screened measurements and flags for every site.
- `08_power_context_ONLY.csv` - optional descriptive power residuals, never used in labels.
- `08_jensen_all_turbines.csv` - farm-wide wake lookup.
- `08_run_settings.json` and `08_INTERPRETATION.txt` - reproducibility and caveats.
- Separate charts for label counts, raw peer differences, and up to three flagged-site timelines.

## How to interpret labels

- **Priority investigation:** predefined recurrence rules triggered with adequate evidence.
- **Monitor / context required:** a candidate pattern or persistent difference needs context.
- **Insufficient evidence:** sparse measurements, unavailable peers, unsupported conditions,
  or inadequate coverage. This is not agreement or proof of health.
- **No strong anomaly in assessed conditions:** no implemented trigger in sufficiently
  covered selected conditions. Not certification of the anemometer.

This is an inspection-prioritization system, not a broken/healthy classifier.
It does not resolve SCADA channel lineage, terrain flow, mast calibration, instrument
heights, upstream operational states, or errors shared by several peers. Power may be
circular evidence if SCADA wind speed is power-derived. Mast inspection notes are
background documentation only; no automatic channel remapping is inferred from them.

## First outputs to review

Review `08_turbine_review_table.csv`, `08_data_coverage.csv`, and `08_fixed_peers.csv`
together. Do not interpret priorities before checking usable samples and peer suitability.

## Test scope

Tested on synthetic CSV inputs with a terrain raster, including planted shifts, flat
signals, persistent offsets, missing turbine data, duplicate rejection, timestamp gaps,
and comparison of the wake equations against 07. Not run on the full local ALTA10
Parquet measurements. Synthetic test results are not ALTA10 findings.
