# ================================================================
# 09C | WTG172: Are the 08 flags explained by wakes or peer sensors?
# Paste ALL of this in ONE new cell after 09B.  Existing wake physics
# and screening flags are READ ONLY; no new thresholds are optimized.
# ================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# -------- Sources: the output created by your last successful cell --------
ROOT_C = Path(PROJECT_ROOT) if 'PROJECT_ROOT' in globals() else Path(
    r'C:\Users\alex-\OneDrive\Documents\GitHub\CU-Capstone-Project')
PREVIOUS_C = ROOT_C / 'data/processed/09_WTG172_power_flag_overlap/peer_wind_power_check'
OUT_C = ROOT_C / 'data/processed/09_WTG172_power_flag_overlap/wake_mast_peer_lidar'
OUT_C.mkdir(parents=True, exist_ok=True)
TARGET_C = 'WTG172'
OFFSETS_C = [-10, -5, 0, 5, 10]  # EXACT offsets used by 08
LOOKUP_STEP_C = 5
LOW_WAKE_PCT_C = 2.0           # EXACT screen from 08
MAST_LAT_C, MAST_LON_C = 35.089338, -118.254800  # confirmed mast GPS

# Pin the 08 run to the one used in 09B, when still present in notebook.
if 'RUN_09B' in globals() and Path(RUN_09B).is_dir():
    RUN_C = Path(RUN_09B)
elif 'screening_run' in globals() and Path(screening_run).is_dir():
    RUN_C = Path(screening_run)
elif 'run' in globals() and isinstance(run, (Path, str)) and Path(run).is_dir():
    RUN_C = Path(run)
else:
    candidates_c = sorted(
        (ROOT_C / 'data/processed/08_scada_screening').glob('run_*'), reverse=True)
    RUN_C = next((p for p in candidates_c
                  if (p / '08_jensen_all_turbines.csv').is_file()
                  and (p / '08_turbine_geometry.csv').is_file()
                  and (p / '08_fixed_peers.csv').is_file()
                  and (p / 'intervals/WTG172_screened_intervals.csv').is_file()), None)
if RUN_C is None:
    raise FileNotFoundError('No complete 08 screening run. Run 08 first.')

paired_path_c = PREVIOUS_C / 'WTG172_four_turbines_flagged_days_10min.csv'
if not paired_path_c.is_file():
    raise FileNotFoundError('Run the previous 09B cell first: ' + str(paired_path_c))

# This is the SAME set of 10-minute rows used in the previous peer/power plots.
g = pd.read_csv(paired_path_c, low_memory=False)
g['timestamp'] = pd.to_datetime(g['timestamp'], errors='coerce', utc=True)
if g.timestamp.isna().any() or g.timestamp.duplicated().any():
    raise ValueError('Missing/duplicated 09B time labels; inspect the saved file.')

def as_bool_c(x):
    s = x.astype(str).str.strip().str.lower()
    ok = s.isin(['true', 'false', '1', '0', '1.0', '0.0'])
    if not ok.all():
        raise ValueError('Unknown/missing Boolean flag(s): ' + str(s.loc[~ok].unique()))
    return s.isin(['true', '1', '1.0'])

g['flagged_08'] = as_bool_c(g['flagged_08'])
g = g.set_index('timestamp').sort_index()
fixed_peers_c = pd.read_csv(RUN_C / '08_fixed_peers.csv')
PEERS_C = fixed_peers_c.loc[fixed_peers_c.turbine.eq(TARGET_C)].sort_values('rank').peer.tolist()
TURBINES_C = [TARGET_C] + PEERS_C
if len(PEERS_C) != 3 or len(set(PEERS_C)) != 3:
    raise ValueError('Need exactly three fixed 08 comparison turbines.')
for t in TURBINES_C:
    for field in [f'wind_{t}_ms', f'power_{t}_kw']:
        if field not in g:
            raise KeyError(f'Missing {field} in 09B exported data.')
    for field in [f'wind_{t}_ms', f'power_{t}_kw']:
        g[field] = pd.to_numeric(g[field], errors='coerce')
for field in ['wind172_minus_peer_ms', 'power172_minus_peer_kw']:
    g[field] = pd.to_numeric(g[field], errors='coerce')

# Check flags directly against the completed 08 run (do not mix run versions).
flags_csv_c = pd.read_csv(RUN_C / 'intervals/WTG172_screened_intervals.csv')
flags_csv_c['timestamp'] = pd.to_datetime(flags_csv_c.timestamp, utc=True, errors='coerce')
if flags_csv_c.timestamp.isna().any() or flags_csv_c.timestamp.duplicated().any():
    raise ValueError('08 export has invalid/duplicate timestamps.')
actual_flag_c = (
    as_bool_c(flags_csv_c.departure_episode) |
    as_bool_c(flags_csv_c.flat_episode)
)
if 'period' in flags_csv_c:
    actual_flag_c &= flags_csv_c.period.eq('later_evaluation')
expected_flags_c = set(flags_csv_c.loc[actual_flag_c, 'timestamp'])
observed_flags_c = set(g.index[g.flagged_08])
if expected_flags_c != observed_flags_c:
    raise ValueError('09B flags and selected 08 run differ. Set RUN_09B to the original 08 run.')
print(f'08 run: {RUN_C.name} | comparison peers: {PEERS_C}')
print('Exact flag count:', len(observed_flags_c))

# -------- 1. Met mast measurements on the SAME timestamps --------
m = pd.read_parquet(ROOT_C / 'data/processed/metmasts_data.parquet')
if not any(x in m for x in ['utc', 'timestamp']):
    m = m.reset_index()
time_name_c = 'utc' if 'utc' in m else 'timestamp'
m['timestamp'] = pd.to_datetime(m[time_name_c], errors='coerce', utc=True)
if m.timestamp.isna().any() or m.timestamp.duplicated().any():
    raise ValueError('Mast timestamps invalid or duplicated; do not average or shift them.')
mast_cols_c = ['mm_wind_speed_3', 'mm_wind_direction_3',
               'mm_wind_speed_hh', 'mm_wind_direction_hh']
for col in mast_cols_c:
    if col not in m:
        m[col] = np.nan
    m[col] = pd.to_numeric(m[col], errors='coerce')
m = m.set_index('timestamp')[mast_cols_c].reindex(g.index)
g = g.join(m)
for field in ['mm_wind_direction_3', 'mm_wind_direction_hh']:
    g.loc[~g[field].between(0, 360), field] = np.nan
for field in ['mm_wind_speed_3', 'mm_wind_speed_hh']:
    g.loc[~g[field].between(0, 40), field] = np.nan
valid_direction_c = g.mm_wind_direction_3.between(0, 360)

# -------- 2. Reuse the EXACT Jensen lookup exported by 08 --------
wake_lookup_c = pd.read_csv(RUN_C / '08_jensen_all_turbines.csv')
wake_lookup_c['wind_from_deg'] = pd.to_numeric(wake_lookup_c.wind_from_deg, errors='coerce')
wake_lookup_c['predicted_deficit_pct'] = pd.to_numeric(
    wake_lookup_c.predicted_deficit_pct, errors='coerce')
if wake_lookup_c.duplicated(['turbine', 'wind_from_deg']).any():
    raise ValueError('Duplicated wake lookup keys.')
wake_grid_c = wake_lookup_c.pivot(
    index='wind_from_deg', columns='turbine', values='predicted_deficit_pct'
).reindex(index=np.arange(0, 360, 5), columns=TURBINES_C)
if not np.isfinite(wake_grid_c.to_numpy(dtype=float)).all():
    raise ValueError('Incomplete wake lookup for WTG172 and its three peers.')

# Calculate nominal and worst offset loss at each observed mast direction.
# Fixed inflow speed of 8.2 m/s, k=.05, candidate Ct=.756 INHERITED from 08.
safe_angle_c = g.mm_wind_direction_3.where(valid_direction_c, 0).to_numpy()
wake_matrix_c = wake_grid_c.to_numpy(dtype=float)
model_checks_c = []
for offset in OFFSETS_C:
    idx = (np.floor(((safe_angle_c + offset) % 360 + 2.5) / 5).astype(int) % 72)
    model_checks_c.append(wake_matrix_c[idx, :])
model_stack_c = np.stack(model_checks_c, axis=0)
for j, turbine in enumerate(TURBINES_C):
    g[f'jensen_nominal_{turbine}_pct'] = np.where(
        valid_direction_c, model_checks_c[OFFSETS_C.index(0)][:, j], np.nan)
    g[f'jensen_max_{turbine}_pct'] = np.where(
        valid_direction_c, model_stack_c[:, :, j].max(axis=0), np.nan)

g['jensen_peer_median_pct'] = g[[
    f'jensen_nominal_{p}_pct' for p in PEERS_C
]].median(axis=1, skipna=False)
g['jensen_172_minus_peer_pctpts'] = (
    g[f'jensen_nominal_{TARGET_C}_pct'] - g.jensen_peer_median_pct)
g['all_four_under_2pct'] = g[[
    f'jensen_max_{t}_pct' for t in TURBINES_C
]].lt(LOW_WAKE_PCT_C).all(axis=1)

# -------- 3. Peer LiDAR corroboration (optional, NOT ground truth) --------
lidar_file_c = ROOT_C / 'data/processed/06_research_synthesis/05_scada_lidar_matched.csv'
lidar_source_c = '06 matched LiDAR intervals'
if lidar_file_c.is_file():
    lidar_raw_c = pd.read_csv(
        lidar_file_c, usecols=lambda x: x in
        {'timestamp', 'utc', 'turbine', 'lidar_speed', 'flagged_interval'}, low_memory=False)
    lt_c = 'timestamp' if 'timestamp' in lidar_raw_c else 'utc'
    lidar_raw_c['timestamp'] = pd.to_datetime(lidar_raw_c[lt_c], errors='coerce', utc=True)
    lidar_raw_c = lidar_raw_c[lidar_raw_c.turbine.isin(PEERS_C)].copy()
    if lidar_raw_c.timestamp.isna().any() or lidar_raw_c.duplicated(['timestamp','turbine']).any():
        raise ValueError('Invalid or duplicate 06 peer LiDAR records.')
    lidar_raw_c['lidar_speed'] = pd.to_numeric(lidar_raw_c.lidar_speed, errors='coerce')
    if 'flagged_interval' in lidar_raw_c:
        flag_c = lidar_raw_c.flagged_interval.astype(str).str.strip().str.lower()
        reliable_bool_c = flag_c.isin(['true','false','1','0','1.0','0.0'])
        lidar_raw_c.loc[~reliable_bool_c, 'lidar_speed'] = np.nan
        lidar_raw_c.loc[flag_c.isin(['true','1','1.0']), 'lidar_speed'] = np.nan
    lidar_raw_c.loc[~lidar_raw_c.lidar_speed.between(3, 25), 'lidar_speed'] = np.nan
    lidar_wide_c = lidar_raw_c.pivot(
        index='timestamp', columns='turbine', values='lidar_speed'
    ).reindex(g.index, columns=PEERS_C)
else:
    lidar_source_c = 'No saved 06 LiDAR matched file; peer LiDAR unavailable'
    lidar_wide_c = pd.DataFrame(np.nan, index=g.index, columns=PEERS_C)

for peer in PEERS_C:
    g[f'lidar_{peer}_ms'] = lidar_wide_c[peer]
    g[f'scada_minus_lidar_{peer}_ms'] = (
        g[f'wind_{peer}_ms'] - g[f'lidar_{peer}_ms'])
g['peer_lidar_median_all3_ms'] = lidar_wide_c.median(axis=1, skipna=False)
g['peer_scada_median_all3_ms'] = g[[
    f'wind_{p}_ms' for p in PEERS_C
]].median(axis=1, skipna=False)
g['peer_scada_minus_lidar_med_ms'] = (
    g.peer_scada_median_all3_ms - g.peer_lidar_median_all3_ms)

# -------- 4. Auditable summaries for flagged timestamps --------
f = g.loc[g.flagged_08].copy()
if f.empty:
    raise ValueError('No flagged observations in 09B dataset.')

def circular_mean_c(deg):
    values = pd.to_numeric(deg, errors='coerce').dropna().to_numpy(dtype=float)
    values = values[(values >= 0) & (values <= 360)]
    if not len(values):
        return np.nan
    theta = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())) % 360)

def circ_abs_c(a, b):
    return np.abs((a - b + 180) % 360 - 180)

peer_lidar_rows_c = []
for p in PEERS_C:
    x = f[f'scada_minus_lidar_{p}_ms'].dropna()
    peer_lidar_rows_c.append({
        'peer': p, 'n_flagged_with_valid_lidar': len(x),
        'mean_scada_minus_lidar_ms': x.mean(),
        'median_scada_minus_lidar_ms': x.median(),
        'median_abs_disagreement_ms': x.abs().median()
    })
peer_lidar_summary_c = pd.DataFrame(peer_lidar_rows_c)

report_rows_c = []
for date, sub in f.groupby(f.index.floor('D')):
    report_rows_c.append({
        'day_utc': date.strftime('%Y-%m-%d'),
        'flagged_intervals': len(sub),
        'complete_four_scada_power': int(sub.complete_both.sum()) if 'complete_both' in sub else np.nan,
        'mast_ch3_available': int(sub.mm_wind_speed_3.notna().sum()),
        'mean_mast_ch3_ms': sub.mm_wind_speed_3.mean(),
        'mean_mast_direction_deg': circular_mean_c(sub.mm_wind_direction_3),
        'median_mast_direction_diff_hh_deg': circ_abs_c(
            sub.mm_wind_direction_3, sub.mm_wind_direction_hh).median(),
        'mean_WTG172_minus_peer_wind_ms': sub.wind172_minus_peer_ms.mean(),
        'mean_WTG172_minus_peer_power_kw': sub.power172_minus_peer_kw.mean(),
        'max_WTG172_modeled_deficit_pct': sub[f'jensen_max_{TARGET_C}_pct'].max(),
        'max_any_peer_modeled_deficit_pct': sub[[
            f'jensen_max_{p}_pct' for p in PEERS_C]].max().max(),
        'passes_08_low_wake_all_four': bool(sub.all_four_under_2pct.all()),
        'all_3_peer_lidars_available_n': int(sub.peer_lidar_median_all3_ms.notna().sum()),
        'mean_peer_scada_minus_lidar_median_ms': sub.peer_scada_minus_lidar_med_ms.mean()
    })
report_c = pd.DataFrame(report_rows_c)

print('\n--- FLAGGED-DAY WTG172 / WAKE / MAST SUMMARY ---')
display(report_c.round(2))
print('\n--- PEER LIDAR QC / AGREEMENT (context only) ---')
print(lidar_source_c)
display(peer_lidar_summary_c.round(3))
print('\nFLAGGED INTERVAL MODEL SCREEN:')
print(f'  {int(f.all_four_under_2pct.sum())} / {len(f)} pass <2% for ALL four turbines at tested angles.')
print('  This is largely by design: 08 chose flags from low-modeled-wake intervals.')
print('  A small Jensen deficit does NOT prove wake-free or terrain-free airflow.')
if int(f.all_four_under_2pct.sum()) != len(f):
    raise ValueError('08 selected flags fail their own wake screen; verify 08 lookup and run consistency.')
print('Peer SCADA–LiDAR comparisons are NOT calibrations of WTG172.')

# All required audit columns plus the prior raw SCADA and power comparisons.
g.reset_index().to_csv(OUT_C / 'WTG172_flagged_days_mast_wake_lidar.csv', index=False)
report_c.to_csv(OUT_C / 'WTG172_flagged_day_diagnostics.csv', index=False)
peer_lidar_summary_c.to_csv(OUT_C / 'WTG172_peer_lidar_quality_summary.csv', index=False)

# -------- 5. Event plots: mast/direction, existing wakes and peer LiDAR --------
for date, sub in g.groupby(g.index.floor('D')):
    selected = sub.loc[sub.flagged_08]
    if selected.empty:
        continue
    fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    ax[0].plot(sub.index, sub[f'wind_{TARGET_C}_ms'], label='WTG172 SCADA', lw=2)
    ax[0].plot(sub.index, sub.peer_scada_median_all3_ms, label='Peer SCADA median', lw=1.5)
    ax[0].plot(sub.index, sub.mm_wind_speed_3, label='Remote mast ch3', lw=1.2)
    if sub.mm_wind_speed_hh.notna().any():
        ax[0].plot(sub.index, sub.mm_wind_speed_hh, label='Remote mast HH', lw=1.1, alpha=.75)
    ax[0].set_ylabel('Reported wind (m/s)')
    ax[0].legend(ncol=4, fontsize=8)

    ax[1].plot(sub.index, sub.mm_wind_direction_3, label='Mast ch3 wind-from', lw=1.3)
    if sub.mm_wind_direction_hh.notna().any():
        ax[1].plot(sub.index, sub.mm_wind_direction_hh, label='Mast HH wind-from', lw=1.1)
    ax[1].set(ylabel='Wind-from direction (°)', ylim=(0, 360))
    ax[1].legend(fontsize=8)

    ax[2].plot(sub.index, sub[f'jensen_max_{TARGET_C}_pct'], label='WTG172 max offset', lw=1.4)
    for peer in PEERS_C:
        ax[2].plot(sub.index, sub[f'jensen_max_{peer}_pct'], label=peer, lw=1.0, alpha=.8)
    ax[2].axhline(2, color='black', ls='--', lw=.9, label='08 <2% threshold')
    ax[2].set_ylabel('Predicted hub deficit (%)')
    ax[2].legend(ncol=5, fontsize=8)

    ax[3].plot(sub.index, sub.peer_scada_median_all3_ms, label='Peer SCADA median', lw=1.3)
    ax[3].plot(sub.index, sub.peer_lidar_median_all3_ms, label='Peer LiDAR median (all 3 valid)', lw=1.3)
    ax[3].set_ylabel('Peer wind report (m/s)')
    ax[3].set_xlabel('UTC timestamp (time-label convention unverified)')
    ax[3].legend(fontsize=8)

    for a in ax:
        for stamp in selected.index:
            a.axvspan(stamp, stamp + pd.Timedelta(minutes=10), color='tab:purple', alpha=.12, lw=0)
        a.grid(alpha=.23)
    ax[0].set_title(f'WTG172 physical-context check | {date:%Y-%m-%d} | {len(selected)} exact 08 flags')
    fig.suptitle('Purple = screening flags | Jensen: fixed 8.2 m/s, unverified Ct, no terrain flow', fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(OUT_C / f'WTG172_wake_mast_lidar_{date:%Y-%m-%d}.png', dpi=180, bbox_inches='tight')
    plt.show()

# -------- 6. Geometry plot: position vs wind-from on flagged days --------
geo_c = pd.read_csv(RUN_C / '08_turbine_geometry.csv')
geo_c = geo_c.set_index('turbine').loc[TURBINES_C]
if not {'x','y','elevation_m'}.issubset(geo_c.columns):
    raise KeyError('08 geometry file missing UTM coordinates/elevation.')
from pyproj import Transformer
mast_e_c, mast_n_c = Transformer.from_crs(
    'EPSG:4326', 'EPSG:32611', always_xy=True
).transform(MAST_LON_C, MAST_LAT_C)
target_geom_c = geo_c.loc[TARGET_C]
peer_dist_c = pd.DataFrame([{
    'peer': p,
    'distance_from_WTG172_km': np.hypot(geo_c.loc[p,'x'] - target_geom_c.x,
                                       geo_c.loc[p,'y'] - target_geom_c.y) / 1000,
    'peer_ground_minus_WTG172_m': geo_c.loc[p,'elevation_m'] - target_geom_c.elevation_m
} for p in PEERS_C])
print('\nGeographic separation (not a terrain-flow calibration):')
display(peer_dist_c.round(2))
peer_dist_c.to_csv(OUT_C / 'WTG172_peer_distances_ground_elevation.csv', index=False)
fig, axs = plt.subplots(1, len(report_c), figsize=(5*len(report_c), 5), squeeze=False)
for ax, row in zip(axs.flat, report_c.itertuples(index=False)):
    for turbine in TURBINES_C:
        xy = geo_c.loc[turbine]
        color = 'tab:purple' if turbine == TARGET_C else 'tab:blue'
        ax.scatter(xy.x, xy.y, c=color, s=110 if turbine == TARGET_C else 50)
        ax.annotate(turbine, (xy.x,xy.y), xytext=(5,5), textcoords='offset points', fontsize=9)
    ax.scatter(mast_e_c, mast_n_c, marker='*', c='gold', edgecolors='black', s=140)
    ax.annotate('Met mast', (mast_e_c,mast_n_c), xytext=(5,5), textcoords='offset points', fontsize=8)
    theta = np.deg2rad(row.mean_mast_direction_deg)
    if np.isfinite(theta):
        xc, yc = geo_c.x.mean(), geo_c.y.mean()
        length = 650  # arrow length in meters, NOT wind speed
        ax.arrow(xc, yc, -np.sin(theta)*length, -np.cos(theta)*length,
                 width=12, head_width=90, length_includes_head=True,
                 color='black', zorder=5)
        ax.text(xc, yc-90, f'Flow to; from {row.mean_mast_direction_deg:.0f}°', fontsize=8)
    ax.set_title(f'{row.day_utc}\n{row.flagged_intervals} flagged intervals')
    ax.set_aspect('equal')
    ax.grid(alpha=.22)
    ax.set_xlabel('UTM Zone 11N easting (m)')
    ax.set_ylabel('UTM Zone 11N northing (m)')
fig.suptitle('Turbine geography and provisional mast wind direction on flagged dates')
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(OUT_C / 'WTG172_peer_layout_wind_on_flagged_days.png', dpi=180, bbox_inches='tight')
plt.show()

print('\nSAVED:', OUT_C)
print('Diagnoses requiring external evidence: actual SCADA wind-speed lineage, verified operating/curtailment status,')
print('local calibrated wind at WTG172, instrument calibration/maintenance, verified direction and timestamps.')
print('No fault or bias diagnosis can be confirmed from remote mast and SCADA/power alone.')
