# %% [markdown]
# # 09 — WTG172: Are SCADA screening events also power-curve anomalies?
#
# Run this AFTER running all cells in power_curve_ALTA10_adapted.ipynb.
# Reuses that notebook's `scada`, `all_turbine_bins`, `PROJECT_ROOT`, and
# `THEORETICAL_PLOT_LABEL`; does not modify the 08 wake screening or the
# partner's density-adjustment / power-curve definitions.
#
# March 2026 is OUTSIDE the 08 screening period (begins May 1).
# Screen flags are NOT diagnoses of anemometer bias, curtailment, or failure.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x.to_string(index=False))

TARGET = 'WTG172'
# Leave None to choose newest COMPLETE 08 screening run.
EIGHT_RUN_DIR = None
COMPARE_MONTHS = ['2026-03', '2026-08']  # Context; includes flagged months automatically.
MIN_REFERENCE_BIN_N = 10
MAX_EVENT_DAYS_TO_PLOT = 10

# Require original adapted partner notebook variables rather than silently
# generating a different curve using different wind corrections/units.
required_globals = ['scada', 'all_turbine_bins', 'PROJECT_ROOT', 'THEORETICAL_PLOT_LABEL']
missing_globals = [s for s in required_globals if s not in globals()]
if missing_globals:
    raise RuntimeError(
        'Run ALL cells in power_curve_ALTA10_adapted.ipynb first. '
        f'Missing variables: {missing_globals}'
    )

root = Path(PROJECT_ROOT)
runs_dir = root / 'data' / 'processed' / '08_scada_screening'
if EIGHT_RUN_DIR is None:
    candidates = sorted(runs_dir.glob('run_*'), reverse=True)
    run = next((p for p in candidates
                if (p / '08_flagged_episodes.csv').is_file()
                and (p / '08_fixed_peers.csv').is_file()
                and (p / 'intervals' / f'{TARGET}_screened_intervals.csv').is_file()), None)
    if run is None:
        raise FileNotFoundError('No COMPLETE 08 run found. Check data/processed/08_scada_screening/.')
else:
    run = Path(EIGHT_RUN_DIR)

out = root / 'data' / 'processed' / '09_WTG172_power_flag_overlap'
out.mkdir(parents=True, exist_ok=True)
print('Using 08 screening run:', run)
print('Output folder:', out)

# ---------------- 1. Load EXACT ten-minute screening flags ----------------
episodes = pd.read_csv(run / '08_flagged_episodes.csv')
episodes = episodes.loc[episodes['turbine'].astype(str).eq(TARGET)].copy()
for c in ['first_timestamp_label_utc', 'last_timestamp_label_utc']:
    episodes[c] = pd.to_datetime(episodes[c], errors='coerce', utc=True)
if episodes[['first_timestamp_label_utc', 'last_timestamp_label_utc']].isna().any().any():
    raise ValueError('08 event CSV has invalid timestamp labels.')

screen = pd.read_csv(run / 'intervals' / f'{TARGET}_screened_intervals.csv', low_memory=False)
screen['timestamp'] = pd.to_datetime(screen['timestamp'], errors='coerce', utc=True)
if screen['timestamp'].isna().any() or screen['timestamp'].duplicated().any():
    raise ValueError('Invalid or duplicate WTG172 08 screening timestamps.')
for col in ['departure_episode', 'flat_episode']:
    if col not in screen.columns:
        raise KeyError(f'Expected {col} in 08 WTG172 interval CSV.')
    screen[col] = screen[col].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
screen['screen_flag'] = screen['departure_episode'] | screen['flat_episode']
if 'period' in screen.columns:
    screen = screen.loc[screen['period'].eq('later_evaluation')].copy()

flags = pd.DatetimeIndex(screen.loc[screen.screen_flag, 'timestamp']).unique().sort_values()
if len(flags) == 0:
    raise ValueError('No WTG172 flagged ten-minute timestamps in this 08 run.')
print(f'{len(flags):,} flagged 10-minute timestamps, {len(flags.normalize().unique())} distinct days.')
print('Flagged months:', ', '.join(sorted({d.strftime('%Y-%m') for d in flags})))
print('Event records from 08:')
display(episodes[['reason', 'first_timestamp_label_utc', 'last_timestamp_label_utc', 'records']])

peers_df = pd.read_csv(run / '08_fixed_peers.csv')
peer_rows = peers_df.loc[peers_df['turbine'].eq(TARGET)].sort_values('rank')
peers = peer_rows['peer'].astype(str).tolist()
if len(peers) != 3:
    raise ValueError(f'Expected three WTG172 peer turbines; got {peers}')
print('Comparison peers:', ', '.join(peers))

# ---------------- 2. Use EXACT power definitions from partner notebook ----------------
needed = ['utc', 'turbine', 'wind_speed', 'corrected_ws',
          'KPI_real_power_ac', 'comparison_theoretical_power']
missing_cols = [c for c in needed if c not in scada.columns]
if missing_cols:
    raise KeyError(f'Partner notebook scada DataFrame missing: {missing_cols}')

s = scada.loc[scada['turbine'].isin([TARGET] + peers), needed].copy()
s['utc'] = pd.to_datetime(s['utc'], errors='coerce', utc=True)
if s.utc.isna().any() or s.duplicated(['utc', 'turbine']).any():
    raise ValueError('Check SCADA timestamp validity and turbine-time duplicates.')
for col in ['wind_speed', 'corrected_ws', 'KPI_real_power_ac', 'comparison_theoretical_power']:
    s[col] = pd.to_numeric(s[col], errors='coerce')
s['month'] = s.utc.dt.strftime('%Y-%m')
s172 = s.loc[s.turbine.eq(TARGET)].copy()
s172['08_flag'] = s172['utc'].isin(flags)
s172['power_difference_kw'] = (
    s172['KPI_real_power_ac'] - s172['comparison_theoretical_power']
)

# Match the exact event-screened sample to the same SCADA records.
event_matched = screen.merge(
    s172[['utc', 'corrected_ws', 'KPI_real_power_ac',
          'comparison_theoretical_power', 'power_difference_kw', 'month']],
    left_on='timestamp', right_on='utc', how='left', validate='one_to_one'
)
event_matched.to_csv(out / 'WTG172_08_screened_with_power.csv', index=False)
print('08 flagged timestamps with actual power:', int(event_matched.loc[event_matched.screen_flag, 'KPI_real_power_ac'].notna().sum()))
print('08 flagged timestamps with theoretical/reference power:', int(event_matched.loc[event_matched.screen_flag, 'comparison_theoretical_power'].notna().sum()))

# Descriptive comparison of flagged vs nonflagged eligible 08 intervals.
summary = event_matched.groupby(['month', 'screen_flag'], dropna=False).agg(
    intervals=('timestamp', 'size'),
    actual_power_available=('KPI_real_power_ac', 'count'),
    actual_power_mean_kw=('KPI_real_power_ac', 'mean'),
    theoretical_power_mean_kw=('comparison_theoretical_power', 'mean'),
    mean_actual_minus_theoretical_kw=('power_difference_kw', 'mean'),
    median_actual_minus_theoretical_kw=('power_difference_kw', 'median')
).reset_index()
summary.to_csv(out / 'WTG172_flagged_unflagged_power_summary.csv', index=False)
print('\nPower summaries (descriptive; groups need not have identical wind distributions):')
display(summary.round(2))

# Compare event and non-event power gaps within the SAME month and corrected
# wind-speed bin, keeping only non-event bins with >=10 comparator points.
valid = event_matched.loc[
    event_matched['corrected_ws'].between(3, 15)
    & event_matched['power_difference_kw'].notna()
].copy()
valid['half_ms_bin'] = np.floor((valid['corrected_ws'] - 3) / .5).astype(int)
non_event_bins = valid.loc[~valid.screen_flag].groupby(
    ['month', 'half_ms_bin']
)['power_difference_kw'].agg(non_event_mean='mean', non_event_n='size').reset_index()
matched_bin = valid.loc[valid.screen_flag].merge(
    non_event_bins, on=['month', 'half_ms_bin'], how='left', validate='many_to_one'
)
matched_bin = matched_bin.loc[
    matched_bin['non_event_n'].ge(MIN_REFERENCE_BIN_N)
].copy()
matched_bin['difference_from_comparable_nonflagged_kw'] = (
    matched_bin['power_difference_kw'] - matched_bin['non_event_mean']
)
matched_bin.to_csv(out / 'WTG172_event_matched_power_gaps.csv', index=False)
print(f'Flagged intervals with >= {MIN_REFERENCE_BIN_N} nonflagged SAME-month/speed-bin comparators:', len(matched_bin))
if len(matched_bin):
    print('Mean event power-gap difference from matched nonflagged bins (kW):',
          round(matched_bin['difference_from_comparable_nonflagged_kw'].mean(), 2))
else:
    print('No sufficiently supported within-month/speed-bin comparison. Do not infer association.')

# ---------------- 3. Reproduce MONTHLY curves and add exact flag markers ----------------
bins = all_turbine_bins.loc[all_turbine_bins.turbine.eq(TARGET)].copy()
bins['month'] = bins['period'].astype(str)
flag_months = {t.strftime('%Y-%m') for t in flags}
months_to_show = sorted(set(COMPARE_MONTHS).union(flag_months))
ncols = 2
nrows = int(np.ceil(len(months_to_show) / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4.4 * nrows), squeeze=False, sharex=True, sharey=True)
for ax, month in zip(axs.flat, months_to_show):
    g = bins.loc[(bins.month.eq(month)) & bins.n.ge(10)].sort_values('corrected_ws')
    ax.plot(g.corrected_ws, g.mean_real_power, color='tab:blue', lw=1.8, label='Actual (monthly bin mean)')
    ax.plot(g.corrected_ws, g.mean_theoretical_power, color='tab:red', lw=1.5, label=str(THEORETICAL_PLOT_LABEL))
    ax.plot(g.corrected_ws, g.mean_fleet_power, color='black', lw=1.5, label='Other-turbine fleet')
    e = s172.loc[s172.month.eq(month) & s172['08_flag'] &
                 s172.corrected_ws.between(3, 15) & s172.KPI_real_power_ac.notna()]
    if not e.empty:
        ax.scatter(e.corrected_ws, e.KPI_real_power_ac, marker='X', s=62,
                   color='tab:purple', edgecolors='white', linewidths=.3,
                   label=f'08 flagged timestamps (n={len(e)})', zorder=5)
    message = ('Outside 08 date window' if month < '2026-05' else
               f'08 flagged points displayed: {len(e)}')
    ax.text(.02, .96, message, transform=ax.transAxes, va='top', fontsize=9)
    ax.set(title=month, xlabel='Density-adjusted SCADA wind speed (m/s)',
           ylabel='WTG172 electrical power (kW)', xlim=(3, 15), ylim=(0, 3000))
    ax.grid(alpha=.23)
for ax in axs.flat[len(months_to_show):]:
    ax.axis('off')
h, labs = axs.flat[0].get_legend_handles_labels()
# Prefer a panel that has highlighted event markers when available.
for a in axs.flat[:len(months_to_show)]:
    hh, ll = a.get_legend_handles_labels()
    if len(ll) > len(labs):
        h, labs = hh, ll
fig.legend(h, labs, loc='upper center', bbox_to_anchor=(.5, .963),
           ncol=min(4, len(labs)), fontsize=9)
fig.suptitle('WTG172: monthly power curves with exact SCADA-screening flags', y=.995)
fig.tight_layout(rect=[0, 0, 1, .89])
fig.savefig(out / 'WTG172_monthly_power_curves_with_08_flags.png', dpi=180, bbox_inches='tight')
plt.show()

# ---------------- 4. Exact day-level event study (target + individual peers) ----------------
flag_days = sorted(pd.DatetimeIndex(flags.normalize().unique()))
for date in flag_days[:MAX_EVENT_DAYS_TO_PLOT]:
    start, end = date, date + pd.Timedelta(days=1)
    # A complete 10-minute grid prevents visual lines over missing intervals.
    t_axis = pd.date_range(start, end, freq='10min', inclusive='left', tz='UTC')
    day_s = s.loc[s.utc.ge(start) & s.utc.lt(end)]
    day_screen = screen.loc[screen.timestamp.ge(start) & screen.timestamp.lt(end)].set_index('timestamp')
    day_screen = day_screen.reindex(t_axis)
    current_flags = pd.DatetimeIndex(flags[(flags >= start) & (flags < end)])
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1.6, 1.2, 1.4]})
    # SCADA wind-speed comparison with each individual peer.
    for turbine in [TARGET] + peers:
        g = day_s.loc[day_s.turbine.eq(turbine)].set_index('utc').reindex(t_axis)
        axes[0].plot(t_axis, g.wind_speed, lw=2 if turbine == TARGET else 1.1,
                     label=turbine, alpha=1 if turbine == TARGET else .75)
    axes[0].set(ylabel='SCADA wind speed (m/s)',
                title=f'{TARGET} | {date.strftime("%Y-%m-%d")} | {len(current_flags)} flagged 10-min records')
    axes[0].legend(loc='upper right', ncol=4, fontsize=9)

    # 08 adjusted residual and its original two-sided trigger threshold.
    axes[1].plot(t_axis, day_screen.adjusted_residual_ms, '.-', label='Deviation from earlier peer relationship')
    axes[1].plot(t_axis, day_screen.departure_limit_ms, ls='--', label='+ threshold')
    axes[1].plot(t_axis, -day_screen.departure_limit_ms, ls='--', label='− threshold')
    axes[1].axhline(0, lw=.8, color='black')
    axes[1].set(ylabel='Adjusted peer residual (m/s)')
    axes[1].legend(loc='upper right', fontsize=8)

    # Power at WTG172; theoretical may be SCADA KPI or GE fallback.
    p = day_s.loc[day_s.turbine.eq(TARGET)].set_index('utc').reindex(t_axis)
    axes[2].plot(t_axis, p.KPI_real_power_ac, label='Actual WTG172 power', lw=1.7)
    axes[2].plot(t_axis, p.comparison_theoretical_power,
                 label=str(THEORETICAL_PLOT_LABEL), lw=1.4)
    axes[2].set(ylabel='Electrical power (kW)', xlabel='Time (UTC)')
    axes[2].legend(loc='upper right', fontsize=9)

    for ax in axes:
        for stamp in current_flags:
            ax.axvspan(stamp, stamp + pd.Timedelta(minutes=10),
                       color='tab:purple', alpha=.16, lw=0)
        ax.grid(alpha=.23)
    fig.suptitle('Purple shading = exact WTG172 SCADA-screening flags; not a confirmed sensor fault', fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, .955])
    path = out / f'WTG172_screened_event_{date.strftime("%Y-%m-%d")}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()

if len(flag_days) > MAX_EVENT_DAYS_TO_PLOT:
    print(f'Only first {MAX_EVENT_DAYS_TO_PLOT} flagged days plotted; {len(flag_days)} exist.')
print('\nDone. Figures and CSVs saved to:', out)
print('Interpretation: matching anomaly times does not prove the power gap was caused by a wind-speed sensor.')
print('March 2026 is not part of the 08 screening period.')
