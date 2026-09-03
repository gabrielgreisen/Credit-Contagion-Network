"""6.5 + 6.6 + 6.7: Aggregate trade-level spreads, merge into node features, validate."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw"
CLEAN = PROJECT_ROOT / "data/clean"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
fu['gvkey'] = fu['gvkey'].astype(int)
gvkey_to_name = dict(zip(fu['gvkey'], fu['conm']))
gvkey_to_node_class = dict(zip(fu['gvkey'], fu['node_class']))

# === 6.5 Aggregation ===
stamp('loading trade-level spreads...')
trades = pd.read_parquet(CLEAN / 'trace_trade_spreads.parquet')
print(f'  trades: {len(trades):,}')
print(f'  unique firms: {trades["gvkey"].nunique():,}')
print(f'  date range: {trades["trd_dt"].min()} → {trades["trd_dt"].max()}')

# Volume-weighted median helper
def vw_median(s, w):
    """Volume-weighted median."""
    if len(s) == 0:
        return np.nan
    if w.isna().all() or w.sum() == 0:
        return s.median()
    order = np.argsort(s.values)
    s_sorted = s.values[order]
    w_sorted = w.fillna(0).values[order]
    cw = np.cumsum(w_sorted)
    target = cw[-1] / 2
    idx = np.searchsorted(cw, target)
    return s_sorted[min(idx, len(s_sorted) - 1)]

stamp('aggregating to firm-quarter...')
def agg_q(grp):
    return pd.Series({
        'median_spread_bps': grp['spread_bps'].median(),
        'mean_spread_bps':   grp['spread_bps'].mean(),
        'spread_std_bps':    grp['spread_bps'].std(),
        'vw_median_spread_bps': vw_median(grp['spread_bps'], grp['vol']),
        'n_bonds':           grp['cusip_id'].nunique(),
        'n_trades':          len(grp),
        'total_volume':      grp['vol'].sum(),
    })

# Faster path: groupby agg without lambda for most cols
g = trades.groupby(['gvkey', 'year', 'quarter'])
basic = g.agg(
    median_spread_bps=('spread_bps', 'median'),
    mean_spread_bps=('spread_bps', 'mean'),
    spread_std_bps=('spread_bps', 'std'),
    n_bonds=('cusip_id', 'nunique'),
    n_trades=('cusip_id', 'count'),
    total_volume=('vol', 'sum'),
).reset_index()

stamp('  computing volume-weighted median...')
vw = (g.apply(lambda x: vw_median(x['spread_bps'], x['vol']), include_groups=False)
        .rename('vw_median_spread_bps').reset_index())
quarterly = basic.merge(vw, on=['gvkey', 'year', 'quarter'])

# Quarter-end datadate (calendar quarter)
def qend(year, q):
    months = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
    return pd.Timestamp(f'{int(year)}-{months[int(q)]}')
quarterly['datadate'] = [qend(y, q) for y, q in zip(quarterly['year'], quarterly['quarter'])]
quarterly['log_credit_spread'] = np.log(quarterly['median_spread_bps'].where(quarterly['median_spread_bps'] > 0))

quarterly = quarterly[['gvkey', 'year', 'quarter', 'datadate',
                        'median_spread_bps', 'vw_median_spread_bps', 'mean_spread_bps',
                        'spread_std_bps', 'n_bonds', 'n_trades', 'total_volume',
                        'log_credit_spread']]
out_q = CLEAN / 'credit_spreads_quarterly.parquet'
quarterly.to_parquet(out_q, index=False)
print(f'  wrote {out_q.name}: {len(quarterly):,} firm-quarters, '
      f'{out_q.stat().st_size/1e6:.1f} MB')

# === Annual aggregation: aggregate at fiscal-year level ===
# For each (gvkey, fyear) — find datadate from firm_years and roll up trades within fyear
stamp('aggregating to firm-fiscal-year (matched to firm_years datadate)...')
fy = pd.read_parquet(CLEAN / 'firm_years.parquet', columns=['gvkey', 'datadate'])
fy['datadate'] = pd.to_datetime(fy['datadate'])
fy['gvkey'] = fy['gvkey'].astype(int)
fy['fyear'] = fy['datadate'].dt.year.astype(int)

# For each firm-year, aggregate the trades whose calendar year == fyear
trades['cal_year'] = trades['year'].astype(int)
g_y = trades.groupby(['gvkey', 'cal_year'])
basic_y = g_y.agg(
    median_spread_bps=('spread_bps', 'median'),
    mean_spread_bps=('spread_bps', 'mean'),
    spread_std_bps=('spread_bps', 'std'),
    n_bonds=('cusip_id', 'nunique'),
    n_trades=('cusip_id', 'count'),
    total_volume=('vol', 'sum'),
).reset_index().rename(columns={'cal_year': 'fyear'})

vw_y = (g_y.apply(lambda x: vw_median(x['spread_bps'], x['vol']), include_groups=False)
            .rename('vw_median_spread_bps').reset_index())
vw_y = vw_y.rename(columns={'cal_year': 'fyear'})
annual = basic_y.merge(vw_y, on=['gvkey', 'fyear'])
annual['log_credit_spread'] = np.log(annual['median_spread_bps'].where(annual['median_spread_bps'] > 0))

# Merge to firm_years datadate
annual_keyed = fy.merge(annual, on=['gvkey', 'fyear'], how='inner')
annual_keyed = annual_keyed[['gvkey', 'datadate', 'fyear', 'median_spread_bps',
                              'vw_median_spread_bps', 'mean_spread_bps', 'spread_std_bps',
                              'n_bonds', 'n_trades', 'total_volume', 'log_credit_spread']]
out_a = CLEAN / 'credit_spreads_annual.parquet'
annual_keyed.to_parquet(out_a, index=False)
print(f'  wrote {out_a.name}: {len(annual_keyed):,} firm-years, '
      f'{out_a.stat().st_size/1e6:.1f} MB')

# === 6.6 Merge into node features ===
stamp('merging into quarterly node features...')
SP_COLS = ['median_spread_bps', 'log_credit_spread', 'n_bonds']

# Quarterly raw
qf = pd.read_parquet(CLEAN / 'node_features_quarterly.parquet')
print(f'  qf shape before: {qf.shape}')
qf['datadate'] = pd.to_datetime(qf['datadate'])
qf = qf.merge(quarterly[['gvkey', 'datadate'] + SP_COLS],
               on=['gvkey', 'datadate'], how='left')
qf.to_parquet(CLEAN / 'node_features_quarterly.parquet', index=False)
print(f'  qf shape after: {qf.shape}, spread non-null: '
      f'{qf["median_spread_bps"].notna().sum():,} ({qf["median_spread_bps"].notna().mean():.1%})')

# Standardized quarterly: standardize log_credit_spread by (fyearq, fqtr); keep raw median
stamp('updating standardized quarterly...')
qfs = pd.read_parquet(CLEAN / 'node_features_quarterly_standardized.parquet')
qfs['datadate'] = pd.to_datetime(qfs['datadate'])
qfs = qfs.merge(quarterly[['gvkey', 'datadate'] + SP_COLS],
                  on=['gvkey', 'datadate'], how='left')
# Standardize log_credit_spread by period
g = qfs.groupby(['fyearq', 'fqtr'], sort=False, observed=True)['log_credit_spread']
qfs['log_credit_spread'] = ((qfs['log_credit_spread'] - g.transform('mean')) /
                              g.transform('std').where(g.transform('std') > 0))
qfs.to_parquet(CLEAN / 'node_features_quarterly_standardized.parquet', index=False)
print(f'  qfs shape: {qfs.shape}')

# Annual raw
stamp('merging into annual node features...')
afr = pd.read_parquet(CLEAN / 'node_features_raw.parquet')
afr['datadate'] = pd.to_datetime(afr['datadate'])
afr = afr.merge(annual_keyed[['gvkey', 'datadate'] + SP_COLS],
                  on=['gvkey', 'datadate'], how='left')
afr.to_parquet(CLEAN / 'node_features_raw.parquet', index=False)
print(f'  afr shape after: {afr.shape}, spread non-null: '
      f'{afr["median_spread_bps"].notna().sum():,} ({afr["median_spread_bps"].notna().mean():.1%})')

# Annual standardized
stamp('updating standardized annual...')
afs = pd.read_parquet(CLEAN / 'node_features_standardized.parquet')
afs['datadate'] = pd.to_datetime(afs['datadate'])
afs = afs.merge(annual_keyed[['gvkey', 'datadate'] + SP_COLS],
                  on=['gvkey', 'datadate'], how='left')
g = afs.groupby('fyear', sort=False, observed=True)['log_credit_spread']
afs['log_credit_spread'] = ((afs['log_credit_spread'] - g.transform('mean')) /
                              g.transform('std').where(g.transform('std') > 0))
afs.to_parquet(CLEAN / 'node_features_standardized.parquet', index=False)
print(f'  afs shape: {afs.shape}')

# === 6.7 Validation ===
stamp('\n=== Validation ===')

# Distributions
print('\n--- median_spread_bps distribution (overall) ---')
m = quarterly['median_spread_bps']
print(f'  mean:   {m.mean():.0f} bps')
print(f'  median: {m.median():.0f} bps')
print(f'  p25:    {m.quantile(0.25):.0f} bps')
print(f'  p75:    {m.quantile(0.75):.0f} bps')
print(f'  p95:    {m.quantile(0.95):.0f} bps')

print('\n--- median_spread_bps by year (median across firm-quarters) ---')
yr_med = quarterly.groupby('year')['median_spread_bps'].median()
for y in (2002, 2005, 2007, 2008, 2009, 2010, 2015, 2019, 2020, 2021, 2022, 2024):
    if y in yr_med.index:
        print(f'  {y}: {yr_med[y]:>5.0f} bps')

print('\n--- median spread by node_class ---')
quarterly['node_class'] = quarterly['gvkey'].map(gvkey_to_node_class)
nc_med = quarterly.groupby('node_class')['median_spread_bps'].agg(['median', 'mean', 'count'])
print(nc_med.to_string())

# Spot checks
print('\n--- spot checks ---')
def spot(name, gvkey, year_lo=None, year_hi=None):
    sub = quarterly[quarterly['gvkey'] == gvkey].sort_values(['year', 'quarter'])
    if year_lo is not None:
        sub = sub[sub['year'] >= year_lo]
    if year_hi is not None:
        sub = sub[sub['year'] <= year_hi]
    if len(sub) == 0:
        print(f'  {name}: NO TRADES')
        return
    print(f'  {name} ({sub["year"].min()}–{sub["year"].max()}, {len(sub)} firm-qtrs):')
    sub2 = sub.tail(8)
    for _, row in sub2.iterrows():
        print(f'    {int(row["year"])}-Q{int(row["quarter"])}: median={row["median_spread_bps"]:>6.0f} bps, '
              f'n_bonds={int(row["n_bonds"]):>3d}, n_trades={int(row["n_trades"]):>5,}')

spot('Enron (gvkey 6127)', 6127, 2001, 2002)
spot('Lehman Brothers (gvkey 30128)', 30128, 2007, 2008)
spot('Apple (gvkey 1690)', 1690, 2018, 2024)
spot('JPMorgan (gvkey 2968)', 2968, 2018, 2024)

# Coverage report
print('\n--- coverage by year (quarterly) ---')
qcov = qf.groupby(qf['datadate'].dt.year).agg(
    rows=('gvkey', 'count'),
    with_spread=('median_spread_bps', lambda x: x.notna().sum())
)
qcov['coverage_pct'] = qcov['with_spread'] / qcov['rows'] * 100
print('  year  rows  with_spread  coverage_pct')
for y, row in qcov[(qcov.index >= 2002) & (qcov.index <= 2024)].iterrows():
    print(f'  {int(y)}: {int(row["rows"]):>7,}  {int(row["with_spread"]):>7,}  {row["coverage_pct"]:>6.1f}%')

# Coverage by firm size (log_assets quartile)
print('\n--- coverage by log_assets quartile (quarterly, 2010-2020) ---')
recent = qf[(qf['datadate'].dt.year >= 2010) & (qf['datadate'].dt.year <= 2020)]
recent = recent.dropna(subset=['log_assets'])
# log_assets is standardized to z-score? In quarterly_standardized yes; in raw yes
# In node_features_quarterly.parquet, log_assets is RAW (since this is the raw file).
# Compute quartile and coverage
recent_q = recent.copy()
recent_q['size_qtl'] = pd.qcut(recent_q['log_assets'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
size_cov = recent_q.groupby('size_qtl', observed=True).agg(
    rows=('gvkey', 'count'),
    with_spread=('median_spread_bps', lambda x: x.notna().sum())
)
size_cov['coverage_pct'] = size_cov['with_spread'] / size_cov['rows'] * 100
print(size_cov.to_string())

# Coverage by node_class
print('\n--- coverage by node_class (quarterly) ---')
qf2 = qf.copy()
qf2['node_class'] = qf2['gvkey'].map(gvkey_to_node_class)
nc_cov = qf2.groupby('node_class').agg(
    rows=('gvkey', 'count'),
    with_spread=('median_spread_bps', lambda x: x.notna().sum())
)
nc_cov['coverage_pct'] = nc_cov['with_spread'] / nc_cov['rows'] * 100
print(nc_cov.to_string())

# === Save phase6_summary.json ===
summary = {
    'treasury_curve': {
        'tenors': ['GS1','GS2','GS3','GS5','GS7','GS10','GS20','GS30'],
        'n_dates': 276,
        'date_range': '2002-01-01 to 2024-12-01',
    },
    'bond_firm_map': {
        'master_rows': 3_223_786,
        'corporate_after_filter': 2_711_043,
        'mapped_to_firm_universe': len(pd.read_parquet(CLEAN / 'trace_bond_firm_map.parquet')),
        'unique_firms': int(pd.read_parquet(CLEAN / 'trace_bond_firm_map.parquet')['gvkey'].nunique()),
    },
    'trade_spreads': {
        'total_trades': int(len(trades)),
        'unique_firms': int(trades['gvkey'].nunique()),
        'date_range': f'{trades["trd_dt"].min()} → {trades["trd_dt"].max()}',
    },
    'quarterly_aggregate': {
        'firm_quarters': int(len(quarterly)),
        'unique_firms': int(quarterly['gvkey'].nunique()),
        'median_spread_overall_bps': float(quarterly['median_spread_bps'].median()),
        'median_by_year': {int(y): float(round(v, 0)) for y, v in yr_med.items() if y >= 2002},
    },
    'annual_aggregate': {
        'firm_years': int(len(annual_keyed)),
        'unique_firms': int(annual_keyed['gvkey'].nunique()),
    },
    'coverage': {
        'quarterly_with_spread_pct': float(round(qf['median_spread_bps'].notna().mean() * 100, 2)),
        'annual_with_spread_pct': float(round(afr['median_spread_bps'].notna().mean() * 100, 2)),
        'unique_firms_with_spread': int(quarterly['gvkey'].nunique()),
    },
    'node_class_coverage': {k: {'rows': int(v['rows']), 'with_spread': int(v['with_spread']),
                                  'coverage_pct': float(round(v['coverage_pct'], 2))}
                              for k, v in nc_cov.iterrows()},
    'spot_checks': {
        'crisis_yrs_median_spread_bps': {int(y): float(round(yr_med[y], 0))
                                          for y in (2007, 2008, 2009, 2020) if y in yr_med.index},
    },
}
summary_path = CLEAN / 'phase6_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f'\nwrote {summary_path}')

print('\n' + '='*72)
print(' PHASE 6 — FINAL SUMMARY')
print('='*72)
print(f'  Treasury curve:           276 monthly dates × 8 tenors')
print(f'  Bonds in TRACE master:    3,223,786')
print(f'  Mapped corporate bonds:   {summary["bond_firm_map"]["mapped_to_firm_universe"]:,}')
print(f'  Unique firms with bonds:  {summary["bond_firm_map"]["unique_firms"]:,}')
print(f'  Clean trade-level rows:   {len(trades):,}')
print(f'  Firm-quarters with spread: {len(quarterly):,}')
print(f'  Quarterly node coverage:  {summary["coverage"]["quarterly_with_spread_pct"]:.1f}%')
print(f'  Annual node coverage:     {summary["coverage"]["annual_with_spread_pct"]:.1f}%')
