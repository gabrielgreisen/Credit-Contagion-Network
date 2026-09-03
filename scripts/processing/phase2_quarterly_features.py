"""Prototype quarterly Phase 2 pipeline.

End-to-end:
  1. Load + filter compustat_CIQ_quarterly.csv
  2. Map quarterly cols -> annual names (snapshot for balance sheet, TTM for flows)
  3. Compute features via node_processing/ modules
  4. Attach CRSP trailing market features (12m return, 3m return, 12m vol, mktcap, M/B, volume, turnover)
  5. As-of merge FRED macro
  6. Save node_features_quarterly.parquet
  7. Validation
"""
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from node_processing import (
    levereage_solvency as lev,
    profitability as prof,
    liquidity as liq,
    activity_efficiency as act,
    growth as grow,
    cash_flow as cf,
    composite_scores as cs,
)

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ============================================================
# STEP 1 — Load + filter quarterly Compustat
# ============================================================
stamp("loading compustat quarterly...")

QCOLS = [
    # IDs / filters
    'gvkey', 'datadate', 'conm', 'sic', 'fyearq', 'fqtr',
    'datafmt', 'consol', 'curcdq', 'costat', 'indfmt',
    # Balance sheet snapshot (q-suffixed)
    'atq', 'actq', 'lctq', 'ltq', 'cheq', 'wcapq',
    'dlcq', 'dlttq', 'ceqq', 'req', 'invtq', 'rectq',
    # Income statement, single-quarter (q-suffixed flow)
    'niq', 'oiadpq', 'oibdpq', 'piq', 'xintq', 'cogsq',
    'saleq', 'revtq', 'dpq', 'xrdq', 'xsgaq',
    # YTD cash flow (y-suffix; need to diff within fyearq to get quarterly flow)
    'oancfy', 'capxy',
    # Market snapshot
    'cshoq', 'prccq', 'mkvaltq',
]

dq = pd.read_csv(
    PROJECT_ROOT / 'data/raw/compustat/compustat_CIQ_quarterly.csv',
    usecols=QCOLS,
    dtype={'gvkey': 'Int64', 'sic': 'Int64', 'fyearq': 'Int64', 'fqtr': 'Int64'},
    low_memory=False,
)
stamp(f"raw quarterly: {len(dq):,} rows × {dq.shape[1]} cols")

# Quality filters (mirror annual)
before = len(dq)
dq = dq[(dq['datafmt'] == 'STD') & (dq['consol'] == 'C') & (dq['curcdq'] == 'USD')]
stamp(f"after STD/C/USD filter: {len(dq):,} (-{before-len(dq):,})")

dq['datadate'] = pd.to_datetime(dq['datadate'])
before = len(dq)
dq = dq.drop_duplicates(['gvkey', 'datadate'], keep='last')
stamp(f"after dedup gvkey+datadate: {len(dq):,} (-{before-len(dq):,})")

before = len(dq)
empty = dq['atq'].isna() & dq['saleq'].isna() & dq['revtq'].isna()
dq = dq[~empty]
stamp(f"after drop empty (atq+saleq+revtq all null): {len(dq):,} (-{before-len(dq):,})")

dq = dq.sort_values(['gvkey', 'datadate']).reset_index(drop=True)

# ============================================================
# STEP 2 — Build TTM flows + rename to annual names
# ============================================================
stamp("building TTM flows from quarterly...")

# Recover quarterly OCF / CAPX from YTD (oancfy resets each fiscal year)
def ytd_to_q(s_ytd, fyearq):
    """Quarterly value = current YTD - prior YTD within same (gvkey, fyearq)."""
    grp = dq.groupby(['gvkey', fyearq.name], sort=False)[s_ytd.name]
    diffed = grp.diff()
    first_in_year = grp.cumcount() == 0
    out = diffed.where(~first_in_year, dq[s_ytd.name])
    return out

dq['oancfq_built'] = ytd_to_q(dq['oancfy'], dq['fyearq'])
dq['capxq_built']  = ytd_to_q(dq['capxy'],  dq['fyearq'])

# TTM = trailing 4-quarter sum, requires 4 consecutive quarters (min_periods=4)
def ttm(col):
    return (dq.groupby('gvkey', sort=False)[col]
              .rolling(window=4, min_periods=4).sum()
              .reset_index(level=0, drop=True))

flow_q_cols = ['niq', 'oiadpq', 'oibdpq', 'piq', 'xintq', 'cogsq',
               'saleq', 'revtq', 'dpq', 'xrdq', 'xsgaq',
               'oancfq_built', 'capxq_built']
ttm_cols = {c: ttm(c) for c in flow_q_cols}
for c, s in ttm_cols.items():
    dq[c + '_ttm'] = s

# Rename map: snapshot q -> annual; TTM flow -> annual
RENAME = {
    # snapshots
    'atq': 'at', 'actq': 'act', 'lctq': 'lct', 'ltq': 'lt',
    'cheq': 'che', 'wcapq': 'wcap',
    'dlcq': 'dlc', 'dlttq': 'dltt',
    'ceqq': 'ceq', 'req': 're',
    'invtq': 'invt', 'rectq': 'rect',
    'cshoq': 'csho', 'prccq': 'prcc_f', 'mkvaltq': 'mkvalt',
    # TTM flows -> annual flow names
    'niq_ttm': 'ni',
    'oiadpq_ttm': 'oiadp',
    'oibdpq_ttm': 'ebitda',     # oibdp ≈ ebitda
    'piq_ttm': 'pi',
    'xintq_ttm': 'xint',
    'cogsq_ttm': 'cogs',
    'saleq_ttm': 'sale',
    'revtq_ttm': 'revt',
    'dpq_ttm': 'dp',
    'xrdq_ttm': 'xrd',
    'xsgaq_ttm': 'xsga',
    'oancfq_built_ttm': 'oancf',
    'capxq_built_ttm':  'capx',
}
df = dq.rename(columns=RENAME).copy()

# Derived: gp = sale - cogs (annual `gp` not present in quarterly)
df['gp'] = df['sale'] - df['cogs']
# ebit not in quarterly; the EBIT fallback below handles it via oiadp -> pi+xint
df['ebit'] = np.nan

# Annual cols with NO quarterly equivalent
NO_Q_EQUIV = ['emp']
for c in NO_Q_EQUIV:
    df[c] = np.nan

stamp(f"TTM frame: {df.shape}")

# ============================================================
# STEP 3 — Denominator hygiene + feature computation
# ============================================================
stamp("computing features...")
ZERO_TO_NAN = ['at', 'ceq', 'sale', 'lct', 'lt', 'xint', 'rect', 'invt', 'cogs']
for c in ZERO_TO_NAN:
    df[c] = df[c].replace(0, np.nan)

feat = df[['gvkey', 'datadate', 'conm', 'sic', 'fyearq', 'fqtr']].copy()
feat['node_type'] = np.where(df['sic'].between(6000, 6999), 'financial', 'nonfinancial')

# Leverage / solvency
total_debt = df['dltt'].add(df['dlc'], fill_value=0)
total_debt = total_debt.where(total_debt != 0, np.nan)
feat['total_debt']        = total_debt
feat['debt_to_assets']    = lev.debt_to_assets(total_debt, df['at'])
feat['debt_to_equity']    = lev.debt_to_equity(total_debt, df['ceq'])
feat['lt_debt_ratio']     = df['dltt'] / df['at']
feat['st_debt_ratio']     = lev.st_debt_ration(df['dlc'], df['at'])
feat['interest_coverage'] = lev.interest_coverage(df['ebitda'], df['xint'])
feat['st_debt_share']     = lev.st_debt_share(df['dlc'], total_debt)

# Profitability
feat['roa']                = prof.roa(df['ni'], df['at'])
feat['roe']                = prof.roe(df['ni'], df['ceq'])
feat['ebitda_margin']      = prof.ebitda_margin(df['ebitda'], df['sale'])
feat['gross_margin']       = prof.gross_margin(df['gp'], df['sale'])
feat['operational_margin'] = prof.operational_margin(df['oiadp'], df['sale'])

# Liquidity
wc = df['wcap'].where(df['wcap'].notna(), df['act'] - df['lct'])
feat['current_ratio']  = liq.current_ratio(df['act'], df['lct'])
feat['quick_ratio']    = liq.quick_ration(df['che'], df['rect'], df['lct'])
feat['cash_to_assets'] = liq.cash_to_assets(df['che'], df['at'])
feat['wc_to_assets']   = wc / df['at']

# Size
def safe_log(s): return np.log(s.where(s > 0))
feat['log_assets']  = safe_log(df['at'])
feat['log_revenue'] = safe_log(df['sale'])
feat['log_mktcap']  = safe_log(df['csho'] * df['prcc_f'])
feat['emp']         = df['emp']  # all NaN — no quarterly emp

# Activity / efficiency
feat['asset_turnover']       = act.asset_turnover(df['sale'], df['at'])
feat['receivables_turnover'] = act.receivables_turnover(df['sale'], df['rect'])
feat['inventory_turnover']   = act.inventory_turnover(df['cogs'], df['invt'])

# Growth — same-fiscal-quarter year-over-year (lag 4 quarters within gvkey)
g = df.groupby('gvkey', sort=False)
sale_lag = g['sale'].shift(4)
at_lag   = g['at'].shift(4)
feat['revenue_growth'] = grow.revenue_growth(df['sale'], sale_lag)
feat['asset_growth']   = grow.asset_growth(df['at'],   at_lag)
feat['emp_growth']     = np.nan  # no quarterly emp

# Cash flow
feat['opcf_to_assets']  = cf.opcf_to_assets(df['oancf'], df['at'])
feat['capex_to_assets'] = cf.capex_to_assets(df['capx'], df['at'])
feat['fcf_to_assets']   = cf.fcf_to_assets(df['oancf'], df['capx'], df['at'])

# EBIT fallback ebit -> oiadp -> pi+xint (ebit always NaN here, so falls through)
ebit_proxy = df['ebit'].where(df['ebit'].notna(), df['oiadp'])
ebit_proxy = ebit_proxy.where(ebit_proxy.notna(), df['pi'].add(df['xint'], fill_value=np.nan))
feat['ebit_proxy'] = ebit_proxy

# Altman Z routing (same as annual)
v_z   = np.vectorize(cs.altman_z,              otypes=[float])
v_zp  = np.vectorize(cs.altman_z_prime,        otypes=[float])
v_zpp = np.vectorize(cs.altman_z_double_prime, otypes=[float])

sic_s = df['sic'].astype('Int64')
is_fin = sic_s.between(6000, 6999).fillna(False)
is_mfr = sic_s.between(2000, 3999).fillna(False)
has_market = df['csho'].notna() & df['prcc_f'].notna()

altman_z_score = np.full(len(df), np.nan)
altman_variant = np.full(len(df), '', dtype=object)

m = (is_mfr & has_market & ~is_fin).values
if m.any():
    altman_z_score[m] = v_z(
        df.loc[m, 'act'].values, df.loc[m, 'lct'].values,
        df.loc[m, 're'].values,  ebit_proxy[m].values,
        df.loc[m, 'csho'].values, df.loc[m, 'prcc_f'].values,
        df.loc[m, 'lt'].values,  df.loc[m, 'sale'].values,
        df.loc[m, 'at'].values,  df.loc[m, 'wcap'].values,
    )
    altman_variant[m] = 'z'
m = (is_mfr & ~has_market & ~is_fin).values
if m.any():
    altman_z_score[m] = v_zp(
        df.loc[m, 'act'].values, df.loc[m, 'lct'].values,
        df.loc[m, 're'].values,  ebit_proxy[m].values,
        df.loc[m, 'ceq'].values, df.loc[m, 'lt'].values,
        df.loc[m, 'sale'].values, df.loc[m, 'at'].values,
        df.loc[m, 'wcap'].values,
    )
    altman_variant[m] = 'z_prime'
m = (~is_mfr & ~is_fin).values
if m.any():
    altman_z_score[m] = v_zpp(
        df.loc[m, 'act'].values, df.loc[m, 'lct'].values,
        df.loc[m, 're'].values,  ebit_proxy[m].values,
        df.loc[m, 'ceq'].values, df.loc[m, 'lt'].values,
        df.loc[m, 'at'].values,  df.loc[m, 'wcap'].values,
    )
    altman_variant[m] = 'z_double_prime'
altman_variant[is_fin.values] = 'financial_excluded'

feat['altman_z']       = altman_z_score
feat['altman_variant'] = altman_variant
zone = np.full(len(df), '', dtype=object)
for var in ('z', 'z_prime', 'z_double_prime'):
    sel = (altman_variant == var)
    if sel.any():
        zone[sel] = [cs.z_zone(v, var) if not np.isnan(v) else np.nan
                     for v in altman_z_score[sel]]
feat['altman_zone'] = zone

stamp(f"variant breakdown: {feat['altman_variant'].value_counts(dropna=False).to_dict()}")

# Replace inf with nan
num_cols = feat.select_dtypes(include=[np.number]).columns
inf_before = {c: int(np.isinf(feat[c]).sum()) for c in num_cols if np.isinf(feat[c]).any()}
feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan)
stamp(f"inf cleanup: {inf_before}")

# ============================================================
# STEP 4 — CRSP trailing market features
# ============================================================
stamp("loading CRSP monthly...")
crsp = pd.read_csv(
    PROJECT_ROOT / 'data/raw/crsp/CRSP.csv',
    usecols=['PERMNO', 'date', 'RET', 'PRC', 'VOL', 'SHROUT'],
    dtype={'PERMNO': 'Int64'},
    low_memory=False,
)
stamp(f"crsp raw: {len(crsp):,}")
crsp['date'] = pd.to_datetime(crsp['date'])
# RET sometimes encodes as 'C', 'B', 'A' for missing — coerce
crsp['RET'] = pd.to_numeric(crsp['RET'], errors='coerce')
crsp = crsp.sort_values(['PERMNO', 'date']).reset_index(drop=True)

# Compute monthly mktcap and turnover at row level
crsp['mktcap'] = crsp['PRC'].abs() * crsp['SHROUT']            # $thousands (PRC $/sh × SHROUT k)
crsp['turnover'] = crsp['VOL'] / crsp['SHROUT']                # vol-shares / shares-outstanding (SHROUT in k → ratio in 1k)

# Rolling per-permno features (12m returns: cum prod; 3m: cum prod; 12m vol: std; volume mean; turnover mean)
gcrsp = crsp.groupby('PERMNO', sort=False)
log1p_ret = np.log1p(crsp['RET'])
crsp['logret'] = log1p_ret
roll12 = gcrsp['logret'].rolling(window=12, min_periods=12)
roll3  = gcrsp['logret'].rolling(window=3,  min_periods=3)
crsp['ret_12m'] = (np.exp(roll12.sum().reset_index(level=0, drop=True)) - 1)
crsp['ret_3m']  = (np.exp(roll3.sum().reset_index(level=0, drop=True))  - 1)
crsp['volatility_12m'] = (gcrsp['RET'].rolling(window=12, min_periods=12).std()
                              .reset_index(level=0, drop=True))
crsp['avg_volume']     = (gcrsp['VOL'].rolling(window=12, min_periods=12).mean()
                              .reset_index(level=0, drop=True))
crsp['share_turnover'] = (gcrsp['turnover'].rolling(window=12, min_periods=12).mean()
                              .reset_index(level=0, drop=True))
crsp['log_mktcap_crsp'] = np.log(crsp['mktcap'].where(crsp['mktcap'] > 0))
stamp("crsp rolling features computed")

# As-of merge: each (gvkey, datadate, permno) row gets the last CRSP row at-or-before datadate
fu = pd.read_parquet(PROJECT_ROOT / 'data/clean/firm_universe.parquet').reset_index()[['gvkey', 'permno', 'has_crsp']]
feat = feat.merge(fu, on='gvkey', how='left')
feat['permno'] = feat['permno'].astype('Int64')

# Build the panel for asof: sort both by date globally
crsp_keep = crsp[['PERMNO', 'date', 'PRC', 'SHROUT',
                   'ret_12m', 'ret_3m', 'volatility_12m',
                   'avg_volume', 'share_turnover', 'log_mktcap_crsp', 'mktcap']].rename(
    columns={'PERMNO': 'permno', 'date': 'crsp_date'})
crsp_keep = crsp_keep.sort_values(['crsp_date', 'permno'])

feat_for_merge = feat[feat['has_crsp'] & feat['permno'].notna()].copy()
feat_for_merge = feat_for_merge.sort_values(['datadate', 'permno']).reset_index(drop=True)

stamp(f"asof merge: {len(feat_for_merge):,} crsp-eligible quarters x crsp panel")
m = pd.merge_asof(
    feat_for_merge[['gvkey', 'datadate', 'permno']],
    crsp_keep,
    by='permno',
    left_on='datadate',
    right_on='crsp_date',
    direction='backward',
    tolerance=pd.Timedelta(days=45),  # quarter-end may fall mid-month; require recent CRSP obs
)
stamp(f"asof merge done. rows w/ crsp_date: {m['crsp_date'].notna().sum():,}")

# Attach back to feat
m_features = ['ret_12m', 'ret_3m', 'volatility_12m',
              'avg_volume', 'share_turnover', 'log_mktcap_crsp', 'PRC', 'SHROUT']
m_keyed = m[['gvkey', 'datadate'] + m_features]
feat = feat.merge(m_keyed, on=['gvkey', 'datadate'], how='left')

# Market-to-book: (abs(prc) * shrout / 1000) [millions $] / ceq [millions $]
mktcap_millions = (feat['PRC'].abs() * feat['SHROUT']) / 1000.0
feat['market_to_book'] = mktcap_millions / df['ceq'].values
feat = feat.drop(columns=['PRC', 'SHROUT'])

stamp(f"crsp features attached. ret_12m non-null: {feat['ret_12m'].notna().sum():,}")

# ============================================================
# STEP 5 — FRED macro merge (as-of)
# ============================================================
stamp("loading FRED master...")
fred = pd.read_csv(PROJECT_ROOT / 'data/raw/fred/fred_master.csv')
fred['date'] = pd.to_datetime(fred['date'])
fred = fred.sort_values('date').reset_index(drop=True)
# Derived spread
fred['BAA_AAA_spread'] = fred['DBAA'] - fred['DAAA']

feat_sorted = feat.sort_values('datadate').reset_index(drop=True)
feat_sorted = pd.merge_asof(
    feat_sorted, fred,
    left_on='datadate', right_on='date', direction='backward',
)
feat_sorted = feat_sorted.drop(columns=['date'])
feat = feat_sorted
stamp(f"fred attached. e.g. VIXCLS non-null: {feat['VIXCLS'].notna().sum():,}")

# Final inf sweep + save
num_cols = feat.select_dtypes(include=[np.number]).columns
feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan)

out = PROJECT_ROOT / 'data/clean/node_features_quarterly.parquet'
feat.to_parquet(out, index=False)
stamp(f"wrote {out} — {feat.shape[0]:,} rows × {feat.shape[1]} cols, {out.stat().st_size/1e6:.1f} MB")

# ============================================================
# STEP 6 — Validation
# ============================================================
print("\n=== shape ===")
print(f"  rows: {len(feat):,}  unique gvkeys: {feat['gvkey'].nunique():,}")
print(f"  date range: {feat['datadate'].min()} -> {feat['datadate'].max()}")

print("\n=== coverage (top + bottom) ===")
nn = feat.notna().sum().sort_values()
print((nn / len(feat) * 100).round(1).to_string())

print("\n=== infinity check ===")
inf = {c: int(np.isinf(feat[c]).sum()) for c in num_cols if np.isinf(feat[c]).any()}
print(inf or 'no infinities')

print("\n=== Enron quarterly trajectory (gvkey 6127) ===")
en = feat[feat['gvkey'] == 6127].sort_values('datadate')
en_view = en[['datadate', 'fyearq', 'fqtr', 'altman_z', 'altman_zone',
              'debt_to_assets', 'roa', 'log_assets',
              'ret_12m', 'volatility_12m', 'market_to_book']]
with pd.option_context('display.max_rows', 80, 'display.width', 200, 'display.max_colwidth', 30):
    print(en_view.tail(20))

print("\n=== market features NaN-by-design check ===")
no_crsp = feat[~feat['has_crsp']]
print(f"  rows w/o CRSP: {len(no_crsp):,}")
print(f"  ret_12m non-null among them: {no_crsp['ret_12m'].notna().sum()} (expect 0)")
print(f"  market_to_book non-null among them: {no_crsp['market_to_book'].notna().sum()} (expect 0)")
