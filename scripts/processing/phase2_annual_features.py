"""Prototype Phase 2 node feature pipeline. Run end-to-end against firm_years.parquet
to validate logic before porting to notebook."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from node_processing import (
    levereage_solvency as lev,
    profitability as prof,
    liquidity as liq,
    size_features as size,
    activity_efficiency as act,
    growth as grow,
    cash_flow as cf,
    composite_scores as cs,
)

# -------- 1. Load --------
KEEP = [
    'gvkey', 'datadate', 'conm', 'sic', 'node_type',
    'at', 'ceq', 'sale', 'lct', 'lt', 'xint', 'rect', 'invt', 'cogs',
    'act', 'che', 'wcap', 'dlc', 'dltt',
    'ni', 'ebitda', 'gp', 'oiadp', 'pi', 'ebit', 're',
    'oancf', 'capx', 'csho', 'prcc_f', 'mkvalt', 'emp',
]
df = pd.read_parquet(PROJECT_ROOT / 'data/clean/firm_years.parquet', columns=KEEP)
df['datadate'] = pd.to_datetime(df['datadate'])
df = df.sort_values(['gvkey', 'datadate']).reset_index(drop=True)
print(f"loaded {len(df):,} firm-years, {df['gvkey'].nunique():,} gvkeys")

# -------- 2. Denominator hygiene --------
ZERO_TO_NAN = ['at', 'ceq', 'sale', 'lct', 'lt', 'xint', 'rect', 'invt', 'cogs']
for c in ZERO_TO_NAN:
    df[c] = df[c].replace(0, np.nan)

# -------- 3. Compute features (modules used directly where they vectorize) --------
feat = df[['gvkey', 'datadate', 'conm', 'sic', 'node_type']].copy()

# leverage / solvency — total_debt has a scalar zero-guard, do inline; sum becomes nan if both nan
total_debt = df['dltt'].add(df['dlc'], fill_value=0)
total_debt = total_debt.where(total_debt != 0, np.nan)
feat['total_debt']        = total_debt
feat['debt_to_assets']    = lev.debt_to_assets(total_debt, df['at'])
feat['debt_to_equity']    = lev.debt_to_equity(total_debt, df['ceq'])
feat['lt_debt_ratio']     = df['dltt'] / df['at']                      # module fn is buggy (returns tuple)
feat['st_debt_ratio']     = lev.st_debt_ration(df['dlc'], df['at'])
feat['interest_coverage'] = lev.interest_coverage(df['ebitda'], df['xint'])
feat['st_debt_share']     = lev.st_debt_share(df['dlc'], total_debt)

# profitability
feat['roa']                = prof.roa(df['ni'], df['at'])
feat['roe']                = prof.roe(df['ni'], df['ceq'])
feat['ebitda_margin']      = prof.ebitda_margin(df['ebitda'], df['sale'])
feat['gross_margin']       = prof.gross_margin(df['gp'], df['sale'])
feat['operational_margin'] = prof.operational_margin(df['oiadp'], df['sale'])

# liquidity — wc_to_assets in module is buggy (== np.nan never True); inline preferred-wcap-then-fallback
wc = df['wcap'].where(df['wcap'].notna(), df['act'] - df['lct'])
feat['current_ratio']  = liq.current_ratio(df['act'], df['lct'])
feat['quick_ratio']    = liq.quick_ration(df['che'], df['rect'], df['lct'])
feat['cash_to_assets'] = liq.cash_to_assets(df['che'], df['at'])
feat['wc_to_assets']   = wc / df['at']

# size — log of nonpositive returns nan; replace nonpositives with nan first
def safe_log(s):
    return np.log(s.where(s > 0))
feat['log_assets']  = safe_log(df['at'])
feat['log_revenue'] = safe_log(df['sale'])
mve = df['csho'] * df['prcc_f']
feat['log_mktcap']  = safe_log(mve)
feat['emp']         = df['emp']

# activity / efficiency
feat['asset_turnover']       = act.asset_turnover(df['sale'], df['at'])
feat['receivables_turnover'] = act.receivables_turnover(df['sale'], df['rect'])
feat['inventory_turnover']   = act.inventory_turnover(df['cogs'], df['invt'])

# growth — need lagged values within gvkey; assume datadate-sorted rows are sequential fiscal years
g = df.groupby('gvkey', sort=False)
sale_lag = g['sale'].shift(1)
at_lag   = g['at'].shift(1)
emp_lag  = g['emp'].shift(1)
feat['revenue_growth'] = grow.revenue_growth(df['sale'], sale_lag)
feat['asset_growth']   = grow.asset_growth(df['at'], at_lag)
feat['emp_growth']     = grow.emp_growth(df['emp'], emp_lag)

# cash flow
feat['opcf_to_assets']  = cf.opcf_to_assets(df['oancf'], df['at'])
feat['capex_to_assets'] = cf.capex_to_assets(df['capx'], df['at'])
feat['fcf_to_assets']   = cf.fcf_to_assets(df['oancf'], df['capx'], df['at'])

# -------- 4. EBIT fallback: ebit -> oiadp -> pi + xint --------
ebit_proxy = df['ebit'].where(df['ebit'].notna(), df['oiadp'])
ebit_proxy = ebit_proxy.where(ebit_proxy.notna(), df['pi'].add(df['xint'], fill_value=np.nan))
feat['ebit_proxy'] = ebit_proxy
print(f"ebit_proxy non-null: {ebit_proxy.notna().sum():,} / {len(ebit_proxy):,}")
print(f"  from ebit:   {df['ebit'].notna().sum():,}")
print(f"  from oiadp:  {(df['ebit'].isna() & df['oiadp'].notna()).sum():,}")
print(f"  from pi+xint: {(df['ebit'].isna() & df['oiadp'].isna() & ebit_proxy.notna()).sum():,}")

# -------- 5. Z-score routing --------
# Vectorize the scalar Altman fns; signatures match positionally.
v_z   = np.vectorize(cs.altman_z,                excluded=set(), otypes=[float])
v_zp  = np.vectorize(cs.altman_z_prime,          excluded=set(), otypes=[float])
v_zpp = np.vectorize(cs.altman_z_double_prime,   excluded=set(), otypes=[float])

sic = df['sic']
is_fin = sic.between(6000, 6999)
is_mfr = sic.between(2000, 3999)
has_market = df['csho'].notna() & df['prcc_f'].notna()

# Compute Z'' for everyone first (cheap, no market data dep)
zpp = v_zpp(
    df['act'].values, df['lct'].values, df['re'].values, ebit_proxy.values,
    df['ceq'].values, df['lt'].values, df['at'].values, df['wcap'].values,
)
# Compute Z (mfr w/ market data)
mfr_mask = is_mfr & has_market & ~is_fin
z = np.full(len(df), np.nan)
if mfr_mask.any():
    idx = mfr_mask.values
    z[idx] = v_z(
        df.loc[mfr_mask, 'act'].values, df.loc[mfr_mask, 'lct'].values,
        df.loc[mfr_mask, 're'].values,  ebit_proxy[mfr_mask].values,
        df.loc[mfr_mask, 'csho'].values, df.loc[mfr_mask, 'prcc_f'].values,
        df.loc[mfr_mask, 'lt'].values,  df.loc[mfr_mask, 'sale'].values,
        df.loc[mfr_mask, 'at'].values,  df.loc[mfr_mask, 'wcap'].values,
    )
# Compute Z' (mfr w/o market data — fallback)
mfr_book_mask = is_mfr & ~has_market & ~is_fin
zp = np.full(len(df), np.nan)
if mfr_book_mask.any():
    idx = mfr_book_mask.values
    zp[idx] = v_zp(
        df.loc[mfr_book_mask, 'act'].values, df.loc[mfr_book_mask, 'lct'].values,
        df.loc[mfr_book_mask, 're'].values,  ebit_proxy[mfr_book_mask].values,
        df.loc[mfr_book_mask, 'ceq'].values, df.loc[mfr_book_mask, 'lt'].values,
        df.loc[mfr_book_mask, 'sale'].values, df.loc[mfr_book_mask, 'at'].values,
        df.loc[mfr_book_mask, 'wcap'].values,
    )

# Route per row:
#   financials -> nan; manufacturers -> z (or zp fallback); other nonfinancials -> zpp
altman_z_score = np.full(len(df), np.nan)
altman_variant = np.full(len(df), '', dtype=object)
nonfin_mask = (~is_fin).values
# Manufacturers w/ market data
m1 = (mfr_mask).values
altman_z_score[m1] = z[m1]
altman_variant[m1] = 'z'
# Manufacturers w/o market data
m2 = (mfr_book_mask).values
altman_z_score[m2] = zp[m2]
altman_variant[m2] = 'z_prime'
# Other nonfinancials
m3 = (nonfin_mask & ~is_mfr.values)
altman_z_score[m3] = zpp[m3]
altman_variant[m3] = 'z_double_prime'
# Financials stay nan with empty variant -> mark explicitly
altman_variant[is_fin.values] = 'financial_excluded'

feat['altman_z']         = altman_z_score
feat['altman_variant']   = altman_variant
# Zone label
zone = np.full(len(df), '', dtype=object)
for var in ('z', 'z_prime', 'z_double_prime'):
    sel = (altman_variant == var)
    if sel.any():
        zone[sel] = [cs.z_zone(v, var) if not np.isnan(v) else np.nan
                     for v in altman_z_score[sel]]
feat['altman_zone'] = zone

print(f"\nAltman variant breakdown:")
print(feat['altman_variant'].value_counts(dropna=False))

# -------- 6. Save --------
out_path = PROJECT_ROOT / 'data/clean/node_features_raw.parquet'
feat.to_parquet(out_path, index=False)
print(f"\nwrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

# -------- 7. Validation --------
print("\n--- non-null counts per feature ---")
nn = feat.notna().sum().sort_values()
print(nn.to_string())

print("\n--- infinity check ---")
num_cols = feat.select_dtypes(include=[np.number]).columns
inf_counts = {c: int(np.isinf(feat[c]).sum()) for c in num_cols if np.isinf(feat[c]).any()}
print(inf_counts if inf_counts else "no infinities")

# -------- 8. Spot check Enron 2000 --------
print("\n--- Enron 2000 (gvkey=6127, fy2000) ---")
enron = feat[(feat['gvkey'] == 6127) & (feat['datadate'].dt.year == 2000)]
with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', 30):
    print(enron.T)
