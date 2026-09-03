"""Phase 2.4–2.6: data quality + standardization + validation for both annual and quarterly.

Inputs:
    data/clean/node_features_raw.parquet              (441,934 × 39, annual)
    data/clean/node_features_quarterly.parquet        (1,556,311 × 65, quarterly)

Outputs:
    data/clean/node_features_standardized.parquet
    data/clean/node_features_quarterly_standardized.parquet
    data/clean/feature_coverage_annual.csv
    data/clean/feature_coverage_quarterly.csv
    data/clean/feature_correlations_annual.csv
    data/clean/feature_correlations_quarterly.csv
    data/clean/phase2_summary.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT_ROOT / "data/clean"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ============================================================
# Column groupings
# ============================================================
WINSORIZE_ACCOUNTING = [
    'debt_to_assets', 'debt_to_equity', 'interest_coverage',
    'roa', 'roe', 'ebitda_margin', 'gross_margin', 'operational_margin',
    'current_ratio', 'quick_ratio', 'cash_to_assets', 'wc_to_assets',
    'asset_turnover', 'receivables_turnover', 'inventory_turnover',
    'revenue_growth', 'asset_growth', 'emp_growth',
    'opcf_to_assets', 'capex_to_assets', 'fcf_to_assets',
    'lt_debt_ratio', 'st_debt_ratio', 'st_debt_share',
]
WINSORIZE_MARKET = [
    'ret_12m', 'ret_3m', 'volatility_12m', 'market_to_book', 'share_turnover',
]
WINSORIZE_ALL = WINSORIZE_ACCOUNTING + WINSORIZE_MARKET

# Standardize (z-score by period) — log-transformed sizes are included
STANDARDIZE_EXTRAS = ['log_assets', 'log_revenue', 'log_mktcap',
                       'log_mktcap_crsp', 'log_avg_volume']

# Excluded entirely from both winsorization and standardization
MACRO_COLS = ['A191RL1Q225SBEA', 'BAA10Y', 'BAMLC0A4CBBB', 'BAMLH0A0HYM2',
              'DAAA', 'DBAA', 'FEDFUNDS', 'GS10', 'INDPRO', 'T10Y2Y',
              'TB3MS', 'TEDRATE', 'UNRATE', 'VIXCLS', 'BAA_AAA_spread',
              'sp500_ret_12m']
ALTMAN_COLS = ['altman_z', 'altman_zone', 'altman_variant']
ID_COLS = ['gvkey', 'datadate', 'conm', 'sic', 'permno', 'has_crsp',
           'node_type', 'fyear', 'fyearq', 'fqtr',
           'total_debt', 'emp', 'ebit_proxy']  # raw dollar / count fields kept untouched


# ============================================================
# Helpers
# ============================================================
def winsorize_by_period(df: pd.DataFrame, cols, period_cols, low=0.01, high=0.99):
    """Clip each col to [period-quantile(low), period-quantile(high)] in place."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, []
    # Vectorized per-period quantiles, broadcast back via .transform
    for c in cols:
        g = df.groupby(period_cols, sort=False, observed=True)[c]
        lo = g.transform(lambda s: s.quantile(low))
        hi = g.transform(lambda s: s.quantile(high))
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df, cols


def standardize_by_period(df: pd.DataFrame, cols, period_cols):
    """Z-score each col within (period_cols) groups in place. ddof=1 (pandas default)."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df, []
    means = df.groupby(period_cols, sort=False, observed=True)[cols].transform('mean')
    stds  = df.groupby(period_cols, sort=False, observed=True)[cols].transform('std')
    df[cols] = (df[cols] - means) / stds.where(stds > 0)
    return df, cols


def coverage_table(df: pd.DataFrame, decade_col='__decade'):
    """Per-feature missingness rate, plus by-decade breakdown."""
    overall = (1 - df.isna().mean()).rename('coverage_pct') * 100
    by_decade = (1 - df.groupby(decade_col).apply(lambda g: g.isna().mean())) * 100
    by_decade = by_decade.T  # rows=feature, cols=decade
    by_decade.columns = [f'{int(d)}s' for d in by_decade.columns]
    out = pd.DataFrame({'overall_coverage_pct': overall.round(2)})
    out = out.join(by_decade.round(2))
    out.index.name = 'feature'
    return out.sort_values('overall_coverage_pct')


def correlation_pairs(df: pd.DataFrame, cols, threshold=0.90):
    """Return pairs with |corr| > threshold from the listed cols."""
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr().abs()
    # Upper triangle
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > threshold:
                # Sign by re-checking signed correlation
                signed = df[[a, b]].corr().iloc[0, 1]
                pairs.append({'feature_a': a, 'feature_b': b, 'corr': float(round(signed, 4))})
    pairs.sort(key=lambda x: abs(x['corr']), reverse=True)
    return pairs


def ensure_log_avg_volume(df: pd.DataFrame):
    """Replace nonpositive avg_volume with NaN, take log."""
    if 'avg_volume' in df.columns and 'log_avg_volume' not in df.columns:
        df['log_avg_volume'] = np.log(df['avg_volume'].where(df['avg_volume'] > 0))
    return df


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(label, raw_path, out_path, period_cols, fy_col):
    print('\n' + '='*72)
    print(f' {label.upper()} pipeline')
    print('='*72)

    stamp(f'loading {raw_path.name}...')
    df = pd.read_parquet(raw_path)
    print(f'  shape: {df.shape}')

    # ---- derive fyear if missing (annual) ----
    if fy_col == 'fyear' and 'fyear' not in df.columns:
        df['fyear'] = df['datadate'].dt.year.astype('Int64')

    # ---- log_avg_volume (quarterly only — avg_volume not in annual) ----
    df = ensure_log_avg_volume(df)

    # ---- inf cleanup ----
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_total = int(np.isinf(df[num_cols]).sum().sum())
    if inf_total:
        print(f'  found {inf_total} infinities; replacing with NaN')
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # ---- decade column for coverage ----
    df['__decade'] = (df['datadate'].dt.year // 10 * 10).astype('Int64')

    # ---- coverage report ----
    stamp('coverage report...')
    cov = coverage_table(df)
    cov_path = CLEAN / f'feature_coverage_{label}.csv'
    cov.to_csv(cov_path)
    print(f'  wrote {cov_path.name} ({len(cov)} features)')

    high_missing = cov[cov['overall_coverage_pct'] < 20]
    if not high_missing.empty:
        print(f'  ⚠ features with >80% missing (candidates for removal):')
        for f, r in high_missing['overall_coverage_pct'].items():
            print(f'    {f}: {100-r:.1f}% missing')
    else:
        print('  no features with >80% missing')

    df = df.drop(columns='__decade')

    # ---- winsorize ----
    stamp(f'winsorizing by {period_cols}...')
    df, winsorized = winsorize_by_period(df, WINSORIZE_ALL, period_cols)
    print(f'  winsorized {len(winsorized)} cols: {winsorized}')

    # ---- standardize ----
    stamp(f'standardizing by {period_cols}...')
    standardize_cols = WINSORIZE_ALL + STANDARDIZE_EXTRAS
    df, standardized = standardize_by_period(df, standardize_cols, period_cols)
    print(f'  standardized {len(standardized)} cols')

    # ---- save ----
    stamp(f'saving {out_path.name}...')
    # Drop helper col if any
    df.to_parquet(out_path, index=False)
    print(f'  wrote {out_path.name} ({df.shape[0]:,} × {df.shape[1]}, '
          f'{out_path.stat().st_size/1e6:.1f} MB)')

    # ---- distribution sanity (mean ≈ 0, std ≈ 1 within periods) ----
    print('\n  --- distribution sanity (sample periods) ---')
    sample_periods = sorted(df[fy_col].dropna().unique())
    sample_periods = [sample_periods[len(sample_periods)//4],
                      sample_periods[len(sample_periods)//2],
                      sample_periods[3*len(sample_periods)//4],
                      sample_periods[-1]]
    sample_cols = ['debt_to_assets', 'roa', 'log_assets']
    for p in sample_periods:
        sub = df[df[fy_col] == p]
        if sub.empty:
            continue
        stats = sub[sample_cols].agg(['mean', 'std']).round(3)
        print(f'  {fy_col}={p}: ' +
              ', '.join(f'{c} (μ={stats.loc["mean", c]:+.3f}, σ={stats.loc["std", c]:.3f})'
                         for c in sample_cols if c in df.columns))

    # ---- correlation matrix ----
    stamp('correlation analysis...')
    corr_features = [c for c in (WINSORIZE_ALL + STANDARDIZE_EXTRAS) if c in df.columns]
    pairs = correlation_pairs(df, corr_features, threshold=0.90)
    corr_path = CLEAN / f'feature_correlations_{label}.csv'
    pd.DataFrame(pairs).to_csv(corr_path, index=False)
    print(f'  wrote {corr_path.name} ({len(pairs)} pairs with |r|>0.90)')
    if pairs:
        for p in pairs[:10]:
            print(f'    {p["feature_a"]:>22s}  ↔  {p["feature_b"]:<22s}  r = {p["corr"]:+.3f}')
    else:
        print('  no high-correlation pairs')

    return df, cov, winsorized, standardized, pairs


# ============================================================
# Run both pipelines
# ============================================================
df_a, cov_a, wins_a, stand_a, pairs_a = run_pipeline(
    'annual',
    CLEAN / 'node_features_raw.parquet',
    CLEAN / 'node_features_standardized.parquet',
    period_cols=['fyear'],
    fy_col='fyear',
)

df_q, cov_q, wins_q, stand_q, pairs_q = run_pipeline(
    'quarterly',
    CLEAN / 'node_features_quarterly.parquet',
    CLEAN / 'node_features_quarterly_standardized.parquet',
    period_cols=['fyearq', 'fqtr'],
    fy_col='fyearq',
)

# ============================================================
# Spot checks
# ============================================================
print('\n' + '='*72)
print(' SPOT CHECKS (z-scored values: positive = above period avg, negative = below)')
print('='*72)

def show_spot(df, name, mask, cols):
    sub = df[mask]
    if sub.empty:
        print(f'  {name}: NO ROW MATCHED')
        return None
    row = sub.iloc[0]
    print(f'\n  {name}:')
    for c in cols:
        if c not in row:
            continue
        v = row[c]
        if pd.isna(v):
            print(f'    {c:25s}      NaN')
        elif isinstance(v, str):
            print(f'    {c:25s} {v}')
        else:
            print(f'    {c:25s} {v:+8.3f}')
    return row


def row_to_dict(row, cols):
    """Coerce row values to JSON-friendly types."""
    out = {}
    for c in cols:
        if c not in row:
            continue
        v = row[c]
        if pd.isna(v):
            out[c] = None
        elif isinstance(v, str):
            out[c] = v
        else:
            out[c] = float(v)
    return out

spot_results = {}

# ANNUAL
print('\n--- annual (node_features_standardized.parquet) ---')
key_cols_acct = ['debt_to_assets', 'lt_debt_ratio', 'roa', 'roe',
                  'ebitda_margin', 'log_assets', 'sp500_ret_12m',
                  'altman_z', 'altman_zone', 'node_type']

r = show_spot(df_a, 'Enron fy2000 (gvkey 6127)',
               (df_a['gvkey']==6127) & (df_a['fyear']==2000),
               key_cols_acct)
spot_results['enron_2000_annual'] = row_to_dict(r, key_cols_acct) if r is not None else None

r = show_spot(df_a, 'Apple fy2023 (gvkey 1690)',
               (df_a['gvkey']==1690) & (df_a['fyear']==2023),
               key_cols_acct)
spot_results['apple_2023_annual'] = row_to_dict(r, key_cols_acct) if r is not None else None

r = show_spot(df_a, 'JPMorgan fy2023 (gvkey 2968)',
               (df_a['gvkey']==2968) & (df_a['fyear']==2023),
               key_cols_acct)
spot_results['jpmorgan_2023_annual'] = row_to_dict(r, key_cols_acct) if r is not None else None

# QUARTERLY
print('\n--- quarterly (node_features_quarterly_standardized.parquet) ---')
key_cols_q = key_cols_acct + ['ret_12m', 'ret_3m', 'volatility_12m', 'market_to_book']

r = show_spot(df_q, 'Enron Q3-2001 (gvkey 6127)',
               (df_q['gvkey']==6127) & (df_q['datadate']=='2001-09-30'),
               key_cols_q)
spot_results['enron_q3_2001'] = row_to_dict(r, key_cols_q) if r is not None else None

# Apple latest quarter in 2023 (Q4 calendar)
r = show_spot(df_q, 'Apple fy2023 Q4 (gvkey 1690)',
               (df_q['gvkey']==1690) & (df_q['fyearq']==2023) & (df_q['fqtr']==4),
               key_cols_q)
spot_results['apple_2023_quarterly'] = row_to_dict(r, key_cols_q) if r is not None else None

r = show_spot(df_q, 'JPMorgan fy2023 Q4 (gvkey 2968)',
               (df_q['gvkey']==2968) & (df_q['fyearq']==2023) & (df_q['fqtr']==4),
               key_cols_q)
spot_results['jpmorgan_2023_quarterly'] = row_to_dict(r, key_cols_q) if r is not None else None

# ============================================================
# Save phase2_summary.json
# ============================================================
def coverage_summary(cov):
    return {
        'features': int(len(cov)),
        'mean_coverage_pct': float(cov['overall_coverage_pct'].mean().round(2)),
        'features_above_80_coverage': int((cov['overall_coverage_pct'] >= 80).sum()),
        'features_below_20_coverage': int((cov['overall_coverage_pct'] < 20).sum()),
    }

summary = {
    'annual': {
        'rows': int(len(df_a)),
        'features': int(len(df_a.columns)),
        'period_cols': ['fyear'],
        'winsorized_cols': wins_a,
        'standardized_cols': stand_a,
        'excluded_macro': MACRO_COLS,
        'excluded_altman': ALTMAN_COLS,
        'excluded_ids': ID_COLS,
        'coverage_summary': coverage_summary(cov_a),
        'high_correlation_pairs': pairs_a,
        'spot_checks': {k: v for k, v in spot_results.items() if 'annual' in k or 'enron_2000' in k},
    },
    'quarterly': {
        'rows': int(len(df_q)),
        'features': int(len(df_q.columns)),
        'period_cols': ['fyearq', 'fqtr'],
        'winsorized_cols': wins_q,
        'standardized_cols': stand_q,
        'excluded_macro': MACRO_COLS,
        'excluded_altman': ALTMAN_COLS,
        'excluded_ids': ID_COLS,
        'coverage_summary': coverage_summary(cov_q),
        'high_correlation_pairs': pairs_q,
        'spot_checks': {k: v for k, v in spot_results.items() if 'quarterly' in k or 'q3_2001' in k},
    },
}
summary_path = CLEAN / 'phase2_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f'\nwrote {summary_path}')

# ============================================================
# Final summary table
# ============================================================
print('\n' + '='*72)
print(' PHASE 2 SUMMARY')
print('='*72)
print(f'{"":40s} {"annual":>12s} {"quarterly":>12s}')
print(f'{"rows":40s} {len(df_a):>12,} {len(df_q):>12,}')
print(f'{"total features":40s} {len(df_a.columns):>12d} {len(df_q.columns):>12d}')
print(f'{"winsorized cols":40s} {len(wins_a):>12d} {len(wins_q):>12d}')
print(f'{"standardized cols":40s} {len(stand_a):>12d} {len(stand_q):>12d}')
print(f'{"high-correlation pairs (|r|>0.9)":40s} {len(pairs_a):>12d} {len(pairs_q):>12d}')
print(f'{"mean overall coverage (%)":40s} {cov_a["overall_coverage_pct"].mean():>12.1f} '
      f'{cov_q["overall_coverage_pct"].mean():>12.1f}')

stamp('done.')
