"""6.2 + 6.4: Stream-clean TRACE BTDS and compute trade-level credit spreads."""
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw"
CLEAN = PROJECT_ROOT / "data/clean"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# === Bond → firm map ===
bond_map = pd.read_parquet(CLEAN / 'trace_bond_firm_map.parquet')
bond_map['cusip_id'] = bond_map['cusip_id'].astype(str).str.upper()
print(f'bond_map: {len(bond_map):,} bonds')
cusip_to_gvkey = dict(zip(bond_map['cusip_id'], bond_map['gvkey'].astype(int)))
cusip_to_mat = dict(zip(bond_map['cusip_id'], bond_map['mtrty_dt']))
mapped_cusips = set(bond_map['cusip_id'].tolist())
print(f'  unique cusips: {len(mapped_cusips):,}')

# === Treasury curve ===
tc = pd.read_csv(RAW / 'fred/treasury_curve/treasury_curve.csv')
tc['date'] = pd.to_datetime(tc['date'])
tc = tc.sort_values('date').reset_index(drop=True)
TENORS = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float32)
treasury_dates = tc['date'].values
treasury_matrix = tc[['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10', 'GS20', 'GS30']].values.astype(np.float32)
print(f'treasury_curve: {len(tc):,} dates × 8 tenors')

# Helper: vectorized linear interp across tenors
def interp_treasury(rates_rows: np.ndarray, mat_yrs: np.ndarray) -> np.ndarray:
    """rates_rows: (N, 8) treasury yield matrix per row; mat_yrs: (N,) remaining maturity in years."""
    # searchsorted finds insertion index in TENORS
    idx = np.searchsorted(TENORS, mat_yrs, side='right') - 1
    L = np.clip(idx, 0, len(TENORS) - 2)
    R = L + 1
    t_l = TENORS[L]
    t_r = TENORS[R]
    w = (mat_yrs - t_l) / (t_r - t_l)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    rows = np.arange(len(L))
    rate_l = rates_rows[rows, L]
    rate_r = rates_rows[rows, R]
    return rate_l * (1 - w) + rate_r * w

# Helper: parse FINRA volume strings ('1MM+' = 1,000,000 lower bound, etc.)
def parse_vol(s):
    if not isinstance(s, str): return np.nan
    s = s.strip().upper()
    if s.endswith('MM+'): return float(s[:-3] or '1') * 1_000_000
    if s.endswith('MM'):  return float(s[:-2] or '1') * 1_000_000
    if s.endswith('M+'):  return float(s[:-2] or '1') * 1_000
    if s.endswith('M'):   return float(s[:-1] or '1') * 1_000
    if s.endswith('K+'):  return float(s[:-2] or '1') * 1_000
    try: return float(s)
    except (ValueError, TypeError): return np.nan

# Pre-build vectorized parser for typical numeric volumes
def parse_vol_array(arr):
    """Vectorized — fast path for pure-digit strings, slow path for special codes."""
    s = pd.Series(arr).astype(str)
    # Fast path: pure numeric
    out = pd.to_numeric(s, errors='coerce')
    # Slow path: handle MM+, M+ etc on rows that failed
    bad_mask = out.isna() & s.notna() & (s != 'nan')
    if bad_mask.any():
        out.loc[bad_mask] = s.loc[bad_mask].apply(parse_vol)
    return out.values

# === Streaming reader ===
TRACE_PATH = str(RAW / 'trace/trace_standard_BTDS.csv')

# Status codes to drop
DROP_STATUS = {'C', 'W', 'Y'}

# Schema for trade-level output
out_schema = pa.schema([
    ('cusip_id', pa.string()),
    ('gvkey', pa.int32()),
    ('trd_dt', pa.date32()),
    ('year', pa.int16()),
    ('quarter', pa.int8()),
    ('yld_pt', pa.float32()),
    ('rptd_pr', pa.float32()),
    ('vol', pa.float64()),
    ('mat_yrs', pa.float32()),
    ('treas_yld', pa.float32()),
    ('spread_bps', pa.float32()),
])

# Treasury merge_asof helper: pre-build sorted treasury date array
treas_df = tc.copy()
treas_df['date_num'] = treas_df['date'].astype('int64') // 10**9  # secs since epoch

# === Process ===
stamp('opening pyarrow CSV stream...')
ro = pacsv.ReadOptions(use_threads=True, block_size=1<<26)  # 64 MB blocks
co = pacsv.ConvertOptions(
    include_columns=['cusip_id', 'trd_exctn_dt', 'rptd_pr', 'yld_pt',
                      'ascii_rptd_vol_tx', 'trc_st'],
    column_types={'cusip_id': 'string', 'trc_st': 'string',
                  'rptd_pr': 'float64', 'yld_pt': 'float64',
                  'ascii_rptd_vol_tx': 'string',
                  'trd_exctn_dt': 'date32[day]'},
    null_values=['', 'NA', 'NaN'],
)
reader = pacsv.open_csv(TRACE_PATH, read_options=ro, convert_options=co)

out_path = CLEAN / 'trace_trade_spreads.parquet'
writer = pq.ParquetWriter(str(out_path), out_schema, compression='zstd')

stats = {
    'rows_seen': 0,
    'after_status': 0,
    'after_price': 0,
    'after_yield': 0,
    'after_cusip_filter': 0,
    'after_maturity': 0,
    'after_spread_filter': 0,
    'final': 0,
}
batch_count = 0

stamp('streaming TRACE BTDS...')
for batch in reader:
    batch_count += 1
    n0 = batch.num_rows
    stats['rows_seen'] += n0

    df = batch.to_pandas()

    # 1. Status filter — drop {C, W, Y}
    df = df[~df['trc_st'].isin(DROP_STATUS)]
    stats['after_status'] += len(df)
    if len(df) == 0: continue

    # 2. Price filter — between 1 and 200
    df = df[(df['rptd_pr'] >= 1) & (df['rptd_pr'] <= 200)]
    stats['after_price'] += len(df)
    if len(df) == 0: continue

    # 3. Yield filter — between 0 and 50
    df = df[df['yld_pt'].notna() & (df['yld_pt'] >= 0) & (df['yld_pt'] <= 50)]
    stats['after_yield'] += len(df)
    if len(df) == 0: continue

    # 4. CUSIP filter — keep only mapped corporate bonds
    df['cusip_id'] = df['cusip_id'].astype(str).str.upper()
    df = df[df['cusip_id'].isin(mapped_cusips)]
    stats['after_cusip_filter'] += len(df)
    if len(df) == 0: continue

    # 5. Compute remaining maturity
    df['gvkey'] = df['cusip_id'].map(cusip_to_gvkey).astype('int32')
    df['mat_dt'] = df['cusip_id'].map(cusip_to_mat)
    df['trd_dt_pd'] = pd.to_datetime(df['trd_exctn_dt'])
    df['mat_yrs'] = ((pd.to_datetime(df['mat_dt']) - df['trd_dt_pd']).dt.days / 365.25).astype('float32')
    df = df[(df['mat_yrs'] >= 0.25) & (df['mat_yrs'] <= 30)]
    stats['after_maturity'] += len(df)
    if len(df) == 0: continue

    # 6. Vol parse
    df['vol'] = parse_vol_array(df['ascii_rptd_vol_tx'].values)

    # 7. Treasury interpolation
    # Treasury merge_asof: for each trade date, find latest treasury_date ≤ trd_dt
    df_sorted = df.sort_values('trd_dt_pd').reset_index()
    treas_match = pd.merge_asof(
        df_sorted[['index', 'trd_dt_pd', 'mat_yrs']],
        treas_df.rename(columns={'date': 'tr_date'}),
        left_on='trd_dt_pd', right_on='tr_date', direction='backward')
    # Order back
    treas_match = treas_match.set_index('index').reindex(df.index)
    rates = treas_match[['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10', 'GS20', 'GS30']].values.astype(np.float32)
    mat_arr = df['mat_yrs'].values.astype(np.float32)
    treas_yld = interp_treasury(rates, mat_arr)
    df['treas_yld'] = treas_yld
    df['spread_bps'] = ((df['yld_pt'].values.astype(np.float32) - treas_yld) * 100).astype(np.float32)

    # 8. Spread filter — drop < -100 or > 5000
    df = df[(df['spread_bps'] >= -100) & (df['spread_bps'] <= 5000)]
    stats['after_spread_filter'] += len(df)
    if len(df) == 0: continue

    # 9. Build output
    df['year'] = df['trd_dt_pd'].dt.year.astype('int16')
    df['quarter'] = df['trd_dt_pd'].dt.quarter.astype('int8')
    out = df[['cusip_id', 'gvkey', 'trd_exctn_dt', 'year', 'quarter',
              'yld_pt', 'rptd_pr', 'vol', 'mat_yrs', 'treas_yld', 'spread_bps']]
    out = out.rename(columns={'trd_exctn_dt': 'trd_dt'})
    out['yld_pt'] = out['yld_pt'].astype('float32')
    out['rptd_pr'] = out['rptd_pr'].astype('float32')
    out['vol'] = out['vol'].astype('float64')

    table = pa.Table.from_pandas(out, schema=out_schema, preserve_index=False)
    writer.write_table(table)
    stats['final'] += len(out)

    if batch_count % 20 == 0:
        stamp(f'  batch {batch_count}: {stats["rows_seen"]/1e6:.1f}M scanned, '
              f'{stats["final"]/1e6:.2f}M kept ({stats["final"]/stats["rows_seen"]:.1%})')

writer.close()
stamp(f'streaming complete — {batch_count} batches')

print(f'\n=== TRACE Cleaning Stats ===')
for k, v in stats.items():
    print(f'  {k:<25s}: {v:>15,}')
print(f'\n  output: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)')
