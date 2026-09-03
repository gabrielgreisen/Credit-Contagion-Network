"""3.5 Geographic Edges — same state."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT_ROOT / "data/clean"
EDGES = CLEAN / "edges"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
GVKEYS = set(fu['gvkey'].astype(int).tolist())
fu['gvkey'] = fu['gvkey'].astype(int)
fu_state = fu.dropna(subset=['state'])[['gvkey', 'state']].copy()
print(f'  firms with state: {len(fu_state):,}')

# Active firms per year: from firm_years.parquet (gvkey × year)
stamp('loading firm_years for active years...')
fy = pd.read_parquet(CLEAN / 'firm_years.parquet', columns=['gvkey', 'datadate'])
fy['datadate'] = pd.to_datetime(fy['datadate'])
fy['year'] = fy['datadate'].dt.year.astype(int)
fy['gvkey'] = fy['gvkey'].astype(int)
fy = fy[fy['gvkey'].isin(GVKEYS)].drop_duplicates(['gvkey', 'year'])
print(f'  firm-years: {len(fy):,}')

# Join state
fy = fy.merge(fu_state, on='gvkey', how='left')
fy = fy.dropna(subset=['state'])
print(f'  firm-years with state: {len(fy):,}')

YEAR_MIN = int(fy['year'].min())
YEAR_MAX = int(fy['year'].max())
print(f'  year range: {YEAR_MIN}–{YEAR_MAX}')

quarters = np.array([1, 2, 3, 4], dtype=np.int8)

schema = pa.schema([
    ('gvkey_1', pa.int32()),
    ('gvkey_2', pa.int32()),
    ('year', pa.int16()),
    ('quarter', pa.int8()),
    ('state', pa.string()),
])
out_path = EDGES / 'geographic_edges.parquet'
writer = pq.ParquetWriter(str(out_path), schema, compression='zstd')

stamp(f'building geographic edges per (year, state) — {YEAR_MAX-YEAR_MIN+1} years...')
total = 0
counts_per_year = {}

for year in range(YEAR_MIN, YEAR_MAX + 1):
    yr = fy[fy['year'] == year]
    if len(yr) < 2:
        continue
    yr_pairs = []
    for state, sub in yr.groupby('state', sort=False):
        gvks = np.sort(sub['gvkey'].values)
        n = len(gvks)
        if n < 2:
            continue
        ii, jj = np.triu_indices(n, k=1)
        yr_pairs.append(pd.DataFrame({
            'gvkey_1': gvks[ii],
            'gvkey_2': gvks[jj],
            'state': state,
        }))
    if not yr_pairs:
        continue
    df_year_pairs = pd.concat(yr_pairs, ignore_index=True)
    counts_per_year[year] = len(df_year_pairs)

    # Expand to 4 quarters
    df_year_q = df_year_pairs.assign(_k=1).merge(
        pd.DataFrame({'quarter': quarters, '_k': 1}), on='_k').drop(columns='_k')
    df_year_q['year'] = np.int16(year)
    df_year_q['gvkey_1'] = df_year_q['gvkey_1'].astype('int32')
    df_year_q['gvkey_2'] = df_year_q['gvkey_2'].astype('int32')
    df_year_q['quarter'] = df_year_q['quarter'].astype('int8')
    df_year_q = df_year_q[['gvkey_1', 'gvkey_2', 'year', 'quarter', 'state']]

    table = pa.Table.from_pandas(df_year_q, schema=schema, preserve_index=False)
    writer.write_table(table)
    total += len(df_year_q)
    if year % 5 == 0 or year == YEAR_MAX:
        stamp(f'  year {year}: {len(df_year_pairs):,} pairs (cum quarterly: {total:,})')
    del df_year_pairs, df_year_q

writer.close()
stamp(f'wrote {out_path.name} — {total:,} rows, {out_path.stat().st_size/1e6:.1f} MB')

print(f'\n=== Geographic Edge Stats ===')
print(f'  total quarterly edges: {total:,}')
print(f'  edges per year (annual unique pairs):')
for y in (1995, 2000, 2005, 2010, 2015, 2020):
    if y in counts_per_year:
        print(f'    {y}: {counts_per_year[y]:>12,}')
