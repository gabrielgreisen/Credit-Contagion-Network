"""3.3 + 3.4 Industry Edges — 4-digit + 3-digit-only SIC."""
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
stamp(f'firm_universe: {len(GVKEYS):,} gvkeys')

# Get firm-year SIC from firm_years.parquet (annual; we'll expand to quarterly)
stamp('loading firm_years sich...')
fy = pd.read_parquet(CLEAN / 'firm_years.parquet',
                     columns=['gvkey', 'datadate', 'sich', 'sic'])
fy['datadate'] = pd.to_datetime(fy['datadate'])
fy['year'] = fy['datadate'].dt.year.astype(int)
fy['gvkey'] = fy['gvkey'].astype(int)
fy = fy[fy['gvkey'].isin(GVKEYS)]
fy['sic_use'] = fy['sich'].fillna(fy['sic'])
fy = fy.dropna(subset=['sic_use'])
fy['sic4'] = fy['sic_use'].astype(int).astype(str).str.zfill(4)
fy['sic3'] = fy['sic4'].str[:3]
# Dedup to one (gvkey, year, sic4) row
fy = fy.drop_duplicates(['gvkey', 'year']).reset_index(drop=True)
print(f'  {len(fy):,} firm-years with sich/sic')
YEAR_MIN = int(fy['year'].min())
YEAR_MAX = int(fy['year'].max())
print(f'  year range: {YEAR_MIN}–{YEAR_MAX}')

# Quarter expansion
quarters = np.array([1, 2, 3, 4], dtype=np.int8)


def emit_pairs(group, label_col):
    """Given a DataFrame with gvkey + label_col, emit all (g1<g2, label) pairs."""
    arrs = []
    for label, sub in group.groupby(label_col, sort=False):
        gvks = sub['gvkey'].values
        if len(gvks) < 2:
            continue
        gvks = np.sort(gvks)
        n = len(gvks)
        ii, jj = np.triu_indices(n, k=1)
        arrs.append(pd.DataFrame({
            'gvkey_1': gvks[ii],
            'gvkey_2': gvks[jj],
            label_col: label,
        }))
    if not arrs:
        return pd.DataFrame(columns=['gvkey_1', 'gvkey_2', label_col])
    return pd.concat(arrs, ignore_index=True)


# === 3.3: Industry 4-digit ===
stamp('building 4-digit industry edges (per year, then × 4 quarters)...')
schema_4 = pa.schema([
    ('gvkey_1', pa.int32()),
    ('gvkey_2', pa.int32()),
    ('year', pa.int16()),
    ('quarter', pa.int8()),
    ('sic_4digit', pa.string()),
])
out_4 = EDGES / 'industry_4digit_edges.parquet'
writer4 = pq.ParquetWriter(str(out_4), schema_4, compression='zstd')
total_4 = 0
counts_4_per_year = {}

# Track 4-digit pairs per year for 3-digit subtraction
sic4_pairs_per_year = {}

for year in range(YEAR_MIN, YEAR_MAX + 1):
    yr = fy[fy['year'] == year]
    if len(yr) < 2:
        continue
    pairs4 = emit_pairs(yr, 'sic4')
    if len(pairs4) == 0:
        continue
    # Store the (gvkey_1, gvkey_2) set for 3-digit subtraction (no need to keep sic4 in set)
    sic4_pairs_per_year[year] = set(zip(pairs4['gvkey_1'].values, pairs4['gvkey_2'].values))
    # Expand to 4 quarters
    pairs4_q = pairs4.assign(_k=1).merge(
        pd.DataFrame({'quarter': quarters, '_k': 1}), on='_k').drop(columns='_k')
    pairs4_q['year'] = np.int16(year)
    pairs4_q = pairs4_q.rename(columns={'sic4': 'sic_4digit'})
    pairs4_q['gvkey_1'] = pairs4_q['gvkey_1'].astype('int32')
    pairs4_q['gvkey_2'] = pairs4_q['gvkey_2'].astype('int32')
    pairs4_q['quarter'] = pairs4_q['quarter'].astype('int8')
    pairs4_q = pairs4_q[['gvkey_1', 'gvkey_2', 'year', 'quarter', 'sic_4digit']]
    table = pa.Table.from_pandas(pairs4_q, schema=schema_4, preserve_index=False)
    writer4.write_table(table)
    total_4 += len(pairs4_q)
    counts_4_per_year[year] = len(pairs4)  # annual unique pairs (× 4 = quarterly)
    if year % 5 == 0 or year == YEAR_MAX:
        stamp(f'  4-digit year {year}: {len(pairs4):,} pairs (cum quarterly: {total_4:,})')

writer4.close()
stamp(f'wrote {out_4.name} — {total_4:,} rows, {out_4.stat().st_size/1e6:.1f} MB')


# === 3.4: Industry 3-digit (additional, exclude same 4-digit) ===
stamp('building 3-digit-only industry edges...')
schema_3 = pa.schema([
    ('gvkey_1', pa.int32()),
    ('gvkey_2', pa.int32()),
    ('year', pa.int16()),
    ('quarter', pa.int8()),
    ('sic_3digit', pa.string()),
])
out_3 = EDGES / 'industry_3digit_edges.parquet'
writer3 = pq.ParquetWriter(str(out_3), schema_3, compression='zstd')
total_3 = 0
counts_3_per_year = {}

for year in range(YEAR_MIN, YEAR_MAX + 1):
    yr = fy[fy['year'] == year]
    if len(yr) < 2:
        continue
    pairs3 = emit_pairs(yr, 'sic3')
    if len(pairs3) == 0:
        continue
    # Exclude pairs already in 4-digit set
    sic4_set = sic4_pairs_per_year.get(year, set())
    if sic4_set:
        # Build a key for fast filter
        pair_keys = list(zip(pairs3['gvkey_1'].values, pairs3['gvkey_2'].values))
        mask = np.array([p not in sic4_set for p in pair_keys])
        pairs3 = pairs3[mask]
    if len(pairs3) == 0:
        continue
    pairs3_q = pairs3.assign(_k=1).merge(
        pd.DataFrame({'quarter': quarters, '_k': 1}), on='_k').drop(columns='_k')
    pairs3_q['year'] = np.int16(year)
    pairs3_q = pairs3_q.rename(columns={'sic3': 'sic_3digit'})
    pairs3_q['gvkey_1'] = pairs3_q['gvkey_1'].astype('int32')
    pairs3_q['gvkey_2'] = pairs3_q['gvkey_2'].astype('int32')
    pairs3_q['quarter'] = pairs3_q['quarter'].astype('int8')
    pairs3_q = pairs3_q[['gvkey_1', 'gvkey_2', 'year', 'quarter', 'sic_3digit']]
    table = pa.Table.from_pandas(pairs3_q, schema=schema_3, preserve_index=False)
    writer3.write_table(table)
    total_3 += len(pairs3_q)
    counts_3_per_year[year] = len(pairs3)
    if year % 5 == 0 or year == YEAR_MAX:
        stamp(f'  3-digit-only year {year}: {len(pairs3):,} pairs (cum quarterly: {total_3:,})')
writer3.close()
stamp(f'wrote {out_3.name} — {total_3:,} rows, {out_3.stat().st_size/1e6:.1f} MB')

# === Stats ===
print(f'\n=== Industry Edge Stats ===')
print(f'  4-digit: {total_4:,} quarterly edges')
print(f'  3-digit-only (additional): {total_3:,} quarterly edges')
print(f'  combined: {total_4 + total_3:,} quarterly edges')
print(f'  edges per year (annual unique pairs):')
for y in (1995, 2000, 2005, 2010, 2015, 2020):
    p4 = counts_4_per_year.get(y, 0)
    p3 = counts_3_per_year.get(y, 0)
    print(f'    {y}: 4-digit={p4:>9,}  3-digit-only={p3:>9,}')
