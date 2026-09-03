"""3.2 Common Creditor Edges — DealScan + Roberts (lead-arranger filter)."""
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw"
CLEAN = PROJECT_ROOT / "data/clean"
EDGES = CLEAN / "edges"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# Setup
fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
GVKEYS = set(fu['gvkey'].astype(int).tolist())
stamp(f'firm_universe: {len(GVKEYS):,} gvkeys')

# === Load Roberts linking table ===
stamp('loading Roberts DealScan-Compustat link...')
roberts = pd.read_excel(RAW / 'LSEG/Dealscan-Compustat_Linking_Database012024.xlsx',
                         sheet_name='links',
                         usecols=['borrowercompanyid', 'gvkey'])
roberts = roberts.dropna(subset=['borrowercompanyid', 'gvkey'])
roberts['gvkey'] = roberts['gvkey'].astype(int)
roberts = roberts[roberts['gvkey'].isin(GVKEYS)]
# One borrower can map to multiple gvkeys (M&A); take most common
borrower_to_gvkey = (roberts.groupby('borrowercompanyid')['gvkey']
                            .agg(lambda x: x.mode().iloc[0]).to_dict())
print(f'  Roberts: {len(roberts):,} rows, {len(borrower_to_gvkey):,} Borrower_Id → gvkey')

# === Load DealScan with required columns (streaming via pyarrow) ===
stamp('loading DealScan (pyarrow streaming)...')
ds_cols = ['Borrower_Id', 'Lender_Parent_Id', 'Tranche_Amount',
           'Deal_Active_Date', 'Tranche_Active_Date', 'Tranche_Maturity_Date',
           'Primary_Role']
read_opts = pacsv.ReadOptions(use_threads=True, block_size=1<<24)
parse_opts = pacsv.ParseOptions(invalid_row_handler=lambda x: 'skip')
convert_opts = pacsv.ConvertOptions(include_columns=ds_cols)
ds_table = pacsv.read_csv(str(RAW / 'LSEG/LSEG_Dealscan.csv'),
                           read_options=read_opts,
                           parse_options=parse_opts,
                           convert_options=convert_opts)
ds = ds_table.to_pandas()
del ds_table
stamp(f'DealScan loaded: {len(ds):,} rows')

# === Map Borrower_Id → gvkey ===
ds['Borrower_Id'] = pd.to_numeric(ds['Borrower_Id'], errors='coerce')
ds['Lender_Parent_Id'] = pd.to_numeric(ds['Lender_Parent_Id'], errors='coerce')
ds = ds.dropna(subset=['Borrower_Id', 'Lender_Parent_Id'])

ds['gvkey'] = ds['Borrower_Id'].astype('int64').map(borrower_to_gvkey)
before = len(ds)
ds = ds.dropna(subset=['gvkey'])
ds['gvkey'] = ds['gvkey'].astype(int)
print(f'  rows mapped to gvkey: {len(ds):,} ({len(ds)/before:.1%})')

# === Lead-arranger filter ===
stamp('applying lead-arranger filter...')
LEAD_PATTERN = (r'lead arranger|admin.*agent|administrative agent|'
                r'bookrunner|lead manager|book manager')
ds['is_lead'] = ds['Primary_Role'].astype(str).str.contains(
    LEAD_PATTERN, case=False, regex=True, na=False)
print(f'  Primary_Role distribution (top 10):')
for role, cnt in ds['Primary_Role'].value_counts().head(10).items():
    flag = '✓' if any(p in role.lower() for p in ['lead', 'admin', 'book']) else ' '
    print(f'    {flag} {role!r:35s} {cnt:>10,}')

ds_lead = ds[ds['is_lead']].copy()
print(f'  rows after lead-arranger filter: {len(ds_lead):,} ({len(ds_lead)/len(ds):.1%})')

# === Active period: Tranche_Active_Date → Tranche_Maturity_Date (or +5y) ===
stamp('parsing dates...')
ds_lead['active_start'] = pd.to_datetime(
    ds_lead['Tranche_Active_Date'].fillna(ds_lead['Deal_Active_Date']),
    errors='coerce')
ds_lead['active_end'] = pd.to_datetime(ds_lead['Tranche_Maturity_Date'], errors='coerce')
# Fill missing maturity with start + 5y
missing_mat = ds_lead['active_end'].isna()
ds_lead.loc[missing_mat, 'active_end'] = (
    ds_lead.loc[missing_mat, 'active_start'] + pd.DateOffset(years=5))
ds_lead = ds_lead.dropna(subset=['active_start'])
ds_lead['Tranche_Amount'] = pd.to_numeric(ds_lead['Tranche_Amount'], errors='coerce').fillna(0)
print(f'  rows with valid active period: {len(ds_lead):,}')
print(f'  date range: {ds_lead["active_start"].min().date()} → {ds_lead["active_start"].max().date()}')

# === Compute quarter range ===
YEAR_MIN = max(int(ds_lead['active_start'].dt.year.min()), 1980)
YEAR_MAX = min(int(ds_lead['active_end'].dt.year.max()), 2025)
print(f'  iterating quarters {YEAR_MIN}–{YEAR_MAX}')

def quarter_bounds(year, q):
    starts = [pd.Timestamp(f'{year}-{m:02d}-01') for m in (1, 4, 7, 10)]
    ends = [pd.Timestamp(f'{year}-{m:02d}-{d}') for m, d in
             ((3, 31), (6, 30), (9, 30), (12, 31))]
    return starts[q-1], ends[q-1]

# === Iterate quarters, build edges ===
schema = pa.schema([
    ('gvkey_1', pa.int32()),
    ('gvkey_2', pa.int32()),
    ('year', pa.int16()),
    ('quarter', pa.int8()),
    ('shared_lender_count', pa.int16()),
    ('shared_exposure', pa.float64()),
    ('shared_lead_arranger', pa.bool_()),
])
out_path = EDGES / 'creditor_edges.parquet'
writer = pq.ParquetWriter(str(out_path), schema, compression='zstd')
total_edges = 0
edge_counts_per_year = {}

stamp('building edges per quarter...')
for year in range(YEAR_MIN, YEAR_MAX + 1):
    yr_records = []
    yr_edges = 0
    for q in range(1, 5):
        q_start, q_end = quarter_bounds(year, q)
        active = ds_lead[(ds_lead['active_start'] <= q_end) &
                          (ds_lead['active_end'] >= q_start)]
        if len(active) == 0:
            continue
        # Reduce to (lender, gvkey, total_amount)
        agg = (active.groupby(['Lender_Parent_Id', 'gvkey'])['Tranche_Amount']
                     .sum().reset_index())
        # Per lender, generate pairs
        pair_records = []
        for lid, grp in agg.groupby('Lender_Parent_Id'):
            gvks = grp['gvkey'].values
            amts = grp['Tranche_Amount'].values
            if len(gvks) < 2:
                continue
            # Build pairs via combinations (vectorize via index pairs)
            n = len(gvks)
            # Create index arrays
            ii, jj = np.triu_indices(n, k=1)
            g1 = gvks[ii]
            g2 = gvks[jj]
            # Ensure gvkey_1 < gvkey_2
            mask = g1 > g2
            g1[mask], g2[mask] = g2[mask], g1[mask]
            amt = amts[ii] + amts[jj]
            pair_records.append(pd.DataFrame({
                'gvkey_1': g1, 'gvkey_2': g2,
                'lender_id': lid, 'amount': amt,
            }))
        if not pair_records:
            continue
        df_q = pd.concat(pair_records, ignore_index=True)
        agg_q = df_q.groupby(['gvkey_1', 'gvkey_2'], sort=False).agg(
            shared_lender_count=('lender_id', 'nunique'),
            shared_exposure=('amount', 'sum'),
        ).reset_index()
        agg_q['year'] = year
        agg_q['quarter'] = q
        agg_q['shared_lead_arranger'] = True  # by construction
        yr_records.append(agg_q)
        yr_edges += len(agg_q)
    if yr_records:
        df_year = pd.concat(yr_records, ignore_index=True)
        df_year['gvkey_1'] = df_year['gvkey_1'].astype('int32')
        df_year['gvkey_2'] = df_year['gvkey_2'].astype('int32')
        df_year['shared_lender_count'] = df_year['shared_lender_count'].astype('int16')
        df_year['shared_exposure'] = df_year['shared_exposure'].astype('float64')
        df_year['shared_lead_arranger'] = df_year['shared_lead_arranger'].astype('bool')
        df_year['year'] = df_year['year'].astype('int16')
        df_year['quarter'] = df_year['quarter'].astype('int8')
        df_year = df_year[['gvkey_1', 'gvkey_2', 'year', 'quarter',
                           'shared_lender_count', 'shared_exposure',
                           'shared_lead_arranger']]
        table = pa.Table.from_pandas(df_year, schema=schema, preserve_index=False)
        writer.write_table(table)
        total_edges += len(df_year)
        edge_counts_per_year[year] = yr_edges
        if year % 5 == 0 or year == YEAR_MAX:
            stamp(f'  year {year}: {yr_edges:>9,} edges  (cumulative {total_edges:,})')

writer.close()
stamp(f'wrote {out_path.name} — {total_edges:,} rows, {out_path.stat().st_size/1e6:.1f} MB')

# === Print stats ===
print(f'\n=== Common Creditor Edge Stats ===')
print(f'  total quarterly edges: {total_edges:,}')
print(f'  edges per year (sample):')
for y in (1990, 1995, 2000, 2005, 2007, 2010, 2015, 2020, 2024):
    if y in edge_counts_per_year:
        print(f'    {y}: {edge_counts_per_year[y]:>10,}')

# Spot check — re-load to verify
edges = pd.read_parquet(out_path)
print(f'\n  --- Spot check: Lehman Brothers (gvkey 9529 not standard, finding...) ---')
lehman = fu[fu['conm'].str.contains('LEHMAN', case=False, na=False)]
print(f'    Lehman gvkeys: {lehman[["gvkey","conm"]].head(10).to_string(index=False)}')
print(f'\n  --- Spot check: Enron (gvkey 6127) common-creditor links ---')
enron_edges = edges[(edges['gvkey_1']==6127) | (edges['gvkey_2']==6127)]
enron_pre = enron_edges[enron_edges['year'] < 2002]
print(f'    Enron pre-2002 quarterly edges: {len(enron_pre):,}')
if len(enron_pre):
    print(f'    distinct counterparties: {pd.concat([enron_pre.gvkey_1, enron_pre.gvkey_2]).nunique() - 1}')
print(f'\n  --- Spot check: JPMorgan (gvkey 2968) ---')
jpm_edges = edges[(edges['gvkey_1']==2968) | (edges['gvkey_2']==2968)]
print(f'    JPMorgan total quarterly creditor edges: {len(jpm_edges):,}')

print('\n  --- shared_exposure distribution ---')
exp = edges['shared_exposure']
print(f'    median: ${exp.median()/1e6:,.1f}M, p90: ${exp.quantile(0.9)/1e6:,.1f}M, max: ${exp.max()/1e9:,.1f}B')
print(f'  --- shared_lender_count distribution ---')
print(edges['shared_lender_count'].value_counts().head(8).to_string())
