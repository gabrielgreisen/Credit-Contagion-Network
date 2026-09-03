"""3.7 Board Interlock Edges — BoardEx Networks + Analytics."""
import time
import re
from pathlib import Path
import numpy as np
import pandas as pd

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

# Map: cusip(9) → gvkey, ticker → gvkey (handle dupes by taking first)
fu['cusip_str'] = fu['cusip'].astype(str).str.upper().str.strip()
fu_cusip = fu.dropna(subset=['cusip']).drop_duplicates('cusip_str')
cusip_to_gvkey = dict(zip(fu_cusip['cusip_str'], fu_cusip['gvkey'].astype(int)))
print(f'  cusip→gvkey map: {len(cusip_to_gvkey):,}')

fu_tic = fu.dropna(subset=['tic']).copy()
fu_tic['tic_norm'] = fu_tic['tic'].astype(str).str.upper().str.strip()
fu_tic = fu_tic.drop_duplicates('tic_norm')
ticker_to_gvkey = dict(zip(fu_tic['tic_norm'], fu_tic['gvkey'].astype(int)))
print(f'  ticker→gvkey map: {len(ticker_to_gvkey):,}')

# Name→gvkey for fallback fuzzy
def normalize_name(s):
    if not isinstance(s, str): return ''
    s = s.upper()
    # Strip parenthetical
    s = re.sub(r'\(.*?\)', '', s)
    # Common suffixes
    s = re.sub(r'\b(INC|CORP|LTD|LLC|PLC|HOLDINGS|CO|COMPANY|GROUP|HLDG|HLDGS)\b', '', s)
    # Remove non-alphanumeric
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s.strip()

fu['conm_norm'] = fu['conm'].apply(normalize_name)
fu_name = fu[fu['conm_norm'] != ''].drop_duplicates('conm_norm')
name_to_gvkey = dict(zip(fu_name['conm_norm'], fu_name['gvkey'].astype(int)))
print(f'  name→gvkey map: {len(name_to_gvkey):,}')

# === Build companyid → gvkey from BoardEx analytics ===
stamp('loading BoardEx analytics for ID mapping...')
analytics_path = RAW / 'bordex/bordex_organization_analytics.csv'
# Stream — we only need (companyid, isin, ticker, boardname/companyname)
ana = pd.read_csv(analytics_path,
                   usecols=['rowtype', 'companyid', 'boardid', 'isin', 'ticker', 'boardname'],
                   dtype={'companyid': 'Int64', 'boardid': 'Int64'})
print(f'  analytics rows: {len(ana):,}')

# Build companyid → identifiers via boardid (rows have boardid==companyid often)
# We collapse to (boardid, isin, ticker, boardname) since boardid is the more stable per-firm key
ana_firms = (ana.dropna(subset=['boardid'])
                 .drop_duplicates('boardid')
                 [['boardid', 'isin', 'ticker', 'boardname']])
print(f'  unique boardids: {len(ana_firms):,}')

# ISIN → cusip(9): ISIN format = CC + national-id (9 for US) + check digit (12 chars total)
# So ISIN[2:11] is CUSIP9 for US. For other countries no match.
ana_firms['isin_cusip'] = ana_firms['isin'].astype(str).str[2:11].str.upper()
ana_firms['gvkey_via_cusip'] = ana_firms['isin_cusip'].map(cusip_to_gvkey).astype('Int64')

ana_firms['ticker_norm'] = ana_firms['ticker'].astype(str).str.upper().str.strip()
ana_firms['gvkey_via_ticker'] = ana_firms['ticker_norm'].map(ticker_to_gvkey).astype('Int64')

ana_firms['name_norm'] = ana_firms['boardname'].apply(normalize_name)
ana_firms['gvkey_via_name'] = ana_firms['name_norm'].map(name_to_gvkey).astype('Int64')

# Coalesce: cusip > ticker > name
ana_firms['gvkey'] = (ana_firms['gvkey_via_cusip']
                       .fillna(ana_firms['gvkey_via_ticker'])
                       .fillna(ana_firms['gvkey_via_name']))

n_total = len(ana_firms)
n_via_cusip = ana_firms['gvkey_via_cusip'].notna().sum()
n_via_tic   = ana_firms['gvkey_via_ticker'].notna().sum() - ana_firms['gvkey_via_cusip'].notna().sum()
n_via_name = ana_firms['gvkey'].notna().sum() - n_via_cusip - n_via_tic
print(f'  boardid→gvkey resolution:')
print(f'    via CUSIP (ISIN[2:11]):    {n_via_cusip:>6,}')
print(f'    via ticker (after CUSIP):  {n_via_tic:>6,}')
print(f'    via name (after CUSIP+tic):{n_via_name:>6,}')
print(f'    total resolved:            {ana_firms["gvkey"].notna().sum():>6,} / {n_total:,} ({ana_firms["gvkey"].notna().mean():.1%})')

ana_firms = ana_firms.dropna(subset=['gvkey'])
ana_firms['gvkey'] = ana_firms['gvkey'].astype(int)
boardid_to_gvkey = dict(zip(ana_firms['boardid'].astype(int), ana_firms['gvkey']))
print(f'  boardid→gvkey dict: {len(boardid_to_gvkey):,}')

# === Load networks file in chunks; filter ===
stamp('loading BoardEx networks (in chunks)...')
nets_path = RAW / 'bordex/bordex_networks_associations.csv'

EXEC_PATTERN = re.compile(r'CEO|CFO|COO|Chief Executive|Chief Financial|Chief Operating|Chairman|President',
                           re.IGNORECASE)

def is_exec(role):
    if not isinstance(role, str):
        return False
    return bool(EXEC_PATTERN.search(role))

# Process in chunks
chunks = []
total_chunks = 0
total_after_filter = 0
for chunk in pd.read_csv(nets_path, chunksize=500_000,
                          usecols=['companyid', 'boardid', 'directorid',
                                   'overlapyearstart_int', 'overlapyearend_int',
                                   'roletitle', 'associatedrole', 'conncompanyorgtype']):
    total_chunks += len(chunk)
    chunk = chunk.dropna(subset=['companyid', 'boardid', 'directorid',
                                  'overlapyearstart_int', 'overlapyearend_int'])
    # Map both sides to gvkey
    chunk['boardid'] = chunk['boardid'].astype(int)
    chunk['companyid'] = chunk['companyid'].astype(int)
    chunk['gvkey_board'] = chunk['boardid'].map(boardid_to_gvkey)
    chunk['gvkey_conn'] = chunk['companyid'].map(boardid_to_gvkey)
    # Both must resolve and be different
    chunk = chunk.dropna(subset=['gvkey_board', 'gvkey_conn'])
    chunk = chunk[chunk['gvkey_board'] != chunk['gvkey_conn']]
    chunk['gvkey_board'] = chunk['gvkey_board'].astype(int)
    chunk['gvkey_conn'] = chunk['gvkey_conn'].astype(int)
    # Both must be in firm_universe (already true via map but double-check)
    chunk = chunk[chunk['gvkey_board'].isin(GVKEYS) & chunk['gvkey_conn'].isin(GVKEYS)]
    chunk['is_exec'] = (chunk['roletitle'].apply(is_exec) |
                        chunk['associatedrole'].apply(is_exec))
    total_after_filter += len(chunk)
    chunks.append(chunk)
print(f'  total raw rows: {total_chunks:,}')
print(f'  rows w/ both gvkey + in universe: {total_after_filter:,}')

if not chunks:
    print('NO BOARD INTERLOCK ROWS — skipping')
    raise SystemExit

nets = pd.concat(chunks, ignore_index=True)
del chunks

# === Construct edges ===
stamp('constructing pair-year edges...')
# Order gvkeys for undirected key
nets['gvkey_1'] = np.minimum(nets['gvkey_board'], nets['gvkey_conn'])
nets['gvkey_2'] = np.maximum(nets['gvkey_board'], nets['gvkey_conn'])
nets['ystart'] = nets['overlapyearstart_int'].astype(int)
nets['yend'] = nets['overlapyearend_int'].astype(int)

# Each row contributes (gvkey1, gvkey2, year, directorid, is_exec) for year in [ystart, yend]
# Expand using a year list; cap range to plausible
nets['ystart'] = nets['ystart'].clip(lower=1980, upper=2026)
nets['yend'] = nets['yend'].clip(lower=1980, upper=2026)
nets = nets[nets['yend'] >= nets['ystart']]

# Vectorized year expansion via repeat
n_rows = len(nets)
year_lengths = (nets['yend'] - nets['ystart'] + 1).values
print(f'  expanding {n_rows:,} association rows × avg {year_lengths.mean():.1f} years each = '
      f'{int(year_lengths.sum()):,} year-rows')

expanded = pd.DataFrame({
    'gvkey_1': np.repeat(nets['gvkey_1'].values, year_lengths),
    'gvkey_2': np.repeat(nets['gvkey_2'].values, year_lengths),
    'directorid': np.repeat(nets['directorid'].astype(int).values, year_lengths),
    'is_exec': np.repeat(nets['is_exec'].values, year_lengths),
    'year': np.concatenate([np.arange(s, e+1) for s, e in zip(nets['ystart'], nets['yend'])]),
})
print(f'  expanded rows: {len(expanded):,}')

# Aggregate per (gvkey_1, gvkey_2, year): distinct directors + any exec
stamp('aggregating per (pair, year)...')
agg = expanded.groupby(['gvkey_1', 'gvkey_2', 'year']).agg(
    shared_director_count=('directorid', 'nunique'),
    shared_executive=('is_exec', 'any'),
).reset_index()
print(f'  unique (pair, year) rows: {len(agg):,}')

# === Expand year → quarter ===
stamp('expanding year → quarters...')
quarters = pd.DataFrame({'quarter': [1, 2, 3, 4]})
edges = agg.assign(_k=1).merge(quarters.assign(_k=1), on='_k').drop(columns='_k')
edges = edges[['gvkey_1', 'gvkey_2', 'year', 'quarter',
                'shared_director_count', 'shared_executive']]
edges['gvkey_1'] = edges['gvkey_1'].astype('int32')
edges['gvkey_2'] = edges['gvkey_2'].astype('int32')
edges['year'] = edges['year'].astype('int16')
edges['quarter'] = edges['quarter'].astype('int8')
edges['shared_director_count'] = edges['shared_director_count'].astype('int16')

out_path = EDGES / 'board_interlock_edges.parquet'
edges.to_parquet(out_path, index=False)
stamp(f'wrote {out_path.name} — {len(edges):,} rows, {out_path.stat().st_size/1e6:.1f} MB')

# === Stats ===
print(f'\n=== Board Interlock Edge Stats ===')
print(f'  total quarterly edges: {len(edges):,}')
print(f'  unique pairs: {edges[["gvkey_1","gvkey_2"]].drop_duplicates().shape[0]:,}')
print(f'  year range: {edges["year"].min()} → {edges["year"].max()}')
print(f'  edges per year (sample):')
yc = edges.groupby('year').size()
for y in (1995, 2000, 2005, 2010, 2015, 2020):
    if y in yc.index:
        print(f'    {y}: {yc[y]:>10,}')
print(f'  shared_director_count distribution:')
print(edges['shared_director_count'].value_counts().head(8).to_string())
print(f'  shared_executive: {edges["shared_executive"].sum():,} edges have exec link '
      f'({edges["shared_executive"].mean():.1%})')

# Spot checks
print(f'\n  --- Spot check: Lehman Brothers (gvkey 30128) ---')
le = edges[(edges['gvkey_1']==30128) | (edges['gvkey_2']==30128)]
le_pre = le[le['year'] < 2009]
print(f'    pre-2009 quarterly edges: {len(le_pre):,}')
print(f'    distinct counterparties (all): {pd.concat([le.gvkey_1, le.gvkey_2]).nunique() - 1}')

print(f'\n  --- Spot check: Apple (gvkey 1690) ---')
ap = edges[(edges['gvkey_1']==1690) | (edges['gvkey_2']==1690)]
print(f'    total quarterly edges: {len(ap):,}')
print(f'    distinct counterparties: {pd.concat([ap.gvkey_1, ap.gvkey_2]).nunique() - 1}')
