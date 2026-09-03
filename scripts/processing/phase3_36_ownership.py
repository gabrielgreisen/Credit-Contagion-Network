"""3.6 Ownership Edges — Orbis Subsidiaries (US-filtered, fuzzy-matched)."""
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

# Build name map
def normalize_name(s):
    if not isinstance(s, str): return ''
    s = s.upper()
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'\b(INC|INCORPORATED|CORP|CORPORATION|LTD|LIMITED|LLC|PLC|HOLDINGS|CO|COMPANY|GROUP|HLDG|HLDGS|TRUST|FUND)\b', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s.strip()

fu['conm_norm'] = fu['conm'].apply(normalize_name)
fu_n = fu[fu['conm_norm'].str.len() >= 4].drop_duplicates('conm_norm')
name_to_gvkey = dict(zip(fu_n['conm_norm'], fu_n['gvkey'].astype(int)))
print(f'  name→gvkey map: {len(name_to_gvkey):,}')

# === Stream Orbis, filter to US rows ===
stamp('streaming Orbis (24.7M rows) — filtering to US...')
orbis_path = RAW / 'moodys_orbis/bvd_orbis_large_subsidiaries.csv'
chunks_kept = []
total_rows = 0
total_us = 0
chunk_size = 1_000_000
for chunk_idx, chunk in enumerate(pd.read_csv(orbis_path,
        usecols=['bvdid', 'category_of_company', '_9305', '_9306', 'contact_ctryiso',
                 '_9300', '_9302', 'subsidiaries_isinwocoformatted',
                 '_9308', '_9309', '_9345'],
        dtype={'_9305': str, '_9306': str, '_9300': str, '_9302': str,
               'subsidiaries_isinwocoformatted': str, '_9345': str,
               '_9308': str, '_9309': str},
        chunksize=chunk_size,
        low_memory=False)):
    total_rows += len(chunk)
    # US filter: parent country (contact_ctryiso) OR subsidiary country (_9302) = US
    us_mask = ((chunk['contact_ctryiso'] == 'US') | (chunk['_9302'] == 'US') |
               chunk['bvdid'].astype(str).str.startswith('US'))
    us = chunk[us_mask]
    total_us += len(us)
    if len(us):
        chunks_kept.append(us)
    if chunk_idx % 5 == 0:
        stamp(f'  chunk {chunk_idx}: {total_rows:,} rows scanned, {total_us:,} US so far')

orbis_us = pd.concat(chunks_kept, ignore_index=True) if chunks_kept else pd.DataFrame()
stamp(f'total Orbis rows: {total_rows:,}; US rows: {len(orbis_us):,}')

if orbis_us.empty:
    print('No US Orbis rows — saving empty parquet')
    pd.DataFrame(columns=['parent_gvkey', 'subsidiary_gvkey', 'year', 'quarter', 'ownership_pct']
                 ).to_parquet(EDGES / 'ownership_edges.parquet', index=False)
    raise SystemExit

# === Inspect breakdown ===
print('\n  US row breakdown:')
print(f'    contact_ctryiso==US:    {(orbis_us["contact_ctryiso"]=="US").sum():,}')
print(f'    _9302 (other side)==US: {(orbis_us["_9302"]=="US").sum():,}')
print(f'    bvdid starts US:        {orbis_us["bvdid"].astype(str).str.startswith("US").sum():,}')
print(f'    BOTH contact==US AND _9302==US: '
      f'{((orbis_us["contact_ctryiso"]=="US") & (orbis_us["_9302"]=="US")).sum():,}')

# Determine which side is parent vs subsidiary.
# Per Orbis layout, bvdid = the FOCAL entity, _9300 = related entity name.
# category_of_company: s = subsidiary, m = parent, etc.
# We try both directions: focal as parent, focal as subsidiary
print('\n  category_of_company in US rows:')
print(orbis_us['category_of_company'].value_counts().head(10))

# === Match strategy ===
# Side A: focal entity (bvdid). No direct name in this file → effectively unmatched
#          unless we have an external bvdid → name/gvkey map. We don't.
# Side B: related entity (_9300 name). Match via fuzzy name → gvkey.
#
# Without a parent-name field, we can resolve only ONE side reliably. To get parent→subsidiary
# edges, we'd need both sides resolved. We'll approximate: keep _9300-name-resolved rows where
# _9302 == 'US', AND treat bvdid-starts-with-'US' rows separately.

stamp('matching subsidiary/related-entity name (_9300) → gvkey...')
orbis_us['_9300_norm'] = orbis_us['_9300'].astype(str).apply(normalize_name)
orbis_us['gvkey_9300'] = (orbis_us['_9300_norm']
                            .where(orbis_us['_9300_norm'].str.len() >= 4)
                            .map(name_to_gvkey))
n_resolved = orbis_us['gvkey_9300'].notna().sum()
print(f'  _9300 → gvkey resolved: {n_resolved:,} / {len(orbis_us):,} ({n_resolved/len(orbis_us):.1%})')

# Bvdid → gvkey: we don't have a direct map. Best we can do is to keep rows where
# bvdid contains a US national ID pattern and try matching via cusip-like substrings.
# Per sample, bvdid format like "US*904315628" — the last 9 digits could be a tax ID,
# not a CUSIP. So this is unlikely to match firm_universe.cusip.
# We'll attempt anyway: extract trailing 9 digits and look up in cusip→gvkey map.
stamp('attempting bvdid → gvkey via tail-9 digit lookup against CUSIP9...')
fu_cusip_str = fu['cusip'].astype(str).str.upper().str.strip()
cusip_to_gvkey = dict(zip(fu_cusip_str.dropna(), fu['gvkey'].astype(int)))
bvdid_tail = orbis_us['bvdid'].astype(str).str.replace(r'[^A-Z0-9]', '', regex=True).str[-9:].str.upper()
orbis_us['gvkey_bvdid_cusip'] = bvdid_tail.map(cusip_to_gvkey)
n_bv_resolved = orbis_us['gvkey_bvdid_cusip'].notna().sum()
print(f'  bvdid tail9 → CUSIP9 → gvkey: {n_bv_resolved:,} (likely 0)')

# === Construct edges ===
# For each row where _9300 resolves AND bvdid resolves, we have a parent/subsidiary pair.
# Direction: per Orbis convention, bvdid is the focal "parent record" with subsidiaries.
# So bvdid → _9300 = parent → subsidiary.
both_resolved = orbis_us[orbis_us['gvkey_9300'].notna() & orbis_us['gvkey_bvdid_cusip'].notna()].copy()
print(f'\n  rows with BOTH parent and subsidiary resolved: {len(both_resolved):,}')

# Fallback: when only _9300 is resolved, we have a known subsidiary but unknown parent.
# These rows are useful only if we can identify the parent through some other means.
# Without a bvdid→gvkey map, those rows must be discarded (cannot form edge).

# Build edge dataframe
if len(both_resolved):
    edges_static = pd.DataFrame({
        'parent_gvkey': both_resolved['gvkey_bvdid_cusip'].astype(int).values,
        'subsidiary_gvkey': both_resolved['gvkey_9300'].astype(int).values,
        'ownership_pct': pd.to_numeric(both_resolved['_9308'].str.rstrip('%').replace('>75.00', '85'),
                                         errors='coerce'),
    })
    edges_static = edges_static.dropna(subset=['parent_gvkey', 'subsidiary_gvkey'])
    edges_static = edges_static[edges_static['parent_gvkey'] != edges_static['subsidiary_gvkey']]
    edges_static = edges_static.drop_duplicates(['parent_gvkey', 'subsidiary_gvkey'])
    print(f'  unique (parent, subsidiary) edges: {len(edges_static):,}')
else:
    edges_static = pd.DataFrame(columns=['parent_gvkey', 'subsidiary_gvkey', 'ownership_pct'])

# === Static snapshot: replicate across ALL quarters in our universe ===
qf = pd.read_parquet(CLEAN / 'node_features_quarterly.parquet',
                     columns=['fyearq', 'fqtr'])
qrange = qf[['fyearq', 'fqtr']].drop_duplicates().sort_values(['fyearq', 'fqtr'])
qrange.columns = ['year', 'quarter']
print(f'  expanding to {len(qrange):,} (year, quarter) snapshots...')

if len(edges_static):
    edges = edges_static.assign(_k=1).merge(qrange.assign(_k=1), on='_k').drop(columns='_k')
else:
    edges = pd.DataFrame(columns=['parent_gvkey', 'subsidiary_gvkey', 'year', 'quarter', 'ownership_pct'])

edges = edges[['parent_gvkey', 'subsidiary_gvkey', 'year', 'quarter', 'ownership_pct']]
out_path = EDGES / 'ownership_edges.parquet'
edges.to_parquet(out_path, index=False)
stamp(f'wrote {out_path.name} — {len(edges):,} rows, {out_path.stat().st_size/1e6:.2f} MB')

# === Stats ===
print(f'\n=== Ownership Edge Stats ===')
print(f'  total quarterly edges: {len(edges):,}')
if len(edges):
    print(f'  unique (parent, subsidiary) pairs: '
          f'{edges[["parent_gvkey","subsidiary_gvkey"]].drop_duplicates().shape[0]:,}')
    print(f'  unique parents: {edges["parent_gvkey"].nunique():,}')
    print(f'  unique subsidiaries: {edges["subsidiary_gvkey"].nunique():,}')
    print(f'  ownership_pct distribution:')
    pct = edges['ownership_pct'].dropna()
    if len(pct):
        print(f'    median: {pct.median():.1f}%, p25: {pct.quantile(0.25):.1f}%, p75: {pct.quantile(0.75):.1f}%')
print(f'\n  ⚠ DATA CAVEAT: Orbis ownership match rate is low. The "subsidiaries_large" file is')
print(f'  global (24.7M rows, ~95% non-US), and has no canonical parent name field. We can only')
print(f'  resolve subsidiary side via name match; parent side requires bvdid→gvkey which we lack.')
print(f'  Result is an aggressively conservative US-US ownership layer — expected to be very small.')
