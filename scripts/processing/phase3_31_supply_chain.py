"""3.1 Supply Chain Edges — WRDS Supply Chain with IDs."""
import time
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

# === Load WRDS Supply Chain with IDs ===
stamp('loading WRDS Supply Chain...')
sc = pd.read_csv(RAW / 'WRDS_linking/wrdsapps_link_supplychain.csv')
print(f'  raw rows: {len(sc):,}')

sc['srcdate'] = pd.to_datetime(sc['srcdate'], errors='coerce')
sc = sc.dropna(subset=['srcdate', 'gvkey', 'cgvkey'])
sc['year'] = sc['srcdate'].dt.year.astype(int)

# Filter both sides in firm_universe
before = len(sc)
sc = sc[sc['gvkey'].astype(int).isin(GVKEYS) & sc['cgvkey'].astype(int).isin(GVKEYS)]
print(f'  after universe filter: {len(sc):,} (-{before-len(sc):,})')

# Rename to canonical edge fields
sc = sc.rename(columns={'gvkey': 'supplier_gvkey', 'cgvkey': 'customer_gvkey'})
sc['supplier_gvkey'] = sc['supplier_gvkey'].astype(int)
sc['customer_gvkey'] = sc['customer_gvkey'].astype(int)

# Deduplicate to one row per (supplier, customer, year), prefer most recent srcdate
sc = sc.sort_values(['supplier_gvkey', 'customer_gvkey', 'year', 'srcdate'])
sc = sc.drop_duplicates(['supplier_gvkey', 'customer_gvkey', 'year'], keep='last')
print(f'  unique (supplier, customer, year) tuples: {len(sc):,}')

# === Compute customer_concentration ===
stamp('computing customer_concentration via firm_years.sale...')
fy = pd.read_parquet(CLEAN / 'firm_years.parquet', columns=['gvkey', 'datadate', 'sale'])
fy['datadate'] = pd.to_datetime(fy['datadate'])
fy['year'] = fy['datadate'].dt.year.astype(int)
fy = fy.rename(columns={'gvkey': 'supplier_gvkey', 'sale': 'supplier_total_sale'})
fy = fy.dropna(subset=['supplier_total_sale'])
fy = fy[['supplier_gvkey', 'year', 'supplier_total_sale']].drop_duplicates(['supplier_gvkey', 'year'])

sc = sc.merge(fy, on=['supplier_gvkey', 'year'], how='left')
# Convert salecs to numeric
sc['salecs'] = pd.to_numeric(sc['salecs'], errors='coerce')
sc['customer_concentration'] = np.where(
    (sc['supplier_total_sale'].notna()) & (sc['supplier_total_sale'] > 0),
    sc['salecs'] / sc['supplier_total_sale'],
    np.nan,
)
print(f'  customer_concentration computed: {sc["customer_concentration"].notna().sum():,} / {len(sc):,} '
      f'({sc["customer_concentration"].notna().mean():.1%})')

# === Compute relationship_duration (consecutive years per pair) ===
stamp('computing relationship_duration...')
sc = sc.sort_values(['supplier_gvkey', 'customer_gvkey', 'year'])
sc['year_diff'] = sc.groupby(['supplier_gvkey', 'customer_gvkey'])['year'].diff()
sc['new_streak'] = ((sc['year_diff'] != 1) | sc['year_diff'].isna()).astype(int)
sc['streak_id'] = sc.groupby(['supplier_gvkey', 'customer_gvkey'])['new_streak'].cumsum()
sc['relationship_duration'] = (
    sc.groupby(['supplier_gvkey', 'customer_gvkey', 'streak_id']).cumcount() + 1
).astype(int)
sc = sc.drop(columns=['year_diff', 'new_streak', 'streak_id'])

# === Expand to quarterly snapshots ===
stamp('expanding to quarterly snapshots...')
quarters = pd.DataFrame({'quarter': [1, 2, 3, 4]})
edges = sc.assign(_key=1).merge(quarters.assign(_key=1), on='_key').drop(columns='_key')

edges = edges[['supplier_gvkey', 'customer_gvkey', 'year', 'quarter',
                'salecs', 'customer_concentration', 'relationship_duration']]
edges['source'] = 'wrds_supply_chain'

# === Save ===
out = EDGES / 'supply_chain_edges.parquet'
edges.to_parquet(out, index=False)
stamp(f'wrote {out.name} — {len(edges):,} rows, {out.stat().st_size/1e6:.1f} MB')

# === Print stats ===
print(f'\n=== Supply Chain Edge Stats ===')
print(f'  total quarterly edges: {len(edges):,}')
print(f'  unique pairs: {edges[["supplier_gvkey","customer_gvkey"]].drop_duplicates().shape[0]:,}')
print(f'  year range: {edges["year"].min()} → {edges["year"].max()}')
print(f'  unique suppliers: {edges["supplier_gvkey"].nunique():,}')
print(f'  unique customers: {edges["customer_gvkey"].nunique():,}')

# Edges per year (sample)
yr_counts = edges.groupby('year').size()
print(f'  edges per year (sample):')
for y in (1995, 2000, 2005, 2010, 2015, 2020, 2024):
    if y in yr_counts.index:
        print(f'    {y}: {yr_counts[y]:>10,}')

# Customer concentration distribution
cc = edges['customer_concentration'].dropna()
print(f'  customer_concentration: median={cc.median():.3f}, p90={cc.quantile(0.9):.3f}, max={cc.max():.3f}')

# Spot checks
print('\n  --- Spot check: Apple suppliers (gvkey 1690) ---')
apple = edges[edges['customer_gvkey']==1690].drop_duplicates(['supplier_gvkey', 'year'])
print(f'    distinct suppliers (Apple as customer) historically: {apple["supplier_gvkey"].nunique():,}')

print('\n  --- Spot check: Enron supply chain (gvkey 6127) ---')
enron_sup = edges[edges['supplier_gvkey']==6127].drop_duplicates(['customer_gvkey', 'year'])
enron_cus = edges[edges['customer_gvkey']==6127].drop_duplicates(['supplier_gvkey', 'year'])
print(f'    Enron as supplier: {enron_sup["customer_gvkey"].nunique()} distinct customers')
print(f'    Enron as customer: {enron_cus["supplier_gvkey"].nunique()} distinct suppliers')
