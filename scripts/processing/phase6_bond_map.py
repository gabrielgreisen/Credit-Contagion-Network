"""6.3 Bond → Firm Mapping (TRACE Master File → gvkey)."""
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw"
CLEAN = PROJECT_ROOT / "data/clean"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# Setup
fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
fu['gvkey'] = fu['gvkey'].astype(int)
GVKEYS = set(fu['gvkey'].tolist())

# CUSIP6 → gvkey map (CUSIP9 first 6 chars = issuer)
fu_cu = fu.dropna(subset=['cusip']).copy()
fu_cu['cusip_str'] = fu_cu['cusip'].astype(str).str.upper().str.strip()
fu_cu['cusip6'] = fu_cu['cusip_str'].str[:6]
# Multiple gvkeys may share CUSIP6 (M&A); keep first
cusip6_to_gvkey = (fu_cu.drop_duplicates('cusip6').set_index('cusip6')['gvkey']
                         .astype(int).to_dict())
print(f'cusip6→gvkey: {len(cusip6_to_gvkey):,}')

# === Load TRACE master ===
stamp('loading TRACE master...')
master = pd.read_csv(RAW / 'trace/trace_standard_master_file.csv',
                      usecols=['cusip_id', 'issuer_nm', 'cpn_rt', 'mtrty_dt',
                               'sub_prdct_type', 'debt_type_cd', 'scrty_type_cd',
                               'scrty_sbtp_cd', 'grade'],
                      dtype={'cusip_id': str, 'cpn_rt': 'float64'},
                      low_memory=False)
print(f'  master rows: {len(master):,}')

# === Filter to corporate bonds ===
stamp('inspecting bond type fields...')
print('  sub_prdct_type top values:')
print(master['sub_prdct_type'].value_counts(dropna=False).head(15).to_string())
print('\n  debt_type_cd top values:')
print(master['debt_type_cd'].value_counts(dropna=False).head(15).to_string())
print('\n  scrty_type_cd top values:')
print(master['scrty_type_cd'].value_counts(dropna=False).head(15).to_string())
print('\n  scrty_sbtp_cd top values:')
print(master['scrty_sbtp_cd'].value_counts(dropna=False).head(20).to_string())

# Filter: keep corporate bonds only
# sub_prdct_type categorizes asset class: 'CORP' for corporate, exclude 'AGCY','MTGE','MUNI','TRSY','144A' as needed
# debt_type_cd has more specific bond types

# Conservative corporate filter:
#   keep sub_prdct_type IN {NaN/empty, 'CORP'} (some are unlabeled but corporate)
#   exclude scrty_type_cd in obvious non-corporate set
EXCLUDE_PRDCT = {'AGCY', 'MTGE', 'MBS', 'CMO', 'ABS', 'MUNI', 'TRSY', 'SOV', 'GOVT'}
EXCLUDE_DEBT = {'AGCY', 'MTGE', 'MBS', 'CMO', 'ABS', 'MUNI', 'TRSY', 'SOV', 'GOVT'}

before = len(master)
mask_keep = (~master['sub_prdct_type'].astype(str).str.upper().isin(EXCLUDE_PRDCT)
              & ~master['debt_type_cd'].astype(str).str.upper().isin(EXCLUDE_DEBT))
corp = master[mask_keep].copy()
print(f'\n  after corporate filter: {len(corp):,} (-{before-len(corp):,})')

# === Drop missing maturity ===
corp['mtrty_dt'] = pd.to_datetime(corp['mtrty_dt'], errors='coerce')
before = len(corp)
corp = corp.dropna(subset=['mtrty_dt'])
print(f'  with maturity date: {len(corp):,} (-{before-len(corp):,})')

# === Map CUSIP6 → gvkey ===
stamp('mapping bond CUSIP6 → firm gvkey...')
corp['cusip_id'] = corp['cusip_id'].astype(str).str.upper().str.strip()
corp['cusip6'] = corp['cusip_id'].str[:6]
corp['gvkey'] = corp['cusip6'].map(cusip6_to_gvkey)
n_via_cusip = corp['gvkey'].notna().sum()
print(f'  matched via CUSIP6: {n_via_cusip:,} / {len(corp):,} ({n_via_cusip/len(corp):.1%})')

# === Bond CRSP Link fallback ===
bcl = pd.read_csv(RAW / 'crsp/Bond_CRSP_link.csv',
                   usecols=['CUSIP', 'PERMNO', 'link_startdt', 'link_enddt'])
bcl = bcl.rename(columns={'CUSIP': 'cusip_id', 'PERMNO': 'permno'})
permno_to_gvkey = (fu.dropna(subset=['permno'])
                       .drop_duplicates('permno').set_index('permno')['gvkey']
                       .astype(int).to_dict())
bcl['gvkey'] = bcl['permno'].map(permno_to_gvkey)
bcl = bcl.dropna(subset=['gvkey'])
bcl_map = (bcl.drop_duplicates('cusip_id').set_index('cusip_id')['gvkey']
              .astype(int).to_dict())
print(f'  Bond CRSP link → gvkey: {len(bcl_map):,} unique cusips mappable')

# Apply Bond CRSP fallback for unmapped corporate bonds
unmapped_mask = corp['gvkey'].isna()
corp.loc[unmapped_mask, 'gvkey'] = corp.loc[unmapped_mask, 'cusip_id'].map(bcl_map)
n_after_bcl = corp['gvkey'].notna().sum()
print(f'  +after Bond CRSP fallback: {n_after_bcl:,} ({n_after_bcl/len(corp):.1%})')

# === Filter to mapped + universe ===
mapped = corp.dropna(subset=['gvkey']).copy()
mapped['gvkey'] = mapped['gvkey'].astype(int)
mapped = mapped[mapped['gvkey'].isin(GVKEYS)]
print(f'  in firm_universe: {len(mapped):,} bonds, {mapped["gvkey"].nunique():,} unique firms')

# === Save ===
out = mapped[['cusip_id', 'gvkey', 'mtrty_dt', 'cpn_rt', 'issuer_nm']].copy()
out_path = CLEAN / 'trace_bond_firm_map.parquet'
out.to_parquet(out_path, index=False)
stamp(f'wrote {out_path.name} — {len(out):,} bonds, {out_path.stat().st_size/1e6:.1f} MB')

print(f'\n=== Summary ===')
print(f'  master rows: {len(master):,}')
print(f'  corporate (post-filter): {len(corp):,}')
print(f'  mapped to firm_universe: {len(mapped):,}')
print(f'  unique firms with bonds: {mapped["gvkey"].nunique():,}')
print(f'  date range of maturities: {mapped["mtrty_dt"].min().date()} → {mapped["mtrty_dt"].max().date()}')
