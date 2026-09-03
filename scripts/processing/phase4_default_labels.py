"""Phase 4: Default Labels — process LoPucki, Compustat delistings, build labels."""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw"
CLEAN = PROJECT_ROOT / "data/clean"
LOPUCKI_CSV = (RAW / 'lopucki' /
               'Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023' /
               'Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv')

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# Setup
fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
fu['gvkey'] = fu['gvkey'].astype(int)
GVKEYS = set(fu['gvkey'].tolist())
gvkey_to_name = dict(zip(fu['gvkey'], fu['conm']))
stamp(f'firm_universe: {len(GVKEYS):,} gvkeys')

# Maps for fallback matching
fu_cik = fu.dropna(subset=['cik']).copy()
fu_cik['cik'] = fu_cik['cik'].astype('Int64').astype(str)
cik_to_gvkey = dict(zip(fu_cik['cik'], fu_cik['gvkey']))
print(f'  CIK→gvkey: {len(cik_to_gvkey):,}')

def normalize_name(s):
    if not isinstance(s, str): return ''
    s = s.upper()
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[,\.]', ' ', s)
    s = re.sub(r'\b(INC|INCORPORATED|CORP|CORPORATION|LTD|LIMITED|LLC|PLC|HOLDINGS|HOLDING|CO|COMPANY|GROUP|HLDG|HLDGS|TRUST|FUND|INTL|INTERNATIONAL|US|USA)\b', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s.strip()

fu['conm_norm'] = fu['conm'].apply(normalize_name)
fu_n = fu[fu['conm_norm'].str.len() >= 4].drop_duplicates('conm_norm')
name_to_gvkey = dict(zip(fu_n['conm_norm'], fu_n['gvkey']))
print(f'  name→gvkey: {len(name_to_gvkey):,}')

# ============================================================
# 4.1 Process LoPucki
# ============================================================
stamp('\n=== 4.1 LoPucki ===')
lo = pd.read_csv(LOPUCKI_CSV, low_memory=False, encoding='latin-1')
print(f'  raw rows: {len(lo):,}')
lo['DateFiled'] = pd.to_datetime(lo['DateFiled'], errors='coerce')
lo['DateDisposed'] = pd.to_datetime(lo['DateDisposed'], errors='coerce')
lo['Chapter'] = lo['Chapter'].astype(str)

# Match strategy: GvkeyBefore (primary) → CIK → fuzzy name
lo['matched_gvkey'] = pd.NA
lo['match_source'] = pd.NA

# Stage 1: GvkeyBefore
mask = lo['GvkeyBefore'].notna() & lo['GvkeyBefore'].astype('Int64').isin(GVKEYS)
lo.loc[mask, 'matched_gvkey'] = lo.loc[mask, 'GvkeyBefore'].astype(int)
lo.loc[mask, 'match_source'] = 'gvkey_before'
n1 = mask.sum()
print(f'  matched via GvkeyBefore: {n1:,} ({n1/len(lo):.1%})')

# Stage 2: CIK fallback
mask2 = lo['matched_gvkey'].isna() & lo['CikBefore'].notna()
lo.loc[mask2, 'cik_str'] = lo.loc[mask2, 'CikBefore'].astype('Int64').astype(str)
lo.loc[mask2, 'matched_gvkey'] = lo.loc[mask2, 'cik_str'].map(cik_to_gvkey)
new_match = mask2 & lo['matched_gvkey'].notna()
lo.loc[new_match, 'match_source'] = 'cik'
n2 = new_match.sum()
print(f'  +matched via CIK fallback:  {n2:,}')

# Stage 3: fuzzy name
mask3 = lo['matched_gvkey'].isna()
lo.loc[mask3, 'name_norm'] = lo.loc[mask3, 'NameCorp'].apply(normalize_name)
lo.loc[mask3, 'matched_gvkey'] = (lo.loc[mask3, 'name_norm']
                                     .where(lo.loc[mask3, 'name_norm'].str.len() >= 4)
                                     .map(name_to_gvkey))
new_match = mask3 & lo['matched_gvkey'].notna()
lo.loc[new_match, 'match_source'] = 'name'
n3 = new_match.sum()
print(f'  +matched via name (norm):   {n3:,}')

n_total_matched = lo['matched_gvkey'].notna().sum()
n_unmatched = len(lo) - n_total_matched
print(f'  total matched: {n_total_matched:,} / {len(lo):,} ({n_total_matched/len(lo):.1%})')
print(f'  unmatched: {n_unmatched:,}')

# Map chapter to default_type
def chapter_to_type(ch):
    s = str(ch).strip().lower()
    if s == '7': return 'bankruptcy_ch7'
    if s == '11': return 'bankruptcy_ch11'
    return f'bankruptcy_other'  # 'no order for relief' etc

lo['default_type_lopucki'] = lo['Chapter'].apply(chapter_to_type)

# Build LoPucki match table
lopucki_matched = lo[lo['matched_gvkey'].notna()].copy()
lopucki_matched['matched_gvkey'] = lopucki_matched['matched_gvkey'].astype(int)
lopucki_matched['outcome'] = lopucki_matched['Disposition']
lopucki_matched['assets_at_filing'] = lopucki_matched['AssetsPetition']
lopucki_matched = lopucki_matched.rename(columns={
    'matched_gvkey': 'gvkey',
    'DateFiled': 'lopucki_filing_date',
    'Chapter': 'chapter',
})
lopucki_matched = lopucki_matched[['gvkey', 'lopucki_filing_date', 'chapter',
                                    'outcome', 'assets_at_filing',
                                    'default_type_lopucki', 'NameCorp', 'match_source']]

# Save match report
lo['matched'] = lo['matched_gvkey'].notna()
match_report = lo[['NameCorp', 'CikBefore', 'GvkeyBefore', 'DateFiled', 'Chapter',
                    'Disposition', 'AssetsPetition', 'matched_gvkey', 'matched',
                    'match_source']]
match_report_path = CLEAN / 'lopucki_match_report.csv'
match_report.to_csv(match_report_path, index=False)
print(f'  wrote {match_report_path.name}')

# Save unmatched separately
unmatched = lo[~lo['matched']]
unmatched_path = CLEAN / 'lopucki_unmatched.csv'
unmatched[['NameCorp', 'CikBefore', 'GvkeyBefore', 'DateFiled', 'Chapter',
            'Disposition']].to_csv(unmatched_path, index=False)
print(f'  wrote {unmatched_path.name} ({len(unmatched)} rows)')

# Dedup LoPucki to one row per gvkey (keep earliest filing — first bankruptcy)
lopucki_matched = (lopucki_matched.sort_values('lopucki_filing_date')
                                    .drop_duplicates('gvkey', keep='first'))
print(f'  unique LoPucki gvkeys: {len(lopucki_matched):,}')

# ============================================================
# 4.2 Compustat Delistings
# ============================================================
stamp('\n=== 4.2 Compustat Delistings ===')
fu_del = fu.dropna(subset=['dlrsn', 'dldte']).copy()
fu_del['dlrsn'] = fu_del['dlrsn'].astype(int)
fu_del['dldte'] = pd.to_datetime(fu_del['dldte'], errors='coerce')
fu_del = fu_del.dropna(subset=['dldte'])
print(f'  firms with dlrsn + dldte: {len(fu_del):,}')

bk_2 = fu_del[fu_del['dlrsn'] == 2].copy()
bk_3 = fu_del[fu_del['dlrsn'] == 3].copy()
print(f'  dlrsn==2 (bankruptcy):  {len(bk_2):,}')
print(f'  dlrsn==3 (liquidation): {len(bk_3):,}')

# Map to default_type
fu_del = fu_del[fu_del['dlrsn'].isin([2, 3])].copy()
fu_del['default_type_compustat'] = fu_del['dlrsn'].map({2: 'bankruptcy_compustat', 3: 'liquidation'})
fu_del = fu_del[['gvkey', 'dldte', 'dlrsn', 'default_type_compustat']].rename(
    columns={'dldte': 'compustat_dldte'})

# ============================================================
# 4.3 Combine & Deduplicate
# ============================================================
stamp('\n=== 4.3 Combine ===')
combined = lopucki_matched.merge(fu_del, on='gvkey', how='outer')
print(f'  combined rows (unique gvkeys with default events): {len(combined):,}')

# Determine source
combined['has_lopucki'] = combined['lopucki_filing_date'].notna()
combined['has_compustat'] = combined['compustat_dldte'].notna()
combined['source'] = np.where(
    combined['has_lopucki'] & combined['has_compustat'], 'both',
    np.where(combined['has_lopucki'], 'lopucki', 'compustat'))

# Pick earliest date as default_date; prefer LoPucki chapter / outcome when both
combined['default_date'] = combined[['lopucki_filing_date', 'compustat_dldte']].min(axis=1)
combined['default_type'] = np.where(
    combined['has_lopucki'],
    combined['default_type_lopucki'],
    combined['default_type_compustat'])

# Cross-reference: date discrepancies for 'both' source
both = combined[combined['source'] == 'both'].copy()
both['date_diff_days'] = (both['compustat_dldte'] - both['lopucki_filing_date']).dt.days
print(f'  cross-source firms (in both LoPucki and Compustat): {len(both):,}')
if len(both):
    print(f'    median |date_diff|: {both["date_diff_days"].abs().median():.0f} days')
    print(f'    median date_diff (signed):    {both["date_diff_days"].median():.0f} days '
          f'(positive = Compustat after LoPucki)')
    big_diff = both[both['date_diff_days'].abs() > 365]
    print(f'    pairs with |diff| > 365 days: {len(big_diff):,}')

# Final default_events table
default_events = combined[['gvkey', 'default_date', 'default_type',
                           'chapter', 'outcome', 'source',
                           'lopucki_filing_date', 'compustat_dldte', 'assets_at_filing',
                           'NameCorp']].copy()
default_events = default_events.dropna(subset=['default_date'])
default_events['gvkey'] = default_events['gvkey'].astype(int)
default_events = default_events.sort_values('default_date').reset_index(drop=True)

out_events = CLEAN / 'default_events.parquet'
default_events.to_parquet(out_events, index=False)
print(f'  wrote {out_events.name} — {len(default_events):,} unique default events')

# Counts by source / type / decade
print(f'\n  by source: {default_events["source"].value_counts().to_dict()}')
print(f'  by type:')
for t, n in default_events['default_type'].value_counts().items():
    print(f'    {t:25s} {n:>5,}')
default_events['decade'] = (default_events['default_date'].dt.year // 10 * 10).astype(int)
print(f'  by decade:')
for d, n in default_events.groupby('decade').size().items():
    print(f'    {d}s: {n:>5,}')

# ============================================================
# 4.4 Annual Default Labels
# ============================================================
stamp('\n=== 4.4 Annual Default Labels ===')
fy = pd.read_parquet(CLEAN / 'firm_years.parquet', columns=['gvkey', 'datadate'])
fy['datadate'] = pd.to_datetime(fy['datadate'])
fy['gvkey'] = fy['gvkey'].astype(int)
fy['fyear'] = fy['datadate'].dt.year.astype(int)
print(f'  annual rows: {len(fy):,}')

# Build gvkey → earliest default_date map
gvkey_to_default = (default_events.groupby('gvkey')['default_date'].min()).to_dict()
fy['default_date'] = fy['gvkey'].map(gvkey_to_default)
fy['days_to_default'] = (fy['default_date'] - fy['datadate']).dt.days

# Drop firm-years AFTER default
n_before = len(fy)
fy_kept = fy[(fy['default_date'].isna()) | (fy['days_to_default'] >= 0)].copy()
n_dropped = n_before - len(fy_kept)
print(f'  rows dropped (datadate after default_date): {n_dropped:,}')

# Labels: strictly AFTER datadate (days_to_default > 0)
# For a firm that defaults on datadate's same day, treat as same-period (do NOT label as future)
fy_kept['default_next_1y'] = ((fy_kept['days_to_default'] > 0) &
                                (fy_kept['days_to_default'] <= 365)).astype('int8')
fy_kept['default_next_2y'] = ((fy_kept['days_to_default'] > 0) &
                                (fy_kept['days_to_default'] <= 730)).astype('int8')

# Save
fy_out = fy_kept[['gvkey', 'datadate', 'fyear', 'default_next_1y', 'default_next_2y']]
out_annual = CLEAN / 'default_labels_annual.parquet'
fy_out.to_parquet(out_annual, index=False)
print(f'  wrote {out_annual.name} — {len(fy_out):,} rows')
print(f'  positive 1y: {fy_out["default_next_1y"].sum():,} ({fy_out["default_next_1y"].mean():.4%})')
print(f'  positive 2y: {fy_out["default_next_2y"].sum():,} ({fy_out["default_next_2y"].mean():.4%})')

# ============================================================
# 4.5 Quarterly Default Labels
# ============================================================
stamp('\n=== 4.5 Quarterly Default Labels ===')
qf = pd.read_parquet(CLEAN / 'node_features_quarterly.parquet',
                     columns=['gvkey', 'datadate'])
qf['datadate'] = pd.to_datetime(qf['datadate'])
qf['gvkey'] = qf['gvkey'].astype(int)
print(f'  quarterly rows: {len(qf):,}')

qf['default_date'] = qf['gvkey'].map(gvkey_to_default)
qf['days_to_default'] = (qf['default_date'] - qf['datadate']).dt.days

n_before = len(qf)
qf_kept = qf[(qf['default_date'].isna()) | (qf['days_to_default'] >= 0)].copy()
n_dropped = n_before - len(qf_kept)
print(f'  rows dropped (datadate after default_date): {n_dropped:,}')

qf_kept['default_next_1q'] = ((qf_kept['days_to_default'] > 0) &
                                (qf_kept['days_to_default'] <= 90)).astype('int8')
qf_kept['default_next_4q'] = ((qf_kept['days_to_default'] > 0) &
                                (qf_kept['days_to_default'] <= 365)).astype('int8')
qf_kept['default_next_8q'] = ((qf_kept['days_to_default'] > 0) &
                                (qf_kept['days_to_default'] <= 730)).astype('int8')

qf_out = qf_kept[['gvkey', 'datadate', 'default_next_1q',
                   'default_next_4q', 'default_next_8q']]
out_quarterly = CLEAN / 'default_labels_quarterly.parquet'
qf_out.to_parquet(out_quarterly, index=False)
print(f'  wrote {out_quarterly.name} — {len(qf_out):,} rows')
print(f'  positive 1q: {qf_out["default_next_1q"].sum():,} ({qf_out["default_next_1q"].mean():.5%})')
print(f'  positive 4q: {qf_out["default_next_4q"].sum():,} ({qf_out["default_next_4q"].mean():.5%})')
print(f'  positive 8q: {qf_out["default_next_8q"].sum():,} ({qf_out["default_next_8q"].mean():.5%})')

# ============================================================
# 4.6 Class Balance Assessment
# ============================================================
stamp('\n=== 4.6 Class Balance ===')
print('\n  Annual default rates by year:')
print(f'  {"year":>6s} {"obs":>8s} {"def_1y":>7s} {"rate_1y":>8s} {"rate_2y":>8s}')
fy_annual = fy_out.copy()
yr = fy_annual.groupby('fyear').agg(
    obs=('gvkey', 'count'),
    def_1y=('default_next_1y', 'sum'),
    def_2y=('default_next_2y', 'sum'),
).reset_index()
yr['rate_1y'] = yr['def_1y'] / yr['obs'] * 100
yr['rate_2y'] = yr['def_2y'] / yr['obs'] * 100

for _, row in yr[(yr['fyear'] >= 1980) & (yr['fyear'] <= 2024)].iterrows():
    print(f'  {int(row["fyear"]):>6d} {int(row["obs"]):>8,} {int(row["def_1y"]):>7,} '
          f'{row["rate_1y"]:>7.3f}% {row["rate_2y"]:>7.3f}%')

# Crisis year flags
crisis = yr[yr['fyear'].isin([2001, 2002, 2008, 2009, 2020])][['fyear', 'def_1y', 'rate_1y']]
print(f'\n  Crisis years (1y default rate):')
for _, row in crisis.iterrows():
    print(f'    {int(row["fyear"])}: {int(row["def_1y"]):>3,} defaults '
          f'({row["rate_1y"]:.3f}%)')

# ============================================================
# 4.7 Validation
# ============================================================
stamp('\n=== 4.7 Validation ===')

# Spot checks
def check(label, gvkey, year, expected_1y, expected_2y):
    row = fy_out[(fy_out['gvkey'] == gvkey) & (fy_out['fyear'] == year)]
    if row.empty:
        print(f'  {label}: NO ROW (gvkey={gvkey}, fyear={year})')
        return False
    r = row.iloc[0]
    actual_1y = int(r['default_next_1y'])
    actual_2y = int(r['default_next_2y'])
    ok = actual_1y == expected_1y and actual_2y == expected_2y
    flag = '✓' if ok else '✗'
    print(f'  {flag} {label} (gvkey={gvkey}, fyear={year}): 1y={actual_1y} (exp {expected_1y}), '
          f'2y={actual_2y} (exp {expected_2y})')
    return ok

print('\n  spot checks:')
checks = [
    ('Enron fy2000',    6127,  2000, 1, 1),
    ('Enron fy1999',    6127,  1999, 0, 1),
    ('Lehman fy2007',   30128, 2007, 1, 1),
    ('WorldCom fy2001', 143972, 2001, 1, 1),
    ('GM fy2008',       5073,  2008, 1, 1),
    ('Apple fy2023',    1690,  2023, 0, 0),
]
all_ok = True
for label, gvkey, year, e1, e2 in checks:
    if not check(label, gvkey, year, e1, e2):
        all_ok = False
print(f'\n  spot-check verdict: {"ALL PASS" if all_ok else "SOME FAILED"}')

# Temporal consistency
print('\n  temporal consistency:')
temporal_violations = fy_out[(fy_out['default_next_1y'] == 1)].merge(
    default_events[['gvkey', 'default_date']], on='gvkey', how='left')
n_bad = (temporal_violations['default_date'] <= temporal_violations['datadate']).sum()
print(f'    rows where default_date <= datadate but label=1: {n_bad} (expect 0)')

# Coverage of default events
print('\n  every default event produces ≥1 positive label?')
event_gvkeys = set(default_events['gvkey'])
positive_gvkeys_annual = set(fy_out[fy_out['default_next_2y'] == 1]['gvkey'])
positive_gvkeys_quarterly = set(qf_out[qf_out['default_next_8q'] == 1]['gvkey'])
missing_annual = event_gvkeys - positive_gvkeys_annual
missing_quarterly = event_gvkeys - positive_gvkeys_quarterly
print(f'    events not covered by ≥1 annual 2y label: {len(missing_annual)} '
      f'(of {len(event_gvkeys)} events; explainable by firms with no firm-years '
      f'in the 2-year pre-default window)')
print(f'    events not covered by ≥1 quarterly 8q label: {len(missing_quarterly)}')

# Cross-references
print('\n  cross-reference (LoPucki vs Compustat):')
print(f'    LoPucki only (Compustat has no/different deletion code): '
      f'{(default_events["source"] == "lopucki").sum()}')
print(f'    Compustat only (smaller bankruptcies under LoPucki size threshold): '
      f'{(default_events["source"] == "compustat").sum()}')
print(f'    Both sources: {(default_events["source"] == "both").sum()}')

# ============================================================
# Save phase4_summary.json
# ============================================================
stamp('\n=== save phase4_summary.json ===')
summary = {
    'lopucki': {
        'raw_rows': int(len(lo)),
        'matched_total': int(n_total_matched),
        'matched_via_gvkey_before': int(n1),
        'matched_via_cik': int(n2),
        'matched_via_name': int(n3),
        'unmatched': int(n_unmatched),
        'match_rate_pct': float(round(n_total_matched / len(lo) * 100, 2)),
        'unique_gvkeys': int(len(lopucki_matched)),
    },
    'compustat_delistings': {
        'dlrsn_2_bankruptcy': int(len(bk_2)),
        'dlrsn_3_liquidation': int(len(bk_3)),
    },
    'default_events': {
        'total_unique_gvkeys': int(len(default_events)),
        'by_source': {k: int(v) for k, v in default_events['source'].value_counts().items()},
        'by_type': {k: int(v) for k, v in default_events['default_type'].value_counts().items()},
        'by_decade': {f'{int(d)}s': int(v) for d, v in
                       default_events.groupby('decade').size().items()},
    },
    'annual_labels': {
        'rows': int(len(fy_out)),
        'rows_dropped_after_default': int(n_dropped),
        'positive_1y': int(fy_out['default_next_1y'].sum()),
        'positive_2y': int(fy_out['default_next_2y'].sum()),
        'rate_1y_pct': float(round(fy_out['default_next_1y'].mean() * 100, 4)),
        'rate_2y_pct': float(round(fy_out['default_next_2y'].mean() * 100, 4)),
    },
    'quarterly_labels': {
        'rows': int(len(qf_out)),
        'positive_1q': int(qf_out['default_next_1q'].sum()),
        'positive_4q': int(qf_out['default_next_4q'].sum()),
        'positive_8q': int(qf_out['default_next_8q'].sum()),
        'rate_1q_pct': float(round(qf_out['default_next_1q'].mean() * 100, 5)),
        'rate_4q_pct': float(round(qf_out['default_next_4q'].mean() * 100, 5)),
        'rate_8q_pct': float(round(qf_out['default_next_8q'].mean() * 100, 5)),
    },
    'crisis_year_default_rates_1y_pct': {
        str(int(row['fyear'])): float(round(row['rate_1y'], 4))
        for _, row in crisis.iterrows()
    },
    'validation': {
        'spot_checks_all_passed': bool(all_ok),
        'temporal_violations': int(n_bad),
        'events_not_covered_annual_2y': int(len(missing_annual)),
        'events_not_covered_quarterly_8q': int(len(missing_quarterly)),
    },
}
summary_path = CLEAN / 'phase4_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f'  wrote {summary_path.name}')

# Final summary
print('\n' + '='*72)
print(' PHASE 4 — FINAL SUMMARY')
print('='*72)
print(f'  LoPucki match rate:      {n_total_matched}/{len(lo)} = {n_total_matched/len(lo):.1%}')
print(f'  Compustat bankruptcies:  {len(bk_2):,}')
print(f'  Compustat liquidations:  {len(bk_3):,}')
print(f'  Total default events:    {len(default_events):,}')
print(f'  Annual rows kept:        {len(fy_out):,} ({n_dropped:,} dropped)')
print(f'  Annual 1y default rate:  {fy_out["default_next_1y"].mean():.4%} '
      f'({fy_out["default_next_1y"].sum():,} positive)')
print(f'  Quarterly rows kept:     {len(qf_out):,}')
print(f'  Quarterly 4q rate:       {qf_out["default_next_4q"].mean():.4%} '
      f'({qf_out["default_next_4q"].sum():,} positive)')
print()
print('  Class imbalance is severe (~0.3-1% positive). Recommendations:')
print('    - Use focal loss (γ=2) or class-balanced cross-entropy')
print('    - Class weights inverse to frequency (~100-300x for positives)')
print('    - Consider time-aware splits to avoid leakage across firms')
print('    - For training: oversample positives or use weighted sampler')
