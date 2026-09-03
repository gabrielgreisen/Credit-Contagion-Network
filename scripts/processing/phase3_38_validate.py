"""3.8 Validation & Network Statistics + phase3_summary.json"""
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT_ROOT / "data/clean"
EDGES = CLEAN / "edges"

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
GVKEYS = set(fu['gvkey'].astype(int).tolist())
gvkey_to_name = dict(zip(fu['gvkey'].astype(int), fu['conm']))
N_UNIVERSE = len(GVKEYS)

LAYERS = [
    {'name': 'supply_chain',  'file': 'supply_chain_edges.parquet',
     'cols': ('supplier_gvkey', 'customer_gvkey'), 'directional': True},
    {'name': 'creditor',      'file': 'creditor_edges.parquet',
     'cols': ('gvkey_1', 'gvkey_2'), 'directional': False},
    {'name': 'industry_4digit', 'file': 'industry_4digit_edges.parquet',
     'cols': ('gvkey_1', 'gvkey_2'), 'directional': False},
    {'name': 'industry_3digit', 'file': 'industry_3digit_edges.parquet',
     'cols': ('gvkey_1', 'gvkey_2'), 'directional': False},
    {'name': 'geographic',    'file': 'geographic_edges.parquet',
     'cols': ('gvkey_1', 'gvkey_2'), 'directional': False},
    {'name': 'ownership',     'file': 'ownership_edges.parquet',
     'cols': ('parent_gvkey', 'subsidiary_gvkey'), 'directional': True},
    {'name': 'board_interlock','file': 'board_interlock_edges.parquet',
     'cols': ('gvkey_1', 'gvkey_2'), 'directional': False},
]

summary = {'layers': {}}

def degree_stats(unique_pairs, all_gvks, directional=False):
    """Degree from a unique-pair frame."""
    if len(unique_pairs) == 0:
        return {'mean_degree': 0, 'median_degree': 0, 'max_degree': 0,
                 'connected_firms': 0, 'isolated_firms': N_UNIVERSE,
                 'max_degree_gvkey': None, 'max_degree_firm': None}
    g1, g2 = unique_pairs.columns[:2]
    if directional:
        out_deg = unique_pairs.groupby(g1).size()
        in_deg = unique_pairs.groupby(g2).size()
        deg = out_deg.add(in_deg, fill_value=0)
    else:
        all_g = pd.concat([unique_pairs[g1], unique_pairs[g2]])
        deg = all_g.value_counts()
    connected = set(deg.index)
    max_g = int(deg.idxmax())
    return {
        'mean_degree': float(round(deg.mean(), 2)),
        'median_degree': int(deg.median()),
        'max_degree': int(deg.max()),
        'max_degree_gvkey': max_g,
        'max_degree_firm': gvkey_to_name.get(max_g, '?'),
        'connected_firms': len(connected),
        'isolated_firms': N_UNIVERSE - len(connected),
    }

# === Per-layer stats ===
for layer in LAYERS:
    path = EDGES / layer['file']
    if not path.exists() or path.stat().st_size == 0:
        print(f'\n--- {layer["name"]}: SKIP (no file) ---')
        continue
    print(f'\n=== {layer["name"]} ({path.name}) ===')
    g1, g2 = layer['cols']
    df = pd.read_parquet(path, columns=[g1, g2, 'year'])
    n = len(df)
    print(f'  total quarterly edges: {n:,}')
    print(f'  size on disk: {path.stat().st_size/1e6:.1f} MB')

    # Per year aggregation (unique pairs)
    yr_counts = df.groupby('year').size().to_dict()
    print(f'  edges per year (sample):')
    sample_years = [y for y in (1995, 2000, 2005, 2007, 2010, 2015, 2020) if y in yr_counts]
    for y in sample_years:
        print(f'    {y}: {yr_counts[y]:>12,}')

    # Unique pairs across all time (for global degree stats)
    unique_pairs_all = df[[g1, g2]].drop_duplicates()
    print(f'  unique (g1,g2) pairs: {len(unique_pairs_all):,}')
    deg = degree_stats(unique_pairs_all, GVKEYS, layer['directional'])
    print(f'  connected firms: {deg["connected_firms"]:,} / {N_UNIVERSE:,} '
          f'(isolated: {deg["isolated_firms"]:,})')
    print(f'  degree (any-time): mean={deg["mean_degree"]}, '
          f'median={deg["median_degree"]}, max={deg["max_degree"]:,} '
          f'({deg["max_degree_firm"]})')

    # Density (relative to all possible firm-pairs)
    possible = N_UNIVERSE * (N_UNIVERSE - 1) / (1 if layer['directional'] else 2)
    density = len(unique_pairs_all) / possible
    print(f'  density: {density:.4f}% ({len(unique_pairs_all):,} / {int(possible):,})')

    summary['layers'][layer['name']] = {
        'file': layer['file'],
        'directional': layer['directional'],
        'total_quarterly_edges': int(n),
        'unique_pairs_all_time': int(len(unique_pairs_all)),
        'connected_firms': int(deg['connected_firms']),
        'isolated_firms': int(deg['isolated_firms']),
        'mean_degree': deg['mean_degree'],
        'median_degree': deg['median_degree'],
        'max_degree': deg['max_degree'],
        'max_degree_firm': deg['max_degree_firm'],
        'density_pct': float(round(density * 100, 6)),
        'edges_per_year_sample': {str(y): int(yr_counts[y]) for y in sample_years},
        'on_disk_mb': float(round(path.stat().st_size / 1e6, 2)),
    }

# === Cross-layer overlap for 2007 ===
print('\n=== Cross-layer overlap (sample year 2007) ===')
year_target = 2007
pair_layer_count = defaultdict(int)
pair_layers = defaultdict(set)

for layer in LAYERS:
    path = EDGES / layer['file']
    if not path.exists() or path.stat().st_size == 0:
        continue
    g1, g2 = layer['cols']
    df = pd.read_parquet(path, columns=[g1, g2, 'year'])
    df_2007 = df[df['year'] == year_target][[g1, g2]].drop_duplicates()
    if len(df_2007) == 0:
        continue
    print(f'  {layer["name"]}: {len(df_2007):,} unique pairs in 2007')
    if layer['directional']:
        # canonicalize for cross-layer comparison: min,max
        a = df_2007[g1].values
        b = df_2007[g2].values
        canon = np.where(a < b, a, b)
        canon2 = np.where(a < b, b, a)
        pair_keys = list(zip(canon, canon2))
    else:
        pair_keys = list(zip(df_2007[g1].values, df_2007[g2].values))
    for pk in pair_keys:
        pair_layer_count[pk] += 1
        pair_layers[pk].add(layer['name'])
    del df, df_2007

# Distribution
overlap_dist = {}
for pk, cnt in pair_layer_count.items():
    overlap_dist[cnt] = overlap_dist.get(cnt, 0) + 1
print(f'\n  pair count by layer multiplicity (2007):')
for k in sorted(overlap_dist):
    print(f'    in exactly {k} layer(s): {overlap_dist[k]:>10,} pairs')

# Top pair-pair overlaps between layers
print(f'\n  pair-counts where pair appears in ≥ 4 layers (top 10):')
high_mult = [(pk, cnt, sorted(pair_layers[pk])) for pk, cnt in pair_layer_count.items() if cnt >= 4]
high_mult.sort(key=lambda x: -x[1])
for pk, cnt, layers_lst in high_mult[:10]:
    n1 = gvkey_to_name.get(int(pk[0]), '?')[:25]
    n2 = gvkey_to_name.get(int(pk[1]), '?')[:25]
    print(f'    [{cnt}] {n1:<25s} ↔ {n2:<25s}  layers={layers_lst}')

summary['cross_layer_2007'] = {
    'total_unique_pairs_any_layer': len(pair_layer_count),
    'pairs_by_layer_count': {int(k): int(v) for k, v in overlap_dist.items()},
}

# === Spot checks ===
print('\n=== SPOT CHECKS ===')

def gvkey_for(name_substr):
    m = fu[fu['conm'].str.contains(name_substr, case=False, na=False)]
    return m[['gvkey', 'conm']].head(5).to_string(index=False)

def edges_for_gvkey(gvkey, layer_filter=None, year_max=None):
    """Per-layer edge counts for a focal gvkey."""
    out = {}
    for layer in LAYERS:
        if layer_filter and layer['name'] not in layer_filter:
            continue
        path = EDGES / layer['file']
        if not path.exists() or path.stat().st_size == 0:
            continue
        g1, g2 = layer['cols']
        df = pd.read_parquet(path, columns=[g1, g2, 'year'])
        m = (df[g1] == gvkey) | (df[g2] == gvkey)
        if year_max is not None:
            m &= (df['year'] <= year_max)
        sub = df[m]
        # distinct counterparties
        cps = pd.concat([sub[g1], sub[g2]]).unique()
        cps = [int(c) for c in cps if int(c) != gvkey]
        out[layer['name']] = {
            'quarterly_edge_rows': int(len(sub)),
            'distinct_counterparties': int(len(set(cps))),
        }
    return out

# Enron (gvkey 6127, pre-2002)
print('\n--- Enron (gvkey 6127, year ≤ 2001) ---')
e_stats = edges_for_gvkey(6127, year_max=2001)
for layer, s in e_stats.items():
    print(f'  {layer:<20s}: {s["quarterly_edge_rows"]:>8,} edges, '
          f'{s["distinct_counterparties"]:>4,} counterparties')
summary['spot_checks'] = {'enron_pre_2002': e_stats}

# Lehman Brothers Holdings (gvkey 30128, pre-2009)
print('\n--- Lehman Brothers Holdings (gvkey 30128, year ≤ 2008) ---')
l_stats = edges_for_gvkey(30128, year_max=2008)
for layer, s in l_stats.items():
    print(f'  {layer:<20s}: {s["quarterly_edge_rows"]:>8,} edges, '
          f'{s["distinct_counterparties"]:>4,} counterparties')
summary['spot_checks']['lehman_pre_2009'] = l_stats

# Apple (gvkey 1690, all time)
print('\n--- Apple (gvkey 1690, all time) ---')
a_stats = edges_for_gvkey(1690)
for layer, s in a_stats.items():
    print(f'  {layer:<20s}: {s["quarterly_edge_rows"]:>8,} edges, '
          f'{s["distinct_counterparties"]:>4,} counterparties')
summary['spot_checks']['apple_all_time'] = a_stats

# === Save summary ===
summary_path = EDGES / 'phase3_summary.json'
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f'\nwrote {summary_path}')

# === Final summary table ===
print('\n' + '='*78)
print(' PHASE 3 — FINAL SUMMARY')
print('='*78)
print(f'{"layer":<20s} {"directional":>11s} {"edges":>15s} {"pairs":>13s} {"connected":>10s} {"density":>9s} {"MB":>7s}')
for layer in LAYERS:
    info = summary['layers'].get(layer['name'])
    if info is None:
        print(f'{layer["name"]:<20s} {"":>11s} {"(empty)":>15s}')
        continue
    print(f'{layer["name"]:<20s} {str(info["directional"]):>11s} '
          f'{info["total_quarterly_edges"]:>15,} '
          f'{info["unique_pairs_all_time"]:>13,} '
          f'{info["connected_firms"]:>10,} '
          f'{info["density_pct"]:>9.4f}% '
          f'{info["on_disk_mb"]:>7.1f}')

total_edges = sum(s.get('total_quarterly_edges', 0) for s in summary['layers'].values())
total_mb = sum(s.get('on_disk_mb', 0) for s in summary['layers'].values())
print(f'\nTOTAL: {total_edges:,} quarterly edges, {total_mb:.1f} MB on disk')
