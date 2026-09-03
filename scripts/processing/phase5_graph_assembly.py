"""Phase 5: Graph Assembly — build per-quarter HeteroData snapshots.

Run with the repo as cwd, or pass --project-root explicitly:
    python scripts/processing/phase5_graph_assembly.py
    python scripts/processing/phase5_graph_assembly.py --project-root /path/to/repo
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_parser = argparse.ArgumentParser()
_parser.add_argument("--project-root", type=Path, default=_DEFAULT_ROOT,
                     help="Repo root containing data/clean/ (default: inferred from script location)")
_args, _ = _parser.parse_known_args()

PROJECT_ROOT = _args.project_root.resolve()
CLEAN = PROJECT_ROOT / "data/clean"
EDGES = CLEAN / "edges"
SNAP_DIR = CLEAN / "graph_snapshots/quarterly"
IDX_DIR = CLEAN / "node_index_maps"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
def stamp(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ============================================================
# 5.1 Define feature columns
# ============================================================
EXCLUDE = {
    'gvkey', 'datadate', 'conm', 'sic', 'fyearq', 'fqtr',
    'node_type', 'altman_variant', 'altman_zone',
    'total_debt', 'ebit_proxy',
    'permno', 'has_crsp',
    'avg_volume',                      # log_avg_volume (z-scored) is included instead
    'median_spread_bps', 'n_bonds',    # raw spread + count (target/aux)
}

stamp('loading quarterly features for column inspection...')
feats_full = pd.read_parquet(CLEAN / 'node_features_quarterly_standardized.parquet')
feats_full['datadate'] = pd.to_datetime(feats_full['datadate'])
feats_full['cal_year'] = feats_full['datadate'].dt.year.astype('int32')
feats_full['cal_quarter'] = feats_full['datadate'].dt.quarter.astype('int8')

ALL_COLS = list(feats_full.columns)
FEATURE_COLS = [c for c in ALL_COLS if c not in EXCLUDE
                 and c not in {'cal_year', 'cal_quarter'}]
N_FEATURES = len(FEATURE_COLS)

print(f'\n=== 5.1 Feature columns ({N_FEATURES}) ===')
for c in FEATURE_COLS:
    print(f'  {c}')
print(f'\n=== Excluded from x ({len(EXCLUDE)}) ===')
for c in sorted(EXCLUDE):
    print(f'  {c}')

# Cast features to float32 for the x tensor
feats_full[FEATURE_COLS] = feats_full[FEATURE_COLS].astype('float32')

# Map node_type to int class
feats_full['node_class_int'] = (feats_full['node_type'] == 'financial').astype('int8')
# Coerce has_crsp: load from firm_universe (since the standardized parquet may have it as object dtype)
fu = pd.read_parquet(CLEAN / 'firm_universe.parquet').reset_index()
fu['gvkey'] = fu['gvkey'].astype(int)
gvkey_to_has_crsp = dict(zip(fu['gvkey'], fu['has_crsp'].astype(bool)))
gvkey_to_node_class = dict(zip(fu['gvkey'], fu['node_class']))

# ============================================================
# 5.1b Load labels + spreads
# ============================================================
stamp('loading default labels...')
labels = pd.read_parquet(CLEAN / 'default_labels_quarterly.parquet')
labels['datadate'] = pd.to_datetime(labels['datadate'])
labels['gvkey'] = labels['gvkey'].astype(int)

stamp('loading credit spreads...')
spreads = pd.read_parquet(CLEAN / 'credit_spreads_quarterly.parquet')
spreads['datadate'] = pd.to_datetime(spreads['datadate'])
spreads['gvkey'] = spreads['gvkey'].astype(int)

# Build per-quarter lookup dicts
labels_by_dt = {dt: g for dt, g in labels.groupby('datadate')}
spreads_by_yq = {(int(y), int(q)): g for (y, q), g in spreads.groupby(['year', 'quarter'])}

# ============================================================
# 5.2 Edge layers — config + reader helpers
# ============================================================
EDGE_LAYERS = [
    {'key': 'supply_chain', 'file': 'supply_chain_edges.parquet',
     'src': 'supplier_gvkey', 'dst': 'customer_gvkey',
     'attr_cols': ['salecs', 'customer_concentration', 'relationship_duration'],
     'undirected': False, 'add_reverse': True,
     'edge_type': ('firm', 'supplies', 'firm'),
     'reverse_type': ('firm', 'supplied_by', 'firm')},
    {'key': 'creditor', 'file': 'creditor_edges.parquet',
     'src': 'gvkey_1', 'dst': 'gvkey_2',
     'attr_cols': ['shared_lender_count', 'shared_exposure', 'shared_lead_arranger'],
     'undirected': True,
     'edge_type': ('firm', 'shares_creditor', 'firm')},
    {'key': 'industry_4d', 'file': 'industry_4digit_edges.parquet',
     'src': 'gvkey_1', 'dst': 'gvkey_2',
     'attr_cols': [],
     'undirected': True,
     'edge_type': ('firm', 'same_industry_4d', 'firm')},
    {'key': 'industry_3d', 'file': 'industry_3digit_edges.parquet',
     'src': 'gvkey_1', 'dst': 'gvkey_2',
     'attr_cols': [],
     'undirected': True,
     'edge_type': ('firm', 'same_industry_3d', 'firm')},
    {'key': 'geographic', 'file': 'geographic_edges.parquet',
     'src': 'gvkey_1', 'dst': 'gvkey_2',
     'attr_cols': [],
     'undirected': True,
     'edge_type': ('firm', 'same_state', 'firm')},
    {'key': 'ownership', 'file': 'ownership_edges.parquet',
     'src': 'parent_gvkey', 'dst': 'subsidiary_gvkey',
     'attr_cols': ['ownership_pct'],
     'undirected': False, 'add_reverse': False,
     'edge_type': ('firm', 'owns', 'firm')},
    {'key': 'board', 'file': 'board_interlock_edges.parquet',
     'src': 'gvkey_1', 'dst': 'gvkey_2',
     'attr_cols': ['shared_director_count', 'shared_executive'],
     'undirected': True,
     'edge_type': ('firm', 'shares_director', 'firm')},
]

# Open ParquetFile handles for filter-pushdown reads
edge_pq_files = {layer['key']: pq.ParquetFile(EDGES / layer['file']) for layer in EDGE_LAYERS}

def read_edges_q(layer, year, quarter):
    """Filter-pushdown read of edges for a given (year, quarter)."""
    cols = [layer['src'], layer['dst']] + layer['attr_cols'] + ['year', 'quarter']
    try:
        tbl = pq.read_table(EDGES / layer['file'], columns=cols,
                             filters=[('year', '=', int(year)),
                                      ('quarter', '=', int(quarter))])
        return tbl.to_pandas()
    except Exception:
        # If file is empty, return an empty DataFrame with the right schema
        return pd.DataFrame(columns=cols)

# ============================================================
# Splits
# ============================================================
def assign_split(year, quarter):
    """Main split."""
    if year < 1990: return 'pretrain'
    if year <= 2006: return 'train'
    if year <= 2012: return 'val'
    return 'test'

def assign_split_alt_a(year, quarter):
    if year < 1990: return 'pretrain'
    if year <= 2004: return 'train'
    if year <= 2009: return 'val'
    return 'test'

def assign_split_alt_b(year, quarter):
    if year < 1990: return 'pretrain'
    if year in (2008, 2009) or (year == 2020): return 'test'
    return 'train'

# ============================================================
# Main per-quarter build
# ============================================================
def build_quarter(year, quarter, feats_q):
    """Build one HeteroData snapshot for (year, quarter)."""
    # Active firms — those with a row in this quarter
    feats_q = feats_q.sort_values('gvkey').reset_index(drop=True)
    feats_q['node_idx'] = np.arange(len(feats_q), dtype=np.int32)
    gvkey_to_idx = dict(zip(feats_q['gvkey'].astype(int), feats_q['node_idx']))
    n_nodes = len(feats_q)

    # Features + mask
    raw = feats_q[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    mask = ~np.isnan(raw)
    raw[~mask] = 0.0
    # Replace any infs that snuck through
    bad = ~np.isfinite(raw)
    if bad.any():
        raw[bad] = 0.0
        mask[bad] = False
    x_tensor = torch.from_numpy(raw)
    x_mask_tensor = torch.from_numpy(mask)

    # Build HeteroData
    data = HeteroData()
    data['firm'].x = x_tensor
    data['firm'].x_mask = x_mask_tensor
    data['firm'].node_class = torch.tensor(feats_q['node_class_int'].values, dtype=torch.int8)
    data['firm'].has_crsp = torch.tensor(
        feats_q['gvkey'].astype(int).map(gvkey_to_has_crsp).fillna(False).astype(bool).values,
        dtype=torch.bool)
    data['firm'].gvkeys = torch.tensor(feats_q['gvkey'].astype(int).values, dtype=torch.int32)
    data['firm'].datadates = feats_q['datadate'].astype('int64').values  # ns since epoch

    # Labels
    quarter_dt = pd.Timestamp(f'{year}-{[3,6,9,12][quarter-1]:02d}-01') + pd.offsets.MonthEnd(0)
    # Default labels: match by gvkey + datadate exactly (not calendar quarter end)
    fy_dts = feats_q['datadate'].dt.normalize().values
    # Build a label map keyed on gvkey for this quarter
    # Get labels rows matching feats_q's exact datadates
    feats_q_lookup = feats_q[['gvkey', 'datadate']].copy()
    # Merge labels
    labels_q = labels[labels['datadate'].isin(feats_q['datadate'].unique())]
    label_merged = feats_q_lookup.merge(labels_q[['gvkey', 'datadate',
                                                    'default_next_1q', 'default_next_4q',
                                                    'default_next_8q']],
                                         on=['gvkey', 'datadate'], how='left')
    def label_tensor(name):
        v = label_merged[name].astype('float32').values
        v = np.where(np.isnan(v), -1.0, v)
        return torch.from_numpy(v.astype(np.float32))
    data['firm'].y_default_1q = label_tensor('default_next_1q')
    data['firm'].y_default_4q = label_tensor('default_next_4q')
    data['firm'].y_default_8q = label_tensor('default_next_8q')

    # Credit spread target: keyed on (gvkey, year, quarter)
    sp_q = spreads_by_yq.get((year, quarter))
    if sp_q is not None and len(sp_q) > 0:
        sp_map = dict(zip(sp_q['gvkey'].astype(int), sp_q['median_spread_bps']))
    else:
        sp_map = {}
    y_spread = (feats_q['gvkey'].astype(int).map(sp_map)
                  .astype('float32').fillna(-1.0).values)
    data['firm'].y_spread = torch.from_numpy(y_spread)

    # ----- Edges -----
    edge_stats = {}
    for layer in EDGE_LAYERS:
        edges_df = read_edges_q(layer, year, quarter)
        # Map gvkeys to indices
        if len(edges_df) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            attr_dim = max(1, len(layer['attr_cols']))
            edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32) if layer['attr_cols'] else None
            data[layer['edge_type']].edge_index = edge_index
            if edge_attr is not None:
                data[layer['edge_type']].edge_attr = edge_attr
            if layer.get('add_reverse'):
                rev_type = layer['reverse_type']
                data[rev_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if edge_attr is not None:
                    data[rev_type].edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32)
            edge_stats[layer['key']] = 0
            continue

        src_idx = edges_df[layer['src']].astype(int).map(gvkey_to_idx)
        dst_idx = edges_df[layer['dst']].astype(int).map(gvkey_to_idx)
        valid = src_idx.notna() & dst_idx.notna()
        edges_df = edges_df[valid].copy()
        src_idx = src_idx[valid].astype(int).values
        dst_idx = dst_idx[valid].astype(int).values

        if len(src_idx) == 0:
            data[layer['edge_type']].edge_index = torch.zeros((2, 0), dtype=torch.long)
            if layer['attr_cols']:
                data[layer['edge_type']].edge_attr = torch.zeros((0, len(layer['attr_cols'])),
                                                                    dtype=torch.float32)
            if layer.get('add_reverse'):
                data[layer['reverse_type']].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if layer['attr_cols']:
                    data[layer['reverse_type']].edge_attr = torch.zeros(
                        (0, len(layer['attr_cols'])), dtype=torch.float32)
            edge_stats[layer['key']] = 0
            continue

        # Forward edges
        if layer['undirected']:
            # Add both directions
            ei = np.stack([np.concatenate([src_idx, dst_idx]),
                           np.concatenate([dst_idx, src_idx])])
            edge_index_t = torch.from_numpy(ei.astype(np.int64))
            if layer['attr_cols']:
                attr_arr = edges_df[layer['attr_cols']].astype('float32').to_numpy()
                attr_arr = np.nan_to_num(attr_arr, nan=0.0, posinf=0.0, neginf=0.0)
                attr_t = torch.from_numpy(np.concatenate([attr_arr, attr_arr], axis=0))
                data[layer['edge_type']].edge_attr = attr_t
            data[layer['edge_type']].edge_index = edge_index_t
            edge_stats[layer['key']] = ei.shape[1]
        else:
            # Directional — forward edges
            ei_fwd = np.stack([src_idx, dst_idx])
            data[layer['edge_type']].edge_index = torch.from_numpy(ei_fwd.astype(np.int64))
            if layer['attr_cols']:
                attr_arr = edges_df[layer['attr_cols']].astype('float32').to_numpy()
                attr_arr = np.nan_to_num(attr_arr, nan=0.0, posinf=0.0, neginf=0.0)
                data[layer['edge_type']].edge_attr = torch.from_numpy(attr_arr)
            edge_stats[layer['key']] = ei_fwd.shape[1]
            # Add reverse as separate type if specified
            if layer.get('add_reverse'):
                ei_rev = np.stack([dst_idx, src_idx])
                data[layer['reverse_type']].edge_index = torch.from_numpy(ei_rev.astype(np.int64))
                if layer['attr_cols']:
                    data[layer['reverse_type']].edge_attr = torch.from_numpy(attr_arr.copy())
                edge_stats[layer['key'] + '_reverse'] = ei_rev.shape[1]

    return data, gvkey_to_idx, edge_stats, n_nodes

# ============================================================
# Iterate quarters
# ============================================================
stamp('iterating quarters...')
quarters_processed = []
quarters_with_split = []

# Group features by (cal_year, cal_quarter)
feats_groups = feats_full.groupby(['cal_year', 'cal_quarter'], sort=True)
total_q = len(feats_groups)
print(f'  total quarters to process: {total_q}')

for i, ((year, quarter), feats_q) in enumerate(feats_groups):
    year, quarter = int(year), int(quarter)
    data, gvkey_to_idx, edge_stats, n_nodes = build_quarter(year, quarter, feats_q)

    # Save snapshot
    out_pt = SNAP_DIR / f'{year}_Q{quarter}.pt'
    torch.save(data, out_pt)

    # Save node index map
    idx_df = pd.DataFrame({
        'gvkey': list(gvkey_to_idx.keys()),
        'node_idx': list(gvkey_to_idx.values()),
    })
    idx_df.to_parquet(IDX_DIR / f'{year}_Q{quarter}.parquet', index=False)

    quarters_processed.append({
        'year': year, 'quarter': quarter, 'n_nodes': n_nodes,
        'split': assign_split(year, quarter),
        'split_alt_a': assign_split_alt_a(year, quarter),
        'split_alt_b': assign_split_alt_b(year, quarter),
        'pt_size_bytes': out_pt.stat().st_size,
        **{f'edges_{k}': v for k, v in edge_stats.items()},
    })

    if (i + 1) % 20 == 0 or i == total_q - 1:
        stamp(f'  [{i+1:3d}/{total_q}] {year}-Q{quarter}: {n_nodes:,} nodes, '
              f'{sum(edge_stats.values()):,} total edges, '
              f'{out_pt.stat().st_size/1e6:.1f} MB')

stamp(f'all {len(quarters_processed)} quarters built')

# ============================================================
# 5.3 Splits — save assignments
# ============================================================
splits_df = pd.DataFrame(quarters_processed)
splits_meta = splits_df[['year', 'quarter', 'split', 'split_alt_a', 'split_alt_b']]
splits_meta.to_parquet(CLEAN / 'split_assignments.parquet', index=False)
print(f'  wrote split_assignments.parquet ({len(splits_meta)} quarters)')

# Print split counts
print(f'\nMain split distribution:')
print(splits_meta['split'].value_counts().to_string())

# ============================================================
# 5.4 Save graph_metadata.json
# ============================================================
edge_attr_columns = {}
for layer in EDGE_LAYERS:
    edge_attr_columns[str(layer['edge_type'])] = layer['attr_cols']
    if layer.get('add_reverse'):
        edge_attr_columns[str(layer['reverse_type'])] = layer['attr_cols']

metadata = {
    'feature_columns': FEATURE_COLS,
    'n_features': N_FEATURES,
    'edge_types': [str(layer['edge_type']) for layer in EDGE_LAYERS] +
                  [str(layer['reverse_type']) for layer in EDGE_LAYERS if layer.get('add_reverse')],
    'edge_attr_columns': edge_attr_columns,
    'split_boundaries': {
        'train': '1990-Q1 to 2006-Q4',
        'val':   '2007-Q1 to 2012-Q4',
        'test':  '2013-Q1 to 2024-Q4',
        'pretrain': 'before 1990-Q1',
    },
    'total_quarters': len(splits_df),
    'label_columns': ['y_default_1q', 'y_default_4q', 'y_default_8q', 'y_spread'],
    'metadata_columns': ['gvkeys', 'datadates', 'node_class', 'has_crsp'],
    'node_class_encoding': {'nonfinancial': 0, 'financial': 1},
    'nan_imputation': '0.0 with x_mask boolean indicator',
    'label_nan_sentinel': -1.0,
}
(CLEAN / 'graph_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
print(f'  wrote graph_metadata.json')

# ============================================================
# 5.5 Validation
# ============================================================
print('\n' + '='*72)
print(' 5.5 VALIDATION')
print('='*72)

def inspect_quarter(year, q):
    pth = SNAP_DIR / f'{year}_Q{q}.pt'
    if not pth.exists():
        print(f'  {year}-Q{q}: snapshot missing')
        return None
    data = torch.load(pth, weights_only=False)
    n_nodes = data['firm'].x.size(0)
    print(f'\n--- {year}-Q{q} ---')
    print(f'  nodes: {n_nodes:,}')
    print(f'  feature shape: {tuple(data["firm"].x.shape)}')
    print(f'  feature non-NaN ratio (mask True): {data["firm"].x_mask.float().mean().item():.3f}')
    # Verify no NaN/inf in x
    bad = (~torch.isfinite(data['firm'].x)).sum().item()
    print(f'  feature NaN/inf count: {bad} (expect 0)')

    print(f'  edges:')
    for et in data.edge_types:
        ei = data[et].edge_index
        n = ei.size(1)
        if n == 0:
            print(f'    {et}: 0')
            continue
        max_idx = int(ei.max())
        within = max_idx < n_nodes
        # Self-loops
        self_loops = (ei[0] == ei[1]).sum().item()
        # Undirected check (count of (i,j) == count of (j,i))
        undirected = '?'
        # Quick sanity for undirected types: forward+reverse should match
        et_str = str(et)
        if 'shares_creditor' in et_str or 'same_' in et_str or 'shares_director' in et_str:
            # Should have equal counts in both directions
            edge_set = set(map(tuple, ei.t().tolist()))
            rev_set = set((b, a) for a, b in edge_set)
            symmetry = len(edge_set & rev_set) / max(len(edge_set), 1)
            undirected = f'sym={symmetry:.2%}'
        print(f'    {et}: {n:,} edges, max_idx={max_idx} (within={within}), '
              f'self_loops={self_loops}, {undirected}')

    # Labels
    y1 = data['firm'].y_default_1q
    y4 = data['firm'].y_default_4q
    y8 = data['firm'].y_default_8q
    ys = data['firm'].y_spread
    print(f'  labels:')
    print(f'    y_default_1q: pos={(y1==1).sum().item()}, neg={(y1==0).sum().item()}, '
          f'masked={(y1==-1).sum().item()}')
    print(f'    y_default_4q: pos={(y4==1).sum().item()}, neg={(y4==0).sum().item()}')
    print(f'    y_default_8q: pos={(y8==1).sum().item()}, neg={(y8==0).sum().item()}')
    print(f'    y_spread:     observed={(ys>=0).sum().item()}, masked={(ys==-1).sum().item()}, '
          f'median={ys[ys>=0].median().item():.0f}' if (ys>=0).sum() > 0 else '    y_spread: all masked')
    return data

# Three sample quarters
for (y, q) in [(2001, 3), (2007, 3), (2020, 1)]:
    inspect_quarter(y, q)

# Connectivity check on 2007-Q3
print('\n--- Connectivity (2007-Q3) ---')
data_07q3 = torch.load(SNAP_DIR / '2007_Q3.pt', weights_only=False)
n = data_07q3['firm'].x.size(0)
deg = torch.zeros(n, dtype=torch.long)
sc_deg = torch.zeros(n, dtype=torch.long)
cred_deg = torch.zeros(n, dtype=torch.long)
for et in data_07q3.edge_types:
    ei = data_07q3[et].edge_index
    if ei.size(1) == 0: continue
    deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
    deg.scatter_add_(0, ei[1], torch.ones(ei.size(1), dtype=torch.long))
    if 'supplies' in str(et) or 'supplied_by' in str(et):
        sc_deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
        sc_deg.scatter_add_(0, ei[1], torch.ones(ei.size(1), dtype=torch.long))
    if 'shares_creditor' in str(et):
        cred_deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
        cred_deg.scatter_add_(0, ei[1], torch.ones(ei.size(1), dtype=torch.long))
isolated = (deg == 0).sum().item()
has_sc = (sc_deg > 0).sum().item()
has_cred = (cred_deg > 0).sum().item()
print(f'  total nodes: {n:,}')
print(f'  isolated (zero edges): {isolated:,} ({isolated/n:.1%})')
print(f'  with ≥1 supply-chain edge: {has_sc:,} ({has_sc/n:.1%})')
print(f'  with ≥1 creditor edge:     {has_cred:,} ({has_cred/n:.1%})')
print(f'  mean total degree: {deg.float().mean().item():.1f}')

# ============================================================
# Default rate by split
# ============================================================
print('\n--- Default rates by split (4q label) ---')
split_stats = {'train': {'pos': 0, 'total': 0},
                'val':   {'pos': 0, 'total': 0},
                'test':  {'pos': 0, 'total': 0},
                'pretrain': {'pos': 0, 'total': 0}}
for q in quarters_processed:
    pth = SNAP_DIR / f'{q["year"]}_Q{q["quarter"]}.pt'
    data = torch.load(pth, weights_only=False)
    y4 = data['firm'].y_default_4q
    pos = int((y4 == 1).sum())
    valid = int((y4 != -1).sum())
    s = q['split']
    split_stats[s]['pos'] += pos
    split_stats[s]['total'] += valid
for s, vv in split_stats.items():
    rate = vv['pos'] / vv['total'] * 100 if vv['total'] else 0
    print(f'  {s:10s}: {vv["pos"]:>5,} positive / {vv["total"]:>9,} valid ({rate:.3f}%)')

# ============================================================
# Size estimates
# ============================================================
total_size = sum(p['pt_size_bytes'] for p in quarters_processed)
max_size = max(p['pt_size_bytes'] for p in quarters_processed)
max_size_q = max(quarters_processed, key=lambda x: x['pt_size_bytes'])
print(f'\n--- Storage ---')
print(f'  total snapshots: {len(quarters_processed)}')
print(f'  total disk size: {total_size/1e9:.2f} GB')
print(f'  largest snapshot: {max_size_q["year"]}-Q{max_size_q["quarter"]} '
      f'({max_size/1e6:.1f} MB, {max_size_q["n_nodes"]:,} nodes)')

# Spot checks
print('\n--- Spot checks ---')
def gvkey_in_q(gvkey, year, q):
    data = torch.load(SNAP_DIR / f'{year}_Q{q}.pt', weights_only=False)
    gvks = data['firm'].gvkeys.numpy()
    if gvkey not in gvks:
        print(f'  gvkey {gvkey} NOT in {year}-Q{q}')
        return None
    idx = int(np.where(gvks == gvkey)[0][0])
    name = gvkey_to_node_class.get(gvkey, '?')
    print(f'\n  {gvkey} ({fu[fu["gvkey"]==gvkey]["conm"].iloc[0]}) — {year}-Q{q} [idx={idx}]')
    # Edge counts by type
    for et in data.edge_types:
        ei = data[et].edge_index
        if ei.size(1) == 0: continue
        cnt = int(((ei[0]==idx) | (ei[1]==idx)).sum())
        if cnt > 0:
            print(f'    {et}: {cnt:,} edges')
    # Feature values
    feat = data['firm'].x[idx].numpy()
    feat_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    for col in ['debt_to_assets', 'roa', 'log_assets', 'altman_z', 'log_credit_spread', 'ret_12m']:
        if col in feat_idx:
            print(f'    {col}: {feat[feat_idx[col]]:+.3f}')
    # Labels
    print(f'    y_default_1q: {data["firm"].y_default_1q[idx].item():.0f}')
    print(f'    y_default_4q: {data["firm"].y_default_4q[idx].item():.0f}')
    print(f'    y_spread:     {data["firm"].y_spread[idx].item():.0f}')

# Lehman 2008 Q2
gvkey_in_q(30128, 2008, 2)
# Apple 2023 Q4
gvkey_in_q(1690, 2023, 4)

# ============================================================
# Save phase5_summary.json
# ============================================================
summary = {
    'total_quarters': len(quarters_processed),
    'date_range': f'{splits_df.iloc[0]["year"]}-Q{splits_df.iloc[0]["quarter"]} → '
                  f'{splits_df.iloc[-1]["year"]}-Q{splits_df.iloc[-1]["quarter"]}',
    'total_disk_size_gb': float(round(total_size / 1e9, 2)),
    'largest_snapshot_mb': float(round(max_size / 1e6, 1)),
    'largest_quarter': f'{max_size_q["year"]}-Q{max_size_q["quarter"]}',
    'largest_n_nodes': int(max_size_q['n_nodes']),
    'split_counts': {k: int(v) for k, v in splits_meta['split'].value_counts().to_dict().items()},
    'feature_columns': FEATURE_COLS,
    'n_features': N_FEATURES,
    'edge_types_count': len(metadata['edge_types']),
    'default_rates_by_split_4q': {
        s: {'pos': int(vv['pos']),
             'total': int(vv['total']),
             'rate_pct': float(round(vv['pos']/vv['total']*100, 4)) if vv['total'] else 0.0}
        for s, vv in split_stats.items()
    },
    'connectivity_2007Q3': {
        'n_nodes': int(n),
        'isolated_pct': float(round(isolated/n*100, 2)),
        'with_sc_edge_pct': float(round(has_sc/n*100, 2)),
        'with_cred_edge_pct': float(round(has_cred/n*100, 2)),
        'mean_total_degree': float(round(deg.float().mean().item(), 1)),
    },
}
(CLEAN / 'phase5_summary.json').write_text(json.dumps(summary, indent=2))
print(f'\nwrote phase5_summary.json')

print('\n' + '='*72)
print(' PHASE 5 — FINAL SUMMARY')
print('='*72)
print(f'  total quarter snapshots: {summary["total_quarters"]}')
print(f'  total disk size: {summary["total_disk_size_gb"]} GB')
print(f'  largest snapshot: {summary["largest_snapshot_mb"]} MB ({summary["largest_quarter"]})')
print(f'  features per node: {summary["n_features"]}')
print(f'  edge types: {summary["edge_types_count"]}')
print(f'  splits: {summary["split_counts"]}')
