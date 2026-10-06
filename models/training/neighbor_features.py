"""Hand-built network (neighbor) features per layer, for the XGBoost+network baseline.

For each firm-quarter and each network layer, all measured at or before time t:

  <layer>_degree            number of (live) neighbors in the layer at t
  <layer>_nbr_def_cnt       neighbor default events in the 4 quarters up to t
  <layer>_nbr_def_share     same, scaled by max(degree, 1)
  <layer>_nbr_<stat>        mean of a neighbor risk stat (standardized panel value)

Neighbor risk stats: altman_z, roa, debt_to_assets, ret_12m, volatility_12m,
log_mktcap — a compact distress block from the firm's own feature set.

Neighbor defaults (the contagion signal)
----------------------------------------
A defaulting firm usually leaves the panel at default, so it is nobody's *live*
neighbor afterwards. Counting defaults among live neighbors at t therefore
yields ~zero everywhere. Instead, for each default event we take the
defaulter's neighbors **as of its last active quarter** (at or before the
default), and flag those neighbors for the default quarter and the three
quarters after it. Everything is still measured at or before t.

Layers
------
Sparse layers (explicit edge lists, merge-based aggregation):
  supply_cust   the firm's customers        (supply_chain: supplier -> customer)
  supply_supp   the firm's suppliers        (supply_chain reversed)
  creditor      shared-lender neighbors     (creditor_edges, undirected)
  board         board-interlock neighbors   (board_interlock_edges, undirected)

Clique layers (every member of a group is connected; aggregated by group,
mirroring how phase3 builds these edges, so the 78M-294M row edge lists are
never scanned):
  ind4          same 4-digit SIC   (firm_years sich/sic, as in phase3_33_34)
  ind3          same 3-digit SIC
  state         same headquarters state (firm_universe, as in phase3_35)

The ownership layer is empty in the current data and is skipped.

Output cached to data/clean/neighbor_features_quarterly.parquet, keyed by
(gvkey, year, quarter).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN = PROJECT_ROOT / "data" / "clean"
EDGES = CLEAN / "edges"
CACHE = CLEAN / "neighbor_features_quarterly.parquet"

NEIGHBOR_STATS = ["altman_z", "roa", "debt_to_assets", "ret_12m",
                  "volatility_12m", "log_mktcap"]

KEY = ["gvkey", "year", "quarter"]
DEFAULT_WINDOW_Q = 4      # a default stays "recent" for this many quarters
MAX_EXIT_GAP_Q = 8        # defaulter's last panel quarter must be within this of the event


def _qidx(year, quarter):
    return year * 4 + (quarter - 1)


def _with_qidx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qidx"] = _qidx(df["year"].astype(int), df["quarter"].astype(int))
    return df


# ---------------------------------------------------------------------------
# Default events, anchored at the defaulter's last active quarter
# ---------------------------------------------------------------------------

def _default_events(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per usable default event: gvkey, qidx_last, qidx_def.

    qidx_def  calendar quarter of the default event
    qidx_last the defaulter's last panel quarter at or before the event
              (events with no panel row within MAX_EXIT_GAP_Q are dropped)
    """
    de = pd.read_parquet(CLEAN / "default_events.parquet")[["gvkey", "default_date"]]
    de["default_date"] = pd.to_datetime(de["default_date"])
    de["qidx_def"] = _qidx(de["default_date"].dt.year, de["default_date"].dt.quarter)

    pq = _with_qidx(panel[KEY])
    firm_quarters = {g: np.sort(s.values) for g, s in pq.groupby("gvkey")["qidx"]}

    rows = []
    for gvkey, qdef in zip(de["gvkey"], de["qidx_def"]):
        qs = firm_quarters.get(gvkey)
        if qs is None:
            continue
        pos = np.searchsorted(qs, qdef, side="right") - 1
        if pos < 0:
            continue
        qlast = int(qs[pos])
        if qdef - qlast > MAX_EXIT_GAP_Q:
            continue
        rows.append((gvkey, qlast, qdef))
    ev = pd.DataFrame(rows, columns=["gvkey", "qidx_last", "qidx_def"])
    ev = ev.drop_duplicates()
    print(f"default events anchored to the panel: {len(ev):,} / {len(de):,}")
    return ev


def _expand_event_window(ev: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """(id_cols, qidx_def) -> one row per affected quarter t in [qidx_def, qidx_def+3]."""
    reps = pd.concat(
        [ev[id_cols + ["qidx_def"]].assign(qidx=ev["qidx_def"] + k)
         for k in range(DEFAULT_WINDOW_Q)],
        ignore_index=True,
    )
    return reps.drop(columns="qidx_def")


# ---------------------------------------------------------------------------
# Live-neighbor aggregates (degree + mean stats)
# ---------------------------------------------------------------------------

def _node_stats(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[KEY + NEIGHBOR_STATS].copy()


def _aggregate_neighbors(pairs: pd.DataFrame, stats: pd.DataFrame,
                         prefix: str) -> pd.DataFrame:
    """pairs: (gvkey, gvkey_nbr, year, quarter) -> per-firm neighbor aggregates."""
    merged = pairs.merge(
        stats.rename(columns={"gvkey": "gvkey_nbr"}),
        on=["gvkey_nbr", "year", "quarter"],
        how="inner",
    )
    agg = merged.groupby(KEY).agg(
        degree=("gvkey_nbr", "size"),
        **{f"nbr_{c}": (c, "mean") for c in NEIGHBOR_STATS},
    )
    return agg.add_prefix(f"{prefix}_").reset_index()


def _bidirectional(edges: pd.DataFrame) -> pd.DataFrame:
    fwd = edges.rename(columns={"gvkey_1": "gvkey", "gvkey_2": "gvkey_nbr"})
    rev = edges.rename(columns={"gvkey_2": "gvkey", "gvkey_1": "gvkey_nbr"})
    cols = ["gvkey", "gvkey_nbr", "year", "quarter"]
    return pd.concat([fwd[cols], rev[cols]], ignore_index=True)


def _sparse_layer(stats: pd.DataFrame, events: pd.DataFrame, fname: str,
                  prefix: str, directed: tuple[str, str] | None = None,
                  year_min: int = 1988) -> pd.DataFrame:
    """One edge-list layer: live aggregates per year + event-based default counts."""
    cols = (list(directed) if directed else ["gvkey_1", "gvkey_2"]) + ["year", "quarter"]
    edges = pd.read_parquet(EDGES / fname, columns=cols)
    edges = edges[edges["year"] >= year_min - (MAX_EXIT_GAP_Q // 4)]
    if directed:
        src, dst = directed
        pairs = edges.rename(columns={src: "gvkey", dst: "gvkey_nbr"})
        pairs = pairs[["gvkey", "gvkey_nbr", "year", "quarter"]]
    else:
        pairs = _bidirectional(edges)
    del edges

    # live aggregates, chunked by year to bound merge memory
    blocks = []
    for year in sorted(pairs.loc[pairs["year"] >= year_min, "year"].unique()):
        blocks.append(_aggregate_neighbors(pairs[pairs["year"] == year],
                                           stats[stats["year"] == year], prefix))
    out = pd.concat(blocks, ignore_index=True)

    # default events: neighbors of each defaulter at its last active quarter
    dp = _with_qidx(pairs[pairs["gvkey"].isin(events["gvkey"])])
    nbrs_at_exit = dp.merge(events, left_on=["gvkey", "qidx"],
                            right_on=["gvkey", "qidx_last"])
    hits = _expand_event_window(nbrs_at_exit, ["gvkey_nbr"])
    cnt = (hits.groupby(["gvkey_nbr", "qidx"]).size()
           .rename(f"{prefix}_nbr_def_cnt").reset_index()
           .rename(columns={"gvkey_nbr": "gvkey"}))

    out = _with_qidx(out).merge(cnt, on=["gvkey", "qidx"], how="left")
    out[f"{prefix}_nbr_def_cnt"] = out[f"{prefix}_nbr_def_cnt"].fillna(0).astype("int32")
    out[f"{prefix}_nbr_def_share"] = (
        out[f"{prefix}_nbr_def_cnt"] / out[f"{prefix}_degree"].clip(lower=1))
    out = out.drop(columns="qidx")
    print(f"  {prefix}: {len(out):,} firm-quarters, "
          f"{int((out[f'{prefix}_nbr_def_cnt'] > 0).sum()):,} with a recent neighbor default")
    return out


# ---------------------------------------------------------------------------
# Clique layers: aggregate by group; exclude self via (sum - own) / (n - 1)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _industry_membership() -> pd.DataFrame:
    """(gvkey, year) -> sic4, sic3 exactly as phase3_33_34 defines them."""
    fy = pd.read_parquet(CLEAN / "firm_years.parquet",
                         columns=["gvkey", "datadate", "sich", "sic"])
    fy["year"] = pd.to_datetime(fy["datadate"]).dt.year.astype("int16")
    fy["gvkey"] = fy["gvkey"].astype(int)
    fy["sic_use"] = fy["sich"].fillna(fy["sic"])
    fy = fy.dropna(subset=["sic_use"]).drop_duplicates(["gvkey", "year"])
    fy["sic4"] = fy["sic_use"].astype(int).astype(str).str.zfill(4)
    fy["sic3"] = fy["sic4"].str[:3]
    return fy[["gvkey", "year", "sic4", "sic3"]]


@lru_cache(maxsize=1)
def _state_membership() -> pd.DataFrame:
    """(gvkey) -> headquarters state, as phase3_35 defines it."""
    fu = pd.read_parquet(CLEAN / "firm_universe.parquet").reset_index()
    fu["gvkey"] = fu["gvkey"].astype(int)
    return fu.dropna(subset=["state"])[["gvkey", "state"]]


def _attach_group(df: pd.DataFrame, member: pd.DataFrame, group_col: str) -> pd.DataFrame:
    on = [c for c in ["gvkey", "year"] if c in member.columns]
    return df.merge(member[on + [group_col]], on=on, how="inner").dropna(subset=[group_col])


def _clique_default_counts(events: pd.DataFrame, member: pd.DataFrame,
                           group_col: str) -> pd.DataFrame:
    """(group, qidx) -> number of group members that defaulted in the window."""
    ev = events.copy()
    ev["year"] = ev["qidx_last"] // 4
    ev["quarter"] = ev["qidx_last"] % 4 + 1
    ev = _attach_group(ev, member, group_col)  # defaulter's group at its last quarter
    hits = _expand_event_window(ev, [group_col, "gvkey"])
    return hits  # per (group, defaulter gvkey, affected qidx)


def _clique_layer(stats: pd.DataFrame, events: pd.DataFrame, member: pd.DataFrame,
                  group_col: str, prefix: str) -> pd.DataFrame:
    df = _attach_group(stats, member, group_col)

    gkey = [group_col, "year", "quarter"]
    grp = df.groupby(gkey)
    size = grp["gvkey"].transform("size")

    out = df[KEY].copy()
    out[f"{prefix}_degree"] = (size - 1).astype("int32")
    for c in NEIGHBOR_STATS:
        total = grp[c].transform("sum")
        n = grp[c].transform("count")
        own = df[c].notna().astype(int)
        denom = (n - own).replace(0, np.nan)
        out[f"{prefix}_nbr_{c}"] = (total - df[c].fillna(0)) / denom
    out[group_col] = df[group_col].values
    out = out[out[f"{prefix}_degree"] > 0]

    # event-based default counts within the group, excluding own default
    hits = _clique_default_counts(events, member, group_col)
    cnt = (hits.groupby([group_col, "qidx"]).size()
           .rename("grp_def_cnt").reset_index())
    own_hits = hits.rename(columns={"gvkey": "own_gvkey"})

    out = _with_qidx(out).merge(cnt, on=[group_col, "qidx"], how="left")
    out["grp_def_cnt"] = out["grp_def_cnt"].fillna(0)
    own = out.merge(
        own_hits[[group_col, "own_gvkey", "qidx"]].drop_duplicates(),
        left_on=["gvkey", group_col, "qidx"],
        right_on=["own_gvkey", group_col, "qidx"],
        how="left", indicator=True)["_merge"].eq("both").astype(int)
    out[f"{prefix}_nbr_def_cnt"] = (out["grp_def_cnt"] - own.values).clip(lower=0).astype("int32")
    out[f"{prefix}_nbr_def_share"] = (
        out[f"{prefix}_nbr_def_cnt"] / out[f"{prefix}_degree"].clip(lower=1))
    out = out.drop(columns=["qidx", group_col, "grp_def_cnt"])
    print(f"  {prefix}: {len(out):,} firm-quarters, "
          f"{int((out[f'{prefix}_nbr_def_cnt'] > 0).sum()):,} with a recent neighbor default")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_neighbor_features(panel: pd.DataFrame, force: bool = False,
                            year_min: int = 1988) -> pd.DataFrame:
    """Neighbor features for every row of the (standardized) quarterly panel.

    `panel` must carry gvkey/datadate/year/quarter and NEIGHBOR_STATS columns.
    Firm-quarters with no neighbors in a layer get NaN stats (XGBoost handles
    these natively) and degree/default counts of 0. Cached; force=True rebuilds.
    """
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    stats = _node_stats(panel[panel["year"] >= year_min])
    events = _default_events(panel)
    print(f"node stats: {len(stats):,} firm-quarters (year >= {year_min})")

    blocks = [
        _sparse_layer(stats, events, "supply_chain_edges.parquet", "supply_cust",
                      directed=("supplier_gvkey", "customer_gvkey"), year_min=year_min),
        _sparse_layer(stats, events, "supply_chain_edges.parquet", "supply_supp",
                      directed=("customer_gvkey", "supplier_gvkey"), year_min=year_min),
        _sparse_layer(stats, events, "creditor_edges.parquet", "creditor",
                      year_min=year_min),
        _sparse_layer(stats, events, "board_interlock_edges.parquet", "board",
                      year_min=year_min),
        _clique_layer(stats, events, _industry_membership(), "sic4", "ind4"),
        _clique_layer(stats, events, _industry_membership(), "sic3", "ind3"),
        _clique_layer(stats, events, _state_membership(), "state", "state"),
    ]

    out = panel.loc[panel["year"] >= year_min, KEY].copy()
    for block in blocks:
        out = out.merge(block, on=KEY, how="left")

    # absent from a layer => degree 0 / count 0, not NaN
    for c in out.columns:
        if c.endswith("_degree") or c.endswith("_nbr_def_cnt"):
            out[c] = out[c].fillna(0).astype("int32")
        elif c.endswith("_nbr_def_share"):
            out[c] = out[c].fillna(0.0)

    out.to_parquet(CACHE, index=False)
    print(f"cached {out.shape} -> {CACHE}")
    return out


NEIGHBOR_FEATURE_PREFIXES = ["supply_cust", "supply_supp", "creditor", "board",
                             "ind4", "ind3", "state"]


def neighbor_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if any(c.startswith(p + "_") for p in NEIGHBOR_FEATURE_PREFIXES)]
