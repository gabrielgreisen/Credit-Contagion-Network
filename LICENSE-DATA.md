# Data Licensing

This project integrates 21 source datasets under three different licensing
regimes. Code is unreleased (all rights reserved); data is mixed and is described
below. **Read this file before redistributing anything from `data/`.**

## 1. Sources that are NOT redistributable (academic license)

The following source datasets are licensed for academic use through Wharton
Research Data Services (WRDS) and may **not** be redistributed in raw or
derivative form. Anyone reproducing this pipeline must re-acquire them through
their own institutional WRDS subscription.

| Dataset | Licensing entity |
|---|---|
| Compustat North America Annual (Fundamentals Annual) | S&P Global / Wharton |
| Compustat North America Quarterly (Fundamentals Quarterly) | S&P Global / Wharton |
| Compustat Historical Segments (Customer & Geographic) | S&P Global / Wharton |
| Compustat-CRSP Merged Link Table (CCM) | S&P Global / CRSP / Wharton |
| CRSP Monthly Stock | CRSP / University of Chicago |
| CRSP Bond–CRSP Link | CRSP / University of Chicago |
| TRACE Enhanced (BTDS trade-by-trade) | FINRA / Wharton |
| TRACE Master File | FINRA / Wharton |
| LSEG / Refinitiv DealScan (loan facilities, lenders) | LSEG / Refinitiv / Wharton |
| Roberts DealScan-Compustat Link | LSEG / Refinitiv / Wharton |
| WRDS BoardEx (Networks, Analytics) | WRDS / BoardEx / Management Diagnostics |
| FactSet Revere (supply chain relationships, trial access) | FactSet |
| Bureau van Dijk Orbis Subsidiaries (large-firm extract) | Bureau van Dijk / Moody's |

### Derivative files that are also non-redistributable

Any artifact built from one or more of the sources above inherits their
licensing restrictions. The following files in this repository were produced
from non-redistributable sources and may **not** be shared:

- `data/clean/firm_universe.parquet`
- `data/clean/firm_years.parquet`
- `data/clean/single_year_firms.parquet`
- `data/clean/sub_treshold_firms.parquet`
- `data/clean/node_features_raw.parquet`
- `data/clean/node_features_standardized.parquet`
- `data/clean/node_features_quarterly.parquet`
- `data/clean/node_features_quarterly_standardized.parquet`
- `data/clean/edges/supply_chain_edges.parquet`
- `data/clean/edges/creditor_edges.parquet`
- `data/clean/edges/industry_4digit_edges.parquet`
- `data/clean/edges/industry_3digit_edges.parquet`
- `data/clean/edges/geographic_edges.parquet`
- `data/clean/edges/ownership_edges.parquet`
- `data/clean/edges/board_interlock_edges.parquet`
- `data/clean/default_events.parquet` (combined LoPucki + Compustat dlrsn)
- `data/clean/default_labels_annual.parquet`
- `data/clean/default_labels_quarterly.parquet`
- `data/clean/credit_spreads_annual.parquet`
- `data/clean/credit_spreads_quarterly.parquet`
- `data/clean/trace_bond_firm_map.parquet`
- `data/clean/trace_trade_spreads.parquet`
- `data/clean/graph_snapshots/quarterly/*.pt` (all 261 quarterly HeteroData snapshots)
- `data/clean/node_index_maps/*.parquet`
- `data/clean/split_assignments.parquet`

These files are excluded from version control via `.gitignore` and are not
included in the SHA-256 manifest of redistributable files
(`data/MANIFEST.sha256`).

## 2. Sources that ARE redistributable

### LoPucki Bankruptcy Research Database (free with attribution)

The Florida–UCLA-LoPucki Bankruptcy Research Database is provided free of
charge for academic and policy research with an attribution requirement.
Source: <https://lopucki.law.ufl.edu/>

**Required citation:**

> Lynn M. LoPucki, *UCLA-LoPucki Bankruptcy Research Database*,
> <http://lopucki.law.ufl.edu/>.

Redistributable files in this repository:

- The raw LoPucki download under `data/raw/lopucki/`
- `data/processed/default_events_lopucki_only.parquet` — the LoPucki-only
  subset of matched default events (1,055 rows). Compustat-sourced columns
  (`compustat_dldte`) and the combined `default_date` (which mixed the LoPucki
  filing date with the Compustat delisting date) have been replaced with the
  pure LoPucki filing date so no Compustat information is carried over. See
  `scripts/processing/split_lopucki_only.py` for the exact filter.

### FRED — Federal Reserve Economic Data (public domain)

FRED time series are works of the U.S. federal government and are in the
public domain. Source: <https://fred.stlouisfed.org/>

**Suggested citation:**

> Federal Reserve Bank of St. Louis, *Federal Reserve Economic Data (FRED)*,
> <https://fred.stlouisfed.org/>.

Redistributable files in this repository:

- All CSVs under `data/raw/fred/` (27 macro series + Treasury curve tenors +
  `fred_master.csv`)

These are also covered by the SHA-256 manifest.

## 3. Code

All source code, configuration files, scripts, documentation, notebooks, and
metadata files in this repository — including everything under `scripts/`,
`node_processing/`, `docs/`, this file, the `Snakefile`, and `metadata.json` —
are unreleased work, all rights reserved. No public license is granted.

This section covers the code only. The data terms in sections 1 and 2 are
imposed by the upstream providers and apply regardless.
