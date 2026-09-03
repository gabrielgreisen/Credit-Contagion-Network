# Data Dictionary

This document describes every artifact produced by the pipeline. The dataset
spans **1950–2025** (firm panel) and **1961-Q1 through 2026-Q1** (quarterly
graph snapshots). All firm identifiers are Compustat `gvkey` (int64).

Columns sourced directly from Compustat are cited by their Compustat
mnemonic; the WRDS Compustat data dictionary
(<https://wrds-www.wharton.upenn.edu/>) is authoritative for full definitions.

---

## Section 1 — Tabular outputs

### 1.1  `data/clean/firm_universe.parquet`

**Purpose:** Authoritative list of firms in the dataset. One row per gvkey.
**Shape:** 32,135 rows × 30 columns.

| Column | Type | Source | Description |
|---|---|---|---|
| `gvkey` | int64 | Compustat | Compustat firm identifier (primary key). |
| `conm` | string | Compustat | Company name (uppercase). |
| `conml` | string | Compustat | Company legal name (mixed case). |
| `tic` | string | Compustat | Ticker symbol (most recent). |
| `cusip` | string | Compustat | CUSIP (most recent 9-character). |
| `cik` | float (nullable) | Compustat | SEC Central Index Key. |
| `exchg` | float (nullable) | Compustat | Exchange code (e.g., 11=NYSE, 14=NASDAQ). |
| `sic` | int64 | Compustat | Standard Industrial Classification code (modal across history). |
| `sich_mode` | float (nullable) | Compustat | Modal historical SIC across firm-years. |
| `sich_recent` | float (nullable) | Compustat | Most recent historical SIC. |
| `naics` | float (nullable) | Compustat | NAICS code (most recent). |
| `naicsh` | float (nullable) | Compustat | NAICS historical. |
| `gsector`, `ggroup`, `gind`, `gsubind` | float (nullable) | Compustat | GICS sector / industry-group / industry / sub-industry. |
| `state`, `city`, `addzip` | string | Compustat | Headquarters state, city, ZIP. |
| `fic`, `loc` | string | Compustat | Country of incorporation, country of HQ. |
| `costat` | string | Compustat | Active (`A`) or inactive (`I`). |
| `dldte` | string | Compustat | Delisting date (Compustat). |
| `dlrsn` | float (nullable) | Compustat | Delisting reason code (`2`=bankruptcy, `3`=liquidation). |
| `ipodate` | string | Compustat | IPO date. |
| `first_year`, `last_year` | int64 | Derived | First/last fiscal year present in `firm_years`. |
| `year_count` | int64 | Derived | Number of fiscal years observed. |
| `node_class` | string | Derived | `nonfinancial` (26,656) or `financial` (5,479) — partition by SIC code (financials = SIC 6000–6799). |
| `permno` | float (nullable) | CRSP | CRSP-Compustat-Merged link permno (most recent). |
| `has_crsp` | bool | Derived | True if firm has a valid CRSP link. |

### 1.2  `data/clean/firm_years.parquet`

**Purpose:** Firm-year panel (Compustat Fundamentals Annual filtered to
`indfmt IN ('INDL','FS') AND datafmt='STD' AND popsrc='D' AND consol='C'`),
joined with CRSP descriptive fields. Used as input to feature engineering.

**Shape:** 441,934 rows × 981 columns. The full Compustat
[`funda`](https://wrds-www.wharton.upenn.edu/) field set is preserved; the
columns most relevant downstream are listed below. Refer to the Compustat
data dictionary for the remainder.

| Column | Type | Description |
|---|---|---|
| `gvkey` | int64 | Firm identifier. |
| `datadate` | datetime | Fiscal year-end date. |
| `fyear` | int (nullable) | Fiscal year. |
| `at` | float | Total assets, USD millions. |
| `lt` | float | Total liabilities, USD millions. |
| `sale`, `revt` | float | Sales / revenue. |
| `ni`, `ib` | float | Net income / income before extraordinary items. |
| `ebit`, `ebitda` | float | Earnings before interest & tax / depreciation. |
| `oancf` | float | Operating cash flow. |
| `capx` | float | Capital expenditures. |
| `dlc`, `dltt` | float | Debt in current liabilities / long-term debt. |
| `che` | float | Cash and short-term investments. |
| `act`, `lct` | float | Current assets / current liabilities. |
| `ceq` | float | Common equity. |
| `csho`, `prcc_f` | float | Common shares outstanding / fiscal-year close price (used for market cap). |
| `re` | float | Retained earnings. |
| `xint` | float | Interest expense. |
| `emp` | float | Employees (thousands). |
| `sic`, `sich`, `state` | various | Identifiers as in `firm_universe`. |
| `dldte`, `dlrsn` | various | Delisting date / reason. |
| `node_type` | string | `financial` if SIC ∈ [6000, 6799] else `nonfinancial`. |
| ...remaining 950+ Compustat columns | various | See Compustat `funda` documentation. |

### 1.3  `data/clean/node_features_quarterly_standardized.parquet`

**Purpose:** Standardized quarterly node features used by Phase 5 graph
assembly. One row per gvkey × fiscal quarter.

**Shape:** 1,556,311 rows × 69 columns. 53 of those columns are the model
feature set (the others are identifiers, targets, and intermediate fields
excluded by Phase 5; see `phase2_summary.json` and `phase5_summary.json`).

**Standardization.** Continuous fundamentals are winsorized at the 1st/99th
percentile within each fiscal-quarter cross-section, then z-scored
(within-quarter mean=0, std=1). Macro variables and Altman Z-Score are
attached *raw* — they are global time series and have meaningful units.

#### Identifiers and metadata (excluded from `x` tensor)

| Column | Type | Description |
|---|---|---|
| `gvkey` | int64 | Firm identifier. |
| `datadate` | datetime | Quarter-end date. |
| `fyearq`, `fqtr` | int | Fiscal year and quarter. |
| `conm` | string | Company name. |
| `sic` | int | SIC code. |
| `node_type` | string | `financial` / `nonfinancial`. |
| `permno`, `has_crsp` | various | CRSP link. |
| `total_debt`, `ebit_proxy` | float | Intermediate quantities used in feature construction. |
| `altman_zone`, `altman_variant` | string | Distress zone label, Z / Z′ / Z″ variant flag. |

#### Feature columns (53; included in `x` tensor of every snapshot)

**Leverage** — `debt_to_assets`, `debt_to_equity`, `lt_debt_ratio`,
`st_debt_ratio`, `interest_coverage`, `st_debt_share`. Built from Compustat
`dlc`, `dltt`, `at`, `ceq`, `xint`. See [`node_processing/levereage_solvency.py`](../node_processing/levereage_solvency.py).

**Profitability** — `roa` (`ni/at`), `roe` (`ni/ceq`), `ebitda_margin`
(`ebitda/sale`), `gross_margin` (`gp/sale`), `operational_margin`
(`oiadp/sale`). See [`node_processing/profitability.py`](../node_processing/profitability.py).

**Liquidity** — `current_ratio` (`act/lct`), `quick_ratio`
((`act – invt`)/`lct`), `cash_to_assets` (`che/at`), `wc_to_assets`
(working capital / `at`). See [`node_processing/liquidity.py`](../node_processing/liquidity.py).

**Size** — `log_assets` (`log(at)`), `log_revenue` (`log(sale)`), `log_mktcap`
(`log(csho × prcc_f)`), `emp` (employees). See [`node_processing/size_features.py`](../node_processing/size_features.py).

**Activity / efficiency** — `asset_turnover` (`sale/at`),
`receivables_turnover` (`sale/rect`), `inventory_turnover` (`cogs/invt`).
See [`node_processing/activity_efficiency.py`](../node_processing/activity_efficiency.py).

**Growth (year-over-year)** — `revenue_growth`, `asset_growth`, `emp_growth`.
See [`node_processing/growth.py`](../node_processing/growth.py).

**Cash flow** — `opcf_to_assets` (`oancf/at`), `capex_to_assets` (`capx/at`),
`fcf_to_assets` ((`oancf – capx`) / `at`). See [`node_processing/cash_flow.py`](../node_processing/cash_flow.py).

**Composite distress score** — `altman_z` (Altman Z-Score; uses Z, Z′, or Z″
depending on firm type — see [`node_processing/composite_scores.py`](../node_processing/composite_scores.py)).

**Market (from CRSP)** — `ret_12m` (12-month trailing return), `ret_3m`
(3-month trailing), `volatility_12m` (rolling 12-month return σ),
`share_turnover`, `log_avg_volume`, `log_mktcap_crsp`, `market_to_book`.

**Macro (FRED, attached as-of)** — `FEDFUNDS`, `GS10`, `T10Y2Y`, `BAA10Y`,
`BAMLH0A0HYM2`, `BAMLC0A4CBBB`, `DBAA`, `DAAA`, `BAA_AAA_spread` (derived),
`VIXCLS`, `sp500_ret_12m` (derived), `A191RL1Q225SBEA` (real GDP growth),
`UNRATE`, `INDPRO`, `TEDRATE`, `TB3MS`. See [`scripts/fred/fred_config.yaml`](../scripts/fred/fred_config.yaml).

**Credit spread** — `log_credit_spread` (z-scored log of the median
TRACE-derived OAS for the quarter; see Section 1.6).

#### Targets (excluded from `x`, attached as labels)

| Column | Type | Description |
|---|---|---|
| `median_spread_bps` | float32 | Median bond OAS over the quarter, basis points. |
| `n_bonds` | float | Number of unique bonds contributing to the spread. |

### 1.4  `data/clean/default_events.parquet` (combined; NOT redistributable)

**Purpose:** Unique default events from LoPucki (matched) plus Compustat
delisting supplements (`dlrsn ∈ {2, 3}`).

**Shape:** 2,594 rows × 10 columns. **By source:** compustat 1,539 ·
lopucki 855 · both 200.

| Column | Type | Description |
|---|---|---|
| `gvkey` | int64 | Firm identifier. |
| `default_date` | datetime | Earliest of LoPucki filing date and Compustat dldte. |
| `default_type` | string | `bankruptcy_ch11`, `bankruptcy_ch7`, `bankruptcy_other`, `bankruptcy_compustat`, or `liquidation`. Compustat-derived for source=compustat rows. |
| `chapter` | string | LoPucki Chapter (`7`, `11`, `15`, etc.); null for Compustat-only. |
| `outcome` | string | LoPucki Disposition (`Confirmed Plan`, `Acquired`, `Dismissed`, etc.). |
| `source` | string | `lopucki`, `compustat`, or `both`. |
| `lopucki_filing_date` | datetime | LoPucki DateFiled. |
| `compustat_dldte` | datetime | Compustat delisting date (Compustat-licensed). |
| `assets_at_filing` | float | LoPucki AssetsPetition (USD millions). |
| `NameCorp` | string | LoPucki NameCorp (corporate name at filing). |

### 1.5  `data/processed/default_events_lopucki_only.parquet` (redistributable)

**Purpose:** LoPucki-only subset of `default_events.parquet`, stripped of
Compustat-sourced columns. **Redistributable** (LoPucki license: free with
attribution).

**Shape:** 1,055 rows × 9 columns. **By source:** lopucki 855 · both 200
(no compustat-only rows).

Same schema as `default_events.parquet` but **without** `compustat_dldte`,
and with `default_date := lopucki_filing_date` so the date column is purely
LoPucki-sourced. See [`scripts/processing/split_lopucki_only.py`](../scripts/processing/split_lopucki_only.py).

### 1.6  `data/clean/default_labels_quarterly.parquet`

**Purpose:** Quarterly forward-looking default indicators for every active
firm-quarter.

**Shape:** 1,540,647 rows × 5 columns. **Positives:** 793 (1q), 5,004 (4q),
12,558 (8q).

| Column | Type | Description |
|---|---|---|
| `gvkey` | int64 | Firm identifier. |
| `datadate` | datetime | Fiscal quarter-end date (the as-of date). |
| `default_next_1q` | int8 | 1 if firm defaults within the next 1 quarter, else 0. |
| `default_next_4q` | int8 | 1 if within next 4 quarters. |
| `default_next_8q` | int8 | 1 if within next 8 quarters. |

`data/clean/default_labels_annual.parquet` has the same structure with
`fyear` and `default_next_1y` / `default_next_2y` columns (438,085 rows).

### 1.7  `data/clean/credit_spreads_quarterly.parquet`

**Purpose:** Quarterly aggregation of TRACE-derived bond spreads at the
firm level, used as the regression target and as a feature.

**Shape:** 139,624 firm-quarters × 12 columns. Coverage: 6.5% of firm
quarters (3,668 unique firms with at least one spread observation,
2002-Q3 to 2026-Q1).

| Column | Type | Description |
|---|---|---|
| `gvkey` | int32 | Firm identifier. |
| `year`, `quarter` | int | Calendar year / quarter. |
| `datadate` | datetime | Quarter-end date. |
| `median_spread_bps` | float | Median trade-level OAS over the quarter, basis points. |
| `vw_median_spread_bps` | float | Volume-weighted median OAS. |
| `mean_spread_bps`, `spread_std_bps` | float | Mean and standard deviation of OAS. |
| `n_bonds` | int64 | Distinct CUSIPs contributing this quarter. |
| `n_trades` | int64 | Total trade count this quarter. |
| `total_volume` | double | Total face-value volume traded. |
| `log_credit_spread` | float | `log(median_spread_bps)` standardized within-quarter (z-scored). |

---

## Section 2 — Edge layers

All edge files live under `data/clean/edges/`. Edges are produced quarterly
or as static peer-group memberships keyed by `(year, quarter)`, and are
materialized into HeteroData snapshots in Phase 5 along with their reverse
edges (Phase 5 produces 8 edge types in HeteroData; see Section 3).

### 2.1 Supply chain — `supply_chain_edges.parquet`

- **Source:** Compustat Historical Segments (Customer file: `wrds_seg_customer`),
  combined with the WRDS Supply Chain link table (`wrdsapps_link_supplychain`)
  to resolve customer names to gvkey.
- **Directionality:** Directed (supplier → customer).
- **Rows:** 531,816 quarterly edges.
- **Construction:** Phase 3.1
  ([`scripts/processing/phase3_31_supply_chain.py`](../scripts/processing/phase3_31_supply_chain.py)).

| Column | Type | Description |
|---|---|---|
| `supplier_gvkey` | int64 | Source firm. |
| `customer_gvkey` | int64 | Target firm. |
| `year`, `quarter` | int64 | Calendar year / quarter the relationship is active. |
| `salecs` | float | Sales to customer, USD millions. |
| `customer_concentration` | float | Share of supplier's total sales going to this customer (0–1). |
| `relationship_duration` | int64 | Number of consecutive quarters the relationship has been observed. |
| `source` | string | Provenance flag (segment-disclosure source). |

### 2.2 Common creditor — `creditor_edges.parquet`

- **Source:** LSEG / Refinitiv DealScan facilities, with lender resolution via
  the Roberts Dealscan-Compustat link table.
- **Directionality:** Undirected.
- **Rows:** 45,536,702 quarterly edges. (Largest layer by edge count.)
- **Construction:** Phase 3.2
  ([`scripts/processing/phase3_32_creditor.py`](../scripts/processing/phase3_32_creditor.py)).
  Two firms are connected in quarter `(y, q)` if both have an active
  syndicated facility (start date ≤ q-end < end date) sharing at least one
  lead arranger.

| Column | Type | Description |
|---|---|---|
| `gvkey_1`, `gvkey_2` | int32 | Firm identifiers (gvkey_1 < gvkey_2). |
| `year`, `quarter` | int | Calendar year / quarter. |
| `shared_lender_count` | int16 | Number of distinct shared lenders. |
| `shared_exposure` | double | Sum of shared facility amounts (USD millions). |
| `shared_lead_arranger` | bool | True if the shared lender is a lead arranger. |

### 2.3 Industry 4-digit SIC — `industry_4digit_edges.parquet`

- **Source:** Compustat `sic` (4-digit). **Directionality:** Undirected.
- **Rows:** 78,408,260.
- **Construction:** Phase 3.3
  ([`scripts/processing/phase3_33_34_industry.py`](../scripts/processing/phase3_33_34_industry.py)).
  Two firms are connected each quarter if they share the same 4-digit SIC.

| Column | Type | Description |
|---|---|---|
| `gvkey_1`, `gvkey_2` | int32 | Firm identifiers. |
| `year`, `quarter` | int | Calendar year / quarter. |
| `sic_4digit` | string | The shared SIC code. |

### 2.4 Industry 3-digit SIC — `industry_3digit_edges.parquet`

- **Source:** Compustat `sic` (3-digit prefix). **Directionality:** Undirected.
- **Rows:** 65,597,236. Excludes firms already linked at 4-digit (so this layer
  captures the *additional* peer-group from the broader 3-digit grouping).
- **Construction:** Phase 3.4 (same script as 3.3).

| Column | Type | Description |
|---|---|---|
| `gvkey_1`, `gvkey_2` | int32 | Firm identifiers. |
| `year`, `quarter` | int | Calendar year / quarter. |
| `sic_3digit` | string | Shared 3-digit SIC prefix. |

### 2.5 Geographic — `geographic_edges.parquet`

- **Source:** Compustat `state` (HQ state). **Directionality:** Undirected.
- **Rows:** 294,384,508. (Second-largest layer.)
- **Construction:** Phase 3.5
  ([`scripts/processing/phase3_35_geographic.py`](../scripts/processing/phase3_35_geographic.py)).
  Two firms are connected each quarter if they share an HQ state.

| Column | Type | Description |
|---|---|---|
| `gvkey_1`, `gvkey_2` | int32 | Firm identifiers. |
| `year`, `quarter` | int | Calendar year / quarter. |
| `state` | string | The shared US state code. |

### 2.6 Ownership — `ownership_edges.parquet`

- **Source:** Bureau van Dijk Orbis Subsidiaries (large-firm extract).
- **Directionality:** Directed (parent → subsidiary).
- **Rows:** 0 currently (the BvD extract did not yield gvkey-resolvable
  parent–subsidiary links above the size threshold; the layer is materialized
  as an empty edge table for HeteroData compatibility).
- **Construction:** Phase 3.6
  ([`scripts/processing/phase3_36_ownership.py`](../scripts/processing/phase3_36_ownership.py)).

| Column | Type | Description |
|---|---|---|
| `parent_gvkey`, `subsidiary_gvkey` | int (nullable) | Firm identifiers. |
| `year`, `quarter` | int (nullable) | Calendar year / quarter. |
| `ownership_pct` | float (nullable) | Equity ownership share (0–1). |

### 2.7 Board interlock — `board_interlock_edges.parquet`

- **Source:** WRDS BoardEx (Networks + Organization Composition).
- **Directionality:** Undirected.
- **Rows:** 1,676,148.
- **Construction:** Phase 3.7
  ([`scripts/processing/phase3_37_board.py`](../scripts/processing/phase3_37_board.py)).
  Two firms are connected each quarter if at least one director / officer is
  on both boards.

| Column | Type | Description |
|---|---|---|
| `gvkey_1`, `gvkey_2` | int32 | Firm identifiers. |
| `year`, `quarter` | int | Calendar year / quarter. |
| `shared_director_count` | int16 | Number of overlapping board members. |
| `shared_executive` | bool | True if at least one of the shared members is a C-suite executive. |

---

## Section 3 — Graph dataset schema

### 3.1 File layout

- **Format:** PyTorch Geometric [`HeteroData`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html)
  serialized via `torch.save`.
- **Snapshots:** one `.pt` per quarter, total **261 snapshots**.
- **Path:** `data/clean/graph_snapshots/quarterly/{YYYY}_Q{q}.pt`
  (e.g., `2007_Q3.pt`).
- **Index maps:** `data/clean/node_index_maps/{YYYY}_Q{q}.parquet` — maps the
  snapshot's contiguous node index to gvkey.
- **Total disk size:** 12.95 GB. Largest quarter: 1999-Q4 (10,463 nodes,
  137 MB).
- **Date range:** 1961-Q1 through 2026-Q1.

### 3.2 Node schema (single node type: `firm`)

Each snapshot has one node type. Per-node attributes:

| Attribute | Shape | dtype | Description |
|---|---|---|---|
| `x` | `(N, 53)` | float32 | The 53 standardized feature columns listed in [`graph_metadata.json`](../data/clean/graph_metadata.json) and Section 1.3 of this dictionary. |
| `x_mask` | `(N, 53)` | bool | True where the original value was non-null. NaNs in `x` are imputed to 0.0; the mask preserves missingness. |
| `y_default_1q` | `(N,)` | float32 | 1.0 if firm defaults within next 1 quarter, 0.0 otherwise; `-1.0` sentinel where label undefined. |
| `y_default_4q` | `(N,)` | float32 | Same, 4-quarter horizon. |
| `y_default_8q` | `(N,)` | float32 | Same, 8-quarter horizon. |
| `y_spread` | `(N,)` | float32 | Quarterly median credit spread (bps). `-1.0` sentinel for firms without bond spread coverage. |
| `gvkeys` | `(N,)` | int64 | Original Compustat gvkey for each node. |
| `datadates` | `(N,)` | int64 | Datetime of the as-of date. |
| `node_class` | `(N,)` | int8 | `0` = nonfinancial, `1` = financial. |
| `has_crsp` | `(N,)` | bool | True if firm has CRSP coverage. |

### 3.3 Edge schema (8 edge types)

The 7 directed source layers expand to **8** HeteroData edge types after
adding the reverse direction for the supply chain:

| Edge type | Directionality | Edge attributes |
|---|---|---|
| `('firm', 'supplies', 'firm')` | directed | `salecs`, `customer_concentration`, `relationship_duration` |
| `('firm', 'supplied_by', 'firm')` | directed (reverse of `supplies`) | same as `supplies` |
| `('firm', 'shares_creditor', 'firm')` | undirected | `shared_lender_count`, `shared_exposure`, `shared_lead_arranger` |
| `('firm', 'same_industry_4d', 'firm')` | undirected | (none — peer-group membership) |
| `('firm', 'same_industry_3d', 'firm')` | undirected | (none) |
| `('firm', 'same_state', 'firm')` | undirected | (none) |
| `('firm', 'owns', 'firm')` | directed | `ownership_pct` (currently empty) |
| `('firm', 'shares_director', 'firm')` | undirected | `shared_director_count`, `shared_executive` |

`edge_index` for every edge type is a `(2, E)` long tensor in PyG's standard
COO layout. Where the source layer carries edge attributes, they are stored
on `edge_attr` aligned with `edge_index`.

### 3.4 Train / val / test split (temporal)

The split is *temporal* — assignments are made by quarter, not by firm — so
that no future information leaks into training. Every snapshot carries an
implicit split tag derived from its calendar quarter.

| Split | Quarter range | Number of snapshots |
|---|---|---|
| `pretrain` | before 1990-Q1 | 116 |
| `train` | 1990-Q1 – 2006-Q4 | 68 |
| `val` | 2007-Q1 – 2012-Q4 | 24 |
| `test` | 2013-Q1 – 2024-Q4 | 53 |

The mapping from quarter to split is also written to
[`data/clean/split_assignments.parquet`](../data/clean/split_assignments.parquet)
(one row per snapshot with columns `year`, `quarter`, `split`,
`n_nodes`, `n_default_4q`, `n_default_4q_pos`).

Default rates by split (4-quarter horizon, from [`graph_metadata.json`](../data/clean/graph_metadata.json)):

| Split | Positives | Total | Rate |
|---|---|---|---|
| pretrain | 998 | 415,987 | 0.24% |
| train | 2,133 | 608,357 | 0.35% |
| val | 827 | 173,037 | 0.48% |
| test | 1,046 | 343,266 | 0.30% |

### 3.5 Auxiliary metadata files

- [`data/clean/graph_metadata.json`](../data/clean/graph_metadata.json) —
  full feature column list, edge-type definitions, edge-attr columns per
  type, split boundaries, NaN imputation policy.
- [`data/clean/phase5_summary.json`](../data/clean/phase5_summary.json) —
  per-snapshot statistics, connectivity diagnostics, edge-type counts.
