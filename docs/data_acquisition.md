# Data Acquisition Guide

This project integrates data from **WRDS** (academic-licensed), **FRED**
(public domain), and the **LoPucki Bankruptcy Research Database** (free with
attribution). The repository ships acquisition scripts and configuration but
**does not redistribute the WRDS-derived raw files** — see [LICENSE-DATA.md](../LICENSE-DATA.md).

> **Note for reproducers:** raw CSVs from WRDS sources are
> not included in this repository. The scripts and documentation below let a
> user with WRDS access (such as a university student or affiliate) regenerate the
> entire raw layer. FRED files and the LoPucki download are included
> directly.

## Run order summary

The full pipeline runs in seven phases. Phase 0 is data acquisition and is
documented here; Phases 1–6 are documented in `docs/data_dictionary.md` and
chained in [`Snakefile`](../Snakefile).

| Step | Script / notebook | Purpose |
|---|---|---|
| 0a | `scripts/fred/download_fred.py` | Download 27 FRED macro series + Treasury curve |
| 0b | `scripts/wrds/download_wrds.py` | Download WRDS tables (requires credentials) |
| 0c | (manual download) | LoPucki Cases Table from <https://lopucki.law.ufl.edu/> |
| 1  | `data/clean/firm_cleanup.ipynb` | Build the firm universe & firm-year panel |
| 2  | `scripts/processing/phase2_annual_features.py` | Annual node features |
| 2  | `scripts/processing/phase2_quarterly_features.py` | Quarterly node features |
| 2  | `scripts/processing/phase2_standardize.py` | Winsorize + z-score |
| 3  | `scripts/processing/phase3_3{1..7}*.py` | Edge layers (7 scripts, one per layer) |
| 3  | `scripts/processing/phase3_38_validate.py` | Edge validation summary |
| 4  | `scripts/processing/phase4_default_labels.py` | LoPucki + Compustat default labels |
| 4b | `scripts/processing/split_lopucki_only.py` | Redistributable LoPucki-only subset |
| 6  | `scripts/processing/phase6_bond_map.py` | TRACE bond → gvkey mapping |
| 6  | `scripts/processing/phase6_stream_spreads.py` | Trade-level credit spreads |
| 6  | `scripts/processing/phase6_aggregate.py` | Quarterly/annual spreads + log_credit_spread feature |
| 5  | `scripts/processing/phase5_graph_assembly.py` | 261 quarterly HeteroData snapshots |

Phase 6 must run **before** Phase 5 because Phase 5 reads the credit-spread
augmented node-feature parquets.

## Section A — WRDS datasets (academic license, NOT redistributable)

All WRDS extracts are produced by [`scripts/wrds/download_wrds.py`](../scripts/wrds/download_wrds.py),
driven by [`scripts/wrds/wrds_config.yaml`](../scripts/wrds/wrds_config.yaml)
and per-table column lists in [`scripts/wrds/_columns/`](../scripts/wrds/_columns/).
The script logs every download to `data/raw/wrds_download_log.json`.

**Credentials.** The WRDS PostgreSQL connection is established by
[`scripts/wrds/wrds_client.py`](../scripts/wrds/wrds_client.py), which uses the
standard `wrds` Python package. Set up `~/.pgpass` per WRDS instructions
(<https://wrds-www.wharton.upenn.edu/pages/grid-items/python-from-your-computer/>),
or set `WRDS_USERNAME` in the environment.

| # | Dataset | Library / table | Output path | Approx. size | Date range |
|---|---|---|---|---|---|
| 1 | Compustat North America Annual (Fundamentals Annual) | `comp.funda` | `data/raw/compustat/compustat_CIQ_yearly.csv` | 1.5 GB | 1950 – 2024 |
| 2 | Compustat North America Quarterly (Fundamentals Quarterly) | `comp.fundq` | `data/raw/compustat/compustat_CIQ_quarterly.csv` | 2.0 GB | 1961 – 2024 |
| 3 | Compustat Historical Segments (merged) | `comp_segments_hist_daily.wrds_segmerged` | `data/raw/compustat/comp_segments_hist_daily.csv` | 42 MB | 1976 – 2024 |
| 4 | Compustat Historical Segments — Customer | `comp_segments_hist_daily.wrds_seg_customer` | `data/raw/compustat/comp_segments_hist_daily_customer.csv` | 30 MB | 1976 – 2024 |
| 5 | CRSP Monthly Stock (msf ⨝ msenames ⨝ msedelist) | `crsp.msf` etc. | `data/raw/crsp/CRSP.csv` | 42 MB | 1925 – 2024 |
| 6 | CRSP-Compustat Merged link table | `crsp.ccmxpf_linktable` | `data/raw/crsp/crsp_a_ccm.csv` | 4.3 MB | n/a (snapshot) |
| 7 | Bond–CRSP link table | `wrdsapps.bondcrsp_link` | `data/raw/crsp/Bond_CRSP_link.csv` | 2.7 MB | n/a (snapshot) |
| 8 | TRACE Enhanced (BTDS trade-by-trade) | `trace.trace_enhanced` | `data/raw/trace/trace_standard_BTDS.csv` | 4.3 GB | 2002-07 – 2026-02 |
| 9 | TRACE Master File | `trace.camasterfile` | `data/raw/trace/trace_standard_master_file.csv` | 2.7 GB | n/a (bond reference) |
| 10 | LSEG / Refinitiv DealScan (loan facilities, lenders) | `dealscan.dealscan` | `data/raw/LSEG/LSEG_Dealscan.csv` | 1.4 GB | 1980s – 2024 |
| 11 | Dealscan–Compustat Linking Database (Roberts link) | manual download (Excel) | `data/raw/LSEG/Dealscan-Compustat_Linking_Database012024.xlsx` | 1.5 MB | n/a (snapshot) |
| 12 | WRDS Supply Chain link (Compustat segments → gvkey) | `wrdsapps_link_supplychain.seglink` | `data/raw/WRDS_linking/wrdsapps_link_supplychain.csv` | 18 MB | n/a (snapshot) |
| 13 | BoardEx — Networks (associations) | `boardex.na_wrds_company_networks` | `data/raw/bordex/bordex_networks_associations.csv` | varies | 2000 – 2024 |
| 14 | BoardEx — Organization Analytics | `boardex.na_wrds_org_summary` | `data/raw/bordex/bordex_organization_analytics.csv` | varies | 2000 – 2024 |
| 15 | BoardEx — Officers / Composition | `boardex.na_wrds_org_composition` | `data/raw/bordex/bordex_organization_officers.csv` | varies | 2000 – 2024 |
| 16 | FactSet Revere — Companies (historical) | `factsamp_revere.wrds_company_hist` | `data/raw/factset/factset_revere_companies.csv` | varies | 2003 – 2024 |
| 17 | FactSet Revere — Relationships | `factsamp_revere.wrds_relationship` | `data/raw/factset/factset_revere_relationships.csv` | varies | 2003 – 2024 |
| 18 | Bureau van Dijk Orbis — Subsidiaries (large extract) | `bvd.ob_all_subs_first_level_lms` | `data/raw/moodys_orbis/bvd_orbis_large_subsidiaries.csv` | 146 KB filtered | snapshot |

**Run command.**

```bash
python scripts/wrds/download_wrds.py
```

By default this iterates every entry under `datasets:` in the YAML. Pipeline
behavior:

1. Resolve the columns to `SELECT` from `_columns/<columns_file>`.
2. Build a SQL query (with `where` and `join_sql` from the YAML) and either
   execute it as a single round-trip (`single_query`), chunk by year
   (`by_year`), chunk by year+month (`by_year_month`), or stream via pandas
   `chunksize` (`pandas_chunksize`).
3. Append rows to `output_root/<output_path>` and log row counts plus a
   SHA-256 of the resulting CSV to `wrds_download_log.json`.

Item 11 (the Dealscan-Compustat / Roberts link) is distributed as an Excel
file via the WRDS portal and is not available as a SQL table — see the
`manual_artifacts:` section in `wrds_config.yaml`.

## Section B — FRED (public domain, scripted)

All FRED extracts are produced by [`scripts/fred/download_fred.py`](../scripts/fred/download_fred.py),
driven by [`scripts/fred/fred_config.yaml`](../scripts/fred/fred_config.yaml).
**FRED files ARE redistributable** and are included in this repository under
`data/raw/fred/`.

**API key.** Free key required from
<https://fred.stlouisfed.org/docs/api/api_key.html>. The script reads it from
`keys/fed_cc_key.txt` (one line, no quotes). The key file is gitignored.

**Series list.** 27 series across five categories, covering 1980-01 → 2024-12.

| Category | Series IDs |
|---|---|
| Monetary policy & rates | `FEDFUNDS`, `GS10`, `T10Y2Y` |
| Credit conditions | `BAA10Y`, `BAMLH0A0HYM2`, `BAMLC0A4CBBB`, `DBAA`, `DAAA` |
| Market stress | `VIXCLS`, `SP500` |
| Real economy | `A191RL1Q225SBEA` (real GDP growth, quarterly), `UNRATE`, `INDPRO` |
| Banking & liquidity | `TEDRATE`, `TB3MS` |
| Treasury curve (additional) | `GS1`, `GS2`, `GS3`, `GS5`, `GS7`, `GS10`, `GS20`, `GS30` (under `data/raw/fred/treasury_curve/`) |

**Outputs.**

- One CSV per series in `data/raw/fred/<SERIES_ID>.csv`
- Treasury-curve tenors under `data/raw/fred/treasury_curve/`
- Combined master file `data/raw/fred/fred_master.csv` (all series resampled to
  a common monthly grid; daily series are averaged within month, quarterly
  series are forward-filled across the three months of each quarter)
- `data/raw/fred/download_log.json` audit trail

**Run command.**

```bash
python scripts/fred/download_fred.py
```

## Section C — LoPucki Bankruptcy Research Database (manual download)

The Florida-UCLA LoPucki Bankruptcy Research Database is **free with
attribution**. Required citation:

> Lynn M. LoPucki, *UCLA-LoPucki Bankruptcy Research Database*,
> <http://lopucki.law.ufl.edu/>.

**Manual steps to acquire.**

1. Open <https://lopucki.law.ufl.edu/> in a browser.
2. Register (free) and log in.
3. Download the **Cases Table** as both CSV and XLSX, plus the codebook
   (`Read Me First.pdf`, `User's manual.pdf`, `Protocols.pdf`).
4. Drop the unzipped folder under `data/raw/lopucki/`. Expected layout:
   ```
   data/raw/lopucki/Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023/
       Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv
       Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.xlsx
       BRD Import.do
       BRD Labels.do
       Read Me First.pdf
       User's manual.pdf
       Protocols.pdf
   ```
   The 1-12-2023 release matches the version this pipeline was built against
   (1,218 cases). Newer releases will work but will give different counts.

**Output of the acquisition step.** The raw download is **redistributable** as-is
and is included in this repository (≈ 4 MB). It is consumed by
`scripts/processing/phase4_default_labels.py`, which:

- Matches LoPucki cases to Compustat `gvkey` via `GvkeyBefore` (primary), CIK,
  and normalized name fallback (1,138 / 1,218 = 93.4% match rate).
- Combines matched LoPucki bankruptcies with Compustat `dlrsn ∈ {2, 3}`
  delisting supplements to produce 2,594 unique default events.

**Redistributable derivative.** [`scripts/processing/split_lopucki_only.py`](../scripts/processing/split_lopucki_only.py)
filters the combined `default_events.parquet` to rows sourced from LoPucki
(`source IN ('lopucki', 'both')`), drops the Compustat-sourced
`compustat_dldte` column, and replaces `default_date` with the pure LoPucki
`lopucki_filing_date`. Output:

- `data/processed/default_events_lopucki_only.parquet` (1,055 rows, 1980–2022).
  Included in `data/MANIFEST.sha256`.

## Phase 1 manual notebook step

[`data/clean/firm_cleanup.ipynb`](../data/clean/firm_cleanup.ipynb) constructs
the firm universe from the raw Compustat / CRSP files. It is intentionally a
notebook because most of its logic is exploratory data-quality work
(de-duplicating GVKEY-PERMNO links, dropping single-year and sub-threshold
firms, classifying nonfinancial vs. financial via SIC code).

Run it interactively or non-interactively before invoking the Snakemake
pipeline:

```bash
jupyter nbconvert --to notebook --execute \
    data/clean/firm_cleanup.ipynb --inplace
```

Outputs (consumed by all downstream phases):

- `data/clean/firm_universe.parquet` — 32,135 rows × 30 cols
- `data/clean/firm_years.parquet` — 441,934 firm-years × 981 Compustat fields
- `data/clean/single_year_firms.parquet`
- `data/clean/sub_treshold_firms.parquet`
- `data/clean/phase1_summary.json`

The Phase 2 notebooks under `data/clean/phase2_*.ipynb` are alternate
exploratory entry points to the Phase 2 logic. The committed `.py` scripts
under `scripts/processing/phase2_*.py` are the standalone executable
equivalents wired into the Snakefile.
