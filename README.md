# Credit Contagion and Corporate Default Risk

## Summary

This project builds a curated, reproducible data pipeline for studying corporate default risk and credit contagion. The motivating idea is that firms do not fail in isolation. Corporate credit risk is usually studied at the level of an individual firm, looking at leverage, profitability, liquidity, and other firm-specific financial indicators. Major economic events such as the 2008 financial crisis and the COVID-19 supply chain disruptions show that financial distress can spread across connected firms, industries, creditors, and markets.

The primary artifact of this project is a multiplex temporal graph dataset that integrates 21 source datasets into a single curated artifact. The dataset combines firm-level financial data, macroeconomic indicators, bankruptcy records, stock market data, syndicated loan data, supply chain relationships, ownership ties, and board interlocks. The primary sources are Compustat fundamentals from WRDS, FRED macroeconomic indicators from the Federal Reserve Bank of St. Louis, the LoPucki Bankruptcy Research Database, and CRSP monthly stock data. Additional sources include DealScan, TRACE, BoardEx, FactSet Revere, Orbis Subsidiaries, the Roberts DealScan-Compustat link, the Compustat-CRSP link, the Bond CRSP link, and Compustat Historical Segments.

The research questions focus on how firm financial characteristics relate to bankruptcy outcomes, how macroeconomic stress changes the profile of firms that default, and whether an integrated panel dataset can reveal patterns consistent with credit contagion. In other words, the project tries to understand whether distress is only explained by a firm's own financial health, or whether broader economic conditions and firm connections also matter.

A central part of the project is the data curation process itself. The datasets come from different sources, use different identifiers, have different access rules, and are structured at different levels of detail. Compustat is firm-year financial data, FRED is macroeconomic time series data, and LoPucki is bankruptcy event-level data. The project required cleaning, identifier matching, frequency alignment, license-aware partitioning, and decisions about how to preserve records without making irreversible cuts too early.

The final artifact is a set of 261 quarterly multiplex graph snapshots covering 32,135 firms and 441,934 firm-years from 1950 through 2025. Each snapshot contains 53 standardized node features (accounting ratios, market signals, macroeconomic controls, and TRACE-derived credit spreads), seven edge layers covering distinct contagion channels (supply chain, common creditor, industry 4-digit, industry 3-digit, geographic, ownership, board interlock), default labels at multiple horizons, and credit spread regression targets. The dataset is suitable for downstream analysis of corporate default risk, including network-aware predictive modeling. The headline findings include a 32,135-firm universe with a 26,656 / 5,479 nonfinancial-financial split, 25,726 firms successfully linked to CRSP (~80% match rate), 2,594 default events combining a 93.4% LoPucki match rate with Compustat delisting supplements, and 486 million edges across the seven contagion layers.

This project demonstrates how difficult corporate default research becomes when trying to make the data reproducible and transparent. Even before any modeling work, there are many decisions about what data to include, how to match firms across sources, how to handle missing values, and how to document data that cannot be redistributed publicly because of licensing restrictions.

## Data Profile

### Compustat Corporate Fundamentals

The Compustat North America Annual and Quarterly Fundamentals files provide firm-level financial data for public companies, including total assets, total debt, revenue, net income, current assets, current liabilities, SIC industry code, and other accounting values. The primary identifier is GVKEY, which allows firm records to be tracked across time. Compustat is the backbone of the firm universe: leverage, liquidity, profitability, size, growth, and cash-flow features are all derived from Compustat fields. These variables describe whether firms that eventually enter bankruptcy look financially weaker than firms that survive.

The annual file produced 441,934 firm-year observations after standard quality filtering, covering fiscal years 1950 through 2025. The quarterly file produced 1,556,311 firm-quarter observations and is the panel granularity used for the final node feature matrix, because quarterly fundamentals capture intra-year deterioration that annual snapshots miss (a firm can look healthy in December and file for bankruptcy by June). Quarterly Compustat uses a parallel set of mnemonics with a `q` suffix (`atq` for total assets, `saleq` for revenue, etc.) and required a separate field-mapping pass during feature engineering.

Compustat was accessed through WRDS under an academic license. Because of the WRDS license terms, the raw Compustat files cannot be redistributed in this repository. The repository documents the acquisition process and expected file locations so that someone with WRDS access can reproduce the download. Acquisition is automated by `scripts/wrds/download_wrds.py` driven by `scripts/wrds/wrds_config.yaml`.

Repository locations:
- `scripts/wrds/download_wrds.py`, `scripts/wrds/wrds_config.yaml`
- `data/raw/` (WRDS-derived, not redistributed)
- `data/clean/firm_universe.parquet`, `data/clean/firm_years.parquet` (derivative, not redistributable)

### FRED Macroeconomic Indicators

The FRED dataset contains macroeconomic time series from the Federal Reserve Bank of St. Louis. The project uses 25 indicators covering interest rates, credit spreads, market volatility, GDP, unemployment, industrial production, the TED spread, the term spread, and money supply. These indicators provide the broader economic context surrounding firm distress.

The FRED data is structured differently from the firm-level data. It is organized as time series rather than firm-year records, and the variables appear at different frequencies. Some series are daily, some are monthly, and others are quarterly. The data therefore had to be aligned to the firm-quarter panel using explicit aggregation rules: rate variables use end-of-quarter values, while flow and level variables use period averages.

FRED data is in the public domain and is included in this repository.

Repository locations:
- `scripts/fred/download_fred.py`, `scripts/fred/fred_config.yaml`
- `data/raw/fred/` (included in repo, redistributable)
- `data/clean/fred_aligned_quarterly.parquet` (derivative)

### LoPucki Bankruptcy Research Database

The LoPucki Bankruptcy Research Database provides bankruptcy event records for major public-company Chapter 11 bankruptcies from October 1979 through December 2022. It includes company name, CIK, ticker, filing date, chapter, assets at filing, disposition date, and bankruptcy outcome. The dataset is used to identify firms that entered bankruptcy and connect those events back to firm-level financial information.

LoPucki does not include Compustat GVKEY identifiers, so LoPucki cases are matched to Compustat firms using a tiered matching strategy: CIK exact match first, then ticker match, then fuzzy company-name match with manual review of borderline cases. Matched events become the primary source of default labels because Compustat delisting codes alone miss many bankruptcies that are coded as mergers (for example, Lehman Brothers and WorldCom both have Compustat `dlrsn = 10` even though they filed for bankruptcy).

LoPucki is free to use with attribution. The cleaned LoPucki cases file is included in this repository.

Repository locations:
- `data/raw/lopucki/` (included in repo, redistributable)
- `data/clean/default_events.parquet` (combined with Compustat supplements, not redistributable)

### CRSP Monthly Stock File

CRSP provides market-based information about firms: monthly returns, prices, share counts, and delisting codes. Market data complements accounting data because it captures investor expectations and changes in firm value before default. CRSP is linked to Compustat through the Compustat-CRSP linking table, with link types filtered to high-confidence values (`linktype` in `LC` or `LU` and `linkprim` in `P` or `C`). About 80% of firms in the universe have a CRSP linkage; unmatched firms are flagged via `has_crsp` rather than removed.

CRSP is licensed through WRDS and cannot be redistributed.

Repository locations:
- `scripts/wrds/download_wrds.py` (acquisition)
- `data/clean/firm_universe.parquet` (linkage embedded, derivative)

### TRACE Corporate Bond Transactions

TRACE Enhanced contains transaction-level data for all public corporate bond trades reported to FINRA. The project uses TRACE in combination with the TRACE Master File (which provides bond-level metadata such as coupon, maturity, and CUSIP) and the Bond CRSP Link (which maps issuer CUSIPs to PERMNOs and then to GVKEYs). Together these three files allow trade-level data to be aggregated into firm-quarter median option-adjusted spreads, producing the credit spread node feature and prediction target. The cleaned credit spread panel covers approximately 139,000 firm-quarters, concentrated among large investment-grade and high-yield firms with active bond markets. TRACE Enhanced is the largest single dataset in the project (tens of gigabytes raw) and required chunked streaming during processing.

TRACE is licensed through WRDS and cannot be redistributed.

Repository locations:
- `scripts/processing/phase6_*.py` (bond mapping, spread streaming, aggregation)
- `data/clean/credit_spreads_quarterly.parquet` (derivative, not redistributable)

### Network-Layer Datasets

The project integrates additional WRDS sources to construct contagion-channel edges. These are part of the curated artifact rather than analytical add-ons.

- **WRDS Supply Chain with IDs** and **Compustat Historical Segments** provide customer-supplier disclosures used to build the supply chain edge layer.
- **FactSet Revere** (trial license, 2003–2015) supplements supply chain coverage for the post-2003 period.
- **DealScan**, linked to Compustat via the **Roberts DealScan-Compustat link**, provides syndicated loan facilities used to identify firms sharing common creditors.
- **BoardEx** provides director and executive employment histories used to build the board interlock layer (firms that share at least one director).
- **Orbis Subsidiaries** provides parent-subsidiary ownership relationships.
- **TRACE Enhanced** and the **TRACE Master File**, linked through the **Bond CRSP Link**, are used to compute firm-level credit spreads from corporate bond transactions.

All WRDS-licensed sources cannot be redistributed.

### Storage and Organization

All processed and intermediate outputs are stored as Apache Parquet files. Parquet was chosen over CSV for three reasons: column-oriented storage allows reading only the fields needed for a given analysis (the Compustat firm-year file has hundreds of columns but most analyses use a few dozen); built-in compression reduces footprint roughly 5–10× compared with CSV at no read-time cost; and Parquet preserves data types, which matters for fields like GVKEY (categorical string) and `dlrsn` (nullable integer code) where CSV round-trips silently coerce types and introduce bugs. The exception is the LoPucki source file, which arrives as Excel and is converted to CSV during cleaning to preserve human-readability of the small underlying record set.

The repository follows a three-tier directory layout. `data/raw/` holds original downloads from WRDS, FRED, and LoPucki, never modified. `data/clean/` holds cleaned and integrated parquet files plus the per-phase summary JSON files that record counts and validation checks. `data/clean/graph_snapshots/quarterly/` holds the 261 quarterly PyTorch Geometric `HeteroData` files in a `{YYYY}_{Qq}.pt` naming convention. Scripts are organized by lifecycle stage: `scripts/wrds/` and `scripts/fred/` for acquisition, `scripts/processing/` for the six phase scripts, `scripts/analysis/` for figure generation. This separation makes it easy to reason about provenance: anything in `data/clean/` was produced by a script in `scripts/processing/`, and anything in `scripts/processing/` reads only from `data/raw/` or earlier-phase outputs in `data/clean/`.

### Ethical and Legal Constraints

A major legal constraint is that most source datasets cannot be redistributed publicly. WRDS data, including Compustat, CRSP, TRACE, DealScan, BoardEx, FactSet Revere, Orbis Subsidiaries, and all WRDS linking tables, can only be used under the academic license. As a consequence, derivative files built from these sources, including the firm universe, node features, edge layers, and assembled graph snapshots, also cannot be redistributed. Per-source licensing is documented in detail in `LICENSE-DATA.md`.

This project does not use individual-level personal data or personally identifiable information. All data describes public companies, financial markets, bankruptcy records, or macroeconomic conditions. Even so, ethical data handling matters because financial datasets shape decisions about risk, investment, and credit. The project documents its sources, limitations, and cleaning decisions so that anyone using the resulting dataset understands its provenance and constraints.

## Data Quality

The data quality assessment focused on completeness, consistency, validity, identifier integrity, and integration readiness. Because the project uses many datasets from different sources, the binding question is not just whether each dataset is clean by itself, but whether the datasets can be linked together correctly.

For Compustat, the main quality issues were missing values, duplicate firm-date records, and inconsistent representation of corporate structure. Many financial variables are missing for smaller firms or earlier time periods, which creates problems for derived ratios such as debt-to-assets, current ratio, and profitability measures. Duplicate `GVKEY × datadate` pairs would corrupt the firm-year panel structure and downstream merges, so the cleaning process explicitly deduplicates on this key. Quality filters retain only consolidated statements (`consol = 'C'`), standard format (`datafmt = 'STD'`), domestic population (`popsrc = 'D'`), and USD reporting (`curcd = 'USD'`).

The CRSP-Compustat link required quality filtering. Not every possible link between CRSP and Compustat is equally reliable, so the project filtered to high-confidence link types and prime links. This produced a firm-level match rate of approximately 80% (25,726 of 32,135 firms matched). Unmatched firms were flagged with `has_crsp = False` rather than deleted, so that downstream analyses can choose to include or exclude them as appropriate.

For FRED, the main quality concern was frequency mismatch. The series are not all measured at the same frequency: some are daily, some monthly, some quarterly. If handled incorrectly, macroeconomic values could be matched to the wrong firm-quarter. The project addressed this with explicit aggregation rules where rate variables use end-of-quarter values and flow or level variables use period averages.

For LoPucki, the main quality issue was identifier mismatch. Because LoPucki does not include GVKEY identifiers, company names had to be matched to Compustat records, with the risk of false matches or missed matches because company names may change, use abbreviations, or include different legal suffixes. The project used a tiered matching approach (CIK first, then ticker, then fuzzy name match), achieved a 93.4% match rate (1,138 of 1,218 cases matched), and saved unmatched cases for manual review.

For the network-layer datasets, the main quality issues were coverage gaps and identifier translation. The supply chain layer suffers from a known coverage gap before 2003 because FactSet Revere starts in 2003 and historical Compustat customer disclosures are sparse. The DealScan-Compustat link via Roberts loses some borrowers that don't have GVKEYs assigned. BoardEx company IDs and Orbis BvD IDs require translation to GVKEYs before edges can be constructed. Each translation step is documented and the resulting match rates are recorded.

Overall, the quality assessment showed that the data is usable but only after careful documentation and cleaning. The main quality issues are missing values, identifier mismatches, frequency differences, large file sizes, and licensing restrictions that affect reproducibility. A summary of integration-readiness checks (firm-year deduplication, CRSP linkage rate, LoPucki match rate, FRED frequency alignment, edge-layer coverage by year) is captured in the per-phase summary JSON files under `data/clean/` and is referenced from the data dictionary so that quality metrics travel with the artifact rather than living only in interactive notebooks.

## Data Cleaning

The data cleaning process followed the principle that records should not be deleted unless there is a clear and documented reason. Because this project is about building a reproducible data pipeline, it was important to preserve as much information as possible and document any decisions that changed the data.

For Compustat, the first cleaning step was applying standard quality filters (consolidated statements, standard format, domestic population, USD reporting). Records that were completely empty across the major financial fields (`at`, `sale`, `revt` all null simultaneously) were removed because they did not provide useful information. Financial firms with SIC codes 6000–6999 were not removed; they were kept as a separate `node_class` because financial firms play an important role in credit contagion through common-creditor channels. Removing them would lose an important part of the financial network. Firms with only one year of data were also not permanently deleted; they were set aside in `single_year_firms.parquet`. Single-year firms cannot be used for year-over-year features but may be useful for cross-sectional snapshots, so the pipeline preserves them rather than dropping them irreversibly.

For the CRSP-Compustat merge, the project filtered the linking table to high-confidence link types (`linktype` in `LC` or `LU` and `linkprim` in `P` or `C`). Open-ended link spells (where the link end date is null because the link is still active) were handled by filling missing end dates with a far-future placeholder so that interval matching works correctly. Firms that did not match CRSP were flagged with `has_crsp = False` rather than removed.

For FRED, cleaning focused on temporal alignment. The project created aggregation rules for converting macroeconomic indicators to the quarterly frequency used by the firm panel. Rate variables were aligned using end-of-quarter values; flow and level variables were aligned using period averages. These rules are documented in `scripts/fred/fred_config.yaml`.

For missing values in node features, the project used a tiered approach. Variables with low missingness are forward-filled within firms. Variables with moderate missingness are kept with null indicators so that downstream analyses can account for the fact that the value was missing rather than imputed. Variables with very high missingness are excluded from the core feature set but preserved in the raw data so that the decision is reversible.

For LoPucki, the cleaning steps were converting the original Excel source file into CSV format, normalizing company names (stripping legal suffixes such as "Inc.", "Corp.", "LLC", uppercasing, removing punctuation), and applying the tiered matching strategy described above. Each candidate match was further constrained to require the LoPucki filing date to fall within the firm's active period in the firm universe; this prevented false positives where a current Compustat firm shared a normalized name with a long-defunct bankruptcy filer. Unmatched cases were saved to `lopucki_unmatched.csv` for transparency.

When the same GVKEY appeared in both LoPucki and the Compustat delisting supplements, the project took the **earliest** of the two dates (LoPucki filing date or Compustat `dldte`) as the canonical default date and recorded `source = 'both'`. The median gap between the two dates was 535 days, with LoPucki almost always preceding the Compustat deletion — consistent with the bankruptcy filing happening months before the formal delisting. Taking the earlier date avoids labeling a firm as "not yet defaulted" during a quarter when it had already filed for Chapter 11.

For composite scores, Altman Z-score is mathematically defined for nonfinancial firms only — its inputs (working capital, retained earnings, EBIT, market value of equity, sales, all over total assets) interact pathologically with the leverage structures of banks, insurers, and REITs. The cleaning pass therefore computes Z-score only for `node_class == 'nonfinancial'` and writes null for financial firms. The same logic applies to interest coverage where `xint` is zero or null for many financial firms.

For node feature standardization, all numeric features were z-scored by quarter so that downstream analyses see relative cross-sectional positions rather than raw scale. Outliers were winsorized at the 1st and 99th percentiles by quarter to prevent extreme values from dominating; size variables (log assets, log sales, log market cap) were exempted from winsorization since their distributions are already log-transformed.

Two source-scope decisions were made on file-size grounds. The Orbis Shareholders file was dropped after determining that its size was disproportionate to its usefulness here; ownership edges are built from Orbis Subsidiaries instead. TRACE Enhanced and Orbis Subsidiaries were retained but require chunked processing rather than whole-file loads, which is why their acquisition and processing steps stream in batches.

For edge construction, deduplication was applied within each layer: a directed firm pair appears at most once per quarter, with weights aggregated when the same relationship was observed across multiple sources or facilities. Self-loops (a firm linked to itself through any channel) were removed. The supply chain layer in particular required source-priority logic because WRDS Supply Chain, Compustat Historical Segments, and FactSet Revere can all assert the same customer-supplier relationship; the pipeline keeps the earliest disclosed quarter and records the source set rather than double-counting. The common creditor layer was constructed by joining DealScan facilities on shared lender IDs, then aggregating across all facilities active in a given quarter to produce a single edge weight per firm pair.

## Findings

The output of this project is the curated dataset itself rather than a downstream model, so the findings describe what was built and the patterns visible in the curated data.

**Firm coverage.** The pipeline produced a firm universe of 32,135 unique firms covering 441,934 firm-year observations from 1950 through 2025. Coverage rises steeply from the 1950s through the 1990s, peaks at roughly 10,500 firm-year observations annually around 1996–2000, and gradually declines through the 2000s and 2010s as public-company delistings outpace new IPOs. See `results/figures/firm_year_coverage.png`. Of the 32,135 firms, 26,656 are nonfinancial and 5,479 are financial (SIC 6000–6799), an 83/17 split visible in `results/figures/node_class_split.png`. About 25,726 firms (~80%) successfully link to CRSP monthly stock data through the Compustat-CRSP linking table.

**Industry composition.** The single largest industry by firm count is State Commercial Banks (SIC 6020, 1,509 firms), reflecting the long history of US bank consolidation captured in Compustat. Biological Products (2836), Computer Services (7370), Prepackaged Software (7372), and Crude Petroleum & Natural Gas (1311) round out the top five, each with over 1,000 firms. See `results/figures/top_industries.png`. The distribution is consistent with prior literature on the Compustat universe.

**Default events.** The pipeline produced 2,594 default events combining 1,138 LoPucki matches (93.4% match rate against 1,218 LoPucki cases) with 1,539 Compustat-derived events from delisting reason codes 2 (bankruptcy) and 3 (liquidation). Two hundred events appear in both sources. The annual default rate visible in the panel varies substantially over time, with elevated rates during the early 1970s tail of the secondary banking crisis, the 2001 dot-com bust (1.12%), the 2009 financial crisis peak (1.68% — Lehman filed in September 2008 but the bankruptcy cluster including Chrysler, GM, and CIT clustered in 2009), and the 2020 COVID-19 period (0.85%). The long-run mean over 1970–2024 is 0.61%. See `results/figures/default_rate_over_time.png`. The temporal variation in the default rate is the signal that any downstream credit-risk model would need to learn.

**Edge layer composition.** The graph component contains 486,134,670 edges across seven contagion-channel layers. Geographic (firms in the same state) is by far the largest at 294 million edges; industry 4-digit and 3-digit SIC follow with 78 and 66 million respectively; common creditor (firms sharing a syndicated lender via DealScan) contains 46 million; board interlock contains 1.7 million; supply chain contains 532 thousand; ownership contains a small number of parent-subsidiary edges from Orbis. See `results/figures/edge_layer_composition.png`. The relative sizes are consistent with the underlying coverage of each source — industry and geographic are dense by construction, while supply chain and ownership reflect specific disclosed relationships.

**Credit spreads as a contagion signal.** The credit spread sub-pipeline computes firm-quarter median bond option-adjusted spreads from TRACE and links them to firms via the Bond CRSP Link. Coverage is 139,000 firm-quarters, concentrated among large investment-grade and high-yield firms with active bond markets. The Lehman Brothers Holdings (GVKEY 6669) trajectory illustrates what the curated data captures. From 2006Q1 through 2007Q1, Lehman's median bond OAS fluctuates between roughly 125 and 160 basis points. It then reaches 295 bps by 2007Q3, 626 bps by 2007Q4, 904 bps by 2008Q1, and 893 bps by 2008Q3 (just before the September 15 Chapter 11 filing). See `results/figures/lehman_spread_trajectory.png`. The trajectory shows the bond-market view of firm distress at quarterly resolution, anchored to the same GVKEY as the accounting and network data.

**Final artifact.** The curated graph dataset comprises 261 quarterly snapshots from 1961Q1 through 2026Q1, totaling approximately 12.95 GB. Each snapshot is a PyTorch Geometric `HeteroData` object containing 53 standardized node features per firm, seven edge layers (eight edge types when reverse edges are counted), default labels at 1-quarter, 4-quarter, and 8-quarter horizons, and the credit spread regression target.

## Future Work

Several extensions are natural next steps now that the curated dataset is in place.

**Pre-2003 supply chain coverage.** The most immediate data-curation gap is the supply chain layer before 2003. FactSet Revere coverage starts in 2003, and Compustat's historical customer disclosures are sparse. A future extension could systematically parse SEC 10-K filings from EDGAR for major-customer disclosures (firms must report customers representing more than 10% of revenue), entity-resolve those disclosures back to GVKEYs, and backfill the supply chain layer for the 1990–2002 period. This is a substantial natural-language-processing and entity-resolution effort, but it would close the largest known gap in the dataset.

**Notebook-to-script extraction.** Phases 1 and 2 currently exist as Jupyter notebooks under `data/clean/`. The remaining phases (3, 4, 5, 6) are Python scripts. A future cleanup pass should extract the notebooks to scripts so that the entire pipeline can be invoked through Snakemake without manual notebook execution. The Snakefile in this repository documents the intended chain and runs the parts that already exist as scripts.

**Bloomberg SPLC and Capital IQ supplements.** Bloomberg's SPLC supply chain function and Capital IQ's customer relationship data could provide additional supply chain coverage, particularly for international firms not well captured by SEC disclosures. Both are subject to licensing constraints similar to WRDS, but they would extend the data horizontally.

**Downstream modeling.** With the curated graph dataset in place, the natural downstream step is graph-based predictive modeling (for example, graph neural networks for default prediction or credit spread forecasting). This is out of scope for the current curation effort but is the motivating use case that drove the project's network-layer data integration choices.

**Edge-attribute enrichment.** The current edge layers carry minimal attributes (presence and a coarse weight where applicable). Future work could enrich each layer with provenance and strength metadata: for supply chain, the percentage-of-revenue threshold met by each disclosed customer relationship; for common creditor, the dollar-weighted shared exposure rather than a count of shared lenders; for board interlock, the count and seniority of shared directors; for ownership, the percentage held by the parent. Richer edge attributes would let downstream analyses weight contagion channels more precisely and would make the curated artifact more useful as a general-purpose research dataset.

**Broader default definition.** The current default label captures Chapter 7 and Chapter 11 bankruptcies plus Compustat liquidation codes. Credit research often uses a broader definition that also includes credit rating downgrades to D or SD, distressed exchanges, and missed coupon payments. Moody's Default and Recovery Database and S&P's CreditPro contain the necessary records but require separate licensing. Incorporating these would roughly double the labeled-event count and reduce the class imbalance that makes default prediction difficult.

**External validation against published default series.** The 2,594 events in the panel can be compared against published US corporate default counts from Moody's annual default studies and the NYU Salomon Center's bankruptcy database. Cross-checking the 2009 peak rate of 1.68% against these external benchmarks would either validate the curation choices or surface systematic biases (for example, LoPucki's $100M-asset floor causes the panel to undercount smaller defaults). A short validation notebook with year-by-year comparison plots would be a useful addition.

**International expansion.** The current dataset is US-only. Compustat Global, BoardEx International, and Orbis cover non-US firms with the same identifier conventions used here, so the pipeline would extend naturally to Europe, Asia, and Latin America with relatively modest changes — primarily expanding the FRED macro panel to include country-specific series and adding currency normalization. Cross-border supply chain and ownership edges become particularly interesting in a global panel because they capture contagion channels that purely domestic networks miss.

## Reproducing

Reproducing the pipeline requires WRDS access through a subscribing institution, a free FRED API key, and a manual LoPucki download. The full reproduction sequence is:

**1. Clone the repository.**
```bash
git clone https://github.com/gabrielgreisen/Credit-Contagion-Network.git
cd Credit-Contagion-Network
```

**2. Create a Python environment and install dependencies.**
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**3. Acquire the WRDS data.** Configure your WRDS credentials per WRDS instructions, then run the WRDS download script:
```bash
python scripts/wrds/download_wrds.py
```
This pulls Compustat Annual, Compustat Quarterly, CRSP Monthly, DealScan, the Roberts DealScan-Compustat link, the Compustat-CRSP link, Compustat Historical Segments, FactSet Revere (trial), BoardEx, Orbis Subsidiaries, TRACE Enhanced, the TRACE Master File, and the Bond CRSP Link. Files land under `data/raw/`. Approximate total size is several tens of gigabytes; expect long download times for TRACE and Orbis. Not all files are required for every downstream phase; partial pipelines are possible.

**4. Acquire the FRED data.** Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html and export it as `FRED_API_KEY`, then run:
```bash
python scripts/fred/download_fred.py
```
The 25 series are listed in `scripts/fred/fred_config.yaml`. Files land under `data/raw/fred/`. FRED files are public domain and are also already included in the repository, so this step is optional for reproduction.

**5. Acquire the LoPucki data.** Visit https://lopucki.law.ufl.edu/, register for free access, navigate to "Data access — Download cases table", and download the Cases Table Excel file to `data/raw/lopucki/`. The cleaned LoPucki file is also already included in the repository.

**6. Run Phases 1 and 2 (Jupyter notebooks).** These phases are currently notebook-based. From the project root:
```bash
jupyter notebook data/clean/firm_cleanup.ipynb           # Phase 1
jupyter notebook data/clean/phase2_node_features.ipynb   # Phase 2 (annual)
jupyter notebook data/clean/phase2_node_features_quarterly.ipynb
jupyter notebook data/clean/phase2_standardization.ipynb
```
Run all cells in each notebook in order. Outputs land under `data/clean/`.

**7. Run Phases 3–6 via the Snakefile.** From the project root:
```bash
snakemake --cores 4
```
This chains the Phase 3 edge construction, Phase 4 default labels, Phase 6 credit spreads, and Phase 5 graph assembly scripts under `scripts/processing/`. Final outputs are the 261 quarterly graph snapshots under `data/clean/graph_snapshots/quarterly/` plus their metadata. The Snakefile header documents how to run partial pipelines (e.g., `snakemake --until phase4`).

**8. Regenerate Findings figures (optional).**
```bash
python scripts/analysis/figures.py
```
Outputs the six PNGs and `figure_index.md` under `results/figures/`.

Detailed acquisition documentation for each of the 21 source datasets, including license terms and exact file paths, is in `docs/data_acquisition.md`. The data dictionary describing every column of every output file, every edge layer schema, and the HeteroData graph schema is in `docs/data_dictionary.md`. Per-source licensing and the rules around derivative-file redistribution are in `LICENSE-DATA.md`.

## References

### Datasets

- Compustat North America Fundamentals (Annual and Quarterly). Standard & Poor's / Wharton Research Data Services. Accessed through an academic WRDS subscription.
- CRSP Monthly Stock File. Center for Research in Security Prices / WRDS. Accessed through an academic WRDS subscription.
- DealScan. Refinitiv / WRDS.
- TRACE Enhanced and TRACE Master File. Financial Industry Regulatory Authority / WRDS.
- BoardEx. Wharton Research Data Services.
- FactSet Revere. FactSet Research Systems (trial license, 2003–2015).
- Orbis Subsidiaries. Bureau van Dijk / Moody's Analytics.
- Compustat Historical Segments, Compustat-CRSP Linking Table, Bond CRSP Link, Roberts DealScan-Compustat Link. WRDS.
- Federal Reserve Bank of St. Louis. Federal Reserve Economic Data (FRED). https://fred.stlouisfed.org/. Public domain.
- LoPucki, L. M. UCLA-LoPucki Bankruptcy Research Database. http://lopucki.law.ufl.edu/. Free with attribution.

### Software

- Python Software Foundation. Python (version 3.11+). https://www.python.org/
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*. (pandas)
- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585, 357–362.
- Apache Software Foundation. Apache Parquet. https://parquet.apache.org/
- Köster, J., & Rahmann, S. (2012). Snakemake — a scalable bioinformatics workflow engine. *Bioinformatics* 28(19), 2520–2522.
- Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems* 32.
- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*.
- Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. *Computing in Science & Engineering* 9(3), 90–95.

### Methodological Background

- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance* 23(4), 589–609. (Z-score)
- Merton, R. C. (1974). On the pricing of corporate debt: the risk structure of interest rates. *The Journal of Finance* 29(2), 449–470. (Distance-to-default)
- Roberts, M. R. & Sufi, A. (2009). Renegotiation of financial contracts: evidence from private credit agreements. *Journal of Financial Economics* 93(2), 159–184. (DealScan-Compustat link)
