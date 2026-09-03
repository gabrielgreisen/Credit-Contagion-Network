# Snakefile — Credit Contagion Network pipeline
# ----------------------------------------------------------------
# Chains the data-processing phases that exist
# as standalone Python scripts in scripts/processing/ and the acquisition
# scripts in scripts/wrds/ and scripts/fred/.
#
# What this Snakefile DOES cover
# ------------------------------
#   - Phase 0 (data acquisition):  scripts/fred/download_fred.py
#                                  scripts/wrds/download_wrds.py
#   - Phase 2 (node features):     scripts/processing/phase2_annual_features.py
#                                  scripts/processing/phase2_quarterly_features.py
#                                  scripts/processing/phase2_standardize.py
#   - Phase 3 (edges):             scripts/processing/phase3_31_supply_chain.py
#                                  scripts/processing/phase3_32_creditor.py
#                                  scripts/processing/phase3_33_34_industry.py
#                                  scripts/processing/phase3_35_geographic.py
#                                  scripts/processing/phase3_36_ownership.py
#                                  scripts/processing/phase3_37_board.py
#                                  scripts/processing/phase3_38_validate.py
#   - Phase 4 (default labels):    scripts/processing/phase4_default_labels.py
#                                  scripts/processing/split_lopucki_only.py
#   - Phase 6 (credit spreads):    scripts/processing/phase6_bond_map.py
#                                  scripts/processing/phase6_stream_spreads.py
#                                  scripts/processing/phase6_aggregate.py
#   - Phase 5 (graph assembly):    scripts/processing/phase5_graph_assembly.py
#
# What this Snakefile does NOT cover
# ----------------------------------
#   - Phase 1 (firm universe construction) lives in a Jupyter notebook
#     at data/clean/firm_cleanup.ipynb. It reads the raw Compustat /
#     CRSP / link-table CSVs in data/raw/ and writes:
#         data/clean/firm_universe.parquet
#         data/clean/firm_years.parquet
#         data/clean/single_year_firms.parquet
#         data/clean/sub_treshold_firms.parquet
#         data/clean/phase1_summary.json
#     Run it manually before invoking this workflow:
#         jupyter nbconvert --to notebook --execute \
#             data/clean/firm_cleanup.ipynb --inplace
#     See docs/data_acquisition.md for the full manual run order.
#
#   - WRDS data acquisition is wrapped here as a single rule, but anyone
#     reproducing the pipeline must supply their own WRDS credentials
#     (see scripts/wrds/wrds_client.py). The raw CSVs are not
#     redistributed in this repository.
#
# Usage
# -----
#   snakemake all -j 4              # build the full graph dataset
#   snakemake --until phase4_done   # stop after default labels
#   snakemake --until phase3_done   # build only edges (no labels / spreads)
#   snakemake -n                    # dry run
#   snakemake --dag | dot -Tpng > dag.png   # generate dependency graph
# ----------------------------------------------------------------

from pathlib import Path

CLEAN = "data/clean"
EDGES = "data/clean/edges"
PROCESSED = "data/processed"
GRAPH = "data/clean/graph_snapshots/quarterly"

# Phase 1 outputs are produced manually by data/clean/firm_cleanup.ipynb.
# They are listed here as inputs (not outputs of any rule) so missing files
# raise a clear "produce these via the notebook first" error instead of an
# opaque MissingInputException.
FIRM_UNIVERSE = f"{CLEAN}/firm_universe.parquet"
FIRM_YEARS    = f"{CLEAN}/firm_years.parquet"

rule all:
    input:
        # Phase 5 sentinel — all 261 quarterly snapshots produced in one run
        f"{CLEAN}/graph_metadata.json",
        f"{CLEAN}/phase5_summary.json",
        # Redistributable LoPucki-only events
        f"{PROCESSED}/default_events_lopucki_only.parquet",

# ----------------------------------------------------------------
# Phase 0 — Data acquisition
# ----------------------------------------------------------------

rule fred_download:
    output:
        master = "data/raw/fred/fred_master.csv",
        log    = "data/raw/fred/download_log.json",
    shell:
        "python scripts/fred/download_fred.py"

rule wrds_download:
    # Produces all raw Compustat / CRSP CSVs under data/raw/. Requires WRDS
    # credentials. Listed here for documentation; phases below depend on the
    # individual CSVs, not on this rule.
    output:
        log = "data/raw/wrds_download_log.json",
    shell:
        "python scripts/wrds/download_wrds.py"

# ----------------------------------------------------------------
# Phase 2 — Node features
# ----------------------------------------------------------------

rule phase2_annual_features:
    input:
        FIRM_YEARS,
        "data/raw/fred/fred_master.csv",
    output:
        f"{CLEAN}/node_features_raw.parquet",
    shell:
        "python scripts/processing/phase2_annual_features.py"

rule phase2_quarterly_features:
    input:
        FIRM_UNIVERSE,
        "data/raw/compustat/compustat_CIQ_quarterly.csv",
        "data/raw/crsp/CRSP.csv",
        "data/raw/fred/fred_master.csv",
    output:
        f"{CLEAN}/node_features_quarterly.parquet",
    shell:
        "python scripts/processing/phase2_quarterly_features.py"

rule phase2_standardize:
    input:
        f"{CLEAN}/node_features_raw.parquet",
        f"{CLEAN}/node_features_quarterly.parquet",
    output:
        f"{CLEAN}/node_features_standardized.parquet",
        f"{CLEAN}/node_features_quarterly_standardized.parquet",
        f"{CLEAN}/phase2_summary.json",
    shell:
        "python scripts/processing/phase2_standardize.py"

# ----------------------------------------------------------------
# Phase 3 — Edge construction
# ----------------------------------------------------------------

rule phase3_supply_chain:
    input:
        FIRM_UNIVERSE,
        "data/raw/compustat/comp_segments_hist_daily_customer.csv",
    output:
        f"{EDGES}/supply_chain_edges.parquet",
    shell:
        "python scripts/processing/phase3_31_supply_chain.py"

rule phase3_creditor:
    input:
        FIRM_UNIVERSE,
        "data/raw/LSEG/LSEG_Dealscan.csv",
        "data/raw/LSEG/Dealscan-Compustat_Linking_Database012024.xlsx",
    output:
        f"{EDGES}/creditor_edges.parquet",
    shell:
        "python scripts/processing/phase3_32_creditor.py"

rule phase3_industry:
    input:
        FIRM_UNIVERSE,
    output:
        f"{EDGES}/industry_4digit_edges.parquet",
        f"{EDGES}/industry_3digit_edges.parquet",
    shell:
        "python scripts/processing/phase3_33_34_industry.py"

rule phase3_geographic:
    input:
        FIRM_UNIVERSE,
    output:
        f"{EDGES}/geographic_edges.parquet",
    shell:
        "python scripts/processing/phase3_35_geographic.py"

rule phase3_ownership:
    input:
        FIRM_UNIVERSE,
        "data/raw/moodys_orbis/bvd_orbis_large_subsidiaries.csv",
    output:
        f"{EDGES}/ownership_edges.parquet",
    shell:
        "python scripts/processing/phase3_36_ownership.py"

rule phase3_board:
    input:
        FIRM_UNIVERSE,
    output:
        f"{EDGES}/board_interlock_edges.parquet",
    shell:
        "python scripts/processing/phase3_37_board.py"

rule phase3_validate:
    input:
        f"{EDGES}/supply_chain_edges.parquet",
        f"{EDGES}/creditor_edges.parquet",
        f"{EDGES}/industry_4digit_edges.parquet",
        f"{EDGES}/industry_3digit_edges.parquet",
        f"{EDGES}/geographic_edges.parquet",
        f"{EDGES}/ownership_edges.parquet",
        f"{EDGES}/board_interlock_edges.parquet",
    output:
        f"{EDGES}/phase3_summary.json",
    shell:
        "python scripts/processing/phase3_38_validate.py"

rule phase3_done:
    input:
        f"{EDGES}/phase3_summary.json",

# ----------------------------------------------------------------
# Phase 4 — Default labels
# ----------------------------------------------------------------

rule phase4_default_labels:
    input:
        FIRM_UNIVERSE,
        FIRM_YEARS,
        # The LoPucki Cases Table is downloaded manually from
        # https://lopucki.law.ufl.edu/ and dropped under data/raw/lopucki/.
    output:
        f"{CLEAN}/default_events.parquet",
        f"{CLEAN}/default_labels_annual.parquet",
        f"{CLEAN}/default_labels_quarterly.parquet",
        f"{CLEAN}/lopucki_match_report.csv",
        f"{CLEAN}/lopucki_unmatched.csv",
        f"{CLEAN}/phase4_summary.json",
    shell:
        "python scripts/processing/phase4_default_labels.py"

rule phase4_lopucki_only:
    # Redistributable subset (no Compustat-sourced columns / dates).
    input:
        f"{CLEAN}/default_events.parquet",
    output:
        f"{PROCESSED}/default_events_lopucki_only.parquet",
    shell:
        "python scripts/processing/split_lopucki_only.py"

rule phase4_done:
    input:
        f"{CLEAN}/default_labels_quarterly.parquet",
        f"{PROCESSED}/default_events_lopucki_only.parquet",

# ----------------------------------------------------------------
# Phase 6 — Credit spreads (must run before Phase 5)
# ----------------------------------------------------------------

rule phase6_bond_map:
    input:
        FIRM_UNIVERSE,
        "data/raw/trace/trace_standard_master_file.csv",
        "data/raw/crsp/Bond_CRSP_link.csv",
    output:
        f"{CLEAN}/trace_bond_firm_map.parquet",
    shell:
        "python scripts/processing/phase6_bond_map.py"

rule phase6_stream_spreads:
    input:
        f"{CLEAN}/trace_bond_firm_map.parquet",
        "data/raw/trace/trace_standard_BTDS.csv",
        "data/raw/fred/treasury_curve/treasury_curve.csv",
    output:
        f"{CLEAN}/trace_trade_spreads.parquet",
    shell:
        "python scripts/processing/phase6_stream_spreads.py"

rule phase6_aggregate:
    input:
        f"{CLEAN}/trace_trade_spreads.parquet",
        f"{CLEAN}/node_features_quarterly_standardized.parquet",
        f"{CLEAN}/node_features_standardized.parquet",
    output:
        f"{CLEAN}/credit_spreads_quarterly.parquet",
        f"{CLEAN}/credit_spreads_annual.parquet",
        f"{CLEAN}/phase6_summary.json",
    shell:
        "python scripts/processing/phase6_aggregate.py"

# ----------------------------------------------------------------
# Phase 5 — Graph assembly (depends on all upstream phases)
# ----------------------------------------------------------------

rule phase5_graph_assembly:
    input:
        f"{CLEAN}/node_features_quarterly_standardized.parquet",
        f"{CLEAN}/default_labels_quarterly.parquet",
        f"{CLEAN}/credit_spreads_quarterly.parquet",
        f"{EDGES}/supply_chain_edges.parquet",
        f"{EDGES}/creditor_edges.parquet",
        f"{EDGES}/industry_4digit_edges.parquet",
        f"{EDGES}/industry_3digit_edges.parquet",
        f"{EDGES}/geographic_edges.parquet",
        f"{EDGES}/ownership_edges.parquet",
        f"{EDGES}/board_interlock_edges.parquet",
    output:
        f"{CLEAN}/graph_metadata.json",
        f"{CLEAN}/phase5_summary.json",
        f"{CLEAN}/split_assignments.parquet",
    shell:
        "python scripts/processing/phase5_graph_assembly.py"
