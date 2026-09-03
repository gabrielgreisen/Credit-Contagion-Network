# Findings figures

Each figure is rendered at 150 dpi by [`scripts/analysis/figures.py`](../../scripts/analysis/figures.py). All inputs come from `data/clean/` and `data/processed/`.

- `firm_year_coverage.png` — Firm-year observations per fiscal year, 1950–2025. Total 441,934 firm-years across the panel.
- `default_rate_over_time.png` — Annual default rate, computed as firms experiencing a default event in fiscal year ÷ active firms in fiscal year, 1970–2024. Crisis years (2001, 2008, 2020) annotated with dashed vertical lines.
- `node_class_split.png` — Composition of the 32,135-firm universe: 26,656 nonfinancial vs. 5,479 financial (split by SIC 6000–6799).
- `top_industries.png` — Top 15 SIC codes by firm count in the firm universe.
- `edge_layer_composition.png` — Edge counts by relationship layer (log scale). Total: 486,134,670 edges across 7 layers.
- `lehman_spread_trajectory.png` — Lehman Brothers Holdings (gvkey 6669) median quarterly credit spread from TRACE, 2006Q1 through 2008Q3 (Chapter 11 filing 2008-09-15).
