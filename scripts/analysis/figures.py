"""Generate the six findings figures reported in the README Findings section.

Inputs (read from data/clean/ and data/processed/):
  - firm_universe.parquet
  - firm_years.parquet
  - default_events.parquet
  - default_labels_annual.parquet
  - credit_spreads_quarterly.parquet
  - edges/*.parquet  (row counts via parquet metadata only — no full read)

Outputs (results/figures/, all 150 dpi PNG):
  - firm_year_coverage.png
  - default_rate_over_time.png
  - node_class_split.png
  - top_industries.png
  - edge_layer_composition.png
  - lehman_spread_trajectory.png
  - figure_index.md
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

SIC_LABELS = {
    1311: "Crude Petroleum & Natural Gas",
    2834: "Pharmaceutical Preparations",
    2836: "Biological Products",
    3674: "Semiconductors",
    3711: "Motor Vehicles & Passenger Car Bodies",
    4813: "Telephone Communications",
    4911: "Electric Services",
    5812: "Eating Places",
    6020: "State Commercial Banks",
    6021: "National Commercial Banks",
    6022: "State Commercial Banks",
    6199: "Finance Services",
    6311: "Life Insurance",
    6321: "Accident & Health Insurance",
    6331: "Fire, Marine & Casualty Insurance",
    6512: "Operators of Apartment Buildings",
    6770: "Blank Checks (Holding Companies)",
    6798: "Real Estate Investment Trusts",
    7370: "Computer Services",
    7372: "Prepackaged Software",
    7389: "Business Services",
    8731: "Commercial Physical Research",
}


def fig_firm_year_coverage(project_root: Path, out_dir: Path) -> str:
    fy = pd.read_parquet(project_root / "data/clean/firm_years.parquet",
                         columns=["fyear"])
    counts = fy["fyear"].dropna().astype(int).value_counts().sort_index()
    counts = counts.loc[(counts.index >= 1950) & (counts.index <= 2025)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts.values, width=0.85, color="#3b6cb6",
           edgecolor="none")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Firm-year observations")
    ax.set_title("Firm-year coverage by fiscal year (1950–2025)")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_xlim(1949, 2026)
    fig.tight_layout()
    out = out_dir / "firm_year_coverage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Firm-year observations per fiscal year, 1950–2025. " \
           f"Total {len(fy):,} firm-years across the panel."


def fig_default_rate_over_time(project_root: Path, out_dir: Path) -> str:
    events = pd.read_parquet(
        project_root / "data/clean/default_events.parquet",
        columns=["gvkey", "default_date"])
    events["year"] = events["default_date"].dt.year

    fy = pd.read_parquet(
        project_root / "data/clean/firm_years.parquet",
        columns=["gvkey", "fyear"])

    numerator = (events.dropna(subset=["year"])
                       .groupby("year")["gvkey"].nunique()
                       .rename("n_defaults"))
    denominator = (fy.dropna(subset=["fyear"])
                     .groupby("fyear")["gvkey"].nunique()
                     .rename("n_active"))

    annual = pd.concat([numerator, denominator], axis=1).reset_index()
    annual.columns = ["fyear", "n_defaults", "n_active"]
    annual["n_defaults"] = annual["n_defaults"].fillna(0).astype(int)
    annual = annual[annual["n_active"].notna()]
    annual["rate_pct"] = 100.0 * annual["n_defaults"] / annual["n_active"]
    annual = annual[(annual["fyear"] >= 1970) & (annual["fyear"] <= 2024)] \
                  .sort_values("fyear").reset_index(drop=True)

    crisis_years = [2001, 2008, 2020]
    print("default_rate_over_time sanity check (rate as %):")
    for y in crisis_years:
        row = annual.loc[annual["fyear"] == y]
        if len(row):
            print(f"  {y}: {float(row['rate_pct'].iloc[0]):.3f}%  "
                  f"({int(row['n_defaults'].iloc[0])} defaults / "
                  f"{int(row['n_active'].iloc[0])} active firms)")
    print(f"  long-run mean (1970-2024): {annual['rate_pct'].mean():.3f}%")
    print(f"  min / max: {annual['rate_pct'].min():.3f}% / "
          f"{annual['rate_pct'].max():.3f}%")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(annual["fyear"], annual["rate_pct"], color="#b62b2b",
            linewidth=1.6, marker="o", markersize=3)

    for year in crisis_years:
        if year in annual["fyear"].values:
            ax.axvline(year, color="grey", linestyle="--",
                       linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("1-year default rate (%)")
    ax.set_title("Annual 1-year default rate, 1970–2024")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = out_dir / "default_rate_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Annual default rate, computed as firms experiencing " \
           f"a default event in fiscal year ÷ active firms in fiscal year, " \
           f"1970–2024. Crisis years (2001, 2008, 2020) annotated with dashed " \
           f"vertical lines."


def fig_node_class_split(project_root: Path, out_dir: Path) -> str:
    fu = pd.read_parquet(project_root / "data/clean/firm_universe.parquet",
                         columns=["node_class"])
    counts = fu["node_class"].value_counts()
    nonfin = int(counts.get("nonfinancial", 0))
    fin = int(counts.get("financial", 0))

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(
        [nonfin, fin],
        labels=[f"Nonfinancial\n{nonfin:,}", f"Financial\n{fin:,}"],
        colors=["#3b6cb6", "#e08a3c"],
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    total = nonfin + fin
    ax.text(0, 0, f"{total:,}\nfirms", ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.set_title("Firm universe: nonfinancial vs. financial")
    fig.tight_layout()
    out = out_dir / "node_class_split.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Composition of the {total:,}-firm universe: " \
           f"{nonfin:,} nonfinancial vs. {fin:,} financial (split by SIC 6000–6799)."


def fig_top_industries(project_root: Path, out_dir: Path) -> str:
    fu = pd.read_parquet(project_root / "data/clean/firm_universe.parquet",
                         columns=["sic"])
    counts = fu["sic"].dropna().astype(int).value_counts().head(15)

    labels = [f"{sic}  {SIC_LABELS.get(sic, '—')}" for sic in counts.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(counts))
    ax.barh(y_pos, counts.values, color="#3b6cb6", edgecolor="none")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of firms")
    ax.set_title("Top 15 SIC industries by firm count")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts.values) * 0.005, i, f"{v:,}",
                va="center", fontsize=8)
    fig.tight_layout()
    out = out_dir / "top_industries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Top 15 SIC codes by firm count in the firm universe."


def fig_edge_layer_composition(project_root: Path, out_dir: Path) -> str:
    edges_dir = project_root / "data/clean/edges"
    layer_files = [
        ("Supply chain",        "supply_chain_edges.parquet"),
        ("Common creditor",     "creditor_edges.parquet"),
        ("Industry 4-digit",    "industry_4digit_edges.parquet"),
        ("Industry 3-digit",    "industry_3digit_edges.parquet"),
        ("Geographic",          "geographic_edges.parquet"),
        ("Ownership",           "ownership_edges.parquet"),
        ("Board interlock",     "board_interlock_edges.parquet"),
    ]
    counts = []
    for name, fname in layer_files:
        n = pq.ParquetFile(edges_dir / fname).metadata.num_rows
        counts.append((name, n))

    counts.sort(key=lambda x: x[1], reverse=True)
    names = [c[0] for c in counts]
    values = [c[1] for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color="#3b6cb6", edgecolor="none")
    ax.set_ylabel("Edge count")
    ax.set_title("Edge counts by layer (across all 261 quarterly snapshots)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle=":", which="both")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                v * 1.1 if v > 0 else 1,
                f"{v:,}", ha="center", fontsize=8)
    fig.tight_layout()
    out = out_dir / "edge_layer_composition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Edge counts by relationship layer (log scale). " \
           f"Total: {sum(values):,} edges across 7 layers."


def fig_lehman_spread_trajectory(project_root: Path, out_dir: Path) -> str:
    sp = pd.read_parquet(
        project_root / "data/clean/credit_spreads_quarterly.parquet",
        columns=["gvkey", "year", "quarter", "median_spread_bps"])
    leh = sp[sp["gvkey"] == 6669].copy()
    leh = leh[((leh["year"] == 2006)) |
              ((leh["year"] == 2007)) |
              ((leh["year"] == 2008) & (leh["quarter"] <= 3))]
    leh = leh.sort_values(["year", "quarter"]).reset_index(drop=True)
    leh["label"] = leh["year"].astype(str) + "Q" + leh["quarter"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(leh["label"], leh["median_spread_bps"],
            color="#b62b2b", marker="o", markersize=6, linewidth=1.8)
    for x, y in zip(leh["label"], leh["median_spread_bps"]):
        ax.text(x, y + max(leh["median_spread_bps"]) * 0.03,
                f"{y:.0f}", ha="center", fontsize=8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Median bond OAS (basis points)")
    ax.set_title("Lehman Brothers Holdings (gvkey=6669) — quarterly credit spread, 2006Q1–2008Q3")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = out_dir / "lehman_spread_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return f"`{out.name}` — Lehman Brothers Holdings (gvkey 6669) median quarterly " \
           f"credit spread from TRACE, 2006Q1 through 2008Q3 (Chapter 11 filing 2008-09-15)."


def main(project_root: Path) -> None:
    out_dir = project_root / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    captions = []
    captions.append(fig_firm_year_coverage(project_root, out_dir))
    captions.append(fig_default_rate_over_time(project_root, out_dir))
    captions.append(fig_node_class_split(project_root, out_dir))
    captions.append(fig_top_industries(project_root, out_dir))
    captions.append(fig_edge_layer_composition(project_root, out_dir))
    captions.append(fig_lehman_spread_trajectory(project_root, out_dir))

    index_path = out_dir / "figure_index.md"
    with index_path.open("w") as f:
        f.write("# Findings figures\n\n")
        f.write("Each figure is rendered at 150 dpi by ")
        f.write("[`scripts/analysis/figures.py`](../../scripts/analysis/figures.py). ")
        f.write("All inputs come from `data/clean/` and `data/processed/`.\n\n")
        for cap in captions:
            f.write(f"- {cap}\n")
    print(f"\nwrote {index_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    main(args.project_root.resolve())
