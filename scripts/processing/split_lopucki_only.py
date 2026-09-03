"""Split out the LoPucki-derived subset of default_events for redistribution.

The combined data/clean/default_events.parquet mixes LoPucki bankruptcy filings
with Compustat dlrsn delisting supplements. Compustat-derived rows and fields
cannot be redistributed under the WRDS academic license. This script writes a
LoPucki-only subset stripped of Compustat-sourced columns.

Filter:
  source IN ('lopucki', 'both')

Column treatment:
  - drop  compustat_dldte (Compustat-sourced)
  - reset default_date := lopucki_filing_date (the original is min(lopucki, compustat))

Output:
  data/processed/default_events_lopucki_only.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def main(project_root: Path) -> None:
    src = project_root / "data/clean/default_events.parquet"
    out_dir = project_root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "default_events_lopucki_only.parquet"

    df = pd.read_parquet(src)
    n_in = len(df)

    keep = df["source"].isin(["lopucki", "both"])
    sub = df.loc[keep].copy()

    # default_date in the combined file is min(lopucki_filing_date, compustat_dldte).
    # For redistribution we use the LoPucki filing date directly so no Compustat
    # information leaks into the date column.
    sub["default_date"] = sub["lopucki_filing_date"]

    # Drop the Compustat-sourced delisting date column entirely.
    sub = sub.drop(columns=["compustat_dldte"])

    cols = ["gvkey", "default_date", "default_type", "chapter", "outcome",
            "source", "lopucki_filing_date", "assets_at_filing", "NameCorp"]
    sub = sub[cols].sort_values(["default_date", "gvkey"]).reset_index(drop=True)

    sub.to_parquet(out, index=False)

    print(f"input rows : {n_in:,}")
    print(f"kept rows  : {len(sub):,} (source IN lopucki/both)")
    print(f"by source  : {sub['source'].value_counts().to_dict()}")
    print(f"date range : {sub['default_date'].min()} -> {sub['default_date'].max()}")
    print(f"unique gvk : {sub['gvkey'].nunique():,}")
    print(f"wrote      : {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    main(args.project_root.resolve())
