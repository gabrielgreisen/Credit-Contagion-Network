"""Construct the Campbell-Hilscher-Szilagyi (2008) predictor variables.

The eight CHS variables, built from raw Compustat quarterly and CRSP monthly
files at the (gvkey, datadate) grain of the quarterly panel:

  NIMTA   net income / (market equity + total liabilities)
  TLMTA   total liabilities / (market equity + total liabilities)
  EXRET   quarterly log return minus log S&P 500 return
  SIGMA   annualized volatility of monthly returns, trailing 12 months
  RSIZE   log(firm market equity / total CRSP market equity)
  CASHMTA cash & equivalents / (market equity + total liabilities)
  MB      market-to-book, with the CHS 10% book-equity adjustment
  PRICE   log(share price, capped at $15)

Deviations from the original paper (monthly panel, daily data):
  * EXRET is quarterly (sum of monthly log excess returns) instead of monthly.
  * SIGMA uses 12 monthly returns (annualized) instead of 3 months of daily
    returns — the raw CRSP file here is monthly.
  * RSIZE benchmarks against total CRSP market equity rather than the total
    market value of S&P 500 firms (no constituent list in the raw data);
    the two differ by a slowly-moving scalar.

Results are cached to data/clean/chs_variables_quarterly.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
CLEAN = PROJECT_ROOT / "data" / "clean"
CACHE = CLEAN / "chs_variables_quarterly.parquet"

CHS_VARS = ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "mb", "price"]


def _load_compustat_fundamentals() -> pd.DataFrame:
    """niq, ltq, cheq, ceqq per (gvkey, datadate) from the raw quarterly file."""
    usecols = ["gvkey", "datadate", "indfmt", "datafmt", "consol", "curcdq",
               "niq", "ltq", "cheq", "ceqq"]
    df = pd.read_csv(
        RAW / "compustat" / "compustat_CIQ_quarterly.csv",
        usecols=usecols,
        dtype={"gvkey": "int32", "indfmt": "category", "datafmt": "category",
               "consol": "category", "curcdq": "category"},
        parse_dates=["datadate"],
    )
    df = df[(df["datafmt"] == "STD") & (df["consol"] == "C") & (df["curcdq"] == "USD")]
    # prefer the industrial format where a firm reports both INDL and FS
    df["_fmt_rank"] = (df["indfmt"] != "INDL").astype(int)
    df = (
        df.sort_values(["gvkey", "datadate", "_fmt_rank"])
        .drop_duplicates(["gvkey", "datadate"], keep="first")
    )
    return df[["gvkey", "datadate", "niq", "ltq", "cheq", "ceqq"]]


def _load_crsp_quarterly() -> pd.DataFrame:
    """Quarter-level market variables per (permno, year, quarter) from CRSP monthly."""
    df = pd.read_csv(
        RAW / "crsp" / "CRSP.csv",
        usecols=["PERMNO", "date", "PRC", "RET", "SHROUT", "sprtrn"],
        dtype={"PERMNO": "int32"},
        parse_dates=["date"],
    )
    df = df.rename(columns={"PERMNO": "permno"})
    df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
    df["PRC"] = pd.to_numeric(df["PRC"], errors="coerce").abs()  # negatives are bid/ask midpoints
    df["SHROUT"] = pd.to_numeric(df["SHROUT"], errors="coerce")
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    df["me"] = df["PRC"] * df["SHROUT"] / 1000.0  # $M, matching Compustat units
    df["log_exret_m"] = np.log1p(df["RET"]) - np.log1p(df["sprtrn"])

    # trailing 12-month volatility of monthly returns, annualized (min 6 obs)
    df["sigma"] = (
        df.groupby("permno")["RET"]
        .transform(lambda s: s.rolling(12, min_periods=6).std())
        * np.sqrt(12.0)
    )

    # relative size vs total CRSP market equity that month
    total_me = df.groupby("date")["me"].transform("sum")
    df["rsize"] = np.log(df["me"] / total_me)

    df["price"] = np.log(df["PRC"].clip(upper=15.0))

    df["year"] = df["date"].dt.year.astype("int16")
    df["quarter"] = df["date"].dt.quarter.astype("int8")

    q = (
        df.groupby(["permno", "year", "quarter"])
        .agg(
            me=("me", "last"),
            sigma=("sigma", "last"),
            rsize=("rsize", "last"),
            price=("price", "last"),
            exret=("log_exret_m", "sum"),
            n_months=("log_exret_m", "count"),
        )
        .reset_index()
    )
    q.loc[q["n_months"] < 2, "exret"] = np.nan
    return q.drop(columns="n_months")


def build_chs_variables(panel_ids: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """CHS variables for each row of `panel_ids` (gvkey, datadate, permno).

    Uses the cache when present; pass force=True to rebuild from raw files.
    """
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    ids = panel_ids[["gvkey", "datadate", "permno"]].copy()
    ids["datadate"] = pd.to_datetime(ids["datadate"])
    ids["year"] = ids["datadate"].dt.year.astype("int16")
    ids["quarter"] = ids["datadate"].dt.quarter.astype("int8")

    print("reading Compustat fundamentals (large file, ~minutes)...")
    fund = _load_compustat_fundamentals()
    print(f"  {len(fund):,} firm-quarters")

    print("reading CRSP monthly and building market variables...")
    mkt = _load_crsp_quarterly()
    print(f"  {len(mkt):,} permno-quarters")

    df = ids.merge(fund, on=["gvkey", "datadate"], how="left")
    df = df.merge(mkt, on=["permno", "year", "quarter"], how="left")

    mta = df["me"] + df["ltq"]  # market value of total assets
    mta = mta.where(mta > 0)
    df["nimta"] = df["niq"] / mta
    df["tlmta"] = df["ltq"] / mta
    df["cashmta"] = df["cheq"] / mta

    # CHS adjusted book equity: BE + 0.1 * (ME - BE), floored at a small positive
    be_adj = (df["ceqq"] + 0.1 * (df["me"] - df["ceqq"])).where(lambda s: s > 0)
    df["mb"] = df["me"] / be_adj

    out = df[["gvkey", "datadate", *CHS_VARS]]
    out.to_parquet(CACHE, index=False)
    print(f"cached -> {CACHE}")
    return out
