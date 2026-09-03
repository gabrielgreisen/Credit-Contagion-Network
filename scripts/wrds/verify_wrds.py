"""
WRDS Download Verification

Compares re-downloaded CSVs against the original versions that already
exist under data/raw/. Writes data/raw/wrds_verify_report.json with
per-dataset column, row count, and date-range diagnostics.

This script expects the pipeline to have written new files to paths
suffixed with '.new' (i.e., run download_wrds.py with --force into a
temporary location via manual copy). If only the current files exist,
verification reports "no comparison available" and exits cleanly.

Usage:
  python scripts/wrds/verify_wrds.py                 # compare *.csv vs *.csv.new
  python scripts/wrds/verify_wrds.py --live          # run against live files only
"""


import argparse
import json
import sys
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "wrds_config.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data/raw/wrds_verify_report.json"



def count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def read_header(path: Path) -> list[str]:
    with open(path) as f:
        return f.readline().rstrip("\n").split(",")


def max_date_in_csv(path: Path, col: str) -> str | None:
    max_d = None
    try:
        for chunk in pd.read_csv(
            path, usecols=lambda c: c.lower() == col.lower(),
            chunksize=500_000, dtype=str, low_memory=False,
        ):
            if chunk.empty:
                continue
            name = chunk.columns[0]
            s = pd.to_datetime(chunk[name], errors="coerce")
            m = s.max()
            if pd.notna(m) and (max_d is None or m > max_d):
                max_d = m
    except Exception as e:
        return f"ERROR: {e}"
    return max_d.date().isoformat() if max_d is not None else None


def verify_dataset(dataset: dict, output_root: Path, compare_new: bool) -> dict:
    name = dataset["name"]
    path = output_root / dataset["output_path"]
    new_path = path.with_suffix(path.suffix + ".new")

    rec = {"name": name, "path": str(path.relative_to(PROJECT_ROOT))}

    if not path.exists():
        rec["status"] = "missing"
        return rec

    rec["columns_existing"] = read_header(path)
    rec["row_count_existing"] = count_rows(path)

    if dataset.get("date_column"):
        csv_col = dataset["date_column"].split(".")[-1]
        rec["max_date_existing"] = max_date_in_csv(path, csv_col)

    if compare_new and new_path.exists():
        rec["columns_new"] = read_header(new_path)
        rec["row_count_new"] = count_rows(new_path)
        rec["columns_only_in_existing"] = sorted(set(rec["columns_existing"]) - set(rec["columns_new"]))
        rec["columns_only_in_new"] = sorted(set(rec["columns_new"]) - set(rec["columns_existing"]))
        delta = rec["row_count_new"] - rec["row_count_existing"]
        rec["row_count_delta"] = delta
        base = rec["row_count_existing"] or 1
        rec["row_count_delta_pct"] = round(100 * delta / base, 3)
        if dataset.get("date_column"):
            csv_col = dataset["date_column"].split(".")[-1]
            rec["max_date_new"] = max_date_in_csv(new_path, csv_col)
        rec["status"] = "compared"
    else:
        rec["status"] = "existing_only"

    return rec



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-compare", action="store_true",
                   help="skip looking for .new files; report on existing only")
    return p.parse_args()


def main():
    args = parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    output_root = PROJECT_ROOT / config["output_root"]

    print("=" * 68)
    print("WRDS VERIFICATION")
    print("=" * 68)
    print(f"  Output root: {output_root}")
    print(f"  Datasets:    {len(config['datasets'])}")
    print()

    records = []
    for d in config["datasets"]:
        print(f"  [{d['name']}] ...")
        rec = verify_dataset(d, output_root, compare_new=not args.no_compare)
        records.append(rec)

    report = {
        "timestamp": datetime.now().isoformat(),
        "datasets": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  {'Dataset':<32} {'Status':<16} {'Rows exist':>12} {'Rows new':>12} {'Δ%':>8}")
    print(f"  {'-'*32} {'-'*16} {'-'*12} {'-'*12} {'-'*8}")
    for r in records:
        n = r["name"][:32]
        s = r["status"]
        e = r.get("row_count_existing", "")
        nv = r.get("row_count_new", "")
        p = r.get("row_count_delta_pct", "")
        e_s = f"{e:,}" if isinstance(e, int) else str(e)
        n_s = f"{nv:,}" if isinstance(nv, int) else str(nv)
        p_s = f"{p:+.2f}" if isinstance(p, (int, float)) else ""
        print(f"  {n:<32} {s:<16} {e_s:>12} {n_s:>12} {p_s:>8}")
    print()
    print(f"  Report saved: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
