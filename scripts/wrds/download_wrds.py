"""
WRDS Data Download Pipeline

Reproduces every WRDS-sourced raw CSV under data/raw/ by querying
wrds-pgdata.wharton.upenn.edu. Config lives in wrds_config.yaml; each
dataset declares a library, table, column list, and chunking strategy.

Usage:
  python scripts/wrds/download_wrds.py --list           # show accessible libs
  python scripts/wrds/download_wrds.py --dry-run        # print SQL + counts
  python scripts/wrds/download_wrds.py --only NAME      # subset
  python scripts/wrds/download_wrds.py --force          # overwrite existing
  python scripts/wrds/download_wrds.py                  # full run
"""


import argparse
import hashlib
import json
import os
import sys
import time
import yaml
import pandas as pd
from datetime import datetime, date
from pathlib import Path

from wrds_client import WRDSClient



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "wrds_config.yaml"


# ---------- config / column loading ----------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_columns(columns_file: Path) -> list[str]:
    """Read one-column-per-line file, strip comments and blanks."""
    cols = []
    with open(columns_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols.append(line)
    return cols


def quote_identifier(col: str) -> str:
    """
    Quote a Postgres identifier when it needs it: starts with digit or
    underscore, contains uppercase letters, or contains non-alphanumeric
    characters. Handles table-qualified columns (alias.column),
    already-quoted columns, and column expressions with ' AS ' aliases
    (used for explicit renames in JOIN queries).
    """
    col = col.strip()
    if col == "*":
        return "*"
    # Pass through expressions with explicit aliasing — user knows best.
    if " AS " in col.upper():
        return col
    # Pre-quoted passes through.
    if col.startswith('"') and col.endswith('"'):
        return col
    # Qualified column: alias.name — quote each part separately.
    if "." in col:
        parts = col.split(".", 1)
        return f"{quote_identifier(parts[0])}.{quote_identifier(parts[1])}"

    needs_quote = (
        not col[0].isalpha()
        or any(c.isupper() for c in col)
        or any((not c.isalnum()) and c != "_" for c in col)
    )
    return f'"{col}"' if needs_quote else col


# ---------- SQL building ----------

def build_from_clause(dataset: dict) -> str:
    if dataset.get("join_sql"):
        return dataset["join_sql"].strip()
    return f"FROM {dataset['library']}.{dataset['table']}"


def build_sql(dataset: dict, columns: list[str], extra_where: str | None = None) -> str:
    """Build full SELECT statement. extra_where is ANDed with the config where."""
    from_clause = build_from_clause(dataset)

    if columns == ["*"]:
        select_list = "*"
    else:
        # When using a JOIN with table alias 'a', we don't alias columns —
        # Postgres will resolve plain column names against the unambiguous
        # table. If a collision happens, user must edit the column file
        # to include explicit a.foo / b.bar.
        select_list = ", ".join(quote_identifier(c) for c in columns)

    where_parts = []
    if dataset.get("where"):
        where_parts.append(f"({dataset['where']})")
    if extra_where:
        where_parts.append(f"({extra_where})")
    where_clause = ""
    if where_parts:
        where_clause = " WHERE " + " AND ".join(where_parts)

    order_clause = ""
    if dataset.get("order_by"):
        order_clause = f" ORDER BY {dataset['order_by']}"

    return f"SELECT {select_list} {from_clause}{where_clause}{order_clause}"


def query_hash(sql: str) -> str:
    return "sha256:" + hashlib.sha256(sql.encode()).hexdigest()[:16]


# ---------- date-end derivation ----------

def derive_date_end(dataset: dict, existing_path: Path) -> str | None:
    """
    Find max(date_column) in an existing CSV to cap the re-download date range.
    Returns ISO date string, or None if no existing file or no date_column.
    """
    date_col = dataset.get("date_column")
    if not date_col or not existing_path.exists():
        return None

    # Strip table alias like 'a.date' -> 'date' for CSV column lookup
    csv_col = date_col.split(".")[-1]

    try:
        max_date = None
        for chunk in pd.read_csv(
            existing_path, usecols=lambda c: c.lower() == csv_col.lower(),
            chunksize=500_000, dtype=str, low_memory=False,
        ):
            if chunk.empty:
                continue
            col_name = chunk.columns[0]
            series = pd.to_datetime(chunk[col_name], errors="coerce")
            m = series.max()
            if pd.notna(m) and (max_date is None or m > max_date):
                max_date = m
        if max_date is not None:
            return max_date.date().isoformat()
    except Exception as e:
        print(f"    WARN: could not read max date from {existing_path.name}: {e}")

    return None


def effective_date_range(dataset: dict, config: dict, output_path: Path) -> tuple[str, str]:
    start = dataset.get("date_start") or config["default_date_start"]
    end = dataset.get("date_end")
    if not end:
        end = derive_date_end(dataset, output_path) or config["default_date_end"]
    return start, end


# ---------- chunk iterators ----------

def year_ranges(start_iso: str, end_iso: str):
    start_y = int(start_iso[:4])
    end_y = int(end_iso[:4])
    for y in range(start_y, end_y + 1):
        left = f"{y}-01-01" if y > start_y else start_iso
        right = f"{y+1}-01-01" if y < end_y else _next_day(end_iso)
        yield (f"year={y}", left, right)


def year_month_ranges(start_iso: str, end_iso: str):
    y, m = int(start_iso[:4]), int(start_iso[5:7])
    end_y, end_m = int(end_iso[:4]), int(end_iso[5:7])
    while (y < end_y) or (y == end_y and m <= end_m):
        left = f"{y}-{m:02d}-01" if (y, m) != (int(start_iso[:4]), int(start_iso[5:7])) else start_iso
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        right = f"{ny}-{nm:02d}-01"
        if (y, m) == (end_y, end_m):
            right = _next_day(end_iso)
        yield (f"{y}-{m:02d}", left, right)
        y, m = ny, nm


def _next_day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return (d.fromordinal(d.toordinal() + 1)).isoformat()


# ---------- checkpoint ----------

def checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".chunks.json")


def load_checkpoint(output_path: Path) -> dict:
    cp = checkpoint_path(output_path)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {"completed_chunks": [], "started_at": datetime.now().isoformat()}


def save_checkpoint(output_path: Path, data: dict) -> None:
    cp = checkpoint_path(output_path)
    tmp = cp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, cp)


# ---------- downloaders (one per strategy) ----------

def _write_df(df: pd.DataFrame, path: Path, header: bool, column_rename: dict | None = None):
    if column_rename:
        df = df.rename(columns=column_rename)
    mode = "w" if header else "a"
    df.to_csv(path, index=False, mode=mode, header=header)


def run_single_query(client, dataset, sql, output_path, column_rename):
    df = client.raw_sql(sql)
    tmp = output_path.with_suffix(output_path.suffix + ".partial")
    _write_df(df, tmp, header=True, column_rename=column_rename)
    os.replace(tmp, output_path)
    return {"rows": len(df), "chunks": 1}


def run_by_year(client, dataset, columns, output_path, column_rename, date_col,
                start_iso, end_iso, checkpoint):
    tmp = output_path.with_suffix(output_path.suffix + ".partial")
    first_write = not tmp.exists()
    rows_total = 0
    chunks_done = 0

    completed = set(checkpoint["completed_chunks"])
    ranges = list(year_ranges(start_iso, end_iso))

    for chunk_key, left, right in ranges:
        if chunk_key in completed:
            continue
        extra = f"{date_col} >= '{left}' AND {date_col} < '{right}'"
        sql = build_sql(dataset, columns, extra_where=extra)
        print(f"    chunk {chunk_key}: {left} <= {date_col} < {right}")
        df = client.raw_sql(sql)
        _write_df(df, tmp, header=first_write, column_rename=column_rename)
        first_write = False
        rows_total += len(df)
        chunks_done += 1

        completed.add(chunk_key)
        checkpoint["completed_chunks"] = sorted(completed)
        save_checkpoint(output_path, checkpoint)
        print(f"      rows: {len(df):,}  cumulative: {rows_total:,}")

    os.replace(tmp, output_path)
    cp = checkpoint_path(output_path)
    if cp.exists():
        cp.unlink()
    return {"rows": rows_total, "chunks": chunks_done}


def run_by_year_month(client, dataset, columns, output_path, column_rename, date_col,
                      start_iso, end_iso, checkpoint):
    tmp = output_path.with_suffix(output_path.suffix + ".partial")
    first_write = not tmp.exists()
    rows_total = 0
    chunks_done = 0

    completed = set(checkpoint["completed_chunks"])
    ranges = list(year_month_ranges(start_iso, end_iso))
    print(f"    {len(ranges)} month-chunks ({start_iso} to {end_iso})")

    for chunk_key, left, right in ranges:
        if chunk_key in completed:
            continue
        extra = f"{date_col} >= '{left}' AND {date_col} < '{right}'"
        sql = build_sql(dataset, columns, extra_where=extra)
        t0 = time.time()
        df = client.raw_sql(sql)
        _write_df(df, tmp, header=first_write, column_rename=column_rename)
        first_write = False
        rows_total += len(df)
        chunks_done += 1
        elapsed = time.time() - t0

        completed.add(chunk_key)
        checkpoint["completed_chunks"] = sorted(completed)
        save_checkpoint(output_path, checkpoint)
        print(f"    chunk {chunk_key}: rows={len(df):>8,}  elapsed={elapsed:6.1f}s  cumulative={rows_total:,}")

    os.replace(tmp, output_path)
    cp = checkpoint_path(output_path)
    if cp.exists():
        cp.unlink()
    return {"rows": rows_total, "chunks": chunks_done}


def run_pandas_chunksize(client, dataset, sql, output_path, column_rename, chunksize):
    tmp = output_path.with_suffix(output_path.suffix + ".partial")
    first_write = True
    rows_total = 0
    chunks_done = 0
    for df in client.stream_sql(sql, chunksize=chunksize):
        _write_df(df, tmp, header=first_write, column_rename=column_rename)
        first_write = False
        rows_total += len(df)
        chunks_done += 1
        print(f"    chunk {chunks_done}: rows={len(df):,}  cumulative={rows_total:,}")
    os.replace(tmp, output_path)
    return {"rows": rows_total, "chunks": chunks_done}


# ---------- per-dataset orchestrator ----------

def process_dataset(client, dataset, config, accessible_libs, args) -> dict:
    name = dataset["name"]
    library = dataset["library"]
    table = dataset["table"]
    output_path = PROJECT_ROOT / config["output_root"] / dataset["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "name": name,
        "library": library,
        "table": table,
        "chunk_strategy": dataset["chunk_strategy"],
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "start_time": datetime.now().isoformat(),
    }

    print()
    print(f"  [{name}] {library}.{table}")

    # license check
    if not check_access(library, table, accessible_libs, client):
        print(f"    SKIPPED: {library}.{table} not accessible on this subscription")
        record.update({
            "status": "skipped_no_license",
            "reason": f"library '{library}' not accessible",
            "end_time": datetime.now().isoformat(),
        })
        return record

    # idempotency: skip if exists (bypassed by --dry-run, which always previews)
    if (output_path.exists() and output_path.stat().st_size > 0
            and not args.force and not args.dry_run):
        cp_file = checkpoint_path(output_path)
        if cp_file.exists() and not args.force_missing_chunks:
            print(f"    EXISTS (with partial checkpoint) — use --force-missing-chunks to resume or --force to redo")
        else:
            print(f"    EXISTS ({output_path.stat().st_size:,} bytes) — skipping (use --force to overwrite)")
        record.update({
            "status": "skipped_existing",
            "bytes_existing": output_path.stat().st_size,
            "end_time": datetime.now().isoformat(),
        })
        return record

    # load columns and build SQL
    columns_file = SCRIPT_DIR / dataset["columns_file"]
    columns = load_columns(columns_file)

    # compute effective date range
    start_iso, end_iso = effective_date_range(dataset, config, output_path)
    if dataset.get("date_column"):
        print(f"    date range: {start_iso} to {end_iso}")

    column_rename = dataset.get("column_rename")

    # dry run
    if args.dry_run:
        sql = build_sql(dataset, columns)
        print(f"    SQL preview (first 400 chars):")
        print(f"      {sql[:400]}")
        try:
            count = client.count_rows(build_from_clause(dataset) + (f" WHERE {dataset['where']}" if dataset.get("where") else ""))
            print(f"    estimated rows: {count:,}")
            record["estimated_rows"] = count
        except Exception as e:
            print(f"    count error: {e}")
        record.update({"status": "dry_run", "end_time": datetime.now().isoformat()})
        return record

    # execute
    checkpoint = load_checkpoint(output_path) if args.force_missing_chunks else \
                 {"completed_chunks": [], "started_at": datetime.now().isoformat()}

    strategy = dataset["chunk_strategy"]
    try:
        if strategy == "single_query":
            sql = build_sql(dataset, columns)
            record["query_hash"] = query_hash(sql)
            result = run_single_query(client, dataset, sql, output_path, column_rename)
        elif strategy == "by_year":
            record["query_hash"] = query_hash(build_sql(dataset, columns))
            result = run_by_year(
                client, dataset, columns, output_path, column_rename,
                dataset["date_column"], start_iso, end_iso, checkpoint,
            )
        elif strategy == "by_year_month":
            record["query_hash"] = query_hash(build_sql(dataset, columns))
            result = run_by_year_month(
                client, dataset, columns, output_path, column_rename,
                dataset["date_column"], start_iso, end_iso, checkpoint,
            )
        elif strategy == "pandas_chunksize":
            sql = build_sql(dataset, columns)
            record["query_hash"] = query_hash(sql)
            result = run_pandas_chunksize(
                client, dataset, sql, output_path, column_rename,
                config["defaults"]["pandas_chunksize"],
            )
        else:
            raise ValueError(f"unknown chunk_strategy '{strategy}'")

        record.update({
            "status": "success",
            "rows_downloaded": result["rows"],
            "chunks_completed": result["chunks"],
            "bytes_written": output_path.stat().st_size if output_path.exists() else 0,
            "date_range": {"start": start_iso, "end": end_iso} if dataset.get("date_column") else None,
            "columns_count": len(columns),
            "end_time": datetime.now().isoformat(),
        })
        print(f"    SUCCESS: {result['rows']:,} rows, {record['bytes_written']:,} bytes")

    except Exception as e:
        record.update({
            "status": "error",
            "error": str(e).splitlines()[0][:500],
            "end_time": datetime.now().isoformat(),
        })
        print(f"    ERROR: {e}")

    return record


# ---------- access detection ----------

def check_access(library: str, table: str, accessible_libs: set[str], client: WRDSClient) -> bool:
    if library in accessible_libs:
        return True
    probe = client.probe_table(library, table)
    if probe["ok"]:
        accessible_libs.add(library)
        return True
    return False


def print_access_table(client: WRDSClient, datasets: list[dict], primary_libs: set[str]):
    print("=" * 68)
    print("ACCESSIBLE LIBRARIES")
    print("=" * 68)
    print(f"  {'Library':<24} {'Primary':<8} {'Probe':<8} Status")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*16}")

    seen = {}
    for d in datasets:
        lib = d["library"]
        if lib in seen:
            continue
        in_primary = lib in primary_libs
        probe = client.probe_table(lib, d["table"])
        seen[lib] = (in_primary, probe["ok"])

    for lib, (in_primary, probe_ok) in sorted(seen.items()):
        primary_str = "yes" if in_primary else "no"
        probe_str = "ok" if probe_ok else "denied"
        if in_primary and probe_ok:
            status = "accessible"
        elif probe_ok:
            status = "accessible (trial)"
        elif in_primary:
            status = "listed but denied"
        else:
            status = "no access"
        print(f"  {lib:<24} {primary_str:<8} {probe_str:<8} {status}")
    print()


# ---------- manual artifact check ----------

def check_manual_artifacts(config: dict) -> list[dict]:
    records = []
    for art in config.get("manual_artifacts") or []:
        expected = PROJECT_ROOT / config["output_root"] / art["expected_path"]
        rec = {
            "name": art["name"],
            "expected_path": str(expected.relative_to(PROJECT_ROOT)),
            "note": art.get("note", ""),
        }
        if expected.exists():
            rec.update({
                "status": "manual_present",
                "bytes": expected.stat().st_size,
                "mtime": datetime.fromtimestamp(expected.stat().st_mtime).isoformat(),
            })
        else:
            rec["status"] = "manual_missing"
        records.append(rec)
    return records


# ---------- log writer ----------

def write_log(config: dict, records: list[dict], manual: list[dict], primary_libs: set[str]):
    log_path = PROJECT_ROOT / config["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    statuses = [r["status"] for r in records]
    log = {
        "download_timestamp": datetime.now().isoformat(),
        "config_file": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "subscribed_libraries": sorted(primary_libs),
        "totals": {
            "datasets_requested": len(records),
            "success": statuses.count("success"),
            "skipped_existing": statuses.count("skipped_existing"),
            "skipped_no_license": statuses.count("skipped_no_license"),
            "error": statuses.count("error"),
            "dry_run": statuses.count("dry_run"),
        },
        "datasets": records,
        "manual_artifacts": manual,
    }
    tmp = log_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, log_path)
    print(f"  Log saved: {log_path}")


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="WRDS data download pipeline")
    p.add_argument("--only", nargs="+", default=None, help="dataset name(s) to include")
    p.add_argument("--skip", nargs="+", default=None, help="dataset name(s) to exclude")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.add_argument("--force-missing-chunks", action="store_true",
                   help="resume partial chunked downloads from checkpoint")
    p.add_argument("--dry-run", action="store_true", help="print SQL and count estimate, no download")
    p.add_argument("--list", action="store_true", help="print accessible-libraries diagnostic and exit")
    p.add_argument("--config", default=str(CONFIG_PATH), help="path to YAML config")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print("STEP 1: Loading configuration")
    print("=" * 68)
    print(f"  Config: {args.config}")
    config = load_config(Path(args.config))
    print(f"  Datasets defined: {len(config['datasets'])}")
    print(f"  Manual artifacts: {len(config.get('manual_artifacts') or [])}")
    print()

    print("=" * 68)
    print("STEP 2: Connecting to WRDS")
    print("=" * 68)
    client = WRDSClient(verbose=True)
    client.connect()
    print()

    # primary libs + probe-based diagnostic
    primary_libs = client.list_libraries_primary()
    print_access_table(client, config["datasets"], primary_libs)

    if args.list:
        client.close()
        return 0

    # filter datasets
    ds_all = config["datasets"]
    if args.only:
        ds = [d for d in ds_all if d["name"] in args.only]
        missing = set(args.only) - {d["name"] for d in ds}
        if missing:
            print(f"WARN: --only names not in config: {missing}")
    else:
        ds = ds_all
    if args.skip:
        ds = [d for d in ds if d["name"] not in args.skip]

    print("=" * 68)
    mode = "DRY RUN" if args.dry_run else "DOWNLOADING"
    print(f"STEP 3: {mode} ({len(ds)} datasets)")
    print("=" * 68)

    accessible = set(primary_libs)  # may grow via probes
    records = []
    for i, dataset in enumerate(ds, 1):
        print(f"\n[{i}/{len(ds)}]", end="")
        rec = process_dataset(client, dataset, config, accessible, args)
        records.append(rec)

    print()
    print("=" * 68)
    print("STEP 4: Manual artifact check")
    print("=" * 68)
    manual = check_manual_artifacts(config)
    for m in manual:
        status = m["status"]
        print(f"  [{status}] {m['name']}: {m['expected_path']}")
        if m.get("note"):
            print(f"    note: {m['note']}")

    print()
    print("=" * 68)
    print("STEP 5: Writing download log")
    print("=" * 68)
    write_log(config, records, manual, primary_libs)

    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  {'Dataset':<32} {'Status':<22} {'Rows':>12}")
    print(f"  {'-'*32} {'-'*22} {'-'*12}")
    for r in records:
        n = r["name"][:32]
        s = r["status"]
        rows = r.get("rows_downloaded") or r.get("estimated_rows") or ""
        rows_str = f"{rows:,}" if isinstance(rows, int) else str(rows)
        print(f"  {n:<32} {s:<22} {rows_str:>12}")
    print()

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
