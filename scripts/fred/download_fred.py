"""
FRED Data Download Script
"""



from fredapi import Fred
import yaml
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime



SCRIPT_DIR = Path(__file__).resolve().parent        
PROJECT_ROOT = SCRIPT_DIR.parent.parent             
CONFIG_PATH = SCRIPT_DIR / "fred_config.yaml"       # scripts/fred/fred_config.yaml



def load_config():
    """Read the YAML config and return it as a Python dictionary."""
    print("=" * 60)
    print("STEP 1: Loading configuration")
    print("=" * 60)
    print(f"  Config file: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Count how many series we have across all categories
    total_series = 0
    for cat_name, cat_data in config["categories"].items():
        total_series += len(cat_data["series"])

    print(f"  Date range:  {config['start_date']} to {config['end_date']}")
    print(f"  Categories:  {len(config['categories'])}")
    print(f"  Total series: {total_series}")
    print()

    return config




def load_api_key():
    """Read the FRED API key from the project's key file."""
    print("=" * 60)
    print("STEP 2: Loading FRED API key")
    print("=" * 60)

    key_path = PROJECT_ROOT / "keys/fed_cc_key.txt"
    print(f"  Key file: {key_path}")

    with open(key_path, "r") as f:
        api_key = f.read().strip()

    # Show just the first 4 chars so you can verify it's the right key
    # without exposing the full key in logs
    print(f"  Key loaded: {api_key[:4]}...{'*' * 28}")
    print()

    return api_key



def flatten_series(config):
    """Extract all series from all categories into a flat list."""
    all_series = []
    for cat_name, cat_data in config["categories"].items():
        for s in cat_data["series"]:
            # Add the category name to each series dict for logging
            s["category"] = cat_name
            all_series.append(s)
    return all_series



def download_one_series(fred, series_id, start_date, end_date):
    """
    Download a single FRED series.

    Returns:
        pandas.Series on success, None on failure
    """
    try:
        data = fred.get_series(series_id, start_date, end_date)
        return data
    except Exception as e:
        print(f"    ERROR: {e}")
        return None




def save_individual_csv(data, series_id, output_dir):
    """Save a single series as {series_id}.csv with columns: date, value."""
    
    df = data.dropna().reset_index()
    df.columns = ["date", "value"]

    filepath = output_dir / f"{series_id}.csv"
    df.to_csv(filepath, index=False)

    return len(df)



def resample_to_monthly(data, frequency):
    """Resample a series to monthly frequency based on its native frequency."""
    if frequency == "daily":
        # Average daily values within each month
        return data.resample("MS").mean()
    elif frequency == "quarterly":
        # Forward-fill quarterly values into each month of the quarter
        return data.resample("MS").ffill()
    else:
        # Monthly — just align to month-start dates for consistency
        return data.resample("MS").last()



def main():
    # --- Load config and API key ---
    config = load_config()
    api_key = load_api_key()

    # --- Create the FRED API connection ---
    # Fred() creates a reusable connection object. All subsequent
    # get_series() calls go through this object.
    fred = Fred(api_key=api_key)

    # --- Set up output directory ---
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # parents=True  → creates parent directories if they don't exist
    # exist_ok=True → doesn't error if the directory already exists

    # --- Flatten all series into a simple list ---
    all_series = flatten_series(config)

    # --- Prepare tracking variables ---
    # These accumulate results as we download each series.
    downloaded_data = {}    # series_id → raw pandas Series (for master file)
    frequency_map = {}      # series_id → frequency string (for resampling)
    log_entries = []        # metadata about each download (for the log file)
    errors = []             # any failures

    

    print("=" * 60)
    print("STEP 3: Downloading series from FRED")
    print("=" * 60)

    for i, s in enumerate(all_series, 1):
        sid = s["series_id"]
        desc = s["description"]
        freq = s["frequency"]

        print(f"\n  [{i}/{len(all_series)}] {sid}")
        print(f"    {desc}")
        print(f"    Frequency: {freq}")

        # Download from FRED
        data = download_one_series(
            fred, sid, config["start_date"], config["end_date"]
        )

        if data is not None and len(data) > 0:
            # Save individual CSV
            row_count = save_individual_csv(data, sid, output_dir)

            # Store for master file construction
            downloaded_data[sid] = data
            frequency_map[sid] = freq

            # Record metadata for the log
            clean = data.dropna()
            log_entries.append({
                "series_id": sid,
                "description": desc,
                "category": s["category"],
                "frequency": freq,
                "rows_downloaded": row_count,
                "date_range": {
                    "start": str(clean.index.min().date()),
                    "end": str(clean.index.max().date()),
                },
                "status": "success",
            })

            print(f"    Saved: {sid}.csv ({row_count} rows)")
            print(f"    Range: {clean.index.min().date()} to {clean.index.max().date()}")
        else:
            # Download failed — log the error but keep going
            error_msg = f"Failed to download {sid} or series is empty"
            errors.append(error_msg)
            log_entries.append({
                "series_id": sid,
                "description": desc,
                "category": s["category"],
                "frequency": freq,
                "rows_downloaded": 0,
                "status": "error",
                "error": error_msg,
            })
            print(f"    FAILED — skipping")

        # Rate limiting: pause between requests
        # Skip the delay after the last series (no need to wait)
        if i < len(all_series):
            time.sleep(0.5)


    print()
    print("=" * 60)
    print("STEP 4: Building master file (monthly frequency)")
    print("=" * 60)

    if downloaded_data:
        monthly_dict = {}
        for sid, data in downloaded_data.items():
            freq = frequency_map[sid]
            resampled = resample_to_monthly(data, freq)
            monthly_dict[sid] = resampled
            print(f"  Resampled {sid} ({freq} → monthly): {len(resampled)} months")

        # pd.DataFrame combines all series, aligning on the date index
        master = pd.DataFrame(monthly_dict)
        master.index.name = "date"

        # Sort columns alphabetically for consistent ordering
        master = master.sort_index(axis=1)

        master_path = output_dir / config["master_file"]
        master.to_csv(master_path)
        print(f"\n  Master file saved: {master_path}")
        print(f"  Shape: {master.shape[0]} months × {master.shape[1]} series")
        print(f"  Date range: {master.index.min().date()} to {master.index.max().date()}")
    else:
        print("  No data downloaded — skipping master file")


    print()
    print("=" * 60)
    print("STEP 5: Writing download log")
    print("=" * 60)

    log = {
        "download_timestamp": datetime.now().isoformat(),
        "config_file": str(CONFIG_PATH),
        "date_range": {
            "start": config["start_date"],
            "end": config["end_date"],
        },
        "total_series_requested": len(all_series),
        "total_series_downloaded": len(downloaded_data),
        "errors": errors,
        "series": log_entries,
    }

    log_path = output_dir / config["log_file"]
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Log saved: {log_path}")

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Series downloaded: {len(downloaded_data)}/{len(all_series)}")
    print(f"  Output directory:  {output_dir}")
    print()

    # Print a table of all series with their row counts
    print(f"  {'Series ID':<20} {'Rows':>6}  {'Date Range':<25} Status")
    print(f"  {'-'*20} {'-'*6}  {'-'*25} {'-'*8}")
    for entry in log_entries:
        sid = entry["series_id"]
        rows = entry["rows_downloaded"]
        status = entry["status"]
        if status == "success":
            dr = f"{entry['date_range']['start']} to {entry['date_range']['end']}"
        else:
            dr = "N/A"
        print(f"  {sid:<20} {rows:>6}  {dr:<25} {status}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  No errors!")

    print()


if __name__ == "__main__":
    main()
