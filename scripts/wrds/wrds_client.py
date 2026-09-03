"""
WRDS Connection Wrapper

Thin layer around the `wrds` Python package. Handles:
  - pgpass-based authentication (with env-var fallback)
  - library access detection (explicit list + probe queries for trial-tab libs)
  - column introspection via describe_table()
  - streaming queries via SQLAlchemy's chunksize mechanism
"""


import os
import sys
import stat
from pathlib import Path

import pandas as pd


PGPASS_HOST = "wrds-pgdata.wharton.upenn.edu"



class WRDSClient:
    """Thin wrapper over wrds.Connection."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.db = None
        self._accessible_libraries: set[str] | None = None
        self._probe_results: dict[str, dict] = {}

    # ---------- connection ----------

    def connect(self) -> None:
        """Open a WRDS connection using pgpass or env vars."""
        self._check_credentials()

        try:
            import wrds
        except ImportError:
            print("ERROR: the `wrds` Python package is not installed.")
            print("       Install with: pip install wrds")
            sys.exit(2)

        username = os.environ.get("WRDS_USERNAME") or self._username_from_pgpass()

        if self.verbose:
            print(f"  Connecting to {PGPASS_HOST} as {username or '<prompt>'} ...")

        # wrds.Connection always prompts for username unless passed explicitly.
        # Password is read from ~/.pgpass automatically by libpq.
        if username:
            self.db = wrds.Connection(wrds_username=username)
        else:
            self.db = wrds.Connection()

        if self.verbose:
            print(f"  Connected.")

    def _username_from_pgpass(self) -> str | None:
        """Extract WRDS username from the first wrds-pgdata line in ~/.pgpass."""
        pgpass = Path.home() / ".pgpass"
        if not pgpass.exists():
            return None
        try:
            with open(pgpass) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":")
                    if len(parts) >= 5 and PGPASS_HOST in parts[0]:
                        return parts[3]
        except Exception:
            return None
        return None

    def _check_credentials(self) -> None:
        pgpass = Path.home() / ".pgpass"
        env_user = os.environ.get("WRDS_USERNAME")
        env_pass = os.environ.get("WRDS_PASSWORD")

        if pgpass.exists():
            mode = pgpass.stat().st_mode & 0o777
            if mode != 0o600:
                print(f"WARNING: {pgpass} has mode {oct(mode)}; should be 0600.")
                print(f"         Run: chmod 600 {pgpass}")
            return

        if env_user and env_pass:
            return

        print("ERROR: no WRDS credentials found.")
        print(f"       Create ~/.pgpass with:")
        print(f'         echo "{PGPASS_HOST}:9737:wrds:<username>:<password>" >> ~/.pgpass')
        print(f"         chmod 600 ~/.pgpass")
        print(f"       Or set WRDS_USERNAME and WRDS_PASSWORD environment variables.")
        sys.exit(2)

    def close(self) -> None:
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

    # ---------- library / table access ----------

    def list_libraries_primary(self) -> set[str]:
        """Libraries returned by the WRDS subscription list."""
        libs = self.db.list_libraries()
        return set(libs) if libs else set()

    def probe_table(self, library: str, table: str) -> dict:
        """
        Try `SELECT 1 FROM {library}.{table} LIMIT 1`. Fast access check
        that catches trial-tab libraries not surfaced by list_libraries().
        Returns dict with keys: ok (bool), error (str or None).
        """
        key = f"{library}.{table}"
        if key in self._probe_results:
            return self._probe_results[key]

        sql = f"SELECT 1 FROM {library}.{table} LIMIT 1"
        try:
            self.db.raw_sql(sql)
            result = {"ok": True, "error": None}
        except Exception as e:
            result = {"ok": False, "error": str(e).splitlines()[0][:200]}
        self._probe_results[key] = result
        return result

    def describe_table(self, library: str, table: str) -> pd.DataFrame:
        """Return a DataFrame with column names and types for a given table."""
        return self.db.describe_table(library=library, table=table)

    def get_actual_columns(self, library: str, table: str) -> list[str]:
        """Return the actual column names from the information_schema."""
        sql = (
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema = '{library}' AND table_name = '{table}' "
            "ORDER BY ordinal_position"
        )
        df = self.db.raw_sql(sql)
        return df["column_name"].tolist()

    # ---------- queries ----------

    def raw_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        return self.db.raw_sql(sql, params=params)

    def stream_sql(self, sql: str, chunksize: int = 500_000):
        """
        Yield DataFrames for a SQL query, chunk_size rows at a time.
        Uses the SQLAlchemy engine exposed by the wrds connection.
        """
        engine = self.db.engine
        with engine.connect() as conn:
            for chunk in pd.read_sql(sql, conn, chunksize=chunksize):
                yield chunk

    def count_rows(self, from_and_where: str) -> int:
        """
        Run SELECT count(*) against an arbitrary FROM/WHERE fragment.
        `from_and_where` should begin with 'FROM ...' and may include WHERE.
        """
        sql = f"SELECT COUNT(*) AS n {from_and_where}"
        df = self.db.raw_sql(sql)
        return int(df.iloc[0]["n"])
