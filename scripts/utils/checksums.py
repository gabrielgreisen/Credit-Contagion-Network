"""Compute SHA-256 checksums for the redistributable subset of data/.

Writes data/MANIFEST.sha256 in the standard `sha256sum` format
(`<hash>  <relative-path>`), with paths relative to the repository root.

Only files with redistribution rights are included:
  - data/raw/fred/        — FRED downloads (public domain)
  - data/raw/lopucki/     — LoPucki Cases Table (free with attribution)
  - data/processed/default_events_lopucki_only.parquet
                          — LoPucki-only subset of default events

Files derived from WRDS-licensed sources (Compustat, CRSP, TRACE, DealScan,
BoardEx, FactSet, Orbis) are intentionally NOT manifested. See LICENSE-DATA.md.
"""
import argparse
import hashlib
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

REDISTRIBUTABLE_DIRS = [
    Path("data/raw/fred"),
    Path("data/raw/lopucki"),
]
REDISTRIBUTABLE_FILES = [
    Path("data/processed/default_events_lopucki_only.parquet"),
]
SKIP_NAMES = {".DS_Store"}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for d in REDISTRIBUTABLE_DIRS:
        full = project_root / d
        if not full.is_dir():
            continue
        for p in sorted(full.rglob("*")):
            if p.is_file() and p.name not in SKIP_NAMES:
                files.append(p)
    for f in REDISTRIBUTABLE_FILES:
        full = project_root / f
        if full.is_file():
            files.append(full)
    return files


def main(project_root: Path) -> None:
    files = collect_files(project_root)
    if not files:
        print("No redistributable files found. Nothing written.")
        return

    out_path = project_root / "data/MANIFEST.sha256"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for path in files:
        rel = path.relative_to(project_root)
        digest = sha256_of(path)
        size_mb = path.stat().st_size / 1e6
        lines.append(f"{digest}  {rel}")
        print(f"  {digest[:12]}…  {rel}  ({size_mb:.2f} MB)")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path} ({len(files)} files)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    main(args.project_root.resolve())
