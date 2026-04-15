"""Ingest stat.ink Splatoon 3 CSV files into the battles table.

Usage:
    from src.data.ingest import ingest_csv
    ingest_csv("data/battle-results.csv")

Or via run_pipeline.py --ingest-only.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from tqdm import tqdm

from src.db.models import Battle
from src.db.session import SessionLocal, engine

# Columns we actually store; everything else in the CSV is discarded.
_KEEP_META = [
    "period", "game-ver", "lobby", "mode", "stage", "win", "knockout",
    "rank", "power",
    "alpha-inked", "alpha-ink-percent", "bravo-inked", "bravo-ink-percent",
]

_PLAYER_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
_PLAYER_STATS = ["weapon", "kill", "assist", "death", "special", "inked"]

# Build the full list of per-player columns expected in the CSV
_PLAYER_COLS = [f"{pid}-{stat}" for pid in _PLAYER_IDS for stat in _PLAYER_STATS]

# Rename map: CSV hyphenated names → DB underscore names
_RENAME = {col: col.replace("-", "_").lower() for col in _KEEP_META + _PLAYER_COLS}

_BATCH_SIZE = 2000


def _get_insert_stmt(is_sqlite: bool):
    """Return the correct dialect upsert (ignore duplicates)."""
    if is_sqlite:
        return sqlite_insert(Battle)
    return pg_insert(Battle)


def ingest_csv(csv_path: str | Path, batch_size: int = _BATCH_SIZE) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}.\n"
            "Download the stat.ink Splatoon 3 battle CSV from https://stat.ink/downloads\n"
            "and place it in the data/ directory."
        )

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only the columns we care about
    wanted = [c for c in _KEEP_META + _PLAYER_COLS if c in df.columns]
    df = df[wanted].copy()
    df.rename(columns=_RENAME, inplace=True)

    # Drop rows with no win outcome recorded
    df = df[df["win"].notna() & df["win"].isin(["alpha", "bravo"])]

    is_sqlite = str(engine.url).startswith("sqlite")
    total = len(df)
    inserted = 0

    with SessionLocal() as session:
        for start in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
            batch = df.iloc[start : start + batch_size]
            records = batch.where(batch.notna(), other=None).to_dict(orient="records")

            stmt = _get_insert_stmt(is_sqlite)(records)
            if is_sqlite:
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["period", "a1_weapon", "a1_kill", "a1_death"]
                )
            else:
                stmt = stmt.on_conflict_do_nothing(constraint="uq_battle_dedup")

            result = session.execute(stmt)
            inserted += result.rowcount
            session.commit()

    skipped = total - inserted
    print(f"Done. Inserted {inserted:,} rows, skipped {skipped:,} duplicates.")


def ingest_directory(data_dir: str | Path = "data", **kwargs) -> None:
    """Ingest all CSV files found in data_dir."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}/.\n"
            "Download the stat.ink Splatoon 3 battle CSV from https://stat.ink/downloads\n"
            "and place it in the data/ directory."
        )
    for path in csv_files:
        ingest_csv(path, **kwargs)
