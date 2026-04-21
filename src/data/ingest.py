"""Ingest stat.ink Splatoon 3 CSV files into the battles table."""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from src.db.models import Battle
from src.db.session import SessionLocal, engine
from src.logging_config import get_logger

logger = get_logger(__name__)

# Columns we actually store; everything else in the CSV is discarded.
_KEEP_META = [
    "period", "game-ver", "lobby", "mode", "stage", "win", "knockout",
    "rank", "power",
    "alpha-inked", "alpha-ink-percent", "bravo-inked", "bravo-ink-percent",
]

_PLAYER_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
_PLAYER_STATS = ["weapon", "kill", "assist", "death", "special", "inked"]
_PLAYER_COLS = [f"{pid}-{stat}" for pid in _PLAYER_IDS for stat in _PLAYER_STATS]

_REQUIRED = ["period", "win", "mode", "stage", "A1-weapon"]  # sanity-check fields

# Rename map: CSV hyphenated names → DB underscore names
_RENAME = {col: col.replace("-", "_").lower() for col in _KEEP_META + _PLAYER_COLS}

_BATCH_SIZE = 2000


def _build_upsert_stmt(is_sqlite: bool, records: list[dict]):
    """Build a dialect-specific upsert statement that ignores duplicates."""
    if is_sqlite:
        stmt = sqlite_insert(Battle).values(records)
        return stmt.on_conflict_do_nothing(
            index_elements=["period", "a1_weapon", "a1_kill", "a1_death"]
        )
    stmt = pg_insert(Battle).values(records)
    return stmt.on_conflict_do_nothing(constraint="uq_battle_dedup")


def _validate_csv_columns(df: pd.DataFrame, csv_path: Path) -> None:
    """Fail fast if required columns are missing."""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)[:15]}..."
        )


def ingest_csv(csv_path: str | Path, batch_size: int = _BATCH_SIZE) -> tuple[int, int]:
    """Ingest a single CSV. Returns (inserted, skipped)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            "Download from https://stat.ink/downloads and place in data/raw/."
        )

    logger.info("Reading %s ...", csv_path)
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("  Read %d rows × %d cols in %.1fs",
                len(df), len(df.columns), time.perf_counter() - t0)

    _validate_csv_columns(df, csv_path)

    # Keep only the columns we care about
    wanted = [c for c in _KEEP_META + _PLAYER_COLS if c in df.columns]
    df = df[wanted].copy()
    df.rename(columns=_RENAME, inplace=True)

    # Drop rows with no valid win outcome
    df = df[df["win"].notna() & df["win"].isin(["alpha", "bravo"])]
    if df.empty:
        logger.warning("  No valid rows in %s — skipping.", csv_path.name)
        return 0, 0

    # Deduplicate within this file by the same key as the DB unique constraint.
    dedup_keys = ["period", "a1_weapon", "a1_kill", "a1_death"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="first")
    if before != len(df):
        logger.info("  Dropped %d in-file duplicate rows.", before - len(df))

    is_sqlite = str(engine.url).startswith("sqlite")
    total = len(df)
    inserted = 0

    df = df.astype(object).where(df.notna(), other=None)

    try:
        with SessionLocal() as session:
            for start in tqdm(range(0, total, batch_size), desc=f"  {csv_path.name}"):
                batch = df.iloc[start : start + batch_size]
                records = batch.to_dict(orient="records")
                if not records:
                    continue

                stmt = _build_upsert_stmt(is_sqlite, records)
                result = session.execute(stmt)
                inserted += result.rowcount
                session.commit()
    except SQLAlchemyError:
        logger.exception("  Database error while ingesting %s", csv_path.name)
        raise

    skipped = total - inserted
    logger.info("  Inserted %d, skipped %d duplicates (%.1fs)",
                inserted, skipped, time.perf_counter() - t0)
    return inserted, skipped


def ingest_directory(data_dir: str | Path = "data/raw", **kwargs) -> None:
    """Ingest every CSV file in data_dir."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}/. "
            "Download the stat.ink CSV from https://stat.ink/downloads "
            "and place it (or its extracted files) in that directory."
        )

    logger.info("Found %d CSV file(s) in %s/", len(csv_files), data_dir)
    total_inserted = total_skipped = 0
    t0 = time.perf_counter()

    for idx, path in enumerate(csv_files, 1):
        logger.info("[%d/%d] %s", idx, len(csv_files), path.name)
        try:
            ins, skp = ingest_csv(path, **kwargs)
            total_inserted += ins
            total_skipped += skp
        except Exception as e:
            logger.exception("  Failed to ingest %s: %s", path.name, e)
            raise

    logger.info(
        "Ingest complete: %d inserted, %d skipped across %d files (%.1fs total).",
        total_inserted, total_skipped, len(csv_files), time.perf_counter() - t0,
    )
