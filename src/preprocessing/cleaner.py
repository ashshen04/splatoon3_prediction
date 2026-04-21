"""Clean raw battle data loaded from the battles table.

Returns a cleaned DataFrame ready for feature engineering.
"""

import time

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from src.db.session import engine
from src.logging_config import get_logger

logger = get_logger(__name__)

_PLAYER_IDS = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]

# Columns to drop before feature engineering
_DROP_COLS = [
    "knockout",             # post-match outcome — leakage
    "rank",                 # categorical with many nulls; low signal vs. power
]


_LOAD_CHUNK = 100_000


def load_battles() -> pd.DataFrame:
    """Load all battles from the DB into a DataFrame, chunked with progress bar."""
    t0 = time.perf_counter()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM battles")).scalar_one()
        if total == 0:
            raise RuntimeError(
                "battles table is empty. Run `python run_pipeline.py --ingest-only` first."
            )
        logger.info("Loading %d battles from DB (chunks of %d)...", total, _LOAD_CHUNK)

        chunks = []
        with tqdm(total=total, desc="  load_battles", unit="rows") as pbar:
            for chunk in pd.read_sql(
                text("SELECT * FROM battles"), conn, chunksize=_LOAD_CHUNK,
            ):
                chunks.append(chunk)
                pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d battles in %.1fs", len(df), time.perf_counter() - t0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps and return a cleaned DataFrame.

    Steps:
    1. Drop leakage / irrelevant columns.
    2. Drop rows with null win, mode, or stage.
    3. Impute power (median per lobby).
    4. Fill player stats with 0 (absent players).
    5. Fill missing weapon names with 'Unknown'.
    6. Create binary target label; drop raw win column.
    """
    df = df.copy()

    # 1. Drop leakage columns (ignore missing ones gracefully)
    df.drop(columns=[c for c in _DROP_COLS if c in df.columns], inplace=True)

    # 2. Drop rows where we can't define the target or key categoricals
    df = df[df["win"].isin(["alpha", "bravo"])]
    df = df[df["mode"].notna() & df["stage"].notna()]
    df.reset_index(drop=True, inplace=True)

    # 3. Impute power per lobby (median)
    if "power" in df.columns:
        df["power"] = df.groupby("lobby")["power"].transform(
            lambda x: x.fillna(x.median())
        )
        # Any remaining nulls (lobbies with all-null power) → global median
        df["power"] = df["power"].fillna(df["power"].median())

    # 4. Fill player numeric stats with 0
    numeric_player_cols = [
        f"{pid}_{stat}"
        for pid in _PLAYER_IDS
        for stat in ["kill", "assist", "death", "special", "inked"]
    ]
    for col in numeric_player_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 5. Fill missing weapon names
    weapon_cols = [f"{pid}_weapon" for pid in _PLAYER_IDS]
    for col in weapon_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 6. Target label
    df["target"] = (df["win"] == "alpha").astype(int)
    df.drop(columns=["win"], inplace=True)

    logger.info(
        "After cleaning: %d rows. Target balance: %.1f%% alpha wins.",
        len(df), df["target"].mean() * 100,
    )
    return df
