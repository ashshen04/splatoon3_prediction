"""Clean raw battle data loaded from the battles table.

Returns a cleaned DataFrame ready for feature engineering.
"""

import pandas as pd
from sqlalchemy import text

from src.db.session import engine

_PLAYER_IDS = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]

# Columns to drop before feature engineering
_DROP_COLS = [
    "knockout",             # post-match outcome — leakage
    "rank",                 # categorical with many nulls; low signal vs. power
]


def load_battles() -> pd.DataFrame:
    """Load all battles from the DB into a DataFrame."""
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM battles"), conn)
    print(f"Loaded {len(df):,} battles from DB.")
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

    print(
        f"After cleaning: {len(df):,} rows. "
        f"Target balance: {df['target'].mean():.1%} alpha wins."
    )
    return df
