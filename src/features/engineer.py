"""Feature engineering: transform cleaned battle DataFrame into a feature matrix.

Reads from the battles table (via cleaner.py), computes features, writes
results to the feature_rows table, and returns (X, y) for model training.
"""

import json

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from src.db.models import FeatureRow
from src.db.session import SessionLocal, engine
from src.features.weapon_classes import WEAPON_CLASS_LIST, get_weapon_class
from src.preprocessing.cleaner import clean, load_battles

_ALPHA_IDS = ["a1", "a2", "a3", "a4"]
_BRAVO_IDS = ["b1", "b2", "b3", "b4"]
_STATS = ["kill", "death", "assist", "special", "inked"]

_BATCH_SIZE = 5000


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered features. Returns a new DataFrame."""
    feat = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------ Team sums
    for stat in _STATS:
        feat[f"alpha_{stat}_sum"] = df[[f"{p}_{stat}" for p in _ALPHA_IDS]].sum(axis=1)
        feat[f"bravo_{stat}_sum"] = df[[f"{p}_{stat}" for p in _BRAVO_IDS]].sum(axis=1)
        feat[f"{stat}_differential"] = feat[f"alpha_{stat}_sum"] - feat[f"bravo_{stat}_sum"]

    # ------------------------------------------------------------------ KD ratios
    feat["alpha_kd"] = feat["alpha_kill_sum"] / (feat["alpha_death_sum"] + 1)
    feat["bravo_kd"] = feat["bravo_kill_sum"] / (feat["bravo_death_sum"] + 1)
    feat["kd_ratio_differential"] = feat["alpha_kd"] - feat["bravo_kd"]

    # ------------------------------------------------------------------ Ink
    if "alpha_inked" in df.columns and "bravo_inked" in df.columns:
        feat["ink_differential"] = df["alpha_inked"] - df["bravo_inked"]
    else:
        # Fall back to player-sum inked differential
        feat["ink_differential"] = feat["alpha_inked_sum"] - feat["bravo_inked_sum"]

    if "alpha_ink_percent" in df.columns and "bravo_ink_percent" in df.columns:
        feat["ink_percent_differential"] = df["alpha_ink_percent"] - df["bravo_ink_percent"]
    else:
        feat["ink_percent_differential"] = 0.0

    # ------------------------------------------------------------------ Power
    feat["power"] = df["power"].fillna(df["power"].median()) if "power" in df.columns else 0.0

    # ------------------------------------------------------------------ Weapon class counts
    for team, pids in [("alpha", _ALPHA_IDS), ("bravo", _BRAVO_IDS)]:
        weapon_cols = [f"{p}_weapon" for p in pids if f"{p}_weapon" in df.columns]
        # Map each player's weapon to a class. Use DataFrame.map (pandas 2.1+);
        # falls back to applymap for older pandas.
        try:
            class_df = df[weapon_cols].map(get_weapon_class)
        except AttributeError:
            class_df = df[weapon_cols].applymap(get_weapon_class)
        for cls in WEAPON_CLASS_LIST:
            feat[f"{team}_{cls.lower()}_count"] = (class_df == cls).sum(axis=1).astype(float)

    # ------------------------------------------------------------------ Categoricals (one-hot)
    for col in ["mode", "stage", "lobby"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            feat = pd.concat([feat, dummies], axis=1)

    return feat


def engineer_features(
    df: pd.DataFrame | None = None,
    write_to_db: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix from battles data.

    Args:
        df: Pre-loaded and cleaned DataFrame. If None, loads from DB.
        write_to_db: Whether to persist feature rows to the feature_rows table.

    Returns:
        (X, y) — feature DataFrame and target Series.
    """
    if df is None:
        raw = load_battles()
        df = clean(raw)

    print("Engineering features...")
    feat = _build_features(df)
    y = df["target"]

    # Ensure no NaN values remain
    feat = feat.fillna(0.0)

    if write_to_db:
        _write_feature_rows(df, feat, y)

    print(f"Feature matrix: {feat.shape[0]:,} rows × {feat.shape[1]} features.")
    return feat, y


def load_feature_rows() -> tuple[pd.DataFrame, pd.Series]:
    """Load pre-computed feature rows from the DB (skip re-engineering)."""
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM feature_rows"), conn)

    if df.empty:
        raise RuntimeError(
            "feature_rows table is empty. Run engineer_features(write_to_db=True) first."
        )

    y = df.pop("target").astype(int)
    # Drop metadata columns not used as features
    df.drop(columns=["id", "battle_id", "mode_onehot", "stage_onehot", "lobby_onehot"],
            errors="ignore", inplace=True)

    print(f"Loaded {len(df):,} feature rows from DB.")
    return df, y


def _write_feature_rows(
    battles_df: pd.DataFrame,
    feat: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Persist engineered features to feature_rows table.

    Truncates the table first so re-runs don't duplicate rows.
    """
    print("Writing feature rows to DB (clearing existing rows first)...")
    with SessionLocal() as session:
        session.execute(text("DELETE FROM feature_rows"))
        session.commit()

    # Separate out the one-hot columns from the numeric columns
    onehot_mode_cols = [c for c in feat.columns if c.startswith("mode_")]
    onehot_stage_cols = [c for c in feat.columns if c.startswith("stage_")]
    onehot_lobby_cols = [c for c in feat.columns if c.startswith("lobby_")]
    numeric_cols = [
        c for c in feat.columns
        if not (c.startswith("mode_") or c.startswith("stage_") or c.startswith("lobby_"))
    ]

    # We need battle IDs to write FKs — assumes battles_df has an 'id' column
    if "id" not in battles_df.columns:
        print("Warning: no 'id' column in battles_df; skipping DB write.")
        return

    rows = []
    for idx in feat.index:
        row = {col: feat.at[idx, col] for col in numeric_cols}
        row["battle_id"] = int(battles_df.at[idx, "id"])
        row["target"] = int(y.at[idx])
        row["mode_onehot"] = json.dumps(
            {c: feat.at[idx, c] for c in onehot_mode_cols}
        ) if onehot_mode_cols else None
        row["stage_onehot"] = json.dumps(
            {c: feat.at[idx, c] for c in onehot_stage_cols}
        ) if onehot_stage_cols else None
        row["lobby_onehot"] = json.dumps(
            {c: feat.at[idx, c] for c in onehot_lobby_cols}
        ) if onehot_lobby_cols else None
        rows.append(row)

    with SessionLocal() as session:
        for start in tqdm(range(0, len(rows), _BATCH_SIZE), desc="Writing feature_rows"):
            batch = rows[start : start + _BATCH_SIZE]
            session.bulk_insert_mappings(FeatureRow, batch)
            session.commit()

    print(f"Wrote {len(rows):,} feature rows to DB.")
