"""Feature engineering: transform cleaned battle DataFrame into a feature matrix.

Reads from the battles table (via cleaner.py), computes features, writes
results to the feature_rows table, and returns (X, y) for model training.
"""

import io
import json
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from src.db.session import SessionLocal, engine
from src.features.weapon_classes import WEAPON_CLASS_LIST, get_weapon_class
from src.logging_config import get_logger
from src.preprocessing.cleaner import clean, load_battles

logger = get_logger(__name__)

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

    logger.info("Engineering features for %d rows...", len(df))
    t0 = time.perf_counter()
    feat = _build_features(df)
    logger.info("  Built %d feature columns in %.1fs",
                feat.shape[1], time.perf_counter() - t0)

    # Sanity check: warn if all weapon counts are zero (weapon mapping failure)
    weapon_cols = [c for c in feat.columns if c.endswith("_count")]
    if weapon_cols and feat[weapon_cols].sum().sum() == 0:
        logger.warning(
            "All weapon class counts are zero — weapon_classes mapping may be out of date."
        )

    y = df["target"]
    feat = feat.fillna(0.0)

    if write_to_db:
        _write_feature_rows(df, feat, y)

    logger.info("Feature matrix: %d rows × %d features.", feat.shape[0], feat.shape[1])
    return feat, y


_LOAD_CHUNK = 100_000


def load_feature_rows() -> tuple[pd.DataFrame, pd.Series]:
    """Load pre-computed feature rows from the DB (skip re-engineering), chunked with progress."""
    t0 = time.perf_counter()
    # Skip JSON one-hot columns in the SELECT — training doesn't use them and they
    # bloat the transfer considerably.
    drop_cols = ["id", "battle_id", "mode_onehot", "stage_onehot", "lobby_onehot"]

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM feature_rows")).scalar_one()
        if total == 0:
            raise RuntimeError(
                "feature_rows table is empty. Run engineer_features(write_to_db=True) first."
            )
        logger.info("Loading %d feature rows from DB (chunks of %d)...", total, _LOAD_CHUNK)

        chunks = []
        with tqdm(total=total, desc="  feature_rows", unit="rows") as pbar:
            for chunk in pd.read_sql(
                text("SELECT * FROM feature_rows"), conn, chunksize=_LOAD_CHUNK,
            ):
                chunk.drop(columns=drop_cols, errors="ignore", inplace=True)
                chunks.append(chunk)
                pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    y = df.pop("target").astype(int)
    logger.info("Loaded %d feature rows in %.1fs.", len(df), time.perf_counter() - t0)
    return df, y


_WRITE_CHUNK = 100_000


def _bulk_copy_postgres(df: pd.DataFrame, table_name: str) -> None:
    """Postgres bulk insert via COPY FROM STDIN. ~20–50× faster than INSERT-based writes."""
    cols_sql = ", ".join(f'"{c}"' for c in df.columns)
    copy_sql = f"COPY {table_name} ({cols_sql}) FROM STDIN WITH (FORMAT CSV)"

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            total = len(df)
            with tqdm(total=total, desc="  feature_rows", unit="rows") as pbar:
                for start in range(0, total, _WRITE_CHUNK):
                    chunk = df.iloc[start : start + _WRITE_CHUNK]
                    buf = io.StringIO()
                    chunk.to_csv(buf, index=False, header=False)
                    buf.seek(0)
                    cur.copy_expert(copy_sql, buf)
                    pbar.update(len(chunk))
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()


def _write_feature_rows(
    battles_df: pd.DataFrame,
    feat: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Persist engineered features to feature_rows.

    Uses Postgres COPY FROM STDIN for speed; falls back to DataFrame.to_sql on SQLite.
    """
    if "id" not in battles_df.columns:
        logger.warning("No 'id' column in battles_df; skipping DB write.")
        return

    is_postgres = engine.dialect.name == "postgresql"

    logger.info("Clearing existing feature_rows before insert...")
    try:
        with SessionLocal() as session:
            if is_postgres:
                # TRUNCATE is effectively instant; DELETE on 4M+ rows took ~54s.
                session.execute(text("TRUNCATE TABLE feature_rows RESTART IDENTITY"))
            else:
                session.execute(text("DELETE FROM feature_rows"))
            session.commit()
    except SQLAlchemyError:
        logger.exception("Failed to clear feature_rows table.")
        raise

    onehot_mode_cols = [c for c in feat.columns if c.startswith("mode_")]
    onehot_stage_cols = [c for c in feat.columns if c.startswith("stage_")]
    onehot_lobby_cols = [c for c in feat.columns if c.startswith("lobby_")]
    numeric_cols = [
        c for c in feat.columns
        if not (c.startswith("mode_") or c.startswith("stage_") or c.startswith("lobby_"))
    ]

    out = feat[numeric_cols].copy()
    out["battle_id"] = battles_df["id"].to_numpy()
    out["target"] = y.to_numpy()

    def _rows_to_json(sub: pd.DataFrame) -> pd.Series:
        return pd.Series(sub.to_dict(orient="records"), index=sub.index).map(json.dumps)

    out["mode_onehot"] = _rows_to_json(feat[onehot_mode_cols]) if onehot_mode_cols else None
    out["stage_onehot"] = _rows_to_json(feat[onehot_stage_cols]) if onehot_stage_cols else None
    out["lobby_onehot"] = _rows_to_json(feat[onehot_lobby_cols]) if onehot_lobby_cols else None

    t0 = time.perf_counter()
    logger.info(
        "Writing %d rows to feature_rows (%s, chunks of %d)...",
        len(out),
        "Postgres COPY" if is_postgres else "to_sql",
        _WRITE_CHUNK,
    )
    try:
        if is_postgres:
            _bulk_copy_postgres(out, "feature_rows")
        else:
            out.to_sql(
                "feature_rows",
                engine,
                if_exists="append",
                index=False,
                chunksize=5000,
            )
    except Exception:
        logger.exception("Bulk insert into feature_rows failed.")
        raise
    logger.info("Wrote %d feature rows in %.1fs.", len(out), time.perf_counter() - t0)
