"""End-to-end Splatoon 3 battle outcome prediction pipeline.

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --ingest-only    # only load CSV → DB
    python run_pipeline.py --skip-ingest    # skip CSV ingest, use existing DB data
    python run_pipeline.py --skip-features  # skip re-engineering, load from feature_rows table
"""

import argparse
import sys
import time

from src.db.models import Base
from src.db.session import check_db_connection, engine
from src.logging_config import get_logger

logger = get_logger(__name__)


def setup_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    logger.info("Database tables ready.")


def _run_step(label: str, fn, *args, **kwargs):
    """Run a pipeline step with timing + error logging. Exits on failure."""
    logger.info("=== %s ===", label)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception:
        logger.exception("Step failed: %s", label)
        sys.exit(1)
    logger.info("%s finished in %.1fs", label, time.perf_counter() - t0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Splatoon 3 battle outcome prediction pipeline")
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Only ingest CSV files into the database, then exit."
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip CSV ingest; use whatever battles are already in the DB."
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Skip feature engineering; load pre-computed feature_rows from DB."
    )
    parser.add_argument(
        "--data-dir", default="data/raw/battle-results-csv",
        help="Directory containing stat.ink CSV files (default: data/raw/battle-results-csv)."
    )
    args = parser.parse_args()

    pipeline_t0 = time.perf_counter()
    logger.info("Starting Splatoon 3 prediction pipeline.")

    # ------------------------------------------------------------------ Pre-flight
    try:
        check_db_connection()
    except Exception:
        logger.exception("Pre-flight DB check failed — aborting.")
        sys.exit(1)

    _run_step("Step 0: DB setup", setup_db)

    # ------------------------------------------------------------------ Ingest
    if not args.skip_ingest:
        from src.data.ingest import ingest_directory
        _run_step("Step 1: Ingest CSV → DB", ingest_directory, args.data_dir)
    else:
        logger.info("=== Step 1: Skipping CSV ingest ===")

    if args.ingest_only:
        logger.info("--ingest-only flag set. Done in %.1fs.",
                    time.perf_counter() - pipeline_t0)
        return

    # ------------------------------------------------------------------ Features
    if args.skip_features:
        from src.features.engineer import load_feature_rows
        X, y = _run_step("Step 2: Load pre-computed features from DB", load_feature_rows)
    else:
        from src.features.engineer import engineer_features
        X, y = _run_step("Step 2: Feature engineering", engineer_features, write_to_db=True)

    # ------------------------------------------------------------------ Training
    from src.modeling.trainer import run_training
    _, best_model, best_name = _run_step("Step 3: Model training", run_training, X, y)

    # ------------------------------------------------------------------ SHAP
    from src.analysis.shap_analysis import run_shap_analysis
    _run_step(
        f"Step 4: SHAP analysis on {best_name}",
        run_shap_analysis, best_model, X, model_name=best_name,
    )

    logger.info("Pipeline complete in %.1fs. Results saved to results/.",
                time.perf_counter() - pipeline_t0)


if __name__ == "__main__":
    main()
