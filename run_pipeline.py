"""End-to-end Splatoon 3 battle outcome prediction pipeline.

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --ingest-only    # only load CSV → DB
    python run_pipeline.py --skip-ingest    # skip CSV ingest, use existing DB data
    python run_pipeline.py --skip-features  # skip re-engineering, load from feature_rows table
"""

import argparse
from pathlib import Path

from src.db.models import Base
from src.db.session import engine


def setup_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    print("Database tables ready.")


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
        "--data-dir", default="data",
        help="Directory containing stat.ink CSV files (default: data/)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ DB setup
    setup_db()

    # ------------------------------------------------------------------ Ingest
    if not args.skip_ingest:
        from src.data.ingest import ingest_directory
        print("\n--- Step 1: Ingest CSV → DB ---")
        ingest_directory(args.data_dir)
    else:
        print("\n--- Step 1: Skipping CSV ingest ---")

    if args.ingest_only:
        print("--ingest-only flag set. Done.")
        return

    # ------------------------------------------------------------------ Features
    if args.skip_features:
        print("\n--- Step 2: Loading pre-computed features from DB ---")
        from src.features.engineer import load_feature_rows
        X, y = load_feature_rows()
    else:
        print("\n--- Step 2: Feature engineering ---")
        from src.features.engineer import engineer_features
        X, y = engineer_features(write_to_db=True)

    # ------------------------------------------------------------------ Training
    print("\n--- Step 3: Model training ---")
    from src.modeling.trainer import run_training
    metrics_df, best_model, best_name = run_training(X, y)

    # ------------------------------------------------------------------ SHAP
    print(f"\n--- Step 4: SHAP analysis on {best_name} ---")
    from src.analysis.shap_analysis import run_shap_analysis
    run_shap_analysis(best_model, X, model_name=best_name)

    print("\n=== Pipeline complete. Results saved to results/ ===")


if __name__ == "__main__":
    main()
