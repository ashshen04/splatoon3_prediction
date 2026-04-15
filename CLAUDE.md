# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Splatoon 3 battle outcome prediction — binary classification (alpha team wins vs. bravo team) using stat.ink community battle data. Trains Logistic Regression, Random Forest, and XGBoost; applies SHAP analysis to interpret feature importance. Stores data in PostgreSQL (SQLite fallback for local dev). Deploys to Railway.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL (optional — defaults to SQLite)
alembic upgrade head          # create database tables
```

Place stat.ink Splatoon 3 battle CSV file(s) in `data/` (download from https://stat.ink/downloads).

## Running the pipeline

```bash
python run_pipeline.py                  # full run: ingest → features → train → SHAP
python run_pipeline.py --ingest-only    # only load CSV into DB
python run_pipeline.py --skip-ingest    # skip CSV ingest, use existing DB data
python run_pipeline.py --skip-features  # skip re-engineering, load feature_rows from DB
python run_pipeline.py --data-dir path/ # point to a custom CSV directory
```

## Database migrations

```bash
alembic revision --autogenerate -m "description"   # generate migration after model changes
alembic upgrade head                                # apply migrations
alembic downgrade -1                                # roll back one revision
```

## Architecture

```
run_pipeline.py
    ├── src/db/session.py          — SQLAlchemy engine; reads DATABASE_URL from .env
    ├── src/db/models.py           — ORM: Battle, FeatureRow, Prediction tables
    ├── src/data/ingest.py         — CSV → battles table (idempotent upsert)
    ├── src/preprocessing/cleaner.py — load from DB, drop leakage cols, impute, create target
    ├── src/features/
    │   ├── weapon_classes.py      — weapon name → class mapping (~150 weapons, 11 classes)
    │   └── engineer.py            — team aggregates, differentials, KD, one-hots → feature_rows table
    ├── src/modeling/trainer.py    — 5-fold CV on 3 models; saves best_model.joblib
    └── src/analysis/shap_analysis.py — TreeExplainer; saves shap_summary.png + dependence plot
```

### Key design decisions

- **Target label**: `win == "alpha"` (1 = alpha team wins). Class balance is ~50/50 since stat.ink records both outcomes.
- **Leakage guard**: `knockout` and `time` are dropped — both are determined post-match and directly correlated with win.
- **Weapon classes**: Individual weapon names (100+) are mapped to 11 class buckets before one-hot encoding to avoid high cardinality.
- **One-hot categoricals** (mode, stage, lobby) are stored as JSON in `feature_rows.mode_onehot` etc.; `load_feature_rows()` expands them back into columns for training.
- **Model serialization**: `results/best_model.joblib` and `results/feature_columns.joblib` are written by `trainer.py` for the future backend to load.
- **SQLite fallback**: `session.py` uses `sqlite:///splatoon3.db` when `DATABASE_URL` is unset, so the pipeline runs without Postgres during local development.

### Future phases

- `backend/` — FastAPI serving predictions, logging to `predictions` table
- `frontend/` — UI to submit a team composition and display the prediction + SHAP explanation
