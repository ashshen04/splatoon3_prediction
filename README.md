# Splatoon 3 Battle Outcome Prediction

Binary classification model that predicts Splatoon 3 match outcomes (alpha team wins vs. bravo team) using community battle data from [stat.ink](https://stat.ink/). Trained on 4.6M battles with Logistic Regression, Random Forest, and XGBoost, with SHAP analysis to explain feature importance.

Built as an end-to-end ML portfolio project: raw CSV → PostgreSQL → feature engineering → model comparison → SHAP interpretability → serialized model ready for API serving.

---

## Results

| Model | CV Accuracy | CV F1 Macro |
|---|---|---|
| Logistic Regression | 0.8853 ± 0.0011 | 0.8849 ± 0.0012 |
| Random Forest | 0.8853 ± 0.0011 | 0.8849 ± 0.0011 |
| **XGBoost** | **0.8881 ± 0.0009** | **0.8877 ± 0.0009** |

Evaluated with 5-fold stratified cross-validation on a 500k stratified sample. Best model (XGBoost) is serialized to `results/best_model.joblib`.

**Top SHAP feature: `ink_percent_differential`** — the difference in territory ink coverage between teams is the strongest predictor of match outcome, which aligns with Splatoon 3's core win condition.

![SHAP Summary](results/shap_summary.png)

---

## Setup

**Prerequisites:** Python 3.11+, PostgreSQL (optional — falls back to SQLite)

```bash
git clone <repo-url>
cd splatoon3_prediction

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# macOS only (required for XGBoost)
brew install libomp
```

**Database setup:**

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL (skip to use SQLite fallback)

alembic upgrade head
```

**Data:** Download the Splatoon 3 battle CSV from [stat.ink/downloads](https://stat.ink/downloads) and place it in `data/raw/battle-results-csv/`.

---

## Running the Pipeline

```bash
# Full run: ingest → feature engineering → train → SHAP
python run_pipeline.py

# Skip CSV ingest (battles already in DB)
python run_pipeline.py --skip-ingest

# Skip both ingest and feature engineering (feature_rows table already populated)
python run_pipeline.py --skip-ingest --skip-features

# Ingest only
python run_pipeline.py --ingest-only

# Custom CSV directory
python run_pipeline.py --data-dir path/to/csvs/
```

Pipeline logs stream to stdout and rotate in `logs/pipeline.log`.

**Rough timing on 4.6M battles:**
| Step | Time |
|---|---|
| Load battles from DB | ~90s |
| Feature engineering (80 features) | ~30s |
| Write feature_rows to DB (Postgres COPY) | ~2–3 min |
| 5-fold CV × 3 models (500k sample) | ~15–20 min |
| SHAP analysis | ~2 min |

---

## Architecture

```
run_pipeline.py              # Entry point — CLI flags + step orchestration
src/
├── db/
│   ├── session.py           # SQLAlchemy engine; reads DATABASE_URL, falls back to SQLite
│   └── models.py            # ORM: Battle, FeatureRow, Prediction tables
├── data/
│   └── ingest.py            # stat.ink CSV → battles table (idempotent upsert)
├── preprocessing/
│   └── cleaner.py           # Drop leakage cols, impute, create binary target label
├── features/
│   ├── engineer.py          # Team aggregates, differentials, KD, one-hots → feature_rows
│   └── weapon_classes.py    # ~130 stat.ink weapon codes → 11 class buckets
├── modeling/
│   └── trainer.py           # 5-fold CV, best model refit, saves joblib artifacts
└── analysis/
    └── shap_analysis.py     # TreeExplainer beeswarm + dependence plot
results/
├── best_model.joblib        # Serialized best model (XGBoost)
├── feature_columns.joblib   # Feature column list (for backend serving)
├── model_performance.csv    # CV accuracy + F1 for all models
├── shap_summary.png         # Beeswarm plot — top 20 features by mean |SHAP|
└── shap_dependence_*.png    # Dependence plot for top SHAP feature
```

### Key design decisions

- **Target label:** `win == "alpha"` → 1. Class balance is ~50/50 since stat.ink records both team perspectives.
- **Leakage guard:** `knockout` dropped — it's determined post-match and directly correlated with win.
- **Weapon classes:** 130+ individual weapon codes (e.g. `sshooter`, `52gal`) mapped to 11 class buckets before one-hot encoding to avoid high cardinality.
- **Bulk write:** Feature rows use Postgres `COPY FROM STDIN` (~20–50× faster than `INSERT`). SQLite falls back to `DataFrame.to_sql`.
- **CV subsampling:** Cross-validation runs on a 500k stratified sample — CV metrics are stable past ~200k rows, and running on all 4.6M would multiply training time by ~9× for negligible gain. Final refit uses the same sample.
- **Serialization:** `best_model.joblib` + `feature_columns.joblib` are written to `results/` for the future FastAPI backend to load.

---

## Database Schema

**`battles`** — raw battle records ingested from stat.ink CSV. Stores match metadata (mode, stage, lobby, power) and per-player stats (weapon, kills, deaths, assists, specials, ink) for all 8 players (A1–B4). Unique constraint on `(period, a1_weapon, a1_kill, a1_death)` as a dedup guard.

**`feature_rows`** — one row per battle of engineered features. Populated by `engineer.py`; training reads from here to skip re-computation.

**`predictions`** — future backend request log. Stores input features, predicted outcome, confidence, and model version for each live prediction.

---

## Features Engineered (80 total)

| Group | Features |
|---|---|
| Team sums | `alpha_{stat}_sum`, `bravo_{stat}_sum` for kill, death, assist, special, inked |
| Differentials | `{stat}_differential` (alpha − bravo) for each stat |
| KD ratios | `alpha_kd`, `bravo_kd`, `kd_ratio_differential` |
| Ink coverage | `ink_differential`, `ink_percent_differential` |
| Weapon composition | `alpha_{class}_count`, `bravo_{class}_count` × 11 classes = 22 features |
| Mode (one-hot) | `mode_*` |
| Stage (one-hot) | `stage_*` |
| Lobby (one-hot) | `lobby_*` |
| Power | `power` (median-imputed per lobby) |

---

## Deployment

Designed to deploy on [Railway](https://railway.app/):

1. Provision a Railway Postgres instance
2. Set `DATABASE_URL` as a Railway environment variable
3. Run `alembic upgrade head` in the Railway shell to create tables
4. Run `python run_pipeline.py --ingest-only` to load battle data
5. Run `python run_pipeline.py --skip-ingest` to train and generate artifacts

The serialized model in `results/best_model.joblib` is ready for a FastAPI backend (reserved in `backend/`).

---

## Future Phases

- **`backend/`** — FastAPI endpoint accepting team compositions, returning win probability + SHAP explanation. Logs each prediction to the `predictions` table.
- **`frontend/`** — UI to submit a team composition (8 weapons + stage + mode) and display the predicted outcome with an interactive SHAP waterfall chart.
