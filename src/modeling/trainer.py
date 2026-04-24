"""Train and evaluate Logistic Regression, Random Forest, and XGBoost.

Uses 5-fold stratified cross-validation. Saves the best model and feature
column list to results/ for use by the backend.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("results")

# Cap for CV subsample. Empirically, CV metrics are stable past ~200k rows; going
# from 500k → 4.6M would multiply RF training time by ~9× for negligible gain.
_CV_SAMPLE_CAP = 500_000

MODELS: dict[str, any] = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


def _maybe_subsample(
    X: pd.DataFrame, y: pd.Series, cap: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified-ish subsample if len(X) > cap. Uses simple random sampling."""
    if len(X) <= cap:
        return X, y
    logger.info("Subsampling %d → %d rows for CV (set higher to use more data).",
                len(X), cap)
    idx = X.sample(n=cap, random_state=42).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, sample_cap: int = _CV_SAMPLE_CAP,
) -> pd.DataFrame:
    """Run 5-fold CV on all models; return a performance metrics DataFrame."""
    X_cv, y_cv = _maybe_subsample(X, y, sample_cap)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in MODELS.items():
        t0 = time.perf_counter()
        logger.info("Cross-validating %s on %d rows...", name, len(X_cv))
        try:
            scores = cross_validate(
                model,
                X_cv,
                y_cv,
                cv=cv,
                scoring=["accuracy", "f1_macro"],
                return_train_score=False,
                n_jobs=-1,
                verbose=1,
            )
        except Exception:
            logger.exception("Cross-validation for %s failed.", name)
            raise
        logger.info(
            "  %s: acc=%.4f±%.4f  f1=%.4f±%.4f  (%.1fs)",
            name,
            scores["test_accuracy"].mean(), scores["test_accuracy"].std(),
            scores["test_f1_macro"].mean(), scores["test_f1_macro"].std(),
            time.perf_counter() - t0,
        )
        results.append(
            {
                "Model": name,
                "CV Accuracy": f"{scores['test_accuracy'].mean():.4f}",
                "Accuracy Std": f"±{scores['test_accuracy'].std():.4f}",
                "CV F1 Macro": f"{scores['test_f1_macro'].mean():.4f}",
                "F1 Std": f"±{scores['test_f1_macro'].std():.4f}",
            }
        )

    return pd.DataFrame(results)


def get_best_model(
    metrics_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[any, str]:
    """Identify the best model by CV F1 Macro, refit on full data, save to disk."""
    best_name = metrics_df.loc[
        metrics_df["CV F1 Macro"].astype(float).idxmax(), "Model"
    ]
    logger.info("Best model: %s — refitting on full dataset...", best_name)
    t0 = time.perf_counter()
    best_model = MODELS[best_name]
    best_model.fit(X, y)
    logger.info("  Refit complete in %.1fs", time.perf_counter() - t0)

    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = RESULTS_DIR / "best_model.joblib"
    cols_path = RESULTS_DIR / "feature_columns.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(X.columns.tolist(), cols_path)
    logger.info("Saved model → %s", model_path)
    logger.info("Saved feature columns → %s", cols_path)

    return best_model, best_name


def run_training(
    X: pd.DataFrame, y: pd.Series, sample_cap: int = _CV_SAMPLE_CAP,
) -> tuple[pd.DataFrame, any, str]:
    """Full training flow: CV evaluation + best model refit.

    Both CV and final refit use the same subsample (if X exceeds sample_cap).
    """
    logger.info("Training models (5-fold CV) on %d rows × %d features...",
                X.shape[0], X.shape[1])
    X_cv, y_cv = _maybe_subsample(X, y, sample_cap)
    metrics_df = train_and_evaluate(X_cv, y_cv, sample_cap=sample_cap)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "model_performance.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Model performance:\n%s", metrics_df.to_string(index=False))
    logger.info("Saved → %s", csv_path)

    best_model, best_name = get_best_model(metrics_df, X_cv, y_cv)
    return metrics_df, best_model, best_name
