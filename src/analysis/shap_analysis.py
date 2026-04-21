"""SHAP analysis on the best trained model.

Generates:
  - results/shap_summary.png     — beeswarm summary of top features
  - results/shap_dependence_<feature>.png  — dependence plot for top feature
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("results")
SAMPLE_SIZE = 5000


def run_shap_analysis(
    model: any,
    X: pd.DataFrame,
    model_name: str = "model",
) -> None:
    """Compute SHAP values and save plots.

    Args:
        model: Fitted model (XGBClassifier, RandomForestClassifier, or Pipeline).
        X: Feature DataFrame (same columns used for training).
        model_name: Used only for console messages.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # Sample for speed
    n = min(SAMPLE_SIZE, len(X))
    X_sample = X.sample(n, random_state=42).reset_index(drop=True)

    # Unwrap pipeline to get the underlying estimator
    estimator = model
    X_for_shap = X_sample
    if isinstance(model, Pipeline):
        # Apply all transforms, feed transformed data to the final estimator
        for name_, step in model.steps[:-1]:
            X_for_shap = step.transform(X_for_shap)
        estimator = model.steps[-1][1]
        # Convert back to DataFrame so feature names are preserved
        if not isinstance(X_for_shap, pd.DataFrame):
            X_for_shap = pd.DataFrame(X_for_shap, columns=X_sample.columns)

    # Choose explainer
    estimator_type = type(estimator).__name__
    logger.info("Running SHAP on %s (%s) — sample size %d",
                model_name, estimator_type, n)
    t0 = time.perf_counter()

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_for_shap)
        # RandomForest returns list [class0, class1]; take class1 (alpha wins)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Logistic Regression fallback
        explainer = shap.LinearExplainer(estimator, X_for_shap)
        shap_values = explainer.shap_values(X_for_shap)

    feature_names = X_for_shap.columns.tolist()

    # ------------------------------------------------------------------ Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_for_shap,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    summary_path = RESULTS_DIR / "shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved → %s", summary_path)

    # ------------------------------------------------------------------ Dependence plot (top feature)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feature = feature_names[top_idx]
    logger.info("Top SHAP feature: %s", top_feature)

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        top_idx,
        shap_values,
        X_for_shap,
        feature_names=feature_names,
        show=False,
    )
    dep_path = RESULTS_DIR / f"shap_dependence_{top_feature}.png"
    plt.savefig(dep_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved → %s", dep_path)
    logger.info("SHAP analysis complete in %.1fs", time.perf_counter() - t0)
