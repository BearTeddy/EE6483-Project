from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .data import LABEL_COL, REVIEW_COL
from .modeling import build_classical_pipeline


def compute_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_params = model_params or {}

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[REVIEW_COL],
        train_df[LABEL_COL],
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[LABEL_COL],
    )

    model = build_classical_pipeline(model_name, random_state=random_state, **model_params)
    model.fit(x_train, y_train)
    val_preds = model.predict(x_val)

    metrics = {
        "model_name": model_name,
        "num_samples": int(len(train_df)),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        **compute_classification_metrics(y_val, val_preds),
        "settings": {
            "test_size": test_size,
            "random_state": random_state,
            **model_params,
        },
    }
    return model, metrics


def benchmark_classical_models(
    train_df: pd.DataFrame,
    *,
    model_names: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    shared_model_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    shared_model_params = shared_model_params or {}

    models: dict[str, Any] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        model, metrics = train_and_evaluate_model(
            train_df,
            model_name=model_name,
            test_size=test_size,
            random_state=random_state,
            model_params=shared_model_params,
        )
        models[model_name] = model
        metrics_by_model[model_name] = metrics

    return models, metrics_by_model


def train_and_evaluate(
    train_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    regularization_c: float = 4.0,
) -> tuple[Any, dict[str, Any]]:
    # Backward-compatible wrapper for existing TF-IDF + Logistic Regression baseline code.
    return train_and_evaluate_model(
        train_df,
        model_name="tfidf_logreg",
        test_size=test_size,
        random_state=random_state,
        model_params={
            "ngram_min": ngram_min,
            "ngram_max": ngram_max,
            "max_features": max_features,
            "min_df": min_df,
            "regularization_c": regularization_c,
        },
    )


def save_artifacts(model: Any, metrics: dict[str, Any], *, model_path: str | Path, metrics_path: str | Path) -> None:
    model_output = Path(model_path)
    metrics_output = Path(metrics_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output)

    with metrics_output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_model(model_path: str | Path) -> Any:
    return joblib.load(model_path)


def predict_texts(model: Any, texts: pd.Series | list[str] | np.ndarray) -> np.ndarray:
    return model.predict(texts)
