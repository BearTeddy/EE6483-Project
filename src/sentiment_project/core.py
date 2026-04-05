from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

REVIEW_COL = "reviews"
LABEL_COL = "sentiments"


def _load_json_records(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected list of records in {file_path}, got: {type(records)!r}")

    return records


def _check_columns(df: pd.DataFrame, required_columns: Iterable[str], path: str | Path) -> None:
    missing = set(required_columns).difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing expected column(s) in {path}: {missing_str}")


def load_train_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(_load_json_records(path))
    _check_columns(df, [REVIEW_COL, LABEL_COL], path)

    df = df[[REVIEW_COL, LABEL_COL]].dropna(subset=[REVIEW_COL, LABEL_COL]).copy()
    df[REVIEW_COL] = df[REVIEW_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    invalid_mask = ~df[LABEL_COL].isin([0, 1])
    if invalid_mask.any():
        invalid_values = sorted(df.loc[invalid_mask, LABEL_COL].unique().tolist())
        raise ValueError(f"Unexpected label values in {path}: {invalid_values}. Expected only 0 or 1.")

    return df.reset_index(drop=True)


def load_test_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(_load_json_records(path))
    _check_columns(df, [REVIEW_COL], path)
    df = df[[REVIEW_COL]].dropna(subset=[REVIEW_COL]).copy()
    df[REVIEW_COL] = df[REVIEW_COL].astype(str)
    return df.reset_index(drop=True)


def sanitize_name(value: str) -> str:
    sanitized = value.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")


def format_test_size_tag(test_size: float) -> str:
    percent = test_size * 100
    rounded_percent = round(percent)
    if abs(percent - rounded_percent) < 1e-8:
        return f"0_{int(rounded_percent):02d}"

    text = f"{test_size:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def format_test_size_label(test_size: float) -> str:
    return f"{test_size:.0%}"


def build_run_label(model_name: str, test_size: float, seed: int | None) -> str:
    size_label = format_test_size_label(test_size)
    if seed is None:
        return f"{model_name} | test={size_label}"
    return f"{model_name} | test={size_label} | seed={seed}"


def build_experiment_paths(project_root: Path, model_name: str, *, test_size: float, seed: int) -> dict[str, Path]:
    safe_model_name = sanitize_name(model_name)
    test_size_dir = f"test_size_{format_test_size_tag(test_size)}"
    seed_dir = f"seed_{seed}"

    return {
        "model_dir": project_root / "models" / "experiments" / safe_model_name / test_size_dir / seed_dir,
        "reports_dir": project_root / "reports" / "experiments" / safe_model_name / test_size_dir / seed_dir,
        "submission_dir": project_root
        / "data"
        / "submissions"
        / "experiments"
        / safe_model_name
        / test_size_dir
        / seed_dir,
    }


def label_distribution(labels: Iterable[int]) -> dict[str, Any]:
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    fractions = {str(label): (count / total if total else 0.0) for label, count in sorted(counts.items())}
    return {
        "counts": {str(label): int(count) for label, count in sorted(counts.items())},
        "fractions": fractions,
        "total": int(total),
    }


def build_split_metadata(
    *,
    train_labels: Iterable[int],
    validation_labels: Iterable[int],
    test_size: float,
    seed: int,
    total_samples: int,
) -> dict[str, Any]:
    train_distribution = label_distribution(train_labels)
    validation_distribution = label_distribution(validation_labels)
    train_samples = int(train_distribution["total"])
    validation_samples = int(validation_distribution["total"])

    return {
        "seed": int(seed),
        "test_size": float(test_size),
        "num_samples": int(total_samples),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_fraction_observed": train_samples / max(1, total_samples),
        "validation_fraction_observed": validation_samples / max(1, total_samples),
        "train_label_distribution": train_distribution,
        "validation_label_distribution": validation_distribution,
    }


def build_submission_dataframe(predictions: np.ndarray, include_id: bool = True) -> pd.DataFrame:
    predictions = predictions.astype(int)
    if include_id:
        return pd.DataFrame({"id": range(len(predictions)), LABEL_COL: predictions})
    return pd.DataFrame({LABEL_COL: predictions})


def save_submission_csv(submission_df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output, index=False)


def _build_tfidf_vectorizer(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )


def build_tfidf_logreg_pipeline(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    regularization_c: float = 4.0,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                _build_tfidf_vectorizer(
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=regularization_c,
                    max_iter=2_000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_tfidf_svm_pipeline(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    regularization_c: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                _build_tfidf_vectorizer(
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=regularization_c,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_tfidf_nb_pipeline(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    alpha: float = 0.1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                _build_tfidf_vectorizer(
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            ("classifier", MultinomialNB(alpha=alpha)),
        ]
    )


def build_tfidf_xgboost_pipeline(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for 'tfidf_xgboost'. Install with: pip install xgboost"
        ) from exc

    return Pipeline(
        steps=[
            (
                "tfidf",
                _build_tfidf_vectorizer(
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=-1,
                    random_state=random_state,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_tfidf_lightgbm_pipeline(
    *,
    ngram_min: int = 1,
    ngram_max: int = 2,
    max_features: int = 50_000,
    min_df: int = 2,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for 'tfidf_lightgbm'. Install with: pip install lightgbm"
        ) from exc

    return Pipeline(
        steps=[
            (
                "tfidf",
                _build_tfidf_vectorizer(
                    ngram_min=ngram_min,
                    ngram_max=ngram_max,
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def get_classical_builder(model_name: str):
    builders = {
        "tfidf_logreg": build_tfidf_logreg_pipeline,
        "tfidf_svm": build_tfidf_svm_pipeline,
        "tfidf_nb": build_tfidf_nb_pipeline,
        "tfidf_xgboost": build_tfidf_xgboost_pipeline,
        "tfidf_lightgbm": build_tfidf_lightgbm_pipeline,
    }
    if model_name not in builders:
        valid = ", ".join(sorted(builders))
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")
    return builders[model_name]


def available_classical_models() -> list[str]:
    models = ["tfidf_logreg", "tfidf_svm", "tfidf_nb"]

    try:
        import xgboost  # noqa: F401

        models.append("tfidf_xgboost")
    except Exception:
        pass

    try:
        import lightgbm  # noqa: F401

        models.append("tfidf_lightgbm")
    except Exception:
        pass

    return models


def build_classical_pipeline(model_name: str, **kwargs: Any) -> Pipeline:
    builder = get_classical_builder(model_name)
    return builder(**kwargs)


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
    total_start = time.perf_counter()

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[REVIEW_COL],
        train_df[LABEL_COL],
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[LABEL_COL],
    )
    split_metadata = build_split_metadata(
        train_labels=y_train.tolist(),
        validation_labels=y_val.tolist(),
        test_size=test_size,
        seed=random_state,
        total_samples=len(train_df),
    )

    model = build_classical_pipeline(model_name, random_state=random_state, **model_params)
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    eval_start = time.perf_counter()
    val_preds = model.predict(x_val)
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "model_name": model_name,
        "seed": int(random_state),
        "test_size": float(test_size),
        "num_samples": int(split_metadata["num_samples"]),
        "train_samples": int(split_metadata["train_samples"]),
        "validation_samples": int(split_metadata["validation_samples"]),
        **compute_classification_metrics(y_val, val_preds),
        "split_metadata": split_metadata,
        "timing": {
            "fit_seconds": fit_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - total_start,
        },
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
