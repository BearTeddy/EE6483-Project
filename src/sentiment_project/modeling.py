from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


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
