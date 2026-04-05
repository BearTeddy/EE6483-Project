from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.core import (
    REVIEW_COL,
    available_classical_models,
    build_experiment_paths,
    build_run_label,
    build_submission_dataframe,
    format_test_size_label,
    load_test_dataframe,
    load_train_dataframe,
    predict_texts,
    save_artifacts,
    save_submission_csv,
    train_and_evaluate_model,
)

DEFAULT_TEST_SIZES = [0.2, 0.3, 0.4]
DEFAULT_SEEDS = [42, 52, 62]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare classical sentiment models.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["all"],
        help="Model list, e.g. tfidf_logreg tfidf_svm tfidf_nb tfidf_xgboost tfidf_lightgbm or 'all'.",
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--test-sizes", nargs="+", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=4.0)
    parser.add_argument("--nb-alpha", type=float, default=0.1)
    parser.add_argument("--generate-submissions", action="store_true")
    return parser.parse_args()


def _resolve_model_names(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return available_classical_models()
    return raw


def _resolve_test_sizes(args: argparse.Namespace) -> list[float]:
    if args.test_sizes:
        return args.test_sizes
    if args.test_size is not None:
        return [args.test_size]
    return DEFAULT_TEST_SIZES


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return args.seeds
    if args.random_state is not None:
        return [args.random_state]
    return DEFAULT_SEEDS


def _params_for_model(args: argparse.Namespace, model_name: str) -> dict:
    base = {
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "max_features": args.max_features,
        "min_df": args.min_df,
    }
    if model_name in {"tfidf_logreg", "tfidf_svm"}:
        base["regularization_c"] = args.regularization_c
    if model_name == "tfidf_nb":
        base["alpha"] = args.nb_alpha
    return base


def _distribution_count(metrics: dict, section: str, label: int) -> int:
    split_metadata = metrics.get("split_metadata", {})
    distribution = split_metadata.get(f"{section}_label_distribution", {})
    counts = distribution.get("counts", {})
    return int(counts.get(str(label), 0))


def _build_run_row(
    metrics: dict,
    *,
    model_name: str,
    seed: int,
    test_size: float,
    model_path: Path,
    metrics_path: Path,
    submission_path: Path | None,
) -> dict[str, object]:
    timing = metrics.get("timing", {})
    return {
        "model_name": model_name,
        "test_size": float(test_size),
        "test_size_label": format_test_size_label(test_size),
        "seed": int(seed),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "train_samples": int(metrics["train_samples"]),
        "validation_samples": int(metrics["validation_samples"]),
        "train_label_0": _distribution_count(metrics, "train", 0),
        "train_label_1": _distribution_count(metrics, "train", 1),
        "validation_label_0": _distribution_count(metrics, "validation", 0),
        "validation_label_1": _distribution_count(metrics, "validation", 1),
        "fit_seconds": float(timing.get("fit_seconds", 0.0)),
        "eval_seconds": float(timing.get("eval_seconds", 0.0)),
        "total_seconds": float(timing.get("total_seconds", 0.0)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "submission_path": str(submission_path.relative_to(PROJECT_ROOT)) if submission_path else "",
    }


def _build_leaderboard(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not summary_rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "test_size",
                "test_size_label",
                "run_count",
                "seeds",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "validation_samples_mean",
                "fit_seconds_mean",
                "total_seconds_mean",
            ]
        )

    runs_df = pd.DataFrame(summary_rows)
    grouped = runs_df.groupby(["model_name", "test_size", "test_size_label"], dropna=False)
    leaderboard = (
        grouped.agg(
            run_count=("seed", "count"),
            seeds=("seed", lambda values: ",".join(str(v) for v in sorted(set(values)))),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            validation_samples_mean=("validation_samples", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
        )
        .reset_index()
        .sort_values(["macro_f1_mean", "accuracy_mean"], ascending=[False, False])
    )

    for column in ["accuracy_std", "macro_f1_std"]:
        leaderboard[column] = leaderboard[column].fillna(0.0)
    return leaderboard


def main() -> None:
    args = parse_args()
    train_df = load_train_dataframe(args.train_path)
    test_df = load_test_dataframe(args.test_path) if args.generate_submissions else None

    model_names = _resolve_model_names(args.model_names)
    test_sizes = _resolve_test_sizes(args)
    seeds = _resolve_seeds(args)
    run_specs = list(product(model_names, test_sizes, seeds))

    print(f"Models to run: {model_names}")
    print(f"Test sizes: {test_sizes}")
    print(f"Seeds: {seeds}")

    summary_rows: list[dict[str, object]] = []
    failures: dict[str, str] = {}

    with tqdm(run_specs, desc="Classical comparison runs", unit="run") as run_bar:
        for model_name, test_size, seed in run_bar:
            run_label = build_run_label(model_name, test_size, seed)
            run_bar.set_postfix(model=model_name, test=format_test_size_label(test_size), seed=seed)
            params = _params_for_model(args, model_name)
            experiment_paths = build_experiment_paths(PROJECT_ROOT, model_name, test_size=test_size, seed=seed)
            model_path = experiment_paths["model_dir"] / "model.joblib"
            metrics_path = experiment_paths["reports_dir"] / "metrics.json"
            submission_path: Path | None = None

            try:
                model, metrics = train_and_evaluate_model(
                    train_df,
                    model_name=model_name,
                    test_size=test_size,
                    random_state=seed,
                    model_params=params,
                )
            except Exception as exc:  # noqa: BLE001
                failures[run_label] = str(exc)
                tqdm.write(f"Failed: {run_label} -> {exc}")
                continue

            save_artifacts(model, metrics, model_path=model_path, metrics_path=metrics_path)

            if args.generate_submissions:
                assert test_df is not None
                preds = predict_texts(model, test_df[REVIEW_COL])
                submission_df = build_submission_dataframe(preds, include_id=True)
                submission_path = experiment_paths["submission_dir"] / "submission.csv"
                save_submission_csv(submission_df, submission_path)

            run_row = _build_run_row(
                metrics,
                model_name=model_name,
                seed=seed,
                test_size=test_size,
                model_path=model_path,
                metrics_path=metrics_path,
                submission_path=submission_path,
            )
            summary_rows.append(run_row)
            tqdm.write(
                json.dumps(
                    {
                        "run": run_label,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
                    },
                    indent=2,
                )
            )

    runs_df = pd.DataFrame(summary_rows)
    runs_path = PROJECT_ROOT / "reports" / "classical_model_runs.csv"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    if not runs_df.empty:
        runs_df = runs_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    runs_df.to_csv(runs_path, index=False)

    leaderboard = _build_leaderboard(summary_rows)
    leaderboard_path = PROJECT_ROOT / "reports" / "classical_model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"\nSaved run ledger: {runs_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved leaderboard: {leaderboard_path.relative_to(PROJECT_ROOT)}")
    if not leaderboard.empty:
        print(leaderboard.to_string(index=False))

    if failures:
        failed_path = PROJECT_ROOT / "reports" / "classical_model_failures.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"\nSome runs failed. Details: {failed_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
