from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.data import REVIEW_COL, load_test_dataframe, load_train_dataframe
from sentiment_project.inference import build_submission_dataframe, save_submission_csv
from sentiment_project.modeling import available_classical_models
from sentiment_project.training import predict_texts, save_artifacts, train_and_evaluate_model


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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    train_df = load_train_dataframe(args.train_path)
    test_df = load_test_dataframe(args.test_path)

    model_names = _resolve_model_names(args.model_names)
    print(f"Models to run: {model_names}")

    summary_rows: list[dict] = []
    failures: dict[str, str] = {}

    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        params = _params_for_model(args, model_name)
        try:
            model, metrics = train_and_evaluate_model(
                train_df,
                model_name=model_name,
                test_size=args.test_size,
                random_state=args.random_state,
                model_params=params,
            )
        except Exception as exc:  # noqa: BLE001
            failures[model_name] = str(exc)
            print(f"Failed: {model_name} -> {exc}")
            continue

        model_path = PROJECT_ROOT / "models" / f"{model_name}.joblib"
        metrics_path = PROJECT_ROOT / "reports" / f"{model_name}_metrics.json"
        save_artifacts(model, metrics, model_path=model_path, metrics_path=metrics_path)

        summary_rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
                "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
            }
        )

        print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))

        if args.generate_submissions:
            preds = predict_texts(model, test_df[REVIEW_COL])
            submission_df = build_submission_dataframe(preds, include_id=True)
            submission_path = PROJECT_ROOT / "data" / "submissions" / f"submission_{model_name}.csv"
            save_submission_csv(submission_df, submission_path)
            print(f"Submission: {submission_path.relative_to(PROJECT_ROOT)}")

    leaderboard = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    leaderboard_path = PROJECT_ROOT / "reports" / "classical_model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"\nSaved leaderboard: {leaderboard_path.relative_to(PROJECT_ROOT)}")
    if not leaderboard.empty:
        print(leaderboard.to_string(index=False))

    if failures:
        failed_path = PROJECT_ROOT / "reports" / "classical_model_failures.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"\nSome models failed. Details: {failed_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
