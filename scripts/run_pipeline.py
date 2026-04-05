from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.core import (
    REVIEW_COL,
    build_submission_dataframe,
    load_test_dataframe,
    load_train_dataframe,
    predict_texts,
    save_artifacts,
    save_submission_csv,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model and generate submission in one run.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "tfidf_logreg.joblib")
    parser.add_argument("--metrics-path", type=Path, default=PROJECT_ROOT / "reports" / "train_metrics.json")
    parser.add_argument(
        "--submission-path", type=Path, default=PROJECT_ROOT / "data" / "submissions" / "submission.csv"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = load_train_dataframe(args.train_path)
    model, metrics = train_and_evaluate(
        train_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_artifacts(model, metrics, model_path=args.model_path, metrics_path=args.metrics_path)

    test_df = load_test_dataframe(args.test_path)
    preds = predict_texts(model, test_df[REVIEW_COL])
    submission_df = build_submission_dataframe(preds, include_id=True)
    save_submission_csv(submission_df, args.submission_path)

    print("Pipeline complete.")
    print(f"Model: {args.model_path}")
    print(f"Metrics: {args.metrics_path}")
    print(f"Submission: {args.submission_path}")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
