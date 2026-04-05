from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.core import (
    REVIEW_COL,
    build_submission_dataframe,
    load_model,
    load_test_dataframe,
    predict_texts,
    save_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv from a trained model.")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "tfidf_logreg.joblib")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument(
        "--output-path", type=Path, default=PROJECT_ROOT / "data" / "submissions" / "submission.csv"
    )
    parser.add_argument("--without-id", action="store_true", help="Output only the sentiments column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)
    test_df = load_test_dataframe(args.test_path)

    preds = predict_texts(model, test_df[REVIEW_COL])
    submission_df = build_submission_dataframe(preds, include_id=not args.without_id)
    save_submission_csv(submission_df, args.output_path)

    print(f"Submission saved to: {args.output_path}")
    print(f"Rows: {len(submission_df)}")
    print(f"Columns: {list(submission_df.columns)}")


if __name__ == "__main__":
    main()
