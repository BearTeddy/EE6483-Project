from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib

from sentiment_project.core import (
    LABEL_COL,
    REVIEW_COL,
    build_classical_pipeline,
    build_submission_dataframe,
    load_test_dataframe,
    load_train_dataframe,
    save_submission_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classical model on all labeled data and generate submission.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument("--model-name", choices=["tfidf_logreg", "tfidf_svm", "tfidf_nb"], default="tfidf_logreg")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "final_tfidf_logreg_full.joblib")
    parser.add_argument("--submission-path", type=Path, default=PROJECT_ROOT / "data" / "submissions" / "submission.csv")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "final_submission_metadata.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=4.0)
    parser.add_argument("--nb-alpha", type=float, default=0.1)
    return parser.parse_args()


def build_model_params(args: argparse.Namespace) -> dict:
    params = {
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "max_features": args.max_features,
        "min_df": args.min_df,
    }
    if args.model_name in {"tfidf_logreg", "tfidf_svm"}:
        params["regularization_c"] = args.regularization_c
    if args.model_name == "tfidf_nb":
        params["alpha"] = args.nb_alpha
    return params


def main() -> None:
    args = parse_args()
    train_df = load_train_dataframe(args.train_path)
    test_df = load_test_dataframe(args.test_path)
    model_params = build_model_params(args)

    model = build_classical_pipeline(args.model_name, random_state=args.random_state, **model_params)
    model.fit(train_df[REVIEW_COL], train_df[LABEL_COL])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)

    preds = model.predict(test_df[REVIEW_COL])
    submission_df = build_submission_dataframe(preds, include_id=True)
    save_submission_csv(submission_df, args.submission_path)

    metadata = {
        "model_name": args.model_name,
        "random_state": args.random_state,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_label_counts": {
            "0": int((train_df[LABEL_COL] == 0).sum()),
            "1": int((train_df[LABEL_COL] == 1).sum()),
        },
        "model_params": model_params,
        "model_path": str(args.model_path.relative_to(PROJECT_ROOT)),
        "submission_path": str(args.submission_path.relative_to(PROJECT_ROOT)),
        "predicted_label_counts": {
            "0": int((submission_df[LABEL_COL] == 0).sum()),
            "1": int((submission_df[LABEL_COL] == 1).sum()),
        },
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Full-data classical training complete.")
    print(f"Model: {args.model_path}")
    print(f"Submission: {args.submission_path}")
    print(f"Metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
