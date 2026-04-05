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
    build_experiment_paths,
    build_submission_dataframe,
    load_test_dataframe,
    load_train_dataframe,
    save_submission_csv,
)
from sentiment_project.deep_learning import predict_transformer_texts, train_transformer_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT/RoBERTa for sentiment classification.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument("--model-name", choices=["bert-base-uncased", "roberta-base"], required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    return parser.parse_args()


def model_alias(model_name: str) -> str:
    if model_name == "bert-base-uncased":
        return "bert_base_uncased"
    if model_name == "roberta-base":
        return "roberta_base"
    return model_name.replace("-", "_")


def main() -> None:
    args = parse_args()
    alias = model_alias(args.model_name)
    experiment_paths = build_experiment_paths(
        PROJECT_ROOT,
        alias,
        test_size=args.test_size,
        seed=args.random_state,
    )

    output_dir = args.output_dir or experiment_paths["model_dir"]
    metrics_path = args.metrics_path or (experiment_paths["reports_dir"] / "metrics.json")
    submission_path = args.submission_path or (experiment_paths["submission_dir"] / "submission.csv")

    train_df = load_train_dataframe(args.train_path)
    test_df = load_test_dataframe(args.test_path)

    _, _, metrics = train_transformer_classifier(
        train_df,
        model_name=args.model_name,
        output_dir=output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds = predict_transformer_texts(
        output_dir,
        test_df[REVIEW_COL].tolist(),
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
    )
    submission_df = build_submission_dataframe(preds, include_id=True)
    save_submission_csv(submission_df, submission_path)

    print("Transformer training complete.")
    print(f"Model dir: {output_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Submission: {submission_path}")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
