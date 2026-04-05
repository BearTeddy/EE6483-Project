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
from sentiment_project.deep_learning import (
    predict_neural_texts,
    save_neural_checkpoint,
    train_neural_text_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural sequence model for sentiment classification.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "test.json")
    parser.add_argument("--model-type", choices=["textcnn", "bilstm", "bigru"], required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-vocab-size", type=int, default=30_000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_train_dataframe(args.train_path)
    test_df = load_test_dataframe(args.test_path)

    experiment_paths = build_experiment_paths(
        PROJECT_ROOT,
        args.model_type,
        test_size=args.test_size,
        seed=args.random_state,
    )
    checkpoint_path = args.checkpoint_path or (experiment_paths["model_dir"] / "model.pt")
    metrics_path = args.metrics_path or (experiment_paths["reports_dir"] / "metrics.json")
    submission_path = args.submission_path or (experiment_paths["submission_dir"] / "submission.csv")

    model, vocab, metrics = train_neural_text_classifier(
        train_df,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
    )

    save_neural_checkpoint(model, vocab, metrics, checkpoint_path=checkpoint_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    preds = predict_neural_texts(
        model,
        vocab,
        test_df[REVIEW_COL].tolist(),
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    submission_df = build_submission_dataframe(preds, include_id=True)
    save_submission_csv(submission_df, submission_path)

    print("Training complete.")
    print(f"Model: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Submission: {submission_path}")
    print(json.dumps(metrics["best_validation"], indent=2))


if __name__ == "__main__":
    main()
