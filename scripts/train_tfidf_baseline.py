from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentiment_project.data import load_train_dataframe
from sentiment_project.training import save_artifacts, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "train.json")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "tfidf_logreg.joblib")
    parser.add_argument("--metrics-path", type=Path, default=PROJECT_ROOT / "reports" / "train_metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_train_dataframe(args.train_path)

    model, metrics = train_and_evaluate(
        train_df,
        test_size=args.test_size,
        random_state=args.random_state,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        min_df=args.min_df,
        regularization_c=args.regularization_c,
    )
    save_artifacts(model, metrics, model_path=args.model_path, metrics_path=args.metrics_path)

    print("Training complete.")
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.metrics_path}")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
