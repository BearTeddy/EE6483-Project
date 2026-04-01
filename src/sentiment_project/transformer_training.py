from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .data import LABEL_COL, REVIEW_COL

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError:  # pragma: no cover - dependency is optional
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    set_seed = None


class TransformerTextDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: list[int] | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def _ensure_transformers_available() -> None:
    if AutoModelForSequenceClassification is None:
        raise ImportError(
            "transformers and torch are required for transformer training. "
            "Install with: pip install -r requirements-bert.txt"
        )


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def train_transformer_classifier(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    max_length: int = 256,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_train_epochs: int = 3,
    warmup_ratio: float = 0.1,
) -> tuple[Any, Any, dict[str, Any]]:
    _ensure_transformers_available()
    assert set_seed is not None

    set_seed(random_state)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[REVIEW_COL].tolist(),
        train_df[LABEL_COL].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[LABEL_COL],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=max_length)
    train_dataset = TransformerTextDataset(train_encodings, labels=y_train)
    val_dataset = TransformerTextDataset(val_encodings, labels=y_val)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        seed=random_state,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    val_pred = trainer.predict(val_dataset)
    val_preds = np.argmax(val_pred.predictions, axis=1)

    metrics = {
        "model_name": model_name,
        "num_samples": int(len(train_df)),
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)),
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "macro_f1": float(f1_score(y_val, val_preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_val, val_preds).tolist(),
        "classification_report": classification_report(y_val, val_preds, output_dict=True),
        "trainer_eval": {k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_result.items()},
        "settings": {
            "test_size": test_size,
            "random_state": random_state,
            "max_length": max_length,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "warmup_ratio": warmup_ratio,
        },
    }

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model, tokenizer, metrics


def predict_transformer_texts(
    model_dir: str | Path,
    texts: list[str] | pd.Series,
    *,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    _ensure_transformers_available()

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
    dataset = TransformerTextDataset(encodings, labels=None)

    args = TrainingArguments(
        output_dir=str(Path(model_dir) / "_predict_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args)
    pred_output = trainer.predict(dataset)
    return np.argmax(pred_output.predictions, axis=1)
