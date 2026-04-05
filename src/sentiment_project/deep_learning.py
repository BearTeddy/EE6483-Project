from __future__ import annotations

import json
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .core import LABEL_COL, REVIEW_COL, build_run_label, build_split_metadata

TORCH_IMPORT_ERROR: ImportError | None = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency is optional
    torch = None  # type: ignore[assignment]
    nn = SimpleNamespace(Module=object)  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _ensure_torch_available() -> None:
    if torch is None:
        raise ImportError(
            "torch is required for neural and transformer models. "
            "Install the project dependencies first."
        ) from TORCH_IMPORT_ERROR


def basic_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


@dataclass
class TextVocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_idx: int
    unk_idx: int

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        min_freq: int = 2,
        max_vocab_size: int = 30_000,
    ) -> "TextVocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))

        vocab_tokens = [token for token, freq in counter.most_common() if freq >= min_freq]
        vocab_tokens = vocab_tokens[: max(0, max_vocab_size - 2)]
        itos = [PAD_TOKEN, UNK_TOKEN] + vocab_tokens
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, pad_idx=stoi[PAD_TOKEN], unk_idx=stoi[UNK_TOKEN])

    def encode(self, text: str, *, max_length: int) -> tuple[list[int], int]:
        tokens = basic_tokenize(text)
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        ids = ids[:max_length]
        length = max(1, len(ids))

        if len(ids) < max_length:
            ids = ids + [self.pad_idx] * (max_length - len(ids))

        return ids, length

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TextVocab":
        return cls(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(t) for t in payload["itos"]],
            pad_idx=int(payload["pad_idx"]),
            unk_idx=int(payload["unk_idx"]),
        )


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        *,
        vocab: TextVocab,
        max_length: int,
        labels: list[int] | None = None,
    ) -> None:
        self.items = [vocab.encode(text, max_length=max_length) for text in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        _ensure_torch_available()
        token_ids, length = self.items[index]
        sample = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.labels is not None:
            sample["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return sample


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 128,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        _ensure_torch_available()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled.append(torch.max(conv_out, dim=2).values)
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features))


class RNNTextClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        pad_idx: int = 0,
    ) -> None:
        _ensure_torch_available()
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        if lengths is None:
            lengths = torch.full((input_ids.size(0),), input_ids.size(1), device=input_ids.device, dtype=torch.long)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)

        hidden_state = hidden[0] if self.rnn_type == "lstm" else hidden
        if self.bidirectional:
            features = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            features = hidden_state[-1]

        return self.fc(self.dropout(features))


def set_torch_seed(seed: int = 42) -> None:
    _ensure_torch_available()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    _ensure_torch_available()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_neural_model(model_type: str, config: dict[str, Any]) -> nn.Module:
    _ensure_torch_available()
    model_type = model_type.lower()
    if model_type == "textcnn":
        return TextCNNClassifier(
            vocab_size=config["vocab_size"],
            embedding_dim=config.get("embedding_dim", 128),
            num_classes=2,
            kernel_sizes=tuple(config.get("kernel_sizes", [3, 4, 5])),
            num_filters=config.get("num_filters", 128),
            dropout=config.get("dropout", 0.3),
            pad_idx=config["pad_idx"],
        )
    if model_type in {"bilstm", "bigru"}:
        return RNNTextClassifier(
            vocab_size=config["vocab_size"],
            embedding_dim=config.get("embedding_dim", 128),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.3),
            num_classes=2,
            rnn_type="lstm" if model_type == "bilstm" else "gru",
            bidirectional=True,
            pad_idx=config["pad_idx"],
        )
    raise ValueError("model_type must be one of: textcnn, bilstm, bigru")


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    desc: str,
) -> tuple[float, float]:
    _ensure_torch_available()
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    seen_samples = 0
    epoch_start = time.perf_counter()

    with tqdm(dataloader, desc=desc, unit="batch", leave=False) as batch_bar:
        for batch in batch_bar:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * input_ids.size(0)
            seen_samples += input_ids.size(0)
            batch_bar.set_postfix(loss=f"{(total_loss / max(1, seen_samples)):.4f}")

    return total_loss / max(1, len(dataloader.dataset)), time.perf_counter() - epoch_start


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    _ensure_torch_available()
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    eval_start = time.perf_counter()

    with torch.no_grad():
        with tqdm(dataloader, desc=desc, unit="batch", leave=False) as batch_bar:
            for batch in batch_bar:
                input_ids = batch["input_ids"].to(device)
                lengths = batch["length"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids, lengths)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return y_true, y_pred, time.perf_counter() - eval_start


def train_neural_text_classifier(
    train_df: pd.DataFrame,
    *,
    model_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
    max_length: int = 160,
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 1,
    num_filters: int = 128,
    kernel_sizes: tuple[int, ...] = (3, 4, 5),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 8,
    device: str | None = None,
) -> tuple[nn.Module, TextVocab, dict[str, Any]]:
    _ensure_torch_available()
    total_start = time.perf_counter()
    set_torch_seed(random_state)
    torch_device = resolve_device(device)

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[REVIEW_COL].tolist(),
        train_df[LABEL_COL].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[LABEL_COL],
    )
    split_metadata = build_split_metadata(
        train_labels=y_train,
        validation_labels=y_val,
        test_size=test_size,
        seed=random_state,
        total_samples=len(train_df),
    )

    vocab = TextVocab.build(x_train, min_freq=min_freq, max_vocab_size=max_vocab_size)
    train_dataset = EncodedTextDataset(x_train, vocab=vocab, max_length=max_length, labels=y_train)
    val_dataset = EncodedTextDataset(x_val, vocab=vocab, max_length=max_length, labels=y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_config = {
        "model_type": model_type,
        "vocab_size": len(vocab.itos),
        "pad_idx": vocab.pad_idx,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_filters": num_filters,
        "kernel_sizes": list(kernel_sizes),
        "dropout": dropout,
        "max_length": max_length,
    }
    model = build_neural_model(model_type, model_config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    best_eval: dict[str, Any] = {}
    train_time_total = 0.0
    eval_time_total = 0.0
    run_label = build_run_label(model_type, test_size, random_state)

    with tqdm(range(1, epochs + 1), desc=run_label, unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            train_loss, train_seconds = _run_epoch(
                model,
                train_loader,
                optimizer,
                torch_device,
                desc=f"Epoch {epoch}/{epochs} train",
            )
            y_true, y_pred, eval_seconds = _evaluate(
                model,
                val_loader,
                torch_device,
                desc=f"Epoch {epoch}/{epochs} eval",
            )
            train_time_total += train_seconds
            eval_time_total += eval_seconds
            accuracy = float(accuracy_score(y_true, y_pred))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": accuracy,
                    "val_macro_f1": macro_f1,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                    "epoch_seconds": train_seconds + eval_seconds,
                }
            )
            epoch_bar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{accuracy:.4f}", f1=f"{macro_f1:.4f}")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_state = deepcopy(model.state_dict())
                best_eval = {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                    "classification_report": classification_report(y_true, y_pred, output_dict=True),
                    "epoch": epoch,
                }

    model.load_state_dict(best_state)
    metrics = {
        "model_name": model_type,
        "seed": int(random_state),
        "test_size": float(test_size),
        "num_samples": int(split_metadata["num_samples"]),
        "train_samples": int(split_metadata["train_samples"]),
        "validation_samples": int(split_metadata["validation_samples"]),
        "best_validation": best_eval,
        "history": history,
        "split_metadata": split_metadata,
        "timing": {
            "fit_seconds": train_time_total,
            "eval_seconds": eval_time_total,
            "total_seconds": time.perf_counter() - total_start,
        },
        "settings": {
            "test_size": test_size,
            "random_state": random_state,
            "batch_size": batch_size,
            "max_length": max_length,
            "max_vocab_size": max_vocab_size,
            "min_freq": min_freq,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_filters": num_filters,
            "kernel_sizes": list(kernel_sizes),
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "device": str(torch_device),
        },
        "model_config": model_config,
    }
    return model, vocab, metrics


def save_neural_checkpoint(
    model: nn.Module,
    vocab: TextVocab,
    metrics: dict[str, Any],
    *,
    checkpoint_path: str | Path,
) -> None:
    _ensure_torch_available()
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "vocab": vocab.to_dict(),
        "metrics": metrics,
        "model_config": metrics["model_config"],
    }
    torch.save(payload, output)


def load_neural_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[nn.Module, TextVocab, dict[str, Any]]:
    _ensure_torch_available()
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    model_config = checkpoint["model_config"]
    model = build_neural_model(model_config["model_type"], model_config)
    model.load_state_dict(checkpoint["state_dict"])
    vocab = TextVocab.from_dict(checkpoint["vocab"])
    metrics = checkpoint.get("metrics", {})
    return model, vocab, metrics


def predict_neural_texts(
    model: nn.Module,
    vocab: TextVocab,
    texts: list[str] | pd.Series,
    *,
    max_length: int,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    _ensure_torch_available()
    torch_device = resolve_device(device)
    model = model.to(torch_device)
    model.eval()

    dataset = EncodedTextDataset(list(texts), vocab=vocab, max_length=max_length, labels=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(torch_device)
            lengths = batch["length"].to(torch_device)
            logits = model(input_ids, lengths)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
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
        _ensure_torch_available()
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def _ensure_transformers_available() -> None:
    _ensure_torch_available()
    if AutoModelForSequenceClassification is None:
        raise ImportError(
            "transformers and torch are required for transformer training. "
            "Install with: pip install -r requirements-bert.txt"
        )


def _compute_transformer_metrics(eval_pred) -> dict[str, float]:
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

    total_start = time.perf_counter()
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
    split_metadata = build_split_metadata(
        train_labels=y_train,
        validation_labels=y_val,
        test_size=test_size,
        seed=random_state,
        total_samples=len(train_df),
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
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_transformer_metrics,
    )

    fit_start = time.perf_counter()
    trainer.train()
    fit_seconds = time.perf_counter() - fit_start
    eval_start = time.perf_counter()
    eval_result = trainer.evaluate()
    val_pred = trainer.predict(val_dataset)
    val_preds = np.argmax(val_pred.predictions, axis=1)
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "model_name": model_name,
        "seed": int(random_state),
        "test_size": float(test_size),
        "num_samples": int(split_metadata["num_samples"]),
        "train_samples": int(split_metadata["train_samples"]),
        "validation_samples": int(split_metadata["validation_samples"]),
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "macro_f1": float(f1_score(y_val, val_preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_val, val_preds).tolist(),
        "classification_report": classification_report(y_val, val_preds, output_dict=True),
        "trainer_eval": {k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_result.items()},
        "split_metadata": split_metadata,
        "timing": {
            "fit_seconds": fit_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - total_start,
        },
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
