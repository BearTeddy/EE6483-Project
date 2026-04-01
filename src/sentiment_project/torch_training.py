from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .data import LABEL_COL, REVIEW_COL
from .nn_models import RNNTextClassifier, TextCNNClassifier
from .torch_data import EncodedTextDataset, TextVocab


def set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_neural_model(model_type: str, config: dict[str, Any]) -> nn.Module:
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


def _run_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * input_ids.size(0)

    return total_loss / max(1, len(dataloader.dataset))


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return y_true, y_pred


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
    set_torch_seed(random_state)
    torch_device = resolve_device(device)

    x_train, x_val, y_train, y_val = train_test_split(
        train_df[REVIEW_COL].tolist(),
        train_df[LABEL_COL].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[LABEL_COL],
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

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, torch_device)
        y_true, y_pred = _evaluate(model, val_loader, torch_device)
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
        }
        history.append(row)

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
        "num_samples": int(len(train_df)),
        "train_samples": int(len(train_dataset)),
        "validation_samples": int(len(val_dataset)),
        "best_validation": best_eval,
        "history": history,
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
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)
