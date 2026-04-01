from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


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
        token_ids, length = self.items[index]
        sample = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.labels is not None:
            sample["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return sample
