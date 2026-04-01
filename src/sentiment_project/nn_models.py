from __future__ import annotations

import torch
from torch import nn


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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]
        embedded = embedded.transpose(1, 2)  # [batch, emb_dim, seq_len]

        pooled = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            max_pooled = torch.max(conv_out, dim=2).values
            pooled.append(max_pooled)

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

        if self.rnn_type == "lstm":
            hidden_state = hidden[0]
        else:
            hidden_state = hidden

        if self.bidirectional:
            features = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            features = hidden_state[-1]

        return self.fc(self.dropout(features))
