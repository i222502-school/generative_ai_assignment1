"""Task 4: vanilla RNN encoder-decoder -- nn.RNN only, no LSTM/GRU/attention/Transformer.

The encoder's final hidden state is the sole bridge to the decoder (no attention
mechanism is permitted), which is the classic Sutskever et al. bottleneck: translation
quality degrades on long source sentences since one fixed-size vector must carry the
whole sentence. MAX_SEQUENCE_LENGTH (config.py) exists specifically to keep that
bottleneck manageable given the architectural constraint.
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(src))
        _, hidden = self.rnn(embedded)
        return hidden  # (1, batch, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_in: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tgt_in))
        output, _ = self.rnn(embedded, hidden)
        return self.output_proj(output)  # (batch, seq_len, vocab_size)


class Seq2Seq(nn.Module):
    def __init__(
        self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, dropout)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(src)
        return self.decoder(tgt_in, hidden)
