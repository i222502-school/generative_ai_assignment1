"""Task 3: per-language vocabularies built from the training split only."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from q2_nmt_rnn.src.config import MIN_TOKEN_FREQUENCY, SPECIAL_TOKENS, UNK_IDX


@dataclass(slots=True)
class Vocabulary:
    token_to_index: dict[str, int]
    index_to_token: list[str]

    @classmethod
    def build(cls, tokenized_sentences: list[list[str]], min_freq: int = MIN_TOKEN_FREQUENCY) -> Vocabulary:
        counts = Counter(token for sentence in tokenized_sentences for token in sentence)
        tokens = list(SPECIAL_TOKENS) + sorted(tok for tok, freq in counts.items() if freq >= min_freq)
        return cls(token_to_index={tok: i for i, tok in enumerate(tokens)}, index_to_token=tokens)

    def __len__(self) -> int:
        return len(self.index_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_index.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.index_to_token[i] for i in indices]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.index_to_token, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        tokens = json.loads(path.read_text())
        return cls(token_to_index={tok: i for i, tok in enumerate(tokens)}, index_to_token=tokens)
