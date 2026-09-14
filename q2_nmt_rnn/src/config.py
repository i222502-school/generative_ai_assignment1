from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED: int = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS: tuple[str, str, str, str] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = range(4)

# A vanilla RNN's vanishing-gradient problem gets worse the longer the sequence;
# capping length is a direct mitigation given the assignment forbids LSTM/GRU/attention.
MAX_SEQUENCE_LENGTH: int = 50
MIN_TOKEN_FREQUENCY: int = 2


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
