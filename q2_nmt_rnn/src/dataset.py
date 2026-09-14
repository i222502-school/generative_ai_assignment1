"""Task 5: integer encoding, padding, and masks for the encoder/decoder batching pipeline."""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from q2_nmt_rnn.src.config import BOS_IDX, EOS_IDX, PAD_IDX
from q2_nmt_rnn.src.tokenizer import tokenize
from q2_nmt_rnn.src.vocab import Vocabulary


class TranslationDataset(Dataset):
    """Decoder input is <bos>-prefixed target tokens; decoder target is the same
    sequence shifted left with a trailing <eos> -- the standard teacher-forcing setup.
    """

    def __init__(self, manifest: pd.DataFrame, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> None:
        self.src_ids = [torch.tensor(src_vocab.encode(tokenize(text)), dtype=torch.long) for text in manifest["en"]]

        self.tgt_in: list[torch.Tensor] = []
        self.tgt_out: list[torch.Tensor] = []
        for text in manifest["ur"]:
            ids = tgt_vocab.encode(tokenize(text))
            self.tgt_in.append(torch.tensor([BOS_IDX, *ids], dtype=torch.long))
            self.tgt_out.append(torch.tensor([*ids, EOS_IDX], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.src_ids[index], self.tgt_in[index], self.tgt_out[index]


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    src, tgt_in, tgt_out = zip(*batch)
    src_padded = pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt_in_padded = pad_sequence(tgt_in, batch_first=True, padding_value=PAD_IDX)
    tgt_out_padded = pad_sequence(tgt_out, batch_first=True, padding_value=PAD_IDX)
    return {
        "src": src_padded,
        "src_mask": src_padded != PAD_IDX,
        "tgt_in": tgt_in_padded,
        "tgt_out": tgt_out_padded,
        "tgt_mask": tgt_in_padded != PAD_IDX,
    }
