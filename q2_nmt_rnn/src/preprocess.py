"""Task 1: punctuation/whitespace normalization, Urdu Unicode normalization, cleaning."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from q2_nmt_rnn.src.config import MAX_SEQUENCE_LENGTH
from q2_nmt_rnn.src.tokenizer import tokenize

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalizes Unicode (critical for Urdu, where the same glyph can have multiple
    codepoint sequences) and collapses whitespace. Applied to both languages for consistency.
    """
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_pairs(df: pd.DataFrame, en_col: str = "en", ur_col: str = "ur") -> pd.DataFrame:
    df = df[[en_col, ur_col]].rename(columns={en_col: "en", ur_col: "ur"}).copy()
    # fillna before astype(str): pandas' string dtype leaves missing cells as NaN through
    # astype(str) rather than stringifying them, which would otherwise reach normalize_text.
    df["en"] = df["en"].fillna("").astype(str).map(normalize_text)
    df["ur"] = df["ur"].fillna("").astype(str).map(normalize_text)

    non_empty = (df["en"].str.len() > 1) & (df["ur"].str.len() > 1)
    deduped = df[non_empty].drop_duplicates(subset=["en", "ur"])
    return filter_by_length(deduped.reset_index(drop=True))


def filter_by_length(df: pd.DataFrame, max_length: int = MAX_SEQUENCE_LENGTH) -> pd.DataFrame:
    """Drops pairs where either side exceeds max_length tokens -- caps how much a vanilla
    RNN's single fixed-size hidden state has to carry (see models/seq2seq.py's docstring).
    """
    short_enough = (df["en"].map(lambda t: len(tokenize(t))) <= max_length) & (
        df["ur"].map(lambda t: len(tokenize(t))) <= max_length
    )
    return df[short_enough].reset_index(drop=True)
