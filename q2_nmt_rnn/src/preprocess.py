"""Task 1: punctuation/whitespace normalization, Urdu Unicode normalization, cleaning."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

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
    return deduped.reset_index(drop=True)
