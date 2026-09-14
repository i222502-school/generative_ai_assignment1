"""Task 3: word-level tokenization, shared between English and Urdu.

\\w matches Unicode word characters (including the Urdu/Arabic script) under Python's
default Unicode regex mode, so one pattern serves both languages; punctuation is kept
as its own token rather than stripped, since it's part of what an NMT model should learn.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)
