"""Appends markdown-formatted result sections to a running notes file, so the Colab
run produces one document summarizing everything needed to write the report.
"""

from __future__ import annotations

from pathlib import Path


def append_section(notes_path: Path, title: str, body: str) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with notes_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n{body}\n")
