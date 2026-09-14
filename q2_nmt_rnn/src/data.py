"""Dataset loading and a reproducible 80/10/10 split (Tasks 1-2)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from q2_nmt_rnn.src.config import MANIFEST_DIR, SplitRatios
from q2_nmt_rnn.src.preprocess import clean_pairs

logger = logging.getLogger(__name__)

_COLUMN_ALIASES = {"en": ("en", "english", "eng", "source"), "ur": ("ur", "urdu", "target")}
_TABULAR_READERS: tuple[tuple[str, Callable[..., pd.DataFrame]], ...] = (
    ("*.csv", pd.read_csv),
    ("*.xlsx", pd.read_excel),
    ("*.xls", pd.read_excel),
)


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"none of {aliases} found among columns {columns}")


def load_pairs(raw_dir: Path) -> pd.DataFrame:
    """Handles the shapes a Kaggle parallel corpus like this typically ships as: a single
    CSV/XLSX with English/Urdu columns, or two aligned .txt files (one sentence per line).
    """
    for pattern, reader in _TABULAR_READERS:
        candidates = list(raw_dir.rglob(pattern))
        if candidates:
            df = reader(candidates[0])
            en_col = _find_column(list(df.columns), _COLUMN_ALIASES["en"])
            ur_col = _find_column(list(df.columns), _COLUMN_ALIASES["ur"])
            return df.rename(columns={en_col: "en", ur_col: "ur"})[["en", "ur"]]

    txt_files = sorted(raw_dir.rglob("*.txt"))
    en_file = next((f for f in txt_files if "en" in f.stem.lower()), None)
    ur_file = next((f for f in txt_files if "ur" in f.stem.lower()), None)
    if en_file and ur_file:
        en_lines = en_file.read_text(encoding="utf-8").splitlines()
        ur_lines = ur_file.read_text(encoding="utf-8").splitlines()
        if len(en_lines) != len(ur_lines):
            raise ValueError(f"misaligned parallel files: {len(en_lines)} vs {len(ur_lines)} lines")
        return pd.DataFrame({"en": en_lines, "ur": ur_lines})

    raise FileNotFoundError(
        f"no recognizable English-Urdu corpus under {raw_dir}; inspect the download and adjust load_pairs"
    )


def split(df: pd.DataFrame, ratios: SplitRatios, seed: int) -> dict[str, pd.DataFrame]:
    train, holdout = train_test_split(df, train_size=ratios.train, random_state=seed, shuffle=True)
    val_share = ratios.val / (ratios.val + ratios.test)
    val, test = train_test_split(holdout, train_size=val_share, random_state=seed, shuffle=True)
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def assert_no_overlap(splits: dict[str, pd.DataFrame]) -> None:
    keyed = {name: set(zip(part["en"], part["ur"])) for name, part in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = keyed[a] & keyed[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} overlapping pairs between {a} and {b}")


def build_manifests(
    raw_dir: Path, manifest_dir: Path = MANIFEST_DIR, ratios: SplitRatios = SplitRatios(), seed: int = 42
) -> dict[str, pd.DataFrame]:
    df = clean_pairs(load_pairs(raw_dir))
    splits = split(df, ratios, seed)
    assert_no_overlap(splits)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(manifest_dir / f"{name}.csv", index=False)

    logger.info("split sizes: %s", {name: len(part) for name, part in splits.items()})
    return splits


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifests(args.raw_dir, seed=args.seed)
