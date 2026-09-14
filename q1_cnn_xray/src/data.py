"""Dataset discovery, cleaning, and a leakage-safe patient-grouped split."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from q1_cnn_xray.src.config import CLASS_NAMES, MANIFEST_DIR, SplitRatios

logger = logging.getLogger(__name__)

# "person1946_bacteria_4874.jpeg" -> patient 1946; falls back to the IM-#### study
# id used by the NORMAL filenames (e.g. "NORMAL2-IM-0381-0001.jpeg").
_PERSON_RE = re.compile(r"person(\d+)_", re.IGNORECASE)
_STUDY_RE = re.compile(r"IM-(\d+)", re.IGNORECASE)


def discover_images(raw_dir: Path) -> pd.DataFrame:
    """Pool every NORMAL/PNEUMONIA image under raw_dir, ignoring Kaggle's own split folders."""
    rows: list[dict[str, str]] = []
    for label in CLASS_NAMES:
        for path in raw_dir.rglob(f"*/{label}/*"):
            if path.is_file():
                rows.append({"path": str(path), "label": label})
    if not rows:
        raise FileNotFoundError(f"no images found under {raw_dir} (expected .../NORMAL|PNEUMONIA/*)")
    return pd.DataFrame(rows)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unreadable files, then exact duplicates by content hash."""
    readable = df[df["path"].map(lambda p: _is_readable(Path(p)))].copy()
    dropped_corrupt = len(df) - len(readable)

    readable["sha256"] = readable["path"].map(lambda p: _sha256(Path(p)))
    deduped = readable.drop_duplicates(subset="sha256", keep="first").drop(columns="sha256")
    dropped_dupes = len(readable) - len(deduped)

    logger.info("dropped %d unreadable, %d duplicate images (%d remain)", dropped_corrupt, dropped_dupes, len(deduped))
    return deduped.reset_index(drop=True)


def patient_id(path: str) -> str:
    """Best-effort grouping key so one patient's images never span multiple splits."""
    name = Path(path).name
    if match := _PERSON_RE.search(name):
        return f"pneumonia-person-{match.group(1)}"
    if match := _STUDY_RE.search(name):
        return f"normal-study-{match.group(1)}"
    return f"singleton-{name}"


def add_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(patient_id=df["path"].map(patient_id))


def split(df: pd.DataFrame, ratios: SplitRatios, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified-by-class, grouped-by-patient 80/10/10 split via two StratifiedGroupKFold passes."""
    labels = df["label"].to_numpy()
    groups = df["patient_id"].to_numpy()

    train_folds = round(1 / (1 - ratios.train))
    train_idx, holdout_idx = next(
        StratifiedGroupKFold(n_splits=train_folds, shuffle=True, random_state=seed).split(df, labels, groups)
    )
    holdout = df.iloc[holdout_idx]

    val_share = ratios.val / (ratios.val + ratios.test)
    val_folds = round(1 / val_share)
    val_idx, test_idx = next(
        StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed).split(
            holdout, holdout["label"], holdout["patient_id"]
        )
    )

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": holdout.iloc[val_idx].reset_index(drop=True),
        "test": holdout.iloc[test_idx].reset_index(drop=True),
    }


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    patient_sets = {name: set(part["patient_id"]) for name, part in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[a] & patient_sets[b]
        if overlap:
            raise AssertionError(f"patient leakage between {a} and {b}: {overlap}")


def class_counts(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    counts = {name: part["label"].value_counts() for name, part in splits.items()}
    return pd.DataFrame(counts).fillna(0).astype(int).T


def build_manifests(
    raw_dir: Path, manifest_dir: Path = MANIFEST_DIR, ratios: SplitRatios = SplitRatios(), seed: int = 42
) -> dict[str, pd.DataFrame]:
    df = add_patient_ids(clean(discover_images(raw_dir)))
    splits = split(df, ratios, seed)
    assert_no_leakage(splits)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(manifest_dir / f"{name}.csv", index=False)

    logger.info("class counts:\n%s", class_counts(splits))
    return splits


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_manifests(args.raw_dir, seed=args.seed)
