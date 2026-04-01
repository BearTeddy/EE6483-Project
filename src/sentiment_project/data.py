from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REVIEW_COL = "reviews"
LABEL_COL = "sentiments"


def _load_json_records(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected list of records in {file_path}, got: {type(records)!r}")

    return records


def _check_columns(df: pd.DataFrame, required_columns: Iterable[str], path: str | Path) -> None:
    missing = set(required_columns).difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing expected column(s) in {path}: {missing_str}")


def load_train_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(_load_json_records(path))
    _check_columns(df, [REVIEW_COL, LABEL_COL], path)

    df = df[[REVIEW_COL, LABEL_COL]].dropna(subset=[REVIEW_COL, LABEL_COL]).copy()
    df[REVIEW_COL] = df[REVIEW_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    invalid_mask = ~df[LABEL_COL].isin([0, 1])
    if invalid_mask.any():
        invalid_values = sorted(df.loc[invalid_mask, LABEL_COL].unique().tolist())
        raise ValueError(f"Unexpected label values in {path}: {invalid_values}. Expected only 0 or 1.")

    return df.reset_index(drop=True)


def load_test_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(_load_json_records(path))
    _check_columns(df, [REVIEW_COL], path)
    df = df[[REVIEW_COL]].dropna(subset=[REVIEW_COL]).copy()
    df[REVIEW_COL] = df[REVIEW_COL].astype(str)
    return df.reset_index(drop=True)
