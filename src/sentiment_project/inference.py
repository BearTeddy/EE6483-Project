from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_COL


def build_submission_dataframe(predictions: np.ndarray, include_id: bool = True) -> pd.DataFrame:
    predictions = predictions.astype(int)
    if include_id:
        return pd.DataFrame({"id": range(len(predictions)), LABEL_COL: predictions})
    return pd.DataFrame({LABEL_COL: predictions})


def save_submission_csv(submission_df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output, index=False)
