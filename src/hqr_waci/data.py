"""Loading and preparing the published electricity-price analysis data."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "full_date",
    "date",
    "hour",
    "real",
    "pred1",
    "pred2",
    "pred3",
)


def load_analysis_data(path: Path) -> pd.DataFrame:
    """Load the supplied series and add the forecaster ensemble features."""
    frame = pd.read_csv(path, parse_dates=["full_date"])
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["mean_pred"] = frame[["pred1", "pred2", "pred3"]].mean(axis=1)
    frame["std_pred"] = frame[["pred1", "pred2", "pred3"]].std(axis=1, ddof=0)
    return frame


def select_test_period(
    frame: pd.DataFrame,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Return the chronological evaluation period beginning on ``start_date``."""
    return frame.loc[frame["date"] >= pd.Timestamp(start_date)].copy()
