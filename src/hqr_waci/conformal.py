"""Adaptive conformal prediction methods."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalPredictions:
    """Lower and upper bounds for the chronological evaluation period."""

    lower: np.ndarray
    upper: np.ndarray
    alpha_history: dict[int, list[np.ndarray | float]] | None = None


def _calibration_scores(
    frame: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    calibration_positions: np.ndarray,
) -> np.ndarray:
    real = frame["real"].to_numpy()
    return np.maximum(
        real[calibration_positions] - upper[calibration_positions],
        lower[calibration_positions] - real[calibration_positions],
    )


def _conformal_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    scores: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    if 0 < alpha < 1:
        n = len(scores)
        quantile = np.clip(np.ceil(n + 1) * (1 - alpha) / n, 0, 1)
        adjustment = np.quantile(scores, quantile)
        return lower - adjustment, upper + adjustment
    if alpha >= 1:
        return lower, upper
    return -np.inf, np.inf


def _hour_positions(
    frame: pd.DataFrame,
    row_positions: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    hours = frame["hour"].to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    grouped = {}
    for hour in pd.unique(hours):
        positions = row_positions[hours[row_positions] == hour]
        grouped[int(hour)] = (positions, dates[positions])
    return grouped


def _history_values(
    history: dict[int, list[np.ndarray | float]] | None,
    hour: int,
    alpha: np.ndarray | float,
) -> None:
    if history is not None and hour in history:
        value = alpha.copy() if isinstance(alpha, np.ndarray) else float(alpha)
        history[hour].append(value)


def adaptive_conformal_inference(
    frame: pd.DataFrame,
    base_model: str,
    alpha: float,
    test_start: str = "2024-01-01",
    gamma: float = 0.02,
    max_calibration: int = 500,
    history_hour: int | None = None,
) -> IntervalPredictions:
    """Apply the original per-hour ACI update with rolling calibration scores."""
    lower = frame[f"preds_inf_{base_model}"].to_numpy()
    upper = frame[f"preds_sup_{base_model}"].to_numpy()
    lengths = upper - lower
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    start_date = np.datetime64(pd.Timestamp(test_start), "ns")
    start_position = int(np.searchsorted(dates, start_date, side="left"))
    test_positions = np.arange(start_position, len(frame))
    groups = _hour_positions(frame, np.arange(len(frame)))
    target_lower = np.empty(len(test_positions))
    target_upper = np.empty(len(test_positions))
    alphas = {hour: float(alpha) for hour in groups}
    history = (
        {hour: [float(alpha)] for hour in groups if hour == history_hour}
        if history_hour is not None
        else None
    )

    for output_position, position in enumerate(test_positions):
        hour = int(frame["hour"].iat[position])
        group_positions, group_dates = groups[hour]
        cutoff = np.searchsorted(group_dates, dates[position], side="left")
        calibration = group_positions[:cutoff]
        calibration = calibration[
            (calibration < position) & ~np.isnan(lengths[calibration])
        ][-max_calibration:]
        scores = _calibration_scores(frame, lower, upper, calibration)

        lo, hi = _conformal_bounds(
            lower[position],
            upper[position],
            scores,
            alphas[hour],
        )
        target_lower[output_position] = lo
        target_upper[output_position] = hi

        covered = frame["real"].iat[position] >= lo and frame["real"].iat[position] <= hi
        alphas[hour] += gamma * (alpha - (1 - float(covered)))
        _history_values(history, hour, alphas[hour])

    target_lower, target_upper = _replace_unbounded_bounds(
        start_position,
        target_lower,
        target_upper,
        lower,
        upper,
    )
    return IntervalPredictions(target_lower, target_upper, history)


def width_adaptive_conformal_inference(
    frame: pd.DataFrame,
    base_model: str,
    alpha: float,
    test_start: str = "2024-01-01",
    gamma: float = 0.02,
    sigma: float = 3,
    max_calibration: int = 500,
    history_hour: int | None = None,
    weighting: str = "gaussian",
    exponential_rate: float = 1.02,
) -> IntervalPredictions:
    """Apply WACI while retaining the original 0.1-wide, 0-to-499.9 grid."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if weighting not in {"gaussian", "exponential"}:
        raise ValueError("weighting must be 'gaussian' or 'exponential'")

    lower = frame[f"preds_inf_{base_model}"].to_numpy()
    upper = frame[f"preds_sup_{base_model}"].to_numpy()
    lengths = upper - lower
    width_grid = np.arange(0, 500, 0.1)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    start_date = np.datetime64(pd.Timestamp(test_start), "ns")
    start_position = int(np.searchsorted(dates, start_date, side="left"))
    test_positions = np.arange(start_position, len(frame))
    groups = _hour_positions(frame, np.arange(len(frame)))
    target_lower = np.empty(len(test_positions))
    target_upper = np.empty(len(test_positions))
    alphas = {
        hour: np.full(len(width_grid), alpha, dtype=float)
        for hour in groups
    }
    history = (
        {hour: [alphas[hour].copy()] for hour in groups if hour == history_hour}
        if history_hour is not None
        else None
    )

    for output_position, position in enumerate(test_positions):
        hour = int(frame["hour"].iat[position])
        group_positions, group_dates = groups[hour]
        cutoff = np.searchsorted(group_dates, dates[position], side="left")
        calibration = group_positions[:cutoff]
        calibration = calibration[
            (calibration < position) & ~np.isnan(lengths[calibration])
        ][-max_calibration:]
        scores = _calibration_scores(frame, lower, upper, calibration)

        observed_width = lengths[position]
        distances = np.abs(width_grid - observed_width)
        nearest = int(np.argmin(distances))
        current_alpha = alphas[hour][nearest]
        lo, hi = _conformal_bounds(
            lower[position],
            upper[position],
            scores,
            current_alpha,
        )
        target_lower[output_position] = lo
        target_upper[output_position] = hi

        covered = frame["real"].iat[position] >= lo and frame["real"].iat[position] <= hi
        error = 1 - float(covered)
        if weighting == "gaussian":
            weights = np.exp(-(distances**2) / (2 * sigma**2))
            weights /= weights.max()
        else:
            index_distance = np.abs(np.arange(len(width_grid)) - nearest)
            weights = 1 / (exponential_rate**index_distance)
            weights[(weights < 0) | np.isinf(weights)] = 0

        alphas[hour] = alphas[hour] + gamma * weights * (alpha - error)
        _history_values(history, hour, alphas[hour])

    target_lower, target_upper = _replace_unbounded_bounds(
        start_position,
        target_lower,
        target_upper,
        lower,
        upper,
    )
    return IntervalPredictions(target_lower, target_upper, history)


def _replace_unbounded_bounds(
    start_position: int,
    lower_output: np.ndarray,
    upper_output: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prior_lower = base_lower[:start_position]
    prior_upper = base_upper[:start_position]
    lower_output = lower_output.copy()
    upper_output = upper_output.copy()
    lower_output[lower_output == -np.inf] = pd.Series(prior_lower).min()
    upper_output[upper_output == np.inf] = pd.Series(prior_upper).max()
    return lower_output, upper_output
