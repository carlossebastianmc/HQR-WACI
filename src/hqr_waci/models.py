"""Rolling quantile and distributional conformal prediction models."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

BASE_MODEL_FEATURES = {
    "hqr": ("mean_pred", "std_pred"),
    "hqr_w": ("pred1", "pred2", "pred3", "std_pred"),
    "qra": ("pred1", "pred2", "pred3"),
}
TRAINING_WINDOW_HOURS = 6 * 30 * 24


def rolling_quantile_predictions(
    frame: pd.DataFrame,
    model_name: str,
    alpha: float,
    test_start: str = "2024-01-01",
    show_progress: bool = True,
    training_window_hours: int = TRAINING_WINDOW_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the paper's rolling lower and upper quantile regressions."""
    if model_name not in BASE_MODEL_FEATURES:
        raise ValueError(f"Unknown base model: {model_name}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    features = list(BASE_MODEL_FEATURES[model_name])
    dates = pd.to_datetime(frame["date"])
    test_start_date = pd.Timestamp(test_start)
    test_start_position = int(np.searchsorted(dates, test_start_date, side="left"))
    if training_window_hours < 1:
        raise ValueError("training_window_hours must be positive")
    first_position = max(0, test_start_position - training_window_hours)
    lower = np.full(len(frame), np.nan)
    upper = np.full(len(frame), np.nan)
    target_dates = pd.unique(dates.iloc[first_position:])
    iterator = tqdm(
        target_dates,
        desc=f"{model_name} alpha={alpha}",
        disable=not show_progress,
    )

    for target_date in iterator:
        target_positions = np.flatnonzero(
            (dates.to_numpy() == target_date)
            & (np.arange(len(frame)) >= first_position)
        )
        training = frame.loc[dates < target_date].tail(training_window_hours)
        x_train = sm.add_constant(training[features])
        y_train = training["real"]
        x_test = frame.iloc[target_positions][features].copy()
        x_test.insert(0, "const", 1.0)

        lower_fit = sm.QuantReg(y_train, x_train).fit(q=alpha / 2)
        upper_fit = sm.QuantReg(y_train, x_train).fit(q=1 - alpha / 2)
        lower[target_positions] = lower_fit.predict(x_test).to_numpy()
        upper[target_positions] = upper_fit.predict(x_test).to_numpy()

    return lower, upper


def _fit_quantile_grid(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    quantiles: np.ndarray,
) -> np.ndarray:
    coefficients = np.empty((x_train.shape[1] + 1, len(quantiles)))
    design = sm.add_constant(x_train)
    model = sm.QuantReg(y_train, design)
    for index, quantile in enumerate(quantiles):
        coefficients[:, index] = model.fit(q=float(quantile)).params
    return coefficients


def _distributional_interval(
    y_calibration: np.ndarray,
    x_calibration: np.ndarray,
    y_test: np.ndarray,
    x_test: np.ndarray,
    coefficients: np.ndarray,
    quantiles: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_all = np.vstack((x_calibration, x_test))
    y_all = np.concatenate((y_calibration, y_test))
    quantile_predictions = np.sort(
        sm.add_constant(x_all) @ coefficients,
        axis=1,
    )
    ranks = np.mean(quantile_predictions <= y_all[:, None], axis=1)
    lower_quantiles = quantiles[quantiles <= alpha]
    upper_quantiles = lower_quantiles + 1 - alpha
    upper_indices = np.searchsorted(quantiles, upper_quantiles)
    upper_indices = np.clip(upper_indices, 1, len(quantiles) - 1)
    lower_indices = upper_indices - 1
    upper_weights = (
        (upper_quantiles - quantiles[lower_indices])
        / (quantiles[upper_indices] - quantiles[lower_indices])
    )
    upper_candidates = quantile_predictions[:, lower_indices] + (
        quantile_predictions[:, upper_indices]
        - quantile_predictions[:, lower_indices]
    ) * upper_weights
    candidate_lengths = (
        upper_candidates
        - quantile_predictions[:, : len(lower_quantiles)]
    )
    best_lower_quantile = lower_quantiles[
        np.argmin(candidate_lengths, axis=1)
    ]

    scores = np.abs(
        ranks - best_lower_quantile - (1 - alpha) / 2
    )
    calibration_size = len(y_calibration)
    order_statistic = int(
        np.ceil((1 - alpha) * (1 + calibration_size))
    )
    threshold = np.sort(scores[:calibration_size])[order_statistic]
    test_start = calibration_size
    lower_bounds = []
    upper_bounds = []

    for row in range(test_start, len(y_all)):
        centrality = np.abs(
            quantiles
            - best_lower_quantile[row]
            - (1 - alpha) / 2
        )
        interval_values = quantile_predictions[row, centrality <= threshold]
        upper_bounds.append(np.max(interval_values))
        lower_bounds.append(np.min(interval_values))

    return np.asarray(lower_bounds), np.asarray(upper_bounds)


def distributional_conformal_predictions(
    frame: pd.DataFrame,
    model_name: str,
    alpha: float,
    test_start: str = "2024-01-01",
    quantile_count: int = 200,
    show_progress: bool = True,
    fit_cache: dict[tuple[str, pd.Timestamp, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit rolling DCP-OPT intervals with calibration pooled across hours."""
    if model_name not in BASE_MODEL_FEATURES:
        raise ValueError(f"Unknown base model: {model_name}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    features: Sequence[str] = BASE_MODEL_FEATURES[model_name]
    dates = pd.to_datetime(frame["date"])
    start_date = pd.Timestamp(test_start)
    test_positions = np.flatnonzero(dates.to_numpy() >= start_date)
    lower = np.full(len(frame), np.nan)
    upper = np.full(len(frame), np.nan)
    test_dates = pd.unique(dates.iloc[test_positions])
    quantiles = np.linspace(0.001, 0.999, quantile_count)
    iterator = tqdm(
        test_dates,
        desc=f"DCP {model_name} alpha={alpha}",
        disable=not show_progress,
    )

    for target_date in iterator:
        training_mask = (dates < target_date - pd.Timedelta(days=180)) & (
            dates >= target_date - pd.Timedelta(days=360)
        )
        calibration_mask = (dates < target_date) & (
            dates >= target_date - pd.Timedelta(days=180)
        )
        training = frame.loc[training_mask]
        cache_key = (model_name, pd.Timestamp(target_date), quantile_count)
        if fit_cache is not None and cache_key in fit_cache:
            coefficients = fit_cache[cache_key]
        else:
            coefficients = _fit_quantile_grid(
                training["real"],
                training.loc[:, features],
                quantiles,
            )
            if fit_cache is not None:
                fit_cache[cache_key] = coefficients
        date_positions = np.flatnonzero(
            (dates.to_numpy() == target_date)
            & (np.arange(len(frame)) >= test_positions[0])
        )

        calibration_positions = np.flatnonzero(calibration_mask.to_numpy())
        interval_lower, interval_upper = _distributional_interval(
            frame.iloc[calibration_positions]["real"].to_numpy(),
            frame.iloc[calibration_positions][list(features)].to_numpy(),
            frame.iloc[date_positions]["real"].to_numpy(),
            frame.iloc[date_positions][list(features)].to_numpy(),
            coefficients,
            quantiles,
            alpha,
        )
        lower[date_positions] = interval_lower
        upper[date_positions] = interval_upper

    return lower, upper
