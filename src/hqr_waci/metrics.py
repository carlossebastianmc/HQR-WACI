"""Evaluation metrics matching the definitions in the original analysis."""

import numpy as np
import pandas as pd
from scipy import stats

MODEL_NAMES = (
    "hqr",
    "hqr_w",
    "qra",
    "aci_hqr",
    "aci_hqr_w",
    "aci_qra",
    "waci_hqr",
    "waci_hqr_w",
    "waci_qra",
    "dcp_hqr",
    "dcp_hqr_w",
    "dcp_qra",
)

METRIC_NAMES = (
    "empirical_coverage",
    "average_length",
    "median_length",
    "winkler_score",
    "pearson_correlation",
    "ILS_10_coverage",
    "mean_deviation_coverage_by_interval_length",
    "spearman_correlation",
    "interval_length_std",
)


def winkler_score(
    y_true: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    alpha: float,
) -> float:
    lower_penalty = np.maximum(lower - y_true, 0)
    upper_penalty = np.maximum(y_true - upper, 0)
    length = upper - lower
    score = np.where(
        y_true < lower,
        length + (2 / alpha) * lower_penalty,
        np.where(y_true > upper, length + (2 / alpha) * upper_penalty, length),
    )
    return float(np.mean(score))


def empirical_coverage(
    y_true: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean() * 100)


def average_length(upper: pd.Series, lower: pd.Series) -> float:
    return float(np.mean(upper - lower))


def median_length(upper: pd.Series, lower: pd.Series) -> float:
    return float(np.median(upper - lower))


def pearson_correlation(
    y_true: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
) -> float:
    covered = ((y_true >= lower) & (y_true <= upper)).astype(int)
    return float(stats.pearsonr(covered, upper - lower)[0])


def spearman_correlation(
    absolute_errors: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
) -> float:
    return float(stats.spearmanr(absolute_errors, upper - lower)[0])


def _base_model_for_ils(model: str) -> str | None:
    if model in {
        "hqr",
        "qra",
        "hqr_w",
        "hqr_by_hour",
        "qra_by_hour",
        "hqr_w_by_hour",
    }:
        return None
    if (
        model in {"cqr_hqr", "aci_hqr", "waci_hqr", "dcp_hqr"}
        or "hqr" in model
    ):
        return "hqr"
    if model in {
        "cqr_qra",
        "aci_qra",
        "waci_qra",
        "dcp_qra",
        "cqr_qra_by_hour",
        "aci_qra_by_hour",
        "waci_qra_by_hour",
    }:
        return "qra"
    if model in {
        "cqr_hqr_w",
        "aci_hqr_w",
        "waci_hqr_w",
        "dcp_hqr_w",
        "cqr_hqr_w_by_hour",
        "aci_hqr_w_by_hour",
        "waci_hqr_w_by_hour",
    }:
        return "hqr_w"
    return "initial"


def ils_outlier_mask(
    frame: pd.DataFrame,
    model: str,
    pct_threshold: float = 90,
) -> pd.Series | None:
    """Select observations with unusually large changes in interval length."""
    base_model = _base_model_for_ils(model)
    if base_model is None:
        return None

    lengths = frame[f"preds_sup_{model}"] - frame[f"preds_inf_{model}"]
    base_lengths = frame[f"preds_sup_{base_model}"] - frame[f"preds_inf_{base_model}"]
    threshold = np.quantile((lengths - base_lengths).abs(), pct_threshold / 100)
    return (lengths - base_lengths).abs() > threshold


def ils_coverage(
    frame: pd.DataFrame,
    model: str,
    pct_threshold: float = 90,
    alpha: float = 0.2,
) -> float:
    """Measure coverage deviation on observations with unusual interval changes."""
    selected_mask = ils_outlier_mask(frame, model, pct_threshold)
    if selected_mask is None:
        return float("nan")

    selected = frame.loc[selected_mask]
    coverage = empirical_coverage(
        selected["real"],
        selected[f"preds_sup_{model}"],
        selected[f"preds_inf_{model}"],
    )
    return abs(coverage - (1 - alpha) * 100)


def mean_deviation_coverage_by_interval_length(
    frame: pd.DataFrame,
    model: str,
    quantile_step: float,
    alpha: float,
) -> float:
    """Mean absolute coverage deviation across interval-length quantile bins."""
    lengths = frame[f"preds_sup_{model}"] - frame[f"preds_inf_{model}"]
    quantiles = np.arange(0, 1.01, quantile_step)
    boundaries = np.quantile(lengths, quantiles)
    coverages = []

    for lower_bound, upper_bound in zip(boundaries[:-1], boundaries[1:]):
        selected = frame[(lengths >= lower_bound) & (lengths <= upper_bound)]
        coverages.append(
            empirical_coverage(
                selected["real"],
                selected[f"preds_sup_{model}"],
                selected[f"preds_inf_{model}"],
            )
        )

    target_coverage = (1 - alpha) * 100
    return float(np.mean(np.abs(np.asarray(coverages) - target_coverage)))
