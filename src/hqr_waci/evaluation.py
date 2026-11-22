"""Paper-facing point-estimate tables."""

import pandas as pd

from hqr_waci.metrics import (
    MODEL_NAMES,
    METRIC_NAMES,
    average_length,
    empirical_coverage,
    ils_coverage,
    mean_deviation_coverage_by_interval_length,
    median_length,
    pearson_correlation,
    spearman_correlation,
    winkler_score,
)


def evaluate_models(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Calculate the nine reported metrics for all twelve evaluation methods."""
    absolute_errors = (frame["real"] - frame["mean_pred"]).abs()
    rows = {}

    for model in MODEL_NAMES:
        upper = frame[f"preds_sup_{model}"]
        lower = frame[f"preds_inf_{model}"]
        rows[model] = (
            empirical_coverage(frame["real"], upper, lower),
            average_length(upper, lower),
            median_length(upper, lower),
            winkler_score(frame["real"], upper, lower, alpha),
            pearson_correlation(frame["real"], upper, lower),
            ils_coverage(frame, model, pct_threshold=90, alpha=alpha),
            mean_deviation_coverage_by_interval_length(frame, model, 0.05, alpha),
            spearman_correlation(absolute_errors, upper, lower),
            (upper - lower).std(),
        )

    return pd.DataFrame.from_dict(rows, orient="index", columns=METRIC_NAMES)
