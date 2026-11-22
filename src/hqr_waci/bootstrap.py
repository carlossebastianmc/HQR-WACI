"""Stationary-bootstrap uncertainty estimates for the reported metrics."""

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from tqdm import tqdm

from hqr_waci.metrics import MODEL_NAMES, METRIC_NAMES, ils_coverage

BOOTSTRAP_METRICS = (
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


def bootstrap_ils_coverage(
    frame: pd.DataFrame,
    alpha: float,
    replicates: int = 1000,
    block_size: int = 1000,
    seed: int = 123,
) -> pd.DataFrame:
    """Recalculate the ILS bootstrap using shared seeded stationary samples.

    The original code resets the same bootstrap seed for each model, so each
    model receives the same resampled rows. Reusing each sample across models
    preserves that design while avoiding repeated sample generation.
    """
    samples = {model: [] for model in MODEL_NAMES}
    bootstrap = StationaryBootstrap(block_size, frame, seed=seed)

    for data, _ in bootstrap.bootstrap(replicates):
        sample = data[0]
        for model in MODEL_NAMES:
            samples[model].append(
                ils_coverage(sample, model, pct_threshold=90, alpha=alpha)
            )

    return pd.DataFrame(samples)


def bootstrap_standard_errors(
    metric_samples: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate standard errors for metrics from their bootstrap samples."""
    standard_errors = {}
    for metric, samples in metric_samples.items():
        standard_errors[metric] = samples.std(axis=0) / np.sqrt(len(samples))
    return pd.DataFrame(standard_errors).reindex(MODEL_NAMES)[list(METRIC_NAMES)]


def bootstrap_metric_samples(
    frame: pd.DataFrame,
    alpha: float,
    replicates: int = 1000,
    block_size: int = 1000,
    seed: int = 123,
) -> dict[str, pd.DataFrame]:
    """Estimate all reported metrics on shared stationary-bootstrap samples."""
    from hqr_waci.evaluation import evaluate_models

    collected = {metric: [] for metric in METRIC_NAMES}
    bootstrap = StationaryBootstrap(block_size, frame, seed=seed)
    iterator = tqdm(
        bootstrap.bootstrap(replicates),
        total=replicates,
        desc="Metric bootstrap",
    )

    for data, _ in iterator:
        estimates = evaluate_models(data[0], alpha)
        for metric in METRIC_NAMES:
            collected[metric].append(estimates[metric])

    return {
        metric: pd.DataFrame.from_records(samples, columns=MODEL_NAMES)
        for metric, samples in collected.items()
    }
