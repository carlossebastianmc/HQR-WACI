"""Reproduce the two-state synthetic experiment reported in the paper."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from hqr_waci.metrics import (
    average_length,
    empirical_coverage,
    ils_coverage,
    mean_deviation_coverage_by_interval_length,
    median_length,
    pearson_correlation,
    spearman_correlation,
    winkler_score,
)

ALPHA = 0.2
GAMMA = 0.01
WACI_SIGMA = 1.0
FIRST_TEST_INDEX = 1000
N_OBSERVATIONS = 10_000
N_EXPERIMENTS = 100
N_SAMPLE = 10
MU = 100.0
SIGMA_HIGH = 7.0
SIGMA_LOW = 2.0
AMPLITUDE_HIGH = 2.0
FREQUENCY_HIGH = 0.001
AMPLITUDE_LOW = 1.0
FREQUENCY_LOW = 0.005
WIDTH_GRID = np.arange(0, 40, 0.1)
METHOD_NAMES = ("initial", "aci", "waci")
METRIC_NAMES = (
    "empirical_coverage",
    "average_length",
    "median_length",
    "winkler_score",
    "pearson_correlation",
    "ILS_10_coverage",
    "mean_deviation_coverage_by_interval_length",
    "spearman_correlation",
)
PARTITIONS = ("all", "high_uncertainty", "low_uncertainty")


@dataclass(frozen=True)
class SyntheticRun:
    frame: pd.DataFrame
    true_lower: np.ndarray
    true_upper: np.ndarray
    next_transition_probability: float


@dataclass(frozen=True)
class SyntheticExperimentResult:
    means: dict[str, pd.DataFrame]
    standard_deviations: dict[str, pd.DataFrame]
    example: pd.DataFrame
    true_lower: np.ndarray
    true_upper: np.ndarray


def simulate_synthetic_run(
    seed: int,
    transition_probability: float = 0.0,
) -> SyntheticRun:
    """Generate one run while preserving the original random-draw sequence."""
    rng = np.random.RandomState(seed)
    real = np.empty(N_OBSERVATIONS)
    process_state = np.empty(N_OBSERVATIONS, dtype=np.int8)
    sample_sigma = np.empty(N_OBSERVATIONS)
    current_process = 1
    steps_since_change = 0

    for step in range(N_OBSERVATIONS):
        # Capture the state and sampling scale before any process switch.
        process_state[step] = current_process
        scale = SIGMA_HIGH if current_process == 1 else SIGMA_LOW

        if rng.uniform() < transition_probability and steps_since_change > -1:
            current_process = 2 if current_process == 1 else 1
            # Reset the transition hazard after a process switch.
            transition_probability = 0.0
            steps_since_change = 0
        else:
            transition_probability += 0.0001
        steps_since_change += 1

        # These unused draws are retained because they precede each target draw.
        rng.normal(loc=MU, scale=scale, size=N_SAMPLE)
        # The target uses the pre-switch scale; the interval estimate uses the new state.
        if current_process == 1:
            sample_sigma[step] = SIGMA_HIGH + AMPLITUDE_HIGH * np.sin(
                FREQUENCY_HIGH * step
            )
        else:
            sample_sigma[step] = SIGMA_LOW + AMPLITUDE_LOW * np.sin(
                FREQUENCY_LOW * step
            )
        real[step] = rng.normal(loc=MU, scale=scale, size=1)[0]

    critical_value = stats.t.ppf(1 - ALPHA / 2, df=N_SAMPLE - 1)
    margin = critical_value * sample_sigma * np.sqrt(1 + 1 / N_SAMPLE)
    lower = MU - margin
    upper = MU + margin
    true_lower = np.where(
        process_state == 1,
        stats.norm.ppf(ALPHA / 2, loc=MU, scale=SIGMA_HIGH),
        stats.norm.ppf(ALPHA / 2, loc=MU, scale=SIGMA_LOW),
    )
    true_upper = np.where(
        process_state == 1,
        stats.norm.ppf(1 - ALPHA / 2, loc=MU, scale=SIGMA_HIGH),
        stats.norm.ppf(1 - ALPHA / 2, loc=MU, scale=SIGMA_LOW),
    )
    frame = pd.DataFrame(
        {
            "real": real,
            "preds_inf_initial": lower,
            "preds_sup_initial": upper,
            "actual_process": process_state,
        }
    )
    return SyntheticRun(
        frame,
        true_lower,
        true_upper,
        transition_probability,
    )


def _conformal_adjustment(
    conformity_scores: np.ndarray,
    observation: int,
    alpha: float,
) -> float:
    quantile = np.clip(
        np.ceil(observation + 1) * (1 - alpha) / observation,
        0,
        1,
    )
    return float(np.quantile(conformity_scores[:observation], quantile))


def _replace_infinite_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = lower.copy()
    upper = upper.copy()
    lower[np.isneginf(lower)] = np.min(base_lower)
    upper[np.isposinf(upper)] = np.max(base_upper)
    return lower, upper


def _apply_aci(
    real: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conformity_scores = np.maximum(real - base_upper, base_lower - real)
    lower = np.full(len(real), np.nan)
    upper = np.full(len(real), np.nan)
    alpha_t = ALPHA

    for observation in range(FIRST_TEST_INDEX, len(real)):
        if 0 < alpha_t < 1:
            adjustment = _conformal_adjustment(
                conformity_scores,
                observation,
                alpha_t,
            )
            lower[observation] = base_lower[observation] - adjustment
            upper[observation] = base_upper[observation] + adjustment
        elif alpha_t >= 1:
            lower[observation] = base_lower[observation]
            upper[observation] = base_upper[observation]
        else:
            lower[observation] = -np.inf
            upper[observation] = np.inf

        covered = (
            real[observation] >= lower[observation]
            and real[observation] <= upper[observation]
        )
        alpha_t += GAMMA * (ALPHA - (1 - float(covered)))

    return _replace_infinite_bounds(lower, upper, base_lower, base_upper)


def _apply_waci(
    real: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conformity_scores = np.maximum(real - base_upper, base_lower - real)
    lower = np.full(len(real), np.nan)
    upper = np.full(len(real), np.nan)
    alpha_by_width = np.full(len(WIDTH_GRID), ALPHA)

    for observation in range(FIRST_TEST_INDEX, len(real)):
        observed_width = base_upper[observation] - base_lower[observation]
        distances = np.abs(WIDTH_GRID - observed_width)
        nearest_width = int(np.argmin(distances))
        alpha_t = alpha_by_width[nearest_width]

        if 0 < alpha_t < 1:
            adjustment = _conformal_adjustment(
                conformity_scores,
                observation,
                alpha_t,
            )
            lower[observation] = base_lower[observation] - adjustment
            upper[observation] = base_upper[observation] + adjustment
        elif alpha_t >= 1:
            midpoint = (base_lower[observation] + base_upper[observation]) / 2
            lower[observation] = midpoint
            upper[observation] = midpoint
        else:
            lower[observation] = -np.inf
            upper[observation] = np.inf

        covered = (
            real[observation] >= lower[observation]
            and real[observation] <= upper[observation]
        )
        error = 1 - float(covered)
        weights = np.exp(-(distances**2) / (2 * WACI_SIGMA**2))
        weights /= weights.max()
        alpha_by_width += GAMMA * weights * (ALPHA - error)

    return _replace_infinite_bounds(lower, upper, base_lower, base_upper)


def conformalize_synthetic_run(run: SyntheticRun) -> pd.DataFrame:
    """Apply the original ACI and WACI updates to a generated base interval."""
    frame = run.frame.copy()
    real = frame["real"].to_numpy()
    base_lower = frame["preds_inf_initial"].to_numpy()
    base_upper = frame["preds_sup_initial"].to_numpy()
    aci_lower, aci_upper = _apply_aci(real, base_lower, base_upper)
    waci_lower, waci_upper = _apply_waci(real, base_lower, base_upper)
    frame["preds_inf_aci"] = aci_lower
    frame["preds_sup_aci"] = aci_upper
    frame["preds_inf_waci"] = waci_lower
    frame["preds_sup_waci"] = waci_upper
    return frame


def _method_metrics(frame: pd.DataFrame, model: str) -> dict[str, float]:
    lower = frame[f"preds_inf_{model}"]
    upper = frame[f"preds_sup_{model}"]
    return {
        "empirical_coverage": empirical_coverage(frame["real"], upper, lower),
        "average_length": average_length(upper, lower),
        "median_length": median_length(upper, lower),
        "winkler_score": winkler_score(frame["real"], upper, lower, ALPHA),
        "pearson_correlation": pearson_correlation(frame["real"], upper, lower),
        "ILS_10_coverage": ils_coverage(
            frame,
            model,
            pct_threshold=90,
            alpha=ALPHA,
        ),
        "mean_deviation_coverage_by_interval_length": (
            mean_deviation_coverage_by_interval_length(
                frame,
                model,
                quantile_step=0.05,
                alpha=ALPHA,
            )
        ),
        "spearman_correlation": spearman_correlation(
            (frame["real"] - MU).abs(),
            upper,
            lower,
        ),
    }


def run_synthetic_experiments(
    repetitions: int = N_EXPERIMENTS,
    show_progress: bool = False,
) -> SyntheticExperimentResult:
    """Run the seeded 100-repetition experiment and summarize each state."""
    records = {
        partition: {method: [] for method in METHOD_NAMES}
        for partition in PARTITIONS
    }
    transition_probability = 0.0
    example_frame = None
    example_run = None

    for seed in tqdm(
        range(repetitions),
        desc="Synthetic experiment",
        disable=not show_progress,
    ):
        run = simulate_synthetic_run(seed, transition_probability)
        transition_probability = run.next_transition_probability
        frame = conformalize_synthetic_run(run)
        evaluation = frame.loc[frame["preds_inf_aci"].notna()]
        partitions = {
            "all": evaluation,
            "high_uncertainty": evaluation.loc[
                evaluation["actual_process"] == 1
            ],
            "low_uncertainty": evaluation.loc[
                evaluation["actual_process"] == 2
            ],
        }

        for partition, subset in partitions.items():
            for method in METHOD_NAMES:
                records[partition][method].append(
                    _method_metrics(subset, method)
                )

        if seed == 0:
            example_frame = frame
            example_run = run

    means = {}
    standard_deviations = {}
    for partition in PARTITIONS:
        mean_rows = {}
        deviation_rows = {}
        for method in METHOD_NAMES:
            values = pd.DataFrame(records[partition][method])
            mean_rows[method] = values.mean(axis=0).reindex(METRIC_NAMES)
            deviation_rows[method] = values.std(axis=0, ddof=0).reindex(
                METRIC_NAMES
            )
        means[partition] = pd.DataFrame.from_dict(mean_rows, orient="index")
        standard_deviations[partition] = pd.DataFrame.from_dict(
            deviation_rows,
            orient="index",
        )

    if example_frame is None or example_run is None:
        raise ValueError("At least one synthetic repetition is required")
    return SyntheticExperimentResult(
        means,
        standard_deviations,
        example_frame,
        example_run.true_lower,
        example_run.true_upper,
    )
