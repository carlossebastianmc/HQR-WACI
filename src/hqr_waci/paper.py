"""Run the complete paper analysis from the processed hourly dataset."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning
from tqdm import tqdm

from hqr_waci.bootstrap import bootstrap_metric_samples, bootstrap_standard_errors
from hqr_waci.conformal import (
    adaptive_conformal_inference,
    width_adaptive_conformal_inference,
)
from hqr_waci.data import load_analysis_data, select_test_period
from hqr_waci.evaluation import evaluate_models
from hqr_waci.figures import (
    plot_alpha_progression_figures,
    plot_day_ahead_market_figure,
    plot_day_ahead_market_predictions,
    plot_marginal_vs_conditional_coverage,
    plot_rolling_window,
    plot_sigma_sensitivity,
    plot_significance_results,
    plot_synthetic_intervals,
)
from hqr_waci.metrics import (
    average_length,
    empirical_coverage,
    ils_coverage,
    mean_deviation_coverage_by_interval_length,
    median_length,
    pearson_correlation,
    winkler_score,
)
from hqr_waci.models import (
    BASE_MODEL_FEATURES,
    distributional_conformal_predictions,
    rolling_quantile_predictions,
)
from hqr_waci.simulation import PARTITIONS, run_synthetic_experiments

ALPHAS = (0.1, 0.2)
SIGMAS = np.arange(0.1, 200.1, 0.1)


def _fit_intervals(
    frame: pd.DataFrame,
    alpha: float,
    dcp_fit_cache: dict[tuple[str, pd.Timestamp, int], np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    test_positions = select_test_period(result).index
    alpha_histories = {}

    for model in BASE_MODEL_FEATURES:
        lower, upper = rolling_quantile_predictions(result, model, alpha)
        result[f"preds_inf_{model}"] = lower
        result[f"preds_sup_{model}"] = upper
        result[f"length_{model}"] = upper - lower

    for base_model in BASE_MODEL_FEATURES:
        aci = adaptive_conformal_inference(
            result,
            base_model,
            alpha,
            history_hour=13 if alpha == 0.2 else None,
        )
        waci = width_adaptive_conformal_inference(
            result,
            base_model,
            alpha,
            sigma=3,
            history_hour=13 if alpha == 0.2 else None,
        )
        result.loc[test_positions, f"preds_inf_aci_{base_model}"] = aci.lower
        result.loc[test_positions, f"preds_sup_aci_{base_model}"] = aci.upper
        result.loc[test_positions, f"preds_inf_waci_{base_model}"] = waci.lower
        result.loc[test_positions, f"preds_sup_waci_{base_model}"] = waci.upper

        if alpha == 0.2 and base_model == "hqr":
            waci2 = width_adaptive_conformal_inference(
                result,
                base_model,
                alpha,
                weighting="exponential",
                exponential_rate=1.02,
                history_hour=13,
            )
            result.loc[test_positions, "preds_inf_waci_2_hqr"] = waci2.lower
            result.loc[test_positions, "preds_sup_waci_2_hqr"] = waci2.upper
            alpha_histories = {
                "aci": aci.alpha_history[13],
                "waci": waci.alpha_history[13],
                "waci2": waci2.alpha_history[13],
            }

    for model in BASE_MODEL_FEATURES:
        lower, upper = distributional_conformal_predictions(
            result,
            model,
            alpha,
            fit_cache=dcp_fit_cache,
        )
        result[f"preds_inf_dcp_{model}"] = lower
        result[f"preds_sup_dcp_{model}"] = upper

    return result, alpha_histories


def _write_metrics(
    frame: pd.DataFrame,
    alpha: float,
    results_dir: Path,
) -> pd.DataFrame:
    test = select_test_period(frame)
    point_estimates = evaluate_models(test, alpha)
    point_estimates.to_csv(results_dir / f"df_results_alpha_{alpha}.csv")

    samples = bootstrap_metric_samples(test, alpha)
    for metric, values in samples.items():
        values.to_csv(results_dir / f"df_{metric}_alpha_{alpha}.csv")
    bootstrap_standard_errors(samples).to_csv(
        results_dir / f"df_standard_deviations_bootstrap_alpha_{alpha}.csv"
    )
    return point_estimates


def _write_sigma_sensitivity(frame: pd.DataFrame, results_dir: Path) -> None:
    test = select_test_period(frame)
    rows = []
    for sigma in tqdm(SIGMAS, desc="WACI sigma sensitivity"):
        intervals = width_adaptive_conformal_inference(
            frame,
            "hqr",
            0.2,
            sigma=float(sigma),
        )
        lower_name = f"preds_inf_waci_hqr_{sigma}"
        upper_name = f"preds_sup_waci_hqr_{sigma}"
        values = test.copy()
        values[lower_name] = intervals.lower
        values[upper_name] = intervals.upper
        model_name = f"waci_hqr_{sigma}"
        values[f"preds_inf_{model_name}"] = values[lower_name]
        values[f"preds_sup_{model_name}"] = values[upper_name]
        lower = values[f"preds_inf_{model_name}"]
        upper = values[f"preds_sup_{model_name}"]
        rows.append(
            {
                "sigma": sigma,
                "empirical_coverage": empirical_coverage(
                    values["real"], upper, lower
                ),
                "average_length": average_length(upper, lower),
                "median_length": median_length(upper, lower),
                "winkler_score": winkler_score(
                    values["real"], upper, lower, 0.2
                ),
                "pearson_correlation": pearson_correlation(
                    values["real"], upper, lower
                ),
                "ILS_10_coverage": ils_coverage(
                    values, model_name, pct_threshold=10
                ),
                "mean_deviation_coverage_by_interval_length": (
                    mean_deviation_coverage_by_interval_length(
                        values,
                        model_name,
                        0.05,
                        0.2,
                    )
                ),
            }
        )
    pd.DataFrame(rows).set_index("sigma").to_csv(
        results_dir / "df_results_sigma_impact_alpha_0.2.csv"
    )


def run_paper(
    input_path: Path,
    output_dir: Path,
    run_simulation: bool = True,
) -> None:
    """Fit all models and produce the paper's tables, figures, and diagnostics."""
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_analysis_data(input_path)
    fitted_frames = {}
    alpha_histories = {}
    metric_tables = []
    dcp_fit_cache: dict[tuple[str, pd.Timestamp, int], np.ndarray] = {}
    for alpha in ALPHAS:
        fitted, histories = _fit_intervals(data, alpha, dcp_fit_cache)
        fitted_frames[alpha] = fitted
        alpha_histories[alpha] = histories
        test = select_test_period(fitted)
        test.to_csv(
            results_dir / f"predictions_alpha_{alpha}.csv.gz",
            index=False,
            compression="gzip",
        )
        point_estimates = _write_metrics(fitted, alpha, results_dir)
        metric_tables.append(
            point_estimates.rename_axis("model")
            .reset_index()
            .assign(alpha=alpha)
        )
        print(f"Completed model fits and metrics for alpha={alpha}")

    pd.concat(metric_tables, ignore_index=True).loc[
        :, ["alpha", "model", *point_estimates.columns]
    ].to_csv(results_dir / "paper_metrics.csv", index=False)

    alpha_02 = fitted_frames[0.2]
    plot_rolling_window(figures_dir / "rolling_window.png")
    plot_marginal_vs_conditional_coverage(
        figures_dir / "marginal_vs_conditional_coverage.pdf"
    )
    plot_alpha_progression_figures(
        alpha_histories[0.2]["aci"],
        alpha_histories[0.2]["waci"],
        alpha_histories[0.2]["waci2"],
        figures_dir,
    )
    plot_day_ahead_market_figure(
        data,
        figures_dir / "day_ahead_market_figure.pdf",
    )
    plot_day_ahead_market_predictions(
        data,
        figures_dir / "day_ahead_market_predictions.pdf",
    )
    _write_sigma_sensitivity(alpha_02, results_dir)
    plot_sigma_sensitivity(
        results_dir,
        figures_dir / "sigma_impact.pdf",
    )
    plot_significance_results(results_dir, figures_dir)
    if run_simulation:
        simulation = run_synthetic_experiments(show_progress=True)
        for partition in PARTITIONS:
            simulation.means[partition].to_csv(
                results_dir / f"synthetic_{partition}_mean.csv",
                index_label="method",
            )
            simulation.standard_deviations[partition].to_csv(
                results_dir / f"synthetic_{partition}_std.csv",
                index_label="method",
            )
        plot_synthetic_intervals(
            simulation.example,
            simulation.true_lower,
            simulation.true_upper,
            figures_dir / "synthetic_example_2.pdf",
        )

    print(f"Paper outputs written to {output_dir}")
