"""Reproduction of the paper's EPF and alpha-progression figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

COLORS = sns.color_palette("colorblind")
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


def _configure_paper_style() -> None:
    plt.rcParams.update(
        {
            "ytick.color": "black",
            "xtick.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Serif"],
        }
    )


def plot_marginal_vs_conditional_coverage(output_path: Path, seed: int = 123) -> None:
    _configure_paper_style()
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, 1000))
    y = rng.normal(0, x)
    lower_conditional = -1.95 * x
    upper_conditional = 1.95 * x

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lower, upper, title in (
        (axes[0], -11.5, 11.5, "Marginal coverage"),
        (
            axes[1],
            lower_conditional,
            upper_conditional,
            "Conditional coverage",
        ),
    ):
        ax.scatter(x, y, alpha=0.9, zorder=2, label="Data")
        ax.fill_between(x, lower, upper, alpha=0.5, zorder=1, label="Prediction interval")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X", fontsize=13)
        ax.set_ylabel("Y", fontsize=13)
        ax.grid()
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_intervals(
    frame: pd.DataFrame,
    true_lower: np.ndarray,
    true_upper: np.ndarray,
    output_path: Path,
) -> None:
    _configure_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(
        frame["preds_sup_initial"],
        label="Unconformalized interval",
        color=COLORS[0],
    )
    ax.plot(frame["preds_inf_initial"], color=COLORS[0])
    ax.plot(frame["preds_sup_waci"], label="WACI", color=COLORS[1])
    ax.plot(frame["preds_inf_waci"], color=COLORS[1])
    ax.plot(frame["preds_sup_aci"], label="ACI", color=COLORS[2])
    ax.plot(frame["preds_inf_aci"], color=COLORS[2])
    ax.plot(true_upper, label="True interval", linestyle="--", color="k")
    ax.plot(true_lower, linestyle="--", color="k")
    ax.scatter(
        frame.index,
        frame["real"],
        color="grey",
        alpha=0.2,
        label=r"$y_t$",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid()
    ax.set_xlim(8000, 9000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_xticks(np.arange(8000, 9001, 200))
    ax.set_xticklabels(np.arange(0, 1001, 200))
    ax.set_ylim(85, 115)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_day_ahead_market_figure(
    frame: pd.DataFrame,
    output_path: Path,
    test_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
) -> None:
    _configure_paper_style()
    frame = frame.copy()
    frame["full_date"] = pd.to_datetime(frame["full_date"])
    after_test = frame.index[frame["full_date"] > test_date]
    prior_window = after_test - 6 * 30 * 24

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(frame["full_date"], frame["real"], color=COLORS[0])
    ax.plot(
        frame.loc[prior_window, "full_date"],
        frame.loc[prior_window, "real"],
        color=COLORS[1],
    )
    ax.plot(
        frame.loc[after_test, "full_date"],
        frame.loc[after_test, "real"],
        color=COLORS[2],
    )
    ax.axvline(x=test_date, color="k", linestyle="--", lw=2)
    ax.axvline(x=test_date - pd.Timedelta(days=6 * 30), color="k", linestyle="--", lw=2)
    ax.grid()
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Price (€/MWh)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.text(
        test_date - pd.Timedelta(days=10),
        200,
        "Quantile Regression predictions",
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=(test_date - pd.Timedelta(days=5), 180),
        xytext=(test_date - pd.Timedelta(days=6 * 30), 180),
        arrowprops={"arrowstyle": "->"},
    )
    ax.text(
        test_date + pd.Timedelta(days=6 * 30) - pd.Timedelta(days=15),
        200,
        "Conformal prediction methods",
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=(test_date + pd.Timedelta(days=6 * 30), 180),
        xytext=(test_date, 180),
        arrowprops={"arrowstyle": "->"},
    )
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    fig.tight_layout()
    plt.close(fig)


def plot_day_ahead_market_predictions(frame: pd.DataFrame, output_path: Path) -> None:
    _configure_paper_style()
    frame = frame.copy()
    frame["full_date"] = pd.to_datetime(frame["full_date"])
    frame["date"] = pd.to_datetime(frame["date"])
    start = pd.Timestamp("2024-05-05")
    week = frame.loc[(frame["date"] >= start) & (frame["date"] < start + pd.Timedelta(days=7))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(week["full_date"], week["real"], label="DA market price", color=COLORS[0])
    for column, label, color in (
        ("pred1", "Model 1", COLORS[1]),
        ("pred2", "Model 2", COLORS[2]),
        ("pred3", "Model 3", COLORS[4]),
    ):
        ax.plot(week["full_date"], week[column], "--", label=label, color=color)
    for date in week["date"].unique():
        ax.axvline(x=pd.Timestamp(date), color="k", linestyle="--", lw=1)
    ax.grid()
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Price (€/MWh)", fontsize=13)
    ax.legend(prop={"size": 12}, loc="upper right", frameon=True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rolling_window(output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 4))
    for row, ax in enumerate(axes):
        ax.set_facecolor("black")
        training_start = 2 + row * 0.6
        test_position = training_start + 6
        for offset in range(5):
            position = training_start + offset * 1.2
            ax.add_patch(
                plt.Rectangle(
                    (position, 0.25),
                    0.9,
                    0.5,
                    facecolor="#a6c8e5",
                    edgecolor="none",
                )
            )
            ax.text(
                position + 0.45,
                0.5,
                f"T{f'+{offset - (4 - row)}' if offset > 4 - row else ''}"
                if offset >= 4 - row
                else "",
                ha="center",
                va="center",
            )
        ax.add_patch(
            plt.Rectangle(
                (test_position, 0.25),
                0.9,
                0.5,
                facecolor="#f4c2a8",
                edgecolor="none",
            )
        )
        ax.text(test_position + 0.45, 0.5, f"T+{row + 1}", ha="center", va="center")
        ax.set_xlim(0, 9.5)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="black")
    plt.close(fig)


def plot_alpha_progression_figures(
    aci_history: list[float],
    waci_history: list[np.ndarray],
    waci2_history: list[np.ndarray],
    output_dir: Path,
) -> None:
    _configure_paper_style()
    if not (
        len(aci_history) == len(waci_history) == len(waci2_history)
        and len(aci_history) > 320
    ):
        raise ValueError("Alpha histories must contain at least 321 aligned states")

    width_grid = np.arange(0, 500, 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, iteration in (
        (axes[0, 0], 0),
        (axes[0, 1], 1),
        (axes[1, 0], 100),
        (axes[1, 1], 320),
    ):
        aci = aci_history[iteration]
        waci = waci_history[iteration]
        ax.axhline(0.2, xmin=0, xmax=500, lw=1, color="k", linestyle="--", label="Objective miscoverage", alpha=0.4)
        ax.plot(width_grid, waci, color=COLORS[0], label="WACI", lw=1)
        ax.axhline(aci, xmin=0, xmax=500, lw=1, color=COLORS[1], label="ACI")
        ax.set_xlabel("Interval length")
        ax.set_ylabel(rf"$\alpha_{{{iteration + 1}}}$")
        ax.grid()
        ax.set_ylim(0.1, 0.3)
        ax.set_yticks(np.arange(0.1, 0.3001, 0.025))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlim(0, 100)
        ax.legend(frameon=True, prop={"size": 8})
        ax.set_title(f"$T+1 = {iteration + 1}$")

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_progression_square.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    index = min(366, len(aci_history) - 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axhline(0.2, xmin=0, xmax=500, lw=1, color="k", linestyle="--", label="Objective miscoverage", alpha=0.4)
    ax.plot(width_grid, waci_history[index], color=COLORS[0], label="WACI", lw=1)
    ax.axhline(aci_history[index], xmin=0, xmax=500, lw=1, color=COLORS[1], label="ACI")
    ax.plot(width_grid, waci2_history[index], color=COLORS[2], label="WACI 2", lw=1)
    ax.set_xlabel("Interval length")
    ax.set_ylabel(rf"$\alpha_{{{index + 1}}}$")
    ax.grid()
    ax.set_ylim(0.1, 0.3)
    ax.set_yticks(np.arange(0.1, 0.3001, 0.025))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlim(0, 100)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_progression_hour13.pdf", format="pdf")
    plt.close(fig)


def plot_sigma_sensitivity(
    results_dir: Path,
    output_path: Path,
) -> None:
    _configure_paper_style()
    sensitivity = pd.read_csv(
        results_dir / "df_results_sigma_impact_alpha_0.2.csv",
        index_col=0,
    )
    aci = pd.read_csv(
        results_dir / "df_results_alpha_0.2.csv",
        index_col=0,
    ).loc["aci_hqr"]

    transition_point = 30
    sigma_min = 0.1
    sigma_max = 200
    transition_relative = (transition_point - sigma_min) / (sigma_max - sigma_min)
    color_map = LinearSegmentedColormap(
        "custom_cmap",
        {
            "red": [
                (0.0, 68 / 255, 68 / 255),
                (transition_relative, 68 / 255, 68 / 255),
                (1.0, 253 / 255, 253 / 255),
            ],
            "green": [
                (0.0, 1 / 255, 1 / 255),
                (transition_relative, 148 / 255, 148 / 255),
                (1.0, 231 / 255, 231 / 255),
            ],
            "blue": [
                (0.0, 84 / 255, 84 / 255),
                (transition_relative, 226 / 255, 226 / 255),
                (1.0, 37 / 255, 37 / 255),
            ],
        },
        N=256,
    )
    norm = plt.Normalize(vmin=sensitivity.index.min(), vmax=sensitivity.index.max())

    fig = plt.figure(figsize=(14, 5))
    grid = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    axes = [fig.add_subplot(grid[i]) for i in range(3)]
    colorbar_axis = fig.add_subplot(grid[3])
    for sigma in sensitivity.index:
        color = color_map(norm(sigma))
        axes[0].plot(
            sensitivity.loc[sigma, "empirical_coverage"],
            sensitivity.loc[sigma, "average_length"],
            marker="o",
            color=color,
            zorder=1,
        )
        axes[1].plot(
            sensitivity.loc[sigma, "mean_deviation_coverage_by_interval_length"],
            sensitivity.loc[sigma, "ILS_10_coverage"],
            marker="o",
            color=color,
            zorder=1,
        )
        axes[2].plot(
            sigma,
            sensitivity.loc[sigma, "pearson_correlation"],
            marker="o",
            color=color,
            zorder=1,
        )
    axes[0].scatter(
        aci["empirical_coverage"],
        aci["average_length"],
        marker="*",
        color="k",
        label="ACI-HQR",
        zorder=2,
    )
    axes[1].scatter(
        aci["mean_deviation_coverage_by_interval_length"],
        aci["ILS_10_coverage"],
        marker="*",
        color="k",
        label="ACI-HQR",
        zorder=2,
    )
    axes[2].scatter(
        200.1,
        aci["pearson_correlation"],
        marker="*",
        color="k",
        label="ACI-HQR",
        zorder=2,
    )
    axes[0].axvline(80, color="k", linestyle="--", lw=1.2, label="Objective coverage")
    axes[0].set_xlabel("Mean empirical coverage")
    axes[0].set_ylabel("Average interval length")
    axes[0].legend(frameon=True)
    axes[1].set_xlabel("MCD")
    axes[1].set_ylabel("ILS 10% coverage")
    axes[2].set_xlabel(r"$\sigma$")
    axes[2].set_ylabel("Pearson correlation")
    for ax in axes:
        ax.grid()
        if ax is not axes[0]:
            ax.legend(frameon=True)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=color_map),
        cax=colorbar_axis,
    )
    colorbar.set_label(r"$\sigma$")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_significance_results(results_dir: Path, output_dir: Path) -> None:
    _configure_paper_style()
    axis_names = [
        "HQR",
        "HQR-W",
        "QRA",
        "ACI HQR",
        "ACI HQR-W",
        "ACI QRA",
        "WACI HQR",
        "WACI HQR-W",
        "WACI QRA",
        "DCP HQR",
        "DCP HQR-W",
        "DCP QRA",
    ]
    metric_specs = [
        ("empirical_coverage", "Empirical coverage", 0, 0, ""),
        ("average_length", "Average length", 0, 1, r" $\downarrow$"),
        ("winkler_score", "Winkler score", 0, 2, r" $\downarrow$"),
        ("interval_length_std", "Interval length std", 1, 0, r" $\uparrow$"),
        ("spearman_correlation", "Spearman correlation", 1, 1, r" $\uparrow$"),
        ("mean_deviation_coverage_by_interval_length", "MCD 5", 2, 0, r" $\downarrow$"),
        ("pearson_correlation", "Pearson correlation", 2, 1, r" $\downarrow$"),
        ("ILS_10_coverage", "ILS 10", 2, 2, r" $\downarrow$"),
    ]
    row_titles = [
        "General metrics",
        "Difficulty adaptivity metrics",
        "Coverage--width independence metrics",
    ]
    display_names = dict(zip(MODEL_NAMES, axis_names, strict=True))
    model_by_display_name = dict(zip(axis_names, MODEL_NAMES, strict=True))
    group_colors = {"HQR": COLORS[0], "HQR-W": COLORS[1], "QRA": COLORS[2]}
    group_markers = {"HQR": "o", "HQR-W": "s", "QRA": "^"}
    model_groups = {
        model: (
            "HQR-W"
            if "HQR-W" in display_names[model]
            else "QRA"
            if "QRA" in display_names[model]
            else "HQR"
        )
        for model in MODEL_NAMES
    }
    grouped_models = {
        "Base": axis_names[0:3],
        "ACI": axis_names[3:6],
        "WACI": axis_names[6:9],
        "DCP": axis_names[9:12],
    }

    positions: dict[str, int] = {}
    group_centers = []
    group_boundaries = []
    current_x = 0
    for models in grouped_models.values():
        start_x = current_x
        for model in models:
            positions[model] = current_x
            current_x += 1
        end_x = current_x - 1
        group_centers.append((start_x + end_x) / 2)
        group_boundaries.append(end_x + 0.5)
        current_x += 1

    for alpha in (0.1, 0.2):
        point_results = pd.read_csv(
            results_dir / f"df_results_alpha_{alpha}.csv",
            index_col=0,
        )
        fig = plt.figure(figsize=(12, 10), layout="constrained")
        subfigures = fig.subfigures(3, 1)
        axes = np.array(
            [
                subfigure.subplots(1, 3)
                for subfigure in subfigures
            ]
        )
        for subfigure, title in zip(subfigures, row_titles):
            subfigure.suptitle(title, fontsize=13)

        for metric, title, row, column, arrow in metric_specs:
            samples = pd.read_csv(
                results_dir / f"df_{metric}_alpha_{alpha}.csv",
                index_col=0,
            )
            means = point_results[metric].reindex(samples.columns)
            standard_errors = samples.std(axis=0)
            errors = 1.96 * standard_errors / np.sqrt(samples.shape[0])
            for model in samples.columns:
                display_name = display_names[model]
                group = model_groups[model]
                axes[row, column].errorbar(
                    positions[display_name],
                    means.loc[model],
                    fmt=group_markers[group],
                    yerr=errors.loc[model],
                    color=group_colors[group],
                    zorder=2,
                    markersize=5,
                    capsize=5,
                    elinewidth=2,
                )
            for models in grouped_models.values():
                xs = [positions[model] for model in models]
                ys = [
                    means.loc[model_by_display_name[model]]
                    for model in models
                ]
                axes[row, column].plot(xs, ys, linestyle="--", color="grey", alpha=0.7, zorder=1)
            for boundary in group_boundaries[:-1]:
                axes[row, column].axvline(
                    boundary,
                    color="grey",
                    linestyle="-",
                    lw=1,
                    alpha=0.35,
                    zorder=1,
                )
            if metric == "empirical_coverage":
                axes[row, column].axhline(
                    (1 - alpha) * 100,
                    color="black",
                    linestyle=":",
                    lw=1.5,
                    zorder=1,
                )
                axes[row, column].legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            color="black",
                            linestyle=":",
                            lw=1.5,
                            label=r"$1-\alpha$",
                        )
                    ],
                    loc="best",
                    fontsize=8,
                    frameon=True,
                )
            axes[row, column].set_title(title + arrow)
            axes[row, column].set_xticks(group_centers)
            axes[row, column].set_xticklabels(list(grouped_models), fontsize=8)
            axes[row, column].tick_params(axis="y", labelsize=8)
            axes[row, column].grid(zorder=0)

        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        for spine in axes[1, 2].spines.values():
            spine.set_visible(False)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker=group_markers[group],
                color="none",
                markerfacecolor=group_colors[group],
                markeredgecolor=group_colors[group],
                markersize=7,
                label=group,
            )
            for group in ("HQR", "HQR-W", "QRA")
        ]
        fig.legend(
            handles=legend_handles,
            title="Base model",
            loc="outside upper center",
            ncol=3,
            frameon=True,
        )
        fig.savefig(
            output_dir / f"significance_results_alpha_{alpha}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
