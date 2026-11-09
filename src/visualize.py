"""Plotting helpers for model evaluation and recommendation analysis."""

from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LILAC = "#C8A2C8"
PURPLE = "#8E5B9A"
DEEP_PURPLE = "#6A3F7A"
DARK_PURPLE = "#4A2E5D"

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(
            color=[LILAC, PURPLE, DEEP_PURPLE, DARK_PURPLE]
        ),
        "axes.edgecolor": DARK_PURPLE,
        "axes.labelcolor": DARK_PURPLE,
        "axes.titlecolor": DARK_PURPLE,
        "text.color": DARK_PURPLE,
        "xtick.color": DARK_PURPLE,
        "ytick.color": DARK_PURPLE,
        "grid.color": LILAC,
    }
)


def _save_figure(figure: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f"{path.stem}_", suffix=path.suffix, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        figure.savefig(temporary_path, dpi=160, bbox_inches="tight")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        plt.close(figure)
    return path


def plot_rating_distribution(ratings: pd.DataFrame, output_path: str | Path) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(
        ratings["rating"],
        bins=np.arange(0.25, 5.76, 0.5),
        color=LILAC,
        edgecolor="white",
    )
    axis.set(title="MovieLens rating distribution", xlabel="Rating", ylabel="Count")
    axis.set_xticks(np.arange(0.5, 5.1, 0.5))
    return _save_figure(figure, output_path)


def plot_model_comparison(
    model_metrics: dict[str, dict[str, dict[str, float | int]]],
    output_path: str | Path,
) -> Path:
    model_names = list(model_metrics)
    positions = np.arange(len(model_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7, 4))
    rmse = [model_metrics[name]["test"]["rmse"] for name in model_names]
    mae = [model_metrics[name]["test"]["mae"] for name in model_names]
    axis.bar(positions - width / 2, rmse, width, label="RMSE", color=DEEP_PURPLE)
    axis.bar(positions + width / 2, mae, width, label="MAE", color=LILAC)
    axis.set(
        title="Held-out test rating error",
        ylabel="Error (lower is better)",
        xticks=positions,
        xticklabels=model_names,
    )
    axis.legend()
    return _save_figure(figure, output_path)


def plot_actual_vs_predicted(
    predictions: pd.DataFrame, output_path: str | Path
) -> Path:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        predictions["actual_rating"],
        predictions["predicted_rating"],
        alpha=0.25,
        s=10,
        color=PURPLE,
    )
    axis.plot(
        [0.5, 5.0],
        [0.5, 5.0],
        color=DARK_PURPLE,
        linestyle="--",
        linewidth=1,
    )
    axis.set(
        title="Actual vs. predicted test ratings",
        xlabel="Actual rating",
        ylabel="Predicted rating",
        xlim=(0.25, 5.25),
        ylim=(0.25, 5.25),
    )
    return _save_figure(figure, output_path)


def plot_residual_distribution(
    predictions: pd.DataFrame, output_path: str | Path
) -> Path:
    residuals = predictions["predicted_rating"] - predictions["actual_rating"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(residuals, bins=40, color=PURPLE, edgecolor="white")
    axis.axvline(0, color=DARK_PURPLE, linestyle="--", linewidth=1)
    axis.set(
        title="Test residual distribution",
        xlabel="Predicted rating - actual rating",
        ylabel="Number of ratings",
    )
    return _save_figure(figure, output_path)


def plot_activity_errors(
    activity_metrics: pd.DataFrame, output_path: str | Path
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(activity_metrics))
    width = 0.36
    axis.bar(
        positions - width / 2,
        activity_metrics["rmse"],
        width,
        label="RMSE",
        color=DEEP_PURPLE,
    )
    axis.bar(
        positions + width / 2,
        activity_metrics["mae"],
        width,
        label="MAE",
        color=LILAC,
    )
    axis.set(
        title="Test error by user activity",
        ylabel="Error",
        xticks=positions,
        xticklabels=activity_metrics["activity_group"],
    )
    axis.legend()
    return _save_figure(figure, output_path)


def plot_genre_errors(
    genre_metrics: pd.DataFrame, output_path: str | Path, top_n: int = 12
) -> Path:
    chart_data = genre_metrics.sort_values(
        ["ratings", "genre"], ascending=[False, True], kind="stable"
    ).head(top_n).sort_values("mae")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(chart_data["genre"], chart_data["mae"], color=PURPLE)
    axis.set(title="Test MAE by genre", xlabel="MAE", ylabel="Genre")
    return _save_figure(figure, output_path)


def plot_genre_exposure(
    genre_exposure: pd.DataFrame, output_path: str | Path, top_n: int = 12
) -> Path:
    chart_data = genre_exposure.nlargest(top_n, "recommendations").sort_values(
        "recommendation_share"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(chart_data["genre"], chart_data["recommendation_share"], color=LILAC)
    axis.set(
        title="Share of recommendation slots by genre",
        xlabel="Share (genre memberships can overlap)",
        ylabel="Genre",
    )
    return _save_figure(figure, output_path)


def plot_popularity_exposure(
    popularity_metrics: dict[str, float | int], output_path: str | Path
) -> Path:
    labels = ["Catalog share", "Recommendation share"]
    values = [
        popularity_metrics["popular_catalog_share"],
        popularity_metrics["popular_recommendation_share"],
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, values, color=[LILAC, DARK_PURPLE])
    axis.set(
        title="Exposure of top 20% most-rated movies",
        ylabel="Share",
        ylim=(0, 1),
    )
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    return _save_figure(figure, output_path)
