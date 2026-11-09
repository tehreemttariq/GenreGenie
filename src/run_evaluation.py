"""Reproduce rating metrics, ranking metrics, plots, and bias analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bias_analysis import (
    analyze_genre_errors,
    analyze_recommendation_exposure,
    analyze_user_activity,
    write_bias_report,
)
from dataset import split_data
from evaluate import evaluate_model, ranking_metrics_at_k
from inference import MovieRecommender
from model import CollaborativeFilteringBaseline, MatrixFactorizationSVD
from visualize import (
    plot_actual_vs_predicted,
    plot_activity_errors,
    plot_genre_errors,
    plot_genre_exposure,
    plot_model_comparison,
    plot_popularity_exposure,
    plot_rating_distribution,
    plot_residual_distribution,
)


ROOT = Path(__file__).resolve().parents[1]
RATING_THRESHOLD = 4.0
TOP_K = 10


def _encode_known_rows(
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    known = frame[
        frame["userId"].isin(user_to_idx)
        & frame["movieId"].isin(movie_to_idx)
    ].copy()
    return (
        known,
        known["userId"].map(user_to_idx).to_numpy(dtype=int),
        known["movieId"].map(movie_to_idx).to_numpy(dtype=int),
        known["rating"].to_numpy(dtype=float),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _build_error_analysis(
    predictions: pd.DataFrame, movies: pd.DataFrame
) -> dict[str, list[dict[str, Any]]]:
    detailed = predictions.merge(
        movies[["movieId", "title"]].drop_duplicates("movieId"),
        on="movieId",
        how="left",
        validate="many_to_one",
    )
    detailed["error"] = detailed["predicted_rating"] - detailed["actual_rating"]
    detailed["absolute_error"] = detailed["error"].abs()
    worst_rows = detailed.nlargest(20, "absolute_error")[
        [
            "userId",
            "movieId",
            "title",
            "actual_rating",
            "predicted_rating",
            "error",
            "absolute_error",
        ]
    ]
    by_movie = (
        detailed.groupby(["movieId", "title"], dropna=False)
        .agg(
            ratings=("userId", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .query("ratings >= 10")
        .nlargest(10, "mean_absolute_error")
        .reset_index()
    )
    by_user = (
        detailed.groupby("userId")
        .agg(
            ratings=("movieId", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .query("ratings >= 5")
        .nlargest(10, "mean_absolute_error")
        .reset_index()
    )
    return {
        "largest_individual_errors": worst_rows.to_dict(orient="records"),
        "highest_error_movies_min_10_ratings": by_movie.to_dict(orient="records"),
        "highest_error_users_min_5_ratings": by_user.to_dict(orient="records"),
    }


def run_evaluation(project_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(project_root)
    ratings_path = root / "data" / "cleaned" / "ratings_cleaned.csv"
    movies_path = root / "data" / "cleaned" / "movies_cleaned.csv"
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Cleaned ratings not found: {ratings_path}")
    if not movies_path.is_file():
        raise FileNotFoundError(f"Cleaned movie metadata not found: {movies_path}")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    required_rating_columns = {"userId", "movieId", "rating"}
    missing = required_rating_columns.difference(ratings.columns)
    if missing:
        raise ValueError(f"Cleaned ratings are missing columns: {sorted(missing)}")
    train_df, val_df, test_df = split_data(ratings)

    user_to_idx = {
        int(user_id): index for index, user_id in enumerate(train_df["userId"].unique())
    }
    movie_to_idx = {
        int(movie_id): index for index, movie_id in enumerate(train_df["movieId"].unique())
    }
    (val_known, val_users, val_movies, val_actual) = _encode_known_rows(
        val_df, user_to_idx, movie_to_idx
    )
    (test_known, test_users, test_movies, test_actual) = _encode_known_rows(
        test_df, user_to_idx, movie_to_idx
    )
    if val_known.empty or test_known.empty:
        raise ValueError(
            "The deterministic split has no validation or test rows with known "
            "training users and movies."
        )

    model_loaders = {
        "SVD": (
            root / "models" / "svd_model.pkl",
            MatrixFactorizationSVD.load,
        ),
        "Baseline": (
            root / "models" / "baseline_model.pkl",
            CollaborativeFilteringBaseline.load,
        ),
    }
    model_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    test_predictions: dict[str, pd.DataFrame] = {}
    loaded_models = {}
    for model_name, (model_path, loader) in model_loaders.items():
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained {model_name} model not found: {model_path}")
        model = loader(str(model_path))
        loaded_models[model_name] = model
        expected_shape = (len(user_to_idx), len(movie_to_idx))
        model_shape = (
            (len(model.user_factors), len(model.item_factors))
            if model_name == "SVD"
            else tuple(model.ratings_matrix.shape)
        )
        if model_shape != expected_shape:
            raise ValueError(
                f"{model_name} artifact shape {model_shape} does not match the "
                f"recreated training mapping {expected_shape}; retrain the model "
                "with the current data and split."
            )

        model_metrics[model_name] = {}
        for split_name, users, item_ids, actual in (
            ("validation", val_users, val_movies, val_actual),
            ("test", test_users, test_movies, test_actual),
        ):
            result = evaluate_model(model, users, item_ids, actual)
            model_metrics[model_name][split_name] = {
                "ratings": int(len(actual)),
                "rmse": float(result["rmse"]),
                "mae": float(result["mae"]),
            }
            if split_name == "test":
                model_predictions = test_known[
                    ["userId", "movieId", "rating"]
                ].copy()
                model_predictions["predicted_rating"] = result["predictions"]
                model_predictions = model_predictions.rename(
                    columns={"rating": "actual_rating"}
                )
                test_predictions[model_name] = model_predictions

    best_model_name = min(
        model_metrics,
        key=lambda name: model_metrics[name]["validation"]["rmse"],
    )
    best_model = loaded_models[best_model_name]
    best_predictions = test_predictions[best_model_name]
    training_rating_counts = train_df.groupby("userId").size()
    activity_metrics = analyze_user_activity(
        best_predictions, training_rating_counts
    )
    genre_metrics = analyze_genre_errors(best_predictions, movies)

    recommender = MovieRecommender(
        best_model,
        user_to_idx,
        movie_to_idx,
        train_df,
        movies,
    )
    recommendations_by_user: dict[int, list[int]] = {}
    recommendation_rows = []
    for user_id in sorted(test_known["userId"].unique()):
        recommendations = recommender.recommend_for_user(int(user_id), TOP_K)
        recommendation_rows.extend(
            {
                "userId": int(user_id),
                **row,
            }
            for row in recommendations.to_dict(orient="records")
        )
        recommendations_by_user[int(user_id)] = (
            recommendations["movieId"].astype(int).tolist()
        )
    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=["userId", "movieId", "title", "genres", "predicted_rating"],
    )
    relevant_items = {
        int(user_id): set(group.loc[group["rating"] >= RATING_THRESHOLD, "movieId"].astype(int))
        for user_id, group in test_known.groupby("userId")
    }
    ranking = ranking_metrics_at_k(
        recommendations_by_user, relevant_items, k=TOP_K
    )
    genre_exposure, popularity_metrics = analyze_recommendation_exposure(
        recommendations,
        pd.Series(list(movie_to_idx), dtype=int),
        train_df.groupby("movieId").size(),
        movies,
    )

    output_directory = root / "results"
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "split": {
            "random_state": 42,
            "requested_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "ratings": {
                "total": int(len(ratings)),
                "train": int(len(train_df)),
                "validation": int(len(val_df)),
                "test": int(len(test_df)),
                "validation_known_users_and_movies": int(len(val_known)),
                "test_known_users_and_movies": int(len(test_known)),
            },
            "training_users": int(len(user_to_idx)),
            "training_movies": int(len(movie_to_idx)),
        },
        "models": model_metrics,
        "selected_model": {
            "name": best_model_name,
            "selection_metric": "validation_rmse",
            "rating_relevance_threshold": RATING_THRESHOLD,
        },
        "ranking": ranking,
        "popularity_exposure": popularity_metrics,
        "activity_error_by_group": activity_metrics.to_dict(orient="records"),
        "genre_error": genre_metrics.to_dict(orient="records"),
        "genre_recommendation_exposure": genre_exposure.to_dict(orient="records"),
        "error_analysis": _build_error_analysis(best_predictions, movies),
    }
    metrics_path = output_directory / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    activity_metrics.to_csv(output_directory / "activity_error_by_group.csv", index=False)
    genre_metrics.to_csv(output_directory / "genre_error.csv", index=False)
    genre_exposure.to_csv(output_directory / "genre_recommendation_exposure.csv", index=False)
    recommendations.to_csv(output_directory / "top_recommendations.csv", index=False)
    best_predictions.to_csv(output_directory / "test_predictions.csv", index=False)
    write_bias_report(
        output_directory / "bias_report.txt",
        activity_metrics,
        genre_metrics,
        popularity_metrics,
        genre_exposure,
        ranking,
    )

    plot_rating_distribution(ratings, figure_directory / "rating_distribution.png")
    plot_model_comparison(model_metrics, figure_directory / "model_comparison.png")
    plot_actual_vs_predicted(
        best_predictions, figure_directory / "actual_vs_predicted.png"
    )
    plot_residual_distribution(
        best_predictions, figure_directory / "residual_distribution.png"
    )
    plot_activity_errors(
        activity_metrics, figure_directory / "activity_error.png"
    )
    plot_genre_errors(genre_metrics, figure_directory / "genre_error.png")
    plot_genre_exposure(
        genre_exposure, figure_directory / "genre_recommendation_exposure.png"
    )
    plot_popularity_exposure(
        popularity_metrics, figure_directory / "popularity_exposure.png"
    )

    return metrics


def main() -> None:
    metrics = run_evaluation()
    print(
        json.dumps(
            {
                "selected_model": metrics["selected_model"],
                "models": metrics["models"],
                "ranking": metrics["ranking"],
                "outputs": [
                    "results/evaluation_metrics.json",
                    "results/bias_report.txt",
                    "results/figures/",
                ],
            },
            indent=2,
            default=_json_value,
        )
    )


if __name__ == "__main__":
    main()
