"""User-activity, genre, and popularity analyses for recommendation results."""

from pathlib import Path

import numpy as np
import pandas as pd


def _validate_predictions(predictions: pd.DataFrame) -> None:
    required = {"userId", "movieId", "actual_rating", "predicted_rating"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing required columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Cannot analyze an empty prediction set.")
    values = predictions[["actual_rating", "predicted_rating"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions and actual ratings must be finite.")


def analyze_user_activity(
    predictions: pd.DataFrame,
    training_rating_counts: pd.Series,
    n_groups: int = 3,
) -> pd.DataFrame:
    """Compare errors for users divided into balanced training-activity cohorts."""
    _validate_predictions(predictions)
    if n_groups < 1:
        raise ValueError("n_groups must be at least 1.")

    user_activity = pd.DataFrame(
        {
            "userId": predictions["userId"].drop_duplicates(),
        }
    )
    user_activity["training_ratings"] = (
        user_activity["userId"].map(training_rating_counts).fillna(0).astype(int)
    )
    user_activity = user_activity.sort_values(
        ["training_ratings", "userId"], kind="stable"
    ).reset_index(drop=True)
    group_count = min(n_groups, len(user_activity))
    group_names = ["Low activity", "Medium activity", "High activity"]
    if group_count > len(group_names):
        group_names.extend(
            f"Activity group {index + 1}"
            for index in range(len(group_names), group_count)
        )
    user_activity["activity_group"] = ""
    for group_index, user_indices in enumerate(
        np.array_split(np.arange(len(user_activity)), group_count)
    ):
        user_activity.loc[user_indices, "activity_group"] = group_names[group_index]

    rows = predictions.merge(
        user_activity[["userId", "training_ratings", "activity_group"]],
        on="userId",
        how="left",
        validate="many_to_one",
    )
    rows["error"] = rows["predicted_rating"] - rows["actual_rating"]
    rows["absolute_error"] = rows["error"].abs()
    metrics = (
        rows.groupby("activity_group", sort=False)
        .agg(
            users=("userId", "nunique"),
            ratings=("movieId", "size"),
            mean_training_ratings=("training_ratings", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mae=("absolute_error", "mean"),
        )
        .reset_index()
    )
    group_order = {name: index for index, name in enumerate(group_names)}
    metrics["_group_order"] = metrics["activity_group"].map(group_order)
    return metrics.sort_values("_group_order").drop(columns="_group_order").reset_index(
        drop=True
    )


def analyze_genre_errors(
    predictions: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Compare rating prediction errors across the genres of test movies."""
    _validate_predictions(predictions)
    if not {"movieId", "genres"}.issubset(movies.columns):
        raise ValueError("movies must contain movieId and genres columns.")

    genre_lookup = movies[["movieId", "genres"]].drop_duplicates("movieId")
    rows = predictions.merge(
        genre_lookup, on="movieId", how="left", validate="many_to_one"
    )
    rows["genre"] = rows["genres"].fillna("").str.split("|", regex=False)
    rows = rows.explode("genre")
    rows["genre"] = rows["genre"].str.strip()
    rows = rows[rows["genre"].notna() & rows["genre"].ne("") & rows["genre"].ne("(no genres listed)")].copy()
    if rows.empty:
        return pd.DataFrame(columns=["genre", "ratings", "rmse", "mae"])

    rows["error"] = rows["predicted_rating"] - rows["actual_rating"]
    rows["absolute_error"] = rows["error"].abs()
    return (
        rows.groupby("genre")
        .agg(
            ratings=("movieId", "size"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mae=("absolute_error", "mean"),
        )
        .sort_values(["ratings", "genre"], ascending=[False, True])
        .reset_index()
    )


def analyze_recommendation_exposure(
    recommendations: pd.DataFrame,
    catalog_movie_ids: pd.Series,
    training_rating_counts: pd.Series,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare recommendation exposure with catalog genre and popularity shares."""
    if recommendations.empty:
        raise ValueError("Cannot analyze an empty recommendation set.")
    if not {"movieId", "genres"}.issubset(recommendations.columns):
        raise ValueError("recommendations must contain movieId and genres columns.")
    if len(catalog_movie_ids) == 0:
        raise ValueError("The recommendation catalog must not be empty.")

    catalog = pd.DataFrame({"movieId": catalog_movie_ids.drop_duplicates()})
    catalog["training_ratings"] = catalog["movieId"].map(training_rating_counts).fillna(0)
    popular_count = max(1, int(np.ceil(len(catalog) * 0.2)))
    popular_movie_ids = set(
        catalog.sort_values(
            ["training_ratings", "movieId"], ascending=[False, True], kind="stable"
        )
        .head(popular_count)["movieId"]
    )
    exposed_popular = recommendations["movieId"].isin(popular_movie_ids)
    popularity = {
        "catalog_movies": int(len(catalog)),
        "popular_catalog_movies": int(popular_count),
        "recommendation_count": int(len(recommendations)),
        "popular_recommendation_count": int(exposed_popular.sum()),
        "popular_catalog_share": float(popular_count / len(catalog)),
        "popular_recommendation_share": float(exposed_popular.mean()),
    }

    catalog_genres = movies[["movieId", "genres"]].drop_duplicates("movieId").copy()
    catalog_genres["genre"] = catalog_genres["genres"].fillna("").str.split("|", regex=False)
    catalog_genres = catalog_genres.explode("genre")
    catalog_genres["genre"] = catalog_genres["genre"].str.strip()
    catalog_genres = catalog_genres[
        catalog_genres["genre"].notna()
        & catalog_genres["genre"].ne("")
        & catalog_genres["genre"].ne("(no genres listed)")
    ]
    catalog_genre_counts = catalog_genres.groupby("genre")["movieId"].nunique()

    exposed_genres = recommendations[["movieId", "genres"]].copy()
    exposed_genres["genre"] = exposed_genres["genres"].fillna("").str.split("|", regex=False)
    exposed_genres = exposed_genres.explode("genre")
    exposed_genres["genre"] = exposed_genres["genre"].str.strip()
    exposed_genres = exposed_genres[
        exposed_genres["genre"].notna()
        & exposed_genres["genre"].ne("")
        & exposed_genres["genre"].ne("(no genres listed)")
    ]
    genre_exposure = (
        exposed_genres.groupby("genre")
        .agg(recommendations=("movieId", "size"))
        .join(catalog_genre_counts.rename("catalog_movies"), how="outer")
        .fillna(0)
    )
    genre_exposure["recommendation_share"] = (
        genre_exposure["recommendations"] / len(recommendations)
    )
    total_catalog_memberships = int(catalog_genre_counts.sum())
    genre_exposure["catalog_share"] = (
        genre_exposure["catalog_movies"] / total_catalog_memberships
        if total_catalog_memberships
        else 0.0
    )
    genre_exposure = genre_exposure.reset_index().sort_values(
        "recommendations", ascending=False, kind="stable"
    )
    return genre_exposure, popularity


def write_bias_report(
    output_path: str | Path,
    activity_metrics: pd.DataFrame,
    genre_metrics: pd.DataFrame,
    popularity_metrics: dict[str, float | int],
    genre_exposure: pd.DataFrame,
    ranking_metrics: dict[str, float | int],
) -> Path:
    """Write observed fairness and exposure findings with dataset limitations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if activity_metrics.empty:
        activity_summary = "No activity-group results were available."
    else:
        best_activity = activity_metrics.loc[activity_metrics["rmse"].idxmin()]
        worst_activity = activity_metrics.loc[activity_metrics["rmse"].idxmax()]
        activity_summary = (
            f"Lowest RMSE: {best_activity['activity_group']} "
            f"({best_activity['rmse']:.4f}, {int(best_activity['ratings'])} ratings); "
            f"highest RMSE: {worst_activity['activity_group']} "
            f"({worst_activity['rmse']:.4f}, {int(worst_activity['ratings'])} ratings)."
        )

    if genre_metrics.empty:
        genre_summary = "No genre-level errors could be calculated."
    else:
        supported_genres = genre_metrics[genre_metrics["ratings"] >= 30]
        if supported_genres.empty:
            genre_summary = (
                "No genre had at least 30 held-out ratings; see the CSV/JSON tables "
                "for descriptive results with limited support."
            )
        else:
            best_genre = supported_genres.loc[supported_genres["mae"].idxmin()]
            worst_genre = supported_genres.loc[supported_genres["mae"].idxmax()]
            genre_summary = (
                f"Among genres with at least 30 held-out ratings, lowest MAE: "
                f"{best_genre['genre']} ({best_genre['mae']:.4f}, "
                f"{int(best_genre['ratings'])} ratings); highest MAE: "
                f"{worst_genre['genre']} ({worst_genre['mae']:.4f}, "
                f"{int(worst_genre['ratings'])} ratings)."
            )

    popular_delta = (
        float(popularity_metrics["popular_recommendation_share"])
        - float(popularity_metrics["popular_catalog_share"])
    )
    if popular_delta > 0:
        popularity_summary = (
            f"The top 20% most-rated catalog movies received "
            f"{popularity_metrics['popular_recommendation_share']:.1%} of recommendation "
            f"slots versus {popularity_metrics['popular_catalog_share']:.1%} of catalog "
            f"items (difference: +{popular_delta:.1%})."
        )
    else:
        popularity_summary = (
            f"The top 20% most-rated catalog movies received "
            f"{popularity_metrics['popular_recommendation_share']:.1%} of recommendation "
            f"slots versus {popularity_metrics['popular_catalog_share']:.1%} of catalog "
            f"items (difference: {popular_delta:.1%})."
        )

    if genre_exposure.empty:
        genre_exposure_summary = "No genre recommendation-exposure results were available."
    else:
        exposure = genre_exposure.copy()
        exposure["share_gap"] = (
            exposure["recommendation_share"] - exposure["catalog_share"]
        )
        largest_gap = exposure.loc[exposure["share_gap"].abs().idxmax()]
        genre_exposure_summary = (
            f"Largest genre share difference: {largest_gap['genre']} "
            f"({largest_gap['recommendation_share']:.1%} of recommendation slots "
            f"versus {largest_gap['catalog_share']:.1%} of catalog genre memberships; "
            f"gap {largest_gap['share_gap']:+.1%})."
        )

    report = "\n".join(
        [
            "GenreGenie recommendation bias and fairness analysis",
            "====================================================",
            "",
            "Scope: held-out rating errors by training-activity cohort and movie genre, "
            "plus top-K recommendation exposure by genre and training popularity.",
            "No demographic or other protected-group attributes are present in the "
            "available MovieLens data; this report is not a demographic fairness audit.",
            "",
            "User activity",
            f"- {activity_summary}",
            "",
            "Genre rating error",
            f"- {genre_summary}",
            "",
            "Popularity exposure",
            f"- {popularity_summary}",
            "",
            "Genre recommendation exposure",
            f"- {genre_exposure_summary}",
            "",
            "Top-K ranking evaluation",
            f"- Precision@{ranking_metrics['k']}: "
            f"{ranking_metrics['precision']:.4f}; Recall@{ranking_metrics['k']}: "
            f"{ranking_metrics['recall']:.4f}; F1@{ranking_metrics['k']}: "
            f"{ranking_metrics['f1']:.4f} across "
            f"{ranking_metrics['users_evaluated']} users with at least one held-out "
            "relevant rating (rating >= 4.0).",
            "",
            "Interpretation",
            "- These are descriptive dataset-level comparisons, not evidence that "
            "activity, genre, or popularity caused the observed differences.",
            "- Genre memberships overlap, so genre exposure shares are not mutually "
            "exclusive.",
            "- Results depend on the deterministic random rating-level split and the "
            "selected trained model.",
            "",
        ]
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path
