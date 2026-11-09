"""Evaluation metrics for predicted movie ratings."""

from collections.abc import Mapping, Sequence
from typing import Hashable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RatingPredictor(Protocol):
    """Interface required by :func:`evaluate_model`."""

    def predict(self, user_ids: ArrayLike, movie_ids: ArrayLike) -> ArrayLike:
        """Predict ratings for user/movie index pairs."""
        ...


def _validate_ratings(
    true_ratings: ArrayLike, predicted_ratings: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true_values = np.asarray(true_ratings, dtype=float)
    predicted_values = np.asarray(predicted_ratings, dtype=float)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Ratings must be one-dimensional arrays.")
    if true_values.size == 0:
        raise ValueError("Cannot evaluate an empty set of ratings.")
    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "True and predicted ratings must have the same number of values."
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Ratings must contain only finite values.")

    return true_values, predicted_values


def calculate_rmse(true_ratings: ArrayLike, predicted_ratings: ArrayLike) -> float:
    """Return root mean squared error for true and predicted ratings."""
    true_values, predicted_values = _validate_ratings(
        true_ratings, predicted_ratings
    )
    return float(np.sqrt(mean_squared_error(true_values, predicted_values)))


def calculate_mae(true_ratings: ArrayLike, predicted_ratings: ArrayLike) -> float:
    """Return mean absolute error for true and predicted ratings."""
    true_values, predicted_values = _validate_ratings(
        true_ratings, predicted_ratings
    )
    return float(mean_absolute_error(true_values, predicted_values))


def evaluate_predictions(
    true_ratings: ArrayLike, predicted_ratings: ArrayLike
) -> dict[str, float]:
    """Calculate RMSE and MAE for predicted ratings."""
    true_values, predicted_values = _validate_ratings(
        true_ratings, predicted_ratings
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
        "mae": float(mean_absolute_error(true_values, predicted_values)),
    }


def evaluate_model(
    model: RatingPredictor,
    user_ids: ArrayLike,
    movie_ids: ArrayLike,
    true_ratings: ArrayLike,
    rating_range: tuple[float, float] = (0.5, 5.0),
) -> dict[str, float | NDArray[np.float64]]:
    """Predict and evaluate ratings, clipping predictions to the rating range."""
    minimum_rating, maximum_rating = rating_range
    if not np.isfinite([minimum_rating, maximum_rating]).all():
        raise ValueError("Rating range bounds must be finite.")
    if minimum_rating >= maximum_rating:
        raise ValueError("The minimum rating must be less than the maximum rating.")

    user_values = np.asarray(user_ids)
    movie_values = np.asarray(movie_ids)
    if user_values.ndim != 1 or movie_values.ndim != 1:
        raise ValueError("User and movie IDs must be one-dimensional arrays.")
    if user_values.shape != movie_values.shape:
        raise ValueError("User and movie IDs must have the same number of values.")

    predictions = np.asarray(model.predict(user_values, movie_values), dtype=float)
    predictions = np.clip(predictions, minimum_rating, maximum_rating)
    metrics = evaluate_predictions(true_ratings, predictions)

    return {**metrics, "predictions": predictions}


def ranking_metrics_at_k(
    recommendations: Mapping[Hashable, Sequence[Hashable]],
    relevant_items: Mapping[Hashable, set[Hashable]],
    k: int = 10,
) -> dict[str, float | int]:
    """Calculate macro Precision@K, Recall@K, and F1@K.

    Users with no relevant held-out items are excluded from the macro average.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not relevant_items:
        raise ValueError("At least one user's relevant items is required.")

    precision_scores = []
    recall_scores = []
    for user_id, relevant in relevant_items.items():
        if not relevant:
            continue
        top_k = list(recommendations.get(user_id, ()))[:k]
        hits = len(set(top_k).intersection(relevant))
        precision_scores.append(hits / k)
        recall_scores.append(hits / len(relevant))

    if not precision_scores:
        return {
            "k": k,
            "users_evaluated": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    precision = float(np.mean(precision_scores))
    recall = float(np.mean(recall_scores))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "k": k,
        "users_evaluated": len(precision_scores),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
