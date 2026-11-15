"""Fit the production SVD model on every available MovieLens rating."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import create_user_item_matrix, split_data
from evaluate import evaluate_model
from model import MatrixFactorizationSVD


RATINGS_PATH = PROJECT_ROOT / "data" / "cleaned" / "ratings_cleaned.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "models"
FACTORS = 50


def _encoded_rows(
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known = frame[
        frame["userId"].isin(user_to_idx)
        & frame["movieId"].isin(movie_to_idx)
    ]
    return (
        known["userId"].map(user_to_idx).to_numpy(dtype=int),
        known["movieId"].map(movie_to_idx).to_numpy(dtype=int),
        known["rating"].to_numpy(dtype=float),
    )


def retrain_full_data() -> dict[str, float | int]:
    if not RATINGS_PATH.is_file():
        raise FileNotFoundError(f"Cleaned ratings not found: {RATINGS_PATH}")
    ratings = pd.read_csv(RATINGS_PATH)
    required = {"userId", "movieId", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Cleaned ratings are missing columns: {sorted(missing)}")
    if ratings.empty or ratings[list(required)].isna().any().any():
        raise ValueError("Cleaned ratings must contain non-empty, complete rating rows.")

    train, validation, test = split_data(ratings)
    evaluation_matrix, evaluation_users, evaluation_movies = create_user_item_matrix(
        train
    )
    evaluation_model = MatrixFactorizationSVD(n_factors=FACTORS).fit(
        evaluation_matrix
    )
    del evaluation_matrix
    evaluation_model.user_to_idx = evaluation_users
    evaluation_model.movie_to_idx = evaluation_movies
    test_users, test_movies, test_ratings = _encoded_rows(
        test, evaluation_users, evaluation_movies
    )
    if len(test_ratings) == 0:
        raise ValueError("The test split has no ratings with known users and movies.")
    test_metrics = evaluate_model(
        evaluation_model, test_users, test_movies, test_ratings
    )

    full_matrix, full_users, full_movies = create_user_item_matrix(ratings)
    production_model = MatrixFactorizationSVD(n_factors=FACTORS).fit(full_matrix)
    del full_matrix
    production_model.user_to_idx = full_users
    production_model.movie_to_idx = full_movies

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    evaluation_model.save(str(MODEL_DIRECTORY / "svd_model_evaluation.pkl"))
    production_model.save(str(MODEL_DIRECTORY / "svd_model.pkl"))

    summary: dict[str, float | int] = {
        "ratings_used_for_final_fit": int(len(ratings)),
        "users": int(len(full_users)),
        "movies": int(len(full_movies)),
        "factors": FACTORS,
        "test_ratings_with_known_ids": int(len(test_ratings)),
        "holdout_test_rmse": float(test_metrics["rmse"]),
        "holdout_test_mae": float(test_metrics["mae"]),
        "global_mean": float(production_model.global_mean),
    }
    print("Saved full-data production model and split-only evaluation model.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(
        "TMDB community averages remain catalog metadata; they are not user-level "
        "ratings and were not used as SVD training samples."
    )
    return summary


if __name__ == "__main__":
    retrain_full_data()
