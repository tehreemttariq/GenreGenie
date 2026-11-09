"""Movie recommendation inference helpers."""

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from dataset import split_data
from model import CollaborativeFilteringBaseline, MatrixFactorizationSVD


class RatingPredictor(Protocol):
    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for encoded user/movie IDs."""
        ...


class MovieRecommender:
    """Recommend unseen movies for users represented in a fitted model."""

    def __init__(
        self,
        model: RatingPredictor,
        user_to_idx: dict[int, int],
        movie_to_idx: dict[int, int],
        training_ratings: pd.DataFrame,
        movies: pd.DataFrame,
        rating_range: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        required_rating_columns = {"userId", "movieId"}
        required_movie_columns = {"movieId", "title", "genres"}
        if not required_rating_columns.issubset(training_ratings.columns):
            raise ValueError("training_ratings must contain userId and movieId.")
        if not required_movie_columns.issubset(movies.columns):
            raise ValueError("movies must contain movieId, title, and genres.")
        if not 0 < rating_range[0] < rating_range[1]:
            raise ValueError("rating_range must contain increasing positive bounds.")
        if not user_to_idx or not movie_to_idx:
            raise ValueError("User and movie mappings must not be empty.")

        self.model = model
        self.user_to_idx = {int(key): int(value) for key, value in user_to_idx.items()}
        self.movie_to_idx = {int(key): int(value) for key, value in movie_to_idx.items()}
        self.training_ratings = training_ratings
        self.movies = movies.drop_duplicates("movieId").copy()
        self.rating_range = rating_range

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return the user's highest-scored unseen movies."""
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        try:
            user_id = int(user_id)
        except (TypeError, ValueError) as error:
            raise ValueError("user_id must be an integer MovieLens user ID.") from error
        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} was not present in the training data.")

        watched_movie_ids = set(
            self.training_ratings.loc[
                self.training_ratings["userId"] == user_id, "movieId"
            ].astype(int)
        )
        candidates = [
            movie_id
            for movie_id in self.movie_to_idx
            if movie_id not in watched_movie_ids
        ]
        if not candidates:
            return pd.DataFrame(
                columns=["movieId", "title", "genres", "predicted_rating"]
            )

        candidate_metadata = self.movies[
            self.movies["movieId"].isin(candidates)
        ].copy()
        candidate_metadata["movieId"] = candidate_metadata["movieId"].astype(int)
        candidate_metadata = candidate_metadata[
            candidate_metadata["movieId"].isin(self.movie_to_idx)
        ]
        if candidate_metadata.empty:
            return pd.DataFrame(
                columns=["movieId", "title", "genres", "predicted_rating"]
            )

        movie_ids = candidate_metadata["movieId"].to_numpy(dtype=int)
        user_indices = np.full(len(movie_ids), self.user_to_idx[user_id], dtype=int)
        movie_indices = np.fromiter(
            (self.movie_to_idx[movie_id] for movie_id in movie_ids),
            dtype=int,
            count=len(movie_ids),
        )
        predictions = np.asarray(
            self.model.predict(user_indices, movie_indices), dtype=float
        )
        if predictions.shape != movie_ids.shape or not np.isfinite(predictions).all():
            raise ValueError("The model returned invalid recommendation predictions.")
        candidate_metadata["_raw_prediction"] = predictions
        candidate_metadata["predicted_rating"] = np.clip(
            predictions, self.rating_range[0], self.rating_range[1]
        )
        candidate_metadata = candidate_metadata.sort_values(
            ["_raw_prediction", "movieId"],
            ascending=[False, True],
            kind="stable",
        )
        return candidate_metadata[
            ["movieId", "title", "genres", "predicted_rating"]
        ].head(top_n).reset_index(drop=True)


def load_recommender(
    model_name: str = "baseline",
    project_root: str | Path | None = None,
) -> MovieRecommender:
    """Load a saved SVD or baseline model with mappings from the training split."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    ratings = pd.read_csv(root / "data" / "cleaned" / "ratings_cleaned.csv")
    movies = pd.read_csv(root / "data" / "cleaned" / "movies_cleaned.csv")
    training_ratings, _, _ = split_data(ratings)

    user_to_idx = {
        int(user_id): index
        for index, user_id in enumerate(training_ratings["userId"].unique())
    }
    movie_to_idx = {
        int(movie_id): index
        for index, movie_id in enumerate(training_ratings["movieId"].unique())
    }
    model_loaders = {
        "baseline": (
            root / "models" / "baseline_model.pkl",
            CollaborativeFilteringBaseline.load,
        ),
        "svd": (root / "models" / "svd_model.pkl", MatrixFactorizationSVD.load),
    }
    if model_name not in model_loaders:
        raise ValueError("model_name must be either 'baseline' or 'svd'.")

    model_path, loader = model_loaders[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = loader(str(model_path))
    expected_shape = (len(user_to_idx), len(movie_to_idx))
    model_shape = (
        (len(model.user_factors), len(model.item_factors))
        if model_name == "svd"
        else tuple(model.ratings_matrix.shape)
    )
    if model_shape != expected_shape:
        raise ValueError(
            f"Saved {model_name} model shape {model_shape} does not match the "
            f"current training split {expected_shape}; retrain the model."
        )
    return MovieRecommender(
        model, user_to_idx, movie_to_idx, training_ratings, movies
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend unseen movies for a user.")
    parser.add_argument("--user-id", required=True, type=int)
    parser.add_argument("--model", choices=("baseline", "svd"), default="baseline")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    recommendations = load_recommender(args.model).recommend_for_user(
        args.user_id, args.top_n
    )
    if recommendations.empty:
        print(f"No unseen known movies are available for user {args.user_id}.")
    else:
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
