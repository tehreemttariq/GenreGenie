import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmdb_catalog import TMDB_GENRE_NAMES, TMDB_MOVIE_ID_OFFSET

LILAC = '#C8A2C8'
PURPLE = '#8E5B9A'
DEEP_PURPLE = '#6A3F7A'
DARK_PURPLE = '#4A2E5D'
PURPLE_CMAP = sns.color_palette(
    ['#F1EEFF', '#C4B5FD', '#8B7BE8', '#5946B2', '#30245F'],
    as_cmap=True,
)

sns.set_style('whitegrid', {'grid.color': LILAC})
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.edgecolor'] = DARK_PURPLE
plt.rcParams['axes.labelcolor'] = DARK_PURPLE
plt.rcParams['axes.titlecolor'] = DARK_PURPLE
plt.rcParams['xtick.color'] = DARK_PURPLE
plt.rcParams['ytick.color'] = DARK_PURPLE


def load_cleaned_data():
    print("loading data")

    if os.path.exists('data/cleaned/movies_cleaned.csv'):
        movies_df = pd.read_csv('data/cleaned/movies_cleaned.csv')
        ratings_df = pd.read_csv('data/cleaned/ratings_cleaned.csv')
    else:
        print("cleaned files not found, loading raw data")
        movies_df = pd.read_csv('data/raw/movies.csv')
        ratings_df = pd.read_csv('data/raw/ratings.csv')

    return movies_df, ratings_df


def engineer_user_features(ratings_df):
    print("\ncreating user features")

    user_features = ratings_df.groupby('userId').agg({
        'rating': ['count', 'mean', 'std'],
        'movieId': 'nunique'
    }).reset_index()

    user_features.columns = [
        'userId', 'rating_count', 'rating_mean',
        'rating_std', 'unique_movies'
    ]

    user_features['rating_std'] = user_features['rating_std'].fillna(0)

    scaler = MinMaxScaler()
    user_features['rating_count_normalized'] = scaler.fit_transform(
        user_features[['rating_count']]
    )
    user_features['rating_mean_normalized'] = scaler.fit_transform(
        user_features[['rating_mean']]
    )

    print(f"users: {len(user_features)}")
    return user_features


def engineer_movie_features(movies_df, ratings_df):
    print("\ncreating movie features")

    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean', 'std']
    }).reset_index()

    movie_stats.columns = [
        'movieId', 'num_ratings', 'avg_rating', 'rating_std'
    ]

    movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)

    movies_df = movies_df.merge(movie_stats, on='movieId', how='left')

    movies_df['num_ratings'] = movies_df['num_ratings'].fillna(0)
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(
        movies_df['avg_rating'].mean()
    )
    movies_df['rating_std'] = movies_df['rating_std'].fillna(0)

    # get the first genre
    movies_df['primary_genre'] = movies_df['genres'].apply(
        lambda x: x.split('|')[0] if pd.notna(x) else 'Unknown'
    )

    # convert genres to numbers
    encoder = LabelEncoder()
    movies_df['genre_encoded'] = encoder.fit_transform(
        movies_df['primary_genre']
    )

    # scale numerical features
    scaler = StandardScaler()
    movies_df['num_ratings_scaled'] = scaler.fit_transform(
        movies_df[['num_ratings']]
    )
    movies_df['avg_rating_scaled'] = scaler.fit_transform(
        movies_df[['avg_rating']]
    )

    print(f"movies: {len(movies_df)}")
    print(f"genres: {movies_df['primary_genre'].nunique()}")

    return movies_df


def load_catalog_eda_data(movies_df):
    """Load MovieLens and current TMDB catalog rows for catalog-level figures."""
    catalog = movies_df[["movieId", "genres"]].copy()
    catalog["source"] = "MovieLens"
    popularity = (
        movies_df[["movieId", "num_ratings", "avg_rating"]]
        .rename(columns={"num_ratings": "popularity", "avg_rating": "rating"})
        .assign(source="MovieLens")
    )

    tmdb_rows = []
    raw_paths = (
        Path("data/raw/full_load_2005_2014.json"),
        Path("data/raw/full_load.json"),
        Path("data/raw/tmdb_incremental_catalog.json"),
    )
    for path in raw_paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
        records = payload.get("results", payload.get("movies", []))
        for record in records:
            try:
                tmdb_id = int(record.get("id", 0))
                vote_average = float(record.get("vote_average", 0.0))
                vote_count = int(record.get("vote_count", 0))
            except (TypeError, ValueError):
                continue
            if tmdb_id <= 0 or vote_count < 0:
                continue
            genres = record.get("genres") or [
                {"id": genre_id}
                for genre_id in record.get("genre_ids", [])
            ]
            genre_names = [
                genre.get("name") or TMDB_GENRE_NAMES.get(genre.get("id"))
                for genre in genres
                if isinstance(genre, dict)
            ]
            tmdb_rows.append(
                {
                    "movieId": TMDB_MOVIE_ID_OFFSET + tmdb_id,
                    "genres": "|".join(
                        dict.fromkeys(
                            genre for genre in genre_names if genre
                        )
                    ),
                    "popularity": vote_count,
                    "rating": vote_average / 2,
                    "source": "TMDB",
                }
            )

    tmdb = pd.DataFrame(tmdb_rows).drop_duplicates("movieId")
    links_path = Path("data/raw/links.csv")
    if not tmdb.empty and links_path.is_file():
        linked_tmdb_ids = set(
            pd.to_numeric(
                pd.read_csv(links_path, usecols=["tmdbId"])["tmdbId"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        tmdb = tmdb[
            ~tmdb["movieId"].sub(TMDB_MOVIE_ID_OFFSET).isin(linked_tmdb_ids)
        ]
    if not tmdb.empty:
        catalog = pd.concat(
            [catalog, tmdb[["movieId", "genres", "source"]]],
            ignore_index=True,
        )
        popularity = pd.concat(
            [popularity, tmdb[["movieId", "popularity", "rating", "source"]]],
            ignore_index=True,
        )
    return catalog.merge(
        popularity[["movieId", "popularity", "rating"]],
        on="movieId",
        how="left",
    )


def generate_eda_plots(movies_df, ratings_df, user_features, catalog_data):
    print("\ngenerating plots")

    os.makedirs('docs/figures', exist_ok=True)

    # 1. rating distribution
    plt.figure(figsize=(10, 6))
    ratings_df['rating'].hist(bins=20, color=LILAC, edgecolor='white')
    plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('docs/figures/rating_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. genre distribution
    plt.figure(figsize=(12, 6))
    catalog_genres = (
        catalog_data["genres"]
        .fillna("")
        .str.split("|")
        .explode()
        .loc[lambda values: values.ne("")]
    )
    catalog_genres.value_counts().head(10).plot(
        kind='bar',
        color=[LILAC, PURPLE, DEEP_PURPLE, DARK_PURPLE] * 3,
        edgecolor='white',
    )
    plt.title(
        'Top 10 Genres in MovieLens + TMDB Catalog',
        fontsize=16,
        fontweight='bold',
    )
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('docs/figures/genre_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. user activity
    plt.figure(figsize=(10, 6))
    user_features['rating_count'].hist(
        bins=50, edgecolor='white', color=PURPLE
    )
    plt.title('User Activity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.xlim(0, user_features['rating_count'].quantile(0.95))
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('docs/figures/user_activity.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. correlation heatmap
    features = ['num_ratings', 'avg_rating', 'rating_std', 'genre_encoded']
    corr = movies_df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, cmap=PURPLE_CMAP, center=0,
        square=True, linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('docs/figures/correlation_matrix.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. popularity vs rating
    plt.figure(figsize=(10, 6))
    for source, color in (
        ("MovieLens", DEEP_PURPLE),
        ("TMDB", PURPLE),
    ):
        source_data = catalog_data[
            (catalog_data["source"] == source)
            & (catalog_data["popularity"] > 0)
        ]
        if source_data.empty:
            continue
        sample = source_data.sample(
            min(1000, len(source_data)),
            random_state=42,
        )
        plt.scatter(
            sample['popularity'],
            sample['rating'],
            alpha=0.35,
            color=color,
            label=source,
        )
    plt.title('Catalog Popularity vs Average Rating',
              fontsize=16, fontweight='bold')
    plt.xlabel('Rating Count / TMDB Votes', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.xlim(0, catalog_data['popularity'].quantile(0.95))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('docs/figures/popularity_vs_rating.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("all 5 plots saved to docs/figures/")


def save_engineered_features(movies_df, user_features):
    print("\nsaving features")

    os.makedirs('data/processed', exist_ok=True)

    movies_df.to_csv('data/processed/movies_features.csv', index=False)
    user_features.to_csv('data/processed/user_features.csv', index=False)

    print("saved movies_features.csv")
    print("saved user_features.csv")


def main():
    # move to project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    os.chdir(project_root)

    print("=" * 40)
    print("starting feature engineering")
    print("=" * 40)

    movies_df, ratings_df = load_cleaned_data()

    user_features = engineer_user_features(ratings_df)
    movies_df = engineer_movie_features(movies_df, ratings_df)
    catalog_data = load_catalog_eda_data(movies_df)

    generate_eda_plots(movies_df, ratings_df, user_features, catalog_data)
    save_engineered_features(movies_df, user_features)

    print("\nfeature engineering complete")
    print(f"user features: {len(user_features)}")
    print(f"movie features: {len(movies_df)}")
    print("plots generated: 5")
    print("min-max scaling applied")
    print("standard scaling applied")
    print("label encoding applied")


if __name__ == "__main__":
    main()