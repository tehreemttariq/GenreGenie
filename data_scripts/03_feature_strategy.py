import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


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


def generate_eda_plots(movies_df, ratings_df, user_features):
    print("\ngenerating plots")

    os.makedirs('docs/figures', exist_ok=True)

    # 1. rating distribution
    plt.figure(figsize=(10, 6))
    ratings_df['rating'].hist(bins=20, edgecolor='black')
    plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('docs/figures/rating_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. genre distribution
    plt.figure(figsize=(12, 6))
    movies_df['primary_genre'].value_counts().head(10).plot(
        kind='bar', color='skyblue', edgecolor='black'
    )
    plt.title('Top 10 Movie Genres', fontsize=16, fontweight='bold')
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
        bins=50, edgecolor='black', color='coral'
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
        corr, annot=True, cmap='coolwarm', center=0,
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
    rated_movies = movies_df[movies_df['num_ratings'] > 0]
    sample = rated_movies.sample(min(1000, len(rated_movies)))

    plt.scatter(
        sample['num_ratings'], sample['avg_rating'],
        alpha=0.5, color='green'
    )
    plt.title('Movie Popularity vs Average Rating',
              fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.xlim(0, sample['num_ratings'].quantile(0.95))
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

    generate_eda_plots(movies_df, ratings_df, user_features)
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