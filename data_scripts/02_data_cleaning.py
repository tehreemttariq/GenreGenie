
import pandas as pd
import os


def clean_movies_data(movies_df):
    print("\ncleaning movies")
    print(f"initial records: {len(movies_df)}")

    # check missing values
    print(movies_df.isnull().sum())

    # remove duplicates
    old_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=['movieId'], keep='first')
    print(f"duplicates removed: {old_count - len(movies_df)}")

    # fill missing genres
    movies_df['genres'] = movies_df['genres'].fillna('Unknown')

    # split genres and count them
    movies_df['genre_list'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df['genre_count'] = movies_df['genre_list'].apply(len)

    print(f"final records: {len(movies_df)}")
    return movies_df


def clean_ratings_data(ratings_df):
    print("\ncleaning ratings")
    print(f"initial records: {len(ratings_df)}")

    # check missing values
    print(ratings_df.isnull().sum())

    # remove missing ratings
    ratings_df = ratings_df.dropna(subset=['rating'])

    # keep the latest rating for each user and movie
    old_count = len(ratings_df)
    ratings_df = ratings_df.sort_values('timestamp')
    ratings_df = ratings_df.drop_duplicates(
        subset=['userId', 'movieId'], keep='last'
    )
    print(f"duplicates removed: {old_count - len(ratings_df)}")

    # remove invalid ratings
    outliers = ratings_df[
        (ratings_df['rating'] < 0.5) | (ratings_df['rating'] > 5.0)
    ]
    print(f"rating outliers removed: {len(outliers)}")

    ratings_df = ratings_df[
        (ratings_df['rating'] >= 0.5) & (ratings_df['rating'] <= 5.0)
    ]

    # normalize ratings to 0-1
    ratings_df['rating_normalized'] = ratings_df['rating'] / 5.0

    print(f"final records: {len(ratings_df)}")
    return ratings_df


def clean_links_data(links_df):
    print("\ncleaning links")
    print(f"initial records: {len(links_df)}")

    # check missing values
    print(links_df.isnull().sum())

    # remove duplicates
    old_count = len(links_df)
    links_df = links_df.drop_duplicates(subset=['movieId'], keep='first')
    print(f"duplicates removed: {old_count - len(links_df)}")

    # remove rows without tmdbId
    links_df = links_df.dropna(subset=['tmdbId'])

    # convert tmdbId to integer
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)

    print(f"final records: {len(links_df)}")
    return links_df


def generate_cleaning_report(movies_df, ratings_df, links_df):
    print("\n" + "=" * 40)
    print("data cleaning summary")
    print("=" * 40)

    print(f"\nmovies: {len(movies_df)} records")
    print(f"ratings: {len(ratings_df)} records")
    print(f"links: {len(links_df)} records")

    print(f"\nunique users: {ratings_df['userId'].nunique()}")
    print(f"unique movies: {movies_df['movieId'].nunique()}")
    print(
        f"rating range: [{ratings_df['rating'].min()}, "
        f"{ratings_df['rating'].max()}]"
    )
    print(f"average rating: {ratings_df['rating'].mean():.2f}")

    print("\nall datasets cleaned successfully")


def main():
    # move to project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    os.chdir(project_root)

    print("=" * 40)
    print("starting data cleaning")
    print("=" * 40)

    # load raw data
    movies_df = pd.read_csv('data/raw/movies.csv')
    ratings_df = pd.read_csv('data/raw/ratings.csv')
    links_df = pd.read_csv('data/raw/links.csv')

    # clean data
    movies_clean = clean_movies_data(movies_df)
    ratings_clean = clean_ratings_data(ratings_df)
    links_clean = clean_links_data(links_df)

    # save cleaned data
    movies_clean.to_csv('data/cleaned/movies_cleaned.csv', index=False)
    ratings_clean.to_csv('data/cleaned/ratings_cleaned.csv', index=False)
    links_clean.to_csv('data/cleaned/links_cleaned.csv', index=False)

    print("\ncleaned files saved to data/cleaned/")
    print("movies_cleaned.csv")
    print("ratings_cleaned.csv")
    print("links_cleaned.csv")

    # print summary
    generate_cleaning_report(movies_clean, ratings_clean, links_clean)


if __name__ == "__main__":
    main()
