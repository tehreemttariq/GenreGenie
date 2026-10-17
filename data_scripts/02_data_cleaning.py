"""
Phase 2: Data Cleaning Script
Handles missing values, duplicates, and outliers
"""

import pandas as pd
import numpy as np
import os

def clean_movies_data(movies_df):
    """
    Clean movies dataset
    """
    print("\n=== Cleaning Movies Dataset ===")
    print(f"Initial records: {len(movies_df)}")
    
    # Check for missing values
    missing = movies_df.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    # Remove duplicates based on movieId
    initial_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=['movieId'], keep='first')
    print(f"Duplicates removed: {initial_count - len(movies_df)}")
    
    # Handle missing genres (replace with 'Unknown')
    movies_df['genres'] = movies_df['genres'].fillna('Unknown')
    
    # Split genres into list for future processing
    movies_df['genre_list'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df['genre_count'] = movies_df['genre_list'].apply(len)
    
    print(f"Final records: {len(movies_df)}")
    
    return movies_df

def clean_ratings_data(ratings_df):
    """
    Clean ratings dataset and handle outliers
    """
    print("\n=== Cleaning Ratings Dataset ===")
    print(f"Initial records: {len(ratings_df)}")
    
    # Check for missing values
    missing = ratings_df.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    # Remove rows with missing ratings
    ratings_df = ratings_df.dropna(subset=['rating'])
    
    # Remove duplicates (same user rating same movie multiple times - keep last)
    initial_count = len(ratings_df)
    ratings_df = ratings_df.sort_values('timestamp')
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    print(f"Duplicates removed: {initial_count - len(ratings_df)}")
    
    # Handle outliers: ratings should be between 0.5 and 5.0
    outlier_count = len(ratings_df[(ratings_df['rating'] < 0.5) | (ratings_df['rating'] > 5.0)])
    ratings_df = ratings_df[(ratings_df['rating'] >= 0.5) & (ratings_df['rating'] <= 5.0)]
    print(f"Rating outliers removed: {outlier_count}")
    
    # Create additional features
    ratings_df['rating_normalized'] = ratings_df['rating'] / 5.0  # Normalize to 0-1
    
    print(f"Final records: {len(ratings_df)}")
    
    return ratings_df

def clean_links_data(links_df):
    """
    Clean links dataset
    """
    print("\n=== Cleaning Links Dataset ===")
    print(f"Initial records: {len(links_df)}")
    
    # Check for missing values
    missing = links_df.isnull().sum()
    print(f"\nMissing values:\n{missing}")
    
    # Remove duplicates
    initial_count = len(links_df)
    links_df = links_df.drop_duplicates(subset=['movieId'], keep='first')
    print(f"Duplicates removed: {initial_count - len(links_df)}")
    
    # Remove rows where tmdbId is missing (critical for our project)
    links_df = links_df.dropna(subset=['tmdbId'])
    
    # Convert tmdbId to integer
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    
    print(f"Final records: {len(links_df)}")
    
    return links_df

def generate_cleaning_report(movies_df, ratings_df, links_df):
    """
    Generate summary statistics after cleaning
    """
    print("\n" + "="*50)
    print("=== DATA CLEANING SUMMARY ===")
    print("="*50)
    
    print(f"\n✓ Movies: {len(movies_df)} records")
    print(f"✓ Ratings: {len(ratings_df)} records")
    print(f"✓ Links: {len(links_df)} records")
    
    print(f"\n✓ Unique users: {ratings_df['userId'].nunique()}")
    print(f"✓ Unique movies: {movies_df['movieId'].nunique()}")
    print(f"✓ Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")
    print(f"✓ Average rating: {ratings_df['rating'].mean():.2f}")
    
    print("\n✓ All datasets cleaned successfully!")

def main():
    """
    Main cleaning pipeline
    """
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("="*50)
    print("Starting Data Cleaning Process...")
    print("="*50)
    
    # Load raw data
    movies_df = pd.read_csv('data/raw/movies.csv')
    ratings_df = pd.read_csv('data/raw/ratings.csv')
    links_df = pd.read_csv('data/raw/links.csv')
    
    # Clean each dataset
    movies_clean = clean_movies_data(movies_df)
    ratings_clean = clean_ratings_data(ratings_df)
    links_clean = clean_links_data(links_df)
    
    # Save cleaned data
    movies_clean.to_csv('data/cleaned/movies_cleaned.csv', index=False)
    ratings_clean.to_csv('data/cleaned/ratings_cleaned.csv', index=False)
    links_clean.to_csv('data/cleaned/links_cleaned.csv', index=False)
    
    print("\n✓ Cleaned datasets saved to data/cleaned/:")
    print("  - movies_cleaned.csv")
    print("  - ratings_cleaned.csv")
    print("  - links_cleaned.csv")
    
    # Generate report
    generate_cleaning_report(movies_clean, ratings_clean, links_clean)

if __name__ == "__main__":
    main()
