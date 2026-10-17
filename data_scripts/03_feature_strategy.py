"""
Phase 2: Feature Engineering & EDA
Implements normalization, scaling, encoding, and generates EDA visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import os

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_cleaned_data():
    """
    Load cleaned datasets
    """
    print("Loading cleaned data...")
    
    # Check if cleaned files exist, otherwise use original
    if os.path.exists('movies_cleaned.csv'):
        movies_df = pd.read_csv('movies_cleaned.csv')
        ratings_df = pd.read_csv('ratings_cleaned.csv')
    else:
        print("⚠ Cleaned files not found. Run 02_data_cleaning.py first.")
        print("Loading original files instead...")
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
    
    return movies_df, ratings_df

def engineer_user_features(ratings_df):
    """
    Create user-level features from ratings
    """
    print("\n=== Engineering User Features ===")
    
    user_features = ratings_df.groupby('userId').agg({
        'rating': ['count', 'mean', 'std'],
        'movieId': 'nunique'
    }).reset_index()
    
    user_features.columns = ['userId', 'rating_count', 'rating_mean', 'rating_std', 'unique_movies']
    
    # Fill NaN std with 0 (users who rated only once)
    user_features['rating_std'] = user_features['rating_std'].fillna(0)
    
    # Normalize features using Min-Max scaling (0-1 range)
    scaler = MinMaxScaler()
    user_features['rating_count_normalized'] = scaler.fit_transform(user_features[['rating_count']])
    user_features['rating_mean_normalized'] = scaler.fit_transform(user_features[['rating_mean']])
    
    print(f"✓ Created features for {len(user_features)} users")
    
    return user_features

def engineer_movie_features(movies_df, ratings_df):
    """
    Create movie-level features
    """
    print("\n=== Engineering Movie Features ===")
    
    # Aggregate rating statistics per movie
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['count', 'mean', 'std']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'num_ratings', 'avg_rating', 'rating_std']
    movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
    
    # Merge with movie metadata
    movies_df = movies_df.merge(movie_stats, on='movieId', how='left')
    
    # Fill movies with no ratings
    movies_df['num_ratings'] = movies_df['num_ratings'].fillna(0)
    movies_df['avg_rating'] = movies_df['avg_rating'].fillna(movies_df['avg_rating'].mean())
    movies_df['rating_std'] = movies_df['rating_std'].fillna(0)
    
    # One-hot encode genres (simplified: take first genre only for demo)
    movies_df['primary_genre'] = movies_df['genres'].apply(lambda x: x.split('|')[0] if pd.notna(x) else 'Unknown')
    
    # Label encoding for primary genre
    le = LabelEncoder()
    movies_df['genre_encoded'] = le.fit_transform(movies_df['primary_genre'])
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    movies_df['num_ratings_scaled'] = scaler.fit_transform(movies_df[['num_ratings']])
    movies_df['avg_rating_scaled'] = scaler.fit_transform(movies_df[['avg_rating']])
    
    print(f"✓ Engineered features for {len(movies_df)} movies")
    print(f"✓ Unique genres: {movies_df['primary_genre'].nunique()}")
    
    return movies_df

def generate_eda_plots(movies_df, ratings_df, user_features):
    """
    Generate EDA visualizations for the report
    """
    print("\n=== Generating EDA Visualizations ===")
    
    # Create output directory for figures
    os.makedirs('../docs/figures', exist_ok=True)
    
    # 1. Rating Distribution
    plt.figure(figsize=(10, 6))
    ratings_df['rating'].hist(bins=20, edgecolor='black')
    plt.title('Distribution of Movie Ratings', fontsize=16, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('../docs/figures/rating_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: rating_distribution.png")
    plt.close()
    
    # 2. Genre Distribution
    plt.figure(figsize=(12, 6))
    genre_counts = movies_df['primary_genre'].value_counts().head(10)
    genre_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Top 10 Movie Genres', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../docs/figures/genre_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: genre_distribution.png")
    plt.close()
    
    # 3. User Activity Distribution
    plt.figure(figsize=(10, 6))
    user_features['rating_count'].hist(bins=50, edgecolor='black', color='coral')
    plt.title('User Activity Distribution (Number of Ratings per User)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ratings', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.xlim(0, user_features['rating_count'].quantile(0.95))  # Focus on 95th percentile
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('../docs/figures/user_activity.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: user_activity.png")
    plt.close()
    
    # 4. Correlation Heatmap (Movie Features)
    plt.figure(figsize=(10, 8))
    corr_features = ['num_ratings', 'avg_rating', 'rating_std', 'genre_encoded']
    correlation_matrix = movies_df[corr_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Movies)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../docs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()
    
    # 5. Average Rating vs Number of Ratings (Scatter)
    plt.figure(figsize=(10, 6))
    sample = movies_df[movies_df['num_ratings'] > 0].sample(min(1000, len(movies_df)))
    plt.scatter(sample['num_ratings'], sample['avg_rating'], alpha=0.5, color='green')
    plt.title('Movie Popularity vs Average Rating', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ratings (Popularity)', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.xlim(0, sample['num_ratings'].quantile(0.95))
    plt.grid(alpha=0.3)
    plt.savefig('../docs/figures/popularity_vs_rating.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: popularity_vs_rating.png")
    plt.close()
    
    print("\n✓ All EDA plots saved to ../docs/figures/")

def save_engineered_features(movies_df, user_features):
    """
    Save engineered features for future model training
    """
    print("\n=== Saving Engineered Features ===")
    
    movies_df.to_csv('movies_features.csv', index=False)
    user_features.to_csv('user_features.csv', index=False)
    
    print("✓ Saved: movies_features.csv")
    print("✓ Saved: user_features.csv")

def main():
    """
    Main feature engineering pipeline
    """
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*50)
    print("Starting Feature Engineering & EDA...")
    print("="*50)
    
    # Load cleaned data
    movies_df, ratings_df = load_cleaned_data()
    
    # Engineer features
    user_features = engineer_user_features(ratings_df)
    movies_df = engineer_movie_features(movies_df, ratings_df)
    
    # Generate EDA visualizations
    generate_eda_plots(movies_df, ratings_df, user_features)
    
    # Save engineered features
    save_engineered_features(movies_df, user_features)
    
    print("\n" + "="*50)
    print("✓ Feature Engineering Complete!")
    print("="*50)
    print("\nSummary:")
    print(f"  - User features: {len(user_features)} records")
    print(f"  - Movie features: {len(movies_df)} records")
    print(f"  - EDA plots: 5 visualizations generated")
    print(f"  - Normalization: Min-Max Scaling applied")
    print(f"  - Standardization: Standard Scaling applied")
    print(f"  - Encoding: Label Encoding for genres")

if __name__ == "__main__":
    main()
