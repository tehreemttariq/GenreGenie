"""
Phase 2: Data Collection Script
Documents the data collection process using MovieLens dataset and TMDB API
"""

import pandas as pd
import json
import os

def load_movielens_data():
    """
    Load and analyze MovieLens dataset
    """
    print("="*60)
    print("=== MovieLens Data Collection Summary ===")
    print("="*60)
    
    # Load datasets
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    links_df = pd.read_csv('links.csv')
    
    print(f"\n✓ Movies dataset: {len(movies_df):,} records")
    print(f"✓ Ratings dataset: {len(ratings_df):,} records")
    print(f"✓ Links dataset: {len(links_df):,} records")
    
    print(f"\n✓ Unique users: {ratings_df['userId'].nunique():,}")
    print(f"✓ Unique movies rated: {ratings_df['movieId'].nunique():,}")
    print(f"✓ Average rating: {ratings_df['rating'].mean():.2f}/5.0")
    print(f"✓ Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")
    
    # Genre distribution
    all_genres = []
    for genres in movies_df['genres'].dropna():
        all_genres.extend(genres.split('|'))
    genre_counts = pd.Series(all_genres).value_counts()
    
    print(f"\n✓ Total unique genres: {len(genre_counts)}")
    print(f"✓ Top 5 genres:")
    for genre, count in genre_counts.head(5).items():
        print(f"   - {genre}: {count}")
    
    return movies_df, ratings_df, links_df

def load_tmdb_data():
    """
    Load and analyze TMDB enriched data (collected via API)
    """
    print("\n" + "="*60)
    print("=== TMDB API Data Collection Summary ===")
    print("="*60)
    
    # Check if TMDB data exists
    if os.path.exists('full_load_2005_2014.json'):
        print("\n✓ Found TMDB enriched data: full_load_2005_2014.json")
        
        with open('full_load_2005_2014.json', 'r', encoding='utf-8') as f:
            tmdb_data = json.load(f)
        
        print(f"✓ Collection date: {tmdb_data.get('pulled_at')}")
        print(f"✓ Year range: {tmdb_data.get('year_range')}")
        print(f"✓ Total movies: {tmdb_data.get('record_count'):,}")
        print(f"✓ Enriched with details: {tmdb_data.get('enriched')}")
        print(f"✓ Enrichment limit: {tmdb_data.get('enrich_limit'):,}")
        
        # Analyze sample of enriched data
        results = tmdb_data.get('results', [])
        if results:
            sample = results[0]
            print(f"\n✓ Enriched fields available:")
            enriched_fields = ['runtime', 'budget', 'revenue', 'genres', 
                             'top_cast', 'keywords', 'production_companies']
            for field in enriched_fields:
                if field in sample:
                    print(f"   - {field}")
        
        return tmdb_data
    else:
        print("\n⚠ TMDB enriched data not found (full_load_2005_2014.json)")
        print("Note: Data was collected using separate TMDB API script")
        return None

def generate_collection_report():
    """
    Generate a summary report of all collected data sources
    """
    print("\n" + "="*60)
    print("=== Data Collection Report ===")
    print("="*60)
    
    report = {
        'data_sources': [
            {
                'name': 'MovieLens',
                'files': ['movies.csv', 'ratings.csv', 'links.csv'],
                'description': 'User-movie ratings and metadata'
            },
            {
                'name': 'TMDB API',
                'files': ['full_load_2005_2014.json'],
                'description': 'Enriched movie details (2005-2014)',
                'collection_method': 'REST API with /discover/movie endpoint',
                'enrichment_fields': ['runtime', 'budget', 'revenue', 'genres', 
                                    'cast', 'keywords', 'production_companies']
            }
        ],
        'collection_methodology': {
            'MovieLens': 'Downloaded from GroupLens research dataset',
            'TMDB': 'API-based ETL pipeline with rate limiting and error handling'
        }
    }
    
    print("\n✓ Data Sources:")
    for source in report['data_sources']:
        print(f"\n  {source['name']}:")
        print(f"    Files: {', '.join(source['files'])}")
        print(f"    Description: {source['description']}")
    
    print("\n✓ Collection completed successfully!")
    print("✓ Ready for Phase 2: Data Cleaning\n")

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load and analyze MovieLens data
    movies_df, ratings_df, links_df = load_movielens_data()
    
    # Load and analyze TMDB data
    tmdb_data = load_tmdb_data()
    
    # Generate collection report
    generate_collection_report()
