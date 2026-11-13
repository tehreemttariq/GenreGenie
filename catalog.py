"""Movie catalog - loads and caches movies.csv and links.csv"""
import pandas as pd
from pathlib import Path
from functools import lru_cache
import re

DATA_DIR = Path("data/raw")

@lru_cache(maxsize=1)
def load_movies_catalog():
    """Load movies.csv and return as dict for fast lookup"""
    movies_path = DATA_DIR / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"movies.csv not found at {movies_path}\n"
            "Download from: https://grouplens.org/datasets/movielens/latest/\n"
            "Extract ml-latest-small/movies.csv to data/raw/"
        )
    
    df = pd.read_csv(movies_path)
    catalog = {}
    
    for _, row in df.iterrows():
        movie_id = int(row['movieId'])
        title = str(row['title'])
        genres = str(row['genres']).split('|') if pd.notna(row['genres']) else []
        
        # Extract year from title: "Toy Story (1995)" -> year=1995, title="Toy Story"
        year_match = re.search(r'\((\d{4})\)$', title)
        if year_match:
            year = year_match.group(1)
            title = title[:year_match.start()].strip()
        else:
            year = ""
        
        catalog[movie_id] = {
            "id": movie_id,
            "title": title,
            "year": year,
            "genres": genres
        }
    
    return catalog

@lru_cache(maxsize=1)
def load_tmdb_links():
    """Load links.csv to map MovieLens IDs to TMDB IDs"""
    links_path = DATA_DIR / "links.csv"
    if not links_path.exists():
        return {}
    
    df = pd.read_csv(links_path)
    links = {}
    
    for _, row in df.iterrows():
        movie_id = int(row['movieId'])
        tmdb_id = row.get('tmdbId')
        if pd.notna(tmdb_id):
            links[movie_id] = int(tmdb_id)
    
    return links

def get_movie(movie_id):
    """
    Get movie info by MovieLens ID
    
    Args:
        movie_id: MovieLens movie ID (integer)
    
    Returns:
        dict: {id, title, year, genres, tmdb_id}
    """
    try:
        catalog = load_movies_catalog()
        links = load_tmdb_links()
        
        if movie_id not in catalog:
            return None
        
        movie = catalog[movie_id].copy()
        movie['tmdb_id'] = links.get(movie_id)
        
        return movie
    except Exception as e:
        print(f"Error getting movie {movie_id}: {e}")
        return None

def get_all_movies():
    """Get all movies in catalog"""
    try:
        return load_movies_catalog()
    except:
        return {}

def search_movies(query):
    """Search movies by title (case-insensitive)"""
    try:
        catalog = load_movies_catalog()
        query_lower = query.lower()
        
        results = [m for m in catalog.values() 
                   if query_lower in m['title'].lower()]
        return results
    except:
        return []

def get_popular_movies(n=40):
    """
    Get most-rated movies (as proxy for popularity).
    
    Args:
        n: number of movies to return
    
    Returns:
        list of movie_ids
    """
    try:
        # Load ratings to find most-rated movies
        ratings_path = DATA_DIR / "ratings.csv"
        if not ratings_path.exists():
            return list(load_movies_catalog().keys())[:n]
        
        df = pd.read_csv(ratings_path)
        popular = df['movieId'].value_counts().head(n).index.tolist()
        
        return popular
    except:
        return list(load_movies_catalog().keys())[:n]

def get_movies_by_genre(genre):
    """Get movies of a specific genre"""
    try:
        catalog = load_movies_catalog()
        
        results = [m for m in catalog.values() 
                   if genre in m['genres']]
        return results
    except:
        return []
