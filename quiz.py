"""Taste quiz - popular movies for initial ratings"""
from catalog import get_popular_movies, get_movie

def get_quiz_movies(n=40):
    """
    Return most-popular movies for quiz (from ratings count).
    Returns movie IDs, which will be resolved to titles via catalog.
    """
    try:
        popular = get_popular_movies(n)
        return popular
    except:
        # Fallback to hardcoded well-known MovieLens IDs
        return [1, 260, 318, 356, 593, 2571, 589, 1210, 1196, 480, 
                590, 527, 1198, 912, 858, 50, 541, 2959, 296, 2762,
                1, 110, 150, 231, 288, 367, 541, 648, 750, 919,
                1084, 1196, 1210, 1259, 1270, 1405, 1566, 1682, 1891, 1907]

def get_movie_info(movie_id):
    """Get movie info using catalog"""
    return get_movie(movie_id) or {
        "title": f"Movie {movie_id}",
        "year": "",
        "genres": []
    }

def validate_quiz_ratings(ratings_dict):
    """Check user gave at least 5 ratings"""
    return len(ratings_dict) >= 5

def validate_quiz_genres(genres):
    """Check user selected at least 1 genre"""
    return len(genres) > 0

