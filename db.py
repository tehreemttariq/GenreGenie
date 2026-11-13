"""Database setup and operations"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("genregenie.db")

def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User ratings
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, movie_id)
        )
    ''')
    
    # Watchlist
    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, movie_id)
        )
    ''')
    
    # User preferences (genres)
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            favorite_genres TEXT,
            quiz_completed BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username, password_hash):
    """Add new user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                  (username, password_hash))
        user_id = c.lastrowid
        c.execute('INSERT INTO user_preferences (user_id) VALUES (?)', (user_id,))
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_user_by_username(username):
    """Get user by username"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, password_hash FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result

def add_rating(user_id, movie_id, rating):
    """Add/update user rating"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO user_ratings (user_id, movie_id, rating)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating
        ''', (user_id, movie_id, rating))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding rating: {e}")
        conn.close()
        return False

def get_user_ratings(user_id):
    """Get all ratings for a user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT movie_id, rating FROM user_ratings WHERE user_id = ?', (user_id,))
    ratings = dict(c.fetchall())
    conn.close()
    return ratings

def set_user_preferences(user_id, favorite_genres, quiz_completed=True):
    """Save user genre preferences"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    genres_json = json.dumps(favorite_genres)
    c.execute('''
        UPDATE user_preferences 
        SET favorite_genres = ?, quiz_completed = ?
        WHERE user_id = ?
    ''', (genres_json, quiz_completed, user_id))
    conn.commit()
    conn.close()

def get_user_preferences(user_id):
    """Get user genre preferences"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT favorite_genres, quiz_completed FROM user_preferences WHERE user_id = ?', 
              (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        genres = json.loads(result[0]) if result[0] else []
        return genres, bool(result[1])
    return [], False

def add_to_watchlist(user_id, movie_id):
    """Add movie to watchlist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO watchlist (user_id, movie_id) VALUES (?, ?)',
                  (user_id, movie_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def get_watchlist(user_id):
    """Get user's watchlist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT movie_id FROM watchlist WHERE user_id = ? ORDER BY added_at DESC',
              (user_id,))
    movies = [row[0] for row in c.fetchall()]
    conn.close()
    return movies

def remove_from_watchlist(user_id, movie_id):
    """Remove movie from watchlist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?',
              (user_id, movie_id))
    conn.commit()
    conn.close()
