"""Results page - display recommendations with TMDB data"""
import streamlit as st
from pathlib import Path
import requests
from functools import lru_cache
from catalog import get_movie

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

@lru_cache(maxsize=1000)
def get_tmdb_movie(tmdb_id):
    """Get movie details from TMDB by TMDB ID"""
    if not TMDB_API_KEY or not tmdb_id:
        return None
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        resp = requests.get(url, params=params, timeout=5)
        
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

def display_movie_card(movie_id, score, show_actions=True):
    """Display a single movie card"""
    # Get movie from catalog
    movie_info = get_movie(movie_id)
    if not movie_info:
        return
    
    title = movie_info['title']
    year = movie_info['year']
    genres = movie_info['genres']
    tmdb_id = movie_info.get('tmdb_id')
    
    # Try to get TMDB data
    tmdb_data = get_tmdb_movie(tmdb_id) if tmdb_id else None
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Poster or placeholder
            if tmdb_data and tmdb_data.get('poster_path'):
                poster_url = f"{TMDB_IMAGE_BASE}{tmdb_data['poster_path']}"
                st.image(poster_url, use_column_width=True)
            else:
                st.markdown(
                    f"<div style='background:#333; height:200px; display:flex; align-items:center; justify-content:center; border-radius:8px;'>"
                    f"<p style='color:#999; text-align:center;'>🎬<br>{title}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            # Title & year
            st.markdown(f"**{title}**")
            if year:
                st.caption(f"📅 {year}")
            
            # Genres
            if genres:
                genre_text = " • ".join(genres[:3])
                st.caption(f"🎭 {genre_text}")
            
            # Rating & match score
            col_a, col_b = st.columns(2)
            with col_a:
                if tmdb_data:
                    rating = tmdb_data.get('vote_average', 0) / 2
                    st.metric("Rating", f"{rating:.1f}⭐")
                else:
                    st.metric("Rating", "N/A")
            
            with col_b:
                match_pct = min(100, max(0, score * 10))
                st.metric("Match", f"{match_pct:.0f}%")
            
            # Synopsis
            if tmdb_data and tmdb_data.get('overview'):
                with st.expander("📖 Synopsis"):
                    st.write(tmdb_data['overview'])
            
            # Action buttons
            if show_actions:
                col_x, col_y = st.columns(2)
                with col_x:
                    if st.button("⭐ Rate", key=f"rate_{movie_id}", use_container_width=True):
                        st.session_state.rating_movie_id = movie_id
                
                with col_y:
                    if st.button("📋 Watchlist", key=f"watch_{movie_id}", use_container_width=True):
                        st.success("Added!")

def display_recommendation_grid(recommendations, show_actions=True):
    """Display recommendations in a grid"""
    if not recommendations:
        st.warning("No recommendations found")
        return
    
    st.success(f"✨ Found {len(recommendations)} recommendations")
    
    # Display in 3-column grid
    cols = st.columns(3, gap="medium")
    
    for i, (movie_id, score) in enumerate(recommendations):
        with cols[i % 3]:
            display_movie_card(movie_id, score, show_actions)

