"""GenreGenie - Movie Recommendation App with Auth & Fold-In"""
import streamlit as st
from model_loader import load_model
from auth import signup, login
from db import init_db, add_rating, get_user_ratings, set_user_preferences, get_user_preferences
from foldin import fold_in_user, recommend_new_user
from quiz import get_quiz_movies, get_movie_info
from results import display_recommendation_grid
from catalog import get_movie
from functools import lru_cache
import requests

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

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

# Initialize
init_db()
model_data, mappings = load_model()

if model_data is None:
    st.error("ERROR: Could not load model")
    st.stop()

# Streamlit config
st.set_page_config(page_title="GenreGenie", page_icon="🎬", layout="wide")

# CSS for dark theme
st.markdown("""
    <style>
    .main { background-color: #0d0d0d; color: #fff; }
    .stButton>button { background-color: #ff6b35; color: white; }
    </style>
""", unsafe_allow_html=True)

# Session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "quiz_genres_set" not in st.session_state:
    st.session_state.quiz_genres_set = False
if "quiz_genres" not in st.session_state:
    st.session_state.quiz_genres = []
if "quiz_ratings" not in st.session_state:
    st.session_state.quiz_ratings = {}

# ==================== AUTHENTICATION ====================
if st.session_state.user_id is None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Sign Up")
        signup_user = st.text_input("Username", key="signup_user")
        signup_pass = st.text_input("Password", type="password", key="signup_pass")
        signup_pass_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm")
        
        if st.button("Sign Up", key="btn_signup"):
            success, result = signup(signup_user, signup_pass, signup_pass_confirm)
            if success:
                st.success("Account created! Logging in...")
                st.session_state.user_id = result
                st.rerun()
            else:
                st.error(result)
    
    with col2:
        st.subheader("Log In")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Log In", key="btn_login"):
            success, result = login(login_user, login_pass)
            if success:
                st.session_state.user_id = result
                st.success(f"Welcome back!")
                st.rerun()
            else:
                st.error(result)

# ==================== MAIN APP ====================
else:
    user_id = st.session_state.user_id
    
    # Sidebar
    with st.sidebar:
        st.title("🎬 GenreGenie")
        page = st.radio("Navigate", ["Home", "Taste Quiz", "Recommendations", "Profile"])
        
        if st.button("Log Out"):
            st.session_state.user_id = None
            st.rerun()
    
    # ==================== HOME ====================
    if page == "Home":
        st.title("🎬 Welcome to GenreGenie")
        st.write("""
        Get personalized movie recommendations using AI.
        
        **How it works:**
        1. Take the Taste Quiz (rate 5+ movies you know)
        2. Get personalized recommendations
        3. Rate more movies to improve suggestions
        """)
        
        # Check if user completed quiz
        genres, quiz_completed = get_user_preferences(user_id)
        if quiz_completed:
            st.success("✓ Quiz completed!")
            st.write(f"Favorite genres: {', '.join(genres)}")
        else:
            st.info("⚠ Complete the Taste Quiz to get started")
    
    # ==================== TASTE QUIZ ====================
    elif page == "Taste Quiz":
        st.title("🎯 Taste Quiz")
        st.write("Rate movies you've seen. You need 5+ ratings to get recommendations.")
        
        # Tab 1: Select Genres
        # Tab 2: Rate Movies (one at a time)
        tab1, tab2 = st.tabs(["Step 1: Genres", "Step 2: Movies"])
        
        with tab1:
            st.subheader("Select Your Favorite Genres")
            st.write("Choose at least 1 genre you enjoy:")
            
            all_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", 
                          "Animation", "Adventure", "Sci-Fi", "Fantasy", "Crime"]
            
            favorite_genres = st.multiselect(
                "Genres:",
                all_genres,
                help="You'll see movies in these genres"
            )
            
            if favorite_genres:
                st.success(f"Selected: {', '.join(favorite_genres)}")
                st.info("👉 Go to Step 2 to rate movies")
            else:
                st.info("Select genres first →")
        
        with tab2:
            if not st.session_state.get('quiz_genres_set'):
                if favorite_genres and st.button("Save Genres & Start Rating"):
                    st.session_state.quiz_genres_set = True
                    st.session_state.quiz_genres = favorite_genres
                    st.rerun()
                else:
                    st.warning("Please select genres in Step 1 first")
                    st.stop()
            
            st.subheader("Rate Movies")
            
            # Initialize session state
            if "quiz_ratings" not in st.session_state:
                st.session_state.quiz_ratings = get_user_ratings(user_id) if user_id else {}
            if "skipped_movies" not in st.session_state:
                st.session_state.skipped_movies = set()
            
            quiz_ratings = st.session_state.quiz_ratings
            num_rated = len(quiz_ratings)
            
            # Progress bar
            st.progress(min(num_rated / 5, 1.0))
            st.write(f"**{num_rated} of 5 rated**")
            
            if num_rated >= 5:
                st.success("✓ You have enough ratings!")
                
                col_finish, col_continue = st.columns([1, 2])
                with col_finish:
                    if st.button("Finish Quiz & Get Recommendations", type="primary"):
                        # Save to database
                        for mid, rating in quiz_ratings.items():
                            add_rating(user_id, mid, rating)
                        set_user_preferences(user_id, favorite_genres, quiz_completed=True)
                        st.success("Quiz saved!")
                        st.rerun()
                
                with col_continue:
                    if st.button("Rate More Movies"):
                        # Continue to next movie
                        st.session_state.skipped_movies.clear()
                        st.rerun()
            else:
                st.info(f"Rate {5 - num_rated} more movies to continue")
            
            st.divider()
            
            # Get movies to show (40 popular, not yet rated or skipped by this user)
            quiz_movies = get_quiz_movies(n=40)
            available_movies = [m for m in quiz_movies 
                               if m not in quiz_ratings and m not in st.session_state.skipped_movies]
            
            if not available_movies:
                st.info("No more movies to rate. You can rate more movies above or finish the quiz.")
                if num_rated >= 5:
                    if st.button("Finish Quiz", type="primary"):
                        for mid, rating in quiz_ratings.items():
                            add_rating(user_id, mid, rating)
                        set_user_preferences(user_id, favorite_genres, quiz_completed=True)
                        st.success("Quiz saved!")
                        st.rerun()
            else:
                # Show ONE movie at a time
                current_movie_id = available_movies[0]
                movie_info = get_movie_info(current_movie_id)
                
                if not movie_info:
                    st.error("Could not load movie")
                    st.stop()
                
                # Display movie card
                col_poster, col_info = st.columns([1, 2])
                
                with col_poster:
                    # Try to get TMDB poster
                    tmdb_data = None
                    if movie_info.get('tmdb_id'):
                        tmdb_data = get_tmdb_movie(movie_info['tmdb_id'])
                    
                    if tmdb_data and tmdb_data.get('poster_path'):
                        st.image(
                            f"https://image.tmdb.org/t/p/w342{tmdb_data['poster_path']}",
                            use_column_width=True
                        )
                    else:
                        st.markdown(
                            f"<div style='background:#333; height:400px; display:flex; align-items:center; "
                            f"justify-content:center; border-radius:8px;'>"
                            f"<p style='color:#999; text-align:center; font-size:18px;'>🎬<br>{movie_info['title']}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                
                with col_info:
                    st.markdown(f"### {movie_info['title']}")
                    if movie_info.get('year'):
                        st.caption(f"📅 {movie_info['year']}")
                    if movie_info.get('genres'):
                        st.caption(f"🎭 {' • '.join(movie_info['genres'][:3])}")
                    
                    # Show synopsis if available
                    if tmdb_data and tmdb_data.get('overview'):
                        st.write(tmdb_data['overview'][:200] + "...")
                    
                    st.divider()
                    
                    # Rating controls
                    st.write("**How do you rate this movie?**")
                    
                    col_rating, col_actions = st.columns([2, 2])
                    
                    with col_rating:
                        # Star rating (1-5)
                        rating = st.feedback("stars", key=f"rating_{current_movie_id}")
                    
                    with col_actions:
                        # Skip / Haven't seen buttons
                        col_skip, col_nosee = st.columns(2)
                        with col_skip:
                            skip = st.button("Skip", key=f"skip_{current_movie_id}")
                        with col_nosee:
                            seen_but_no_rating = st.button("Haven't Seen", key=f"noseen_{current_movie_id}")
                    
                    # Process rating
                    if rating:
                        st.session_state.quiz_ratings[current_movie_id] = rating
                        add_rating(user_id, current_movie_id, rating)
                        st.success(f"Saved: {movie_info['title']} - {'⭐' * rating}")
                        st.rerun()
                    
                    # Skip or Haven't Seen - move to next movie
                    if skip:
                        st.session_state.skipped_movies.add(current_movie_id)
                        st.rerun()
                    
                    if seen_but_no_rating:
                        st.session_state.skipped_movies.add(current_movie_id)
                        st.rerun()

    
    # ==================== RECOMMENDATIONS ====================
    elif page == "Recommendations":
        st.title("🎬 Your Recommendations")
        
        user_ratings = get_user_ratings(user_id)
        
        if len(user_ratings) < 5:
            st.warning(f"Need at least 5 ratings. You have {len(user_ratings)}. Complete the Taste Quiz first.")
        else:
            st.write(f"Based on your {len(user_ratings)} ratings...")
            
            # Fold-in and recommend
            folded_vector = fold_in_user(user_ratings, model_data, mappings, lambda_reg=1.0)
            
            if folded_vector is None:
                st.error("ERROR: Could not generate recommendations")
            else:
                n_recs = st.slider("Number of recommendations:", 5, 50, 20)
                recommendations = recommend_new_user(folded_vector, set(user_ratings.keys()), model_data, mappings, n_recs=n_recs)
                
                if recommendations:
                    display_recommendation_grid(recommendations, show_actions=True)
                else:
                    st.warning("No recommendations found")
    
    # ==================== PROFILE ====================
    elif page == "Profile":
        st.title("👤 Profile")
        
        st.write(f"**User ID:** {user_id}")
        
        user_ratings = get_user_ratings(user_id)
        genres, quiz_completed = get_user_preferences(user_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ratings", len(user_ratings))
        with col2:
            st.metric("Genres", len(genres))
        with col3:
            st.metric("Quiz Status", "✓ Done" if quiz_completed else "⚠ Pending")
