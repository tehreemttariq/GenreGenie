"""GenreGenie: choice-driven movie discovery dashboard."""

from __future__ import annotations

import html
import inspect
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

from input_validation import (
    MAX_UPLOAD_BYTES,
    sanitize_prompt,
    validate_poster,
    validate_watchlist_csv,
)
from movie_metadata import fetch_movie_details_batch
from recommendation_engine import (
    detect_poster_movie,
    find_prompt_movie_ids,
    parse_preferences,
    prepare_catalog,
    rank_recommendations,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies_cleaned.csv"
RATINGS_PATH = PROJECT_ROOT / "data" / "cleaned" / "ratings_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "svd_model.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.json"
LOGGER = logging.getLogger(__name__)
GENRE_CHOICES = (
    ("💖", "Romance", "Discover Romance"),
    ("😂", "Comedy", "Discover Comedy"),
    ("🧠", "Sci-Fi", "Sci-Fi & Mind-Bending"),
    ("😱", "Thriller", "Thriller & Mystery"),
    ("🎭", "Drama", "Discover Drama"),
    ("🔥", "Action", "Action & Adventure"),
)
ERA_CHOICES = (
    "Any Era",
    "Before 1995",
    "1995–2005",
    "2006+",
)
SAFE_POSTER_HOSTS = {
    "upload.wikimedia.org",
    "commons.wikimedia.org",
    "static.tvmaze.com",
    "is1-ssl.mzstatic.com",
}


@st.cache_data(show_spinner=False)
def load_movie_catalog() -> pd.DataFrame:
    return prepare_catalog(pd.read_csv(MOVIES_PATH))


def _training_movie_indices() -> dict[int, int]:
    from sklearn.model_selection import train_test_split

    ratings = pd.read_csv(RATINGS_PATH)
    if not {"userId", "movieId", "rating"}.issubset(ratings.columns) or len(ratings) < 3:
        raise ValueError("Cleaned ratings data cannot be used to load the SVD model.")
    train_validation, _ = train_test_split(
        ratings, test_size=0.15, random_state=42
    )
    train, _ = train_test_split(
        train_validation, test_size=0.15 / 0.85, random_state=42
    )
    return {
        int(movie_id): index
        for index, movie_id in enumerate(train["movieId"].unique())
    }


@st.cache_resource(show_spinner="Loading GenreGenie's SVD engine...")
def load_regression_model() -> dict[str, Any]:
    try:
        with MODEL_PATH.open("rb") as model_file:
            model_data = pickle.load(model_file)
        movie_indices = _training_movie_indices()
        item_factors = np.asarray(model_data["item_factors"], dtype=float)
        global_mean = float(model_data["global_mean"])
        if (
            item_factors.ndim != 2
            or item_factors.shape[0] != len(movie_indices)
            or not np.isfinite(item_factors).all()
            or not np.isfinite(global_mean)
        ):
            raise ValueError("The saved SVD model does not match the cleaned dataset.")
        return {
            "available": True,
            "item_factors": item_factors,
            "global_mean": global_mean,
            "movie_indices": movie_indices,
        }
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        LOGGER.warning("SVD model unavailable; enabling fallback scores: %s", error)
        return {
            "available": False,
            "item_factors": None,
            "global_mean": 3.5,
            "movie_indices": {},
        }


@st.cache_data(show_spinner=False)
def load_model_rmse() -> float | None:
    try:
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        return float(metrics["models"]["SVD"]["test"]["rmse"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _rating_predictions(
    catalog: pd.DataFrame,
    model: dict[str, Any],
    prompt: str,
    mood_labels: list[str],
    selected_movie_ids: set[int],
    lambda_val: float,
) -> dict[int, float]:
    if not model["available"]:
        return {
            int(movie_id): 2.8 + ((int(movie_id) * 37) % 22) / 10
            for movie_id in catalog["movieId"]
        }

    factors = model["item_factors"]
    movie_indices = model["movie_indices"]
    profile = parse_preferences(prompt, mood_labels)
    prompt_movie_ids = set(find_prompt_movie_ids(prompt, catalog))
    explicit_seed_ids = selected_movie_ids | prompt_movie_ids
    seed_ids = set(explicit_seed_ids)
    if profile.genre_groups:
        matching_genre_movies: list[int] = []
        for row in catalog.itertuples(index=False):
            genres = {str(genre).casefold() for genre in row.genre_tags}
            groups_match = all(
                genres.intersection(genre.casefold() for genre in group)
                for group in profile.genre_groups
            )
            year = int(row.year) if pd.notna(row.year) else None
            year_match = (
                (profile.year_min is None or (year is not None and year >= profile.year_min))
                and (profile.year_max is None or (year is not None and year <= profile.year_max))
            )
            if groups_match and year_match and int(row.movieId) in movie_indices:
                matching_genre_movies.append(int(row.movieId))
        seed_ids.update(matching_genre_movies[:100])

    usable_seeds = [movie_id for movie_id in seed_ids if movie_id in movie_indices]
    if usable_seeds:
        seed_indices = np.asarray(
            [movie_indices[movie_id] for movie_id in usable_seeds], dtype=int
        )
        seed_factors = factors[seed_indices]
        targets = np.asarray(
            [4.7 if movie_id in explicit_seed_ids else 4.2 for movie_id in usable_seeds],
            dtype=float,
        )
        centered_targets = targets - model["global_mean"]
        regularizer = max(float(lambda_val), 0.05)
        user_vector = np.linalg.solve(
            seed_factors.T @ seed_factors
            + regularizer * np.eye(seed_factors.shape[1]),
            seed_factors.T @ centered_targets,
        )
    else:
        user_vector = np.mean(factors, axis=0)

    raw_scores = model["global_mean"] + factors @ user_vector
    return {
        int(movie_id): float(np.clip(raw_scores[index], 0.5, 5.0))
        for movie_id, index in movie_indices.items()
    }


def _safe_poster_url(value: Any) -> str:
    if isinstance(value, str) and value.startswith("data:image/svg+xml;base64,"):
        return value
    if not isinstance(value, str):
        return ""
    parsed = urlparse(value)
    if (
        parsed.scheme == "https"
        and parsed.hostname
        and (
            parsed.hostname in SAFE_POSTER_HOSTS
            or parsed.hostname.endswith(".mzstatic.com")
        )
    ):
        return value
    return ""


def _render_movie_card(movie: dict[str, Any], details: dict[str, Any]) -> bool:
    title = html.escape(str(movie["title"]))
    year = html.escape(str(movie["year"]) if movie.get("year") else "Year unknown")
    genres = "".join(
        f'<span class="genre-pill">{html.escape(str(genre))}</span>'
        for genre in movie["genres"][:4]
    )
    overview = html.escape(
        str(
            details.get("overview")
            or details.get("full_overview")
            or "Synopsis details are temporarily unavailable."
        )
    )
    reason = html.escape(str(movie["reason"]))
    poster_url = _safe_poster_url(details.get("poster_url"))
    poster = (
        f'<img class="movie-poster" src="{html.escape(poster_url, quote=True)}" '
        f'alt="Poster for {title}" loading="lazy">'
        if poster_url
        else '<div class="poster-placeholder"><span>GENRE<br>GENIE</span></div>'
    )
    st.markdown(
        f"""
        <article class="movie-card">
          <div class="poster-stage">
            {poster}
            <span class="match-badge poster-match">{movie["match_pct"]}% Match</span>
            <span class="rating-badge poster-rating">★ {movie["predicted_rating"]:.1f}</span>
          </div>
          <div class="movie-copy">
            <div class="movie-title-row">
              <h3>{title}</h3>
              <span class="year-badge">{year}</span>
            </div>
            <div class="genre-list">{genres}</div>
            <p class="movie-overview">{overview}</p>
            <div class="reason-tag">✦ {reason}</div>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    return st.button(
        "Full Synopsis ↗",
        key=f"full_synopsis_{int(movie['movie_id'])}",
        use_container_width=True,
    )


@st.dialog("Movie details", width="large")
def _show_movie_details(
    movie: dict[str, Any],
    details: dict[str, Any],
) -> None:
    poster_url = _safe_poster_url(details.get("poster_url"))
    if poster_url:
        st.markdown(
            f'<img class="dialog-poster" src="{html.escape(poster_url, quote=True)}" '
            f'alt="Poster for {html.escape(str(movie["formatted_title"]), quote=True)}">',
            unsafe_allow_html=True,
        )
    st.subheader(str(movie["formatted_title"]))
    st.write(" · ".join(movie["genres"]) if movie["genres"] else "General")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Match", f'{movie["match_pct"]}%')
    metric_columns[1].metric("SVD rating", f'★ {movie["predicted_rating"]:.1f} / 5')
    metric_columns[2].metric(
        "Release year", str(movie["year"]) if movie.get("year") else "Unknown"
    )
    breakdown = [
        f'Genre fit {movie["genre_score"]:.0%}'
        if movie.get("genre_score") is not None
        else None,
        f'Era fit {movie["era_score"]:.0%}'
        if movie.get("era_score") is not None
        else None,
        f'Predicted rating {movie["predicted_rating"]:.1f}/5',
    ]
    st.caption("Match breakdown · " + " · ".join(item for item in breakdown if item))
    st.markdown("#### Full synopsis")
    st.write(
        details.get("full_overview")
        or details.get("overview")
        or "Synopsis details are temporarily unavailable."
    )
    if st.button(
        "Find More Like This",
        key=f"find_more_like_{int(movie['movie_id'])}",
        type="primary",
        use_container_width=True,
    ):
        active_filters = dict(st.session_state.get("active_filters", {}))
        active_filters.update(
            {
                "genres": [],
                "era": "Any Era",
                "loved_movies": [int(movie["movie_id"])],
                "keywords": "",
            }
        )
        st.session_state["active_filters"] = active_filters
        st.session_state["page"] = "Recommendations"
        st.session_state["shuffle_seed"] = 0
        st.session_state.pop("selected_movie_detail", None)
        st.rerun()
    if st.button("Close", key=f"close_movie_detail_{int(movie['movie_id'])}"):
        st.session_state.pop("selected_movie_detail", None)
        st.rerun()


def _start_recommendations(catalog: pd.DataFrame) -> None:
    prompt = sanitize_prompt(st.session_state.get("custom_keywords", ""))
    selected_genre_values = set(st.session_state.get("selected_genres", []))
    selected_genres = [
        label
        for _, label, value in GENRE_CHOICES
        if value in selected_genre_values
    ]
    era = st.session_state.get("selected_era", "Any Era")
    selected_movies = [
        int(movie_id) for movie_id in st.session_state.get("loved_movies", [])[:3]
    ]
    upload = st.session_state.get("uploaded_file")
    if not prompt and not selected_genres and era == "Any Era" and not selected_movies and not upload:
        st.warning("Pick at least one vibe, era, movie, or reference before revealing your picks.")
        return

    seed_ids = set(selected_movies)
    excluded_ids: set[int] = set()
    detected: dict[str, Any] | None = None
    upload_error: str | None = None
    if upload is not None:
        payload = upload.getvalue()
        if len(payload) > MAX_UPLOAD_BYTES:
            upload_error = "Uploads must be smaller than 5 MB."
        elif upload.name.casefold().endswith(".csv"):
            watchlist, upload_error, _ = validate_watchlist_csv(payload)
            if watchlist is not None and upload_error is None:
                excluded_ids = set(watchlist["movieId"].astype(int).tolist())
        else:
            valid, upload_error = validate_poster(payload, upload.name)
            if valid:
                detection = detect_poster_movie(payload, upload.name, catalog)
                if detection:
                    movie_id, confidence, source = detection
                    row = catalog[catalog["movieId"] == movie_id].iloc[0]
                    detected = {
                        "movie_id": int(movie_id),
                        "title": row["formatted_title"],
                        "confidence": confidence,
                        "source": source,
                    }
                    seed_ids.add(int(movie_id))
                else:
                    upload_error = (
                        "Poster title not recognized. Choose a movie above or add "
                        "its title in custom keywords."
                    )

    if upload_error:
        st.error(upload_error)
        return
    recommendation_history = set(
        st.session_state.get("recommendation_history", [])
    )
    recommendation_history.update(
        st.session_state.get("last_recommendation_ids", [])
    )
    st.session_state["recommendation_history"] = sorted(recommendation_history)
    st.session_state["active_filters"] = {
        "genres": selected_genres,
        "era": era,
        "loved_movies": sorted(seed_ids),
        "keywords": prompt,
    }
    st.session_state["discovery_request"] = {
        "excluded_ids": excluded_ids,
        "detected": detected,
    }
    st.session_state["shuffle_seed"] = 0
    st.session_state["page"] = "Recommendations"
    st.rerun()


def _render_discover(catalog: pd.DataFrame, movie_labels: dict[int, str]) -> None:
    st.markdown(
        '<section class="hero"><div class="eyebrow">YOUR NEXT MOVIE NIGHT STARTS HERE</div>'
        '<h1>What kind of night is it?</h1>'
        '<p>Pick a vibe, choose an era, or tap movies you already love. No typing required.</p></section>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-heading"><span>01</span> Pick your genre / vibe</div>', unsafe_allow_html=True)
    st.caption("Choose one or more — every selected vibe is applied to your results.")
    selected_genres = set(st.session_state.get("selected_genres", []))
    genre_columns = st.columns(3)
    for index, (icon, label, value) in enumerate(GENRE_CHOICES):
        with genre_columns[index % 3]:
            with st.container(border=True):
                active = value in selected_genres
                st.markdown(
                    f'<div class="choice-copy"><span class="choice-icon">{icon}</span>'
                    f'<span>{html.escape(label)}</span></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Selected ✓" if active else "Choose vibe",
                    key=f"genre_choice_{index}",
                    type="primary" if active else "secondary",
                    on_click=_toggle_genre,
                    args=(value,),
                    use_container_width=True,
                )

    st.markdown('<div class="step-heading"><span>02</span> Pick an era</div>', unsafe_allow_html=True)
    st.radio(
        "Release era",
        options=ERA_CHOICES,
        horizontal=True,
        key="selected_era",
        label_visibility="collapsed",
    )

    with st.expander("Optional: movies you liked, poster / watchlist, or keywords", expanded=False):
        st.markdown(
            '<div class="optional-note">Use a reference to personalize the SVD profile. '
            'You can also leave this section untouched.</div>',
            unsafe_allow_html=True,
        )
        movie_column, upload_column, prompt_column = st.columns([1.2, 1, 1])
        with movie_column:
            st.multiselect(
                "Movies you already love (up to 3)",
                options=list(movie_labels),
                format_func=lambda movie_id: movie_labels.get(int(movie_id), str(movie_id)),
                max_selections=3,
                key="loved_movies",
                placeholder="Search movie titles",
            )
        with upload_column:
            st.file_uploader(
                "Drop a poster or watchlist CSV",
                type=["jpg", "jpeg", "png", "csv"],
                key="uploaded_file",
                help="JPG/PNG or cleaned-schema CSV, maximum 5 MB.",
            )
        with prompt_column:
            st.text_input(
                "Custom keywords (optional)",
                key="custom_keywords",
                max_chars=500,
                placeholder="funny, space adventure…",
            )

    if st.button(
        "Reveal My Movies →",
        key="find_movies",
        type="primary",
        use_container_width=True,
    ):
        _start_recommendations(catalog)


def _toggle_genre(value: str) -> None:
    selected = set(st.session_state.get("selected_genres", []))
    if value in selected:
        selected.remove(value)
    else:
        selected.add(value)
    st.session_state["selected_genres"] = [
        genre_value for _, _, genre_value in GENRE_CHOICES if genre_value in selected
    ]


def _render_recommendations(
    catalog: pd.DataFrame,
    model: dict[str, Any],
    lambda_value: float,
    minimum_rating: float,
    top_n: int,
) -> None:
    request = st.session_state.get("discovery_request", {})
    active_filters = st.session_state.get("active_filters")
    if not active_filters:
        legacy_request = st.session_state.get("discovery_request")
        if legacy_request:
            active_filters = {
                "genres": legacy_request.get("genres", []),
                "era": legacy_request.get("era", "Any Era"),
                "loved_movies": list(legacy_request.get("seed_ids", [])),
                "keywords": legacy_request.get("prompt", ""),
            }
    if not active_filters:
        st.info("Start by choosing a vibe or era on the Discover page.")
        if st.button("← Change Choices", key="empty_back"):
            st.session_state["page"] = "Discover"
            st.rerun()
        return
    selected_genres = list(active_filters.get("genres", []))
    era = active_filters.get("era", "Any Era")
    seed_ids = {int(movie_id) for movie_id in active_filters.get("loved_movies", [])}
    prompt = str(active_filters.get("keywords", ""))

    back_column, filter_column, shuffle_column = st.columns([1, 2, 1.15])
    with back_column:
        if st.button("← Back to Choices", key="back_to_choices"):
            st.session_state["page"] = "Discover"
            st.rerun()

    mood_labels = list(selected_genres)
    if era != "Any Era":
        mood_labels.append(era)
    profile = parse_preferences(prompt, mood_labels)
    with filter_column:
        filter_labels = list(selected_genres)
        if profile.era_label:
            filter_labels.append(profile.era_label)
        if seed_ids:
            filter_labels.append(f"{len(seed_ids)} reference movie(s)")
        st.markdown(
            '<div class="filter-pills">'
            + "".join(
                f'<span class="filter-pill">{html.escape(label)}</span>'
                for label in filter_labels
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    with shuffle_column:
        if st.button("🎲 Shuffle New Picks", key="shuffle_picks", type="primary"):
            recommendation_history = set(
                st.session_state.get("recommendation_history", [])
            )
            recommendation_history.update(
                st.session_state.get("last_recommendation_ids", [])
            )
            st.session_state["recommendation_history"] = sorted(
                recommendation_history
            )
            st.session_state["shuffle_seed"] = int(st.session_state.get("shuffle_seed", 0)) + 1
            st.rerun()

    detected = request.get("detected")
    if detected:
        st.success(f"Poster match: {detected['title']} ({detected['source']})")
    if not model["available"]:
        st.warning("The trained SVD artifact is unavailable; deterministic fallback scores are active.")

    try:
        shuffle_seed = int(st.session_state.get("shuffle_seed", 0))
        ranker_parameters = inspect.signature(rank_recommendations).parameters
        ranker_accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in ranker_parameters.values()
        )
        ranker_supports_shuffle = (
            "shuffle_seed" in ranker_parameters or ranker_accepts_kwargs
        )
        ranker_arguments: dict[str, Any] = {}
        ranker_mood_labels = mood_labels
        if "selected_genres" in ranker_parameters or ranker_accepts_kwargs:
            ranker_arguments["selected_genres"] = selected_genres
            ranker_arguments["era"] = era
            ranker_mood_labels = []
        if shuffle_seed and ranker_supports_shuffle:
            ranker_arguments["shuffle_seed"] = shuffle_seed
        if "seen_movie_ids" in ranker_parameters or ranker_accepts_kwargs:
            ranker_arguments["seen_movie_ids"] = set(
                st.session_state.get("recommendation_history", [])
            )
        predictions = _rating_predictions(
            catalog=catalog,
            model=model,
            prompt=prompt,
            mood_labels=mood_labels,
            selected_movie_ids=seed_ids,
            lambda_val=lambda_value,
        )
        ranked_candidates = rank_recommendations(
            catalog=catalog,
            predictions=predictions,
            prompt=prompt,
            mood_labels=ranker_mood_labels,
            seed_movie_ids=seed_ids,
            excluded_movie_ids=set(request.get("excluded_ids", [])),
            top_n=(
                min(30, len(catalog))
                if shuffle_seed and not ranker_supports_shuffle
                else top_n
            ),
            minimum_rating=minimum_rating,
            lambda_val=lambda_value,
            **ranker_arguments,
        )
        recommendations = (
            random.Random(shuffle_seed).sample(
                ranked_candidates[:30],
                k=min(top_n, len(ranked_candidates[:30])),
            )
            if shuffle_seed and not ranker_supports_shuffle and ranked_candidates
            else ranked_candidates[:top_n]
        )
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        LOGGER.exception("Recommendation pipeline failed.")
        st.error(
            "Recommendation pipeline error: "
            f"{type(error).__name__}: {error}"
        )
        return

    if not recommendations:
        st.info(
            "No eligible catalog movies are available. Remove some watchlist exclusions "
            "or change your choices."
        )
        return

    st.session_state["last_recommendation_ids"] = [
        int(movie["movie_id"]) for movie in recommendations
    ]
    fallback_notice = next(
        (movie.get("fallback_notice") for movie in recommendations if movie.get("fallback_notice")),
        "",
    )
    if fallback_notice:
        st.markdown(
            f'<div class="fallback-pill">ⓘ {html.escape(fallback_notice)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="gallery-heading">Your next movie night</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gallery-subtitle">A personalized mix of SVD predictions and your selected filters. '
        'Fresh titles are preferred across searches; earlier picks return only when the matching pool runs short.</div>',
        unsafe_allow_html=True,
    )
    criteria = [
        f"Showing {len(recommendations)} picks",
        "Genres: " + (", ".join(selected_genres) if selected_genres else "Any"),
        "Era: " + (profile.era_label or era),
        f"Min Rating: ≥{minimum_rating:.1f}",
        f"λ: {lambda_value:.1f}",
        f"Pool: {int(recommendations[0].get('pool_size', len(recommendations)))} matching movies",
        f"Shuffle: {shuffle_seed}",
    ]
    st.markdown(
        '<div class="active-criteria">'
        + " <span>·</span> ".join(
            f"<span>{html.escape(item)}</span>" for item in criteria
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    details_by_id = fetch_movie_details_batch(recommendations)
    for start in range(0, len(recommendations), 3):
        columns = st.columns(3, gap="medium")
        for column, movie in zip(columns, recommendations[start : start + 3]):
            with column:
                details = details_by_id.get(
                    int(movie["movie_id"]),
                    {
                        "poster_url": "",
                        "overview": "Synopsis details are temporarily unavailable.",
                        "full_overview": "Synopsis details are temporarily unavailable.",
                    },
                )
                if _render_movie_card(movie, details):
                    st.session_state["selected_movie_detail"] = {
                        "movie": movie,
                        "details": details,
                    }
                    st.rerun()
    selected_detail = st.session_state.get("selected_movie_detail")
    if selected_detail:
        _show_movie_details(
            selected_detail["movie"],
            selected_detail["details"],
        )


def _render_about(model: dict[str, Any], rmse: float | None, movie_count: int) -> None:
    st.markdown('<div class="eyebrow">BEHIND THE RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown('<h1>About GenreGenie</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="about-lead">A thoughtful movie-discovery experience that combines '
        'collaborative filtering with the way people actually choose a film: by mood, '
        'era, and movies they already love.</p>',
        unsafe_allow_html=True,
    )
    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            st.markdown("### What is GenreGenie?")
            st.write(
                "GenreGenie turns a few simple choices into a ranked movie shortlist. "
                "Selected genres and release era are strict filters, while reference "
                "movies help estimate a personal taste vector."
            )
            st.markdown(
                '<div class="workflow-line"><span>01</span> Mood + era filters</div>'
                '<div class="workflow-line"><span>02</span> Regularized SVD fold-in</div>'
                '<div class="workflow-line"><span>03</span> Seeded exploration ranking + metadata</div>',
                unsafe_allow_html=True,
            )
    with second:
        with st.container(border=True):
            st.markdown("### Model & validation")
            st.metric("Catalog titles", f"{movie_count:,}")
            st.metric(
                "SVD test RMSE",
                f"{rmse:.3f}" if rmse is not None else "Not available",
            )
            st.write(
                f"Engine status: **{'Trained SVD active' if model['available'] else 'Fallback scores'}**. "
                "The saved matrix-factorization model predicts ratings from item factors; "
                "λ controls fold-in regularization and list diversity. The RMSE is read "
                "from the project's saved evaluation metrics when available."
            )
    with st.container(border=True):
        st.markdown("### Quality, security & testing")
        quality_columns = st.columns(3)
        with quality_columns[0]:
            st.markdown("**Input handling**")
            st.write("Prompts are normalized and bounded. Genre and era filters are validated against the movie catalog.")
        with quality_columns[1]:
            st.markdown("**File checks**")
            st.write("Uploads are capped at 5 MB. CSV schema and null values are checked; poster headers are verified with Pillow.")
        with quality_columns[2]:
            st.markdown("**Reliable metadata**")
            st.write("Keyless Wikipedia/iTunes metadata is cached and fetched concurrently, with a generated poster fallback.")
        st.caption(
            "Run focused checks with: python -m pytest -q "
            "test_app.py test_recommendation_engine.py test_movie_metadata.py"
        )


st.set_page_config(
    page_title="GenreGenie | Movie discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    :root { color-scheme:dark; }
    .stApp { background:#0b0b0f; color:#f4f4f8; }
        html, body, [class*="st-"] { font-family:'Inter','Plus Jakarta Sans',sans-serif; }
        h1,h2,h3,.brand-title { font-family:'Plus Jakarta Sans','Inter',sans-serif; }
    header[data-testid="stHeader"] { background:transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { visibility:hidden; }
    [data-testid="stMainBlockContainer"] { padding-top:2.5rem; max-width:1500px; }
    [data-testid="stSidebar"] {
      width:260px!important; min-width:260px!important; max-width:260px!important;
      background:#101016; border-right:1px solid #22222e;
    }
    [data-testid="stSidebar"] > div:first-child { width:260px!important; }
    h1,h2,h3,p,label { color:#f4f4f8; }
    .topnav { display:flex; align-items:center; gap:.45rem; }
    .brand-title { font-size:1.55rem; font-weight:850; letter-spacing:-.05em; }
    .brand-gradient {
      background:linear-gradient(110deg,#a855f7,#ec4899);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .nav-note { color:#89899a; font-size:.75rem; }
    .stButton > button {
      min-height:2.3rem; padding:.35rem .75rem; border-radius:999px;
      color:#cac8d6; background:#171720; border:1px solid #2b2b39;
      font-size:.78rem; font-weight:650; transition:all .25s ease;
    }
    .stButton > button:hover {
      color:#fff; border-color:#a855f7; background:#211a30;
      box-shadow:0 0 22px rgba(168,85,247,.45); transform:translateY(-3px);
    }
    .stButton > button[kind="primary"] {
      background:linear-gradient(135deg,#7c3aed,#a855f7); color:#fff;
      border:1px solid #a855f7; border-radius:11px; font-size:.9rem;
      box-shadow:0 6px 18px rgba(124,58,237,.22);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#13131b; border:1px solid #242433!important;
      border-radius:15px!important; padding:.8rem!important;
      transition:all .25s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
      border-color:#a855f7!important;
      box-shadow:0 0 22px rgba(168,85,247,.45);
      transform:translateY(-3px);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(button[kind="primary"]) {
      border-color:#8b5cf6!important;
      box-shadow:0 0 15px rgba(168,85,247,.22);
    }
    div[data-testid="stTextInput"] input {
      background:#111118; border:1px solid #292936; border-radius:10px;
      color:#f4f4f8; min-height:2.6rem;
    }
    div[data-baseweb="select"] > div { background:#111118; border-color:#292936; border-radius:10px; }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
      background:#342151!important; color:#e7dcff!important;
      border:1px solid #6541a2!important; border-radius:999px!important;
    }
    div[data-testid="stFileUploader"] section {
      background:#111118; border:1px dashed #39334a; border-radius:10px;
    }
    div[data-testid="stFileUploader"] small { color:#9997a8; }
    [data-testid="stSlider"] [role="slider"] {
      background:#a855f7!important; border-color:#e9d5ff!important;
      box-shadow:0 0 0 3px rgba(168,85,247,.2)!important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child { background:#a855f7!important; }
    input[type="range"] { accent-color:#a855f7; }
    .hero { max-width:850px; margin:1.3rem auto 2.2rem; text-align:center; }
    .eyebrow { color:#b58af6; font-size:.69rem; letter-spacing:.16em; font-weight:800; }
    .hero h1,.stMarkdown h1 { margin:.5rem 0; font-size:clamp(2rem,4vw,3rem);
      letter-spacing:-.045em; font-weight:820; }
    .hero p,.about-lead { color:#aaa9b8; font-size:1rem; }
    .step-heading { margin:1.65rem 0 .45rem; font-size:1.03rem; font-weight:760; }
    .step-heading span,.workflow-line span { color:#c084fc; margin-right:.55rem; }
    .choice-copy { display:flex; align-items:center; gap:.7rem; min-height:3rem;
      font-size:.94rem; font-weight:730; }
    .choice-icon { font-size:1.45rem; }
    div[role="radiogroup"] { gap:.45rem; }
    div[role="radiogroup"] label {
      background:#171720; border:1px solid #2b2b39; border-radius:999px;
      padding:.35rem .8rem; transition:all .25s ease;
    }
    div[role="radiogroup"] label:hover {
      border-color:#a855f7; box-shadow:0 0 22px rgba(168,85,247,.45);
      transform:translateY(-3px);
    }
    .optional-note { color:#aaa9b8; font-size:.8rem; margin-bottom:.8rem; }
    .st-key-find_movies { max-width:500px; margin:1.1rem auto .3rem; }
    .st-key-find_movies button { min-height:3rem; font-size:1rem; }
    .filter-pills { display:flex; flex-wrap:wrap; justify-content:center;
      align-items:center; gap:.35rem; min-height:2.3rem; }
    .filter-pill,.match-badge,.rating-badge {
      display:inline-block; padding:.28rem .58rem; border-radius:999px;
      color:#e8d9ff; background:#39245a; border:1px solid #6944a3;
      font-size:.72rem; font-weight:780; white-space:nowrap;
    }
    .fallback-pill { display:inline-block; margin:.2rem 0 .65rem; padding:.35rem .7rem;
      color:#d5c4ff; background:#1b1426; border:1px solid #513576;
      border-radius:999px; font-size:.75rem; }
    .gallery-heading { font-size:1.45rem; font-weight:790; margin-top:1.4rem; }
    .gallery-subtitle { color:#9292a2; font-size:.82rem; margin-bottom:.85rem; }
    .active-criteria {
      display:flex; flex-wrap:wrap; gap:.3rem .55rem; margin:.65rem 0 1rem;
      padding:.65rem .8rem; color:#d5c9ed; background:rgba(25,21,34,.78);
      border:1px solid #332744; border-radius:10px; font-size:.72rem;
    }
    .active-criteria > span > span { color:#8f70bd; margin-left:.3rem; }
    .movie-card {
      height:100%; overflow:hidden; background:rgba(17,17,24,.92);
      border:1px solid rgba(255,255,255,.08); border-radius:16px;
      margin:.1rem 0 .35rem; backdrop-filter:blur(14px);
      transition:all .25s ease;
    }
    .movie-card:hover {
      border-color:#c026d3; box-shadow:0 0 24px rgba(139,92,246,.3),0 0 32px rgba(236,72,153,.18);
      transform:translateY(-3px);
    }
    .poster-stage { position:relative; width:100%; aspect-ratio:2/3; overflow:hidden;
      background:#21182d; }
    .movie-poster,.poster-placeholder { display:flex; width:100%; height:100%;
      object-fit:cover; align-items:center; justify-content:center;
      background:linear-gradient(145deg,#241839,#6f36a7);
      color:#e9ddff; font-size:.7rem; font-weight:800; letter-spacing:.14em; text-align:center; }
    .poster-match,.poster-rating { position:absolute; top:.65rem; z-index:1; }
    .poster-match { left:.6rem; color:#fff; background:linear-gradient(120deg,#7c3aed,#c026d3);
      border-color:rgba(255,255,255,.22); backdrop-filter:blur(12px); }
    .poster-rating { right:.6rem; color:#fff4c2; background:rgba(13,13,18,.78);
      border-color:rgba(255,255,255,.15); backdrop-filter:blur(12px); }
    .movie-copy { min-width:0; padding:.9rem .95rem .75rem; }
    .movie-title-row { display:flex; align-items:flex-start; justify-content:space-between; gap:.45rem; }
    .movie-title-row h3 { margin:.05rem 0 0; color:#f4f4f8; font-size:1.05rem;
      font-weight:750; line-height:1.35; }
    .year-badge { flex:0 0 auto; color:#a1a1aa; font-size:.7rem; padding:.15rem .4rem;
      border-radius:999px; background:#20202a; }
    .genre-list { display:flex; flex-wrap:wrap; gap:.3rem; margin-top:.58rem; min-height:1.5rem; }
    .genre-pill { display:inline-block; padding:.18rem .48rem; border-radius:999px;
      color:#d9c8ff; background:#211a2d; border:1px solid #3c2c58; font-size:.67rem; }
    .movie-overview { display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;
      overflow:hidden; color:#a1a1aa; font-size:.82rem; line-height:1.5;
      min-height:3.7rem; margin:.62rem 0 .55rem; }
    .reason-tag { color:#bfa5f5; font-size:.7rem; min-height:1.1rem; }
    [class*="st-key-full_synopsis_"] button { border-color:#3b2b50; color:#d8c9f3; }
    [class*="st-key-full_synopsis_"] button:hover { border-color:#c026d3; color:white; }
    .dialog-poster { display:block; width:100%; max-height:72vh; object-fit:contain;
      border-radius:12px; background:#111118; margin:0 auto 1rem; }
    [data-testid="stDialog"] { background:#111118; border:1px solid #332744; }
    .workflow-line { margin:.75rem 0; color:#c5c3d0; }
    .stAlert { border-radius:10px; }
    @media (max-width:650px) {
      [data-testid="stMainBlockContainer"] { padding-top:1.5rem; }
      .movie-copy { padding:.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = "Discover"
if "selected_genres" not in st.session_state:
    st.session_state["selected_genres"] = []

catalog = load_movie_catalog()
model_state = load_regression_model()
model_rmse = load_model_rmse()
movie_labels = {
    int(row.movieId): row.formatted_title
    for row in catalog.itertuples(index=False)
}

with st.sidebar:
    st.markdown(
        '<div class="brand-title">Genre<span class="brand-gradient">Genie</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f'{"SVD Engine Active" if model_state["available"] else "Fallback Engine"} · '
        f"{len(catalog):,} indexed movies"
    )
    st.markdown("---")
    lambda_value = st.slider(
        "λ Diversity",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Higher λ increases exploration and regularizes taste-profile fold-in.",
        key="lambda_val",
    )
    minimum_rating = st.slider(
        "Minimum predicted rating",
        min_value=0.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        key="min_rating",
    )
    top_n = st.select_slider(
        "Count", options=list(range(3, 13)), value=6, key="count"
    )

nav_logo, nav_discover, nav_results, nav_about = st.columns([1.5, 1, 1.35, 1.15])
with nav_logo:
    st.markdown(
        '<div class="brand-title">Genre<span class="brand-gradient">Genie</span></div>'
        '<div class="nav-note">Smart movie discovery</div>',
        unsafe_allow_html=True,
    )
for column, page_name, label in (
    (nav_discover, "Discover", "✨ Discover"),
    (nav_results, "Recommendations", "🎬 My Recommendations"),
    (nav_about, "About & Model", "ℹ️ About GenreGenie"),
):
    with column:
        if st.button(
            label,
            key=f"nav_{page_name}",
            type="primary" if st.session_state["page"] == page_name else "secondary",
            use_container_width=True,
        ):
            if (
                page_name == "Recommendations"
                and st.session_state["page"] == "Discover"
            ):
                _start_recommendations(catalog)
            else:
                st.session_state["page"] = page_name
                st.rerun()

if not model_state["available"]:
    st.warning("The trained SVD artifact is unavailable; deterministic fallback scores are active.")

if st.session_state["page"] == "Discover":
    _render_discover(catalog, movie_labels)
elif st.session_state["page"] == "Recommendations":
    _render_recommendations(
        catalog=catalog,
        model=model_state,
        lambda_value=float(lambda_value),
        minimum_rating=float(minimum_rating),
        top_n=int(top_n),
    )
else:
    _render_about(model_state, model_rmse, len(catalog))
