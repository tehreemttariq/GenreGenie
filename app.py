"""GenreGenie: choice-driven movie discovery dashboard."""

from __future__ import annotations

import html
import importlib
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
import streamlit.components.v1 as components

import tmdb_catalog
from input_validation import (
    MAX_UPLOAD_BYTES,
    sanitize_prompt,
    validate_poster,
    validate_watchlist_csv,
)
from movie_metadata import fallback_movie_details, fetch_movie_details_batch
from recommendation_engine import (
    detect_poster_movie,
    find_prompt_movie_ids,
    parse_preferences,
    prepare_catalog,
    rank_recommendations,
)


_TMDB_CATALOG_API = (
    "load_or_build_tmdb_full_catalog",
    "merge_tmdb_catalogs",
    "prepare_tmdb_movies",
)
if not all(callable(getattr(tmdb_catalog, name, None)) for name in _TMDB_CATALOG_API):
    tmdb_catalog = importlib.reload(tmdb_catalog)
load_or_build_tmdb_full_catalog = tmdb_catalog.load_or_build_tmdb_full_catalog
merge_tmdb_catalogs = tmdb_catalog.merge_tmdb_catalogs
prepare_tmdb_movies = tmdb_catalog.prepare_tmdb_movies


PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies_cleaned.csv"
LINKS_PATH = PROJECT_ROOT / "data" / "cleaned" / "links_cleaned.csv"
TMDB_FULL_LOAD_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "full_load_2005_2014.json",
    PROJECT_ROOT / "data" / "raw" / "full_load.json",
)
TMDB_INCREMENTAL_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "tmdb_incremental_catalog.json",
    PROJECT_ROOT / "data" / "raw" / "tmdb_horror_catalog.json",
)
TMDB_SOURCE_PATHS = TMDB_FULL_LOAD_PATHS + TMDB_INCREMENTAL_PATHS
TMDB_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "tmdb_catalog.csv"
TMDB_CACHE_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tmdb_catalog_manifest.json"
)
RATINGS_PATH = PROJECT_ROOT / "data" / "cleaned" / "ratings_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "svd_model.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "evaluation_metrics.json"
LOGGER = logging.getLogger(__name__)
GENRE_CHOICES = (
    ("heart", "Romance", "Romance"),
    ("smile", "Comedy", "Comedy"),
    ("orbit", "Sci-Fi", "Sci-Fi"),
    ("eye", "Thriller", "Thriller"),
    ("layers", "Drama", "Drama"),
    ("bolt", "Action", "Action"),
)
ICON_PATHS = {
    "heart": '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21l7.8-7.5 1.1-1.1a5.5 5.5 0 0 0-.1-7.8Z"/>',
    "smile": '<circle cx="12" cy="12" r="9"/><path d="M8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01"/>',
    "orbit": '<circle cx="12" cy="12" r="2"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(-30 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(30 12 12)"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
    "bolt": '<path d="m13 2-3 8h7L9 22l2-9H5l8-11Z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "bookmark": '<path d="M6 4h12v17l-6-4-6 4V4Z"/>',
    "clapperboard": '<path d="M3 9h18v12H3z"/><path d="m3 9 2-6 16 2-2 6M7 4l3 5m4-4 3 5M8 14h4m-4 3h7"/>',
}
ERA_CHOICES = (
    "Any Era",
    "Classic (Before 1980)",
    "1980s",
    "1990s",
    "2000–2009",
    "2010–2019",
    "2020 & Newer",
)
SAFE_POSTER_HOSTS = {
    "upload.wikimedia.org",
    "commons.wikimedia.org",
    "image.tmdb.org",
    "static.tvmaze.com",
    "is1-ssl.mzstatic.com",
}


def _iter_tmdb_incremental_records():
    """Load incremental TMDB records from multiple possible formats."""
    for path in TMDB_INCREMENTAL_PATHS:
        if not path.is_file():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            LOGGER.warning("Could not load incremental catalog %s: %s", path, e)
            continue
        
        if not isinstance(state, dict):
            continue
        
        # Try different possible structures
        records = []
        
        # Format 1: Direct "movies" or "results" array
        if "movies" in state and isinstance(state["movies"], list):
            records = state["movies"]
        elif "results" in state and isinstance(state["results"], list):
            records = state["results"]
        
        # Format 2: Nested structure with categories (trending_today, changed_movies, etc.)
        else:
            for key, value in state.items():
                if isinstance(value, dict) and "results" in value and isinstance(value["results"], list):
                    records.extend(value["results"])
        
        # Yield each valid movie record
        for record in records:
            if isinstance(record, dict):
                yield record


def _linked_tmdb_ids() -> set[int]:
    if not LINKS_PATH.is_file():
        return set()
    links = pd.read_csv(LINKS_PATH, usecols=["tmdbId"])
    return set(
        pd.to_numeric(links["tmdbId"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )


def _catalog_fingerprint() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            str(path),
            path.stat().st_mtime_ns if path.is_file() else 0,
            path.stat().st_size if path.is_file() else 0,
        )
        for path in (MOVIES_PATH, LINKS_PATH, *TMDB_SOURCE_PATHS)
    )


@st.cache_resource(show_spinner="Loading TMDB movie catalog...")
def _load_movie_catalog(
    catalog_fingerprint: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    del catalog_fingerprint
    movies = pd.read_csv(MOVIES_PATH)
    tmdb_catalog = load_or_build_tmdb_full_catalog(
        TMDB_FULL_LOAD_PATHS,
        TMDB_CACHE_PATH,
        TMDB_CACHE_MANIFEST_PATH,
    )
    incremental_catalog = prepare_tmdb_movies(_iter_tmdb_incremental_records())
    tmdb_catalog = merge_tmdb_catalogs(tmdb_catalog, incremental_catalog)
    if not tmdb_catalog.empty:
        excluded_ids = _linked_tmdb_ids()
        tmdb_catalog = tmdb_catalog[
            ~pd.to_numeric(tmdb_catalog["tmdb_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .isin(excluded_ids)
        ]
        movies = pd.concat([movies, tmdb_catalog], ignore_index=True, sort=False)
    return prepare_catalog(movies)


def load_movie_catalog() -> pd.DataFrame:
    return _load_movie_catalog(_catalog_fingerprint())


@st.cache_resource(show_spinner=False)
def _load_movie_labels(
    catalog_fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[int, str]:
    catalog = _load_movie_catalog(catalog_fingerprint)
    return {
        int(row.movieId): row.formatted_title
        for row in catalog.itertuples(index=False)
    }


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
        saved_movie_indices = model_data.get("movie_to_idx")
        if isinstance(saved_movie_indices, dict):
            movie_indices = {
                int(movie_id): int(index)
                for movie_id, index in saved_movie_indices.items()
            }
        else:
            movie_indices = _training_movie_indices()
        item_factors = np.asarray(model_data["item_factors"], dtype=float)
        global_mean = float(model_data["global_mean"])
        if (
            item_factors.ndim != 2
            or item_factors.shape[0] != len(movie_indices)
            or sorted(movie_indices.values()) != list(range(item_factors.shape[0]))
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
        fallback_scores = {
            int(movie_id): 2.8 + ((int(movie_id) * 37) % 22) / 10
            for movie_id in catalog["movieId"]
        }
        for row in catalog.itertuples(index=False):
            try:
                if str(getattr(row, "rating_source", "")).casefold() != "tmdb":
                    continue
                rating = float(getattr(row, "rating_prediction"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(rating):
                fallback_scores[int(row.movieId)] = float(np.clip(rating, 0.5, 5.0))
        return fallback_scores

    factors = model["item_factors"]
    movie_indices = model["movie_indices"]
    profile = parse_preferences(prompt, mood_labels)
    prompt_movie_ids = set(find_prompt_movie_ids(prompt, catalog))
    explicit_seed_ids = selected_movie_ids | prompt_movie_ids
    seed_ids = set(explicit_seed_ids)
    if profile.genre_groups:
        matching_mask = catalog["movieId"].isin(movie_indices)
        normalized_genres = catalog["_genres_norm"]
        for group in profile.genre_groups:
            group_mask = pd.Series(False, index=catalog.index)
            for genre in group:
                group_mask |= normalized_genres.str.contains(
                    genre.casefold(), regex=False, na=False
                )
            matching_mask &= group_mask
        if profile.year_min is not None:
            matching_mask &= catalog["year"].ge(profile.year_min).fillna(False)
        if profile.year_max is not None:
            matching_mask &= catalog["year"].le(profile.year_max).fillna(False)
        seed_ids.update(
            catalog.loc[matching_mask, "movieId"].head(100).astype(int).tolist()
        )

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


def _icon_svg(name: str, class_name: str = "line-icon") -> str:
    path = ICON_PATHS[name]
    return (
        f'<svg class="{class_name}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


def _catalog_genres(catalog: pd.DataFrame) -> list[str]:
    return sorted(
        {
            str(genre).strip()
            for genres in catalog["genre_tags"]
            for genre in genres
            if str(genre).strip()
        },
        key=str.casefold,
    )


def _find_similar(movie_id: int) -> None:
    active_filters = dict(st.session_state.get("active_filters", {}))
    active_filters.update(
        {
            "genres": [],
            "era": "Any Era",
            "loved_movies": [int(movie_id)],
            "keywords": "",
        }
    )
    st.session_state["active_filters"] = active_filters
    st.session_state["similarity_threshold"] = 75
    st.session_state["page"] = "Recommendations"
    st.session_state["shuffle_seed"] = 0
    st.session_state.pop("selected_movie_detail", None)
    st.rerun()


def _render_movie_card(movie: dict[str, Any], details: dict[str, Any]) -> str | None:
    title = html.escape(str(movie["title"]))
    year = html.escape(str(movie["year"]) if movie.get("year") else "Year unknown")
    rating_source = html.escape(str(movie.get("rating_source", "SVD")))
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
            <span class="rating-badge poster-rating">★ {movie["predicted_rating"]:.1f} {rating_source}</span>
          </div>
          <div class="movie-copy">
            <div class="movie-title-row">
              <h3>{title}</h3>
              <span class="year-badge">{year}</span>
            </div>
            <div class="genre-list">{genres}</div>
            <p class="movie-overview">{overview}</p>
            <div class="reason-tag">A note for your queue · {reason}</div>
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    synopsis_column, similar_column = st.columns(2)
    with synopsis_column:
        show_synopsis = st.button(
            "Full Synopsis",
            key=f"full_synopsis_{int(movie['movie_id'])}",
            use_container_width=True,
        )
    with similar_column:
        find_similar = st.button(
            "Find Similar",
            key=f"find_similar_{int(movie['movie_id'])}",
            use_container_width=True,
        )
    if find_similar:
        return "similar"
    return "details" if show_synopsis else None


@st.dialog("Movie details", width="large")
def _show_movie_details(
    movie: dict[str, Any],
    details: dict[str, Any],
) -> None:
    poster_url = _safe_poster_url(details.get("poster_url"))
    poster = (
        f'<img class="dialog-poster" src="{html.escape(poster_url, quote=True)}" '
        f'alt="Poster for {html.escape(str(movie["formatted_title"]), quote=True)}">'
        if poster_url
        else '<div class="dialog-poster-placeholder">GENRE<br>GENIE</div>'
    )
    poster_column, details_column = st.columns([0.72, 1.55], gap="medium")
    with poster_column:
        st.markdown(poster, unsafe_allow_html=True)
    with details_column:
        st.subheader(str(movie["formatted_title"]))
        st.markdown(
            '<div class="detail-badges">'
            f'<span>{movie["match_pct"]}% Match</span>'
            f'<span>★ {movie["predicted_rating"]:.1f} '
            f'{html.escape(str(movie.get("rating_source", "SVD")))}</span>'
            f'<span>{html.escape(str(movie["year"] or "Year unknown"))}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(" · ".join(movie["genres"]) if movie["genres"] else "General")
        credit_rows = []
        for label, field in (
            ("Director", "director"),
            ("Starring", "cast"),
            ("Production", "production"),
        ):
            value = details.get(field)
            if isinstance(value, (list, tuple)):
                value = " · ".join(str(person) for person in value if str(person).strip())
            if isinstance(value, str) and value.strip():
                credit_rows.append(
                    '<div class="credit-row">'
                    f'<span>{label}</span><strong>{html.escape(value.strip())}</strong>'
                    '</div>'
                )
        if credit_rows:
            st.markdown(
                f'<div class="credit-list">{"".join(credit_rows)}</div>',
                unsafe_allow_html=True,
            )
        breakdown = [
            f'Genre fit {movie["genre_score"]:.0%}'
            if movie.get("genre_score") is not None
            else None,
            f'Era fit {movie["era_score"]:.0%}'
            if movie.get("era_score") is not None
            else None,
            f'{html.escape(str(movie.get("rating_source", "SVD")))} rating '
            f'{movie["predicted_rating"]:.1f}/5',
        ]
        st.caption("Match breakdown · " + " · ".join(item for item in breakdown if item))
        synopsis = html.escape(
            str(
                details.get("full_overview")
                or details.get("overview")
                or "Synopsis details are temporarily unavailable."
            )
        )
        st.markdown(
            '<div class="synopsis-label">Full synopsis</div>'
            f'<div class="synopsis-scroll">{synopsis}</div>',
            unsafe_allow_html=True,
        )
    if st.button(
        "Find Similar",
        key=f"find_more_like_{int(movie['movie_id'])}",
        type="primary",
        use_container_width=True,
    ):
        _find_similar(int(movie["movie_id"]))
    if st.button("Close", key=f"close_movie_detail_{int(movie['movie_id'])}"):
        st.session_state.pop("selected_movie_detail", None)
        st.rerun()


def _recent_horror_count(catalog: pd.DataFrame) -> int:
        recent_years = catalog["year"].ge(2020).fillna(False)
        has_horror = catalog["genre_tags"].map(
            lambda genres: "Horror" in genres
        )
        return int((recent_years & has_horror).sum())


def _start_recommendations(catalog: pd.DataFrame) -> None:
    prompt = sanitize_prompt(st.session_state.get("custom_keywords", ""))
    selected_genres = list(st.session_state.get("selected_genres", []))
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
    st.session_state.pop("similarity_threshold", None)
    st.session_state.pop("selected_movie_detail", None)
    st.session_state["shuffle_seed"] = 0
    st.session_state["page"] = "Recommendations"
    st.rerun()


def _compute_recommendations(
    catalog: pd.DataFrame,
    model: dict[str, Any],
    prompt: str,
    mood_labels: list[str],
    selected_genres: list[str],
    era: str,
    seed_ids: set[int],
    excluded_ids: set[int],
    seen_movie_ids: set[int],
    top_n: int,
    minimum_rating: float,
    lambda_value: float,
    shuffle_seed: int,
    similarity_threshold: int | None,
) -> list[dict[str, Any]]:
    ranker_parameters = inspect.signature(rank_recommendations).parameters
    ranker_accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in ranker_parameters.values()
    )
    ranker_supports_shuffle = (
        "shuffle_seed" in ranker_parameters or ranker_accepts_kwargs
    )
    ranker_supports_match_threshold = (
        "minimum_match_pct" in ranker_parameters or ranker_accepts_kwargs
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
        ranker_arguments["seen_movie_ids"] = seen_movie_ids
    if similarity_threshold is not None and ranker_supports_match_threshold:
        ranker_arguments["minimum_match_pct"] = similarity_threshold

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
        excluded_movie_ids=excluded_ids,
        top_n=(
            min(30, len(catalog))
            if shuffle_seed and not ranker_supports_shuffle
            else (
                len(catalog)
                if similarity_threshold is not None
                and not ranker_supports_match_threshold
                else top_n
            )
        ),
        minimum_rating=minimum_rating,
        lambda_val=lambda_value,
        **ranker_arguments,
    )
    recommendation_pool = ranked_candidates
    if similarity_threshold is not None:
        recommendation_pool = [
            movie
            for movie in ranked_candidates
            if int(movie["match_pct"]) >= similarity_threshold
            and float(movie["predicted_rating"]) >= minimum_rating
        ]
    return (
        random.Random(shuffle_seed).sample(
            recommendation_pool[:30],
            k=min(top_n, len(recommendation_pool[:30])),
        )
        if shuffle_seed
        and not ranker_supports_shuffle
        and recommendation_pool
        else recommendation_pool[:top_n]
    )


def _render_recommendation_controls() -> None:
    st.markdown(
        '<div class="control-heading">Set your shortlist</div>',
        unsafe_allow_html=True,
    )
    rating_column, count_column, lambda_column = st.columns(3, gap="large")
    with rating_column:
        st.slider(
            "Minimum Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Filter out films below this predicted rating.",
            key="min_rating",
        )
    with count_column:
        st.slider(
            "Number of Picks",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="count",
        )
    with lambda_column:
        st.slider(
            "Diversity / Regularization (λ)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Balance familiar ratings with a little room for discovery.",
            key="lambda_val",
        )


def _render_discover(catalog: pd.DataFrame, movie_labels: dict[int, str]) -> None:
    st.markdown(
        '<section class="hero"><div class="eyebrow">A LITTLE HELP FROM THE PICTURE HOUSE</div>'
        f'<h1>What are you in the mood to watch? '
        f'<span class="hero-clapperboard">{_icon_svg("clapperboard")}</span></h1>'
        '<p>Choose a feeling, a decade, or a favorite. We’ll find the film that fits tonight.</p></section>',
        unsafe_allow_html=True,
    )

    recent_horror_count = _recent_horror_count(catalog)
    if recent_horror_count:
        st.caption(
            f"{recent_horror_count} horror titles from 2020 onward are in the catalog."
        )
    else:
        latest_year = catalog["year"].max()
        latest_year_label = (
            str(int(latest_year)) if pd.notna(latest_year) else "an unknown year"
        )
        st.info(
            "No horror titles from 2020 onward are loaded. "
            f"The current catalog ends in {latest_year_label}; run the TMDB "
            "incremental sync and refresh Discover to add recent releases."
        )

    _render_recommendation_controls()

    st.markdown('<div class="step-heading"><span>01</span> Choose a feeling</div>', unsafe_allow_html=True)
    st.caption("Select one or more; your shortlist will stay true to the moods you choose.")
    selected_genres = set(st.session_state.get("selected_genres", []))
    available_genres = _catalog_genres(catalog)
    primary_choices = [
        choice for choice in GENRE_CHOICES if choice[2] in available_genres
    ]
    genre_columns = st.columns(3)
    for index, (icon, label, value) in enumerate(primary_choices):
        with genre_columns[index % 3]:
            with st.container(border=True):
                active = value in selected_genres
                st.markdown(
                    f'<div class="choice-copy">{_icon_svg(icon, "choice-icon")}'
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

    remaining_genres = [
        genre for genre in available_genres
        if genre not in {choice[2] for choice in primary_choices}
    ]
    if remaining_genres:
        with st.expander("Explore all catalog genres", expanded=False):
            st.caption("Choose any additional genre to shape your shortlist.")
            extra_columns = st.columns(4, gap="small")
            for index, genre in enumerate(remaining_genres):
                with extra_columns[index % 4]:
                    st.button(
                        f"{'✓ ' if genre in selected_genres else ''}{genre}",
                        key=f"catalog_genre_{index}",
                        type=(
                            "primary"
                            if genre in selected_genres
                            else "secondary"
                        ),
                        on_click=_toggle_genre,
                        args=(genre,),
                        use_container_width=True,
                    )

    st.markdown('<div class="step-heading"><span>02</span> Set the scene</div>', unsafe_allow_html=True)
    st.radio(
        "Release era",
        options=ERA_CHOICES,
        horizontal=True,
        key="selected_era",
        label_visibility="collapsed",
    )

    with st.expander("Fine-tune with a favorite, watchlist, or a few words", expanded=False):
        st.markdown(
            '<div class="optional-note">A favorite film or a few clues can bring your shortlist closer to your taste.</div>',
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
        "Show Me the Films →",
        key="find_movies",
        type="primary",
        use_container_width=True,
    ):
        _start_recommendations(catalog)


def _toggle_genre(value: str) -> None:
    selected = list(st.session_state.get("selected_genres", []))
    if value in selected:
        selected.remove(value)
    else:
        selected.append(value)
    st.session_state["selected_genres"] = selected


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
    similarity_threshold = st.session_state.get("similarity_threshold")

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
        reference_rows = catalog[catalog["movieId"].isin(seed_ids)]
        for row in reference_rows.itertuples(index=False):
            reference_title = str(row.formatted_title)
            filter_labels.append(
                f"Similar to: {reference_title}"
                if similarity_threshold is not None
                else f"With reference: {reference_title}"
            )
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
        if st.button("Shuffle New Picks", key="shuffle_picks", type="primary"):
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

    shuffle_seed = int(st.session_state.get("shuffle_seed", 0))
    excluded_ids = {int(movie_id) for movie_id in request.get("excluded_ids", [])}
    seen_movie_ids = {
        int(movie_id) for movie_id in st.session_state.get("recommendation_history", [])
    }
    catalog_version = tuple(
        path.stat().st_mtime_ns if path.is_file() else 0
        for path in (MOVIES_PATH, LINKS_PATH) + TMDB_SOURCE_PATHS
    )
    model_version = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else 0
    cache_key = (
        catalog_version,
        model_version,
        bool(model["available"]),
        tuple(selected_genres),
        str(era),
        tuple(sorted(seed_ids)),
        prompt,
        tuple(sorted(excluded_ids)),
        tuple(sorted(seen_movie_ids)),
        int(top_n),
        float(minimum_rating),
        float(lambda_value),
        shuffle_seed,
        int(similarity_threshold) if similarity_threshold is not None else None,
    )
    cache_entry = st.session_state.get("_recommendation_cache", {})
    try:
        if cache_entry.get("key") == cache_key:
            recommendations = list(cache_entry["recommendations"])
        else:
            recommendations = _compute_recommendations(
                catalog=catalog,
                model=model,
                prompt=prompt,
                mood_labels=mood_labels,
                selected_genres=selected_genres,
                era=era,
                seed_ids=seed_ids,
                excluded_ids=excluded_ids,
                seen_movie_ids=seen_movie_ids,
                top_n=top_n,
                minimum_rating=minimum_rating,
                lambda_value=lambda_value,
                shuffle_seed=shuffle_seed,
                similarity_threshold=(
                    int(similarity_threshold)
                    if similarity_threshold is not None
                    else None
                ),
            )
            st.session_state["_recommendation_cache"] = {
                "key": cache_key,
                "recommendations": recommendations,
            }
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        LOGGER.exception("Recommendation pipeline failed.")
        st.error(
            "Recommendation pipeline error: "
            f"{type(error).__name__}: {error}"
        )
        return

    if not recommendations:
        if similarity_threshold is not None:
            st.info(
                "No films in this shortlist reached a 75% match. Try another favorite "
                "or adjust your rating filter."
            )
        else:
            st.info(
                "No eligible catalog movies are available. Remove some watchlist exclusions "
                "or change your choices."
            )
        st.session_state["last_recommendation_ids"] = []
        return

    st.session_state["last_recommendation_ids"] = [
        int(movie["movie_id"]) for movie in recommendations
    ]
    if similarity_threshold is not None and len(recommendations) < top_n:
        st.info(
            f"Only {len(recommendations)} films reached the 75% match threshold; "
            "showing every qualifying pick."
        )
    fallback_notice = (
        ""
        if similarity_threshold is not None
        else next(
            (
                movie.get("fallback_notice")
                for movie in recommendations
                if movie.get("fallback_notice")
            ),
            "",
        )
    )
    if fallback_notice:
        st.markdown(
            f'<div class="fallback-pill">ⓘ {html.escape(fallback_notice)}</div>',
            unsafe_allow_html=True,
        )
    gallery_heading = (
        "Close companions to your favorite"
        if similarity_threshold is not None
        else "Your next movie night"
    )
    st.markdown(
        f'<div class="gallery-heading">{gallery_heading}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gallery-subtitle">A considered shortlist, shaped by your choices and a little cinematic serendipity. '
        'Fresh titles are preferred across searches; earlier picks return only when the matching pool runs short.</div>',
        unsafe_allow_html=True,
    )
    criteria = [
        f"Showing {len(recommendations)} picks",
        "Genres: " + (", ".join(selected_genres) if selected_genres else "Any"),
        "Era: " + (profile.era_label or era),
        (
            "Min Rating: Any"
            if minimum_rating <= 0.5
            else f"Min Rating: ≥{minimum_rating:.1f}"
        ),
        *(
            [f"Match: ≥{int(similarity_threshold)}%"]
            if similarity_threshold is not None
            else []
        ),
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
    details_by_id: dict[int, dict[str, Any]] = {}
    for movie in recommendations:
        movie_id = int(movie["movie_id"])
        details = fallback_movie_details(
            str(movie["formatted_title"]),
            movie.get("year"),
            movie.get("genres", ()),
        )
        poster_url = movie.get("poster_url")
        overview = movie.get("overview")
        if isinstance(poster_url, str) and poster_url:
            details["poster_url"] = poster_url
        if isinstance(overview, str) and overview:
            details["overview"] = overview
            details["full_overview"] = overview
        for field in ("director", "cast", "production"):
            value = movie.get(field)
            if (isinstance(value, str) and value.strip()) or (
                isinstance(value, (list, tuple)) and value
            ):
                details[field] = value
        details_by_id[movie_id] = details
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
                card_action = _render_movie_card(movie, details)
                if card_action == "similar":
                    _find_similar(int(movie["movie_id"]))
                if card_action == "details":
                    fetched_details = fetch_movie_details_batch([movie])
                    details = fetched_details.get(
                        int(movie["movie_id"]),
                        fallback_movie_details(
                            str(movie["formatted_title"]),
                            movie.get("year"),
                            movie.get("genres", ()),
                        ),
                    )
                    poster_url = movie.get("poster_url")
                    overview = movie.get("overview")
                    if isinstance(poster_url, str) and poster_url:
                        details["poster_url"] = poster_url
                    if isinstance(overview, str) and overview:
                        details["overview"] = overview
                        details["full_overview"] = overview
                    for field in ("director", "cast", "production"):
                        value = movie.get(field)
                        if (isinstance(value, str) and value.strip()) or (
                            isinstance(value, (list, tuple)) and value
                        ):
                            details[field] = value
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
    copy_column, art_column = st.columns([1.15, 0.85], gap="large")
    with copy_column:
        st.markdown(
            '<div class="eyebrow">A MORE PERSONAL WAY TO PRESS PLAY</div>'
            '<h1 class="about-title">About GenreGenie</h1>'
            '<p class="about-lead">Every great movie night begins with a feeling. '
            'GenreGenie turns yours into a short, thoughtful list of films worth staying in for.</p>'
            '<p class="about-detail">Follow a mood, travel to another era, or start with a film you love. '
            'We’ll take it from there.</p>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Get Started →",
            key="get_started",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["page"] = "Discover"
            st.rerun()
    with art_column:
        st.markdown(
            '<div class="cinema-art" aria-hidden="true">'
            '<div class="art-glow"></div>'
            '<div class="film-stack film-stack-back"><span>02</span><i></i><i></i><i></i></div>'
            '<div class="film-stack film-stack-front"><span>TONIGHT</span><strong>Find your<br>next favorite.</strong>'
            '<div class="art-rule"></div><small>A GenreGenie selection</small></div>'
            f'{_icon_svg("orbit", "art-orbit")}'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="landing-rule"></div>'
        '<div class="landing-caption">THE PLEASURE IS IN THE PICKING</div>',
        unsafe_allow_html=True,
    )
    steps = st.columns(3, gap="large")
    for column, number, title, description in (
        (steps[0], "01", "Name the mood", "Romance, suspense, laughter—or a little of everything."),
        (steps[1], "02", "Set the scene", "Choose a favorite era, or let the whole catalog surprise you."),
        (steps[2], "03", "Meet your next film", "A personal shortlist, ready for the opening credits."),
    ):
        with column:
            st.markdown(
                f'<div class="landing-step"><span>{number}</span><h3>{title}</h3>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )
    with st.expander("Behind the curtain · recommendation notes"):
        st.markdown("#### Quality, security & provenance")
        st.write(
            "GenreGenie uses a trained ratings model when available and keeps discovery usable "
            "with deterministic fallback scores. Prompts are bounded, uploaded lists are schema-checked, "
            "and reference images are validated before they are read."
        )
        metrics_columns = st.columns(3)
        metrics_columns[0].metric("Films in the catalog", f"{movie_count:,}")
        metrics_columns[1].metric(
            "Model status",
            "Trained" if model["available"] else "Fallback",
        )
        metrics_columns[2].metric(
            "SVD test RMSE",
            f"{rmse:.3f}" if rmse is not None else "Not available",
        )


st.set_page_config(
    page_title="GenreGenie | Movie discovery",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    :root { color-scheme:dark; }
    .stApp {
      position:relative; isolation:isolate; color:#f4f4f8;
      background-color:#0b0b0f;
      background-image:
        radial-gradient(ellipse at 12% 18%,rgba(83,52,153,.22),transparent 42%),
        radial-gradient(ellipse at 86% 30%,rgba(84,39,109,.18),transparent 38%),
        radial-gradient(ellipse at 58% 92%,rgba(31,52,105,.17),transparent 44%),
        linear-gradient(135deg,#0b0a10,#0d0e1b 48%,#100c17);
      background-size:150% 150%;
      animation:cinema-ambient-drift 32s ease-in-out infinite alternate;
      --cursor-x:-500px; --cursor-y:-500px;
    }
    @property --cursor-x {
      syntax:"<length>"; inherits:true; initial-value:-500px;
    }
    @property --cursor-y {
      syntax:"<length>"; inherits:true; initial-value:-500px;
    }
    .stApp::before,.stApp::after {
      content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
    }
    .stApp::before {
      opacity:.045;
      background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.32'/%3E%3C/svg%3E");
      background-size:180px 180px; mix-blend-mode:screen;
    }
    .stApp::after {
      background:
        radial-gradient(circle 190px at var(--cursor-x) var(--cursor-y),rgba(146,80,255,.13),transparent 78%),
        radial-gradient(circle 18px at var(--cursor-x) var(--cursor-y),transparent 11px,rgba(186,134,255,.7) 12px,rgba(94,224,255,.62) 13px,transparent 17px);
      opacity:.8; transition:opacity .25s ease,--cursor-x .16s ease-out,--cursor-y .16s ease-out;
    }
    [data-testid="stAppViewContainer"],[data-testid="stSidebar"] {
      position:relative; z-index:1;
    }
    @keyframes cinema-ambient-drift {
      0% { background-position:0% 15%; }
      100% { background-position:100% 85%; }
    }
        html, body, .stApp, [data-testid="stSidebar"] { font-family:'Inter','Plus Jakarta Sans',sans-serif; }
        [data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded'!important; }
        [data-testid="stExpanderToggleIcon"] { font-size:0!important; width:.9rem; }
        [data-testid="stExpanderToggleIcon"]::after {
          content:""; display:block; width:.38rem; height:.38rem;
          border-right:1.5px solid currentColor; border-bottom:1.5px solid currentColor;
          transform:rotate(-45deg); transition:transform .2s ease;
        }
        [data-testid="stExpander"][aria-expanded="true"] [data-testid="stExpanderToggleIcon"]::after {
          transform:rotate(45deg);
        }
        h1,h2,h3,.brand-title { font-family:'Plus Jakarta Sans','Inter',sans-serif; }
    header[data-testid="stHeader"] { background:transparent; }
    [data-testid="stDecoration"], #MainMenu, footer { visibility:hidden; }
    [data-testid="stMainBlockContainer"] { padding:2.2rem 2.4rem 4rem; max-width:1240px; margin:0 auto; }
    [data-testid="stSidebar"] {
      width:260px!important; min-width:260px!important; max-width:260px!important;
      background:#101016; border-right:1px solid #22222e;
    }
    [data-testid="stSidebar"] > div:first-child { width:260px!important; }
    h1,h2,h3,p,label { color:#f4f4f8; }
    .brand-title { font-size:1.55rem; font-weight:850; letter-spacing:-.05em; }
    .brand-gradient {
      background:linear-gradient(110deg,#a855f7,#ec4899);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .nav-note { color:#89899a; font-size:.75rem; }
    .brand-tagline { color:#aaa9b8; font-size:.76rem; line-height:1.5; margin:.25rem 0 1.15rem; }
    .sidebar-section-title { color:#827f91; font-size:.65rem; font-weight:800;
      letter-spacing:.14em; text-transform:uppercase; margin:1.1rem 0 .45rem; }
    .line-icon { width:1.1rem; height:1.1rem; flex:0 0 auto; }
    .sidebar-nav-icon { color:#b8a2dc; display:flex; justify-content:center;
      align-items:center; min-height:2.35rem; }
    .sidebar-nav-icon .line-icon { width:1.05rem; height:1.05rem; }
    [data-testid="stSidebar"] .stButton > button {
      width:100%; min-height:2.35rem; text-align:left; border-radius:9px;
      justify-content:flex-start; font-size:.82rem;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] { padding-bottom:.3rem; }
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
    [data-testid="stVerticalBlockBorderWrapper"]:has(.choice-copy) {
      transform:perspective(1000px) rotateX(0deg) rotateY(0deg);
      transform-style:preserve-3d;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.choice-copy):hover {
      border-color:#a855f7!important;
      box-shadow:0 18px 38px rgba(0,0,0,.38),0 0 24px rgba(168,85,247,.2);
      transform:perspective(1000px) rotateX(2deg) rotateY(-2deg) translateY(-4px);
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
    .hero { max-width:820px; margin:1.3rem auto 2.2rem; text-align:center; }
    .eyebrow { color:#b58af6; font-size:.69rem; letter-spacing:.16em; font-weight:800; }
    .hero h1,.stMarkdown h1 { margin:.5rem 0; font-size:clamp(2rem,4vw,3rem);
      letter-spacing:-.045em; font-weight:820; }
    .hero p,.about-lead { color:#aaa9b8; font-size:1rem; }
    .hero-clapperboard { display:inline-flex; vertical-align:middle; margin-left:.45rem;
      color:#d5b8ff; filter:drop-shadow(0 5px 16px rgba(192,132,252,.44)); }
    .hero-clapperboard .line-icon { width:clamp(2rem,3vw,2.8rem);
      height:clamp(2rem,3vw,2.8rem); transform:rotate(-7deg); }
    .control-heading { color:#f0eaf8; font-size:.94rem; font-weight:760;
      letter-spacing:-.015em; margin:.1rem 0 .45rem; }
    .about-title { font-size:clamp(2.6rem,5vw,4.6rem)!important; line-height:1.02; margin:.7rem 0 1rem!important; }
    .about-lead { max-width:650px; font-size:1.18rem; line-height:1.75; }
    .about-detail { color:#888696; font-size:.91rem; line-height:1.7; max-width:560px; margin:1rem 0 1.4rem; }
    .cinema-art { position:relative; min-height:370px; display:flex; align-items:center;
      justify-content:center; perspective:1000px; overflow:hidden; }
    .art-glow { position:absolute; width:290px; height:290px; border-radius:50%;
      background:radial-gradient(circle,rgba(142,81,197,.28),rgba(70,36,95,.04) 68%,transparent 72%);
      filter:blur(10px); }
    .film-stack { position:absolute; display:flex; flex-direction:column; justify-content:center;
      gap:.55rem; width:235px; height:300px; padding:1.5rem; border:1px solid rgba(255,255,255,.16);
      border-radius:18px; backdrop-filter:blur(16px); box-shadow:0 28px 75px rgba(0,0,0,.48); }
    .film-stack-back { background:linear-gradient(145deg,rgba(58,37,78,.92),rgba(25,23,35,.9));
      transform:rotateY(-21deg) rotateZ(-11deg) translate(-34px,4px); }
    .film-stack-back span { color:#d5b9ff; font-size:.68rem; letter-spacing:.2em; }
    .film-stack-back i { height:1px; background:rgba(255,255,255,.15); }
    .film-stack-front { z-index:1; background:linear-gradient(145deg,rgba(55,40,73,.84),rgba(20,19,29,.94));
      transform:rotateY(-12deg) rotateZ(5deg) translate(20px,-3px); }
    .film-stack-front span,.film-stack-front small { color:#bda4db; font-size:.62rem; letter-spacing:.16em; }
    .film-stack-front strong { color:#f8f3ff; font-size:1.55rem; letter-spacing:-.045em; line-height:1.12; }
    .art-rule { height:1px; background:linear-gradient(90deg,#c59af4,transparent); margin:.4rem 0; }
    .art-orbit { position:absolute; z-index:2; width:65px; height:65px; right:8%; top:13%;
      color:#deb4ef; filter:drop-shadow(0 10px 24px rgba(186,128,255,.4));
      transform:rotate(12deg); }
    .landing-rule { height:1px; background:linear-gradient(90deg,transparent,#3b3248,transparent); margin:1.5rem 0 1rem; }
    .landing-caption { color:#928aa0; text-align:center; font-size:.62rem; letter-spacing:.2em; font-weight:800; }
    .landing-step { padding:1rem .25rem; }
    .landing-step > span { color:#bd91f2; font-size:.7rem; font-weight:800; letter-spacing:.15em; }
    .landing-step h3 { margin:.65rem 0 .35rem; font-size:1.05rem; }
    .landing-step p { color:#9794a3; font-size:.84rem; line-height:1.6; }
    .step-heading { margin:1.65rem 0 .45rem; font-size:1.03rem; font-weight:760; }
    .step-heading span,.workflow-line span { color:#c084fc; margin-right:.55rem; }
    .choice-copy { display:flex; align-items:center; gap:.7rem; min-height:3rem;
      font-size:.94rem; font-weight:730; }
    .choice-icon { color:#c7a6fa; width:1.35rem; height:1.35rem; }
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
      transition:transform .28s ease, box-shadow .28s ease,border-color .28s ease;
      transform-style:preserve-3d;
    }
    .movie-card:hover {
      border-color:rgba(192,38,211,.62);
      box-shadow:0 24px 48px rgba(0,0,0,.42),0 0 30px rgba(139,92,246,.22);
      transform:perspective(1000px) rotateX(1.2deg) rotateY(-1deg) translateY(-5px);
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
    .dialog-poster,.dialog-poster-placeholder { display:block; width:100%; max-width:210px;
      aspect-ratio:2/3; max-height:360px; object-fit:cover; border-radius:12px;
      background:linear-gradient(145deg,#322143,#17151e); margin:0 auto; }
    .dialog-poster-placeholder { display:flex; justify-content:center; align-items:center;
      color:#d8c8ed; font-size:.68rem; font-weight:800; letter-spacing:.18em; text-align:center; }
    [data-testid="stDialog"] { background:#111118; border:1px solid #332744; }
    .detail-badges { display:flex; flex-wrap:wrap; gap:.35rem; margin:.35rem 0 .6rem; }
    .detail-badges span { padding:.28rem .55rem; border-radius:999px; color:#e7d8fb;
      background:rgba(89,58,119,.3); border:1px solid rgba(181,138,246,.3); font-size:.68rem; }
    .credit-list { margin:.55rem 0 .45rem; }
    .credit-row { display:grid; grid-template-columns:5rem 1fr; gap:.45rem;
      padding:.25rem 0; color:#aaa6b5; font-size:.74rem; line-height:1.45; }
    .credit-row strong { color:#ded9e7; font-weight:550; }
    .synopsis-label { color:#d7c3ef; font-size:.76rem; font-weight:750; margin:.6rem 0 .35rem; }
    .synopsis-scroll { max-height:160px; overflow-y:auto; padding:.7rem .75rem;
      color:#b7b3c1; background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08);
      border-radius:9px; font-size:.82rem; line-height:1.65; }
    [data-testid="stExpander"] summary { font-family:'Inter','Plus Jakarta Sans',sans-serif!important; }
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
      position:relative; width:.85rem; height:.85rem; font-size:0!important;
      color:transparent!important; overflow:visible;
    }
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"]::after {
      content:""; position:absolute; top:50%; left:50%; width:.38rem; height:.38rem;
      border-right:1.5px solid #bca3d8; border-bottom:1.5px solid #bca3d8;
      transform:translate(-65%,-65%) rotate(-45deg);
    }
    [data-testid="stExpander"] details[open] summary [data-testid="stIconMaterial"]::after {
      transform:translate(-65%,-65%) rotate(45deg);
    }
    .stAlert { border-radius:10px; }
    @media (prefers-reduced-motion: reduce) {
      .stApp { animation:none; }
      .stApp::after { display:none; }
    }
    @media (max-width:650px) {
      [data-testid="stMainBlockContainer"] { padding:1.5rem 1rem 3rem; }
      .movie-copy { padding:.75rem; }
      .cinema-art { min-height:320px; }
      .about-title { font-size:2.8rem!important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <script>
    (() => {
      const parentWindow = window.parent;
      let parentDocument;
      try {
        parentDocument = parentWindow.document;
      } catch {
        return;
      }
      const canvas = parentDocument.querySelector(".stApp");
      if (!canvas) return;
      if (
        parentWindow.matchMedia("(prefers-reduced-motion: reduce)").matches ||
        parentWindow.matchMedia("(hover: none)").matches ||
        parentWindow.navigator.maxTouchPoints > 0
      ) return;

      const previousHandlers = parentWindow.__genreGenieCursorHandlers;
      if (previousHandlers) {
        parentWindow.removeEventListener("pointermove", previousHandlers.move);
        parentWindow.removeEventListener("pointerleave", previousHandlers.leave);
      }
      const moveHandler = (event) => {
        if (event.pointerType === "touch") return;
        canvas.style.setProperty("--cursor-x", `${event.clientX}px`);
        canvas.style.setProperty("--cursor-y", `${event.clientY}px`);
      };
      const leaveHandler = () => {
        canvas.style.setProperty("--cursor-x", "-500px");
        canvas.style.setProperty("--cursor-y", "-500px");
      };
      parentWindow.addEventListener("pointermove", moveHandler, { passive: true });
      parentWindow.addEventListener("pointerleave", leaveHandler, { passive: true });
      parentWindow.__genreGenieCursorHandlers = {
        move: moveHandler,
        leave: leaveHandler,
      };
    })();
    </script>
    """,
    height=0,
)

if "page" not in st.session_state:
    st.session_state["page"] = "About & Model"
if "selected_genres" not in st.session_state:
    st.session_state["selected_genres"] = []

catalog = load_movie_catalog()
catalog_fingerprint = _catalog_fingerprint()
model_state = load_regression_model()
model_rmse = load_model_rmse()
movie_labels = _load_movie_labels(catalog_fingerprint)

with st.sidebar:
    st.markdown(
        '<div class="brand-title">Genre<span class="brand-gradient">Genie</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-tagline">A thoughtful guide to your next film.</div>',
        unsafe_allow_html=True,
    )
    for page_name, label, icon_name in (
        ("About & Model", "About GenreGenie", "info"),
        ("Discover", "Discover", "search"),
        ("Recommendations", "My Recommendations", "bookmark"),
    ):
        icon_column, button_column = st.columns([0.16, 0.84], gap="small")
        with icon_column:
            st.markdown(
                f'<div class="sidebar-nav-icon">{_icon_svg(icon_name)}</div>',
                unsafe_allow_html=True,
            )
        with button_column:
            if st.button(
                label,
                key=f"nav_{page_name}",
                type=(
                    "primary"
                    if st.session_state["page"] == page_name
                    else "secondary"
                ),
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

    st.markdown("---")
    st.caption("Set your rating, list size, and discovery balance on the Discover page.")
    if "rating_source" in catalog and catalog["rating_source"].eq("TMDB").any():
        tmdb_movie_count = int(catalog["rating_source"].eq("TMDB").sum())
        st.caption(
            f"Recent catalog data and ratings supplied by TMDB ({tmdb_movie_count} titles). "
            "This product uses the TMDB API but is not endorsed or certified by TMDB."
        )

if st.session_state["page"] == "Discover":
    _render_discover(catalog, movie_labels)
elif st.session_state["page"] == "Recommendations":
    with st.expander("Adjust your shortlist", expanded=False):
        _render_recommendation_controls()
    _render_recommendations(
        catalog=catalog,
        model=model_state,
        lambda_value=float(st.session_state.get("lambda_val", 1.0)),
        minimum_rating=float(st.session_state.get("min_rating", 3.0)),
        top_n=int(st.session_state.get("count", 10)),
    )
else:
    _render_about(model_state, model_rmse, len(catalog))
