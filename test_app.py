"""Focused UI and upload-validation tests for GenreGenie."""

from pathlib import Path
import re

import movie_metadata
import pandas as pd
import recommendation_engine
from streamlit.testing.v1 import AppTest

from input_validation import validate_watchlist_csv
from recommendation_engine import prepare_catalog
from tmdb_catalog import TMDB_MOVIE_ID_OFFSET


APP_PATH = Path(__file__).resolve().parent / "app.py"


def _widget_with_key(widgets, key):
    return next(widget for widget in widgets if widget.key == key)


def _run_app(start_discover=True):
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    if start_discover and app.session_state["page"] == "About & Model":
        _widget_with_key(app.button, "get_started").click()
        app.run(timeout=30)
    return app


def _stub_metadata(monkeypatch):
    monkeypatch.setattr(
        movie_metadata,
        "fetch_movie_details_batch",
        lambda movies: {
            int(movie["movie_id"]): {
                "poster_url": "",
                "overview": "A test plot summary for this movie.",
                "full_overview": (
                    "A test plot summary for this movie. "
                    "The complete synopsis continues beyond the card hook."
                ),
                "director": "Test Director",
                "cast": ["Actor One", "Actor Two", "Actor Three"],
                "production": "Test Pictures",
            }
            for movie in movies
        },
    )


def _stub_movie_lens_ranker(monkeypatch):
    def rank_movies(
        catalog,
        predictions,
        prompt,
        mood_labels=(),
        seed_movie_ids=None,
        excluded_movie_ids=None,
        top_n=6,
        minimum_rating=0.5,
        lambda_val=1.0,
        shuffle_seed=None,
        seen_movie_ids=None,
        selected_genres=None,
        era="Any Era",
        minimum_match_pct=None,
    ):
        del predictions, prompt, mood_labels, top_n, minimum_rating, lambda_val
        del shuffle_seed, seen_movie_ids, era, minimum_match_pct
        excluded = set(seed_movie_ids or ()) | set(excluded_movie_ids or ())
        seed_genres = set()
        if seed_movie_ids:
            seed_row = catalog[catalog["movieId"].isin(seed_movie_ids)].iloc[0]
            seed_genres = set(seed_row["genre_tags"])
        available = catalog[
            ~catalog["movieId"].isin(excluded)
            & catalog["movieId"].lt(TMDB_MOVIE_ID_OFFSET)
        ]
        if seed_genres:
            available = available[
                available["genre_tags"].map(
                    lambda genres: bool(seed_genres.intersection(genres))
                )
            ]
        elif selected_genres:
            available = available[
                available["genre_tags"].map(
                    lambda genres: bool(set(selected_genres).issubset(genres))
                )
            ]
        row = available.iloc[0]
        return [
            {
                "movie_id": int(row["movieId"]),
                "title": row["display_title"],
                "formatted_title": row["formatted_title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "genres": list(row["genre_tags"]),
                "predicted_rating": 4.0,
                "rating_source": "SVD",
                "tmdb_id": None,
                "poster_url": "",
                "overview": "",
                "match_pct": 90,
                "rank_score": 4.0,
                "genre_score": 1.0,
                "era_score": None,
                "matching_groups": 1,
                "seen_before": False,
                "pool_size": 1,
                "reason": "A test selection.",
                "fallback_notice": "",
            }
        ]

    monkeypatch.setattr(recommendation_engine, "rank_recommendations", rank_movies)


def test_valid_text_query_renders_recommendations(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["page"] == "Recommendations"
    assert any("movie-card" in element.value for element in app.markdown)
    assert any("movie-overview" in element.value for element in app.markdown)
    assert any("Showing 10 picks" in element.value for element in app.markdown)
    assert any("Genres: Romance" in element.value for element in app.markdown)
    assert any("Min Rating: ≥3.0" in element.value for element in app.markdown)
    assert any("λ: 1.0" in element.value for element in app.markdown)


def test_empty_text_query_shows_helpful_warning():
    app = _run_app()
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert any("Pick at least one vibe" in element.value for element in app.warning)


def test_about_page_and_shuffle_navigation(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "nav_About & Model").click()
    app.run(timeout=30)

    assert app.session_state["page"] == "About & Model"
    assert any("Quality, security" in element.value for element in app.markdown)

    _widget_with_key(app.button, "nav_Discover").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)
    assert app.session_state["page"] == "Recommendations"
    assert app.session_state["shuffle_seed"] == 0

    _widget_with_key(app.button, "shuffle_picks").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["shuffle_seed"] == 1


def test_about_is_the_default_landing_page():
    app = _run_app(start_discover=False)

    assert not app.exception
    assert app.session_state["page"] == "About & Model"
    assert any("About GenreGenie" in element.value for element in app.markdown)
    assert any(button.key == "get_started" for button in app.button)
    assert any("A thoughtful guide to your next film." in element.value for element in app.markdown)


def test_discover_exposes_every_catalog_genre_and_granular_eras():
    app = _run_app()
    catalog = prepare_catalog(
        pd.read_csv(APP_PATH.parent / "data" / "cleaned" / "movies_cleaned.csv")
    )
    available_genres = {
        genre for tags in catalog["genre_tags"] for genre in tags
    }
    visible_buttons = {button.label for button in app.button}

    assert {"Horror", "Mystery", "Crime", "Fantasy", "Animation"} <= available_genres
    assert {"Horror", "Mystery", "Crime", "Fantasy", "Animation"} <= visible_buttons
    assert any("hero-clapperboard" in item.value for item in app.markdown)
    assert any(
        "horror titles from 2020 onward are in the catalog" in item.value
        for item in app.caption
    )
    assert _widget_with_key(app.slider, "min_rating").min == 1.0
    assert _widget_with_key(app.slider, "count").min == 1
    assert _widget_with_key(app.slider, "count").max == 30
    assert _widget_with_key(app.slider, "count").value == 10
    assert _widget_with_key(app.slider, "lambda_val").min == 0.1
    assert _widget_with_key(app.radio, "selected_era").options == [
        "Any Era",
        "Classic (Before 1980)",
        "1980s",
        "1990s",
        "2000–2009",
        "2010–2019",
        "2020 & Newer",
    ]


def test_recommendation_error_displays_exact_exception(monkeypatch):
    def fail_rank(*_args, **_kwargs):
        raise ValueError("invalid test schema")

    monkeypatch.setattr(recommendation_engine, "rank_recommendations", fail_rank)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert any("ValueError: invalid test schema" in item.value for item in app.error)


def test_shuffle_supports_a_ranker_without_shuffle_seed_parameter(
    monkeypatch,
):
    ranked_counts = []

    def legacy_ranker(
        catalog,
        predictions,
        prompt,
        mood_labels=(),
        seed_movie_ids=None,
        excluded_movie_ids=None,
        top_n=6,
        minimum_rating=0.5,
        lambda_val=1.0,
    ):
        ranked_counts.append(top_n)
        return []

    monkeypatch.setattr(recommendation_engine, "rank_recommendations", legacy_ranker)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "shuffle_picks").click()
    app.run(timeout=30)

    assert not app.exception
    assert not app.error
    assert ranked_counts == [10, 30]


def test_changing_genre_preferences_avoids_recently_shown_movies(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)
    first_ids = set(app.session_state["last_recommendation_ids"])

    _widget_with_key(app.button, "back_to_choices").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "genre_choice_1").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)
    second_ids = set(app.session_state["last_recommendation_ids"])

    assert app.session_state["selected_genres"] == ["Comedy"]
    assert first_ids.isdisjoint(second_ids)


def test_discover_values_reach_ranker_and_criteria_bar(monkeypatch):
    _stub_metadata(monkeypatch)
    original_ranker = recommendation_engine.rank_recommendations
    ranker_calls = []

    def capture_ranker(*args, **kwargs):
        ranker_calls.append(kwargs.copy())
        return original_ranker(*args, **kwargs)

    monkeypatch.setattr(recommendation_engine, "rank_recommendations", capture_ranker)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.slider, "lambda_val").set_value(4.2)
    _widget_with_key(app.slider, "min_rating").set_value(4.0)
    _widget_with_key(app.slider, "count").set_value(8)
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert ranker_calls[-1]["lambda_val"] == 4.2
    assert ranker_calls[-1]["minimum_rating"] == 4.0
    assert ranker_calls[-1]["top_n"] == 8
    assert any("Showing 8 picks" in item.value for item in app.markdown)
    assert any("Min Rating: ≥4.0" in item.value for item in app.markdown)
    assert any("λ: 4.2" in item.value for item in app.markdown)


def test_active_filters_are_the_genres_rendered_and_ranked(monkeypatch):
    _stub_metadata(monkeypatch)
    ranker_calls = []
    original_ranker = recommendation_engine.rank_recommendations

    def capture_ranker(*args, **kwargs):
        ranker_calls.append(kwargs.copy())
        return original_ranker(*args, **kwargs)

    monkeypatch.setattr(recommendation_engine, "rank_recommendations", capture_ranker)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "genre_choice_4").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["active_filters"] == {
        "genres": ["Romance", "Drama"],
        "era": "Any Era",
        "loved_movies": [],
        "keywords": "",
    }
    assert ranker_calls[-1]["selected_genres"] == ["Romance", "Drama"]
    assert any(
        "Romance" in item.value and "Drama" in item.value
        for item in app.markdown
        if "filter-pills" in item.value
    )
    assert any(
        "Genres: Romance, Drama" in item.value for item in app.markdown
    )


def test_romance_choice_only_renders_movies_with_romance(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    catalog = prepare_catalog(
        pd.read_csv(APP_PATH.parent / "data" / "cleaned" / "movies_cleaned.csv")
    )
    movie_ids = set(app.session_state["last_recommendation_ids"])
    displayed = catalog[catalog["movieId"].isin(movie_ids)]
    tmdb_movie_ids = {
        movie_id for movie_id in movie_ids if movie_id >= TMDB_MOVIE_ID_OFFSET
    }

    assert not app.exception
    assert app.session_state["active_filters"]["genres"] == ["Romance"]
    assert len(displayed) + len(tmdb_movie_ids) == len(movie_ids)
    assert all("Romance" in genres for genres in displayed["genre_tags"])


def test_recommendations_navigation_submits_current_discover_filters(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "nav_Recommendations").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["page"] == "Recommendations"
    assert app.session_state["active_filters"]["genres"] == ["Romance"]
    assert any("Genres: Romance" in item.value for item in app.markdown)


def test_full_synopsis_dialog_shows_uncut_plot(monkeypatch):
    _stub_metadata(monkeypatch)
    _stub_movie_lens_ranker(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    movie_id = app.session_state["last_recommendation_ids"][0]
    _widget_with_key(app.button, f"full_synopsis_{movie_id}").click()
    app.run(timeout=30)

    assert not app.exception
    assert any(
        "The complete synopsis continues beyond the card hook." in item.value
        for item in app.markdown
    )
    assert any("Test Director" in item.value for item in app.markdown)
    assert any("Actor One" in item.value for item in app.markdown)
    assert any("Test Pictures" in item.value for item in app.markdown)
    assert any("Find Similar" in item.label for item in app.button)
    _widget_with_key(app.button, f"find_more_like_{movie_id}").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["active_filters"]["genres"] == []
    assert app.session_state["active_filters"]["loved_movies"] == [movie_id]
    assert app.session_state["similarity_threshold"] == 75
    match_values = [
        int(value)
        for element in app.markdown
        for value in re.findall(r'poster-match">(\d+)%', element.value)
    ]
    assert match_values
    assert all(match_value >= 75 for match_value in match_values)
    catalog = prepare_catalog(
        pd.read_csv(APP_PATH.parent / "data" / "cleaned" / "movies_cleaned.csv")
    )
    reference_title = catalog.loc[
        catalog["movieId"] == movie_id, "formatted_title"
    ].iloc[0]
    assert any(
        f"Similar to: {reference_title}" in element.value
        for element in app.markdown
        if "filter-pills" in element.value
    )
    assert not any(
        "no more catalog titles matched the relaxed filters" in element.value.casefold()
        for element in app.markdown
    )


def test_recommendation_cards_defer_remote_metadata_until_details_requested(monkeypatch):
    metadata_requests = []

    def tracked_metadata_batch(movies):
        metadata_requests.append([int(movie["movie_id"]) for movie in movies])
        return {
            int(movie["movie_id"]): {
                "poster_url": "",
                "overview": "A test plot summary.",
                "full_overview": "A full test plot summary.",
                "director": "Test Director",
                "cast": ["Actor One"],
                "production": "Test Pictures",
            }
            for movie in movies
        }

    monkeypatch.setattr(
        movie_metadata, "fetch_movie_details_batch", tracked_metadata_batch
    )
    _stub_movie_lens_ranker(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert metadata_requests == []

    movie_id = app.session_state["last_recommendation_ids"][0]
    _widget_with_key(app.button, f"full_synopsis_{movie_id}").click()
    app.run(timeout=30)

    assert not app.exception
    assert metadata_requests == [[movie_id]]


def test_movie_card_find_similar_enforces_match_threshold(monkeypatch):
    _stub_metadata(monkeypatch)
    app = _run_app()
    _widget_with_key(app.button, "genre_choice_0").click()
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    movie_id = app.session_state["last_recommendation_ids"][0]
    _widget_with_key(app.button, f"find_similar_{movie_id}").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["similarity_threshold"] == 75
    match_values = [
        int(value)
        for element in app.markdown
        for value in re.findall(r'poster-match">(\d+)%', element.value)
    ]
    assert match_values
    assert all(match_value >= 75 for match_value in match_values)


def test_invalid_csv_upload_reports_schema_error():
    app = _run_app()
    _widget_with_key(app.file_uploader, "uploaded_file").upload(
        "watchlist.csv",
        b"movie,title\n1,The Matrix\n",
        "text/csv",
    )
    app.run(timeout=30)
    _widget_with_key(app.button, "find_movies").click()
    app.run(timeout=30)

    assert not app.exception
    assert any("missing required cleaned-dataset columns" in item.value for item in app.error)
    watchlist, error, _ = validate_watchlist_csv(b"movie,title\n1,The Matrix\n")
    assert watchlist is None
    assert error is not None
    assert "movieId" in error
