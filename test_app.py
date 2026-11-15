"""Focused UI and upload-validation tests for GenreGenie."""

from pathlib import Path

import movie_metadata
import pandas as pd
import recommendation_engine
from streamlit.testing.v1 import AppTest

from input_validation import validate_watchlist_csv
from recommendation_engine import prepare_catalog


APP_PATH = Path(__file__).resolve().parent / "app.py"


def _widget_with_key(widgets, key):
    return next(widget for widget in widgets if widget.key == key)


def _run_app():
    return AppTest.from_file(str(APP_PATH)).run(timeout=30)


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
            }
            for movie in movies
        },
    )


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
    assert any("test plot summary" in element.value for element in app.markdown)
    assert any("Showing 6 picks" in element.value for element in app.markdown)
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
    assert ranked_counts == [6, 30]


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

    assert app.session_state["selected_genres"] == ["Discover Comedy"]
    assert first_ids.isdisjoint(second_ids)


def test_current_sidebar_values_reach_ranker_and_criteria_bar(monkeypatch):
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
    _widget_with_key(app.select_slider, "count").set_value(8)
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

    assert not app.exception
    assert app.session_state["active_filters"]["genres"] == ["Romance"]
    assert len(displayed) == len(movie_ids)
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
    assert any("Find More Like This" in item.label for item in app.button)
    _widget_with_key(app.button, f"find_more_like_{movie_id}").click()
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["active_filters"]["genres"] == []
    assert app.session_state["active_filters"]["loved_movies"] == [movie_id]


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
