import json

import pandas as pd

from recommendation_engine import prepare_catalog, rank_recommendations
from tmdb_catalog import (
    TMDB_MOVIE_ID_OFFSET,
    append_tmdb_movies,
    iter_tmdb_results,
    load_or_build_tmdb_full_catalog,
    merge_tmdb_catalogs,
    prepare_tmdb_movies,
)


def _horror_movie(movie_id=1001, **overrides):
    movie = {
        "id": movie_id,
        "title": f"New Haunt {movie_id}",
        "release_date": "2024-10-25",
        "genre_ids": [27, 53],
        "vote_average": 8.0,
        "vote_count": 100,
        "poster_path": "/new-haunt.jpg",
        "overview": "A late-night haunting.",
    }
    movie.update(overrides)
    return movie


def test_tmdb_records_normalize_across_genres_and_release_years():
    movies = prepare_tmdb_movies(
        [
            _horror_movie(),
            _horror_movie(1002, release_date="2019-10-25"),
            _horror_movie(
                1003,
                release_date="2008-05-01",
                genre_ids=[878, 10751],
                genres=[
                    {"id": 878, "name": "Science Fiction"},
                    {"id": 10751, "name": "Family"},
                ],
            ),
        ]
    )

    assert len(movies) == 3
    movie = movies.iloc[0]
    assert movie["movieId"] == TMDB_MOVIE_ID_OFFSET + 1001
    assert movie["title"] == "New Haunt 1001 (2024)"
    assert movie["genres"] == "Horror|Thriller"
    assert movie["rating_source"] == "TMDB"
    assert movie["rating_prediction"] == 3.667
    assert movie["poster_url"] == "https://image.tmdb.org/t/p/w500/new-haunt.jpg"
    assert movies.iloc[1]["title"] == "New Haunt 1002 (2019)"
    assert movies.iloc[2]["genres"] == "Sci-Fi|Family"


def test_full_load_results_stream_across_small_chunks(tmp_path):
    records = [
        {"id": 1, "title": 'Movie with "quotes" and {braces}'},
        {"id": 2, "title": "Another movie", "genres": [{"id": 27, "name": "Horror"}]},
    ]
    path = tmp_path / "full_load.json"
    path.write_text(
        json.dumps({"pulled_at": "2026-10-01", "results": records}),
        encoding="utf-8",
    )

    assert list(iter_tmdb_results(path, chunk_size=13)) == records


def test_full_load_cache_is_reused_and_incremental_rows_preserve_details(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "full_load.json"
    source_path.write_text(
        json.dumps({"results": [_horror_movie(director=["Director"])]}),
        encoding="utf-8",
    )
    cache_path = tmp_path / "processed" / "catalog.csv"
    manifest_path = tmp_path / "processed" / "catalog.json"
    full_catalog = load_or_build_tmdb_full_catalog(
        [source_path], cache_path, manifest_path
    )
    monkeypatch.setattr(
        "tmdb_catalog.iter_tmdb_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit should not parse the full-load JSON")
        ),
    )
    cached_catalog = load_or_build_tmdb_full_catalog(
        [source_path], cache_path, manifest_path
    )
    updates = prepare_tmdb_movies(
        [_horror_movie(1001, director=[], overview="", poster_path=None)]
    )

    merged = merge_tmdb_catalogs(cached_catalog, updates)

    pd.testing.assert_frame_equal(full_catalog, cached_catalog)
    assert len(merged) == 1
    assert merged.iloc[0]["director"] == "Director"
    assert merged.iloc[0]["overview"] == "A late-night haunting."


def test_tmdb_movie_ratings_and_artwork_reach_the_ranker():
    catalog = prepare_catalog(
        append_tmdb_movies(
            pd.DataFrame(columns=["movieId", "title", "genres"]),
            [_horror_movie()],
        )
    )

    recommendations = rank_recommendations(
        catalog=catalog,
        predictions=None,
        prompt="",
        selected_genres=["Horror"],
        top_n=1,
        minimum_rating=4.0,
    )

    assert len(recommendations) == 1
    movie = recommendations[0]
    assert movie["rating_source"] == "TMDB"
    assert movie["tmdb_id"] == 1001
    assert movie["poster_url"] == "https://image.tmdb.org/t/p/w500/new-haunt.jpg"
    assert movie["overview"] == "A late-night haunting."


def test_tmdb_movies_respect_selected_genres():
    catalog = prepare_catalog(
        append_tmdb_movies(
            pd.DataFrame(columns=["movieId", "title", "genres"]),
            [
                _horror_movie(2001, genre_ids=[10749]),
                _horror_movie(2002, genre_ids=[80]),
            ],
        )
    )

    recommendations = rank_recommendations(
        catalog=catalog,
        predictions=None,
        prompt="",
        selected_genres=["Romance"],
        top_n=5,
    )

    assert [movie["tmdb_id"] for movie in recommendations] == [2001]


def test_similar_movie_threshold_filters_before_returning_candidates():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                {"movieId": 1, "title": "Seed (2020)", "genres": "Horror|Mystery"},
                {"movieId": 2, "title": "Close Match (2021)", "genres": "Horror|Mystery"},
                {"movieId": 3, "title": "Loose Match (2022)", "genres": "Horror"},
            ]
        )
    )

    recommendations = rank_recommendations(
        catalog=catalog,
        predictions=None,
        prompt="",
        seed_movie_ids={1},
        top_n=10,
        minimum_match_pct=75,
    )

    assert [movie["title"] for movie in recommendations] == ["Close Match"]
    assert all(movie["match_pct"] >= 75 for movie in recommendations)


def test_low_vote_tmdb_average_is_pulled_toward_neutral_prior():
    movies = prepare_tmdb_movies([_horror_movie(vote_count=1)])

    assert movies.iloc[0]["rating_prediction"] == 3.02


def test_linked_movielens_titles_are_excluded_from_tmdb_additions():
    movies = append_tmdb_movies(
        pd.DataFrame(columns=["movieId", "title", "genres"]),
        [_horror_movie(1001), _horror_movie(1002)],
        excluded_tmdb_ids={1001},
    )

    assert movies["tmdb_id"].tolist() == [1002]


def test_full_load_credits_are_retained_for_movie_details():
    movies = prepare_tmdb_movies(
        [
            _horror_movie(
                director=[{"name": "Test Director"}],
                top_cast=[{"name": "Actor One"}, {"name": "Actor Two"}],
                production_companies=[{"name": "Test Films"}],
            )
        ]
    )

    movie = movies.iloc[0]
    assert movie["director"] == ["Test Director"]
    assert movie["cast"] == ["Actor One", "Actor Two"]
    assert movie["production"] == ["Test Films"]
