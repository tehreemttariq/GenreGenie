"""Regression tests for GenreGenie's metadata filters and title matching."""

import io

import pandas as pd
import pytest
from PIL import Image

import recommendation_engine
from recommendation_engine import (
    detect_poster_movie,
    find_prompt_movie_ids,
    format_movie_title,
    fuzzy_match_movie,
    parse_preferences,
    prepare_catalog,
    rank_recommendations,
)


@pytest.fixture
def mini_catalog():
    return prepare_catalog(
        pd.DataFrame(
            [
                (1, "How to Lose a Guy in 10 Days (2003)", "Comedy|Romance"),
                (2, "Two Weeks Notice (2002)", "Comedy|Romance"),
                (3, "The Notebook (2004)", "Drama|Romance"),
                (4, "Romantic Comedy (2012)", "Comedy|Romance"),
                (5, "Heat (1995)", "Action|Crime|Thriller"),
                (6, "Notebook, The (2004)", "Drama|Romance"),
                (7, "Modern Love (2020)", "Comedy|Romance"),
            ],
            columns=["movieId", "title", "genres"],
        )
    )


def test_genre_synonyms_and_old_keyword_apply_strict_filters(mini_catalog):
    profile = parse_preferences("romance, funny, old")

    assert profile.year_max == 2004
    assert {next(iter(group)) for group in profile.genre_groups} == {
        "Romance",
        "Comedy",
    }

    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions={movie_id: 4.0 for movie_id in range(1, 8)},
        prompt="romance, funny, old",
        top_n=2,
        minimum_rating=3.0,
    )

    assert {movie["movie_id"] for movie in recommendations} == {1, 2}
    assert all(movie["year"] < 2005 for movie in recommendations)
    assert all(
        {"Comedy", "Romance"}.issubset(movie["genres"])
        for movie in recommendations
    )
    assert all(
        "matches matches" not in movie["reason"].casefold()
        for movie in recommendations
    )


def test_era_parser_and_title_article_formatting(mini_catalog):
    profile = parse_preferences("90s nostalgia")

    assert (profile.year_min, profile.year_max) == (1990, 1999)
    assert format_movie_title("Notebook, The (2004)") == "The Notebook (2004)"
    assert find_prompt_movie_ids(
        "We watched The Notebook last night", mini_catalog
    ) == [3, 6]


def test_empty_prompt_does_not_build_a_title_index(mini_catalog, monkeypatch):
    def fail_if_indexed(_catalog):
        raise AssertionError("an empty prompt should not scan every movie title")

    monkeypatch.setattr(recommendation_engine, "_title_aliases", fail_if_indexed)

    assert find_prompt_movie_ids("", mini_catalog) == []


def test_title_alias_index_is_reused_for_repeated_searches(mini_catalog):
    first = recommendation_engine._title_aliases(mini_catalog)

    assert recommendation_engine._title_aliases(mini_catalog) is first


@pytest.mark.parametrize(
    ("raw_title", "formatted_title"),
    [
        (
            "Man with the Movie Camera, The (Chelovek s kino-apparatom) (1929)",
            "The Man with the Movie Camera (1929)",
        ),
        (
            "Illusionist, The (L'illusionniste) (2010)",
            "The Illusionist (2010)",
        ),
        (
            "Maltese Falcon, The (a.k.a. Dangerous Female) (1931)",
            "The Maltese Falcon (1931)",
        ),
    ],
)
def test_format_title_removes_parenthetical_aliases(raw_title, formatted_title):
    assert format_movie_title(raw_title) == formatted_title


@pytest.mark.parametrize(
    ("prompt", "expected_range"),
    [
        ("80s action movies", (1980, 1989)),
        ("90s nostalgia", (1990, 1999)),
        ("2000s romantic comedies", (2000, 2009)),
    ],
)
def test_decade_keywords_create_release_year_ranges(prompt, expected_range):
    profile = parse_preferences(prompt)

    assert (profile.year_min, profile.year_max) == expected_range


def test_love_synonym_does_not_misread_a_common_verb():
    assert parse_preferences("love").genre_groups == (frozenset({"Romance"}),)
    assert parse_preferences("I love sci-fi").genre_groups == (
        frozenset({"Sci-Fi"}),
    )


def test_classic_romance_mood_applies_romance_and_pre_2005_filters(mini_catalog):
    profile = parse_preferences("", ["Classic Romance (<2005)"])

    assert profile.year_max == 2004
    assert profile.genre_groups == (frozenset({"Romance"}),)
    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions={movie_id: 4.0 for movie_id in range(1, 8)},
        prompt="",
        mood_labels=["Classic Romance (<2005)"],
        top_n=4,
    )
    assert all(movie["year"] < 2005 for movie in recommendations)
    assert all("Romance" in movie["genres"] for movie in recommendations)


@pytest.mark.parametrize(
    ("era", "expected_range"),
    [
        ("Before 1995", (None, 1994)),
        ("1995–2005", (1995, 2005)),
        ("2006+", (2006, None)),
    ],
)
def test_discovery_era_ranges(era, expected_range):
    profile = parse_preferences("", [era])

    assert (profile.year_min, profile.year_max) == expected_range


@pytest.mark.parametrize(
    ("era", "expected_range"),
    [
        ("Classic (Before 1980)", (None, 1979)),
        ("1980s", (1980, 1989)),
        ("1990s", (1990, 1999)),
        ("2000–2009", (2000, 2009)),
        ("2010–2019", (2010, 2019)),
        ("2020 & Newer", (2020, None)),
    ],
)
def test_granular_discovery_era_ranges(era, expected_range):
    profile = parse_preferences("", [era])

    assert (profile.year_min, profile.year_max) == expected_range


def test_discovery_genres_are_strict_and_intersect(mini_catalog):
    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions={movie_id: 4.5 for movie_id in range(1, 8)},
        prompt="",
        mood_labels=["Discover Romance", "Discover Comedy"],
        top_n=4,
    )

    assert {movie["movie_id"] for movie in recommendations} == {1, 2, 4, 7}
    assert all(
        {"Romance", "Comedy"}.issubset(movie["genres"])
        for movie in recommendations
    )


def test_selected_genres_use_and_then_or_without_unfiltered_fallback():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (1, "Romance Only (2000)", "Romance"),
                (2, "Drama Only (2001)", "Drama"),
                (3, "Romantic Drama (2002)", "Romance|Drama"),
                (4, "Action Only (2003)", "Action"),
            ],
            columns=["movieId", "title", "genres"],
        )
    )
    recommendations = rank_recommendations(
        catalog=catalog,
        predictions={movie_id: 4.0 for movie_id in range(1, 5)},
        prompt="",
        selected_genres=["Romance", "Drama"],
        top_n=4,
    )

    assert {movie["movie_id"] for movie in recommendations} == {1, 2, 3}
    assert all(movie["pool_size"] == 3 for movie in recommendations)
    assert all(movie["movie_id"] != 4 for movie in recommendations)


def test_pool_size_counts_filtered_movies_before_count_and_shuffle():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (movie_id, f"Romance {movie_id} (2000)", "Romance")
                for movie_id in range(1, 13)
            ],
            columns=["movieId", "title", "genres"],
        )
    )
    recommendations = rank_recommendations(
        catalog=catalog,
        predictions={movie_id: 4.0 for movie_id in range(1, 13)},
        prompt="",
        selected_genres=["Romance"],
        top_n=3,
        shuffle_seed=7,
    )

    assert len(recommendations) == 3
    assert {movie["pool_size"] for movie in recommendations} == {12}


def test_shuffle_samples_different_movies_from_top_30():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (movie_id, f"Adventure {movie_id} ({1980 + movie_id})", "Action|Adventure")
                for movie_id in range(1, 41)
            ],
            columns=["movieId", "title", "genres"],
        )
    )
    predictions = {movie_id: 4.0 for movie_id in range(1, 41)}
    top_candidates = rank_recommendations(
        catalog,
        predictions,
        "",
        ["Action & Adventure"],
        top_n=40,
        lambda_val=0,
    )
    top_candidate_ids = {movie["movie_id"] for movie in top_candidates}
    first = rank_recommendations(
        catalog, predictions, "", ["Action & Adventure"], top_n=6, shuffle_seed=11
    )
    refreshed = rank_recommendations(
        catalog, predictions, "", ["Action & Adventure"], top_n=6, shuffle_seed=12
    )

    assert len(first) == len(refreshed) == 6
    assert {movie["movie_id"] for movie in first} != {
        movie["movie_id"] for movie in refreshed
    }
    assert all(
        movie["movie_id"] in top_candidate_ids for movie in first + refreshed
    )


def test_unfiltered_shuffle_samples_from_the_eligible_pool():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (movie_id, f"Surprise {movie_id} ({2000 + movie_id})", "Drama", 4.0)
                for movie_id in range(1, 13)
            ],
            columns=["movieId", "title", "genres", "rating_prediction"],
        )
    )

    recommendations = rank_recommendations(
        catalog=catalog,
        predictions=None,
        prompt="",
        top_n=3,
        minimum_rating=3.0,
        shuffle_seed=7,
    )

    assert len(recommendations) == 3
    assert len({movie["movie_id"] for movie in recommendations}) == 3
    assert {movie["pool_size"] for movie in recommendations} == {12}


def test_previously_recommended_movies_are_skipped_when_fresh_matches_exist(
    mini_catalog,
):
    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions={movie_id: 4.0 for movie_id in range(1, 8)},
        prompt="",
        mood_labels=["Discover Romance", "Discover Comedy"],
        top_n=2,
        seen_movie_ids={1, 2},
    )

    assert {movie["movie_id"] for movie in recommendations} == {4, 7}


def test_higher_lambda_selects_more_varied_eras():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (1, "Early A (1980)", "Action|Adventure"),
                (2, "Early B (1981)", "Action|Adventure"),
                (3, "Early C (1982)", "Action|Adventure"),
                (4, "Early D (1983)", "Action|Adventure"),
                (5, "Early E (1984)", "Action|Adventure"),
                (6, "Late Z (2020)", "Action|Adventure|Sci-Fi"),
            ],
            columns=["movieId", "title", "genres"],
        )
    )
    predictions = {movie_id: 4.0 for movie_id in range(1, 7)}
    relevant = rank_recommendations(
        catalog, predictions, "", ["Action & Adventure"], top_n=4, lambda_val=0
    )
    diverse = rank_recommendations(
        catalog, predictions, "", ["Action & Adventure"], top_n=4, lambda_val=5
    )
    assert max(movie["year"] for movie in diverse) > max(
        movie["year"] for movie in relevant
    )


def test_catalog_normalizes_list_genres_one_hot_and_prediction_columns():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (10, "Movie A (2001)", ["Comedy", "Romance"], 4.7, 0),
                (11, "Movie B (2002)", [], 4.2, 1),
            ],
            columns=[
                "movie_id",
                "movie_title",
                "genre_list",
                "predicted_rating",
                "genre_Action",
            ],
        )
    )

    assert catalog.loc[0, "genre_tags"] == ["Comedy", "Romance"]
    assert catalog.loc[1, "genre_tags"] == ["Action"]
    assert catalog.loc[0, "_genres_norm"] == "|comedy|romance|"
    assert catalog["rating_prediction"].tolist() == [4.7, 4.2]
    recommendations = rank_recommendations(
        catalog=catalog,
        predictions=None,
        prompt="",
        top_n=2,
        minimum_rating=4.0,
    )
    assert {movie["movie_id"] for movie in recommendations} == {10, 11}


def test_catalog_accepts_one_hot_genres_without_a_genre_list():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (1, "Romance (2000)", 1, 0),
                (2, "Comedy (2001)", 0, 1),
            ],
            columns=["id", "name", "Romance", "Comedy"],
        )
    )

    assert catalog["genre_tags"].tolist() == [["Romance"], ["Comedy"]]
    assert catalog["_genres_norm"].tolist() == ["|romance|", "|comedy|"]


def test_ranker_handles_catalog_without_normalized_rating_column(mini_catalog):
    catalog = mini_catalog.drop(columns="rating_prediction")
    recommendations = rank_recommendations(
        catalog=catalog,
        predictions={1: 4.2},
        prompt="",
        top_n=2,
    )

    assert len(recommendations) == 2
    assert any(movie["movie_id"] == 1 for movie in recommendations)


def test_catalog_parses_serialized_genres_and_prediction_alias():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                {
                    "id": 8,
                    "movie_title": "A List Movie (2000)",
                    "genre_list": "['Comedy', 'Romance']",
                    "rating_prediction": 4.3,
                }
            ]
        )
    )

    assert catalog.loc[0, "genre_tags"] == ["Comedy", "Romance"]
    assert catalog.loc[0, "rating_prediction"] == 4.3


def test_unlisted_genres_are_normalized_to_no_badges():
    catalog = prepare_catalog(
        pd.DataFrame(
            [(1, "Unlisted Movie (2000)", "(no genres listed)")],
            columns=["movieId", "title", "genres"],
        )
    )
    recommendations = rank_recommendations(
        catalog=catalog,
        predictions={1: 3.5},
        prompt="",
        top_n=1,
    )

    assert recommendations[0]["genres"] == []


def test_recommendations_relax_genre_and_rating_without_unfiltered_movies(
    mini_catalog,
):
    predictions = {
        1: 4.0,
        2: 4.0,
        3: 3.8,
        4: 3.5,
        5: 4.9,
        6: 4.1,
        7: 3.2,
    }

    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions=predictions,
        prompt="romance funny old",
        top_n=4,
        minimum_rating=4.5,
    )

    assert len(recommendations) == 4
    assert all(
        {"Comedy", "Romance"}.intersection(movie["genres"])
        for movie in recommendations
    )
    assert all(movie["year"] < 2005 for movie in recommendations)
    assert all(movie["fallback_notice"] for movie in recommendations)


def test_poster_filename_fallback_fuzzy_matches_dataset_title(mini_catalog):
    expected_id, confidence = fuzzy_match_movie(
        "How_to_Lose_a_Guy_in_10_Days.jpg", mini_catalog
    )

    assert expected_id == 1
    assert confidence >= 0.68
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color="white").save(image_buffer, format="PNG")
    assert detect_poster_movie(
        image_buffer.getvalue(),
        "How_to_Lose_a_Guy_in_10_Days.png",
        mini_catalog,
    )[0] == 1


def test_match_percent_is_dynamic(mini_catalog):
    recommendations = rank_recommendations(
        catalog=mini_catalog,
        predictions={1: 4.8, 2: 3.2, 3: 4.4, 4: 4.9, 5: 4.7, 6: 4.5, 7: 4.6},
        prompt="romance funny old",
        top_n=6,
        minimum_rating=3.0,
    )

    assert len({movie["match_pct"] for movie in recommendations}) > 1
    assert all(movie["match_pct"] != 80 for movie in recommendations)


def test_find_similar_uses_match_strength_instead_of_diversity_jitter():
    catalog = prepare_catalog(
        pd.DataFrame(
            [
                (1, "Reference (2000)", "Action|Adventure"),
                (2, "Exact Match (2001)", "Action|Adventure"),
                (3, "Strong Match (2002)", "Action|Adventure|Sci-Fi"),
                (4, "Loose Match (2003)", "Action"),
                (5, "Unrelated (2004)", "Drama"),
            ],
            columns=["movieId", "title", "genres"],
        )
    )
    predictions = {1: 4.0, 2: 3.5, 3: 4.8, 4: 5.0, 5: 5.0}

    conservative = rank_recommendations(
        catalog=catalog,
        predictions=predictions,
        prompt="",
        seed_movie_ids={1},
        top_n=5,
        minimum_rating=0.5,
        lambda_val=0.0,
        minimum_match_pct=75,
    )
    high_diversity = rank_recommendations(
        catalog=catalog,
        predictions=predictions,
        prompt="",
        seed_movie_ids={1},
        top_n=5,
        minimum_rating=0.5,
        lambda_val=5.0,
        minimum_match_pct=75,
    )

    assert [movie["movie_id"] for movie in conservative] == [
        movie["movie_id"] for movie in high_diversity
    ]
    assert [movie["movie_id"] for movie in conservative] == [2, 3]
    assert all(movie["match_pct"] >= 75 for movie in conservative)
