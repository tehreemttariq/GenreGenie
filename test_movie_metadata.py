"""Tests for keyless movie metadata lookups."""

import threading
import time

import movie_metadata
import pytest


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


def _patch_session(monkeypatch, fake_get):
    class FakeSession:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            headers = dict(self.headers)
            headers.update(kwargs.pop("headers", {}))
            kwargs["headers"] = headers
            return fake_get(url, **kwargs)

    monkeypatch.setattr(movie_metadata.requests, "Session", FakeSession)


def test_wikipedia_poster_and_summary_are_cached_and_upscaled(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            200,
            {
                "thumbnail": {
                    "source": "https://upload.wikimedia.org/example/320px-poster.jpg"
                },
                "extract": "A first sentence. A second sentence. A third sentence. More.",
            },
        )

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Example Film (2004)", 2004)

    assert details["poster_url"].endswith("500px-poster.jpg")
    assert details["overview"] == "A first sentence. A second sentence."
    assert details["full_overview"] == (
        "A first sentence. A second sentence. A third sentence. More."
    )
    assert details["source"] == "Wikipedia"
    assert len(calls) == 1
    assert calls[0][1]["headers"]["User-Agent"] == (
        "GenreGenie/2.0 (educational project)"
    )
    assert calls[0][1]["timeout"] == 4


@pytest.mark.parametrize(
    ("raw_title", "year", "expected_slug"),
    [
        (
            "Man with the Movie Camera, The (Chelovek s kino-apparatom) (1929)",
            1929,
            "The_Man_with_the_Movie_Camera",
        ),
        (
            "Illusionist, The (L'illusionniste) (2010)",
            2010,
            "The_Illusionist",
        ),
        (
            "Maltese Falcon, The (a.k.a. Dangerous Female) (1931)",
            1931,
            "The_Maltese_Falcon",
        ),
    ],
)
def test_metadata_lookup_normalizes_parenthetical_aliases(
    monkeypatch, raw_title, year, expected_slug
):
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        return FakeResponse(
            200,
            {
                "thumbnail": {
                    "source": "https://upload.wikimedia.org/example/320px-poster.jpg"
                },
                "extract": "A plot summary.",
            },
        )

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    movie_metadata.fetch_movie_details(raw_title, year)

    assert calls[0].endswith(expected_slug)


@pytest.mark.parametrize(
    ("raw_title", "expected_title"),
    [
        ("Slumdog Millionaire... (2008)", "Slumdog Millionaire"),
        ("Nick and Norah's Infinite Playlist... (2008)", "Nick and Norah's Infinite Playlist"),
        ("What Happens in Vegas... (2008)", "What Happens in Vegas"),
    ],
)
def test_metadata_lookup_strips_trailing_ellipsis(
    monkeypatch, raw_title, expected_title
):
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        return FakeResponse(200, {"extract": "A plot summary."})

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    movie_metadata.fetch_movie_details(raw_title, 2008)

    assert calls[0] == (
        movie_metadata.WIKIPEDIA_SUMMARY
        + expected_title.replace(" ", "_").replace("'", "%27")
    )


def test_wikipedia_checks_base_title_before_year_specific_page(monkeypatch):
    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if url.endswith("/Example_Film"):
            return FakeResponse(404, {})
        return FakeResponse(
            200,
            {
                "thumbnail": {
                    "source": "https://upload.wikimedia.org/example/500px.jpg"
                },
                "extract": "A year-specific synopsis.",
            },
        )

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    movie_metadata.fetch_movie_details("Example Film (2004)", 2004)

    assert calls[:2] == [
        movie_metadata.WIKIPEDIA_SUMMARY + "Example_Film",
        movie_metadata.WIKIPEDIA_SUMMARY + "Example_Film_(2004_film)",
    ]


def test_wikipedia_generator_search_supplies_page_image_and_extract(monkeypatch):
    def fake_get(url, **kwargs):
        if url == movie_metadata.WIKIPEDIA_SUMMARY + "Foreign_film":
            return FakeResponse(
                200,
                {
                    "thumbnail": {
                        "source": "https://upload.wikimedia.org/example/500px.jpg"
                    },
                    "extract": "A plot from Wikipedia search.",
                },
            )
        if url.startswith(movie_metadata.WIKIPEDIA_SUMMARY):
            return FakeResponse(404, {})
        if url == movie_metadata.WIKIPEDIA_SEARCH:
            assert kwargs["params"]["action"] == "query"
            assert kwargs["params"]["list"] == "search"
            return FakeResponse(
                200,
                {"query": {"search": [{"title": "Foreign film"}]}},
            )
        return FakeResponse(404, {})

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Foreign Film (2001)", 2001)

    assert details["poster_url"].endswith("500px.jpg")
    assert details["overview"] == "A plot from Wikipedia search."


def test_itunes_fallback_supplies_artwork_overview_and_runtime(monkeypatch):
    def fake_get(url, **_kwargs):
        if url.startswith(movie_metadata.WIKIPEDIA_SUMMARY):
            return FakeResponse(404, {})
        if url == movie_metadata.WIKIPEDIA_SEARCH:
            return FakeResponse(200, {"query": {"search": []}})
        if url == movie_metadata.ITUNES_SEARCH:
            return FakeResponse(
                200,
                {
                    "results": [
                        {
                            "trackName": "Example Film",
                            "releaseDate": "2004-01-01T00:00:00Z",
                            "artworkUrl100": "https://is1-ssl.mzstatic.com/image/100x100bb.jpg",
                            "longDescription": "A real plot summary from iTunes.",
                            "trackTimeMillis": 6_000_000,
                        }
                    ]
                },
            )
        return FakeResponse(404, {})

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Example Film", 2004)

    assert "600x600bb" in details["poster_url"]
    assert details["overview"] == "A real plot summary from iTunes."
    assert details["runtime"] == "100 min"


def test_itunes_search_matches_release_year_among_results(monkeypatch):
    searches = []

    def fake_get(url, **kwargs):
        if url.startswith(movie_metadata.WIKIPEDIA_SUMMARY):
            return FakeResponse(404, {})
        if url == movie_metadata.WIKIPEDIA_SEARCH:
            return FakeResponse(200, {"query": {"search": []}})
        if url == movie_metadata.ITUNES_SEARCH:
            searches.append(kwargs["params"])
            return FakeResponse(
                200,
                {
                    "results": [
                        {
                            "trackName": "Rare Film",
                            "releaseDate": "1986-03-01T00:00:00Z",
                            "artworkUrl100": "https://is1-ssl.mzstatic.com/image/86x86bb.jpg",
                            "longDescription": "An incorrect-year result.",
                        },
                        {
                            "trackName": "Rare Film",
                            "releaseDate": "1987-04-01T00:00:00Z",
                            "artworkUrl100": "https://is1-ssl.mzstatic.com/image/100x100bb.jpg",
                            "longDescription": "A recovered movie plot.",
                        }
                    ]
                },
            )
        return FakeResponse(404, {})

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Rare Film (1987)", 1987)

    assert len(searches) == 1
    assert searches[0]["term"] == "Rare Film"
    assert searches[0]["entity"] == "movie"
    assert searches[0]["limit"] == 3
    assert "600x600bb" in details["poster_url"]
    assert details["overview"] == "A recovered movie plot."


def test_tvmaze_fallback_supplies_poster_and_synopsis(monkeypatch):
    def fake_get(url, **kwargs):
        if url == movie_metadata.TVMAZE_SEARCH:
            return FakeResponse(
                200,
                {
                    "image": {
                        "original": "https://static.tvmaze.com/poster.jpg"
                    },
                    "summary": "<p>A TVMaze synopsis.</p>",
                },
            )
        if url == movie_metadata.WIKIDATA_API:
            return FakeResponse(200, {"search": []})
        return FakeResponse(404, {})

    _patch_session(monkeypatch, fake_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Fallback Film", 1999)

    assert details["poster_url"] == "https://static.tvmaze.com/poster.jpg"
    assert details["overview"] == "A TVMaze synopsis."
    assert details["source"] == "TVMaze"


def test_generated_poster_is_returned_when_public_sources_fail(monkeypatch):
    _patch_session(
        monkeypatch, lambda *_args, **_kwargs: FakeResponse(404, {})
    )
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details("Offline Film", 1999)

    assert details["poster_url"].startswith("data:image/svg+xml;base64,")
    assert details["source"] == "generated"


def test_generated_poster_includes_initials_year_and_genre_after_http_errors(monkeypatch):
    def failing_get(*_args, **_kwargs):
        raise movie_metadata.requests.Timeout("test timeout")

    _patch_session(monkeypatch, failing_get)
    movie_metadata.fetch_movie_details.clear()

    details = movie_metadata.fetch_movie_details(
        "Notebook, The (2004)", 2004, genres=["Romance", "Drama"]
    )
    import base64

    poster_svg = base64.b64decode(
        details["poster_url"].split(",", 1)[1]
    ).decode("utf-8")
    assert ">N</text>" in poster_svg
    assert "2004" in poster_svg
    assert "Romance" in poster_svg


def test_recommendation_metadata_is_fetched_concurrently(monkeypatch):
    lock = threading.Lock()
    active = 0
    peak_active = 0
    requested_workers = []
    executor_type = movie_metadata.ThreadPoolExecutor

    def recording_executor(max_workers):
        requested_workers.append(max_workers)
        return executor_type(max_workers=max_workers)

    def fake_fetch(title, _year, _tmdb_id=None, _genres=()):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"poster_url": "", "overview": title}

    monkeypatch.setattr(movie_metadata, "fetch_movie_details", fake_fetch)
    monkeypatch.setattr(movie_metadata, "ThreadPoolExecutor", recording_executor)
    recommendations = [
        {"movie_id": movie_id, "formatted_title": f"Movie {movie_id}", "year": 2000}
        for movie_id in range(1, 5)
    ]

    details = movie_metadata.fetch_movie_details_batch(recommendations)

    assert set(details) == {1, 2, 3, 4}
    assert peak_active > 1
    assert requested_workers == [4]


def test_one_failed_movie_lookup_does_not_break_other_cards(monkeypatch):
    def partially_failing_fetch(title, year, tmdb_id=None, genres=()):
        if title == "Movie 2":
            raise RuntimeError("metadata test failure")
        return {"poster_url": "", "overview": title}

    monkeypatch.setattr(
        movie_metadata, "fetch_movie_details", partially_failing_fetch
    )
    recommendations = [
        {
            "movie_id": movie_id,
            "formatted_title": f"Movie {movie_id}",
            "year": 2000,
            "genres": ["Drama"],
        }
        for movie_id in range(1, 4)
    ]

    details = movie_metadata.fetch_movie_details_batch(recommendations)

    assert details[1]["overview"] == "Movie 1"
    assert details[3]["overview"] == "Movie 3"
    assert details[2]["poster_url"].startswith("data:image/svg+xml;base64,")
