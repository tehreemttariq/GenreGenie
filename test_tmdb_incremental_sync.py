import json
import importlib
from datetime import date, timedelta

import pytest
import requests


sync_module = importlib.import_module("data_scripts.04_tmdb_incremental_sync")


@pytest.fixture(autouse=True)
def isolate_full_load_paths(monkeypatch):
    monkeypatch.setattr(sync_module, "FULL_LOAD_PATHS", ())


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _page(page, total_pages, results):
    return FakeResponse(
        {"page": page, "total_pages": total_pages, "results": results}
    )


def test_initial_sync_fetches_all_pages_and_saves_all_genres(tmp_path):
    output = tmp_path / "tmdb_incremental_catalog.json"
    session = FakeSession(
        [
            _page(1, 2, [{"id": 101, "title": "Horror One"}]),
            _page(2, 2, [{"id": 102, "title": "Horror Two"}]),
        ]
    )
    progress = []

    result = sync_module.sync_tmdb_catalog(
        output_path=output,
        api_key="local-test-key",
        today=date(2026, 1, 1),
        session=session,
        on_progress=progress.append,
    )

    state = json.loads(output.read_text(encoding="utf-8"))
    assert result["movie_count"] == 2
    assert result["since"] == "2020-01-01"
    assert [movie["id"] for movie in state["movies"]] == [101, 102]
    assert [call[1]["params"]["page"] for call in session.calls] == [1, 2]
    assert all("with_genres" not in call[1]["params"] for call in session.calls)
    assert state["catalog_scope"] == "all_genres"
    assert "in_progress" not in state
    assert any("Loading 2 page(s)" in message for message in progress)
    assert any("page 2 of 2" in message for message in progress)


def test_interrupted_sync_resumes_at_the_next_page(tmp_path):
    output = tmp_path / "tmdb_incremental_catalog.json"
    first_session = FakeSession(
        [
            _page(1, 2, [{"id": 101, "title": "Horror One"}]),
            requests.Timeout("temporary outage"),
        ]
    )

    with pytest.raises(RuntimeError, match="Timeout"):
        sync_module.sync_tmdb_catalog(
            output_path=output,
            api_key="local-test-key",
            today=date(2026, 1, 1),
            session=first_session,
        )

    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint["in_progress"]["current_window"]["next_page"] == 2
    assert [movie["id"] for movie in checkpoint["movies"]] == [101]

    resumed_session = FakeSession(
        [_page(2, 2, [{"id": 102, "title": "Horror Two"}])]
    )
    sync_module.sync_tmdb_catalog(
        output_path=output,
        api_key="local-test-key",
        today=date(2026, 1, 1),
        session=resumed_session,
    )

    state = json.loads(output.read_text(encoding="utf-8"))
    assert [call[1]["params"]["page"] for call in resumed_session.calls] == [2]
    assert [movie["id"] for movie in state["movies"]] == [101, 102]
    assert "in_progress" not in state


def test_incremental_sync_rechecks_an_overlap_and_updates_existing_movies(tmp_path):
    output = tmp_path / "tmdb_incremental_catalog.json"
    today = date.today()
    initial_session = FakeSession([_page(1, 1, [{"id": 101, "title": "Old title"}])])
    sync_module.sync_tmdb_catalog(
        output_path=output,
        api_key="local-test-key",
        today=today,
        session=initial_session,
    )

    next_session = FakeSession([_page(1, 1, [{"id": 101, "title": "Updated title"}])])
    result = sync_module.sync_tmdb_catalog(
        output_path=output,
        api_key="local-test-key",
        today=today,
        session=next_session,
    )

    state = json.loads(output.read_text(encoding="utf-8"))
    assert result["since"] == (today - timedelta(days=30)).isoformat()
    assert next_session.calls[0][1]["params"]["primary_release_date.gte"] == result["since"]
    assert result["movie_count"] == 1
    assert state["movies"][0]["title"] == "Updated title"


def test_large_discover_windows_split_recursively_at_the_page_limit(tmp_path):
    start = date(2020, 1, 1)
    end = date(2026, 1, 1)
    first_midpoint = start + timedelta(days=(end - start).days // 2)

    class LargeWindowSession:
        def __init__(self):
            self.calls = []
            self.next_id = 100

        def get(self, _url, **kwargs):
            params = kwargs["params"]
            lower = params["primary_release_date.gte"]
            upper = params["primary_release_date.lte"]
            if (lower, upper) == (start.isoformat(), end.isoformat()):
                total_pages = 1001
            elif (lower, upper) == (
                start.isoformat(),
                first_midpoint.isoformat(),
            ):
                total_pages = 501
            else:
                total_pages = 1
            self.calls.append((dict(params), total_pages))
            payload = {
                "page": params["page"],
                "total_pages": total_pages,
                "results": (
                    [{"id": self.next_id, "title": f"Movie {self.next_id}"}]
                    if total_pages == 1
                    else []
                ),
            }
            self.next_id += 1
            return FakeResponse(payload)

    session = LargeWindowSession()
    output = tmp_path / "tmdb_incremental_catalog.json"

    result = sync_module.sync_tmdb_catalog(
        output_path=output,
        api_key="local-test-key",
        today=end,
        session=session,
    )

    leaf_windows = sorted(
        (
            date.fromisoformat(params["primary_release_date.gte"]),
            date.fromisoformat(params["primary_release_date.lte"]),
        )
        for params, total_pages in session.calls
        if total_pages <= sync_module.MAX_DISCOVER_PAGES
    )
    assert len(leaf_windows) == 3
    assert leaf_windows[0][0] == start
    assert leaf_windows[-1][1] == end
    assert all(
        left[1] + timedelta(days=1) == right[0]
        for left, right in zip(leaf_windows, leaf_windows[1:])
    )
    assert result["movie_count"] == 3


def test_jwt_in_api_key_variable_is_sent_as_a_bearer_token(tmp_path):
    token = "eyJheader.eyJpayload.signature"
    session = FakeSession([_page(1, 1, [])])

    sync_module.sync_tmdb_catalog(
        output_path=tmp_path / "tmdb_incremental_catalog.json",
        api_key=token,
        today=date(2026, 1, 1),
        session=session,
    )

    request = session.calls[0][1]
    assert request["headers"]["Authorization"] == f"Bearer {token}"
    assert "api_key" not in request["params"]


def test_unauthorized_error_does_not_expose_request_url_or_credentials(tmp_path):
    credential = "secret-test-value"

    class UnauthorizedResponse:
        def raise_for_status(self):
            error = requests.HTTPError(
                f"401 at https://example.test/?api_key={credential}"
            )
            error.response = type("Response", (), {"status_code": 401})()
            raise error

        def json(self):
            return {}

    with pytest.raises(RuntimeError, match="v4 read token") as error:
        sync_module.sync_tmdb_catalog(
            output_path=tmp_path / "catalog.json",
            api_key="local-test-key",
            session=FakeSession([UnauthorizedResponse()]),
        )

    assert credential not in str(error.value)
    assert "example.test" not in str(error.value)


def test_sync_requires_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_READ_ACCESS_TOKEN", raising=False)

    with pytest.raises(ValueError, match="TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN"):
        sync_module.sync_tmdb_catalog(output_path=tmp_path / "catalog.json")


def test_initial_sync_uses_full_load_pull_date_as_its_incremental_baseline(
    tmp_path, monkeypatch
):
    full_load = tmp_path / "full_load.json"
    full_load.write_text(
        json.dumps(
            {
                "pulled_at": "2026-10-01",
                "year_range": "2015-2026",
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sync_module, "FULL_LOAD_PATHS", (full_load,))
    session = FakeSession([_page(1, 1, [])])

    result = sync_module.sync_tmdb_catalog(
        output_path=tmp_path / "tmdb_incremental_catalog.json",
        api_key="local-test-key",
        today=date(2026, 11, 15),
        session=session,
    )

    assert result["since"] == "2026-09-01"
