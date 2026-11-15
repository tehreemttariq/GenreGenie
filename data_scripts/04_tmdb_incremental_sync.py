"""Incrementally refresh the all-genre TMDB movie catalog."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "tmdb_incremental_catalog.json"
FULL_LOAD_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "full_load_2005_2014.json",
    PROJECT_ROOT / "data" / "raw" / "full_load.json",
)
DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
INITIAL_RELEASE_DATE = date(2020, 1, 1)
MAX_DISCOVER_PAGES = 500
REQUEST_INTERVAL_SECONDS = 0.25


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"movies": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read TMDB sync state at {path}: {error}") from error
    if not isinstance(state, dict) or not isinstance(state.get("movies", []), list):
        raise ValueError(f"TMDB sync state at {path} has an invalid format.")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sync_start_date(
    state: dict[str, Any],
    today: date,
    since: date | None,
    overlap_days: int,
    baseline_date: date | None = None,
) -> date:
    if since is not None:
        return since
    last_synced_at = state.get("last_synced_at")
    if not isinstance(last_synced_at, str):
        if baseline_date is not None:
            return max(
                INITIAL_RELEASE_DATE,
                min(today, baseline_date) - timedelta(days=overlap_days),
            )
        return INITIAL_RELEASE_DATE
    try:
        last_sync_date = datetime.fromisoformat(last_synced_at).date()
    except ValueError as error:
        raise ValueError("TMDB sync state has an invalid last_synced_at value.") from error
    return max(INITIAL_RELEASE_DATE, min(today, last_sync_date) - timedelta(days=overlap_days))


def _full_load_baseline_date() -> date | None:
    candidates: list[tuple[int, date]] = []
    for path in FULL_LOAD_PATHS:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as source:
            metadata = source.read(4096)
        year_range = re.search(r'"year_range"\s*:\s*"(\d{4})-(\d{4})"', metadata)
        pulled_at = re.search(r'"pulled_at"\s*:\s*"(\d{4}-\d{2}-\d{2})"', metadata)
        if year_range and pulled_at:
            candidates.append(
                (int(year_range.group(2)), date.fromisoformat(pulled_at.group(1)))
            )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _is_read_access_token(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and parts[0].startswith("eyJ")


def _split_date_window(
    start_date: date,
    end_date: date,
) -> list[dict[str, str]] | None:
    if start_date >= end_date:
        return None
    midpoint = start_date + timedelta(days=(end_date - start_date).days // 2)
    return [
        {
            "since": start_date.isoformat(),
            "until": midpoint.isoformat(),
        },
        {
            "since": (midpoint + timedelta(days=1)).isoformat(),
            "until": end_date.isoformat(),
        },
    ]


def sync_tmdb_catalog(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    api_key: str | None = None,
    read_access_token: str | None = None,
    since: date | None = None,
    overlap_days: int = 30,
    today: date | None = None,
    session: Any = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch all-genre updates, checkpointing each page."""
    api_key = api_key or os.getenv("TMDB_API_KEY")
    read_access_token = read_access_token or os.getenv("TMDB_READ_ACCESS_TOKEN")
    if not api_key and not read_access_token:
        raise ValueError(
            "Set TMDB_API_KEY or TMDB_READ_ACCESS_TOKEN before syncing TMDB."
        )
    if not read_access_token and api_key and _is_read_access_token(api_key):
        read_access_token, api_key = api_key, None
    if read_access_token:
        api_key = None
    if overlap_days < 0:
        raise ValueError("overlap_days cannot be negative.")

    sync_day = today or datetime.now(timezone.utc).date()
    state = _read_state(output_path)
    stored_movies = state.get("movies", [])
    movies_by_id = {
        int(movie["id"]): movie
        for movie in stored_movies
        if isinstance(movie, dict) and str(movie.get("id", "")).isdigit()
    }

    progress = state.get("in_progress")
    if isinstance(progress, dict):
        start_date = date.fromisoformat(str(progress["since"]))
        end_date = date.fromisoformat(
            str(progress.get("until") or sync_day.isoformat())
        )
        pending_windows = list(progress.get("pending_windows", []))
        current_window = progress.get("current_window")
        if current_window is None and "next_page" in progress:
            current_window = {
                "since": start_date.isoformat(),
                "until": end_date.isoformat(),
                "next_page": int(progress["next_page"]),
                "total_pages": int(progress["total_pages"]),
            }
    else:
        start_date = _sync_start_date(
            state,
            sync_day,
            since,
            overlap_days,
            baseline_date=_full_load_baseline_date(),
        )
        end_date = sync_day
        if start_date > end_date:
            raise ValueError("The TMDB sync start date cannot be after today.")
        pending_windows = [
            {"since": start_date.isoformat(), "until": end_date.isoformat()}
        ]
        current_window = None

    if not isinstance(pending_windows, list):
        raise ValueError("TMDB sync state has invalid pending date windows.")

    owns_session = session is None
    request_session = session or requests.Session()
    headers = (
        {"Authorization": f"Bearer {read_access_token}"}
        if read_access_token
        else {}
    )
    base_params: dict[str, Any] = {
        "language": "en-US",
        "include_adult": "false",
        "sort_by": "primary_release_date.desc",
        "primary_release_date.gte": start_date.isoformat(),
        "primary_release_date.lte": sync_day.isoformat(),
    }
    if api_key:
        base_params["api_key"] = api_key

    last_request_at: float | None = None

    def report_progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    def request_page(window: dict[str, Any], page_number: int) -> dict[str, Any]:
        nonlocal last_request_at
        if last_request_at is not None:
            delay = REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_at)
            if delay > 0:
                time.sleep(delay)
        try:
            response = request_session.get(
                DISCOVER_URL,
                params={
                    **base_params,
                    "primary_release_date.gte": window["since"],
                    "primary_release_date.lte": window["until"],
                    "page": page_number,
                },
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as error:
            status_code = (
                error.response.status_code if error.response is not None else "unknown"
            )
            if status_code == 401:
                raise RuntimeError(
                    "TMDB rejected the credentials (HTTP 401). Use a v3 API key "
                    "as TMDB_API_KEY or a v4 read token as TMDB_READ_ACCESS_TOKEN."
                ) from None
            raise RuntimeError(
                f"TMDB Discover returned HTTP {status_code}."
            ) from None
        except requests.RequestException as error:
            raise RuntimeError(
                f"TMDB Discover request failed ({type(error).__name__})."
            ) from None
        last_request_at = time.monotonic()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("TMDB Discover returned an invalid response.")
        return payload

    def save_progress() -> None:
        state["movies"] = list(movies_by_id.values())
        state["in_progress"] = {
            "since": start_date.isoformat(),
            "until": end_date.isoformat(),
            "pending_windows": pending_windows,
            "current_window": current_window,
        }
        _write_state(output_path, state)

    try:
        while current_window is not None or pending_windows:
            if current_window is None:
                current_window = pending_windows.pop(0)
                current_window["next_page"] = 1
                current_window["total_pages"] = None
                save_progress()
                report_progress(
                    "Checking TMDB releases "
                    f"{current_window['since']} through {current_window['until']}."
                )

            page_number = int(current_window["next_page"])
            total_pages = current_window["total_pages"]
            payload = None
            if total_pages is None:
                payload = request_page(current_window, page_number)
                try:
                    total_pages = int(payload.get("total_pages", 1))
                except (TypeError, ValueError) as error:
                    raise ValueError("TMDB Discover returned an invalid page count.") from error
                if total_pages > MAX_DISCOVER_PAGES:
                    window = _split_date_window(
                        date.fromisoformat(str(current_window["since"])),
                        date.fromisoformat(str(current_window["until"])),
                    )
                    if window is None:
                        raise ValueError(
                            "A single-day TMDB release window exceeds the "
                            f"{MAX_DISCOVER_PAGES}-page API limit."
                        )
                    report_progress(
                        f"TMDB returned {total_pages} pages; splitting this date range."
                    )
                    pending_windows = window + pending_windows
                    current_window = None
                    save_progress()
                    continue
                current_window["total_pages"] = max(1, total_pages)
                total_pages = current_window["total_pages"]
                save_progress()
                report_progress(
                    f"Loading {total_pages} page(s) for "
                    f"{current_window['since']} through {current_window['until']}."
                )
            else:
                report_progress(
                    f"Resuming at page {page_number} of {total_pages} for "
                    f"{current_window['since']} through {current_window['until']}."
                )

            while int(current_window["next_page"]) <= int(total_pages):
                page_number = int(current_window["next_page"])
                if payload is None or page_number != 1:
                    payload = request_page(current_window, page_number)
                for movie in payload["results"]:
                    if isinstance(movie, dict) and str(movie.get("id", "")).isdigit():
                        movies_by_id[int(movie["id"])] = movie
                current_window["next_page"] = page_number + 1
                save_progress()
                if (
                    page_number == 1
                    or page_number % 25 == 0
                    or page_number == total_pages
                ):
                    report_progress(
                        f"{current_window['since']} through {current_window['until']}: "
                        f"page {page_number} of {total_pages}."
                    )
                payload = None

            current_window = None
            save_progress()

        state["last_synced_at"] = datetime.now(timezone.utc).isoformat()
        state.pop("in_progress", None)
        state["catalog_scope"] = "all_genres"
        state["initial_release_date"] = INITIAL_RELEASE_DATE.isoformat()
        _write_state(output_path, state)
    finally:
        if owns_session:
            request_session.close()

    return {
        "movie_count": len(movies_by_id),
        "last_synced_at": state["last_synced_at"],
        "since": start_date.isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        help="Override the release-date lower bound (ISO date, e.g. 2020-01-01).",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=30,
        help="Recheck this many days before the last successful sync (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Incremental catalog path (defaults to data/raw/tmdb_incremental_catalog.json).",
    )
    arguments = parser.parse_args()
    print("Starting all-genre TMDB catalog sync...", flush=True)
    result = sync_tmdb_catalog(
        output_path=arguments.output,
        since=arguments.since,
        overlap_days=arguments.overlap_days,
        on_progress=lambda message: print(message, flush=True),
    )
    print(
        f"TMDB catalog now contains {result['movie_count']} titles "
        f"(synced from {result['since']})."
    )


if __name__ == "__main__":
    main()
