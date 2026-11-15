"""Keyless movie posters and plot metadata from public sources."""

from __future__ import annotations

import base64
import difflib
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote, urlparse

import requests
import streamlit as st


LOGGER = logging.getLogger(__name__)
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"
ITUNES_SEARCH = "https://itunes.apple.com/search"
TVMAZE_SEARCH = "https://api.tvmaze.com/singlesearch/shows"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
ALLOWED_POSTER_HOSTS = {
    "upload.wikimedia.org",
    "commons.wikimedia.org",
    "static.tvmaze.com",
    "is1-ssl.mzstatic.com",
}
HTTP_HEADERS = {"User-Agent": "GenreGenie/2.0 (educational project)"}
REQUEST_TIMEOUT = 4
MAX_METADATA_WORKERS = 4


def _clean_title(title: str) -> str:
    year_match = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    title_without_year = (
        str(title)[: year_match.start()] if year_match else str(title)
    )
    cleaned = re.sub(r"\s*\([^)]*\)", "", title_without_year).strip()
    cleaned = re.sub(r"\.{2,}$", "", cleaned).rstrip(" ,;:")
    article = re.match(r"^(.*),\s*(The|A|An)$", cleaned, flags=re.IGNORECASE)
    if article:
        cleaned = f"{article.group(2)} {article.group(1)}"
    return cleaned.strip()


def _normalize_overview(overview: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(overview or ""))).strip()


def _short_overview(overview: str) -> str:
    text = _normalize_overview(overview)
    if not text:
        return "Synopsis could not be retrieved from public movie sources."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2])


def _valid_poster_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return ""
    if parsed.hostname not in ALLOWED_POSTER_HOSTS and not parsed.hostname.endswith(
        ".mzstatic.com"
    ):
        return ""
    return re.sub(r"/\d+px-", "/500px-", url)


def _summary_from_page(page_title: str, session: requests.Session) -> dict[str, str] | None:
    slug = quote(page_title.replace(" ", "_"), safe="_()")
    try:
        response = session.get(
            WIKIPEDIA_SUMMARY + slug,
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    thumbnail = data.get("thumbnail")
    image_url = _valid_poster_url(
        thumbnail.get("source") if isinstance(thumbnail, dict) else ""
    )
    extract = data.get("extract")
    return {
        "poster_url": image_url,
        "overview": (
            _normalize_overview(extract)
            if isinstance(extract, str) and extract
            else ""
        ),
    }


def _wikipedia_metadata(
    title: str,
    year: int | None,
    session: requests.Session,
) -> dict[str, str]:
    page_names = [title]
    if year:
        page_names.append(f"{title} ({year} film)")
    page_names.append(f"{title} (film)")
    best: dict[str, str] = {"poster_url": "", "overview": ""}
    for page_name in dict.fromkeys(page_names):
        summary = _summary_from_page(page_name, session)
        if not summary:
            continue
        if len(summary["overview"]) > len(best["overview"]):
            best["overview"] = summary["overview"]
        if summary["poster_url"]:
            best["poster_url"] = summary["poster_url"]
        if best["poster_url"] and best["overview"]:
            return best

    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{title} {year or ''} film".strip(),
        "srlimit": 3,
        "format": "json",
    }
    try:
        response = session.get(
            WIKIPEDIA_SEARCH,
            params=params,
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            query_data = data.get("query", {}) if isinstance(data, dict) else {}
            results = query_data.get("search", []) if isinstance(query_data, dict) else []
            if isinstance(results, list):
                for result in results[:3]:
                    if not isinstance(result, dict) or not result.get("title"):
                        continue
                    summary = _summary_from_page(str(result["title"]), session)
                    if not summary:
                        continue
                    if not best["poster_url"] and summary["poster_url"]:
                        best["poster_url"] = summary["poster_url"]
                    if not best["overview"] and summary["overview"]:
                        best["overview"] = summary["overview"]
                    if best["poster_url"] and best["overview"]:
                        break
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("Wikipedia search lookup failed for %s", title)
    return best


def _itunes_metadata(
    title: str,
    year: int | None,
    session: requests.Session,
) -> dict[str, Any]:
    try:
        response = session.get(
            ITUNES_SEARCH,
            params={
                "term": title,
                "media": "movie",
                "entity": "movie",
                "limit": 3,
            },
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return {}
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("iTunes lookup failed for %s", title)
        return {}
    if not isinstance(data, dict):
        return {}
    results = data.get("results", [])
    if not isinstance(results, list):
        return {}
    items = [item for item in results if isinstance(item, dict)]
    if year:
        exact_year_items = [
            item
            for item in items
            if re.match(r"^\d{4}", str(item.get("releaseDate", "")))
            and int(str(item["releaseDate"])[:4]) == year
        ]
        unknown_year_items = [
            item for item in items if not re.match(r"^\d{4}", str(item.get("releaseDate", "")))
        ]
        items = exact_year_items or unknown_year_items
    title_key = _normalize_overview(title).casefold()
    titles = [str(item.get("trackName", "")) for item in items]
    best_titles = difflib.get_close_matches(title_key, [name.casefold() for name in titles], n=1, cutoff=0.4)
    if best_titles:
        selected_title = best_titles[0]
        item = next(
            item for item in items if str(item.get("trackName", "")).casefold() == selected_title
        )
    elif items:
        item = items[0]
    else:
        return {}

    artwork = item.get("artworkUrl100")
    if isinstance(artwork, str):
        artwork = artwork.replace("100x100bb", "600x600bb")
    runtime_ms = item.get("trackTimeMillis")
    try:
        runtime = round(float(runtime_ms) / 60000) if runtime_ms else None
    except (TypeError, ValueError):
        runtime = None
    return {
        "poster_url": _valid_poster_url(artwork),
        "overview": _normalize_overview(item.get("longDescription", "")),
        "runtime": f"{runtime} min" if runtime else None,
    }


def _tvmaze_metadata(title: str, session: requests.Session) -> dict[str, Any]:
    try:
        response = session.get(
            TVMAZE_SEARCH,
            params={"q": title},
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return {}
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("TVMaze lookup failed for %s", title)
        return {}
    if not isinstance(data, dict):
        return {}
    show = data.get("show", data)
    if not isinstance(show, dict):
        return {}
    image = show.get("image")
    poster = ""
    if isinstance(image, dict):
        poster = _valid_poster_url(image.get("original") or image.get("medium"))
    summary_html = show.get("summary")
    overview = ""
    if isinstance(summary_html, str) and summary_html.strip():
        overview = re.sub(r"<[^>]+>", " ", html.unescape(summary_html))
        overview = _normalize_overview(overview)
    return {"poster_url": poster, "overview": overview}


def _wikidata_metadata(
    title: str,
    year: int | None,
    session: requests.Session,
) -> dict[str, Any]:
    try:
        response = session.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": f"{title} {year or ''}".strip(),
                "language": "en",
                "format": "json",
                "limit": 1,
            },
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return {}
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("Wikidata lookup failed for %s", title)
        return {}
    results = data.get("search", []) if isinstance(data, dict) else []
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return {}
    description = results[0].get("description")
    return {
        "poster_url": "",
        "overview": (
            _normalize_overview(description)
            if isinstance(description, str) and description.strip()
            else ""
        ),
    }


def _fallback_poster(
    title: str,
    year: int | None,
    genres: tuple[str, ...] | list[str] = (),
) -> str:
    safe_title = html.escape(_clean_title(title)[:48])
    safe_year = html.escape(str(year) if year else "GENREGENIE PICK")
    words = re.findall(r"[A-Za-z0-9]+", _clean_title(title))
    initials = "".join(word[0] for word in words if word.casefold() not in {"the", "a", "an"})[:2]
    if not initials:
        initials = (words[0][0] if words else "G").upper()
    safe_initials = html.escape(initials.upper())
    safe_genre = html.escape(str(genres[0])[:24] if genres else "MOVIE PICK")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="900">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#31105f"/><stop offset="1" stop-color="#a855f7"/>
      </linearGradient></defs>
      <rect width="600" height="900" fill="#101016"/>
      <rect x="20" y="20" width="560" height="860" rx="34" fill="url(#g)"/>
      <circle cx="470" cy="170" r="180" fill="#fff" opacity=".08"/>
      <text x="300" y="395" text-anchor="middle" fill="#fff" opacity=".94"
        font-family="Arial,sans-serif" font-size="190" font-weight="700">{safe_initials}</text>
      <rect x="54" y="646" width="220" height="42" rx="21" fill="#1a1027"
        stroke="#d8b4fe" stroke-width="2"/>
      <text x="164" y="674" text-anchor="middle" fill="#e9d5ff"
        font-family="Arial,sans-serif" font-size="20" font-weight="700">{safe_genre}</text>
      <text x="54" y="750" fill="white" font-family="Arial,sans-serif"
        font-size="38" font-weight="700">{safe_title}</text>
      <text x="54" y="810" fill="#e9d5ff" font-family="Arial,sans-serif"
        font-size="27">{safe_year} · GENREGENIE</text>
    </svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _fallback_details(
    title: str,
    year: int | None,
    genres: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    synopsis = "Synopsis details are temporarily unavailable."
    return {
        "poster_url": _fallback_poster(title, year, genres),
        "overview": synopsis,
        "full_overview": synopsis,
        "runtime": None,
        "rating": None,
        "source": "generated",
    }


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details(
    title: str,
    year: int | None,
    tmdb_id: int | None = None,
    genres: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Fetch free Wikipedia/iTunes metadata and cache it for one day."""
    del tmdb_id
    title_year = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    if year is None and title_year:
        year = int(title_year.group(1))
    clean_title = _clean_title(title)
    metadata: dict[str, Any] = {
        "poster_url": "",
        "overview": "",
        "runtime": None,
    }
    source = "generated"

    def merge(found: dict[str, Any], provider: str) -> None:
        nonlocal source
        if not metadata["poster_url"] and found.get("poster_url"):
            metadata["poster_url"] = found["poster_url"]
            source = provider
        if not metadata["overview"] and found.get("overview"):
            metadata["overview"] = found["overview"]
            source = provider
        if not metadata["runtime"] and found.get("runtime"):
            metadata["runtime"] = found["runtime"]

    with requests.Session() as session:
        session.headers.update(HTTP_HEADERS)
        try:
            merge(_wikipedia_metadata(clean_title, year, session), "Wikipedia")
        except Exception:
            LOGGER.exception(
                "Wikipedia metadata lookup failed for %s (%s).", clean_title, year
            )
        if not metadata["poster_url"] or not metadata["overview"]:
            try:
                merge(_itunes_metadata(clean_title, year, session), "iTunes")
            except Exception:
                LOGGER.exception(
                    "iTunes metadata lookup failed for %s (%s).", clean_title, year
                )
        if not metadata["poster_url"] or not metadata["overview"]:
            try:
                merge(_tvmaze_metadata(clean_title, session), "TVMaze")
            except Exception:
                LOGGER.exception("TVMaze metadata lookup failed for %s.", clean_title)
        if not metadata["overview"]:
            try:
                merge(_wikidata_metadata(clean_title, year, session), "Wikidata")
            except Exception:
                LOGGER.exception(
                    "Wikidata metadata lookup failed for %s.", clean_title
                )

    full_overview = _normalize_overview(metadata["overview"])
    if not full_overview:
        full_overview = "Synopsis details are temporarily unavailable."

    if not _valid_poster_url(metadata["poster_url"]):
        metadata["poster_url"] = _fallback_poster(clean_title, year, genres)
        if source == "Wikipedia" or source == "iTunes":
            source = "generated poster + " + source
    return {
        "poster_url": metadata["poster_url"],
        "overview": _short_overview(full_overview),
        "full_overview": full_overview,
        "runtime": metadata["runtime"],
        "rating": None,
        "source": source,
    }


def fetch_movie_details_batch(
    recommendations: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Fetch uncached card details concurrently, preserving recommendation IDs."""
    if not recommendations:
        return {}
    output: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=MAX_METADATA_WORKERS) as executor:
        futures = {
            executor.submit(
                fetch_movie_details,
                str(movie["formatted_title"]),
                movie.get("year"),
                movie.get("tmdb_id"),
                tuple(movie.get("genres", ())),
            ): int(movie["movie_id"])
            for movie in recommendations
        }
        for future in as_completed(futures):
            movie_id = futures[future]
            try:
                output[movie_id] = future.result()
            except Exception:
                LOGGER.exception("Metadata lookup failed for movie ID %s", movie_id)
                movie = next(
                    item
                    for item in recommendations
                    if int(item["movie_id"]) == movie_id
                )
                output[movie_id] = _fallback_details(
                    str(movie["formatted_title"]),
                    movie.get("year"),
                    movie.get("genres", ()),
                )
    return output
