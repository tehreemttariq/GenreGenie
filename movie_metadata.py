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
WIKIPEDIA_PAGE_API = "https://en.wikipedia.org/w/api.php"
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


def _summary_from_page(
    page_title: str,
    session: requests.Session,
) -> dict[str, Any] | None:
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
        "page_title": str(data.get("title") or page_title),
        "overview": (
            _normalize_overview(extract)
            if isinstance(extract, str) and extract
            else ""
        ),
    }


def _wikitext_names(value: str, limit: int) -> list[str]:
    cleaned = re.sub(r"<ref\b[^>]*>.*?</ref>", "", value, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<ref\b[^>]*/\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", ",", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\{\{(?:plainlist|unbulleted list|ubl|hlist)\s*\|",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\{\{[^{}]*\}\}", "", cleaned)
    cleaned = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", cleaned)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("}}", "")
    cleaned = re.sub(r"''+", "", cleaned)
    names = []
    for candidate in re.split(r"[\n,;|•]+", cleaned):
        name = re.sub(r"^[*#\s]+|[\s\]]+$", "", candidate).strip()
        name = re.sub(r"\s+", " ", name)
        if name and name.casefold() not in {"and", "unbulleted list"}:
            names.append(name)
    return list(dict.fromkeys(names))[:limit]


def _wikipedia_infobox_metadata(
    page_title: str,
    session: requests.Session,
) -> dict[str, Any]:
    try:
        response = session.get(
            WIKIPEDIA_PAGE_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "pageimages|revisions",
                "piprop": "thumbnail",
                "pithumbsize": 800,
                "rvprop": "content",
                "rvslots": "main",
                "titles": page_title,
            },
            headers=HTTP_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return {}
        data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("Wikipedia page details lookup failed for %s", page_title)
        return {}
    query = data.get("query", {}) if isinstance(data, dict) else {}
    pages = query.get("pages", []) if isinstance(query, dict) else []
    if isinstance(pages, dict):
        pages = list(pages.values())
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        return {}

    page = pages[0]
    thumbnail = page.get("thumbnail")
    output: dict[str, Any] = {
        "poster_url": _valid_poster_url(
            thumbnail.get("source") if isinstance(thumbnail, dict) else ""
        )
    }
    revisions = page.get("revisions", [])
    if not isinstance(revisions, list) or not revisions or not isinstance(revisions[0], dict):
        return output
    revision = revisions[0]
    slots = revision.get("slots", {})
    main_slot = slots.get("main", {}) if isinstance(slots, dict) else {}
    wikitext = (
        main_slot.get("content") or main_slot.get("*")
        if isinstance(main_slot, dict)
        else None
    )
    if not isinstance(wikitext, str):
        wikitext = revision.get("*")
    if not isinstance(wikitext, str):
        return output

    fields = {
        "director": ("director", "directed by"),
        "cast": ("starring", "cast"),
        "production": ("production company", "production companies", "studio"),
    }
    for output_name, keys in fields.items():
        key_pattern = "|".join(re.escape(key) for key in keys)
        match = re.search(
            rf"(?ims)^\|\s*(?:{key_pattern})\s*=\s*(.*?)(?=^\|\s*[^|\n]+\s*=|^\}}\}})",
            wikitext,
        )
        if not match:
            continue
        names = _wikitext_names(match.group(1), 4 if output_name == "cast" else 3)
        if names:
            output[output_name] = names if output_name == "cast" else names[0]
    return output


def _wikipedia_metadata(
    title: str,
    year: int | None,
    session: requests.Session,
) -> dict[str, Any]:
    page_names = [title]
    if year:
        page_names.append(f"{title} ({year} film)")
    page_names.append(f"{title} (film)")
    best: dict[str, Any] = {"poster_url": "", "overview": "", "page_title": ""}
    for page_name in dict.fromkeys(page_names):
        summary = _summary_from_page(page_name, session)
        if not summary:
            continue
        if not best["page_title"]:
            best["page_title"] = summary["page_title"]
        if len(summary["overview"]) > len(best["overview"]):
            best["overview"] = summary["overview"]
        if summary["poster_url"]:
            best["poster_url"] = summary["poster_url"]
        if best["poster_url"] and best["overview"]:
            for key, value in _wikipedia_infobox_metadata(
                best["page_title"], session
            ).items():
                if value:
                    best[key] = value
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
                    if not best["page_title"]:
                        best["page_title"] = summary["page_title"]
                    if not best["poster_url"] and summary["poster_url"]:
                        best["poster_url"] = summary["poster_url"]
                    if not best["overview"] and summary["overview"]:
                        best["overview"] = summary["overview"]
                    if best["poster_url"] and best["overview"]:
                        break
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.debug("Wikipedia search lookup failed for %s", title)
    if best["page_title"]:
        for key, value in _wikipedia_infobox_metadata(
            best["page_title"], session
        ).items():
            if value:
                best[key] = value
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
    director: str | None = None,
) -> str:
    clean_title = _clean_title(title)[:54]
    words = clean_title.split()
    title_lines: list[str] = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) > 18 and current_line:
            title_lines.append(current_line)
            current_line = word
        else:
            current_line = candidate
    if current_line:
        title_lines.append(current_line)
    title_lines = title_lines[:3] or ["A film for tonight"]
    title_markup = "".join(
        f'<tspan x="300" dy="{0 if index == 0 else 56}">'
        f'{html.escape(line)}</tspan>'
        for index, line in enumerate(title_lines)
    )
    title_start_y = 580 - (len(title_lines) - 1) * 28
    safe_year = html.escape(str(year) if year else "YEAR NOT LISTED")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="900">
      <defs>
        <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">
          <stop stop-color="#211a2d"/><stop offset="1" stop-color="#11131b"/>
        </linearGradient>
        <radialGradient id="glow">
          <stop stop-color="#8959b8" stop-opacity=".24"/>
          <stop offset="1" stop-color="#8959b8" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="600" height="900" fill="url(#background)"/>
      <rect x="22" y="22" width="556" height="856" rx="18"
        fill="none" stroke="#9e79c2" stroke-opacity=".38" stroke-width="2"/>
      <circle cx="300" cy="315" r="205" fill="url(#glow)"/>
      <g fill="none" stroke="#d3b5f3" stroke-linecap="round"
        stroke-linejoin="round" stroke-width="12" opacity=".82"
        transform="translate(222 206)">
        <path d="M12 66h132v110H12z"/>
        <path d="m12 66 16-52 120 22-16 52"/>
        <path d="m48 21 25 48m35-40 25 48"/>
        <path d="M42 111h35m-35 30h62" stroke-width="9"/>
      </g>
      <text x="300" y="465" text-anchor="middle" fill="#bba2d2"
        font-family="Arial,sans-serif" font-size="20" font-weight="700"
        letter-spacing="3">ARTWORK UNAVAILABLE</text>
      <text x="300" y="{title_start_y}" text-anchor="middle" fill="#f4effa"
        font-family="Arial,sans-serif" font-size="44" font-weight="700">{title_markup}</text>
      <text x="300" y="730" text-anchor="middle" fill="#aaa0b5"
        font-family="Arial,sans-serif" font-size="24" letter-spacing="2">{safe_year}</text>
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
        "director": None,
        "cast": [],
        "production": None,
        "source": "generated",
    }


def fallback_movie_details(
    title: str,
    year: int | None,
    genres: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Build local card details without waiting for remote metadata providers."""
    return _fallback_details(title, year, genres)


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
        "director": None,
        "cast": [],
        "production": None,
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
        for field in ("director", "cast", "production"):
            if not metadata[field] and found.get(field):
                metadata[field] = found[field]

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
        metadata["poster_url"] = _fallback_poster(
            clean_title,
            year,
            genres,
            str(metadata["director"]) if metadata["director"] else None,
        )
        if source == "Wikipedia" or source == "iTunes":
            source = "generated poster + " + source
    return {
        "poster_url": metadata["poster_url"],
        "overview": _short_overview(full_overview),
        "full_overview": full_overview,
        "runtime": metadata["runtime"],
        "rating": None,
        "director": metadata["director"],
        "cast": metadata["cast"],
        "production": metadata["production"],
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
