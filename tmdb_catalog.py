"""Stream and normalize TMDB full loads into GenreGenie's catalog schema."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


TMDB_MOVIE_ID_OFFSET = 1_000_000_000
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_GENRE_NAMES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}
TMDB_CATALOG_COLUMNS = (
    "movieId",
    "title",
    "genres",
    "rating_prediction",
    "rating_source",
    "tmdb_id",
    "poster_url",
    "overview",
    "director",
    "cast",
    "production",
)
LOGGER = logging.getLogger(__name__)


def iter_tmdb_results(path: Path, chunk_size: int = 65_536) -> Iterator[dict[str, Any]]:
    """Yield objects from a full-load JSON `results` array without loading it all."""
    decoder = json.JSONDecoder()
    results_pattern = re.compile(r'"results"\s*:\s*\[')
    with path.open(encoding="utf-8") as source:
        buffer = ""
        match = None
        while match is None:
            chunk = source.read(chunk_size)
            if not chunk:
                raise ValueError(f"TMDB full load has no results array: {path}")
            buffer += chunk
            match = results_pattern.search(buffer)
        buffer = buffer[match.end() :]
        cursor = 0

        while True:
            while cursor < len(buffer) and (
                buffer[cursor].isspace() or buffer[cursor] == ","
            ):
                cursor += 1
            if cursor == len(buffer):
                chunk = source.read(chunk_size)
                if not chunk:
                    raise ValueError(f"TMDB full load has an incomplete results array: {path}")
                buffer = chunk
                cursor = 0
                continue
            if buffer[cursor] == "]":
                return

            try:
                record, next_cursor = decoder.raw_decode(buffer, cursor)
            except json.JSONDecodeError:
                chunk = source.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f"TMDB full load has an invalid results array: {path}"
                    ) from None
                buffer = buffer[cursor:] + chunk
                cursor = 0
                continue
            if not isinstance(record, dict):
                raise ValueError(f"TMDB result is not a movie object: {path}")
            yield record
            cursor = next_cursor
            if cursor >= chunk_size:
                buffer = buffer[cursor:]
                cursor = 0


def prepare_tmdb_movies(
    records: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Convert TMDB records from any release year or genre to catalog rows."""
    rows: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            tmdb_id = int(record.get("id", 0))
        except (TypeError, ValueError):
            continue
        if tmdb_id <= 0:
            continue

        title = str(record.get("title") or record.get("original_title") or "").strip()
        release_date = str(record.get("release_date") or "")
        year_text = release_date[:4]
        if not title:
            continue
        year = int(year_text) if len(year_text) == 4 and year_text.isdigit() else None
        title_with_year = f"{title} ({year})" if year is not None else title
        genres = _genre_names(record)

        try:
            vote_average = float(record.get("vote_average", 0.0))
            vote_count = max(0, int(record.get("vote_count", 0)))
        except (TypeError, ValueError):
            vote_average, vote_count = 0.0, 0
        if not 0.0 <= vote_average <= 10.0:
            vote_average = 0.0
            vote_count = 0
        prior_rating = 3.0
        prior_votes = 50
        community_rating = (
            (vote_average / 2 * vote_count + prior_rating * prior_votes)
            / (vote_count + prior_votes)
            if vote_count
            else prior_rating
        )

        poster_path = record.get("poster_path")
        poster_url = (
            f"{TMDB_IMAGE_BASE}{poster_path}"
            if isinstance(poster_path, str) and poster_path.startswith("/")
            else ""
        )
        movie_id = TMDB_MOVIE_ID_OFFSET + tmdb_id
        row = {
            "movieId": movie_id,
            "title": title_with_year,
            "genres": "|".join(genres),
            "rating_prediction": round(community_rating, 3),
            "rating_source": "TMDB",
            "tmdb_id": tmdb_id,
            "poster_url": poster_url,
            "overview": str(record.get("overview") or "").strip(),
            "director": _name_values(record.get("director") or record.get("directors")),
            "cast": _name_values(record.get("top_cast") or record.get("cast")),
            "production": _name_values(record.get("production_companies")),
        }
        existing = rows.get(movie_id)
        if existing is not None:
            for field in (
                "genres",
                "poster_url",
                "overview",
                "director",
                "cast",
                "production",
            ):
                if not row[field]:
                    row[field] = existing[field]
        rows[movie_id] = row
    return pd.DataFrame(rows.values(), columns=TMDB_CATALOG_COLUMNS)


def append_tmdb_movies(
    movies: pd.DataFrame,
    records: Iterable[Mapping[str, Any]],
    excluded_tmdb_ids: Collection[int] = (),
) -> pd.DataFrame:
    """Append unique TMDB additions without duplicating linked MovieLens movies."""
    additions = prepare_tmdb_movies(records)
    if excluded_tmdb_ids:
        additions = additions[
            ~additions["tmdb_id"].isin(excluded_tmdb_ids)
        ]
    if additions.empty:
        return movies
    return pd.concat([movies, additions], ignore_index=True, sort=False)


def load_or_build_tmdb_full_catalog(
    source_paths: Iterable[Path],
    cache_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """Load a compact normalized cache, rebuilding it only when source files change."""
    paths = tuple(path for path in source_paths if path.is_file())
    if not paths:
        return pd.DataFrame(columns=TMDB_CATALOG_COLUMNS)

    fingerprint = [
        {
            "path": str(path.resolve()),
            "mtime_ns": path.stat().st_mtime_ns,
            "size": path.stat().st_size,
        }
        for path in paths
    ]
    if cache_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("sources") == fingerprint:
                return pd.read_csv(cache_path, keep_default_na=False)
        except (OSError, TypeError, ValueError, pd.errors.ParserError) as error:
            LOGGER.warning("Rebuilding unreadable TMDB cache: %s", error)

    records = (
        record
        for path in paths
        for record in iter_tmdb_results(path)
    )
    catalog = prepare_tmdb_movies(records)
    for field in ("director", "cast", "production"):
        catalog[field] = catalog[field].map(_format_names)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    catalog.to_csv(temporary_cache, index=False)
    temporary_manifest.write_text(
        json.dumps(
            {"sources": fingerprint, "record_count": len(catalog)},
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_cache.replace(cache_path)
    temporary_manifest.replace(manifest_path)
    LOGGER.info("Prepared %s unique movies from TMDB full loads.", len(catalog))
    return catalog


def merge_tmdb_catalogs(
    base: pd.DataFrame,
    updates: pd.DataFrame,
) -> pd.DataFrame:
    """Upsert refreshed TMDB rows while retaining full-load details that are absent."""
    base = base.reindex(columns=TMDB_CATALOG_COLUMNS).drop_duplicates(
        "tmdb_id", keep="last"
    )
    updates = updates.reindex(columns=TMDB_CATALOG_COLUMNS).drop_duplicates(
        "tmdb_id", keep="last"
    )
    if base.empty:
        return updates.reset_index(drop=True)
    if updates.empty:
        return base.reset_index(drop=True)

    base = base.set_index("tmdb_id", drop=False)
    updates = updates.set_index("tmdb_id", drop=False)
    shared_ids = base.index.intersection(updates.index)
    for tmdb_id in shared_ids:
        for field in (
            "genres",
            "poster_url",
            "overview",
            "director",
            "cast",
            "production",
        ):
            if not _has_catalog_value(updates.at[tmdb_id, field]):
                updates.at[tmdb_id, field] = base.at[tmdb_id, field]

    base_only = base.loc[~base.index.isin(updates.index)]
    return pd.concat([base_only, updates], ignore_index=True).reindex(
        columns=TMDB_CATALOG_COLUMNS
    )


def _genre_names(record: Mapping[str, Any]) -> list[str]:
    raw_genres = record.get("genres")
    genres: list[str] = []
    if isinstance(raw_genres, (list, tuple)):
        for item in raw_genres:
            if isinstance(item, dict):
                genre_id = _as_int(item.get("id"))
                name = TMDB_GENRE_NAMES.get(genre_id) if genre_id is not None else None
                name = name or _canonical_genre(item.get("name"))
            else:
                name = _canonical_genre(item)
            if name:
                genres.append(name)

    if not genres:
        raw_genre_ids = record.get("genre_ids", [])
        if isinstance(raw_genre_ids, (list, tuple, set)):
            genres = [
                TMDB_GENRE_NAMES[genre_id]
                for raw_genre_id in raw_genre_ids
                if (genre_id := _as_int(raw_genre_id)) in TMDB_GENRE_NAMES
            ]
    return list(dict.fromkeys(genres))


def _canonical_genre(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    name = value.strip()
    aliases = {
        "science fiction": "Sci-Fi",
        "sci-fi": "Sci-Fi",
        "tv movie": "TV Movie",
    }
    return aliases.get(name.casefold(), name)


def _name_values(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, (list, tuple)):
        return None
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("original_name")
        else:
            name = item
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names or None


def _format_names(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " · ".join(str(item).strip() for item in value if str(item).strip())
    return value.strip() if isinstance(value, str) else ""


def _has_catalog_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return not pd.isna(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
