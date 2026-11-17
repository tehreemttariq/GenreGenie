"""Metadata parsing and hybrid ranking helpers for GenreGenie."""

from __future__ import annotations

import difflib
import ast
import io
import logging
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


GENRE_SYNONYMS = {
    "Comedy": ("comedy", "funny", "hilarious", "humorous", "feel good", "feel-good"),
    "Romance": ("romance", "romantic", "love", "rom com", "rom-com", "romcom"),
    "Horror": ("horror", "scary", "spooky", "frightening"),
    "Action": ("action", "high-octane"),
    "Adventure": ("adventure", "adventurous"),
    "Animation": ("animation", "animated"),
    "Crime": ("crime", "gangster"),
    "Documentary": ("documentary", "real life"),
    "Drama": ("drama", "dramatic"),
    "Fantasy": ("fantasy", "magical"),
    "Family": ("family", "family film"),
    "History": ("history", "historical"),
    "Mystery": ("mystery", "whodunit"),
    "Music": ("music", "musical"),
    "Sci-Fi": ("sci fi", "sci-fi", "science fiction", "scifi"),
    "Thriller": ("thriller", "suspense", "edge of your seat"),
    "TV Movie": ("tv movie", "made for television"),
    "War": ("war",),
    "Western": ("western",),
    "Musical": ("musical",),
    "Children": ("children", "family"),
    "Film-Noir": ("film noir", "noir"),
}

MOOD_PRESETS = (
    "Romantic",
    "Funny / Feel-Good",
    "Old Classics (<2005)",
    "90s Nostalgia",
    "Mind-Bending",
    "Tearjerker",
)

MOOD_GENRE_GROUPS = {
    "Romantic": (frozenset({"Romance"}),),
    "Classic Romance (<2005)": (frozenset({"Romance"}),),
    "Funny / Feel-Good": (frozenset({"Comedy"}),),
    "Mind-Bending": (frozenset({"Sci-Fi", "Mystery", "Thriller"}),),
    "Tearjerker": (frozenset({"Drama", "Romance"}),),
    "Action & Thrills": (frozenset({"Action", "Thriller"}),),
    "Discover Romance": (frozenset({"Romance"}),),
    "Discover Comedy": (frozenset({"Comedy"}),),
    "Sci-Fi & Mind-Bending": (frozenset({"Sci-Fi", "Mystery"}),),
    "Thriller & Mystery": (frozenset({"Thriller", "Mystery"}),),
    "Discover Drama": (frozenset({"Drama"}),),
    "Action & Adventure": (frozenset({"Action", "Adventure"}),),
}
ERA_RANGES = {
    "Classic (Before 1980)": (None, 1979, "Classic (Before 1980)"),
    "1980s": (1980, 1989, "1980s"),
    "1990s": (1990, 1999, "1990s"),
    "2000–2009": (2000, 2009, "2000–2009"),
    "2010–2019": (2010, 2019, "2010–2019"),
    "2020 & Newer": (2020, None, "2020 & Newer"),
    "Before 1995": (None, 1994, "Before 1995"),
    "1995–2005": (1995, 2005, "1995–2005"),
    "2006+": (2006, None, "2006+"),
}
LOGGER = logging.getLogger(__name__)
KNOWN_GENRES = {
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Family",
    "Film-Noir",
    "History",
    "Horror",
    "IMAX",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "TV Movie",
    "War",
    "Western",
}


@dataclass(frozen=True)
class PreferenceProfile:
    genre_groups: tuple[frozenset[str], ...]
    year_min: int | None
    year_max: int | None
    era_label: str | None


def normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_search_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _title_without_year(title: str) -> tuple[str, int | None]:
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    if not match:
        return title.strip(), None
    return title[: match.start()].strip(), int(match.group(1))


def format_movie_title(title: str) -> str:
    """Remove title aliases and move trailing articles ahead of the title."""
    clean_title, year = _title_without_year(title)
    clean_title = re.sub(r"\s*\([^)]*\)", "", clean_title).strip()
    article = re.match(r"^(.*),\s*(The|A|An)$", clean_title, flags=re.IGNORECASE)
    if article:
        clean_title = f"{article.group(2)} {article.group(1)}"
    return f"{clean_title} ({year})" if year else clean_title


def _normalized_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _find_column(
    columns: list[Any],
    candidates: tuple[str, ...],
) -> Any | None:
    lookup = {_normalized_column_name(column): column for column in columns}
    for candidate in candidates:
        match = lookup.get(_normalized_column_name(candidate))
        if match is not None:
            return match
    return None


def _one_hot_genre_columns(columns: list[Any]) -> dict[str, Any]:
    known_genres = {
        _normalized_column_name(genre): genre for genre in KNOWN_GENRES
    }
    known_genres["scifi"] = "Sci-Fi"
    output: dict[str, Any] = {}
    for column in columns:
        normalized = _normalized_column_name(column)
        stripped = re.sub(r"^(?:genre|is)", "", normalized)
        genre = known_genres.get(normalized) or known_genres.get(stripped)
        if genre:
            output[genre] = column
    return output


def _truthy_genre_flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _parse_genres(value: Any) -> list[str]:
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_genres = list(value)
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "(no genres listed)"}:
            return []
        parsed: Any = None
        if text.startswith(("[", "(", "{")):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, dict):
            raw_genres = [genre for genre, enabled in parsed.items() if enabled]
        elif isinstance(parsed, (list, tuple, set)):
            raw_genres = list(parsed)
        else:
            separator = "|" if "|" in text else ","
            raw_genres = text.split(separator)
    return list(
        dict.fromkeys(
            str(genre).strip()
            for genre in raw_genres
            if genre is not None and str(genre).strip()
        )
    )


def prepare_catalog(movies: pd.DataFrame) -> pd.DataFrame:
    catalog = movies.copy()
    movie_id_column = _find_column(
        list(catalog.columns), ("movieId", "movie_id", "item_id", "id")
    )
    title_column = _find_column(
        list(catalog.columns),
        ("title", "movie_title", "movie name", "name", "original_title"),
    )
    genres_column = _find_column(
        list(catalog.columns),
        ("genres", "genre_list", "genre_tags", "genre", "categories"),
    )
    one_hot_columns = _one_hot_genre_columns(list(catalog.columns))
    if title_column is None:
        raise ValueError("Movie catalog must include a title or movie_title column.")
    if genres_column is None and not one_hot_columns:
        raise ValueError(
            "Movie catalog must include genres, genre_list, or known one-hot genre columns."
        )

    if movie_id_column is None:
        catalog["movieId"] = range(1, len(catalog) + 1)
    else:
        catalog["movieId"] = catalog[movie_id_column]
    catalog["movieId"] = pd.to_numeric(catalog["movieId"], errors="coerce")
    catalog["title"] = catalog[title_column].astype("string").str.strip()
    catalog = catalog[(catalog["movieId"] > 0) & (catalog["movieId"] % 1 == 0)]
    catalog = catalog.dropna(subset=["movieId", "title"])
    catalog["movieId"] = catalog["movieId"].astype(int)
    catalog = catalog[catalog["title"].str.len() > 0]
    catalog = catalog.drop_duplicates("movieId").reset_index(drop=True)
    if catalog.empty:
        raise ValueError("Movie catalog contains no valid movie rows.")

    if genres_column is None:
        catalog["genre_tags"] = [
            [
                genre
                for genre, column in one_hot_columns.items()
                if _truthy_genre_flag(row[column])
            ]
            for _, row in catalog.iterrows()
        ]
    else:
        catalog["genre_tags"] = catalog[genres_column].map(_parse_genres)
        if one_hot_columns:
            for index, row in catalog.iterrows():
                catalog.at[index, "genre_tags"] = list(
                    dict.fromkeys(
                        catalog.at[index, "genre_tags"]
                        + [
                            genre
                            for genre, column in one_hot_columns.items()
                            if _truthy_genre_flag(row[column])
                        ]
                    )
                )

    catalog["genres"] = catalog["genre_tags"].map("|".join)
    catalog["_genres_norm"] = catalog["genre_tags"].map(
        lambda genres: "|" + "|".join(str(genre).casefold() for genre in genres) + "|"
    )
    title_parts = catalog["title"].map(_title_without_year)
    catalog["year"] = pd.array([year for _, year in title_parts], dtype="Int64")
    catalog["display_title"] = [title for title, _ in title_parts]
    catalog["formatted_title"] = catalog["title"].map(format_movie_title)
    catalog["_cast_norm"] = (
        catalog["cast"].map(
            lambda value: normalize_search_text(value)
            if isinstance(value, str)
            else ""
        )
        if "cast" in catalog
        else pd.Series("", index=catalog.index, dtype="string")
    )

    rating_column = _find_column(
        list(catalog.columns),
        (
            "predicted_rating",
            "rating_prediction",
            "prediction",
            "prediction_score",
            "predicted_score",
            "svd_prediction",
            "svd_rating",
            "svd_score",
            "rating/prediction",
            "rating",
            "score",
        ),
    )
    catalog["rating_prediction"] = (
        pd.to_numeric(catalog[rating_column], errors="coerce")
        if rating_column is not None
        else np.nan
    )
    return catalog


def parse_preferences(
    prompt: str,
    mood_labels: tuple[str, ...] | list[str] = (),
) -> PreferenceProfile:
    normalized = normalize_search_text(prompt)
    genre_groups: list[frozenset[str]] = []

    verb_love = re.search(
        r"\b(?:i|we|you|they|he|she)\s+love\b",
        normalized,
    )
    for genre, synonyms in GENRE_SYNONYMS.items():
        if genre == "Romance" and verb_love:
            synonyms = tuple(alias for alias in synonyms if alias != "love")
        if any(_contains_phrase(normalized, synonym) for synonym in synonyms):
            genre_groups.append(frozenset({genre}))
    if _contains_phrase(normalized, "rom com") or _contains_phrase(
        normalized, "romcom"
    ):
        genre_groups.extend(
            (frozenset({"Romance"}), frozenset({"Comedy"}))
        )

    canonical_genres = {genre.casefold(): genre for genre in KNOWN_GENRES}
    for mood in mood_labels:
        groups = MOOD_GENRE_GROUPS.get(mood)
        if groups is None:
            canonical_genre = canonical_genres.get(str(mood).casefold())
            groups = (
                (frozenset({canonical_genre}),)
                if canonical_genre is not None
                else ()
            )
        for group in groups:
            genre_groups.append(group)

    if _contains_phrase(normalized, "mind bending") or _contains_phrase(
        normalized, "mind-bending"
    ):
        genre_groups.append(frozenset({"Sci-Fi", "Mystery", "Thriller"}))
    if _contains_phrase(normalized, "tearjerker") or _contains_phrase(
        normalized, "tear jerker"
    ):
        genre_groups.append(frozenset({"Drama", "Romance"}))

    unique_groups = tuple(dict.fromkeys(genre_groups))

    year_min: int | None = None
    year_max: int | None = None
    era_names: list[str] = []
    if any(
        _contains_phrase(normalized, marker)
        for marker in ("old", "classic", "classics", "pre 2005", "before 2005")
    ) or any(
        mood in mood_labels
        for mood in ("Old Classics (<2005)", "Classic Romance (<2005)")
    ):
        year_max = 2004
        era_names.append("pre-2005")

    for mood in mood_labels:
        era_range = ERA_RANGES.get(mood)
        if era_range:
            era_min, era_max, label = era_range
            if era_min is not None:
                year_min = era_min if year_min is None else max(year_min, era_min)
            if era_max is not None:
                year_max = era_max if year_max is None else min(year_max, era_max)
            era_names.append(label)

    decade_patterns = (
        (r"\b(?:1980s|1980 s|80s|80 s|eighties)\b", 1980, 1989, "1980s"),
        (r"\b(?:1990s|1990 s|90s|90 s|nineties)\b", 1990, 1999, "1990s"),
        (r"\b(?:2000s|2000 s|00s|00 s|two thousands)\b", 2000, 2009, "2000s"),
        (r"\b(?:2010s|2010 s|10s|10 s|twenty tens)\b", 2010, 2019, "2010s"),
        (r"\b(?:2020s|2020 s|20s|20 s|twenty twenties)\b", 2020, 2029, "2020s"),
    )
    for pattern, start, end, label in decade_patterns:
        if re.search(pattern, normalized):
            year_min = start if year_min is None else max(year_min, start)
            year_max = end if year_max is None else min(year_max, end)
            era_names.append(label)
            break
    if "90s Nostalgia" in mood_labels:
        year_min = 1990 if year_min is None else max(year_min, 1990)
        year_max = 1999 if year_max is None else min(year_max, 1999)
        if "1990s" not in era_names:
            era_names.append("1990s")

    return PreferenceProfile(
        genre_groups=unique_groups,
        year_min=year_min,
        year_max=year_max,
        era_label=" + ".join(dict.fromkeys(era_names)) or None,
    )


def _title_aliases(catalog: pd.DataFrame) -> dict[str, list[int]]:
    cached_aliases = catalog.attrs.get("_genre_genie_title_aliases")
    if isinstance(cached_aliases, dict):
        return cached_aliases

    aliases: dict[str, list[int]] = {}
    for row in catalog.itertuples(index=False):
        raw_title = str(row.display_title)
        variants = [raw_title]
        article = re.match(r"^(.*),\s*(The|A|An)$", raw_title, flags=re.IGNORECASE)
        if article:
            variants.append(f"{article.group(2)} {article.group(1)}")
        else:
            moved_article = re.match(
                r"^(The|A|An)\s+(.+)$", raw_title, flags=re.IGNORECASE
            )
            if moved_article:
                variants.append(
                    f"{moved_article.group(2)}, {moved_article.group(1)}"
                )
        for variant in variants:
            alias = normalize_search_text(variant)
            if len(alias) >= 3:
                aliases.setdefault(alias, []).append(int(row.movieId))
    catalog.attrs["_genre_genie_title_aliases"] = aliases
    return aliases


def find_prompt_movie_ids(prompt: str, catalog: pd.DataFrame) -> list[int]:
    normalized_prompt = normalize_search_text(prompt)
    if not normalized_prompt:
        return []
    padded_prompt = f" {normalized_prompt} "
    matches: list[tuple[int, int]] = []
    for alias, movie_ids in _title_aliases(catalog).items():
        if f" {alias} " in padded_prompt:
            matches.extend((len(alias), movie_id) for movie_id in movie_ids)
    matches.sort(key=lambda item: (-item[0], item[1]))
    return list(dict.fromkeys(movie_id for _, movie_id in matches))


def fuzzy_match_movie(
    text: str,
    catalog: pd.DataFrame,
    cutoff: float = 0.68,
) -> tuple[int, float] | None:
    aliases = _title_aliases(catalog)
    exact_matches = find_prompt_movie_ids(text, catalog)
    if exact_matches:
        return exact_matches[0], 1.0

    choices = list(aliases)
    source_lines = [Path(line).stem.replace("_", " ").replace("-", " ") for line in text.splitlines()]
    segments: list[str] = []
    for source in source_lines[:12]:
        normalized = normalize_search_text(source)
        if normalized:
            segments.append(normalized)
            words = normalized.split()
            for window_size in range(2, min(9, len(words) + 1)):
                segments.extend(
                    " ".join(words[start : start + window_size])
                    for start in range(len(words) - window_size + 1)
                )

    best: tuple[float, int] | None = None
    for segment in segments:
        if len(segment) < 4:
            continue
        matches = difflib.get_close_matches(segment, choices, n=1, cutoff=cutoff)
        if not matches:
            continue
        alias = matches[0]
        ratio = difflib.SequenceMatcher(None, segment, alias).ratio()
        movie_id = aliases[alias][0]
        if best is None or ratio > best[0]:
            best = (ratio, movie_id)
    return (best[1], best[0]) if best else None


def detect_poster_movie(
    payload: bytes,
    filename: str,
    catalog: pd.DataFrame,
) -> tuple[int, float, str] | None:
    """Try OCR first, then use a fuzzy match against the uploaded filename."""
    try:
        import pytesseract
    except ImportError:
        LOGGER.info("pytesseract is unavailable; using the poster filename.")
        ocr_text = ""
    else:
        try:
            with Image.open(io.BytesIO(payload)) as poster:
                ocr_text = pytesseract.image_to_string(poster)[:1200]
        except (
            OSError,
            RuntimeError,
            pytesseract.TesseractError,
            pytesseract.TesseractNotFoundError,
        ):
            LOGGER.info("Poster OCR is unavailable; using the poster filename.")
            ocr_text = ""

    match = fuzzy_match_movie(ocr_text, catalog) if ocr_text.strip() else None
    if match is not None:
        return match[0], match[1], "poster text"

    filename_text = Path(filename).stem.replace("_", " ").replace("-", " ")
    match = fuzzy_match_movie(filename_text, catalog)
    if match is not None:
        return match[0], match[1], "filename"
    return None


def _genre_sets_for_movie_ids(
    catalog: pd.DataFrame,
    movie_ids: set[int],
) -> list[set[str]]:
    if not movie_ids:
        return []
    return [
        {str(genre).casefold() for genre in genres}
        for genres in catalog.loc[
            catalog["movieId"].isin(movie_ids), "genre_tags"
        ]
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _movie_year(value: Any) -> int | None:
    return int(value) if pd.notna(value) else None


def _shuffle_unfiltered_catalog(
    catalog: pd.DataFrame,
    predictions: dict[int, float] | None,
    excluded: set[int],
    seen: set[int],
    requested_count: int,
    minimum_rating: float,
    minimum_match_pct: int | None,
    shuffle_seed: int,
    era_label: str | None,
) -> list[dict[str, Any]]:
    catalog_ids = catalog["movieId"].to_numpy(dtype=int)
    ratings = pd.to_numeric(
        catalog.get("rating_prediction", pd.Series(np.nan, index=catalog.index)),
        errors="coerce",
    ).to_numpy(dtype=float, copy=True)
    if predictions is not None:
        model_ratings = pd.to_numeric(
            catalog["movieId"].map(predictions), errors="coerce"
        ).to_numpy(dtype=float)
        ratings = np.where(np.isfinite(model_ratings), model_ratings, ratings)
    ratings[~np.isfinite(ratings)] = 0.5

    match_pcts = np.rint(np.clip((ratings - 0.5) / 4.5, 0.0, 1.0) * 100).astype(int)
    available_mask = ~np.isin(catalog_ids, tuple(excluded))
    if minimum_match_pct is not None:
        available_mask &= match_pcts >= minimum_match_pct

    def mask_at_rating(threshold: float) -> np.ndarray:
        return available_mask & (ratings >= threshold)

    best_mask = mask_at_rating(float(minimum_rating))
    best_count = int(np.count_nonzero(best_mask))
    fallback_notice = ""
    if minimum_match_pct is None and best_count < requested_count:
        for relaxed_rating in (
            max(0.5, float(minimum_rating) - 0.5),
            max(0.5, float(minimum_rating) - 1.0),
            0.5,
        ):
            relaxed_mask = mask_at_rating(relaxed_rating)
            relaxed_count = int(np.count_nonzero(relaxed_mask))
            if relaxed_count > best_count:
                best_mask = relaxed_mask
                best_count = relaxed_count
            if relaxed_count >= requested_count:
                best_mask = relaxed_mask
                best_count = relaxed_count
                if relaxed_rating < float(minimum_rating):
                    fallback_notice = (
                        "Lowered the rating threshold slightly to fill your picks."
                    )
                break

        if best_count < requested_count:
            all_ratings_mask = mask_at_rating(0.5)
            all_ratings_count = int(np.count_nonzero(all_ratings_mask))
            if all_ratings_count > best_count:
                best_mask = all_ratings_mask
                best_count = all_ratings_count
                fallback_notice = (
                    "Expanded the era and rating threshold; showing the best available picks."
                )
        if best_count and best_count < requested_count and not fallback_notice:
            fallback_notice = (
                f"Showing {best_count} available picks; no more catalog titles "
                "matched the relaxed filters."
            )

    eligible_positions = np.flatnonzero(best_mask)
    fresh_positions = [
        int(position)
        for position in eligible_positions
        if int(catalog_ids[position]) not in seen
    ]
    seen_positions = [
        int(position)
        for position in eligible_positions
        if int(catalog_ids[position]) in seen
    ]
    rng = random.Random(int(shuffle_seed))
    fresh_count = min(requested_count, len(fresh_positions))
    chosen_positions = rng.sample(fresh_positions, k=fresh_count)
    seen_count = min(requested_count - fresh_count, len(seen_positions))
    if seen_count:
        chosen_positions.extend(rng.sample(seen_positions, k=seen_count))

    selected_rows = catalog.iloc[chosen_positions].itertuples(index=False)
    recommendations: list[dict[str, Any]] = []
    for position, row in zip(chosen_positions, selected_rows):
        rating = float(ratings[position])
        rating_source = getattr(row, "rating_source", None)
        if not isinstance(rating_source, str) or not rating_source.strip():
            rating_source = "SVD" if predictions and int(row.movieId) in predictions else "Catalog"
        reason = (
            f"{era_label} era."
            if era_label
            else "strong SVD predicted rating."
        )
        recommendations.append(
            {
                "movie_id": int(row.movieId),
                "title": row.display_title,
                "formatted_title": row.formatted_title,
                "year": _movie_year(row.year),
                "genres": list(row.genre_tags),
                "predicted_rating": rating,
                "rating_source": rating_source.strip(),
                "tmdb_id": getattr(row, "tmdb_id", None),
                "poster_url": getattr(row, "poster_url", ""),
                "overview": getattr(row, "overview", ""),
                "director": getattr(row, "director", None),
                "cast": getattr(row, "cast", None),
                "production": getattr(row, "production", None),
                "match_pct": int(match_pcts[position]),
                "rank_score": rating,
                "genre_score": None,
                "era_score": None,
                "matching_groups": 0,
                "seen_before": int(row.movieId) in seen,
                "pool_size": best_count,
                "reason": reason,
                "fallback_notice": fallback_notice,
            }
        )
    return recommendations


def _rank_similar_movies(
    catalog: pd.DataFrame,
    predictions: dict[int, float] | None,
    seed_genres: list[set[str]],
    seed_decades: set[int],
    excluded: set[int],
    seen: set[int],
    requested_count: int,
    minimum_rating: float,
    minimum_match_pct: int,
    shuffle_seed: int | None,
) -> list[dict[str, Any]]:
    movie_ids = catalog["movieId"].to_numpy(dtype=int)
    ratings = pd.to_numeric(
        catalog.get("rating_prediction", pd.Series(np.nan, index=catalog.index)),
        errors="coerce",
    ).to_numpy(dtype=float, copy=True)
    if predictions is not None:
        model_ratings = pd.to_numeric(
            catalog["movieId"].map(predictions), errors="coerce"
        ).to_numpy(dtype=float)
        ratings = np.where(np.isfinite(model_ratings), model_ratings, ratings)
    ratings[~np.isfinite(ratings)] = 0.5

    genre_counts = catalog["genre_tags"].map(len).to_numpy(dtype=int)
    normalized_genres = (
        catalog["_genres_norm"]
        if "_genres_norm" in catalog.columns
        else catalog["genre_tags"].map(
            lambda genres: "|" + "|".join(
                str(genre).casefold() for genre in genres
            ) + "|"
        )
    )
    genre_similarity = np.zeros(len(catalog), dtype=float)
    for seed_set in seed_genres:
        pattern = r"(?=\|(?:" + "|".join(
            re.escape(genre) for genre in sorted(seed_set)
        ) + r")\|)"
        shared_genres = normalized_genres.str.count(pattern).to_numpy(dtype=int)
        union_counts = len(seed_set) + genre_counts - shared_genres
        similarity = np.divide(
            shared_genres,
            union_counts,
            out=np.zeros(len(catalog), dtype=float),
            where=union_counts > 0,
        )
        genre_similarity = np.maximum(genre_similarity, similarity)

    years = pd.to_numeric(catalog["year"], errors="coerce").to_numpy(dtype=float)
    era_scores = np.full(len(catalog), np.nan, dtype=float)
    valid_years = np.isfinite(years)
    if seed_decades:
        era_scores[valid_years] = 0.0
        movie_decades = np.floor_divide(years[valid_years].astype(int), 10)
        for decade in seed_decades:
            era_scores[valid_years] = np.maximum(
                era_scores[valid_years],
                np.maximum(0.0, 1.0 - np.abs(movie_decades - decade) * 0.25),
            )

    rating_scores = np.clip((ratings - 0.5) / 4.5, 0.0, 1.0)
    has_era_score = np.isfinite(era_scores)
    match_scores = (
        0.15 * rating_scores
        + 0.65 * genre_similarity
        + 0.2 * np.nan_to_num(era_scores, nan=0.0)
    ) / (0.8 + 0.2 * has_era_score)
    match_pcts = np.rint(match_scores * 100).astype(int)

    eligible = (
        (ratings >= float(minimum_rating))
        & (match_pcts >= int(minimum_match_pct))
        & ~np.isin(movie_ids, tuple(excluded))
    )
    positions = np.flatnonzero(eligible)
    seen_ids = set(seen)
    title_values = catalog["display_title"].to_numpy()
    positions = sorted(
        positions,
        key=lambda position: (
            int(movie_ids[position] in seen_ids),
            -int(match_pcts[position]),
            -genre_similarity[position],
            -(
                era_scores[position]
                if np.isfinite(era_scores[position])
                else 0.0
            ),
            -ratings[position],
            str(title_values[position]).casefold(),
        ),
    )
    pool_size = len(positions)
    fresh_positions = [
        position for position in positions if movie_ids[position] not in seen_ids
    ]
    seen_positions = [
        position for position in positions if movie_ids[position] in seen_ids
    ]
    if shuffle_seed is None:
        chosen_positions = positions[:requested_count]
    else:
        rng = random.Random(int(shuffle_seed))
        fresh_count = min(requested_count, len(fresh_positions))
        chosen_positions = rng.sample(fresh_positions, k=fresh_count)
        remaining_count = min(
            requested_count - fresh_count, len(seen_positions)
        )
        if remaining_count:
            chosen_positions.extend(
                rng.sample(seen_positions, k=remaining_count)
            )

    recommendations: list[dict[str, Any]] = []
    for position in chosen_positions:
        row = catalog.iloc[position]
        movie_genres = {
            str(genre).casefold() for genre in row["genre_tags"]
        }
        shared_genres = sorted(
            {
                genre.title()
                for seed_set in seed_genres
                for genre in movie_genres.intersection(seed_set)
            }
        )
        reason = "Shares " + " + ".join(shared_genres) + " with your reference."
        rating_source = row.get("rating_source")
        if not isinstance(rating_source, str) or not rating_source.strip():
            rating_source = (
                "SVD"
                if predictions is not None and int(row["movieId"]) in predictions
                else "Catalog"
            )
        recommendations.append(
            {
                "movie_id": int(movie_ids[position]),
                "title": row["display_title"],
                "formatted_title": row["formatted_title"],
                "year": _movie_year(row["year"]),
                "genres": list(row["genre_tags"]),
                "predicted_rating": float(ratings[position]),
                "rating_source": rating_source.strip(),
                "tmdb_id": row.get("tmdb_id"),
                "poster_url": row.get("poster_url", ""),
                "overview": row.get("overview", ""),
                "director": row.get("director"),
                "cast": row.get("cast"),
                "production": row.get("production"),
                "match_pct": int(match_pcts[position]),
                "rank_score": float(match_scores[position]),
                "final_score": float(match_scores[position]),
                "genre_score": float(genre_similarity[position]),
                "era_score": (
                    float(era_scores[position])
                    if np.isfinite(era_scores[position])
                    else None
                ),
                "matching_groups": 0,
                "seen_before": int(movie_ids[position]) in seen_ids,
                "pool_size": pool_size,
                "reason": reason,
                "fallback_notice": "",
            }
        )
    return recommendations


def rank_recommendations(
    catalog: pd.DataFrame,
    predictions: dict[int, float] | None,
    prompt: str,
    mood_labels: tuple[str, ...] | list[str] = (),
    seed_movie_ids: set[int] | None = None,
    excluded_movie_ids: set[int] | None = None,
    top_n: int = 6,
    minimum_rating: float = 0.5,
    lambda_val: float = 1.0,
    shuffle_seed: int | None = None,
    seen_movie_ids: set[int] | None = None,
    selected_genres: tuple[str, ...] | list[str] | None = None,
    era: str = "Any Era",
    minimum_match_pct: int | None = None,
) -> list[dict[str, Any]]:
    if selected_genres is None:
        profile_moods = list(mood_labels)
    else:
        profile_moods = list(selected_genres)
        if era != "Any Era":
            profile_moods.append(era)
    profile = parse_preferences(prompt, profile_moods)
    seeds = set(seed_movie_ids or ()) | set(find_prompt_movie_ids(prompt, catalog))
    excluded = set(excluded_movie_ids or ()) | seeds
    seen = set(seen_movie_ids or ())
    requested_count = max(1, int(top_n))
    if (
        shuffle_seed is not None
        and not seeds
        and not profile.genre_groups
        and profile.year_min is None
        and profile.year_max is None
    ):
        return _shuffle_unfiltered_catalog(
            catalog=catalog,
            predictions=predictions,
            excluded=excluded,
            seen=seen,
            requested_count=requested_count,
            minimum_rating=minimum_rating,
            minimum_match_pct=minimum_match_pct,
            shuffle_seed=shuffle_seed,
            era_label=profile.era_label,
        )
    seed_genres = _genre_sets_for_movie_ids(catalog, seeds)
    seed_years = [
        _movie_year(year)
        for year in catalog.loc[catalog["movieId"].isin(seeds), "year"]
    ]
    seed_decades = {seed_year // 10 for seed_year in seed_years if seed_year}
    if (
        minimum_match_pct is not None
        and seed_genres
        and not profile.genre_groups
        and profile.year_min is None
        and profile.year_max is None
    ):
        return _rank_similar_movies(
            catalog=catalog,
            predictions=predictions,
            seed_genres=seed_genres,
            seed_decades=seed_decades,
            excluded=excluded,
            seen=seen,
            requested_count=requested_count,
            minimum_rating=minimum_rating,
            minimum_match_pct=minimum_match_pct,
            shuffle_seed=shuffle_seed,
        )
    preference_genres = {
        genre.casefold() for group in profile.genre_groups for genre in group
    }

    normalized_genres = (
        catalog["_genres_norm"]
        if "_genres_norm" in catalog.columns
        else catalog["genre_tags"].map(
            lambda genres: "|" + "|".join(
                str(genre).casefold() for genre in genres
            ) + "|"
        )
    )
    matching_group_counts = pd.Series(0, index=catalog.index, dtype=int)
    for group in profile.genre_groups:
        group_matches = pd.Series(False, index=catalog.index)
        for genre in group:
            group_matches |= normalized_genres.str.contains(
                str(genre).casefold(), regex=False, na=False
            )
        matching_group_counts += group_matches.astype(int)
    minimum_candidate_group_matches = (
        1 if len(profile.genre_groups) > 1 else len(profile.genre_groups)
    )
    matching_group_values = matching_group_counts.to_numpy(dtype=int)
    candidate_mask = matching_group_values >= minimum_candidate_group_matches
    if excluded:
        candidate_mask &= ~catalog["movieId"].isin(excluded).to_numpy()
    candidate_positions = np.flatnonzero(candidate_mask)
    candidates: list[dict[str, Any]] = []
    candidate_rows = catalog.iloc[candidate_positions].itertuples(index=False)
    for row_position, row in zip(candidate_positions, candidate_rows):
        movie_id = int(row.movieId)
        matching_groups = int(matching_group_values[row_position])
        year = _movie_year(row.year)
        movie_genres = (
            {str(genre).casefold() for genre in row.genre_tags}
            if seed_genres
            else set()
        )
        seed_similarity = (
            max(
                (_jaccard(movie_genres, seed_set) for seed_set in seed_genres),
                default=0.0,
            )
            if seed_genres
            else 0.0
        )
        if profile.genre_groups:
            genre_score = matching_groups / len(profile.genre_groups)
            if seed_genres:
                genre_score = 0.7 * genre_score + 0.3 * seed_similarity
        elif seed_genres:
            genre_score = seed_similarity
        else:
            genre_score = None

        if profile.year_min is not None or profile.year_max is not None:
            within_era = (
                year is not None
                and (profile.year_min is None or year >= profile.year_min)
                and (profile.year_max is None or year <= profile.year_max)
            )
            distance = (
                min(
                    abs(year - profile.year_min)
                    if profile.year_min is not None
                    else float("inf"),
                    abs(year - profile.year_max)
                    if profile.year_max is not None
                    else float("inf"),
                )
                if year is not None
                else 10
            )
            era_score: float | None = (
                1.0 if within_era else max(0.0, 1.0 - distance / 10)
            )
        elif year is not None and seed_decades:
            era_score = (
                max(
                    max(0.0, 1.0 - abs((year // 10) - decade) * 0.25)
                    for decade in seed_decades
                )
                if seed_decades
                else None
            )
        else:
            era_score = None

        prediction_value = (
            predictions.get(movie_id)
            if predictions is not None and movie_id in predictions
            else getattr(row, "rating_prediction", np.nan)
        )
        try:
            predicted_rating = float(prediction_value)
        except (TypeError, ValueError):
            predicted_rating = 0.5
        if not np.isfinite(predicted_rating):
            predicted_rating = 0.5
        rating_source_value = getattr(row, "rating_source", None)
        rating_source = (
            rating_source_value.strip()
            if isinstance(rating_source_value, str) and rating_source_value.strip()
            else (
                "SVD"
                if predictions is not None and movie_id in predictions
                else "Catalog"
            )
        )
        rating_score = min(1.0, max(0.0, (predicted_rating - 0.5) / 4.5))
        components: list[tuple[float, float]] = [(rating_score, 0.15)]
        if genre_score is not None:
            components.append((genre_score, 0.65))
        if era_score is not None:
            components.append((era_score, 0.2))
        match_score = sum(score * weight for score, weight in components) / sum(
            weight for _, weight in components
        )
        match_pct = round(match_score * 100)
        if minimum_match_pct is not None and match_pct < minimum_match_pct:
            continue
        rank_score = predicted_rating
        if genre_score is not None:
            rank_score += 0.6 * genre_score
        if era_score is not None:
            rank_score += 0.25 * era_score
        candidates.append(
            {
                "movie_id": movie_id,
                "title": row.display_title,
                "formatted_title": row.formatted_title,
                "year": year,
                "genres": list(row.genre_tags),
                "predicted_rating": predicted_rating,
                "rating_source": rating_source,
                "tmdb_id": getattr(row, "tmdb_id", None),
                "poster_url": getattr(row, "poster_url", ""),
                "overview": getattr(row, "overview", ""),
                "director": getattr(row, "director", None),
                "cast": getattr(row, "cast", None),
                "production": getattr(row, "production", None),
                "match_pct": match_pct,
                "rank_score": rank_score,
                "genre_score": genre_score,
                "era_score": era_score,
                "matching_groups": matching_groups,
                "seen_before": movie_id in seen,
            }
        )

    def filter_candidates(
        genre_match_count: int,
        year_expansion: int | None,
        minimum: float,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["matching_groups"] < genre_match_count:
                continue
            if candidate["predicted_rating"] < minimum:
                continue
            year = candidate["year"]
            if year_expansion is not None:
                year_min = (
                    profile.year_min - year_expansion
                    if profile.year_min is not None
                    else None
                )
                year_max = (
                    profile.year_max + year_expansion
                    if profile.year_max is not None
                    else None
                )
                if year_min is not None and (year is None or year < year_min):
                    continue
                if year_max is not None and (year is None or year > year_max):
                    continue
            filtered.append(candidate)
        return filtered

    required_genre_matches = len(profile.genre_groups)
    best_candidates = filter_candidates(
        required_genre_matches, 0, float(minimum_rating)
    )
    fallback_notice = ""
    genre_match_threshold = required_genre_matches
    if minimum_match_pct is None:
        if len(best_candidates) < requested_count and required_genre_matches > 1:
            genre_match_threshold = 1
            genre_fallback = filter_candidates(1, 0, float(minimum_rating))
            if len(genre_fallback) > len(best_candidates):
                best_candidates = genre_fallback
                fallback_notice = (
                    "Relaxed the genre intersection to include movies matching any selected genre."
                )

    if minimum_match_pct is None and len(best_candidates) < requested_count:
        relaxation_steps = (
            (3, max(0.5, float(minimum_rating) - 0.5)),
            (7, max(0.5, float(minimum_rating) - 1.0)),
            (None, 0.5),
        )
        for year_expansion, relaxed_rating in relaxation_steps:
            relaxed_candidates = filter_candidates(
                genre_match_threshold, year_expansion, relaxed_rating
            )
            if len(relaxed_candidates) > len(best_candidates):
                best_candidates = relaxed_candidates
            if len(relaxed_candidates) >= requested_count:
                best_candidates = relaxed_candidates
                if profile.year_min is not None or profile.year_max is not None:
                    fallback_notice = " ".join(
                        part
                        for part in (
                            fallback_notice,
                            f"Expanded the era slightly to fill {requested_count} picks.",
                        )
                        if part
                    )
                if relaxed_rating < float(minimum_rating):
                    fallback_notice = " ".join(
                        part
                        for part in (
                            fallback_notice,
                            "Lowered the rating threshold slightly to fill your picks.",
                        )
                        if part
                    )
                break

        if len(best_candidates) < requested_count:
            all_eras_and_ratings = filter_candidates(
                genre_match_threshold, None, 0.5
            )
            if len(all_eras_and_ratings) > len(best_candidates):
                best_candidates = all_eras_and_ratings
                fallback_notice = " ".join(
                    part
                    for part in (
                        fallback_notice,
                        "Expanded the era and rating threshold; showing the best available picks.",
                    )
                    if part
                )

        if (
            minimum_match_pct is None
            and best_candidates
            and len(best_candidates) < requested_count
            and not fallback_notice
        ):
            fallback_notice = (
                f"Showing {len(best_candidates)} available picks; no more catalog titles "
                "matched the relaxed filters."
            )

    eligible_pool_size = len(best_candidates)
    best_candidates.sort(
        key=lambda item: (
            item["seen_before"],
            -item["rank_score"],
            item["title"].casefold(),
        )
    )
    exploration_pool = best_candidates
    normalized_lambda = min(max(float(lambda_val), 0.0), 5.0) / 5.0
    score_values = [candidate["predicted_rating"] for candidate in exploration_pool]
    score_min = min(score_values, default=0.0)
    score_span = max(score_values, default=0.0) - score_min
    rng_seed = 0 if shuffle_seed is None else int(shuffle_seed)
    rng = random.Random(rng_seed)
    for candidate in exploration_pool:
        svd_rating = (
            1.0
            if score_span == 0
            else (candidate["predicted_rating"] - score_min) / score_span
        )
        diversity_jitter = rng.random()
        candidate["diversity_jitter"] = diversity_jitter
        candidate["final_score"] = (
            (1.0 - normalized_lambda) * svd_rating
            + normalized_lambda * diversity_jitter
        )
        candidate["pool_size"] = eligible_pool_size

    exploration_pool.sort(
        key=lambda item: (
            item["seen_before"],
            -item["final_score"],
            item["title"].casefold(),
        )
    )
    if shuffle_seed is not None:
        fresh_pool = [item for item in exploration_pool if not item["seen_before"]]
        seen_pool = [item for item in exploration_pool if item["seen_before"]]
        fresh_count = min(requested_count, len(fresh_pool))
        chosen = rng.sample(fresh_pool, k=fresh_count)
        remaining_count = min(requested_count - fresh_count, len(seen_pool))
        if remaining_count:
            chosen.extend(rng.sample(seen_pool, k=remaining_count))
    else:
        chosen = exploration_pool[:requested_count]

    for selected in chosen:
        matched_genres = sorted(
            {genre.title() for genre in selected["genres"]}.intersection(
                {genre.title() for genre in preference_genres}
            )
        )
        reason_parts: list[str] = []
        if matched_genres:
            reason_parts.append(" + ".join(matched_genres))
        elif selected["genre_score"] is not None:
            reason_parts.append("shares genres with your picks")
        if profile.era_label:
            reason_parts.append(profile.era_label + " era")
        if not reason_parts:
            reason_parts.append("strong SVD predicted rating")
        selected["reason"] = " & ".join(reason_parts) + "."
        selected["fallback_notice"] = fallback_notice
    return chosen
