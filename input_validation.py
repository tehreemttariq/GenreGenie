"""Input sanitization and upload validation for GenreGenie."""

import io
import re
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
REQUIRED_CSV_COLUMNS = {"movieId", "title", "genres"}
OPTIONAL_CSV_COLUMNS = {"genre_list", "genre_count"}


def sanitize_prompt(prompt: str) -> str:
    """Normalize user-provided prompt text and remove control characters."""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(prompt))
    return re.sub(r"\s+", " ", cleaned).strip()[:500]


def validate_watchlist_csv(
    payload: bytes,
) -> tuple[pd.DataFrame | None, str | None, int]:
    """Validate a watchlist CSV against the cleaned catalog's movie schema."""
    if len(payload) > MAX_UPLOAD_BYTES:
        return None, "Files must be 5 MB or smaller.", 0
    if not payload:
        return None, "The uploaded CSV is empty.", 0
    try:
        watchlist = pd.read_csv(
            io.BytesIO(payload),
            dtype={"movieId": "string", "title": "string", "genres": "string"},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None, "That file could not be read as a CSV. Please check its format.", 0

    missing = REQUIRED_CSV_COLUMNS.difference(watchlist.columns)
    if missing:
        names = ", ".join(sorted(missing))
        return (
            None,
            f"CSV is missing required cleaned-dataset columns: {names}.",
            0,
        )
    unexpected = set(watchlist.columns).difference(
        REQUIRED_CSV_COLUMNS | OPTIONAL_CSV_COLUMNS
    )
    if unexpected:
        names = ", ".join(sorted(unexpected))
        return None, f"CSV contains unsupported columns: {names}.", 0

    original_rows = len(watchlist)
    watchlist["movieId"] = pd.to_numeric(watchlist["movieId"], errors="coerce")
    watchlist = watchlist.dropna(subset=["movieId", "title", "genres"]).copy()
    watchlist = watchlist[
        (watchlist["movieId"] > 0) & (watchlist["movieId"] < 2**63)
    ]
    watchlist["title"] = watchlist["title"].astype(str).str.strip()
    watchlist["genres"] = watchlist["genres"].astype(str).str.strip()
    watchlist = watchlist[(watchlist["title"] != "") & (watchlist["genres"] != "")]
    watchlist = watchlist[watchlist["movieId"] % 1 == 0]
    watchlist["movieId"] = watchlist["movieId"].astype(int)
    watchlist = watchlist.drop_duplicates("movieId").reset_index(drop=True)
    dropped_rows = original_rows - len(watchlist)
    if watchlist.empty:
        return None, "CSV has no valid movie rows after checking required values.", 0
    return watchlist, None, dropped_rows


def validate_poster(payload: bytes, filename: str) -> tuple[bool, str]:
    """Verify an uploaded image's actual header and supported format."""
    if len(payload) > MAX_UPLOAD_BYTES:
        return False, "Files must be 5 MB or smaller."
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
                image_format = image.format
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return False, "This file is not a valid JPG or PNG image."
    extension = Path(filename).suffix.lower()
    expected_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}.get(
        extension
    )
    if image_format not in {"JPEG", "PNG"} or image_format != expected_format:
        return False, "Image content must match its .jpg, .jpeg, or .png extension."
    return True, ""
