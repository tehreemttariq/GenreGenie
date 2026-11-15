# GenreGenie

GenreGenie is a dark, three-page Streamlit movie-discovery app. Choose genres
and an era, optionally reference movies or upload a poster/watchlist, and let a
regularized SVD regression model and strict metadata filters rank suggestions.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects `data/cleaned/movies_cleaned.csv`,
`data/cleaned/ratings_cleaned.csv`, and (for trained scores)
`models/svd_model.pkl`. The catalog, model, and movie metadata are cached. If
the model is missing or incompatible, GenreGenie warns and uses deterministic
fallback scores so the interface remains usable.

## Features

- Discover, Recommendations, and About & Model pages with persistent
  navigation.
- Select multiple genres with strict intersection filtering; when that pool is
  too small, matching relaxes to any selected genre without becoming unfiltered.
  Choose a release era: before 1995, 1995–2005, or 2006+.
- Movie catalog normalization supports pipe-separated, Python-list, and
  one-hot genre columns, plus common title and predicted-rating aliases.
- The active filter state is shared by the badges, ranker, and criteria bar.
  Pool size is counted after filtering and before the requested result count.
- When strict filters yield too few choices, the ranker incrementally relaxes
  era and score thresholds and labels the expanded result set.
- Seeded ranking blends normalized SVD ratings with controlled exploration
  across the eligible pool. Shuffle rotates that pool.
- Previously shown movies are deprioritized across searches in the same
  session; they return only when too few fresh titles match.
- Reference movies through the searchable selector or upload a poster; title
  recognition uses OCR when available and fuzzy filename matching otherwise.
- Tune SVD regularization / exploration, minimum predicted rating, and 3–12
  results.
- Upload a cleaned-schema watchlist CSV with `movieId`, `title`, and `genres`,
  or upload a JPG/PNG poster. Files are limited to 5 MB and image headers are
  validated.
- Three-column movie cards show feature-derived match scores, SVD ratings,
  genres, and a complete two-sentence hook; a detail dialog provides the full
  synopsis and a similar-movie action.
- Titles are normalized before querying Wikipedia, iTunes, TVMaze, and Wikidata
  through cached sessions with four concurrent lookups. Custom violet posters
  with title initials, year, and genre keep cards intact when public poster
  sources are unavailable.

## Tests

```powershell
python -m pytest -q test_app.py test_recommendation_engine.py test_movie_metadata.py
```

The tests cover the Streamlit discovery flow and invalid CSV uploads, genre and
era filtering, poster-title matching, dynamic match scores, and keyless
metadata/fallback behavior.
