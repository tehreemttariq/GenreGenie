# GenreGenie

GenreGenie is an editorial, three-page Streamlit movie-discovery app. Choose genres
and an era, optionally reference movies or upload a poster/watchlist, and let a
regularized SVD regression model and strict metadata filters rank suggestions.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects `data/cleaned/movies_cleaned.csv`,
`data/cleaned/ratings_cleaned.csv`, and (for trained scores)
`models/svd_model.pkl`. On first launch, TMDB full loads are normalized into a
compact `data/processed/tmdb_catalog.csv` cache; later launches reuse it until
the source files change. The in-memory catalog, model, and movie metadata are
also cached. If the model is missing or incompatible, GenreGenie warns and uses
deterministic fallback scores so the interface remains usable.

## Features

- An About / welcome landing page, Discover workspace, and My Recommendations
  gallery with persistent sidebar navigation.
- Select from the catalog's full genre list with strict intersection filtering;
  when that pool is too small, matching relaxes to any selected genre without
  becoming unfiltered. Choose a classic, decade, or modern release era.
- Movie catalog normalization supports pipe-separated, Python-list, and
  one-hot genre columns, plus common title and predicted-rating aliases.
- The active filter state is shared by the badges, ranker, and criteria bar.
  Pool size is counted after filtering and before the requested result count.
- When strict filters yield too few choices, the ranker incrementally relaxes
  era and score thresholds and labels the expanded result set.
- The MovieLens sample catalog is supplemented by TMDB full loads covering
  2005 onward and an all-genre incremental catalog. Linked MovieLens titles
  are not duplicated; TMDB-only records use TMDB community ratings while
  MovieLens titles continue to use SVD predictions.
- Seeded ranking blends normalized SVD ratings with controlled exploration
  across the eligible pool. Shuffle rotates that pool.
- Previously shown movies are deprioritized across searches in the same
  session; they return only when too few fresh titles match.
- Reference movies through the searchable selector or upload a poster; title
  recognition uses OCR when available and fuzzy filename matching otherwise.
- Tune SVD regularization / exploration, minimum rating, and a 1–30-title
  shortlist on Discover; the same controls remain available on Recommendations.
- Upload a cleaned-schema watchlist CSV with `movieId`, `title`, and `genres`,
  or upload a JPG/PNG poster. Files are limited to 5 MB and image headers are
  validated.
- Three-column movie cards show feature-derived match scores, SVD ratings,
  genres, and a complete two-sentence hook; compact details include a scrollable
  full synopsis, available director/cast/production credits, and a
  similar-movie action that requires at least a 75% match.
- Titles are normalized before querying Wikipedia, iTunes, TVMaze, and Wikidata
  through cached sessions with four concurrent lookups. A restrained
  “Artwork unavailable” placeholder is shown when public sources cannot supply
  a real poster.

## TMDB catalog data and incremental refresh

Place the TMDB full-load files at
`data/raw/full_load_2005_2014.json` and `data/raw/full_load.json`. GenreGenie
streams their `results` arrays and adds all available genres, years, posters,
synopses, ratings, and enriched credits to the catalog. It also reads the
legacy `tmdb_horror_catalog.json` when present. To refresh the full catalog
incrementally, set a TMDB API key or read-access token in your local
environment and run:

```powershell
# Use the v3 API key:
$env:TMDB_API_KEY = "<your TMDB API key>"
# Or, instead, use the v4 Read Access Token:
# $env:TMDB_READ_ACCESS_TOKEN = "<your TMDB read access token>"
python data_scripts/04_tmdb_incremental_sync.py
```

The sync includes all TMDB movie genres, uses the latest full-load pull date as
its starting point on the first run, checkpoints after every page, resumes
after interruptions, and upserts by TMDB movie ID. Oversized date ranges are
split automatically to stay within TMDB Discover's 500-page limit. The app
reads `data/raw/tmdb_incremental_catalog.json` automatically on the next rerun.
This product uses TMDB data but is not endorsed or certified by TMDB.

## Retrain the recommendation model

After updating MovieLens ratings, fit the production SVD model on all available
user ratings with:

```powershell
python data_scripts/05_retrain_svd_full.py
```

The script saves `models/svd_model.pkl` with its user/movie ID mappings and a
separate split-only `models/svd_model_evaluation.pkl` for unbiased holdout
evaluation. TMDB vote averages are movie-level community summaries, not
user-level observations, so they enrich catalog ratings but are not fabricated
into SVD training samples.

## Tests

```powershell
python -m pytest -q test_app.py test_recommendation_engine.py test_movie_metadata.py
```

The tests cover the Streamlit discovery flow and invalid CSV uploads, genre and
era filtering, poster-title matching, dynamic match scores, and keyless
metadata/fallback behavior.
