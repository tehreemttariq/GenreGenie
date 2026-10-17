# Data Scripts - GenreGenie

This folder contains all Phase 2 data processing scripts for the GenreGenie project.

## Setup

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Scripts

Execute scripts in order:

### 1. Data Collection (`01_tmdb_collector.py`)
```bash
python 01_tmdb_collector.py
```
- Loads existing MovieLens data (movies.csv, ratings.csv, links.csv)
- **Optional**: Enriches data from TMDB API (requires API key)
- Get free TMDB API key: https://www.themoviedb.org/settings/api

### 2. Data Cleaning (`02_data_cleaning.py`)
```bash
python 02_data_cleaning.py
```
- Handles missing values
- Removes duplicates
- Filters outliers
- Outputs: `movies_cleaned.csv`, `ratings_cleaned.csv`, `links_cleaned.csv`

### 3. Feature Engineering & EDA (`03_feature_strategy.py`)
```bash
python 03_feature_strategy.py
```
- Creates user and movie features
- Applies normalization and scaling
- Encodes categorical variables
- Generates EDA visualizations
- Outputs: `movies_features.csv`, `user_features.csv`, and figures in `../docs/figures/`

## Output Files

### Cleaned Data
- `movies_cleaned.csv` - Cleaned movie metadata
- `ratings_cleaned.csv` - Cleaned user ratings
- `links_cleaned.csv` - Cleaned external IDs

### Engineered Features
- `movies_features.csv` - Movies with engineered features
- `user_features.csv` - User-level aggregated features

### Visualizations
All plots are saved to `../docs/figures/`:
- `rating_distribution.png` - Distribution of user ratings
- `genre_distribution.png` - Top movie genres
- `user_activity.png` - User activity patterns
- `correlation_matrix.png` - Feature correlations
- `popularity_vs_rating.png` - Popularity vs quality analysis

## Notes

- Scripts must be run from this directory (`data_scripts/`)
- Original CSV files (movies.csv, ratings.csv, links.csv) must be present
- TMDB API collection is optional for student demo purposes
