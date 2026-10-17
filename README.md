# GenreGenie
Transforms raw movie streaming telemetry into personalized recommendations by merging TMDB's live API pipelines with collaborative user behaviour models.

## Project Overview
**GenreGenie** addresses choice overload on digital streaming platforms by dynamically synthesizing real-time movie metadata from The Movie Database (TMDB) REST API with collaborative user interaction filtering models.

* **Track:** Track A (Application Development / R&D)
* **Team Members:** 
  * `l242519`
  * `l242590`

---

## Repository Structure
```text
GenreGenie/
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── links.csv
│   ├── cleaned/              # Cleaned datasets
│   │   ├── movies_cleaned.csv
│   │   ├── ratings_cleaned.csv
│   │   └── links_cleaned.csv
│   └── processed/            # Engineered features
│       ├── movies_features.csv
│       └── user_features.csv
├── data_scripts/             # ETL, data cleaning, and feature engineering scripts
│   ├── 01_tmdb_collector.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_strategy.py
│   ├── README.md
│   └── requirements.txt
├── docs/                     # Project deliverables, LaTeX reports, and EDA figures
│   ├── report.tex
│   └── figures/
│       ├── rating_distribution.png
│       ├── genre_distribution.png
│       ├── user_activity.png
│       ├── correlation_matrix.png
│       └── popularity_vs_rating.png
└── README.md
```

## Setup & Installation

1. **Install dependencies:**
```bash
pip install -r data_scripts/requirements.txt
```

2. **Run data pipeline:**
```bash
# From project root
python data_scripts/01_tmdb_collector.py  # Data collection summary
python data_scripts/02_data_cleaning.py    # Clean data
python data_scripts/03_feature_strategy.py # Feature engineering & EDA
```

## Data Pipeline

### Phase 1: Data Collection
- **MovieLens dataset**: User-movie ratings and metadata
- **TMDB API enrichment**: Budget, revenue, runtime, cast, keywords

### Phase 2: Data Cleaning
- Handles missing values and duplicates
- Removes outliers from ratings
- Outputs cleaned datasets to `data/cleaned/`

### Phase 3: Feature Engineering
- User-level features: rating patterns, activity metrics
- Movie-level features: popularity, quality indicators
- Normalization (Min-Max) and standardization (Z-score)
- Genre encoding
- Generates 5 EDA visualizations

## Outputs

- **Cleaned Data**: `data/cleaned/`
- **Engineered Features**: `data/processed/`
- **EDA Visualizations**: `docs/figures/`

## Documentation

Full technical report available in `docs/report.tex` (compile with `pdflatex`).