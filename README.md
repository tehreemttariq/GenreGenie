# GenreGenie

GenreGenie is a data-driven movie recommendation project designed to reduce choice overload in streaming platforms by combining collaborative filtering behavior with enriched movie metadata. The project follows a structured pipeline from data acquisition to cleaning, feature engineering, and exploratory analysis in preparation for recommendation modeling.

## Overview

This project uses the MovieLens dataset as the primary source of user-item interaction data and enriches it with metadata from The Movie Database (TMDB) to provide additional movie context. The goal is to build a recommendation system that uses both user behavior and movie characteristics to generate more relevant suggestions.

- Track: Track A (Application Development / R&D)
- Team members:
  - `l242519`
  - `l242590`

## Project Objectives

- Define a real-world movie recommendation problem
- Explain why static rule-based systems are insufficient for personalization
- Build a robust data pipeline for collection, cleaning, and processing
- Engineer meaningful user and movie features
- Perform exploratory analysis to uncover behavioral and content trends
- Prepare the dataset for future recommendation model development

## Phase 1: Problem Definition and Requirements

Phase 1 establishes the conceptual and technical foundation of the project.

### 1. Problem Definition
- Address the challenge of content overload faced by users on large streaming platforms
- Explain why users struggle to identify relevant content from broad libraries
- Establish the need for personalized and adaptive recommendation systems

### 2. Target Users and Use Case
- End users seeking tailored movie recommendations
- Developers and researchers working on recommendation systems
- Product scenarios where content discoverability and engagement are important

### 3. Machine Learning Justification
- Show why rule-based logic is limited for large-scale personalization
- Highlight sparse interaction patterns, latent preferences, and scalability issues
- Emphasize the role of collaborative filtering and latent representation learning

### 4. Data Source and Methodology Planning
- Use MovieLens as the main user-item interaction dataset
- Incorporate TMDB metadata for richer movie context
- Define a hybrid recommendation strategy combining user behavior and movie attributes

## Phase 2: Data Preparation and Feature Engineering

Phase 2 focuses on transforming raw data into a clean and analysis-ready dataset.

### 1. Data Acquisition
- Load raw MovieLens datasets: `movies.csv`, `ratings.csv`, and `links.csv`
- Examine dataset structure, relationships, and completeness
- Document the TMDB enrichment workflow for movie metadata

### 2. Data Cleaning and Validation
- Remove duplicate records
- Handle missing values appropriately
- Exclude invalid or unusable rows
- Enforce a consistent rating scale for reliability
- Standardize identifiers to support correct joins and merges

### 3. Feature Engineering
- Create user-level behavioral features such as rating count, mean rating, and variability
- Create movie-level features such as popularity, average rating, and genre signal
- Encode categorical variables for modeling compatibility
- Apply normalization and scaling where needed for analysis and downstream modeling

### 4. Exploratory Data Analysis
- Analyze rating distributions and popularity patterns
- Examine genre composition and user activity behavior
- Review relationships between engineered features
- Generate visual summaries to support interpretation and reporting

## Python Scripts and Their Roles

### 1. `data_scripts/01_tmdb_collector.py`
This script documents the data collection process and summarizes the MovieLens dataset while outlining the TMDB enrichment workflow. It is used to verify the raw dataset structure and establish the basis for metadata augmentation.

Key responsibilities:
- reads and summarizes the raw MovieLens files
- reports dataset size and user/movie counts
- analyzes genre distribution and rating ranges
- documents the TMDB data collection strategy and enrichment context

### 2. `data_scripts/02_data_cleaning.py`
This script performs the main preprocessing and cleaning steps required before modeling. It prepares the raw dataset into a cleaner form by handling inconsistent records and validating rating quality.

Key responsibilities:
- removes duplicate rows
- fills missing genre values with `Unknown`
- drops null rating entries
- retains the latest rating for each user-movie pair
- removes invalid ratings outside the `[0.5, 5.0]` range
- normalizes ratings to a 0-1 scale
- cleans and standardizes `tmdbId` values for accurate linkage
- writes cleaned outputs to `data/cleaned/`

### 3. `data_scripts/03_feature_strategy.py`
This script creates the engineered features and exploratory visualizations used to understand the dataset before recommendation modeling. It transforms the cleaned data into a more informative feature representation.

Key responsibilities:
- builds user-level engagement features
- builds movie-level popularity and quality features
- encodes primary genre categories
- applies Min-Max and standard scaling
- generates EDA plots for rating distribution, genre spread, user activity, correlations, and popularity versus rating
- saves processed feature datasets to `data/processed/`

## Repository Structure

```text
GenreGenie/
├── data/
│   ├── raw/                     # Original MovieLens datasets
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── links.csv
│   ├── cleaned/                 # Cleaned datasets after preprocessing
│   │   ├── movies_cleaned.csv
│   │   ├── ratings_cleaned.csv
│   │   └── links_cleaned.csv
│   └── processed/               # Engineered feature outputs
│       ├── movies_features.csv
│       └── user_features.csv
├── data_scripts/
│   ├── 01_tmdb_collector.py     # Data collection and TMDB workflow summary
│   ├── 02_data_cleaning.py      # Data cleaning and validation pipeline
│   ├── 03_feature_strategy.py   # Feature engineering and exploratory analysis
│   ├── README.md
│   └── requirements.txt
├── docs/
│   ├── report.tex               # Technical report
│   └── figures/
│       ├── rating_distribution.png
│       ├── genre_distribution.png
│       ├── user_activity.png
│       ├── correlation_matrix.png
│       └── popularity_vs_rating.png
├── README.md
├── .gitignore
└── LICENSE (if added in the future)
```

## Completed Work

### Data Collection
- Reviewed the raw MovieLens dataset and mapped its structure
- Defined the TMDB enrichment approach for metadata augmentation
- Established a hybrid recommendation foundation combining interaction data and movie metadata

### Cleaning and Preprocessing
- Removed duplicate rows and inconsistent entries
- Resolved missing genre values and invalid ratings
- Standardized identifiers and rating scales
- Prepared clean datasets suitable for feature engineering and analysis

### Feature Engineering
- Built user-centric behavior features
- Built movie-centric quality and popularity features
- Applied scaling and encoding to support meaningful downstream analysis

### Exploratory Data Analysis
- Generated visual summaries of ratings, genres, user activity, and feature relationships
- Used the outputs to understand behavioral patterns and dataset quality

## Generated Outputs

- `data/cleaned/movies_cleaned.csv`
- `data/cleaned/ratings_cleaned.csv`
- `data/cleaned/links_cleaned.csv`
- `data/processed/movies_features.csv`
- `data/processed/user_features.csv`
- `docs/figures/*.png`

## Setup and Execution

### Install dependencies

```bash
pip install -r data_scripts/requirements.txt
```

### Run the pipeline from the project root

```bash
python data_scripts/02_data_cleaning.py
python data_scripts/03_feature_strategy.py
```

### Optional data review

```bash
python data_scripts/01_tmdb_collector.py
```

## Documentation

- `docs/report.tex` contains the project technical report
- The report covers the problem definition, machine learning rationale, dataset strategy, feature engineering, and early exploratory findings

## Current Status

The project has completed the foundational data preparation and analysis stages. The next step is to build and evaluate a recommendation model using the cleaned and engineered data to generate personalized movie recommendations.
