# Data Scripts - GenreGenie

This folder contains all Phase 2 data processing scripts for the GenreGenie project.

## Setup

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Scripts

Execute scripts **from project root** in order:

### 1. Data Collection (`01_tmdb_collector.py`)
```bash
python data_scripts/01_tmdb_collector.py
```
- Analyzes existing MovieLens data in `data/raw/`
- Documents TMDB enriched data collection

### 2. Data Cleaning (`02_data_cleaning.py`)
```bash
python data_scripts/02_data_cleaning.py
```
- Handles missing values
- Removes duplicates
- Filters outliers
- Outputs to: `data/cleaned/`

### 3. Feature Engineering & EDA (`03_feature_strategy.py`)
```bash
python data_scripts/03_feature_strategy.py
```
- Creates user and movie features
- Applies normalization and scaling
- Encodes categorical variables
- Generates EDA visualizations
- Outputs to: `data/processed/` and `docs/figures/`

## Project Structure

```
GenreGenie/
├── data/
│   ├── raw/              # Original datasets
│   ├── cleaned/          # Cleaned datasets
│   └── processed/        # Engineered features
├── data_scripts/         # Processing scripts
└── docs/
    └── figures/          # EDA visualizations
```
