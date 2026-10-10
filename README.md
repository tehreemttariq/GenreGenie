# GenreGenie
Transforms raw movie streaming telemetry into personalized recommenadationsby merging TMDB's live API pipelines with collaborative user behaviour models.

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
├── data_scripts/          # ETL, data cleaning, and feature engineering scripts
│   ├── 01_tmdb_collector.py
│   ├── 02_data_cleaning.py
│   └── 03_feature_strategy.py
├── docs/                  # Project deliverables, LaTeX reports, and EDA figures
│   └── report.tex
└── README.md