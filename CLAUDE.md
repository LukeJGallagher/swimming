# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swimming performance analysis system for Team Saudi. Scrapes World Aquatics API for competition results (2000-2026), stores data in Azure Blob Storage as Parquet, and provides evidence-based coaching analytics with DuckDB queries.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper (saves to Azure Blob)
python scraper_swimming.py                # Scrape current year
python scraper_swimming.py --year 2024    # Scrape specific year
python scraper_swimming.py --migrate      # Migrate local CSVs to Azure

# Launch coaching dashboard
streamlit run coaching_dashboard.py

# Test Azure connection
python blob_storage.py

# Run tests
python test_scraper.py
```

## Architecture

### Data Storage Layer (Azure Blob + DuckDB)

- **blob_storage.py** - Azure Blob Storage with DuckDB queries
  - `load_results()` - Load data from Azure (or local CSV fallback)
  - `save_results()` - Save DataFrame to Azure as Parquet
  - `query(sql)` - Execute SQL against data using DuckDB
  - `get_athlete_results()`, `get_event_rankings()`, `get_yearly_summary()` - Convenience queries
  - `migrate_csv_to_parquet()` - One-time migration from CSV to Azure
  - `create_backup()` - Create timestamped backup before writes

- **scraper_swimming.py** - GitHub Actions scraper
  - `WorldAquaticsAPI` - API client with 1.5s rate limiting
  - `SplitTimeAnalyzer` - Parse splits, calculate lap times, classify pacing
  - Runs weekly via `.github/workflows/scraper.yml`

### Analytics Layer

- **coaching_analytics.py** - Research-backed coaching analytics
  - `AdvancedPacingAnalyzer` - Pacing strategies (U-shape, Inverted-J, Even, Positive, Negative)
  - `TalentDevelopmentTracker` - WR%, competition age, progression
  - `RaceRoundAnalyzer` - Heats-to-finals improvement
  - `CompetitorIntelligence` - Tactical profiling
  - `WORLD_RECORDS_LCM`, `ELITE_BENCHMARKS` - Reference data

### Dashboard Layer

- **coaching_dashboard.py** - Main Streamlit dashboard with Team Saudi branding
- **dashboard.py** - Basic dashboard

### Data Flow

```
World Aquatics API → scraper_swimming.py → Azure Blob (master.parquet)
                                                    ↓
                     blob_storage.py ← load_results() / query()
                                                    ↓
                     coaching_dashboard.py → Interactive Analysis
```

## Azure Blob Storage

Data is stored in Azure Blob Storage, not in git. Configuration in `blob_storage.py`:

```python
CONTAINER_NAME = "swimming-data"
MASTER_FILE = "master.parquet"
STORAGE_ACCOUNT_URL = "https://worldaquatics.blob.core.windows.net/"
```

Connection string in `.env`:
```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

## DuckDB Queries

```python
from blob_storage import query, get_country_summary

# Custom SQL (table name is 'swimming')
df = query("SELECT * FROM swimming WHERE NAT='KSA' ORDER BY year DESC")

# Built-in helpers
df = get_country_summary('KSA')
```

## Key Research Metrics

| Metric | Elite Benchmark |
|--------|-----------------|
| Pacing CV (lap variance) | < 1.3% |
| Years to elite (>900 FINA pts) | ~8 years |
| Peak age (male/female) | 24.2 / 22.5 years |
| Heats-to-finals improvement | > 1.2% for medalists |

## World Aquatics API

Base URL: `https://api.worldaquatics.com/fina`

Endpoints:
- `/competitions` - List by date range
- `/competitions/{id}/events` - Events in competition
- `/events/{id}` - Detailed results with splits

No API key required. Rate limit: 1.5s between requests.

## GitHub Actions

Weekly scraper runs via `.github/workflows/scraper.yml`. Requires `AZURE_STORAGE_CONNECTION_STRING` secret.

## Team Saudi Branding

```python
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
```

## File Organization

- **archive/** - Local CSV backups (not in git)
- **data/** - Enriched data files
- **.claude/** - Agent/skill definitions for Claude Code
