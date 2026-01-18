# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swimming performance analysis system for Team Saudi. Scrapes World Aquatics API for competition results (2000-2026), extracts split times, and provides evidence-based coaching analytics. Built for performance analysts tracking Saudi swimmers against international competition.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper
python enhanced_swimming_scraper.py

# Launch coaching dashboard (recommended)
streamlit run coaching_dashboard.py

# Launch basic dashboard
streamlit run dashboard.py

# Run tests
python test_scraper.py

# Quick demo of coaching analytics
python coaching_analytics.py
```

## Architecture

### Data Collection Layer

- **enhanced_swimming_scraper.py** - Main data collection module
  - `WorldAquaticsAPI` - API client with rate limiting (1.5s between requests)
  - `SplitTimeAnalyzer` - Parses split times, calculates lap times and pacing metrics
  - `EnhancedSwimmingScraper` - Orchestrates scraping by year/athlete

### Analytics Layer

- **coaching_analytics.py** - Evidence-based coaching analytics (research-backed)
  - `AdvancedPacingAnalyzer` - Classifies pacing strategies (U-shape, Inverted-J, Fast-start-even, Positive, Negative, Even)
  - `TalentDevelopmentTracker` - Competition age, WR%, age progression, annual improvement
  - `RaceRoundAnalyzer` - Heats-to-finals progression analysis
  - `CompetitorIntelligence` - Tactical competitor profiling
  - `WORLD_RECORDS_LCM` - Current world records for benchmarking
  - `ELITE_BENCHMARKS` - Research-based thresholds (CV < 1.3%, 8 years to elite, etc.)

- **performance_analyst_tools.py** - Basic analysis workflows
  - `AthleteProfiler`, `ProgressionTracker`, `CompetitionAnalyzer`, `ReportGenerator`

- **ai_enrichment.py** - AI-powered enrichment via OpenRouter free models

### Dashboard Layer

- **coaching_dashboard.py** - Elite coaching dashboard with Team Saudi branding
  - Talent development tracking, pacing analysis, competitor intelligence, race preparation
- **dashboard.py** - Basic Streamlit dashboard

### Data Flow

```
World Aquatics API → enhanced_swimming_scraper.py → Results_YYYY.csv / data/*.csv
                                                              ↓
                    coaching_analytics.py ← load_all_results()
                                                              ↓
                    coaching_dashboard.py → Interactive Analysis
```

## Key Research-Based Metrics

From peer-reviewed sports science (Frontiers, PLOS ONE):

| Metric | Elite Benchmark | Source |
|--------|-----------------|--------|
| Pacing CV (lap variance) | < 1.3% | World Championships 2017-2024 |
| Years to elite (>900 FINA pts) | ~8 years | Career trajectory studies |
| Peak age (male/female) | 24.2 / 22.5 years | PLOS ONE 2024 |
| Heats-to-finals improvement | > 1.2% for medalists | Race analysis |
| Final 100m position for medal | Top 3 | 90%+ correlation |

## Data Files

Root directory:
- `Results_YYYY.csv` - Annual results with splits (2000-2026)

Data directory (`data/`):
- `enriched_Results_YYYY.csv` - AI-enriched historical data
- `competitions_YYYY.csv` - Competition metadata

Key columns: `Time`, `FullName`, `NAT`, `Rank`, `discipline_name`, `splits_json`, `lap_times_json`, `pacing_type`, `lap_variance`, `year`

## World Aquatics API

Base URL: `https://api.worldaquatics.com/fina`

Key endpoints:
- `/competitions` - List competitions by date range
- `/competitions/{id}/events` - Events in a competition
- `/events/{id}` - Detailed results with splits

No API key required. Rate limit: 1.5s between requests.

## Configuration

- **config.py** - API keys from `.env`, swimming constants, Saudi athlete IDs
- **.claude/** - Domain knowledge (agents, skills for swimming analysis)

## Team Saudi Branding

Use these colors for all dashboards:
- Primary Teal: `#007167`
- Gold Accent: `#a08e66`
- Dark Teal: `#005a51`
