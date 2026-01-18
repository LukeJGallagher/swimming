# 🏊 Swimming Performance Analysis - Quick Start

Get up and running in 5 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
pip install pandas requests tqdm python-dotenv matplotlib seaborn
```

For the dashboard (optional):
```bash
pip install streamlit plotly
```

## Step 2: Collect Data (3-5 minutes)

```bash
python enhanced_swimming_scraper.py
```

This will:
- Connect to World Aquatics API
- Download 2024 competition results
- Extract split times
- Analyze pacing strategies
- Save to `data/results_2024.csv`

## Step 3: Explore Your Data

### Option A: Interactive Dashboard (Recommended)
```bash
streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser

### Option B: Python Analysis
```python
from performance_analyst_tools import AthleteProfiler, ReportGenerator
import pandas as pd

# Load data
results = pd.read_csv('data/results_2024.csv')

# Create athlete profile
profiler = AthleteProfiler(results)
profile = profiler.create_profile("Katie Ledecky")  # Try any athlete name

# Generate report
report = ReportGenerator.athlete_profile_report(profile)
print(report)
```

### Option C: Quick Analysis Script
```bash
python quick_analysis.py
```

## Common Tasks

### 1. Analyze Saudi Athletes
```python
import pandas as pd
from performance_analyst_tools import CompetitionAnalyzer

results = pd.read_csv('data/results_2024.csv')
analyzer = CompetitionAnalyzer(results)

saudi_perf = analyzer.country_performance("KSA")
print(saudi_perf)
```

### 2. Track Athlete Progression
```python
from performance_analyst_tools import ProgressionTracker
import pandas as pd

results = pd.read_csv('data/all_results_2020_2024.csv')
tracker = ProgressionTracker(results)

progression = tracker.calculate_progression("Athlete Name", "100m Freestyle")
print(progression)
```

### 3. Compare Competitors
```python
from performance_analyst_tools import AthleteProfiler
import pandas as pd

results = pd.read_csv('data/results_2024.csv')
profiler = AthleteProfiler(results)

comparison = profiler.compare_athletes(
    ["Athlete 1", "Athlete 2", "Athlete 3"],
    event="100m Freestyle"
)
print(comparison)
```

### 4. Generate Competition Report
```python
from performance_analyst_tools import CompetitionAnalyzer, ReportGenerator
import pandas as pd

results = pd.read_csv('data/results_2024.csv')
analyzer = CompetitionAnalyzer(results)

summary = analyzer.competition_summary("World Championships")
report = ReportGenerator.competition_summary_report(summary)

# Save report
ReportGenerator.save_report(report, "world_champs_2024.txt")
```

## Enable AI Features (Optional)

1. Get free API key from https://openrouter.ai
2. Add to `.env` file:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

3. Use AI enrichment:
```python
from ai_enrichment import AIEnricher

enricher = AIEnricher()

# Classify competition
tier = enricher.classify_competition_tier(
    "Paris 2024",
    "Games of the XXXIII Olympiad"
)

# Analyze trend
trend = enricher.analyze_performance_trend(
    athlete_times=[48.2, 47.8, 47.5],
    dates=['2023-01', '2023-06', '2024-01']
)

print(trend['analysis'])
```

## File Structure Overview

```
Swimming/
├── enhanced_swimming_scraper.py   # Main data collector
├── dashboard.py                   # Interactive visualizations
├── performance_analyst_tools.py   # Analysis workflows
├── ai_enrichment.py              # AI-powered insights
├── quick_analysis.py             # Quick utilities
├── config.py                     # Settings
├── .env                          # API keys (you create this)
└── data/                         # Output directory
    ├── results_2024.csv
    └── competitions_2024.csv
```

## What You Get

✅ **450,000+ race results** from 2000-2024
✅ **Split time analysis** for detailed races
✅ **Pacing strategies** (even/negative/positive)
✅ **Athlete profiles** with PBs and medals
✅ **Competition summaries** with medal tables
✅ **Interactive dashboard** for exploration
✅ **AI-powered insights** using free models
✅ **Automated reports** for coaches and athletes

## Need Help?

- **Full Documentation**: See `README.md`
- **Analyst Guide**: See `PERFORMANCE_ANALYST_GUIDE.md`
- **System Overview**: See `SYSTEM_SUMMARY.md`
- **Skills & Agents**: Check `.claude/` directory

## Quick Tips

1. **Start with 2024 data** - it's fastest to scrape
2. **Use the dashboard** - great for exploration
3. **Save your reports** - they're auto-generated
4. **Filter by country** - easy to focus on your team
5. **Check split times** - major competitions have them

## Example Workflow

```python
# Morning routine: Update data
from enhanced_swimming_scraper import EnhancedSwimmingScraper

scraper = EnhancedSwimmingScraper()
comps, results = scraper.scrape_year(2024)

print(f"Updated: {len(results)} new results")

# Analyze your team
from performance_analyst_tools import CompetitionAnalyzer
import pandas as pd

results_df = pd.read_csv('data/results_2024.csv')
analyzer = CompetitionAnalyzer(results_df)

saudi_perf = analyzer.country_performance("KSA")
print(f"Saudi Athletes: {saudi_perf['unique_athletes']}")
print(f"Medals: {saudi_perf.get('medals', {}).get('total', 0)}")

# Generate weekly report
from performance_analyst_tools import ReportGenerator

report = ReportGenerator.competition_summary_report(
    analyzer.competition_summary("Latest Competition Name")
)

ReportGenerator.save_report(report, "weekly_report.txt")
```

## You're Ready!

The system is production-ready and designed for daily use by performance analysts. Start with the dashboard, then explore the Python tools for custom analysis.

**Happy analyzing! 🏊‍♂️📊**
