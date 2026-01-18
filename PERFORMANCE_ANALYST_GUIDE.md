# Performance Analyst Quick Start Guide

A comprehensive guide for swimming performance analysts using the enhanced swimming data platform.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Daily Workflows](#daily-workflows)
3. [Analysis Tools](#analysis-tools)
4. [Dashboard Usage](#dashboard-usage)
5. [Automated Reports](#automated-reports)
6. [Best Practices](#best-practices)

## Initial Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file with your API keys:
```env
OPENROUTER_API_KEY=your_key_here
```
*Note: Only needed for AI-powered enrichment features*

### 3. Initial Data Collection
```bash
# Scrape the most recent year
python enhanced_swimming_scraper.py

# Or specify year range in the script
```

## Daily Workflows

### Workflow 1: Update Competition Database

**When**: After major competitions complete

```python
from enhanced_swimming_scraper import EnhancedSwimmingScraper

scraper = EnhancedSwimmingScraper(output_dir="data")

# Scrape latest year
competitions, results = scraper.scrape_year(2024)

print(f"Added {len(results)} new results from {len(competitions)} competitions")
```

### Workflow 2: Athlete Progress Check

**When**: Weekly for monitored athletes

```python
from performance_analyst_tools import AthleteProfiler, ProgressionTracker, ReportGenerator
import pandas as pd

# Load data
results = pd.read_csv('data/results_2024.csv')

# Create profile
profiler = AthleteProfiler(results)
profile = profiler.create_profile("Athlete Name")

# Generate report
report = ReportGenerator.athlete_profile_report(profile)
print(report)

# Save to file
ReportGenerator.save_report(report, "athlete_profile_2024.txt")
```

### Workflow 3: Pre-Competition Analysis

**When**: 1-2 weeks before major competition

```python
from performance_analyst_tools import CompetitionAnalyzer, AthleteProfiler
import pandas as pd

results = pd.read_csv('data/all_results_2020_2024.csv')

# Analyze target athletes
profiler = AthleteProfiler(results)

target_athletes = ["Athlete 1", "Athlete 2", "Athlete 3"]
comparison = profiler.compare_athletes(target_athletes, event="100m Freestyle")

print(comparison)

# Save comparison
comparison.to_csv('reports/competitor_analysis.csv', index=False)
```

### Workflow 4: Post-Competition Review

**When**: Immediately after competition ends

```python
from performance_analyst_tools import CompetitionAnalyzer, ReportGenerator
import pandas as pd

results = pd.read_csv('data/results_2024.csv')

analyzer = CompetitionAnalyzer(results)

# Full competition summary
summary = analyzer.competition_summary("World Championships")
report = ReportGenerator.competition_summary_report(summary)
print(report)

# Country performance
country_perf = analyzer.country_performance("KSA", "World Championships")
print(country_perf)
```

## Analysis Tools

### Tool 1: Athlete Profiling

```python
from performance_analyst_tools import AthleteProfiler
import pandas as pd

results = pd.read_csv('data/results_2024.csv')
profiler = AthleteProfiler(results)

# Single athlete profile
profile = profiler.create_profile("Athlete Name", include_splits=True)

# Key information
print(f"Country: {profile['country']}")
print(f"Total Races: {profile['total_races']}")
print(f"Medals: {profile['medals']}")
print(f"Best Times: {profile['best_times']}")
print(f"Pacing Preference: {profile['pacing_preference']}")
```

### Tool 2: Progression Tracking

```python
from performance_analyst_tools import ProgressionTracker
import pandas as pd

results = pd.read_csv('data/all_results_2020_2024.csv')
tracker = ProgressionTracker(results)

# Calculate progression
progression = tracker.calculate_progression("Athlete Name", "100m Freestyle")

# Display progression metrics
print(progression[['date_from', 'Time', 'personal_best', 'seconds_off_pb']])

# Identify breakthroughs
breakthroughs = tracker.identify_breakthroughs(progression, threshold=0.5)
print(f"Found {len(breakthroughs)} breakthrough performances")
```

### Tool 3: Competition Analysis

```python
from performance_analyst_tools import CompetitionAnalyzer
import pandas as pd

results = pd.read_csv('data/results_2024.csv')
analyzer = CompetitionAnalyzer(results)

# Full competition summary
summary = analyzer.competition_summary("Olympic Games")

print(f"Athletes: {summary['unique_athletes']}")
print(f"Countries: {summary['unique_countries']}")
print(f"Records: {summary['records_set']}")
print(f"Medal Table: {summary['medal_table']}")
```

### Tool 4: AI-Powered Enrichment

```python
from ai_enrichment import AIEnricher

enricher = AIEnricher()

# Classify competition
tier = enricher.classify_competition_tier(
    "World Championships",
    "World Aquatics Championships Doha 2024"
)
print(f"Competition Tier: {tier}")

# Analyze performance trend
trend = enricher.analyze_performance_trend(
    athlete_times=[48.2, 47.8, 47.5, 47.1],
    dates=['2023-01', '2023-06', '2023-12', '2024-02']
)
print(f"Trend: {trend['trend']}")
print(f"Analysis: {trend['analysis']}")

# Explain split strategy
analysis = enricher.explain_split_strategy(
    lap_times=[26.5, 27.2, 27.8, 28.1],
    distance=100,
    stroke="Freestyle"
)
print(f"Strategy Analysis: {analysis}")
```

### Tool 5: Split Time Analysis

```python
from enhanced_swimming_scraper import SplitTimeAnalyzer
import json
import pandas as pd

analyzer = SplitTimeAnalyzer()
results = pd.read_csv('data/results_2024.csv')

# Get race with splits
race = results[results['splits_json'].notna()].iloc[0]

# Parse splits
splits = json.loads(race['splits_json'])
lap_times = analyzer.calculate_lap_times(splits)

# Analyze pacing
pacing = analyzer.analyze_pacing(lap_times)

print(f"Pacing Type: {pacing['pacing_type']}")
print(f"First Half Avg: {pacing['first_half_avg']}s")
print(f"Second Half Avg: {pacing['second_half_avg']}s")
print(f"Lap Variance: {pacing['lap_variance']}")
```

## Dashboard Usage

### Launch Dashboard
```bash
# Install streamlit if needed
pip install streamlit plotly

# Run dashboard
streamlit run dashboard.py
```

### Dashboard Features

**1. Overview Page**
- Total statistics (competitions, athletes, countries)
- Event distribution charts
- Pacing strategy distribution
- Medal counts

**2. Athlete Profile Page**
- Search and select athlete
- View personal bests
- See career progression charts
- Identify breakthrough performances
- Recent form analysis

**3. Competition Analysis Page**
- Competition summary statistics
- Medal tables
- Event breakdown
- Participation analysis

**4. Split Time Analysis Page**
- Filter by event
- Compare top performers' splits
- View lap-by-lap breakdown
- Pacing strategy comparison

**5. Country Performance Page**
- National team statistics
- Medal counts
- Top performers
- Event coverage

**6. Rankings & Records Page**
- Filter records by type
- View best times by event
- Historical rankings

## Automated Reports

### Generate Athlete Report

```python
from performance_analyst_tools import AthleteProfiler, ReportGenerator
import pandas as pd

results = pd.read_csv('data/all_results_2020_2024.csv')

profiler = AthleteProfiler(results)
profile = profiler.create_profile("Athlete Name")

report = ReportGenerator.athlete_profile_report(profile)

# Save report
filename = ReportGenerator.save_report(
    report,
    "athlete_report_2024.txt",
    output_dir="reports"
)

print(f"Report saved to: {filename}")
```

### Generate Competition Report

```python
from performance_analyst_tools import CompetitionAnalyzer, ReportGenerator
import pandas as pd

results = pd.read_csv('data/results_2024.csv')

analyzer = CompetitionAnalyzer(results)
summary = analyzer.competition_summary("World Championships")

report = ReportGenerator.competition_summary_report(summary)

filename = ReportGenerator.save_report(
    report,
    "world_champs_2024.txt"
)

print(f"Report saved to: {filename}")
```

### Batch Process Multiple Athletes

```python
from performance_analyst_tools import AthleteProfiler, ReportGenerator
import pandas as pd

results = pd.read_csv('data/results_2024.csv')
profiler = AthleteProfiler(results)

# Saudi national team
saudi_athletes = results[results['NAT'] == 'KSA']['FullName'].unique()

for athlete in saudi_athletes:
    profile = profiler.create_profile(athlete)
    report = ReportGenerator.athlete_profile_report(profile)

    # Save individual report
    safe_name = athlete.replace(" ", "_")
    ReportGenerator.save_report(
        report,
        f"saudi_team_{safe_name}.txt",
        output_dir="reports/saudi_team"
    )

print(f"Generated {len(saudi_athletes)} athlete reports")
```

## Best Practices

### Data Management

1. **Regular Updates**: Scrape new data weekly during competition season
2. **Versioning**: Keep dated copies of datasets
3. **Validation**: Always check data quality after scraping
4. **Backup**: Maintain backups of historical data

```python
# Example: Save with timestamp
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d")
results.to_csv(f'data/backup/results_2024_{timestamp}.csv', index=False)
```

### Analysis Workflow

1. **Start Broad**: Overview → Specific athletes
2. **Use Comparisons**: Never analyze in isolation
3. **Context Matters**: Consider competition tier, conditions
4. **Trends Over Points**: Look for patterns, not single results
5. **Document Findings**: Save all reports and analyses

### Performance Review Cycle

**Daily**: Monitor for new competitions
**Weekly**: Update athlete progression charts
**Monthly**: Generate performance summaries
**Quarterly**: Comprehensive season reviews
**Annually**: Year-end reports and goal setting

### Split Time Analysis

1. **Event Appropriate**: Not all events have meaningful splits
   - 50m: Often no intermediate splits
   - 100m+: Focus on lap consistency
   - 400m+: Front-half vs back-half critical

2. **Pool Configuration**: LCM vs SCM affects strategy
   - LCM: Fewer turns, different pacing
   - SCM: Turn efficiency more important

3. **Pacing Expectations**:
   - **Sprints (50-100m)**: Positive split normal
   - **Middle (200-400m)**: Even split optimal
   - **Distance (800-1500m)**: Negative split advantage

### Using AI Features

1. **Supplement, Don't Replace**: AI insights are aids, not decisions
2. **Verify Critical Info**: Double-check important classifications
3. **Context Aware**: AI doesn't know competition-specific context
4. **Free Models**: Use free OpenRouter models for cost-effective analysis

## Quick Reference Commands

```bash
# Scrape latest data
python enhanced_swimming_scraper.py

# Test scraper
python test_scraper.py

# Run analysis examples
python quick_analysis.py

# Launch dashboard
streamlit run dashboard.py

# Test AI enrichment
python ai_enrichment.py
```

## Troubleshooting

### Issue: No data in dashboard
**Solution**: Run scraper first to collect data
```bash
python enhanced_swimming_scraper.py
```

### Issue: Missing split times
**Cause**: Not all events/competitions have split data
**Solution**: Focus on major competitions (Olympics, Worlds)

### Issue: AI enrichment errors
**Cause**: API key not configured
**Solution**: Add OPENROUTER_API_KEY to .env file

### Issue: Slow scraping
**Cause**: Rate limiting
**Solution**: Adjust RATE_LIMIT_DELAY in config.py

## Support & Resources

- **World Aquatics API**: https://api.worldaquatics.com
- **OpenRouter (Free Models)**: https://openrouter.ai
- **Documentation**: See README.md and agent/skill files in `.claude/`

## Summary

This platform provides comprehensive tools for swimming performance analysis:

✅ **Data Collection**: Automated scraping with split times
✅ **Analysis Tools**: Profiling, progression, competition analysis
✅ **Visualization**: Interactive dashboard
✅ **AI Insights**: Free model integration for enrichment
✅ **Reporting**: Automated report generation
✅ **Scalable**: Handles individual athletes to full national teams

Start with the dashboard for interactive exploration, then use Python tools for detailed analysis and automated reporting.
