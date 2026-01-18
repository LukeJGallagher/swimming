# Swimming Performance Analysis System

A comprehensive data scraping and analysis system for competitive swimming, featuring split time extraction and AI-powered insights.

## Features

- **Enhanced Data Scraping**: Extract detailed competition results from World Aquatics API
- **Split Time Analysis**: Parse and analyze lap-by-lap split times
- **Pacing Strategy Detection**: Identify even, negative, and positive split patterns
- **AI-Powered Enrichment**: Use free AI models for data classification and insights
- **Data Quality Validation**: Automatic anomaly detection and data quality checks
- **Comprehensive Coverage**: Competitions, events, results, rankings, and athlete histories

## Project Structure

```
Swimming/
├── .claude/
│   ├── agents/
│   │   ├── swimming_data_enrichment.md    # Swimming domain expert agent
│   │   └── web_scraper_agent.md           # Scraping specialist agent
│   └── skills/
│       ├── split_time_analyzer.md         # Split time analysis skill
│       └── world_aquatics_api.md          # API interaction skill
├── data/                                   # Output directory for scraped data
├── enhanced_swimming_scraper.py            # Main scraper with split extraction
├── ai_enrichment.py                        # AI-powered data enrichment
├── config.py                               # Configuration settings
├── requirements.txt                        # Python dependencies
├── .env                                    # API keys (not in git)
└── README.md                               # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

The `.env` file contains API keys for various services:

```env
OPENROUTER_API_KEY=your_key_here
BRAVE_API_KEY=your_key_here
FIRECRAWL_API_KEY=your_key_here
BRIGHTDATA_API_KEY=your_key_here
```

For AI enrichment, only `OPENROUTER_API_KEY` is required.

### 3. Free AI Models

The system uses **free models** from OpenRouter:
- `google/gemini-flash-1.5:free` (default)
- `meta-llama/llama-3.2-3b-instruct:free`
- `qwen/qwen-2-7b-instruct:free`
- `google/gemini-flash-1.5-8b:free`
- `meta-llama/llama-3.1-8b-instruct:free`

No cost, just need an OpenRouter account.

## Usage

### Basic Scraping

```python
from enhanced_swimming_scraper import EnhancedSwimmingScraper

# Initialize scraper
scraper = EnhancedSwimmingScraper(output_dir="data")

# Scrape a specific year
competitions, results = scraper.scrape_year(2024)

# Scrape a year range
scraper.scrape_year_range(2020, 2024)

# Scrape specific athlete
athlete_results = scraper.scrape_athlete(
    athlete_id=1640444,
    athlete_name="Your Athlete"
)
```

### Split Time Analysis

```python
from enhanced_swimming_scraper import SplitTimeAnalyzer

analyzer = SplitTimeAnalyzer()

# Parse splits from API response
splits = analyzer.parse_splits(result['Splits'])

# Calculate individual lap times
lap_times = analyzer.calculate_lap_times(splits)

# Analyze pacing strategy
pacing = analyzer.analyze_pacing(lap_times)
print(f"Pacing Type: {pacing['pacing_type']}")
print(f"Lap Variance: {pacing['lap_variance']}")
```

### AI Enrichment

```python
from ai_enrichment import AIEnricher

enricher = AIEnricher()

# Classify competition tier
tier = enricher.classify_competition_tier(
    competition_name="Paris 2024",
    official_name="Games of the XXXIII Olympiad"
)

# Analyze performance trend
trend = enricher.analyze_performance_trend(
    athlete_times=[48.2, 47.8, 47.5, 47.1],
    dates=['2023-01-15', '2023-06-20', '2023-12-10', '2024-02-25']
)

# Get split strategy explanation
analysis = enricher.explain_split_strategy(
    lap_times=[26.5, 27.2, 27.8, 28.1],
    distance=100,
    stroke="Freestyle"
)
```

### Data Quality Checks

```python
from ai_enrichment import DataQualityChecker

checker = DataQualityChecker()

# Validate a time
time_check = checker.validate_time("47.52")
print(f"Valid: {time_check['valid']}")

# Flag anomalies in dataset
flagged_df = checker.flag_anomalies(results_df)
anomalies = flagged_df[flagged_df['data_quality_flags'].notna()]
```

## Data Schema

### Results Data

Key fields in the results DataFrame:

| Field | Description |
|-------|-------------|
| `Time` | Final race time |
| `FullName` | Athlete full name |
| `NAT` / `NATName` | Country code and name |
| `Rank` | Overall rank |
| `HeatRank` | Rank within heat |
| `discipline_name` | Event name (e.g., "Men 100m Freestyle") |
| `gender` | M, W, or X (mixed) |
| `splits_json` | Raw split times as JSON |
| `lap_times_json` | Calculated lap times as JSON |
| `pacing_type` | Even, Negative Split, or Positive Split |
| `lap_variance` | Standard deviation of lap times |
| `fastest_lap` | Fastest lap time in seconds |
| `slowest_lap` | Slowest lap time in seconds |
| `RT` | Reaction time |
| `MedalTag` | G, S, B for medals |

## World Aquatics API Endpoints

The scraper uses these main endpoints:

1. **Competitions**: List competitions by date range
   ```
   GET /fina/competitions?pageSize=100&venueDateFrom=...&venueDateTo=...
   ```

2. **Competition Events**: Get events in a competition
   ```
   GET /fina/competitions/{competitionId}/events
   ```

3. **Event Results**: Get detailed results with splits
   ```
   GET /fina/events/{eventId}
   ```

4. **Rankings**: World rankings by event
   ```
   GET /fina/rankings/swimming?gender=...&distance=...&stroke=...
   ```

5. **Athlete Results**: Complete athlete history
   ```
   GET /fina/athletes/{athleteId}/results
   ```

## Swimming Context

### Event Types
- **Sprint**: 50m, 100m
- **Middle Distance**: 200m, 400m
- **Distance**: 800m, 1500m
- **Relays**: 4x50m, 4x100m, 4x200m, Medley

### Pool Configurations
- **LCM**: Long Course (50m) - Olympic standard
- **SCM**: Short Course (25m) - More turns

### Stroke Types
- Freestyle
- Backstroke
- Breaststroke
- Butterfly
- Individual Medley (IM)

### Pacing Strategies
- **Even Split**: Consistent lap times (optimal)
- **Negative Split**: Faster second half (strong endurance)
- **Positive Split**: Slower second half (fatigue/sprint strategy)

## Split Time Analysis

Split times are crucial for understanding race performance:

- **Elite swimmers** maintain split variance within 1-2 seconds
- Analysis reveals strengths in starts, turns, finishes
- Critical for training focus and race strategy
- Enables comparison across races and athletes

Example split analysis output:
```python
{
    'pacing_type': 'Even',
    'first_half_avg': 26.8,
    'second_half_avg': 27.2,
    'split_difference': 0.4,
    'fastest_lap': 26.5,
    'slowest_lap': 28.1,
    'lap_variance': 0.652
}
```

## Performance Tips

1. **Rate Limiting**: Default 1.5s delay between requests - adjust in `config.py`
2. **Incremental Scraping**: Scrape by year to allow resumption
3. **Data Storage**: Results saved to CSV and optional Parquet for better compression
4. **Error Handling**: Automatic retry with exponential backoff
5. **Logging**: All activity logged to `swimming_scraper.log`

## Agent & Skills System

The `.claude` directory contains:

### Agents
- **Swimming Data Enrichment Agent**: Domain expert for analysis
- **Web Scraper Agent**: Specialized in efficient data extraction

### Skills
- **Split Time Analyzer**: Parse and analyze split times
- **World Aquatics API**: Complete API interaction guide

These are used by Claude Code for enhanced assistance.

## Example Workflows

### 1. Update Database with Latest Results
```python
scraper = EnhancedSwimmingScraper()
comps, results = scraper.scrape_year(2024)
# Data automatically saved to data/results_2024.csv
```

### 2. Analyze Athlete Progress
```python
# Get athlete's history
athlete_df = scraper.scrape_athlete(athlete_id=1640444)

# Filter for specific event
event_results = athlete_df[
    athlete_df['DisciplineName'].str.contains('100m Freestyle')
].sort_values('dateFrom')

# Analyze trend
enricher = AIEnricher()
times = event_results['Time'].apply(
    lambda x: SplitTimeAnalyzer.time_to_seconds(x)
).tolist()
dates = event_results['dateFrom'].tolist()

trend = enricher.analyze_performance_trend(times, dates)
print(trend['analysis'])
```

### 3. Compare Competition Levels
```python
# Scrape multiple years
all_comps = []
for year in range(2020, 2025):
    comps, _ = scraper.scrape_year(year)
    all_comps.append(comps)

combined = pd.concat(all_comps)

# Enrich with AI classification
enricher = AIEnricher()
enriched = enricher.enrich_competition_batch(combined)

# Analyze by tier
tier_summary = enriched.groupby('competition_tier').size()
print(tier_summary)
```

## Troubleshooting

### No data returned
- Check internet connection
- Verify World Aquatics API is accessible
- Check date ranges (API may have limited historical data)

### Split times missing
- Not all events have split times in the API
- Older competitions may not have split data
- Relay events often don't include individual splits

### AI enrichment not working
- Verify OPENROUTER_API_KEY in .env
- Check OpenRouter account has access to free models
- Review logs for API error messages

## Future Enhancements

- [ ] Real-time competition monitoring
- [ ] Predictive performance modeling
- [ ] Automated competitor analysis
- [ ] Interactive visualization dashboard
- [ ] Integration with training data
- [ ] Video analysis integration

## License

This tool is for research and analysis purposes. Respect World Aquatics API usage terms.

## Contact

For issues and questions, refer to the logging output and API documentation.
