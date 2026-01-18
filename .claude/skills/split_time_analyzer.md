# Split Time Analyzer Skill

Parse and analyze swimming split times from World Aquatics data.

## Purpose
Extract valuable insights from split time data to understand race pacing, identify strengths/weaknesses, and optimize training strategies.

## Usage

This skill provides functions to:
1. Parse split time arrays from API responses
2. Calculate lap times from cumulative splits
3. Analyze pacing patterns
4. Compare splits across races/athletes
5. Visualize split time progressions

## Swimming Split Time Fundamentals

### What are Split Times?
- Cumulative times at each lap/turn in a race
- Example for 200m race: [50m: 25.3s, 100m: 52.1s, 150m: 1:19.5s, 200m: 1:46.8s]
- Lap times calculated by subtracting consecutive splits

### Pacing Strategies
- **Even Split**: Consistent lap times throughout (optimal for most events)
- **Negative Split**: Faster second half (demonstrates strong endurance)
- **Positive Split**: Slower second half (common in sprints, indicates fatigue)

### Analysis Metrics
- Split variance: Standard deviation of lap times
- Fastest/slowest lap identification
- Front-half vs back-half comparison
- Turn efficiency (based on split differentials)

## Code Examples

### Parse Split Times from API Response
```python
import json
import pandas as pd

def parse_splits(splits_str):
    """
    Parse splits from API response string/JSON
    Returns list of split dictionaries with distance and time
    """
    if pd.isna(splits_str) or splits_str == '[]':
        return []

    try:
        splits = json.loads(splits_str) if isinstance(splits_str, str) else splits_str
        return splits
    except:
        return []

def calculate_lap_times(splits):
    """
    Calculate individual lap times from cumulative splits
    """
    if not splits or len(splits) < 2:
        return []

    lap_times = []
    for i in range(len(splits)):
        if i == 0:
            lap_times.append({
                'lap_number': 1,
                'distance': splits[i].get('distance', 0),
                'cumulative_time': splits[i].get('time'),
                'lap_time': splits[i].get('time')
            })
        else:
            prev_time = time_to_seconds(splits[i-1].get('time', '0'))
            curr_time = time_to_seconds(splits[i].get('time', '0'))
            lap_time = curr_time - prev_time

            lap_times.append({
                'lap_number': i + 1,
                'distance': splits[i].get('distance', 0),
                'cumulative_time': splits[i].get('time'),
                'lap_time': seconds_to_time(lap_time)
            })

    return lap_times

def time_to_seconds(time_str):
    """Convert time string (MM:SS.ss or SS.ss) to seconds"""
    if not time_str or pd.isna(time_str):
        return 0.0

    parts = str(time_str).split(':')
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return float(parts[0])

def seconds_to_time(seconds):
    """Convert seconds to time string"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}:{secs:05.2f}"
    return f"{secs:.2f}"
```

### Analyze Pacing Strategy
```python
def analyze_pacing(lap_times):
    """
    Analyze pacing strategy from lap times
    """
    if not lap_times or len(lap_times) < 2:
        return None

    lap_seconds = [time_to_seconds(lt['lap_time']) for lt in lap_times]

    first_half_avg = sum(lap_seconds[:len(lap_seconds)//2]) / (len(lap_seconds)//2)
    second_half_avg = sum(lap_seconds[len(lap_seconds)//2:]) / (len(lap_seconds) - len(lap_seconds)//2)

    split_diff = second_half_avg - first_half_avg

    if abs(split_diff) < 0.5:
        pacing_type = "Even"
    elif split_diff < 0:
        pacing_type = "Negative Split"
    else:
        pacing_type = "Positive Split"

    return {
        'pacing_type': pacing_type,
        'first_half_avg': first_half_avg,
        'second_half_avg': second_half_avg,
        'split_difference': split_diff,
        'fastest_lap': min(lap_seconds),
        'slowest_lap': max(lap_seconds),
        'lap_variance': pd.Series(lap_seconds).std()
    }
```

## Integration with Scraper

When scraping results, always extract and parse split times:

```python
# In your scraper
for result in heat_results:
    result_data = {
        'time': result.get('Time'),
        'athlete': result.get('FullName'),
        # ... other fields ...
    }

    # Extract and parse splits
    splits = result.get('Splits', [])
    result_data['splits_raw'] = json.dumps(splits)

    lap_times = calculate_lap_times(splits)
    pacing_analysis = analyze_pacing(lap_times)

    result_data['lap_times'] = json.dumps(lap_times)
    result_data['pacing_type'] = pacing_analysis.get('pacing_type') if pacing_analysis else None
    result_data['lap_variance'] = pacing_analysis.get('lap_variance') if pacing_analysis else None
```

## Expected Output

Enriched data should include:
- Original split times array
- Calculated lap times
- Pacing classification
- Statistical metrics (variance, fastest/slowest lap)
- Performance flags (even pacing, fade, strong finish)
