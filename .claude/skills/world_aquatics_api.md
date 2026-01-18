# World Aquatics API Skill

Complete guide to interacting with the World Aquatics (FINA) API for swimming data.

## Base URL
```
https://api.worldaquatics.com/fina/
```

## Authentication
No API key required - public API

## Rate Limiting Best Practices
- Wait 1-2 seconds between requests
- Implement retry logic with exponential backoff
- Respect server responses (503, 429 errors)

## Main Endpoints

### 1. Competitions
**Endpoint**: `/competitions`

**Purpose**: Get list of swimming competitions by date range

**Parameters**:
- `pageSize`: Number of results (max 100)
- `venueDateFrom`: Start date (YYYY-MM-DDTHH:mm:ss+00:00)
- `venueDateTo`: End date (YYYY-MM-DDTHH:mm:ss+00:00)
- `disciplines`: Sport code (SW for Swimming)
- `group`: Organization (FINA, WORLD_MASTERS, etc.)
- `sort`: Sorting (dateFrom,desc or dateFrom,asc)

**Example**:
```python
import requests
import pandas as pd

url = "https://api.worldaquatics.com/fina/competitions"
params = {
    'pageSize': 100,
    'venueDateFrom': '2024-01-01T00:00:00+00:00',
    'venueDateTo': '2024-12-31T23:59:59+00:00',
    'disciplines': 'SW',
    'group': 'FINA',
    'sort': 'dateFrom,desc'
}

response = requests.get(url, params=params)
competitions = pd.DataFrame(response.json()['content'])
```

**Response Fields**:
- `id`: Competition ID (use for next API calls)
- `name`: Short name
- `officialName`: Full official name
- `dateFrom`, `dateTo`: Competition dates
- `location`: Dict with city, country, coordinates

### 2. Competition Events
**Endpoint**: `/competitions/{competitionId}/events`

**Purpose**: Get all events (races) within a competition

**Example**:
```python
comp_id = 3432
url = f"https://api.worldaquatics.com/fina/competitions/{comp_id}/events"
response = requests.get(url)

data = response.json()
for sport in data['Sports']:
    if sport['Name'] == 'Swimming':
        disciplines = pd.DataFrame(sport['DisciplineList'])
        # disciplines contains event IDs and details
```

**Response Structure**:
- `Sports`: Array of sports
  - `Name`: Sport name (filter for 'Swimming')
  - `DisciplineList`: Array of events
    - `Id`: Event ID (for results endpoint)
    - `DisciplineName`: e.g., "Men 100m Freestyle"
    - `Gender`: M, W, or X (mixed)

### 3. Event Results
**Endpoint**: `/events/{eventId}`

**Purpose**: Get detailed results including splits for specific event

**Example**:
```python
event_id = 12345
url = f"https://api.worldaquatics.com/fina/events/{event_id}"
response = requests.get(url)

data = response.json()
heats = pd.DataFrame(data['Heats'])

all_results = []
for idx, heat in heats.iterrows():
    results = pd.DataFrame(heat['Results'])
    results['Heat_Name'] = heat['Name']  # Finals, Semi-Finals, Heats
    all_results.append(results)

results_df = pd.concat(all_results)
```

**Key Result Fields**:
- `Time`: Final time (string format MM:SS.ss or SS.ss)
- `Splits`: **IMPORTANT** Array of split times
- `Rank`: Overall rank
- `HeatRank`: Rank within heat
- `FullName`, `FirstName`, `LastName`: Athlete name
- `NAT`, `NATName`: Country code and name
- `PersonId`: Athlete ID
- `Lane`: Lane number
- `RT`: Reaction time
- `MedalTag`: G, S, B for medals
- `RecordType`: WR, OR, NR, etc.

**Splits Structure**:
```python
# Example splits array
splits = [
    {'distance': 50, 'time': '24.53'},
    {'distance': 100, 'time': '51.22'},
    {'distance': 150, 'time': '1:18.45'},
    {'distance': 200, 'time': '1:45.67'}
]
```

### 4. Rankings
**Endpoint**: `/rankings/swimming`

**Purpose**: Get world rankings for specific events

**Parameters**:
- `gender`: M, W, X
- `distance`: 50, 100, 200, 400, 800, 1500
- `stroke`: FREESTYLE, BACKSTROKE, BREASTSTROKE, BUTTERFLY, FREESTYLE_RELAY, MEDLEY_RELAY
- `poolConfiguration`: LCM, SCM
- `year`: YYYY
- `timesMode`: BEST_TIMES or ALL_TIMES
- `pageSize`: Max 200

**Example**:
```python
url = "https://api.worldaquatics.com/fina/rankings/swimming"
params = {
    'gender': 'M',
    'distance': '100',
    'stroke': 'FREESTYLE',
    'poolConfiguration': 'LCM',
    'year': '2024',
    'timesMode': 'BEST_TIMES',
    'pageSize': 200
}

response = requests.get(url, params=params)
rankings = pd.DataFrame(response.json()['swimmingWorldRankings'])
```

### 5. Athlete Results
**Endpoint**: `/athletes/{athleteId}/results`

**Purpose**: Get complete competition history for an athlete

**Example**:
```python
athlete_id = 1640444
url = f"https://api.worldaquatics.com/fina/athletes/{athlete_id}/results"
response = requests.get(url)

athlete_results = pd.DataFrame(response.json()['Results'])
athlete_name = response.json()['FullName']
```

## Error Handling

```python
import time
from requests.exceptions import RequestException

def safe_api_call(url, params=None, max_retries=3):
    """Make API call with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Request failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
        except json.JSONDecodeError:
            print(f"Invalid JSON response from {url}")
            return None

    return None
```

## Data Quality Checks

```python
def validate_result(result):
    """Check if result data is valid"""
    checks = {
        'has_time': 'Time' in result and result['Time'] is not None,
        'has_athlete': 'FullName' in result and result['FullName'] is not None,
        'valid_rank': 'Rank' in result and (pd.isna(result['Rank']) or result['Rank'] > 0),
    }
    return all(checks.values()), checks
```

## Complete Scraping Workflow

```python
def scrape_year_competitions(year, disciplines='SW'):
    """Scrape all swimming competitions for a given year"""
    # 1. Get competitions
    url = "https://api.worldaquatics.com/fina/competitions"
    params = {
        'pageSize': 100,
        'venueDateFrom': f'{year}-01-01T00:00:00+00:00',
        'venueDateTo': f'{year}-12-31T23:59:59+00:00',
        'disciplines': disciplines,
        'group': 'FINA',
        'sort': 'dateFrom,desc'
    }

    comps_data = safe_api_call(url, params)
    competitions = pd.DataFrame(comps_data['content'])

    # 2. For each competition, get events and results
    all_results = []

    for comp_id in competitions['id']:
        time.sleep(1)  # Rate limiting

        # Get events
        events_url = f"https://api.worldaquatics.com/fina/competitions/{comp_id}/events"
        events_data = safe_api_call(events_url)

        if not events_data:
            continue

        # Extract swimming disciplines
        for sport in events_data.get('Sports', []):
            if sport['Name'] == 'Swimming':
                for discipline in sport['DisciplineList']:
                    time.sleep(1)

                    # Get results for this event
                    event_id = discipline['Id']
                    results_url = f"https://api.worldaquatics.com/fina/events/{event_id}"
                    results_data = safe_api_call(results_url)

                    if not results_data:
                        continue

                    # Process heats
                    for heat in results_data.get('Heats', []):
                        for result in heat.get('Results', []):
                            result['competition_id'] = comp_id
                            result['event_id'] = event_id
                            result['heat_name'] = heat['Name']
                            all_results.append(result)

    return pd.DataFrame(all_results)
```
