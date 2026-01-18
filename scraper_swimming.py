"""
Swimming Data Scraper with Azure Blob Storage
Scrapes World Aquatics API and saves to Azure Blob as Parquet

Usage:
    python scraper_swimming.py              # Scrape current year
    python scraper_swimming.py --year 2024  # Scrape specific year
    python scraper_swimming.py --migrate    # Migrate local CSVs to Azure
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import blob storage module
from blob_storage import (
    save_results, create_backup, load_results,
    migrate_csv_to_parquet, get_connection_mode, _use_azure
)

# Configuration
BASE_URL = "https://api.worldaquatics.com/fina"
REQUEST_DELAY = 1.5  # Seconds between API requests


class WorldAquaticsAPI:
    """World Aquatics API client with rate limiting."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwimmingAnalytics/1.0',
            'Accept': 'application/json'
        })
        self.last_request = 0

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request = time.time()

    def get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to the API."""
        self._rate_limit()

        url = f"{BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API error: {e}")
            return None

    def get_competitions(self, year: int) -> List[dict]:
        """Get all swimming competitions for a year."""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        data = self.get("/competitions", {
            "startDate": start_date,
            "endDate": end_date,
            "discipline": "SW"
        })

        if data and 'Competitions' in data:
            return data['Competitions']
        return []

    def get_competition_events(self, comp_id: int) -> List[dict]:
        """Get events for a competition."""
        data = self.get(f"/competitions/{comp_id}/events")
        return data.get('Events', []) if data else []

    def get_event_results(self, event_id: int) -> dict:
        """Get detailed results for an event."""
        return self.get(f"/events/{event_id}") or {}


class SplitTimeAnalyzer:
    """Extract and analyze split times from results."""

    @staticmethod
    def extract_splits(result: dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract split times from a result entry."""
        splits = result.get('Splits', [])

        if not splits:
            return None, None

        try:
            splits_json = json.dumps(splits)

            # Calculate lap times
            lap_times = []
            prev_time = 0

            for split in splits:
                if 'Time' in split:
                    time_str = split['Time']
                    seconds = SplitTimeAnalyzer._time_to_seconds(time_str)
                    if seconds:
                        lap = seconds - prev_time
                        lap_times.append(round(lap, 2))
                        prev_time = seconds

            lap_times_json = json.dumps(lap_times) if lap_times else None

            return splits_json, lap_times_json
        except:
            return None, None

    @staticmethod
    def _time_to_seconds(time_str: str) -> Optional[float]:
        """Convert time string to seconds."""
        if not time_str:
            return None
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        except:
            return None

    @staticmethod
    def classify_pacing(lap_times: List[float]) -> Optional[str]:
        """Classify the pacing strategy."""
        if not lap_times or len(lap_times) < 2:
            return None

        first = lap_times[0]
        last = lap_times[-1]
        middle = lap_times[1:-1] if len(lap_times) > 2 else []

        avg_middle = sum(middle) / len(middle) if middle else (first + last) / 2

        # Classification based on research
        if first < avg_middle < last:
            return "Positive"  # Slowing down
        elif first > avg_middle > last:
            return "Negative"  # Speeding up
        elif first < avg_middle and last < avg_middle:
            return "U-shape"  # Fast start and finish
        elif first > avg_middle and last > avg_middle:
            return "Inverted-J"  # Slow start and finish
        else:
            return "Even"


class SwimmingScraper:
    """Main scraper class."""

    def __init__(self):
        self.api = WorldAquaticsAPI()
        self.analyzer = SplitTimeAnalyzer()

    def scrape_year(self, year: int) -> pd.DataFrame:
        """Scrape all results for a year."""
        print(f"\n{'='*60}")
        print(f"SCRAPING YEAR {year}")
        print(f"{'='*60}")

        # Get competitions
        competitions = self.api.get_competitions(year)
        print(f"Found {len(competitions)} competitions")

        all_results = []

        for i, comp in enumerate(competitions):
            comp_id = comp.get('Id')
            comp_name = comp.get('Name', 'Unknown')

            print(f"\n[{i+1}/{len(competitions)}] {comp_name[:50]}...")

            # Get events for this competition
            events = self.api.get_competition_events(comp_id)

            for event in events:
                event_id = event.get('Id')
                event_name = event.get('Name', '')
                gender = event.get('Gender', '')

                # Get results
                event_data = self.api.get_event_results(event_id)

                if not event_data:
                    continue

                # Process heats/finals
                for heat_type in ['Heats', 'SemiFinals', 'Finals']:
                    heats = event_data.get(heat_type, [])

                    for heat in heats:
                        heat_name = heat.get('Name', heat_type)
                        results = heat.get('Results', [])

                        for result in results:
                            # Extract splits
                            splits_json, lap_times_json = self.analyzer.extract_splits(result)

                            # Classify pacing
                            pacing_type = None
                            if lap_times_json:
                                try:
                                    lap_times = json.loads(lap_times_json)
                                    pacing_type = self.analyzer.classify_pacing(lap_times)
                                except:
                                    pass

                            row = {
                                'year': year,
                                'competition_id': comp_id,
                                'competition_name': comp_name,
                                'event_id': event_id,
                                'discipline_name': event_name,
                                'gender': gender,
                                'heat_category': heat_name,
                                'FullName': result.get('FullName'),
                                'FirstName': result.get('FirstName'),
                                'LastName': result.get('LastName'),
                                'NAT': result.get('NAT'),
                                'NATName': result.get('NATName'),
                                'PersonId': result.get('PersonId'),
                                'BiographyId': result.get('BiographyId'),
                                'AthleteResultAge': result.get('AthleteResultAge'),
                                'ResultId': result.get('ResultId'),
                                'Lane': result.get('Lane'),
                                'HeatRank': result.get('HeatRank'),
                                'Rank': result.get('Rank'),
                                'Time': result.get('Time'),
                                'RT': result.get('RT'),
                                'TimeBehind': result.get('TimeBehind'),
                                'Points': result.get('Points'),
                                'MedalTag': result.get('MedalTag'),
                                'Qualified': result.get('Qualified'),
                                'RecordType': result.get('RecordType'),
                                'splits_json': splits_json,
                                'lap_times_json': lap_times_json,
                                'pacing_type': pacing_type,
                            }

                            all_results.append(row)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(competitions)} competitions, {len(all_results):,} results")

        df = pd.DataFrame(all_results)
        print(f"\nYear {year} complete: {len(df):,} results")

        return df

    def run(self, year: int = None, append: bool = True):
        """Run the scraper."""
        if year is None:
            year = datetime.now().year

        print(f"\nStorage mode: {get_connection_mode()}")

        # Create backup if using Azure
        if _use_azure():
            print("Creating backup before scraping...")
            create_backup()

        # Scrape data
        df = self.scrape_year(year)

        if df.empty:
            print("No data scraped")
            return

        # Save to Azure Blob or local
        print(f"\nSaving {len(df):,} results...")
        success = save_results(df, append=append)

        if success:
            print(f"Data saved successfully")
        else:
            # Fallback to local CSV
            filename = f"Results_{year}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved locally to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Swimming Data Scraper')
    parser.add_argument('--year', type=int, help='Year to scrape (default: current year)')
    parser.add_argument('--migrate', action='store_true', help='Migrate local CSVs to Azure Blob')
    parser.add_argument('--test', action='store_true', help='Test Azure connection only')

    args = parser.parse_args()

    if args.test:
        from blob_storage import test_connection
        result = test_connection()
        print(json.dumps(result, indent=2))
        return

    if args.migrate:
        migrate_csv_to_parquet()
        return

    scraper = SwimmingScraper()
    scraper.run(year=args.year)


if __name__ == "__main__":
    main()
