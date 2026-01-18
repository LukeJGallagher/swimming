"""
Incremental Data Sync for Swimming Data
Fetches only new competitions since last sync
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from enhanced_swimming_scraper import EnhancedSwimmingScraper, WorldAquaticsAPI, SplitTimeAnalyzer
from process_splits import process_file
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IncrementalSync:
    """Incremental sync manager for swimming data"""

    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.sync_file = self.data_dir / ".sync_state.json"
        self.api = WorldAquaticsAPI()
        self.scraper = EnhancedSwimmingScraper(output_dir=str(self.data_dir / "data"))

    def load_sync_state(self) -> dict:
        """Load the last sync state"""
        if self.sync_file.exists():
            with open(self.sync_file, 'r') as f:
                return json.load(f)
        return {
            'last_sync': None,
            'synced_competitions': [],
            'last_year': None
        }

    def save_sync_state(self, state: dict):
        """Save sync state"""
        with open(self.sync_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def get_new_competitions(self, since_date: str = None) -> pd.DataFrame:
        """Get competitions since a specific date"""
        current_year = datetime.now().year

        if since_date:
            # Parse the date and get competitions from that year onwards
            since = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
            start_year = since.year
        else:
            # Default to current year only
            start_year = current_year

        all_comps = []

        for year in range(start_year, current_year + 1):
            logger.info(f"Fetching competitions for {year}")
            comps = self.api.get_competitions(year)
            if comps is not None and not comps.empty:
                comps['year'] = year
                all_comps.append(comps)

        if not all_comps:
            return pd.DataFrame()

        combined = pd.concat(all_comps, ignore_index=True)

        # Filter to only new competitions if we have a since_date
        if since_date:
            combined['dateFrom_parsed'] = pd.to_datetime(combined['dateFrom'], errors='coerce')
            combined = combined[combined['dateFrom_parsed'] > since_date]
            combined = combined.drop('dateFrom_parsed', axis=1)

        return combined

    def sync_competition(self, comp_id: int, comp_name: str) -> pd.DataFrame:
        """Sync a single competition's results"""
        logger.info(f"Syncing competition: {comp_name} (ID: {comp_id})")

        events = self.api.get_competition_events(comp_id)
        if not events:
            logger.warning(f"No events found for {comp_name}")
            return pd.DataFrame()

        all_results = []
        analyzer = SplitTimeAnalyzer()

        for event in events:
            event_id = event.get('Id')
            discipline_name = event.get('DisciplineName', 'Unknown')

            results = self.api.get_event_results(event_id)
            if not results:
                continue

            for result in results:
                result['discipline_name'] = discipline_name
                result['competition_id'] = comp_id
                result['competition_name'] = comp_name

                # Extract and analyze splits
                splits_raw = result.get('Splits', [])
                splits = analyzer.parse_splits(splits_raw)

                if splits:
                    result['splits_json'] = json.dumps(splits)
                    lap_times = analyzer.calculate_lap_times(splits)
                    result['lap_times_json'] = json.dumps(lap_times)

                    pacing = analyzer.analyze_pacing(lap_times)
                    if pacing:
                        result['pacing_type'] = pacing['pacing_type']
                        result['first_half_avg'] = pacing['first_half_avg']
                        result['second_half_avg'] = pacing['second_half_avg']
                        result['split_difference'] = pacing['split_difference']
                        result['fastest_lap'] = pacing['fastest_lap']
                        result['slowest_lap'] = pacing['slowest_lap']
                        result['lap_variance'] = pacing['lap_variance']

                all_results.append(result)

        return pd.DataFrame(all_results)

    def run_sync(self, full_sync: bool = False, year: int = None):
        """Run incremental sync"""
        state = self.load_sync_state()

        if full_sync:
            logger.info("Running full sync...")
            state = {'last_sync': None, 'synced_competitions': [], 'last_year': None}
        elif year:
            logger.info(f"Syncing year {year}...")
            state['last_sync'] = None  # Force fetch for specific year

        # Determine what to fetch
        if year:
            comps = self.api.get_competitions(year)
            if comps is not None:
                comps['year'] = year
            else:
                comps = pd.DataFrame()
        else:
            since = state.get('last_sync')
            comps = self.get_new_competitions(since)

        if comps.empty:
            logger.info("No new competitions found")
            return

        # Filter out already synced
        synced_ids = set(state.get('synced_competitions', []))
        new_comps = comps[~comps['id'].isin(synced_ids)]

        logger.info(f"Found {len(new_comps)} new competitions to sync")

        all_results = []

        for _, comp in new_comps.iterrows():
            comp_id = comp['id']
            comp_name = comp.get('name', 'Unknown')

            results = self.sync_competition(comp_id, comp_name)
            if not results.empty:
                results['date_from'] = comp.get('dateFrom', '')
                results['year'] = comp.get('year', datetime.now().year)
                all_results.append(results)

            # Mark as synced
            synced_ids.add(comp_id)

        # Save results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)

            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.data_dir / "data" / f"sync_{timestamp}.csv"
            output_file.parent.mkdir(exist_ok=True)

            combined.to_csv(output_file, index=False)
            logger.info(f"Saved {len(combined)} results to {output_file}")

            # Also append to current year file
            year = datetime.now().year
            year_file = self.data_dir / f"Results_{year}.csv"

            if year_file.exists():
                existing = pd.read_csv(year_file)
                combined = pd.concat([existing, combined], ignore_index=True)
                combined = combined.drop_duplicates(subset=['ResultId'], keep='last')

            combined.to_csv(year_file, index=False)
            logger.info(f"Updated {year_file}")

        # Update sync state
        state['last_sync'] = datetime.now().isoformat()
        state['synced_competitions'] = list(synced_ids)
        state['last_year'] = datetime.now().year
        self.save_sync_state(state)

        logger.info("Sync complete!")

        return combined if all_results else pd.DataFrame()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Incremental data sync")
    parser.add_argument("--full", action="store_true", help="Full sync (re-fetch everything)")
    parser.add_argument("--year", type=int, help="Sync specific year")
    parser.add_argument("--status", action="store_true", help="Show sync status")

    args = parser.parse_args()

    sync = IncrementalSync()

    if args.status:
        state = sync.load_sync_state()
        print("\n=== Sync Status ===")
        print(f"Last sync: {state.get('last_sync', 'Never')}")
        print(f"Synced competitions: {len(state.get('synced_competitions', []))}")
        print(f"Last year: {state.get('last_year', 'N/A')}")
        return

    results = sync.run_sync(full_sync=args.full, year=args.year)

    if results is not None and not results.empty:
        print(f"\nSync complete! {len(results)} new results")
        print(f"Pacing distribution:")
        if 'pacing_type' in results.columns:
            print(results['pacing_type'].value_counts())


if __name__ == "__main__":
    main()
