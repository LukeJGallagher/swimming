"""
Split Time Inspector
Tool to investigate, validate, and gather split times from World Aquatics API
"""

import pandas as pd
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplitTimeInspector:
    """Investigate and validate split time availability"""

    BASE_URL = "https://api.worldaquatics.com/fina"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwimmingAnalysis/1.0'
        })

    def inspect_competition_splits(self, competition_id: int, sample_size: int = 5) -> Dict:
        """
        Deep inspection of split time availability for a competition
        """
        logger.info(f"Inspecting competition {competition_id} for split times...")

        # Get competition events
        events_url = f"{self.BASE_URL}/competitions/{competition_id}/events"
        response = self.session.get(events_url)

        if response.status_code != 200:
            return {"error": f"Failed to fetch competition: {response.status_code}"}

        data = response.json()

        split_analysis = {
            "competition_id": competition_id,
            "total_events": 0,
            "events_with_splits": 0,
            "events_without_splits": 0,
            "split_examples": [],
            "event_details": []
        }

        # Find swimming events
        swimming_events = []
        for sport in data.get('Sports', []):
            if sport.get('Name') == 'Swimming':
                swimming_events = sport.get('DisciplineList', [])
                break

        split_analysis["total_events"] = len(swimming_events)

        if not swimming_events:
            logger.warning("No swimming events found")
            return split_analysis

        # Sample events to inspect
        sample_events = swimming_events[:min(sample_size, len(swimming_events))]

        for event in tqdm(sample_events, desc="Inspecting events"):
            time.sleep(1)  # Rate limiting

            event_id = event.get('Id')
            event_name = event.get('DisciplineName', 'Unknown')

            # Get results for this event
            results_url = f"{self.BASE_URL}/events/{event_id}"
            result_response = self.session.get(results_url)

            if result_response.status_code != 200:
                continue

            result_data = result_response.json()

            # Check for splits in results
            has_splits = False
            split_count = 0
            total_results = 0
            split_example = None

            for heat in result_data.get('Heats', []):
                for result in heat.get('Results', []):
                    total_results += 1

                    splits = result.get('Splits', [])

                    if splits and len(splits) > 0:
                        # Check if splits actually have data
                        valid_splits = [s for s in splits if s.get('time') or s.get('distance')]

                        if valid_splits:
                            has_splits = True
                            split_count += 1

                            # Save first example
                            if not split_example:
                                split_example = {
                                    'athlete': result.get('FullName', 'Unknown'),
                                    'time': result.get('Time'),
                                    'splits': valid_splits,
                                    'heat': heat.get('Name')
                                }

            if has_splits:
                split_analysis["events_with_splits"] += 1
            else:
                split_analysis["events_without_splits"] += 1

            event_detail = {
                'event_id': event_id,
                'event_name': event_name,
                'has_splits': has_splits,
                'split_coverage': f"{split_count}/{total_results}" if total_results > 0 else "0/0",
                'coverage_percentage': (split_count / total_results * 100) if total_results > 0 else 0
            }

            split_analysis["event_details"].append(event_detail)

            if split_example:
                split_analysis["split_examples"].append({
                    'event': event_name,
                    'example': split_example
                })

        # Calculate summary statistics
        if split_analysis["total_events"] > 0:
            split_analysis["split_coverage_pct"] = (
                split_analysis["events_with_splits"] / split_analysis["total_events"] * 100
            )

        return split_analysis

    def find_competitions_with_splits(self, year: int, limit: int = 10) -> List[Dict]:
        """
        Find competitions from a year that have split times
        """
        logger.info(f"Searching for competitions with splits in {year}...")

        # Get competitions for year
        url = f"{self.BASE_URL}/competitions"
        params = {
            'pageSize': 100,
            'venueDateFrom': f'{year}-01-01T00:00:00+00:00',
            'venueDateTo': f'{year}-12-31T23:59:59+00:00',
            'disciplines': 'SW',
            'group': 'FINA',
            'sort': 'dateFrom,desc'
        }

        response = self.session.get(url, params=params)

        if response.status_code != 200:
            logger.error(f"Failed to fetch competitions: {response.status_code}")
            return []

        competitions = response.json().get('content', [])
        logger.info(f"Found {len(competitions)} competitions in {year}")

        competitions_with_splits = []

        # Check each competition (limit to avoid excessive API calls)
        for comp in tqdm(competitions[:limit], desc=f"Checking competitions"):
            comp_id = comp.get('id')
            comp_name = comp.get('name')

            # Quick check - inspect just 2 events
            inspection = self.inspect_competition_splits(comp_id, sample_size=2)

            if inspection.get("events_with_splits", 0) > 0:
                competitions_with_splits.append({
                    'id': comp_id,
                    'name': comp_name,
                    'official_name': comp.get('officialName'),
                    'date_from': comp.get('dateFrom'),
                    'location': comp.get('location', {}).get('city'),
                    'split_coverage': inspection.get('split_coverage_pct', 0)
                })

        return competitions_with_splits

    def validate_existing_splits(self, results_csv: str) -> Dict:
        """
        Validate split time coverage in existing dataset
        """
        logger.info(f"Validating splits in {results_csv}...")

        df = pd.read_csv(results_csv)

        validation = {
            'total_results': len(df),
            'results_with_splits_column': 0,
            'results_with_valid_splits': 0,
            'results_without_splits': 0,
            'split_coverage_pct': 0,
            'sample_splits': []
        }

        # Check for Splits column
        if 'Splits' in df.columns:
            validation['results_with_splits_column'] = df['Splits'].notna().sum()

            # Check for valid splits (not empty arrays)
            for idx, row in df[df['Splits'].notna()].head(20).iterrows():
                try:
                    splits = json.loads(row['Splits']) if isinstance(row['Splits'], str) else row['Splits']

                    if splits and len(splits) > 0:
                        # Check if splits have actual data
                        valid = any(s.get('time') or s.get('distance') for s in splits if isinstance(s, dict))

                        if valid:
                            validation['results_with_valid_splits'] += 1

                            if len(validation['sample_splits']) < 5:
                                validation['sample_splits'].append({
                                    'athlete': row.get('FullName', 'Unknown'),
                                    'event': row.get('DisciplineName', 'Unknown'),
                                    'time': row.get('Time'),
                                    'splits': splits
                                })
                except:
                    continue

        # Check splits_json column (enhanced scraper format)
        if 'splits_json' in df.columns:
            splits_json_count = df['splits_json'].notna().sum()
            validation['results_with_splits_json'] = splits_json_count

        validation['results_without_splits'] = (
            validation['total_results'] - validation['results_with_valid_splits']
        )

        if validation['total_results'] > 0:
            validation['split_coverage_pct'] = (
                validation['results_with_valid_splits'] / validation['total_results'] * 100
            )

        return validation

    def extract_split_structure(self, sample_splits: List) -> Dict:
        """
        Analyze the structure of split time data
        """
        if not sample_splits:
            return {"error": "No splits to analyze"}

        # Take first valid split
        for split_data in sample_splits:
            if split_data and len(split_data) > 0:
                first_split = split_data[0]

                return {
                    "fields": list(first_split.keys()),
                    "example": first_split,
                    "total_laps": len(split_data),
                    "structure": type(split_data).__name__
                }

        return {"error": "No valid split structure found"}


class DedicatedSplitScraper:
    """
    Dedicated scraper focused on maximizing split time collection
    """

    BASE_URL = "https://api.worldaquatics.com/fina"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwimmingAnalysis/1.0'
        })

    def scrape_splits_for_competition(self, competition_id: int, output_file: str = None) -> pd.DataFrame:
        """
        Scrape ALL split times for a competition
        """
        logger.info(f"Scraping splits for competition {competition_id}...")

        all_splits = []

        # Get events
        events_url = f"{self.BASE_URL}/competitions/{competition_id}/events"
        response = self.session.get(events_url)

        if response.status_code != 200:
            logger.error(f"Failed to fetch competition events")
            return pd.DataFrame()

        data = response.json()

        # Find swimming events
        swimming_events = []
        for sport in data.get('Sports', []):
            if sport.get('Name') == 'Swimming':
                swimming_events = sport.get('DisciplineList', [])
                break

        logger.info(f"Found {len(swimming_events)} swimming events")

        for event in tqdm(swimming_events, desc="Scraping events"):
            time.sleep(1.5)  # Rate limiting

            event_id = event.get('Id')
            event_name = event.get('DisciplineName')
            gender = event.get('Gender')

            # Get results
            results_url = f"{self.BASE_URL}/events/{event_id}"
            result_response = self.session.get(results_url)

            if result_response.status_code != 200:
                continue

            result_data = result_response.json()

            # Extract all splits
            for heat in result_data.get('Heats', []):
                heat_name = heat.get('Name')

                for result in heat.get('Results', []):
                    athlete = result.get('FullName')
                    final_time = result.get('Time')
                    rank = result.get('Rank')
                    nat = result.get('NAT')
                    reaction_time = result.get('RT')

                    splits = result.get('Splits', [])

                    if splits and len(splits) > 0:
                        # Validate splits have data
                        valid_splits = [s for s in splits if s.get('time') or s.get('distance')]

                        if valid_splits:
                            all_splits.append({
                                'competition_id': competition_id,
                                'event_id': event_id,
                                'event_name': event_name,
                                'gender': gender,
                                'heat_category': heat_name,
                                'athlete_name': athlete,
                                'country': nat,
                                'final_time': final_time,
                                'rank': rank,
                                'reaction_time': reaction_time,
                                'splits_raw': json.dumps(valid_splits),
                                'num_splits': len(valid_splits)
                            })

        df = pd.DataFrame(all_splits)

        if output_file and not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} split records to {output_file}")

        return df

    def scrape_major_competitions_splits(self, year: int, output_dir: str = "splits_data") -> List[str]:
        """
        Target major competitions that typically have splits
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Major competition keywords
        major_keywords = [
            'World Championships',
            'World Aquatics Championships',
            'Olympic',
            'World Cup',
            'Short Course World Championships'
        ]

        # Get competitions
        url = f"{self.BASE_URL}/competitions"
        params = {
            'pageSize': 100,
            'venueDateFrom': f'{year}-01-01T00:00:00+00:00',
            'venueDateTo': f'{year}-12-31T23:59:59+00:00',
            'disciplines': 'SW',
            'group': 'FINA',
            'sort': 'dateFrom,desc'
        }

        response = self.session.get(url, params=params)

        if response.status_code != 200:
            logger.error("Failed to fetch competitions")
            return []

        competitions = response.json().get('content', [])

        # Filter for major competitions
        major_comps = [
            comp for comp in competitions
            if any(keyword.lower() in comp.get('officialName', '').lower()
                   for keyword in major_keywords)
        ]

        logger.info(f"Found {len(major_comps)} major competitions in {year}")

        output_files = []

        for comp in major_comps:
            comp_id = comp.get('id')
            comp_name = comp.get('name', 'unknown').replace(' ', '_')

            logger.info(f"\nScraping: {comp.get('officialName')}")

            output_file = f"{output_dir}/splits_{comp_name}_{comp_id}_{year}.csv"

            df = self.scrape_splits_for_competition(comp_id, output_file)

            if not df.empty:
                output_files.append(output_file)
                logger.info(f"✓ Collected {len(df)} split records")
            else:
                logger.warning(f"✗ No splits found")

        return output_files


def generate_split_coverage_report(inspection_results: Dict) -> str:
    """Generate readable report of split time coverage"""

    report = []
    report.append("=" * 80)
    report.append("SPLIT TIME COVERAGE REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(f"Competition ID: {inspection_results.get('competition_id')}")
    report.append(f"Total Events: {inspection_results.get('total_events', 0)}")
    report.append(f"Events WITH Splits: {inspection_results.get('events_with_splits', 0)}")
    report.append(f"Events WITHOUT Splits: {inspection_results.get('events_without_splits', 0)}")
    report.append(f"Coverage: {inspection_results.get('split_coverage_pct', 0):.1f}%")
    report.append("")

    if inspection_results.get('event_details'):
        report.append("-" * 80)
        report.append("EVENT BREAKDOWN")
        report.append("-" * 80)
        report.append(f"{'Event':<40} {'Has Splits':<12} {'Coverage':<15}")
        report.append("-" * 80)

        for detail in inspection_results['event_details']:
            event_name = detail['event_name'][:38]
            has_splits = "[YES]" if detail['has_splits'] else "[NO]"
            coverage = f"{detail['coverage_percentage']:.1f}% ({detail['split_coverage']})"

            report.append(f"{event_name:<40} {has_splits:<12} {coverage:<15}")

    if inspection_results.get('split_examples'):
        report.append("")
        report.append("-" * 80)
        report.append("SPLIT TIME EXAMPLES")
        report.append("-" * 80)

        for example in inspection_results['split_examples'][:3]:
            report.append(f"\nEvent: {example['event']}")
            ex_data = example['example']
            report.append(f"Athlete: {ex_data.get('athlete')}")
            report.append(f"Final Time: {ex_data.get('time')}")
            report.append(f"Heat: {ex_data.get('heat')}")
            report.append(f"Splits:")

            for split in ex_data.get('splits', [])[:5]:  # Show first 5 splits
                dist = split.get('distance', 'N/A')
                time = split.get('time', 'N/A')
                report.append(f"  {dist}m: {time}")

    report.append("")
    report.append("=" * 80)

    return '\n'.join(report)


def main():
    """Test split time inspector"""

    print("=" * 80)
    print("SPLIT TIME INSPECTOR - Testing")
    print("=" * 80)
    print("")

    inspector = SplitTimeInspector()

    # Test 1: Check recent major competition
    print("Test 1: Inspecting 2024 World Championships (Short Course)")
    print("-" * 80)

    # Competition ID 3433 is World Aquatics Championships (25m) 2024
    inspection = inspector.inspect_competition_splits(3433, sample_size=5)

    report = generate_split_coverage_report(inspection)
    print(report)

    # Test 2: Validate existing data
    print("\n\nTest 2: Validating existing results file")
    print("-" * 80)

    try:
        validation = inspector.validate_existing_splits('data/results_2024.csv')

        print(f"Total Results: {validation['total_results']}")
        print(f"Results with Split Data: {validation['results_with_valid_splits']}")
        print(f"Coverage: {validation['split_coverage_pct']:.2f}%")
        print("")

        if validation['sample_splits']:
            print("Sample Split Found:")
            sample = validation['sample_splits'][0]
            print(f"  Athlete: {sample['athlete']}")
            print(f"  Event: {sample['event']}")
            print(f"  Time: {sample['time']}")
            print(f"  Splits: {len(sample['splits'])} laps")

    except FileNotFoundError:
        print("No existing results file found. Run scraper first.")

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES:")
    print("=" * 80)
    print("")
    print("# Find competitions with splits:")
    print("comps = inspector.find_competitions_with_splits(2024, limit=5)")
    print("")
    print("# Scrape all splits from a competition:")
    print("scraper = DedicatedSplitScraper()")
    print("df = scraper.scrape_splits_for_competition(3433, 'splits_worlds_2024.csv')")
    print("")
    print("# Scrape major competitions:")
    print("files = scraper.scrape_major_competitions_splits(2024)")
    print("")


if __name__ == "__main__":
    main()
