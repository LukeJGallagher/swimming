"""
Enhanced Swimming Data Scraper
Scrapes World Aquatics API with split times extraction and AI-powered enrichment
Supports writing to both CSV (local) and Azure SQL (cloud)
"""

import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure SQL support
try:
    from azure_db import get_azure_connection, _use_azure, get_connection_mode
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('swimming_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WorldAquaticsAPI:
    """Handler for World Aquatics API interactions"""

    BASE_URL = "https://api.worldaquatics.com/fina"

    def __init__(self, rate_limit_delay: float = 1.5):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SwimmingDataAnalysis/1.0'
        })

    def safe_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                time.sleep(self.rate_limit_delay)
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s... Error: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from {url}: {e}")
                return None
        return None

    def get_competitions(self, year: int, disciplines: str = 'SW', group: str = 'FINA') -> Optional[pd.DataFrame]:
        """Get competitions for a specific year"""
        url = f"{self.BASE_URL}/competitions"
        params = {
            'pageSize': 100,
            'venueDateFrom': f'{year}-01-01T00:00:00+00:00',
            'venueDateTo': f'{year}-12-31T23:59:59+00:00',
            'disciplines': disciplines,
            'group': group,
            'sort': 'dateFrom,desc'
        }

        data = self.safe_request(url, params)
        if data and 'content' in data:
            df = pd.DataFrame(data['content'])
            if not df.empty and 'location' in df.columns:
                location_data = pd.json_normalize(df['location'])
                df['host_country'] = location_data.get('countryName', '')
                df['host_city'] = location_data.get('city', '')
            return df
        return None

    def get_competition_events(self, competition_id: int) -> Optional[List[Dict]]:
        """Get events for a specific competition"""
        url = f"{self.BASE_URL}/competitions/{competition_id}/events"
        data = self.safe_request(url)

        if not data:
            return None

        swimming_events = []
        for sport in data.get('Sports', []):
            if sport.get('Name') == 'Swimming':
                for discipline in sport.get('DisciplineList', []):
                    discipline['competition_id'] = competition_id
                    swimming_events.append(discipline)

        return swimming_events

    def get_event_results(self, event_id: int) -> Optional[List[Dict]]:
        """Get detailed results for a specific event"""
        url = f"{self.BASE_URL}/events/{event_id}"
        data = self.safe_request(url)

        if not data:
            return None

        all_results = []
        for heat in data.get('Heats', []):
            heat_name = heat.get('Name', 'Unknown')
            for result in heat.get('Results', []):
                result['heat_category'] = heat_name
                result['event_id'] = event_id
                all_results.append(result)

        return all_results


class SplitTimeAnalyzer:
    """Analyzer for swimming split times"""

    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        """Convert time string (MM:SS.ss or SS.ss) to seconds"""
        if not time_str or pd.isna(time_str):
            return 0.0

        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(parts[0])
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def seconds_to_time(seconds: float) -> str:
        """Convert seconds to time string"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes > 0:
            return f"{minutes}:{secs:05.2f}"
        return f"{secs:.2f}"

    @staticmethod
    def parse_splits(splits_data) -> List[Dict]:
        """Parse splits from API response

        Expected API format:
        {
            "Time": "11.12",        # Cumulative time
            "Distance": "25m",      # Distance marker
            "Order": 1,             # Lap number
            "DifferentialTime": "11.12"  # Lap time
        }
        """
        if not splits_data or splits_data == '[]':
            return []

        try:
            if isinstance(splits_data, str):
                splits = json.loads(splits_data)
            elif isinstance(splits_data, list):
                splits = splits_data
            else:
                return []

            # Normalize to consistent format (handle both capitalized and lowercase)
            normalized_splits = []
            for split in splits:
                if not isinstance(split, dict):
                    continue

                normalized = {
                    'time': split.get('Time') or split.get('time'),
                    'distance': split.get('Distance') or split.get('distance'),
                    'order': split.get('Order') or split.get('order'),
                    'differential_time': split.get('DifferentialTime') or split.get('differentialTime') or split.get('differential_time')
                }

                # Only include if it has actual data
                if normalized['time'] or normalized['distance']:
                    normalized_splits.append(normalized)

            return normalized_splits
        except (json.JSONDecodeError, TypeError):
            return []

    @classmethod
    def calculate_lap_times(cls, splits: List[Dict]) -> List[Dict]:
        """Calculate individual lap times from cumulative splits

        Can use DifferentialTime if available, otherwise calculate from cumulative
        """
        if not splits or len(splits) < 1:
            return []

        lap_times = []
        for i, split in enumerate(splits):
            cumulative_time = split.get('time', '')
            curr_seconds = cls.time_to_seconds(cumulative_time)

            # Check if DifferentialTime is provided (actual lap time)
            differential_time = split.get('differential_time')

            if differential_time:
                # Use provided lap time
                lap_time_seconds = cls.time_to_seconds(differential_time)
            else:
                # Calculate from cumulative
                if i == 0:
                    lap_time_seconds = curr_seconds
                else:
                    prev_seconds = cls.time_to_seconds(splits[i-1].get('time', '0'))
                    lap_time_seconds = curr_seconds - prev_seconds

            # Parse distance (remove 'm' suffix if present)
            distance_str = str(split.get('distance', '0'))
            distance = int(distance_str.replace('m', '').replace('M', '')) if distance_str else 0

            lap_times.append({
                'lap_number': split.get('order', i + 1),
                'distance': distance,
                'cumulative_time': cumulative_time,
                'cumulative_seconds': curr_seconds,
                'lap_time_seconds': lap_time_seconds,
                'lap_time': cls.seconds_to_time(lap_time_seconds)
            })

        return lap_times

    @classmethod
    def analyze_pacing(cls, lap_times: List[Dict]) -> Optional[Dict]:
        """Analyze pacing strategy from lap times"""
        if not lap_times or len(lap_times) < 2:
            return None

        lap_seconds = [lt['lap_time_seconds'] for lt in lap_times]

        # Calculate first half vs second half
        midpoint = len(lap_seconds) // 2
        first_half = lap_seconds[:midpoint] if midpoint > 0 else lap_seconds[:1]
        second_half = lap_seconds[midpoint:] if midpoint > 0 else lap_seconds[1:]

        first_half_avg = sum(first_half) / len(first_half) if first_half else 0
        second_half_avg = sum(second_half) / len(second_half) if second_half else 0

        split_diff = second_half_avg - first_half_avg

        # Classify pacing
        if abs(split_diff) < 0.5:
            pacing_type = "Even"
        elif split_diff < 0:
            pacing_type = "Negative Split"
        else:
            pacing_type = "Positive Split"

        variance = pd.Series(lap_seconds).std() if len(lap_seconds) > 1 else 0

        return {
            'pacing_type': pacing_type,
            'first_half_avg': round(first_half_avg, 2),
            'second_half_avg': round(second_half_avg, 2),
            'split_difference': round(split_diff, 2),
            'fastest_lap': round(min(lap_seconds), 2),
            'slowest_lap': round(max(lap_seconds), 2),
            'lap_variance': round(variance, 3)
        }


class EnhancedSwimmingScraper:
    """Main scraper class with split time extraction"""

    def __init__(self, output_dir: str = "data", use_azure: bool = True):
        self.api = WorldAquaticsAPI()
        self.analyzer = SplitTimeAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Azure SQL configuration
        self.use_azure = use_azure and AZURE_AVAILABLE and _use_azure()
        if self.use_azure:
            logger.info(f"Enhanced Swimming Scraper initialized (Azure SQL mode)")
        else:
            logger.info(f"Enhanced Swimming Scraper initialized (CSV mode)")

    def _time_to_seconds(self, time_str: str) -> Optional[float]:
        """Convert time string (MM:SS.ss or SS.ss) to seconds."""
        if pd.isna(time_str) or not time_str:
            return None
        try:
            time_str = str(time_str).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return None

    def save_results_to_azure(self, results_df: pd.DataFrame, year: int) -> int:
        """Save results DataFrame to Azure SQL database.

        Returns number of rows inserted.
        """
        if results_df.empty:
            return 0

        try:
            conn = get_azure_connection()
            if conn is None:
                logger.error("Failed to get Azure SQL connection")
                return 0

            cursor = conn.cursor()

            # Column mapping from scraper format to Azure schema
            column_mapping = {
                'heat_category': 'heat_category',
                'discipline_name': 'discipline_name',
                'gender': 'gender',
                'event_id': 'event_id',
                'FullName': 'full_name',
                'FirstName': 'first_name',
                'LastName': 'last_name',
                'NAT': 'nationality',
                'NATName': 'nationality_name',
                'PersonId': 'person_id',
                'BiographyId': 'biography_id',
                'AthleteResultAge': 'athlete_age',
                'ResultId': 'result_id',
                'Lane': 'lane',
                'HeatRank': 'heat_rank',
                'Rank': 'final_rank',
                'Time': 'time_raw',
                'RT': 'reaction_time',
                'TimeBehind': 'time_behind',
                'Points': 'fina_points',
                'MedalTag': 'medal_tag',
                'Qualified': 'qualified',
                'RecordType': 'record_type',
                'splits_json': 'splits_json',
            }

            # Prepare DataFrame
            df = results_df.copy()
            df = df.rename(columns=column_mapping)
            df['year'] = year

            # Convert time to seconds
            if 'time_raw' in df.columns:
                df['time_seconds'] = df['time_raw'].apply(self._time_to_seconds)

            # Azure SQL schema columns
            schema_columns = [
                'heat_category', 'discipline_name', 'gender', 'event_id',
                'full_name', 'first_name', 'last_name', 'nationality', 'nationality_name',
                'person_id', 'biography_id', 'athlete_age', 'result_id',
                'lane', 'heat_rank', 'final_rank', 'time_raw', 'time_seconds',
                'reaction_time', 'time_behind', 'fina_points', 'medal_tag',
                'qualified', 'record_type', 'splits_json', 'year'
            ]

            # Add missing columns with None
            for col in schema_columns:
                if col not in df.columns:
                    df[col] = None

            # Only keep schema columns
            df = df[schema_columns]

            # Truncate splits_json if too long
            def truncate_splits(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                s = str(x)
                return s[:3900] if len(s) > 3900 else s

            if 'splits_json' in df.columns:
                df['splits_json'] = df['splits_json'].apply(truncate_splits)

            # Convert numeric columns
            numeric_int_cols = ['lane', 'heat_rank', 'final_rank', 'athlete_age', 'fina_points', 'year']
            for col in numeric_int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].where(pd.notna(df[col]), None)

            float_cols = ['time_seconds', 'reaction_time', 'time_behind']
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].where(pd.notna(df[col]), None)

            # Ensure string columns are proper strings
            str_cols = ['heat_category', 'discipline_name', 'gender', 'full_name',
                        'first_name', 'last_name', 'nationality', 'nationality_name',
                        'person_id', 'biography_id', 'result_id', 'time_raw',
                        'medal_tag', 'qualified', 'record_type', 'splits_json', 'event_id']
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

            # Build INSERT statement
            columns = df.columns.tolist()
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join(columns)
            sql = f"INSERT INTO results_flat ({column_names}) VALUES ({placeholders})"

            # Insert in batches
            batch_size = 200
            total_inserted = 0
            failed_rows = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]

                rows = []
                for _, row in batch.iterrows():
                    row_tuple = tuple(None if pd.isna(v) else v for v in row.values)
                    rows.append(row_tuple)

                try:
                    cursor.executemany(sql, rows)
                    conn.commit()
                    total_inserted += len(rows)
                except Exception as e:
                    # Try row by row on failure
                    for row in rows:
                        try:
                            cursor.execute(sql, row)
                            conn.commit()
                            total_inserted += 1
                        except Exception as row_e:
                            failed_rows += 1
                            logger.debug(f"Failed to insert row: {row_e}")

            conn.close()

            if failed_rows > 0:
                logger.warning(f"Azure SQL: {total_inserted} inserted, {failed_rows} failed")
            else:
                logger.info(f"Azure SQL: {total_inserted} rows inserted for year {year}")

            return total_inserted

        except Exception as e:
            logger.error(f"Failed to save to Azure SQL: {e}")
            return 0

    def scrape_year(self, year: int, save_incremental: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape all competitions and results for a given year"""
        logger.info(f"Starting scrape for year {year}")

        # Get competitions
        competitions = self.api.get_competitions(year)
        if competitions is None or competitions.empty:
            logger.warning(f"No competitions found for {year}")
            return pd.DataFrame(), pd.DataFrame()

        logger.info(f"Found {len(competitions)} competitions for {year}")

        # Save competitions
        competitions['year'] = year
        if save_incremental:
            comp_file = self.output_dir / f"competitions_{year}.csv"
            competitions.to_csv(comp_file, index=False)
            logger.info(f"Saved competitions to {comp_file}")

        # Scrape results
        all_results = []

        for idx, comp in tqdm(competitions.iterrows(), total=len(competitions), desc=f"Processing {year}"):
            comp_id = comp['id']
            comp_name = comp.get('name', 'Unknown')

            logger.info(f"Processing competition: {comp_name} (ID: {comp_id})")

            # Get events for this competition
            events = self.api.get_competition_events(comp_id)
            if not events:
                logger.warning(f"No events found for competition {comp_id}")
                continue

            # Get results for each event
            for event in events:
                event_id = event.get('Id')
                discipline_name = event.get('DisciplineName', 'Unknown')
                gender = event.get('Gender', 'Unknown')

                results = self.api.get_event_results(event_id)
                if not results:
                    continue

                # Process each result
                for result in results:
                    # Add event and competition metadata
                    result['discipline_name'] = discipline_name
                    result['gender'] = gender
                    result['competition_id'] = comp_id
                    result['competition_name'] = comp_name
                    result['competition_official_name'] = comp.get('officialName', '')
                    result['date_from'] = comp.get('dateFrom', '')
                    result['date_to'] = comp.get('dateTo', '')
                    result['host_country'] = comp.get('host_country', '')
                    result['host_city'] = comp.get('host_city', '')
                    result['year'] = year

                    # Extract and analyze split times
                    splits_raw = result.get('Splits', [])
                    splits = self.analyzer.parse_splits(splits_raw)

                    if splits:
                        result['splits_json'] = json.dumps(splits)
                        lap_times = self.analyzer.calculate_lap_times(splits)
                        result['lap_times_json'] = json.dumps(lap_times)

                        pacing = self.analyzer.analyze_pacing(lap_times)
                        if pacing:
                            result['pacing_type'] = pacing['pacing_type']
                            result['first_half_avg'] = pacing['first_half_avg']
                            result['second_half_avg'] = pacing['second_half_avg']
                            result['split_difference'] = pacing['split_difference']
                            result['fastest_lap'] = pacing['fastest_lap']
                            result['slowest_lap'] = pacing['slowest_lap']
                            result['lap_variance'] = pacing['lap_variance']
                    else:
                        result['splits_json'] = None
                        result['lap_times_json'] = None
                        result['pacing_type'] = None

                    all_results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)

        if save_incremental and not results_df.empty:
            # Save to CSV (always, as backup)
            results_file = self.output_dir / f"results_{year}.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved {len(results_df)} results to {results_file}")

            # Also save to Azure SQL if enabled
            if self.use_azure:
                azure_count = self.save_results_to_azure(results_df, year)
                if azure_count > 0:
                    logger.info(f"Synced {azure_count} results to Azure SQL")

        logger.info(f"Completed scrape for {year}: {len(results_df)} results from {len(competitions)} competitions")

        return competitions, results_df

    def scrape_year_range(self, start_year: int, end_year: int):
        """Scrape multiple years"""
        all_competitions = []
        all_results = []

        for year in range(start_year, end_year + 1):
            comps, results = self.scrape_year(year)
            all_competitions.append(comps)
            all_results.append(results)

        # Combine all data
        combined_comps = pd.concat(all_competitions, ignore_index=True)
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save combined files
        combined_comps.to_csv(self.output_dir / f"all_competitions_{start_year}_{end_year}.csv", index=False)
        combined_results.to_csv(self.output_dir / f"all_results_{start_year}_{end_year}.csv", index=False)

        storage_mode = "Azure SQL + CSV" if self.use_azure else "CSV only"
        logger.info(f"Saved combined data: {len(combined_comps)} competitions, {len(combined_results)} results ({storage_mode})")

        return combined_comps, combined_results

    def scrape_athlete(self, athlete_id: int, athlete_name: str = None) -> pd.DataFrame:
        """Scrape all results for a specific athlete"""
        logger.info(f"Scraping athlete {athlete_id} ({athlete_name})")

        url = f"{self.api.BASE_URL}/athletes/{athlete_id}/results"
        data = self.api.safe_request(url)

        if not data:
            logger.warning(f"No data found for athlete {athlete_id}")
            return pd.DataFrame()

        results = pd.DataFrame(data.get('Results', []))
        results['athlete_id'] = athlete_id
        results['athlete_name'] = data.get('FullName', athlete_name)

        logger.info(f"Found {len(results)} results for {results['athlete_name'].iloc[0] if not results.empty else 'athlete'}")

        return results


def main():
    """Main execution function"""
    scraper = EnhancedSwimmingScraper(output_dir="data")

    # Example: Scrape 2024 data
    logger.info("Starting enhanced swimming data scraper")
    storage_mode = "Azure SQL + CSV" if scraper.use_azure else "CSV only"
    logger.info(f"Storage mode: {storage_mode}")

    # Scrape specific year
    competitions, results = scraper.scrape_year(2024)

    # Display summary
    if not results.empty:
        logger.info(f"\n{'='*60}")
        logger.info(f"SCRAPING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Storage mode: {storage_mode}")
        logger.info(f"Total competitions: {len(competitions)}")
        logger.info(f"Total results: {len(results)}")
        logger.info(f"Results with split times: {results['splits_json'].notna().sum()}")
        logger.info(f"Pacing types distribution:")
        if 'pacing_type' in results.columns:
            logger.info(results['pacing_type'].value_counts())
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
