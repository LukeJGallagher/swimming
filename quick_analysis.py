"""
Quick analysis utilities for swimming data
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_swimming_scraper import SplitTimeAnalyzer

sns.set_style('whitegrid')


class SwimmingAnalyzer:
    """Quick analysis tools for swimming data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.analyzer = SplitTimeAnalyzer()

    def load_results(self, year: int = None, file_path: str = None) -> pd.DataFrame:
        """Load results from CSV"""
        if file_path:
            return pd.read_csv(file_path)
        elif year:
            file = self.data_dir / f"results_{year}.csv"
            if file.exists():
                return pd.read_csv(file)
        return pd.DataFrame()

    def get_athlete_progression(self, results_df: pd.DataFrame, athlete_name: str, event: str = None) -> pd.DataFrame:
        """Get athlete's progression over time"""
        athlete_data = results_df[results_df['FullName'].str.contains(athlete_name, case=False, na=False)]

        if event:
            athlete_data = athlete_data[athlete_data['discipline_name'].str.contains(event, case=False, na=False)]

        # Convert times to seconds
        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)

        # Sort by date
        athlete_data = athlete_data.sort_values('date_from')

        return athlete_data[['date_from', 'Time', 'time_seconds', 'discipline_name',
                            'competition_name', 'Rank', 'NAT', 'pacing_type']]

    def analyze_split_patterns(self, results_df: pd.DataFrame, event: str) -> pd.DataFrame:
        """Analyze split patterns for a specific event"""
        event_data = results_df[results_df['discipline_name'].str.contains(event, case=False, na=False)]

        # Filter for results with splits
        with_splits = event_data[event_data['splits_json'].notna()].copy()

        if with_splits.empty:
            print(f"No split data available for {event}")
            return pd.DataFrame()

        # Parse splits and analyze
        split_analyses = []

        for idx, row in with_splits.iterrows():
            splits = json.loads(row['splits_json'])
            lap_times = self.analyzer.calculate_lap_times(splits)

            if lap_times:
                for lap in lap_times:
                    split_analyses.append({
                        'athlete': row['FullName'],
                        'country': row['NAT'],
                        'final_time': row['Time'],
                        'rank': row['Rank'],
                        'lap_number': lap['lap_number'],
                        'lap_time': lap['lap_time_seconds'],
                        'cumulative_time': lap['cumulative_seconds'],
                        'distance': lap['distance']
                    })

        return pd.DataFrame(split_analyses)

    def compare_athletes(self, results_df: pd.DataFrame, athletes: List[str], event: str) -> pd.DataFrame:
        """Compare multiple athletes in a specific event"""
        comparison = []

        for athlete in athletes:
            athlete_data = self.get_athlete_progression(results_df, athlete, event)
            if not athlete_data.empty:
                comparison.append({
                    'athlete': athlete,
                    'best_time': athlete_data['time_seconds'].min(),
                    'recent_time': athlete_data['time_seconds'].iloc[-1] if len(athlete_data) > 0 else None,
                    'num_races': len(athlete_data),
                    'avg_rank': athlete_data['Rank'].mean(),
                    'improvement': athlete_data['time_seconds'].iloc[0] - athlete_data['time_seconds'].iloc[-1]
                                  if len(athlete_data) > 1 else 0
                })

        return pd.DataFrame(comparison).sort_values('best_time')

    def plot_athlete_progression(self, results_df: pd.DataFrame, athlete_name: str, event: str = None):
        """Plot athlete's time progression"""
        progression = self.get_athlete_progression(results_df, athlete_name, event)

        if progression.empty:
            print(f"No data found for {athlete_name}")
            return

        plt.figure(figsize=(12, 6))

        # Convert dates
        progression['date'] = pd.to_datetime(progression['date_from'])

        plt.plot(progression['date'], progression['time_seconds'], marker='o', linewidth=2, markersize=8)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title(f"{athlete_name} - Performance Progression\n{event if event else 'All Events'}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Add best time line
        best_time = progression['time_seconds'].min()
        plt.axhline(y=best_time, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_time:.2f}s')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return progression

    def plot_split_comparison(self, results_df: pd.DataFrame, event: str, top_n: int = 10):
        """Plot split comparison for top athletes in an event"""
        split_data = self.analyze_split_patterns(results_df, event)

        if split_data.empty:
            return

        # Get top N athletes by final time
        top_athletes = split_data.groupby('athlete')['final_time'].first().sort_values().head(top_n).index

        plt.figure(figsize=(14, 8))

        for athlete in top_athletes:
            athlete_splits = split_data[split_data['athlete'] == athlete]
            plt.plot(athlete_splits['distance'], athlete_splits['cumulative_time'],
                    marker='o', label=athlete, linewidth=2)

        plt.xlabel('Distance (m)', fontsize=12)
        plt.ylabel('Cumulative Time (seconds)', fontsize=12)
        plt.title(f'Split Comparison - {event} (Top {top_n})', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def pacing_distribution(self, results_df: pd.DataFrame, event: str = None):
        """Show distribution of pacing types"""
        data = results_df.copy()

        if event:
            data = data[data['discipline_name'].str.contains(event, case=False, na=False)]

        pacing_counts = data['pacing_type'].value_counts()

        plt.figure(figsize=(10, 6))
        pacing_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#3498db'])
        plt.xlabel('Pacing Type', fontsize=12)
        plt.ylabel('Number of Races', fontsize=12)
        plt.title(f'Pacing Strategy Distribution{" - " + event if event else ""}', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return pacing_counts

    def competition_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by competition"""
        summary = results_df.groupby('competition_name').agg({
            'FullName': 'count',  # Number of results
            'NAT': 'nunique',     # Number of countries
            'discipline_name': 'nunique',  # Number of events
            'date_from': 'first',
            'host_country': 'first',
            'host_city': 'first'
        }).rename(columns={
            'FullName': 'total_results',
            'NAT': 'num_countries',
            'discipline_name': 'num_events'
        })

        return summary.sort_values('total_results', ascending=False)

    def identify_records(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Identify world records and other record performances"""
        records = results_df[results_df['RecordType'].notna()].copy()

        if records.empty:
            return pd.DataFrame()

        records = records[['FullName', 'NAT', 'discipline_name', 'Time',
                          'RecordType', 'competition_name', 'date_from']]
        return records.sort_values('date_from', ascending=False)

    def saudi_athletes_analysis(self, results_df: pd.DataFrame) -> Dict:
        """Analyze Saudi athletes' performance"""
        saudi_results = results_df[results_df['NAT'] == 'KSA'].copy()

        if saudi_results.empty:
            return {'message': 'No Saudi athlete results found'}

        analysis = {
            'total_results': len(saudi_results),
            'num_athletes': saudi_results['FullName'].nunique(),
            'events_competed': saudi_results['discipline_name'].unique().tolist(),
            'best_performances': saudi_results.nsmallest(10, 'time_seconds')[
                ['FullName', 'discipline_name', 'Time', 'Rank', 'competition_name', 'date_from']
            ] if 'time_seconds' in saudi_results.columns else None,
            'medals': saudi_results[saudi_results['MedalTag'].notna()].shape[0],
            'competitions': saudi_results['competition_name'].nunique()
        }

        return analysis


def main():
    """Example usage"""
    analyzer = SwimmingAnalyzer()

    # Load data
    print("Loading 2024 results...")
    results = analyzer.load_results(year=2024)

    if not results.empty:
        print(f"Loaded {len(results)} results")

        # Competition summary
        print("\nTop competitions by number of results:")
        print(analyzer.competition_summary(results).head())

        # Pacing distribution
        print("\nPacing type distribution:")
        print(analyzer.pacing_distribution(results))

        # Records
        records = analyzer.identify_records(results)
        if not records.empty:
            print(f"\nFound {len(records)} record performances")
            print(records.head())
    else:
        print("No data found. Run the scraper first:")
        print("python enhanced_swimming_scraper.py")


if __name__ == "__main__":
    main()
