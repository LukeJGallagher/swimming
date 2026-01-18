"""
Performance Analyst Tools
Automated workflows for swimming performance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime, timedelta
from enhanced_swimming_scraper import SplitTimeAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)


class AthleteProfiler:
    """Create comprehensive athlete profiles"""

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def create_profile(self, athlete_name: str, include_splits: bool = True) -> Dict:
        """Generate complete athlete profile"""

        # Get athlete's results
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ].copy()

        if athlete_data.empty:
            return {'error': f'No data found for {athlete_name}'}

        # Convert times to seconds for analysis
        athlete_data['time_seconds'] = athlete_data['Time'].apply(
            self.analyzer.time_to_seconds
        )

        # Basic info
        profile = {
            'name': athlete_data['FullName'].iloc[0],
            'country': athlete_data['NAT'].iloc[0] if 'NAT' in athlete_data.columns else 'Unknown',
            'total_races': len(athlete_data),
            'first_competition': athlete_data['date_from'].min() if 'date_from' in athlete_data.columns else None,
            'latest_competition': athlete_data['date_from'].max() if 'date_from' in athlete_data.columns else None,
        }

        # Best times by event
        best_times = athlete_data.groupby('discipline_name').agg({
            'time_seconds': 'min',
            'Time': 'first',
            'date_from': 'first',
            'competition_name': 'first',
            'Rank': 'min'
        }).reset_index()

        best_times = best_times.sort_values('time_seconds')
        profile['best_times'] = best_times.to_dict('records')

        # Medal count
        if 'MedalTag' in athlete_data.columns:
            medals = athlete_data['MedalTag'].value_counts().to_dict()
            profile['medals'] = {
                'gold': medals.get('G', 0),
                'silver': medals.get('S', 0),
                'bronze': medals.get('B', 0),
                'total': sum(medals.values())
            }

        # Pacing analysis
        if include_splits and 'pacing_type' in athlete_data.columns:
            pacing_dist = athlete_data['pacing_type'].value_counts().to_dict()
            profile['pacing_preference'] = pacing_dist

            if 'lap_variance' in athlete_data.columns:
                avg_variance = athlete_data['lap_variance'].mean()
                profile['avg_lap_consistency'] = round(avg_variance, 3)

        # Recent form (last 3 months)
        if 'date_from' in athlete_data.columns:
            recent_cutoff = datetime.now() - timedelta(days=90)
            athlete_data['date_parsed'] = pd.to_datetime(athlete_data['date_from'], errors='coerce')
            recent_data = athlete_data[athlete_data['date_parsed'] > recent_cutoff]

            if not recent_data.empty:
                profile['recent_form'] = {
                    'races': len(recent_data),
                    'avg_time': round(recent_data['time_seconds'].mean(), 2),
                    'best_recent': round(recent_data['time_seconds'].min(), 2),
                    'events': recent_data['discipline_name'].unique().tolist()
                }

        return profile

    def compare_athletes(self, athlete_names: List[str], event: str = None) -> pd.DataFrame:
        """Compare multiple athletes"""
        comparisons = []

        for name in athlete_names:
            profile = self.create_profile(name, include_splits=False)

            if 'error' not in profile:
                # Get best time for specific event
                if event and 'best_times' in profile:
                    event_times = [bt for bt in profile['best_times']
                                  if event.lower() in bt['discipline_name'].lower()]
                    best_time = event_times[0]['time_seconds'] if event_times else None
                else:
                    best_time = None

                comparisons.append({
                    'athlete': profile['name'],
                    'country': profile['country'],
                    'total_races': profile['total_races'],
                    'medals': profile.get('medals', {}).get('total', 0),
                    'best_time': best_time,
                    'event': event if event else 'Overall'
                })

        df = pd.DataFrame(comparisons)
        if 'best_time' in df.columns:
            df = df.sort_values('best_time')

        return df


class ProgressionTracker:
    """Track athlete progression over time"""

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def calculate_progression(self, athlete_name: str, event: str) -> pd.DataFrame:
        """Calculate progression metrics for athlete in specific event"""

        # Filter data
        athlete_data = self.results[
            (self.results['FullName'].str.contains(athlete_name, case=False, na=False)) &
            (self.results['discipline_name'].str.contains(event, case=False, na=False))
        ].copy()

        if athlete_data.empty:
            return pd.DataFrame()

        # Convert times and sort
        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)
        athlete_data = athlete_data.sort_values('date_from')

        # Calculate metrics
        athlete_data['personal_best'] = athlete_data['time_seconds'].cummin()
        athlete_data['seconds_off_pb'] = athlete_data['time_seconds'] - athlete_data['personal_best']
        athlete_data['improvement_from_prev'] = athlete_data['time_seconds'].diff()

        # Rolling averages
        athlete_data['rolling_avg_3'] = athlete_data['time_seconds'].rolling(window=3, min_periods=1).mean()

        return athlete_data[['date_from', 'Time', 'time_seconds', 'personal_best',
                            'seconds_off_pb', 'improvement_from_prev', 'rolling_avg_3',
                            'competition_name', 'Rank']]

    def identify_breakthroughs(self, progression_df: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
        """Identify significant breakthrough performances"""
        if progression_df.empty:
            return []

        breakthroughs = []
        for idx in range(1, len(progression_df)):
            improvement = -progression_df.iloc[idx]['improvement_from_prev']

            if improvement >= threshold:  # Improved by threshold seconds or more
                breakthroughs.append({
                    'date': progression_df.iloc[idx]['date_from'],
                    'time': progression_df.iloc[idx]['Time'],
                    'improvement_seconds': round(improvement, 2),
                    'competition': progression_df.iloc[idx]['competition_name'],
                    'rank': progression_df.iloc[idx]['Rank']
                })

        return breakthroughs


class CompetitionAnalyzer:
    """Analyze competition-level data"""

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df

    def competition_summary(self, competition_name: str = None, competition_id: int = None) -> Dict:
        """Generate comprehensive competition summary"""

        # Filter by competition
        if competition_name:
            comp_data = self.results[
                self.results['competition_name'].str.contains(competition_name, case=False, na=False)
            ]
        elif competition_id:
            comp_data = self.results[self.results['competition_id'] == competition_id]
        else:
            return {'error': 'Must provide competition_name or competition_id'}

        if comp_data.empty:
            return {'error': 'No data found for this competition'}

        summary = {
            'competition_name': comp_data['competition_official_name'].iloc[0] if 'competition_official_name' in comp_data.columns else comp_data['competition_name'].iloc[0],
            'dates': f"{comp_data['date_from'].min()} to {comp_data['date_to'].max()}" if 'date_from' in comp_data.columns else 'Unknown',
            'location': f"{comp_data['host_city'].iloc[0]}, {comp_data['host_country'].iloc[0]}" if 'host_city' in comp_data.columns else 'Unknown',
            'total_results': len(comp_data),
            'unique_athletes': comp_data['FullName'].nunique() if 'FullName' in comp_data.columns else 0,
            'unique_countries': comp_data['NAT'].nunique() if 'NAT' in comp_data.columns else 0,
            'events': comp_data['discipline_name'].nunique() if 'discipline_name' in comp_data.columns else 0,
        }

        # Medal table
        if 'MedalTag' in comp_data.columns and 'NAT' in comp_data.columns:
            medals_by_country = comp_data[comp_data['MedalTag'].notna()].groupby(['NAT', 'MedalTag']).size().unstack(fill_value=0)

            if not medals_by_country.empty:
                # Calculate points (Gold=3, Silver=2, Bronze=1)
                if 'G' in medals_by_country.columns:
                    medals_by_country['points'] = medals_by_country.get('G', 0) * 3 + medals_by_country.get('S', 0) * 2 + medals_by_country.get('B', 0)
                    summary['medal_table'] = medals_by_country.sort_values('points', ascending=False).head(10).to_dict()

        # Records set
        if 'RecordType' in comp_data.columns:
            records = comp_data[comp_data['RecordType'].notna()]
            summary['records_set'] = len(records)

        return summary

    def country_performance(self, country_code: str, competition_name: str = None) -> Dict:
        """Analyze specific country's performance"""

        data = self.results[self.results['NAT'] == country_code].copy()

        if competition_name:
            data = data[data['competition_name'].str.contains(competition_name, case=False, na=False)]

        if data.empty:
            return {'error': f'No data found for {country_code}'}

        performance = {
            'country': country_code,
            'total_entries': len(data),
            'unique_athletes': data['FullName'].nunique(),
            'events_competed': data['discipline_name'].nunique(),
        }

        # Medals
        if 'MedalTag' in data.columns:
            medals = data['MedalTag'].value_counts().to_dict()
            performance['medals'] = {
                'gold': medals.get('G', 0),
                'silver': medals.get('S', 0),
                'bronze': medals.get('B', 0),
                'total': sum(medals.values())
            }

        # Finals
        if 'heat_category' in data.columns:
            finals = data[data['heat_category'].str.contains('Final', case=False, na=False)]
            performance['finals_reached'] = len(finals)

        # Top performers
        top_performers = data.nsmallest(10, 'Rank')[
            ['FullName', 'discipline_name', 'Time', 'Rank', 'competition_name']
        ] if 'Rank' in data.columns else None

        if top_performers is not None:
            performance['top_performances'] = top_performers.to_dict('records')

        return performance


class ReportGenerator:
    """Generate formatted reports"""

    @staticmethod
    def athlete_profile_report(profile: Dict) -> str:
        """Format athlete profile as readable report"""

        if 'error' in profile:
            return f"ERROR: {profile['error']}"

        report = []
        report.append("=" * 70)
        report.append(f"ATHLETE PROFILE: {profile['name']}")
        report.append("=" * 70)
        report.append(f"Country: {profile['country']}")
        report.append(f"Total Competitions: {profile['total_races']}")

        if profile.get('first_competition'):
            report.append(f"Career Span: {profile['first_competition']} to {profile['latest_competition']}")

        # Medals
        if 'medals' in profile:
            m = profile['medals']
            report.append(f"\nMedals: {m['gold']} Gold, {m['silver']} Silver, {m['bronze']} Bronze (Total: {m['total']})")

        # Best times
        if 'best_times' in profile:
            report.append("\nBEST TIMES:")
            for bt in profile['best_times'][:5]:  # Top 5 events
                report.append(f"  {bt['discipline_name']:30s} {bt['Time']:>10s} "
                            f"(Rank: {bt.get('Rank', 'N/A')}) - {bt.get('competition_name', 'Unknown')[:40]}")

        # Pacing
        if 'pacing_preference' in profile:
            report.append("\nPACING ANALYSIS:")
            for ptype, count in profile['pacing_preference'].items():
                if ptype:
                    report.append(f"  {ptype}: {count} races")

        if 'avg_lap_consistency' in profile:
            report.append(f"  Average Lap Variance: {profile['avg_lap_consistency']}")

        # Recent form
        if 'recent_form' in profile:
            rf = profile['recent_form']
            report.append(f"\nRECENT FORM (Last 90 days):")
            report.append(f"  Races: {rf['races']}")
            report.append(f"  Best Time: {rf['best_recent']}s")
            report.append(f"  Events: {', '.join(rf['events'])}")

        report.append("=" * 70)
        return '\n'.join(report)

    @staticmethod
    def competition_summary_report(summary: Dict) -> str:
        """Format competition summary as readable report"""

        if 'error' in summary:
            return f"ERROR: {summary['error']}"

        report = []
        report.append("=" * 70)
        report.append(f"COMPETITION SUMMARY")
        report.append("=" * 70)
        report.append(f"Event: {summary['competition_name']}")
        report.append(f"Dates: {summary['dates']}")
        report.append(f"Location: {summary['location']}")
        report.append(f"\nParticipation:")
        report.append(f"  Athletes: {summary['unique_athletes']}")
        report.append(f"  Countries: {summary['unique_countries']}")
        report.append(f"  Events: {summary['events']}")
        report.append(f"  Total Results: {summary['total_results']}")

        if 'records_set' in summary:
            report.append(f"\nRecords Set: {summary['records_set']}")

        if 'medal_table' in summary:
            report.append("\nMEDAL TABLE (Top 10):")
            report.append(f"{'Country':<10} {'Gold':>6} {'Silver':>6} {'Bronze':>6} {'Total':>6}")
            report.append("-" * 40)

            table = summary['medal_table']
            for country in list(table.get('points', {}).keys())[:10]:
                gold = table.get('G', {}).get(country, 0)
                silver = table.get('S', {}).get(country, 0)
                bronze = table.get('B', {}).get(country, 0)
                total = gold + silver + bronze
                report.append(f"{country:<10} {gold:>6} {silver:>6} {bronze:>6} {total:>6}")

        report.append("=" * 70)
        return '\n'.join(report)

    @staticmethod
    def save_report(report_text: str, filename: str, output_dir: str = "reports"):
        """Save report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)

        return str(filepath)


def main():
    """Example usage"""
    print("Performance Analyst Tools - Demo")
    print("=" * 70)
    print("\nTo use these tools:")
    print("1. Load your results data:")
    print("   results = pd.read_csv('data/results_2024.csv')")
    print("\n2. Create profiler:")
    print("   profiler = AthleteProfiler(results)")
    print("   profile = profiler.create_profile('Athlete Name')")
    print("\n3. Generate report:")
    print("   report = ReportGenerator.athlete_profile_report(profile)")
    print("   print(report)")
    print("\n4. Track progression:")
    print("   tracker = ProgressionTracker(results)")
    print("   progression = tracker.calculate_progression('Athlete', '100m Freestyle')")
    print("\n5. Analyze competition:")
    print("   analyzer = CompetitionAnalyzer(results)")
    print("   summary = analyzer.competition_summary('World Championships')")
    print("=" * 70)


if __name__ == "__main__":
    main()
