"""
AI-Powered Analysis Agents for Elite Swimming Coaches
Provides intelligent insights, recommendations, and race analysis
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class SwimmingAnalysisAgent:
    """Base class for AI-powered swimming analysis"""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _call_ai(self, prompt: str, system_prompt: str = None) -> str:
        """Make AI API call"""
        if not self.api_key:
            return self._fallback_analysis(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "google/gemini-flash-1.5:free",
            "messages": messages,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return self._fallback_analysis(prompt)

    def _fallback_analysis(self, prompt: str) -> str:
        """Fallback when AI is not available"""
        return "AI analysis unavailable. Using rule-based analysis."

    def time_to_seconds(self, time_str) -> Optional[float]:
        """Convert time string to seconds"""
        if not time_str or pd.isna(time_str):
            return None
        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except:
            return None

    def seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to time string"""
        if pd.isna(seconds) or seconds is None:
            return "N/A"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes > 0:
            return f"{minutes}:{secs:05.2f}"
        return f"{secs:.2f}"


class AthleteProfileAgent(SwimmingAnalysisAgent):
    """Agent for comprehensive athlete profiling"""

    def analyze_athlete(self, athlete_name: str) -> Dict:
        """Generate comprehensive athlete analysis"""
        if self.df is None or 'FullName' not in self.df.columns:
            return {"error": "No data available"}

        athlete_df = self.df[self.df['FullName'] == athlete_name]

        if athlete_df.empty:
            return {"error": f"Athlete {athlete_name} not found"}

        # Basic profile
        profile = {
            'name': athlete_name,
            'country': athlete_df['NAT'].iloc[0] if 'NAT' in athlete_df.columns else 'Unknown',
            'total_races': len(athlete_df),
            'events': [],
            'pacing_analysis': {},
            'progression': {},
            'strengths': [],
            'areas_for_improvement': [],
            'recommendations': []
        }

        # Event analysis
        disc_col = 'DisciplineName' if 'DisciplineName' in athlete_df.columns else 'discipline_name'
        if disc_col in athlete_df.columns:
            for event in athlete_df[disc_col].unique():
                event_df = athlete_df[athlete_df[disc_col] == event].copy()
                event_df['time_seconds'] = event_df['Time'].apply(self.time_to_seconds)
                event_df = event_df.dropna(subset=['time_seconds'])

                if not event_df.empty:
                    profile['events'].append({
                        'name': event,
                        'races': len(event_df),
                        'best_time': self.seconds_to_time(event_df['time_seconds'].min()),
                        'avg_time': self.seconds_to_time(event_df['time_seconds'].mean()),
                        'improvement': event_df['time_seconds'].iloc[0] - event_df['time_seconds'].min() if len(event_df) > 1 else 0
                    })

        # Pacing analysis
        if 'pacing_type' in athlete_df.columns:
            pacing_counts = athlete_df['pacing_type'].value_counts()
            profile['pacing_analysis'] = {
                'dominant_style': pacing_counts.idxmax() if not pacing_counts.empty else 'Unknown',
                'positive_split_pct': pacing_counts.get('Positive Split', 0) / len(athlete_df) * 100,
                'negative_split_pct': pacing_counts.get('Negative Split', 0) / len(athlete_df) * 100,
                'even_pct': pacing_counts.get('Even', 0) / len(athlete_df) * 100
            }

            # Determine strengths and improvements
            if pacing_counts.get('Positive Split', 0) > pacing_counts.get('Negative Split', 0):
                profile['areas_for_improvement'].append("Endurance - tendency to fade in second half")
                profile['recommendations'].append("Focus on aerobic capacity and lactate threshold training")
            else:
                profile['strengths'].append("Strong finishing ability")
                profile['recommendations'].append("Maintain aerobic base, work on faster starts")

        # Consistency analysis
        if 'lap_variance' in athlete_df.columns:
            avg_variance = athlete_df['lap_variance'].mean()
            if avg_variance < 0.5:
                profile['strengths'].append("Excellent pacing consistency")
            elif avg_variance > 1.5:
                profile['areas_for_improvement'].append("Pacing consistency - high variance between laps")

        # Medal performance
        if 'MedalTag' in athlete_df.columns:
            medals = athlete_df['MedalTag'].notna().sum()
            if medals > 0:
                profile['strengths'].append(f"Proven competitor - {medals} medal performances")

        return profile

    def generate_training_recommendations(self, profile: Dict) -> List[str]:
        """Generate AI-powered training recommendations"""
        if 'error' in profile:
            return []

        recommendations = []

        # Based on pacing
        if 'pacing_analysis' in profile:
            pacing = profile['pacing_analysis']
            if pacing.get('positive_split_pct', 0) > 60:
                recommendations.append("🏋️ Increase aerobic base training (Zone 2) to improve endurance")
                recommendations.append("🔄 Practice negative split training sets")
                recommendations.append("📊 Monitor lactate levels during training to find threshold")

            if pacing.get('dominant_style') == 'Even':
                recommendations.append("✅ Good pacing strategy - maintain current approach")
                recommendations.append("⚡ Work on race-specific speed to lower overall times")

        # Based on events
        for event in profile.get('events', []):
            if event['improvement'] > 2:
                recommendations.append(f"📈 Strong improvement in {event['name']} - continue current training")
            elif event['improvement'] < 0.5 and event['races'] > 5:
                recommendations.append(f"⚠️ Plateau in {event['name']} - consider training variation")

        return recommendations


class RaceTacticsAgent(SwimmingAnalysisAgent):
    """Agent for race tactics and strategy planning"""

    def generate_race_plan(self, athlete_name: str, event: str, goal_time: float,
                          competition_tier: str = "Championship") -> Dict:
        """Generate detailed race plan with split targets"""

        if self.df is None:
            return {"error": "No data available"}

        athlete_df = self.df[self.df['FullName'] == athlete_name]
        disc_col = 'DisciplineName' if 'DisciplineName' in self.df.columns else 'discipline_name'

        if disc_col in athlete_df.columns:
            athlete_df = athlete_df[athlete_df[disc_col] == event]

        if athlete_df.empty:
            return {"error": "No data for this athlete/event combination"}

        # Analyze historical pacing
        athlete_df['time_seconds'] = athlete_df['Time'].apply(self.time_to_seconds)
        best_time = athlete_df['time_seconds'].min()

        # Parse event distance
        distance = self._parse_event_distance(event)
        num_laps = distance // 50 if distance > 50 else 1

        # Generate split targets based on goal time and pacing preference
        splits = self._calculate_optimal_splits(goal_time, num_laps, athlete_df)

        plan = {
            'athlete': athlete_name,
            'event': event,
            'goal_time': self.seconds_to_time(goal_time),
            'current_pb': self.seconds_to_time(best_time),
            'improvement_needed': best_time - goal_time,
            'competition_tier': competition_tier,
            'split_targets': splits,
            'tactical_notes': [],
            'key_focus_points': []
        }

        # Add tactical notes
        if 'pacing_type' in athlete_df.columns:
            dominant = athlete_df['pacing_type'].mode()
            if len(dominant) > 0:
                if dominant.iloc[0] == 'Positive Split':
                    plan['tactical_notes'].append("⚠️ Historical tendency to fade - focus on controlled start")
                    plan['key_focus_points'].append("Stay relaxed first 100m, build through race")
                elif dominant.iloc[0] == 'Negative Split':
                    plan['tactical_notes'].append("✅ Strong finisher - can afford conservative start")
                    plan['key_focus_points'].append("Trust your finish, don't panic if behind at halfway")

        # Competition-specific advice
        if competition_tier == "Championship":
            plan['tactical_notes'].append("🏆 Championship racing - expect faster competition")
            plan['key_focus_points'].append("Focus on your own race, don't react to others early")

        return plan

    def _parse_event_distance(self, event: str) -> int:
        """Extract distance from event name"""
        import re
        match = re.search(r'(\d+)m', event)
        return int(match.group(1)) if match else 100

    def _calculate_optimal_splits(self, goal_time: float, num_laps: int, athlete_df: pd.DataFrame) -> List[Dict]:
        """Calculate optimal split times"""
        splits = []
        lap_time = goal_time / num_laps

        # Adjust for typical race pattern (first lap faster due to dive)
        first_lap_adjustment = 0.95  # First lap usually 5% faster
        subsequent_adjustment = 1.02  # Slight fade is normal

        for i in range(num_laps):
            if i == 0:
                split_time = lap_time * first_lap_adjustment
            else:
                split_time = lap_time * subsequent_adjustment

            cumulative = sum(s['split_time'] for s in splits) + split_time

            splits.append({
                'lap': i + 1,
                'distance': (i + 1) * 50,
                'split_time': round(split_time, 2),
                'cumulative': round(cumulative, 2),
                'cumulative_formatted': self.seconds_to_time(cumulative)
            })

        return splits


class CompetitionScoutAgent(SwimmingAnalysisAgent):
    """Agent for competition scouting and opponent analysis"""

    def scout_competitors(self, event: str, athlete_name: str = None) -> Dict:
        """Scout competitors for a specific event"""
        if self.df is None:
            return {"error": "No data available"}

        disc_col = 'DisciplineName' if 'DisciplineName' in self.df.columns else 'discipline_name'

        if disc_col not in self.df.columns:
            return {"error": "Event data not available"}

        event_df = self.df[self.df[disc_col] == event].copy()
        event_df['time_seconds'] = event_df['Time'].apply(self.time_to_seconds)
        event_df = event_df.dropna(subset=['time_seconds'])

        if event_df.empty:
            return {"error": f"No data for event {event}"}

        # Get best time per athlete
        best_times = event_df.loc[event_df.groupby('FullName')['time_seconds'].idxmin()]
        best_times = best_times.sort_values('time_seconds')

        # Top 10 analysis
        top_10 = best_times.head(10)

        scouting_report = {
            'event': event,
            'total_athletes': len(best_times),
            'top_competitors': [],
            'time_standards': {
                'world_best': self.seconds_to_time(top_10['time_seconds'].min()),
                'top_8_cut': self.seconds_to_time(top_10['time_seconds'].iloc[7] if len(top_10) >= 8 else top_10['time_seconds'].iloc[-1]),
                'top_16_cut': self.seconds_to_time(best_times['time_seconds'].iloc[15] if len(best_times) >= 16 else best_times['time_seconds'].iloc[-1])
            },
            'pacing_trends': {}
        }

        # Analyze top competitors
        for _, row in top_10.iterrows():
            competitor = {
                'name': row['FullName'],
                'country': row.get('NAT', 'UNK'),
                'best_time': self.seconds_to_time(row['time_seconds']),
                'pacing_style': row.get('pacing_type', 'Unknown')
            }
            scouting_report['top_competitors'].append(competitor)

        # Pacing trends in top 10
        if 'pacing_type' in top_10.columns:
            pacing_counts = top_10['pacing_type'].value_counts()
            scouting_report['pacing_trends'] = {
                'dominant_strategy': pacing_counts.idxmax() if not pacing_counts.empty else 'Unknown',
                'distribution': pacing_counts.to_dict()
            }

        # If athlete specified, compare
        if athlete_name:
            athlete_best = event_df[event_df['FullName'] == athlete_name]['time_seconds'].min()
            if not pd.isna(athlete_best):
                rank = (best_times['time_seconds'] < athlete_best).sum() + 1
                scouting_report['athlete_position'] = {
                    'name': athlete_name,
                    'best_time': self.seconds_to_time(athlete_best),
                    'current_rank': rank,
                    'gap_to_final': athlete_best - scouting_report['time_standards']['top_8_cut'] if len(top_10) >= 8 else 0
                }

        return scouting_report


class PerformanceProjectionAgent(SwimmingAnalysisAgent):
    """Agent for performance projections and goal setting"""

    def project_performance(self, athlete_name: str, event: str, target_date: str = None) -> Dict:
        """Project future performance based on trend analysis"""
        if self.df is None:
            return {"error": "No data available"}

        athlete_df = self.df[self.df['FullName'] == athlete_name]
        disc_col = 'DisciplineName' if 'DisciplineName' in athlete_df.columns else 'discipline_name'

        if disc_col in athlete_df.columns:
            athlete_df = athlete_df[athlete_df[disc_col] == event]

        if len(athlete_df) < 3:
            return {"error": "Insufficient data for projection (need at least 3 races)"}

        athlete_df = athlete_df.copy()
        athlete_df['time_seconds'] = athlete_df['Time'].apply(self.time_to_seconds)
        athlete_df = athlete_df.dropna(subset=['time_seconds'])

        if 'date_from' in athlete_df.columns:
            athlete_df['date'] = pd.to_datetime(athlete_df['date_from'], errors='coerce')
            athlete_df = athlete_df.sort_values('date')

        # Calculate trend
        times = athlete_df['time_seconds'].values
        x = np.arange(len(times))

        # Linear regression
        slope, intercept = np.polyfit(x, times, 1)

        # Current PB
        current_pb = times.min()

        # Projected improvement (next 10 races)
        projected_times = [slope * (len(times) + i) + intercept for i in range(10)]
        projected_pb = min(current_pb, min(projected_times))

        projection = {
            'athlete': athlete_name,
            'event': event,
            'current_pb': self.seconds_to_time(current_pb),
            'races_analyzed': len(times),
            'trend': {
                'direction': 'improving' if slope < 0 else 'declining' if slope > 0 else 'stable',
                'rate_per_race': round(abs(slope), 3),
                'confidence': 'high' if len(times) >= 10 else 'medium' if len(times) >= 5 else 'low'
            },
            'projections': {
                '5_races': self.seconds_to_time(slope * (len(times) + 5) + intercept),
                '10_races': self.seconds_to_time(slope * (len(times) + 10) + intercept),
                'potential_pb': self.seconds_to_time(projected_pb)
            },
            'realistic_goals': []
        }

        # Generate realistic goals
        if slope < 0:  # Improving
            improvement_rate = abs(slope)
            projection['realistic_goals'] = [
                f"Short-term (3 months): {self.seconds_to_time(current_pb - improvement_rate * 3)}",
                f"Medium-term (6 months): {self.seconds_to_time(current_pb - improvement_rate * 6)}",
                f"Long-term (12 months): {self.seconds_to_time(current_pb - improvement_rate * 12)}"
            ]
        else:
            projection['realistic_goals'] = [
                "Focus on maintaining current level",
                "Address training or recovery factors",
                f"Target: Return to PB of {self.seconds_to_time(current_pb)}"
            ]

        return projection


class CoachInsightsAgent(SwimmingAnalysisAgent):
    """Master agent for generating comprehensive coaching insights"""

    def __init__(self, df: pd.DataFrame = None):
        super().__init__(df)
        self.profile_agent = AthleteProfileAgent(df)
        self.tactics_agent = RaceTacticsAgent(df)
        self.scout_agent = CompetitionScoutAgent(df)
        self.projection_agent = PerformanceProjectionAgent(df)

    def generate_comprehensive_report(self, athlete_name: str, event: str = None) -> Dict:
        """Generate comprehensive coaching report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'athlete': athlete_name,
            'sections': {}
        }

        # Athlete profile
        profile = self.profile_agent.analyze_athlete(athlete_name)
        report['sections']['profile'] = profile

        # Training recommendations
        recommendations = self.profile_agent.generate_training_recommendations(profile)
        report['sections']['recommendations'] = recommendations

        # Event-specific analysis
        if event:
            # Projection
            projection = self.projection_agent.project_performance(athlete_name, event)
            report['sections']['projection'] = projection

            # Scouting
            scouting = self.scout_agent.scout_competitors(event, athlete_name)
            report['sections']['competition'] = scouting

            # Race plan (using projected PB as target)
            if 'current_pb' in projection and projection.get('current_pb') != 'N/A':
                current_pb_seconds = self.time_to_seconds(projection['current_pb'])
                if current_pb_seconds:
                    target = current_pb_seconds * 0.99  # 1% improvement target
                    race_plan = self.tactics_agent.generate_race_plan(athlete_name, event, target)
                    report['sections']['race_plan'] = race_plan

        return report


def main():
    """Demo the agents"""
    from pathlib import Path

    # Load data
    data_files = list(Path(".").glob("enriched_*.csv")) + list(Path(".").glob("Results_*.csv"))

    if not data_files:
        print("No data files found")
        return

    df = pd.read_csv(data_files[0])
    print(f"Loaded {len(df)} records from {data_files[0]}")

    # Test agents
    insights = CoachInsightsAgent(df)

    # Get a sample athlete
    if 'FullName' in df.columns:
        athlete = df['FullName'].dropna().iloc[0]
        print(f"\n=== Analyzing: {athlete} ===")

        profile = insights.profile_agent.analyze_athlete(athlete)
        print(f"\nCountry: {profile.get('country')}")
        print(f"Total Races: {profile.get('total_races')}")
        print(f"Strengths: {profile.get('strengths', [])}")
        print(f"Areas for Improvement: {profile.get('areas_for_improvement', [])}")


if __name__ == "__main__":
    main()
