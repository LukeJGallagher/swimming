"""
Tactical Race Planning System
Generates optimal race strategies and split targets for swimmers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from enhanced_swimming_scraper import SplitTimeAnalyzer
from ai_enrichment import AIEnricher
import warnings
warnings.filterwarnings('ignore')


class TacticalRacePlanner:
    """Generate intelligent race plans with optimal split targets"""

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()
        self.ai_enricher = AIEnricher()

    def generate_race_plan(
        self,
        athlete_name: str,
        event: str,
        goal_time: str,
        competition_tier: str = "Major",
        pool_type: str = "LCM",
        strategy_preference: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive race plan with split targets

        Args:
            athlete_name: Name of the athlete
            event: Event (e.g., "100m Freestyle")
            goal_time: Target time (e.g., "47.50")
            competition_tier: "Olympics", "World Championships", "Major", "Domestic"
            pool_type: "LCM" or "SCM"
            strategy_preference: "conservative", "aggressive", or None (auto)

        Returns:
            Complete race plan with splits, strategy, confidence
        """

        # 1. Analyze athlete's historical performance
        athlete_profile = self._analyze_athlete_history(athlete_name, event)

        if not athlete_profile:
            return {"error": f"No data found for {athlete_name} in {event}"}

        # 2. Determine event characteristics
        event_info = self._parse_event(event)

        # 3. Calculate optimal split distribution
        goal_seconds = self.analyzer.time_to_seconds(goal_time)
        optimal_splits = self._calculate_optimal_splits(
            goal_seconds=goal_seconds,
            event_info=event_info,
            athlete_profile=athlete_profile,
            strategy_preference=strategy_preference
        )

        # 4. Generate alternative strategies
        alternatives = self._generate_alternative_strategies(
            goal_seconds, event_info, athlete_profile
        )

        # 5. Assess feasibility
        confidence = self._calculate_confidence(
            goal_seconds,
            athlete_profile,
            competition_tier
        )

        # 6. Get AI tactical insights
        tactical_notes = self._generate_tactical_notes(
            athlete_profile,
            optimal_splits,
            competition_tier,
            event_info
        )

        # 7. Identify improvement areas
        improvement_areas = self._identify_improvement_areas(athlete_profile)

        return {
            "athlete": athlete_name,
            "event": event,
            "goal_time": goal_time,
            "goal_seconds": goal_seconds,
            "current_pb": athlete_profile.get("pb_time"),
            "improvement_required": goal_seconds - athlete_profile.get("pb_seconds", float('inf')),
            "primary_strategy": optimal_splits,
            "alternative_strategies": alternatives,
            "confidence_assessment": confidence,
            "tactical_notes": tactical_notes,
            "improvement_areas": improvement_areas,
            "athlete_profile_summary": {
                "races_analyzed": athlete_profile["races_analyzed"],
                "pacing_preference": athlete_profile["pacing_preference"],
                "avg_first_half": athlete_profile.get("avg_first_half"),
                "avg_second_half": athlete_profile.get("avg_second_half"),
                "consistency": athlete_profile.get("consistency")
            }
        }

    def _analyze_athlete_history(self, athlete_name: str, event: str) -> Optional[Dict]:
        """Analyze athlete's historical performance in this event"""

        # Get athlete's results for this event
        athlete_data = self.results[
            (self.results['FullName'].str.contains(athlete_name, case=False, na=False)) &
            (self.results['discipline_name'].str.contains(event, case=False, na=False))
        ].copy()

        if athlete_data.empty:
            return None

        # Convert times
        athlete_data['time_seconds'] = athlete_data['Time'].apply(
            self.analyzer.time_to_seconds
        )

        # Get personal best
        pb_idx = athlete_data['time_seconds'].idxmin()
        pb_time = athlete_data.loc[pb_idx, 'Time']
        pb_seconds = athlete_data.loc[pb_idx, 'time_seconds']

        # Analyze split patterns if available
        splits_data = athlete_data[athlete_data['splits_json'].notna()]

        if not splits_data.empty:
            # Analyze pacing preferences
            pacing_dist = splits_data['pacing_type'].value_counts()
            preferred_pacing = pacing_dist.index[0] if not pacing_dist.empty else "Unknown"

            # Calculate average first/second half performance
            first_half_times = []
            second_half_times = []

            for idx, row in splits_data.iterrows():
                try:
                    splits = json.loads(row['splits_json'])
                    lap_times = self.analyzer.calculate_lap_times(splits)

                    if lap_times:
                        lap_seconds = [lt['lap_time_seconds'] for lt in lap_times]
                        midpoint = len(lap_seconds) // 2

                        first_half = sum(lap_seconds[:midpoint]) if midpoint > 0 else lap_seconds[0]
                        second_half = sum(lap_seconds[midpoint:]) if midpoint > 0 else lap_seconds[-1]

                        first_half_times.append(first_half)
                        second_half_times.append(second_half)
                except:
                    continue

            avg_first_half = np.mean(first_half_times) if first_half_times else None
            avg_second_half = np.mean(second_half_times) if second_half_times else None
            consistency = np.std(athlete_data['time_seconds'].head(10))  # Recent consistency

        else:
            preferred_pacing = "Unknown"
            avg_first_half = None
            avg_second_half = None
            consistency = None

        return {
            "pb_time": pb_time,
            "pb_seconds": pb_seconds,
            "races_analyzed": len(athlete_data),
            "pacing_preference": preferred_pacing,
            "avg_first_half": avg_first_half,
            "avg_second_half": avg_second_half,
            "consistency": consistency,
            "recent_form": athlete_data.head(5)['time_seconds'].mean()
        }

    def _parse_event(self, event: str) -> Dict:
        """Parse event string to extract distance and stroke"""
        event_lower = event.lower()

        # Extract distance
        distances = [50, 100, 200, 400, 800, 1500]
        distance = next((d for d in distances if str(d) in event_lower), 100)

        # Extract stroke
        if 'free' in event_lower:
            stroke = 'Freestyle'
        elif 'back' in event_lower:
            stroke = 'Backstroke'
        elif 'breast' in event_lower:
            stroke = 'Breaststroke'
        elif 'fly' in event_lower or 'butterfly' in event_lower:
            stroke = 'Butterfly'
        elif 'medley' in event_lower or 'im' in event_lower:
            stroke = 'IM'
        else:
            stroke = 'Unknown'

        # Determine event type
        if distance <= 100:
            event_type = 'Sprint'
        elif distance <= 400:
            event_type = 'Middle Distance'
        else:
            event_type = 'Distance'

        return {
            'distance': distance,
            'stroke': stroke,
            'type': event_type,
            'laps': distance // 50  # Assume 50m pool
        }

    def _calculate_optimal_splits(
        self,
        goal_seconds: float,
        event_info: Dict,
        athlete_profile: Dict,
        strategy_preference: Optional[str]
    ) -> Dict:
        """Calculate optimal split targets"""

        num_laps = event_info['laps']
        distance = event_info['distance']
        event_type = event_info['type']

        # Determine optimal pacing strategy
        if strategy_preference:
            strategy = strategy_preference
        else:
            # Auto-select based on athlete profile and event type
            if athlete_profile['pacing_preference'] == 'Negative Split' and event_type != 'Sprint':
                strategy = 'negative_split'
            elif event_type == 'Sprint':
                strategy = 'controlled_fade'  # Normal for sprints
            else:
                strategy = 'even_split'

        # Calculate split targets based on strategy
        splits = []

        if strategy == 'even_split':
            # Equal time per lap
            time_per_lap = goal_seconds / num_laps
            for lap in range(1, num_laps + 1):
                splits.append({
                    'lap': lap,
                    'distance': lap * 50,
                    'target_time': time_per_lap,
                    'cumulative_time': lap * time_per_lap
                })

        elif strategy == 'negative_split':
            # Faster second half
            first_half_pct = 0.505  # Slightly slower first half
            second_half_pct = 0.495  # Slightly faster second half

            midpoint = num_laps // 2

            first_half_time = goal_seconds * first_half_pct
            second_half_time = goal_seconds * second_half_pct

            first_lap_time = first_half_time / midpoint
            second_lap_time = second_half_time / (num_laps - midpoint)

            cumulative = 0
            for lap in range(1, num_laps + 1):
                if lap <= midpoint:
                    lap_time = first_lap_time
                else:
                    lap_time = second_lap_time

                cumulative += lap_time
                splits.append({
                    'lap': lap,
                    'distance': lap * 50,
                    'target_time': lap_time,
                    'cumulative_time': cumulative
                })

        else:  # controlled_fade (for sprints)
            # First 50m fast, controlled slowdown
            first_lap_pct = 0.485  # Faster opening
            remaining_pct = 0.515

            first_lap_time = goal_seconds * first_lap_pct
            remaining_time = goal_seconds * remaining_pct
            remaining_lap_time = remaining_time / (num_laps - 1)

            cumulative = 0
            for lap in range(1, num_laps + 1):
                if lap == 1:
                    lap_time = first_lap_time
                else:
                    lap_time = remaining_lap_time

                cumulative += lap_time
                splits.append({
                    'lap': lap,
                    'distance': lap * 50,
                    'target_time': lap_time,
                    'cumulative_time': cumulative
                })

        return {
            'strategy_type': strategy,
            'splits': splits,
            'total_time': goal_seconds
        }

    def _generate_alternative_strategies(
        self,
        goal_seconds: float,
        event_info: Dict,
        athlete_profile: Dict
    ) -> List[Dict]:
        """Generate 1-2 alternative race strategies"""

        alternatives = []

        # Conservative strategy (slower opening)
        conservative = self._calculate_optimal_splits(
            goal_seconds=goal_seconds + 0.3,  # Slightly slower goal
            event_info=event_info,
            athlete_profile=athlete_profile,
            strategy_preference='negative_split' if event_info['type'] != 'Sprint' else 'controlled_fade'
        )
        conservative['name'] = 'Conservative (Safe Qualifier)'
        conservative['description'] = 'Controlled opening, save energy'
        alternatives.append(conservative)

        # Aggressive strategy (faster opening)
        aggressive = self._calculate_optimal_splits(
            goal_seconds=goal_seconds - 0.2,  # Ambitious goal
            event_info=event_info,
            athlete_profile=athlete_profile,
            strategy_preference='controlled_fade'
        )
        aggressive['name'] = 'Aggressive (Championship Final)'
        aggressive['description'] = 'Fast opening, risk of fade'
        alternatives.append(aggressive)

        return alternatives

    def _calculate_confidence(
        self,
        goal_seconds: float,
        athlete_profile: Dict,
        competition_tier: str
    ) -> Dict:
        """Calculate confidence in achieving goal"""

        pb_seconds = athlete_profile['pb_seconds']
        improvement_needed = pb_seconds - goal_seconds

        # Base confidence on improvement required
        if improvement_needed >= 0:  # Goal is slower than PB
            confidence_pct = 85
            assessment = "High"
        elif improvement_needed > -0.5:
            confidence_pct = 70
            assessment = "Moderate-High"
        elif improvement_needed > -1.0:
            confidence_pct = 50
            assessment = "Moderate"
        elif improvement_needed > -2.0:
            confidence_pct = 25
            assessment = "Low-Moderate"
        else:
            confidence_pct = 10
            assessment = "Low"

        # Adjust for competition tier
        tier_adjustments = {
            "Olympics": -15,
            "World Championships": -10,
            "Major": -5,
            "Domestic": 0
        }
        confidence_pct += tier_adjustments.get(competition_tier, 0)
        confidence_pct = max(5, min(95, confidence_pct))  # Clamp between 5-95%

        # Adjust for consistency
        if athlete_profile.get('consistency') and athlete_profile['consistency'] < 0.5:
            confidence_pct += 5  # Bonus for consistency

        return {
            "confidence_percentage": round(confidence_pct),
            "assessment": assessment,
            "improvement_required": abs(improvement_needed),
            "achievability": "Highly Achievable" if confidence_pct >= 70 else
                           "Achievable" if confidence_pct >= 50 else
                           "Challenging" if confidence_pct >= 30 else
                           "Very Challenging"
        }

    def _generate_tactical_notes(
        self,
        athlete_profile: Dict,
        optimal_splits: Dict,
        competition_tier: str,
        event_info: Dict
    ) -> List[str]:
        """Generate tactical notes and recommendations"""

        notes = []

        # Pacing note
        if athlete_profile['pacing_preference'] == 'Negative Split':
            notes.append(f"✓ Your historical data shows {athlete_profile['pacing_preference']} success rate is high")
        elif athlete_profile['pacing_preference'] == 'Positive Split':
            notes.append(f"⚠️ You typically race with {athlete_profile['pacing_preference']} - be aware of fade risk")

        # Strategy alignment
        strategy_type = optimal_splits['strategy_type'].replace('_', ' ').title()
        notes.append(f"Recommended strategy: {strategy_type}")

        # Competition-specific
        if competition_tier in ['Olympics', 'World Championships']:
            notes.append(f"⚡ {competition_tier} environment - expect fast competition")

        # Event-specific advice
        if event_info['type'] == 'Sprint':
            notes.append("Sprint event: Focus on explosive start and maximum effort")
        elif event_info['type'] == 'Distance':
            notes.append("Distance event: Patience in opening laps is critical")

        # Consistency note
        if athlete_profile.get('consistency'):
            if athlete_profile['consistency'] < 0.5:
                notes.append(f"✓ High consistency (σ={athlete_profile['consistency']:.2f}s) - reliable performer")
            else:
                notes.append(f"⚠️ Consistency variance (σ={athlete_profile['consistency']:.2f}s) - focus on execution")

        return notes

    def _identify_improvement_areas(self, athlete_profile: Dict) -> List[Dict]:
        """Identify key areas for improvement"""

        areas = []

        # Check first half vs second half
        if athlete_profile.get('avg_first_half') and athlete_profile.get('avg_second_half'):
            first_half = athlete_profile['avg_first_half']
            second_half = athlete_profile['avg_second_half']

            if second_half > first_half * 1.05:  # More than 5% slower
                areas.append({
                    'area': 'Back-Half Endurance',
                    'priority': 'High',
                    'current': f'{second_half:.2f}s avg',
                    'target': f'{first_half * 1.03:.2f}s (reduce fade)',
                    'potential_improvement': f'{(second_half - first_half * 1.03):.2f}s'
                })

        # Consistency
        if athlete_profile.get('consistency') and athlete_profile['consistency'] > 1.0:
            areas.append({
                'area': 'Race Consistency',
                'priority': 'Medium',
                'current': f'{athlete_profile["consistency"]:.2f}s variance',
                'target': '<0.8s variance',
                'potential_improvement': 'More reliable performances'
            })

        # Pacing optimization
        if athlete_profile['pacing_preference'] == 'Positive Split':
            areas.append({
                'area': 'Pacing Strategy',
                'priority': 'Medium',
                'current': 'Tendency to fade in second half',
                'target': 'Even or negative split capability',
                'potential_improvement': 'Better race execution'
            })

        return areas


def format_race_plan(race_plan: Dict) -> str:
    """Format race plan as readable text"""

    if "error" in race_plan:
        return f"ERROR: {race_plan['error']}"

    output = []
    output.append("=" * 80)
    output.append(f"TACTICAL RACE PLAN: {race_plan['athlete']} - {race_plan['event']}")
    output.append("=" * 80)
    output.append("")

    # Goal and current status
    output.append(f"GOAL TIME: {race_plan['goal_time']}")
    output.append(f"CURRENT PB: {race_plan['current_pb']}")
    improvement = race_plan['improvement_required']
    if improvement < 0:
        output.append(f"IMPROVEMENT NEEDED: {abs(improvement):.2f}s FASTER ⚡")
    else:
        output.append(f"CUSHION: {improvement:.2f}s slower than PB ✓")
    output.append("")

    # Primary strategy
    output.append("━" * 80)
    output.append("PRIMARY RACE STRATEGY")
    output.append("━" * 80)
    strategy = race_plan['primary_strategy']
    output.append(f"Strategy Type: {strategy['strategy_type'].replace('_', ' ').upper()}")
    output.append("")
    output.append("TARGET SPLITS:")
    output.append(f"{'Lap':<6} {'Distance':<12} {'Lap Time':<15} {'Cumulative':<15}")
    output.append("-" * 50)

    for split in strategy['splits']:
        lap = split['lap']
        dist = f"{split['distance']}m"
        lap_time = self.analyzer.seconds_to_time(split['target_time'])
        cumulative = self.analyzer.seconds_to_time(split['cumulative_time'])
        output.append(f"{lap:<6} {dist:<12} {lap_time:<15} {cumulative:<15}")

    output.append("")

    # Confidence assessment
    output.append("━" * 80)
    output.append("CONFIDENCE ASSESSMENT")
    output.append("━" * 80)
    conf = race_plan['confidence_assessment']
    output.append(f"Confidence: {conf['confidence_percentage']}% - {conf['assessment']}")
    output.append(f"Achievability: {conf['achievability']}")
    output.append("")

    # Tactical notes
    output.append("━" * 80)
    output.append("TACTICAL NOTES")
    output.append("━" * 80)
    for note in race_plan['tactical_notes']:
        output.append(f"• {note}")
    output.append("")

    # Improvement areas
    if race_plan['improvement_areas']:
        output.append("━" * 80)
        output.append("KEY IMPROVEMENT AREAS")
        output.append("━" * 80)
        for i, area in enumerate(race_plan['improvement_areas'], 1):
            output.append(f"{i}. {area['area']} (Priority: {area['priority']})")
            output.append(f"   Current: {area['current']}")
            output.append(f"   Target: {area['target']}")
            output.append(f"   Potential: {area['potential_improvement']}")
            output.append("")

    # Alternative strategies
    output.append("━" * 80)
    output.append("ALTERNATIVE STRATEGIES")
    output.append("━" * 80)
    for alt in race_plan['alternative_strategies']:
        output.append(f"\n{alt['name']}: {alt['description']}")
        output.append(f"Target: {self.analyzer.seconds_to_time(alt['total_time'])}")

    output.append("")
    output.append("=" * 80)

    return '\n'.join(output)


# Initialize with corrected reference
TacticalRacePlanner.analyzer = SplitTimeAnalyzer()


def main():
    """Example usage"""
    import pandas as pd

    print("Tactical Race Planner - Demo")
    print("=" * 80)

    # Load data
    try:
        results = pd.read_csv('data/results_2024.csv')
        print(f"Loaded {len(results)} results from 2024")
        print("")

        # Create planner
        planner = TacticalRacePlanner(results)

        # Example: Generate race plan
        # You would replace with actual athlete name and event from your data
        print("To generate a race plan:")
        print("")
        print("race_plan = planner.generate_race_plan(")
        print("    athlete_name='Katie Ledecky',")
        print("    event='800m Freestyle',")
        print("    goal_time='8:15.00',")
        print("    competition_tier='Olympics',")
        print("    pool_type='LCM'")
        print(")")
        print("")
        print("report = format_race_plan(race_plan)")
        print("print(report)")
        print("")
        print("=" * 80)

    except FileNotFoundError:
        print("No data found. Run enhanced_swimming_scraper.py first!")


if __name__ == "__main__":
    main()
