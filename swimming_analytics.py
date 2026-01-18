"""
Swimming Analytics Module
Advanced analytics for elite swimming performance analysis
Based on World Aquatics standards and elite coaching metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


# ===== WORLD AQUATICS POINTS CALCULATOR =====
# Formula: P = 1000 * (B / T)^3
# Where B = Base time (world record), T = Swimmer's time

# Base times for 2024 (World Records as reference - update annually)
# LCM = Long Course Meters (50m pool)
BASE_TIMES_LCM_2024 = {
    'Men': {
        '50m Freestyle': 20.91,
        '100m Freestyle': 46.40,
        '200m Freestyle': 102.00,
        '400m Freestyle': 220.07,
        '800m Freestyle': 452.12,
        '1500m Freestyle': 871.02,
        '50m Backstroke': 23.71,
        '100m Backstroke': 51.60,
        '200m Backstroke': 111.92,
        '50m Breaststroke': 25.95,
        '100m Breaststroke': 56.88,
        '200m Breaststroke': 125.48,
        '50m Butterfly': 22.27,
        '100m Butterfly': 49.45,
        '200m Butterfly': 110.34,
        '200m Individual Medley': 114.00,
        '400m Individual Medley': 243.84,
    },
    'Women': {
        '50m Freestyle': 23.61,
        '100m Freestyle': 51.71,
        '200m Freestyle': 112.98,
        '400m Freestyle': 236.40,
        '800m Freestyle': 493.14,
        '1500m Freestyle': 940.88,
        '50m Backstroke': 26.98,
        '100m Backstroke': 57.33,
        '200m Backstroke': 123.35,
        '50m Breaststroke': 29.30,
        '100m Breaststroke': 64.13,
        '200m Breaststroke': 139.11,
        '50m Butterfly': 24.43,
        '100m Butterfly': 55.18,
        '200m Butterfly': 121.81,
        '200m Individual Medley': 125.03,
        '400m Individual Medley': 266.36,
    }
}


def calculate_fina_points(time_seconds: float, event: str, gender: str) -> Optional[float]:
    """
    Calculate World Aquatics (FINA) points using cubic formula
    P = 1000 * (B / T)^3

    Args:
        time_seconds: Swimmer's time in seconds
        event: Event name (e.g., '100m Freestyle')
        gender: 'Men' or 'Women'

    Returns:
        Points value (typically 0-1100, where 1000 = world record)
    """
    if pd.isna(time_seconds) or time_seconds <= 0:
        return None

    # Normalize gender
    if gender.lower() in ['m', 'male', 'men']:
        gender = 'Men'
    else:
        gender = 'Women'

    # Normalize event name
    event_normalized = normalize_event_name(event)

    base_times = BASE_TIMES_LCM_2024.get(gender, {})
    base_time = base_times.get(event_normalized)

    if not base_time:
        return None

    # FINA points formula
    points = 1000 * (base_time / time_seconds) ** 3

    return round(points, 2)


def normalize_event_name(event: str) -> str:
    """Normalize event name for lookup"""
    if not event:
        return ''

    # Remove gender prefix if present
    event = event.replace('Men ', '').replace('Women ', '')
    event = event.replace("Men's ", '').replace("Women's ", '')

    # Normalize common variations (only if not already full name)
    if 'Individual' not in event and 'Medley' in event and 'Relay' not in event:
        event = event.replace('Medley', 'Individual Medley')
    event = event.replace('IM', 'Individual Medley')

    # Only expand abbreviations if not already full word
    if 'Freestyle' not in event:
        event = event.replace('Free', 'Freestyle')
    if 'Backstroke' not in event:
        event = event.replace('Back', 'Backstroke')
    if 'Breaststroke' not in event:
        event = event.replace('Breast', 'Breaststroke')
    if 'Butterfly' not in event:
        event = event.replace('Fly', 'Butterfly')

    return event.strip()


# ===== QUALIFICATION STANDARDS =====

QUALIFICATION_STANDARDS = {
    'Olympic 2028 OQT': {  # Olympic Qualifying Time (estimated)
        'Men': {
            '50m Freestyle': 21.96,
            '100m Freestyle': 48.00,
            '200m Freestyle': 106.00,
            '400m Freestyle': 228.00,
            '800m Freestyle': 470.00,
            '1500m Freestyle': 900.00,
            '100m Backstroke': 53.50,
            '200m Backstroke': 117.00,
            '100m Breaststroke': 59.50,
            '200m Breaststroke': 130.00,
            '100m Butterfly': 51.50,
            '200m Butterfly': 114.00,
            '200m Individual Medley': 118.50,
            '400m Individual Medley': 250.00,
        },
        'Women': {
            '50m Freestyle': 24.80,
            '100m Freestyle': 53.80,
            '200m Freestyle': 117.50,
            '400m Freestyle': 246.00,
            '800m Freestyle': 512.00,
            '1500m Freestyle': 980.00,
            '100m Backstroke': 60.00,
            '200m Backstroke': 128.50,
            '100m Breaststroke': 67.50,
            '200m Breaststroke': 145.00,
            '100m Butterfly': 57.50,
            '200m Butterfly': 127.00,
            '200m Individual Medley': 130.50,
            '400m Individual Medley': 278.00,
        }
    },
    'Asian Games 2026': {  # Estimated A standards
        'Men': {
            '50m Freestyle': 22.50,
            '100m Freestyle': 49.50,
            '200m Freestyle': 110.00,
            '400m Freestyle': 235.00,
            '800m Freestyle': 485.00,
            '1500m Freestyle': 930.00,
            '100m Backstroke': 55.00,
            '200m Backstroke': 120.00,
            '100m Breaststroke': 61.00,
            '200m Breaststroke': 133.00,
            '100m Butterfly': 53.00,
            '200m Butterfly': 117.00,
            '200m Individual Medley': 121.00,
            '400m Individual Medley': 258.00,
        },
        'Women': {
            '50m Freestyle': 25.50,
            '100m Freestyle': 55.50,
            '200m Freestyle': 121.00,
            '400m Freestyle': 254.00,
            '800m Freestyle': 530.00,
            '1500m Freestyle': 1010.00,
            '100m Backstroke': 62.00,
            '200m Backstroke': 132.00,
            '100m Breaststroke': 69.00,
            '200m Breaststroke': 149.00,
            '100m Butterfly': 59.00,
            '200m Butterfly': 130.00,
            '200m Individual Medley': 134.00,
            '400m Individual Medley': 286.00,
        }
    },
    'World Championships 2025': {  # Singapore
        'Men': {
            '50m Freestyle': 22.08,
            '100m Freestyle': 48.28,
            '200m Freestyle': 107.24,
            '400m Freestyle': 229.84,
            '800m Freestyle': 473.02,
            '1500m Freestyle': 905.80,
            '100m Backstroke': 53.70,
            '200m Backstroke': 117.68,
            '100m Breaststroke': 59.56,
            '200m Breaststroke': 130.62,
            '100m Butterfly': 51.60,
            '200m Butterfly': 114.78,
            '200m Individual Medley': 118.68,
            '400m Individual Medley': 251.36,
        },
        'Women': {
            '50m Freestyle': 24.82,
            '100m Freestyle': 53.78,
            '200m Freestyle': 117.84,
            '400m Freestyle': 246.54,
            '800m Freestyle': 515.46,
            '1500m Freestyle': 983.28,
            '100m Backstroke': 59.90,
            '200m Backstroke': 129.12,
            '100m Breaststroke': 67.22,
            '200m Breaststroke': 145.62,
            '100m Butterfly': 57.68,
            '200m Butterfly': 127.50,
            '200m Individual Medley': 130.84,
            '400m Individual Medley': 279.56,
        }
    }
}


def check_qualification_status(time_seconds: float, event: str, gender: str) -> Dict:
    """
    Check qualification status against major competition standards

    Returns dict with status for each competition
    """
    results = {}
    event_norm = normalize_event_name(event)

    if gender.lower() in ['m', 'male', 'men']:
        gender = 'Men'
    else:
        gender = 'Women'

    for comp_name, standards in QUALIFICATION_STANDARDS.items():
        gender_standards = standards.get(gender, {})
        standard_time = gender_standards.get(event_norm)

        if standard_time:
            qualified = time_seconds <= standard_time
            gap = time_seconds - standard_time
            results[comp_name] = {
                'standard': standard_time,
                'qualified': qualified,
                'gap': round(gap, 2),
                'gap_percent': round((gap / standard_time) * 100, 2)
            }

    return results


# ===== PEAK PERFORMANCE AGE ANALYSIS =====

# Research-based peak performance ages by event type
PEAK_PERFORMANCE_AGES = {
    'sprint': {'men': (23, 26), 'women': (22, 25)},      # 50m, 100m
    'middle': {'men': (22, 26), 'women': (21, 25)},      # 200m, 400m
    'distance': {'men': (21, 25), 'women': (19, 24)},    # 800m, 1500m
    'im': {'men': (22, 26), 'women': (20, 24)},          # IM events
}


def get_event_category(event: str) -> str:
    """Categorize event by distance"""
    event = event.lower()
    if '50m' in event or '100m' in event:
        return 'sprint'
    elif '200m' in event or '400m' in event:
        if 'medley' in event or 'im' in event:
            return 'im'
        return 'middle'
    elif '800m' in event or '1500m' in event:
        return 'distance'
    return 'middle'


def analyze_peak_performance_potential(current_age: float, event: str, gender: str,
                                        current_best: float, df: pd.DataFrame = None) -> Dict:
    """
    Analyze athlete's potential based on peak performance age research
    """
    category = get_event_category(event)
    gender_key = 'men' if gender.lower() in ['m', 'male', 'men'] else 'women'

    peak_range = PEAK_PERFORMANCE_AGES.get(category, {}).get(gender_key, (22, 26))

    years_to_peak_start = max(0, peak_range[0] - current_age)
    years_to_peak_end = max(0, peak_range[1] - current_age)
    in_peak = peak_range[0] <= current_age <= peak_range[1]
    past_peak = current_age > peak_range[1]

    # Expected improvement rates (seconds per year)
    # Based on research: ~1-3% improvement per year for juniors
    if current_age < 16:
        expected_improvement_rate = 0.03  # 3% per year
    elif current_age < 18:
        expected_improvement_rate = 0.02  # 2% per year
    elif current_age < peak_range[0]:
        expected_improvement_rate = 0.015  # 1.5% per year
    elif in_peak:
        expected_improvement_rate = 0.005  # 0.5% per year (maintenance/small gains)
    else:
        expected_improvement_rate = -0.005  # -0.5% decline

    projected_peak_time = current_best * (1 - expected_improvement_rate * years_to_peak_start) if years_to_peak_start > 0 else current_best

    return {
        'current_age': current_age,
        'peak_range': f"{peak_range[0]}-{peak_range[1]}",
        'years_to_peak': years_to_peak_start if years_to_peak_start > 0 else 0,
        'in_peak_window': in_peak,
        'past_peak': past_peak,
        'expected_annual_improvement': f"{expected_improvement_rate*100:.1f}%",
        'projected_peak_time': round(projected_peak_time, 2),
        'improvement_potential': 'High' if current_age < 18 else ('Medium' if not past_peak else 'Low')
    }


# ===== STROKE EFFICIENCY METRICS =====

def calculate_stroke_efficiency(distance: int, time_seconds: float, stroke_count: int = None,
                                 splits: List[Dict] = None) -> Dict:
    """
    Calculate stroke efficiency metrics

    Metrics:
    - Distance Per Stroke (DPS): meters traveled per stroke
    - Stroke Index (SI): velocity x DPS
    - SWOLF: stroke count + time per 50m (lower is better)
    """
    metrics = {}

    velocity = distance / time_seconds if time_seconds > 0 else 0
    metrics['velocity_mps'] = round(velocity, 2)

    # Estimate stroke count if not provided (typical ranges)
    if stroke_count is None and splits:
        # Estimate from splits - typical 12-18 strokes per 50m for elite
        num_lengths = distance // 50
        stroke_count = num_lengths * 15  # Conservative estimate

    if stroke_count and stroke_count > 0:
        dps = distance / stroke_count
        metrics['distance_per_stroke'] = round(dps, 2)
        metrics['stroke_index'] = round(velocity * dps, 2)

        # SWOLF per 50m
        time_per_50 = (time_seconds / distance) * 50
        strokes_per_50 = (stroke_count / distance) * 50
        metrics['swolf'] = round(time_per_50 + strokes_per_50, 1)

    return metrics


# ===== RACE SEGMENT ANALYSIS =====

def analyze_race_segments(splits: List[Dict], total_time: float, distance: int) -> Dict:
    """
    Analyze race in segments: Start, Swimming, Turns, Finish

    Based on research showing distinct performance phases
    """
    if not splits or len(splits) < 2:
        return {}

    segments = {
        'start_to_15m': None,
        'clean_swimming': [],
        'turns': [],
        'finish': None,
        'underwater_estimate': None
    }

    # First split typically includes start + underwater + first length
    first_split = splits[0] if splits else {}
    first_time = float(first_split.get('time', first_split.get('Time', 0)) or 0)

    # Estimate start segment (first 15m typically takes 5.5-7s for elite)
    # Elite underwater: ~0.8-1.0 seconds per meter
    estimated_start_time = min(first_time * 0.3, 7.0)  # ~30% of first 50m or max 7s
    segments['start_to_15m'] = round(estimated_start_time, 2)

    # Calculate turn times (transition between laps)
    # Turn typically takes 0.7-1.2s at wall
    if len(splits) > 1:
        for i, split in enumerate(splits[1:], 1):
            lap_time = float(split.get('differential_time', split.get('DifferentialTime', 0)) or 0)
            if lap_time > 0:
                # Estimate turn component (~5-10% of lap time)
                turn_time = lap_time * 0.08  # 8% estimate
                segments['turns'].append(round(turn_time, 2))

                # Clean swimming is lap minus turn
                clean_time = lap_time - turn_time
                segments['clean_swimming'].append(round(clean_time, 2))

    # Last split includes finish
    if splits:
        last_split = splits[-1]
        last_lap = float(last_split.get('differential_time', last_split.get('DifferentialTime', 0)) or 0)
        # Finish typically adds 0.1-0.3s
        segments['finish'] = round(min(last_lap * 0.02, 0.3), 2)

    # Underwater percentage estimate
    num_turns = len(splits) - 1
    underwater_per_turn = 3.0  # Estimate 3s underwater per turn
    total_underwater = estimated_start_time + (num_turns * underwater_per_turn)
    segments['underwater_estimate'] = round(total_underwater, 2)
    segments['underwater_percent'] = round((total_underwater / total_time) * 100, 1) if total_time > 0 else 0

    return segments


# ===== PERFORMANCE TRAJECTORY ANALYSIS =====

def analyze_performance_trajectory(athlete_results: pd.DataFrame, event: str) -> Dict:
    """
    Analyze an athlete's performance trajectory over time

    Returns progression metrics, career best, consistency, etc.
    """
    if athlete_results.empty:
        return {}

    disc_col = 'DisciplineName' if 'DisciplineName' in athlete_results.columns else 'discipline_name'
    event_results = athlete_results[athlete_results[disc_col] == event].copy()

    if event_results.empty:
        return {}

    # Convert time to seconds
    def time_to_seconds(t):
        if pd.isna(t):
            return None
        try:
            parts = str(t).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except:
            return None

    event_results['time_seconds'] = event_results['Time'].apply(time_to_seconds)
    event_results = event_results.dropna(subset=['time_seconds'])

    if event_results.empty:
        return {}

    # Sort by date if available, otherwise by age
    if 'date_from' in event_results.columns:
        event_results['date'] = pd.to_datetime(event_results['date_from'], errors='coerce')
        event_results = event_results.sort_values('date')
    elif 'AthleteResultAge' in event_results.columns:
        event_results = event_results.sort_values('AthleteResultAge')

    times = event_results['time_seconds'].values

    # Key metrics
    career_best = times.min()
    career_worst = times.max()
    avg_time = times.mean()
    std_dev = times.std()
    consistency = 1 - (std_dev / avg_time) if avg_time > 0 else 0  # Higher = more consistent

    # Progression
    first_time = times[0]
    last_time = times[-1]
    total_improvement = first_time - career_best
    recent_trend = first_time - last_time  # Positive = improving

    # Personal bests progression
    running_pb = np.minimum.accumulate(times)
    pb_count = np.sum(np.diff(running_pb) < 0) + 1  # Count PBs

    return {
        'total_races': len(times),
        'career_best': round(career_best, 2),
        'career_worst': round(career_worst, 2),
        'average_time': round(avg_time, 2),
        'consistency_score': round(consistency * 100, 1),  # 0-100%
        'total_improvement': round(total_improvement, 2),
        'recent_trend': 'Improving' if recent_trend > 0.5 else ('Stable' if abs(recent_trend) <= 0.5 else 'Declining'),
        'pb_count': pb_count,
        'time_range': round(career_worst - career_best, 2)
    }


# ===== COMPETITOR ANALYSIS =====

def find_similar_competitors(df: pd.DataFrame, athlete_name: str, event: str,
                             time_window: float = 2.0) -> pd.DataFrame:
    """
    Find competitors with similar times (within time_window seconds)
    """
    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    # Get athlete's best time
    athlete_df = df[(df['FullName'] == athlete_name) & (df[disc_col] == event)]
    if athlete_df.empty:
        return pd.DataFrame()

    athlete_best = athlete_df['time_seconds'].min()

    # Find competitors in window
    event_df = df[df[disc_col] == event].copy()
    event_df = event_df.dropna(subset=['time_seconds'])

    # Best per athlete
    best_times = event_df.loc[event_df.groupby('FullName')['time_seconds'].idxmin()]

    # Filter to window
    similar = best_times[
        (best_times['time_seconds'] >= athlete_best - time_window) &
        (best_times['time_seconds'] <= athlete_best + time_window) &
        (best_times['FullName'] != athlete_name)
    ]

    return similar.sort_values('time_seconds')


# ===== DATA ENRICHMENT =====

def enrich_with_fina_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add FINA points to dataframe"""
    df = df.copy()

    def calc_points(row):
        time_val = row.get('time_seconds') or row.get('Time')
        if isinstance(time_val, str):
            # Convert time string to seconds
            parts = str(time_val).split(':')
            if len(parts) == 2:
                time_val = float(parts[0]) * 60 + float(parts[1])
            else:
                try:
                    time_val = float(parts[0])
                except:
                    return None

        event = row.get('DisciplineName') or row.get('discipline_name', '')
        gender = row.get('Gender', '')

        return calculate_fina_points(time_val, event, gender)

    df['fina_points'] = df.apply(calc_points, axis=1)

    return df


def generate_athlete_report(df: pd.DataFrame, athlete_name: str) -> Dict:
    """
    Generate comprehensive athlete performance report
    """
    athlete_df = df[df['FullName'] == athlete_name].copy()

    if athlete_df.empty:
        return {'error': 'Athlete not found'}

    disc_col = 'DisciplineName' if 'DisciplineName' in athlete_df.columns else 'discipline_name'

    # Basic info
    report = {
        'name': athlete_name,
        'nationality': athlete_df['NAT'].iloc[0] if 'NAT' in athlete_df.columns else 'Unknown',
        'total_results': len(athlete_df),
        'events': [],
        'medals': {},
        'qualifications': {}
    }

    # Age info
    if 'AthleteResultAge' in athlete_df.columns:
        ages = athlete_df['AthleteResultAge'].dropna()
        if not ages.empty:
            report['age_range'] = f"{int(ages.min())}-{int(ages.max())}"
            report['current_age'] = int(ages.max())

    # Medals
    if 'MedalTag' in athlete_df.columns:
        medals = athlete_df['MedalTag'].value_counts()
        report['medals'] = {
            'gold': int(medals.get('G', 0)),
            'silver': int(medals.get('S', 0)),
            'bronze': int(medals.get('B', 0))
        }

    # Per-event analysis
    for event in athlete_df[disc_col].unique():
        event_df = athlete_df[athlete_df[disc_col] == event]

        if 'time_seconds' in event_df.columns:
            best_time = event_df['time_seconds'].min()
        else:
            continue

        event_info = {
            'event': event,
            'personal_best': round(best_time, 2),
            'total_swims': len(event_df),
            'fina_points': calculate_fina_points(best_time, event, athlete_df['Gender'].iloc[0] if 'Gender' in athlete_df.columns else 'Men')
        }

        # Check qualification status
        gender = athlete_df['Gender'].iloc[0] if 'Gender' in athlete_df.columns else 'Men'
        qual_status = check_qualification_status(best_time, event, gender)
        event_info['qualifications'] = qual_status

        report['events'].append(event_info)

    return report


if __name__ == "__main__":
    # Test the module
    print("Swimming Analytics Module")
    print("=" * 50)

    # Test FINA points
    test_time = 48.0  # 100m freestyle
    points = calculate_fina_points(test_time, "Men 100m Freestyle", "Men")
    print(f"\n100m Freestyle in 48.00s = {points} FINA points")

    # Test qualification check
    qual = check_qualification_status(48.0, "100m Freestyle", "Men")
    print(f"\nQualification status:")
    for comp, status in qual.items():
        print(f"  {comp}: {'✓' if status['qualified'] else '✗'} (gap: {status['gap']:.2f}s)")

    # Test peak performance
    ppa = analyze_peak_performance_potential(19, "100m Freestyle", "Men", 50.0)
    print(f"\nPeak Performance Analysis (Age 19):")
    print(f"  Peak window: {ppa['peak_range']}")
    print(f"  Years to peak: {ppa['years_to_peak']}")
    print(f"  Projected peak time: {ppa['projected_peak_time']}")
