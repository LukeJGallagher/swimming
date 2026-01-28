"""
Coaching Analytics Module
Evidence-based swimming performance analytics for coaching staff

Based on research from:
- Frontiers in Sports Science: Pacing strategies at World Championships 2017-2024
- PLOS ONE: Peak performance age modeling
- Frontiers in Physiology: Junior-to-senior transition studies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from enhanced_swimming_scraper import SplitTimeAnalyzer

# Azure Blob Storage support (preferred - uses Parquet)
try:
    from blob_storage import load_results as blob_load_results, _use_azure as blob_use_azure
    BLOB_AVAILABLE = True
except ImportError:
    BLOB_AVAILABLE = False

# Azure SQL support (fallback)
try:
    from azure_db import load_results as azure_load_results, _use_azure
    AZURE_SQL_AVAILABLE = True
except ImportError:
    AZURE_SQL_AVAILABLE = False

# World Records for benchmarking (as of 2024 - LCM)
WORLD_RECORDS_LCM = {
    # Men
    'Men 50m Freestyle': 20.91,
    'Men 100m Freestyle': 46.40,
    'Men 200m Freestyle': 102.00,
    'Men 400m Freestyle': 220.07,
    'Men 800m Freestyle': 452.12,
    'Men 1500m Freestyle': 871.02,
    'Men 50m Backstroke': 23.71,
    'Men 100m Backstroke': 51.60,
    'Men 200m Backstroke': 111.92,
    'Men 50m Breaststroke': 25.95,
    'Men 100m Breaststroke': 56.88,
    'Men 200m Breaststroke': 125.48,
    'Men 50m Butterfly': 22.27,
    'Men 100m Butterfly': 49.45,
    'Men 200m Butterfly': 110.34,
    'Men 200m Individual Medley': 114.00,
    'Men 400m Individual Medley': 243.84,
    # Women
    'Women 50m Freestyle': 23.61,
    'Women 100m Freestyle': 51.71,
    'Women 200m Freestyle': 112.98,
    'Women 400m Freestyle': 235.82,
    'Women 800m Freestyle': 493.04,
    'Women 1500m Freestyle': 940.34,
    'Women 50m Backstroke': 26.98,
    'Women 100m Backstroke': 57.33,
    'Women 200m Backstroke': 123.35,
    'Women 50m Breaststroke': 29.16,
    'Women 100m Breaststroke': 64.13,
    'Women 200m Breaststroke': 139.11,
    'Women 50m Butterfly': 24.43,
    'Women 100m Butterfly': 55.18,
    'Women 200m Butterfly': 121.81,
    'Women 200m Individual Medley': 126.12,
    'Women 400m Individual Medley': 266.36,
}

# World Records for benchmarking (as of 2024 - SCM / Short Course Meters)
WORLD_RECORDS_SCM = {
    # Men
    'Men 50m Freestyle': 20.16,
    'Men 100m Freestyle': 44.84,
    'Men 200m Freestyle': 99.37,
    'Men 400m Freestyle': 215.14,
    'Men 800m Freestyle': 447.60,
    'Men 1500m Freestyle': 857.90,
    'Men 50m Backstroke': 22.22,
    'Men 100m Backstroke': 48.33,
    'Men 200m Backstroke': 105.63,
    'Men 50m Breaststroke': 25.25,
    'Men 100m Breaststroke': 55.28,
    'Men 200m Breaststroke': 121.67,
    'Men 50m Butterfly': 21.32,
    'Men 100m Butterfly': 47.78,
    'Men 200m Butterfly': 106.29,
    'Men 100m Individual Medley': 49.28,
    'Men 200m Individual Medley': 109.06,
    'Men 400m Individual Medley': 236.25,
    # Women
    'Women 50m Freestyle': 22.93,
    'Women 100m Freestyle': 50.25,
    'Women 200m Freestyle': 110.31,
    'Women 400m Freestyle': 231.82,
    'Women 800m Freestyle': 481.63,
    'Women 1500m Freestyle': 920.36,
    'Women 50m Backstroke': 25.27,
    'Women 100m Backstroke': 54.56,
    'Women 200m Backstroke': 118.94,
    'Women 50m Breaststroke': 28.37,
    'Women 100m Breaststroke': 62.36,
    'Women 200m Breaststroke': 136.03,
    'Women 50m Butterfly': 24.02,
    'Women 100m Butterfly': 53.67,
    'Women 200m Butterfly': 119.38,
    'Women 100m Individual Medley': 56.51,
    'Women 200m Individual Medley': 123.19,
    'Women 400m Individual Medley': 261.29,
}


def get_world_records(course_type: str = 'lcm') -> dict:
    """Get world records for specified course type.

    Args:
        course_type: 'lcm' for Long Course (50m), 'scm' for Short Course (25m)

    Returns:
        Dictionary of world records for that course type
    """
    if course_type.lower() == 'scm':
        return WORLD_RECORDS_SCM
    return WORLD_RECORDS_LCM


# =============================================================================
# OFFICIAL ENTRY STANDARDS (Times in seconds)
# =============================================================================

# LA 2028 Olympic Qualifying Times (OQT) - Estimated based on Tokyo 2020 + Paris 2024
# Note: Official times will be published by World Aquatics closer to 2028
LA_2028_OQT = {
    # Men
    'Men 50m Freestyle': 21.96,
    'Men 100m Freestyle': 48.00,
    'Men 200m Freestyle': 106.52,
    'Men 400m Freestyle': 227.18,
    'Men 800m Freestyle': 468.49,
    'Men 1500m Freestyle': 904.66,
    'Men 100m Backstroke': 53.85,
    'Men 200m Backstroke': 117.24,
    'Men 100m Breaststroke': 59.67,
    'Men 200m Breaststroke': 130.79,
    'Men 100m Butterfly': 51.67,
    'Men 200m Butterfly': 115.06,
    'Men 200m Individual Medley': 118.87,
    'Men 400m Individual Medley': 254.26,
    # Women
    'Women 50m Freestyle': 24.77,
    'Women 100m Freestyle': 53.68,
    'Women 200m Freestyle': 117.50,
    'Women 400m Freestyle': 245.56,
    'Women 800m Freestyle': 511.99,
    'Women 1500m Freestyle': 983.20,
    'Women 100m Backstroke': 60.00,
    'Women 200m Backstroke': 129.23,
    'Women 100m Breaststroke': 67.26,
    'Women 200m Breaststroke': 145.31,
    'Women 100m Butterfly': 57.92,
    'Women 200m Butterfly': 127.88,
    'Women 200m Individual Medley': 131.80,
    'Women 400m Individual Medley': 280.87,
}

# World Championships Entry Standards (Budapest 2027 - Estimated)
WORLD_CHAMPS_2027_ENTRY = {
    # Men - "A" Standards
    'Men 50m Freestyle': 22.31,
    'Men 100m Freestyle': 48.83,
    'Men 200m Freestyle': 108.43,
    'Men 400m Freestyle': 232.00,
    'Men 800m Freestyle': 480.00,
    'Men 1500m Freestyle': 925.00,
    'Men 50m Backstroke': 25.20,
    'Men 100m Backstroke': 54.80,
    'Men 200m Backstroke': 119.50,
    'Men 50m Breaststroke': 27.50,
    'Men 100m Breaststroke': 60.60,
    'Men 200m Breaststroke': 133.00,
    'Men 50m Butterfly': 23.60,
    'Men 100m Butterfly': 52.50,
    'Men 200m Butterfly': 117.00,
    'Men 200m Individual Medley': 121.00,
    'Men 400m Individual Medley': 259.00,
    # Women - "A" Standards
    'Women 50m Freestyle': 25.10,
    'Women 100m Freestyle': 54.50,
    'Women 200m Freestyle': 119.50,
    'Women 400m Freestyle': 250.00,
    'Women 800m Freestyle': 522.00,
    'Women 1500m Freestyle': 1000.00,
    'Women 50m Backstroke': 28.50,
    'Women 100m Backstroke': 61.00,
    'Women 200m Backstroke': 131.50,
    'Women 50m Breaststroke': 31.00,
    'Women 100m Breaststroke': 68.00,
    'Women 200m Breaststroke': 148.00,
    'Women 50m Butterfly': 26.00,
    'Women 100m Butterfly': 58.80,
    'Women 200m Butterfly': 130.00,
    'Women 200m Individual Medley': 134.00,
    'Women 400m Individual Medley': 286.00,
}

# Asian Games 2026 (Nagoya) Entry Standards - Estimated
# Based on 2023 Asian Games Hangzhou medal times
ASIAN_GAMES_2026_ENTRY = {
    # Men - Gold medal pace (approximate)
    'Men 50m Freestyle': 22.00,
    'Men 100m Freestyle': 48.50,
    'Men 200m Freestyle': 107.00,
    'Men 400m Freestyle': 229.00,
    'Men 800m Freestyle': 475.00,
    'Men 1500m Freestyle': 915.00,
    'Men 50m Backstroke': 25.00,
    'Men 100m Backstroke': 54.00,
    'Men 200m Backstroke': 117.50,
    'Men 50m Breaststroke': 27.20,
    'Men 100m Breaststroke': 60.00,
    'Men 200m Breaststroke': 130.00,
    'Men 50m Butterfly': 23.40,
    'Men 100m Butterfly': 52.00,
    'Men 200m Butterfly': 115.50,
    'Men 200m Individual Medley': 119.00,
    'Men 400m Individual Medley': 255.00,
    # Women - Gold medal pace (approximate)
    'Women 50m Freestyle': 24.80,
    'Women 100m Freestyle': 54.00,
    'Women 200m Freestyle': 118.00,
    'Women 400m Freestyle': 248.00,
    'Women 800m Freestyle': 515.00,
    'Women 1500m Freestyle': 985.00,
    'Women 50m Backstroke': 28.00,
    'Women 100m Backstroke': 60.50,
    'Women 200m Backstroke': 129.00,
    'Women 50m Breaststroke': 30.80,
    'Women 100m Breaststroke': 67.50,
    'Women 200m Breaststroke': 146.00,
    'Women 50m Butterfly': 25.80,
    'Women 100m Butterfly': 58.50,
    'Women 200m Butterfly': 128.50,
    'Women 200m Individual Medley': 132.00,
    'Women 400m Individual Medley': 282.00,
}

# Asian Games Medal Pace (Bronze threshold - what you need to be in contention)
ASIAN_GAMES_2026_MEDAL = {
    # Men - Bronze medal pace
    'Men 50m Freestyle': 22.40,
    'Men 100m Freestyle': 49.20,
    'Men 200m Freestyle': 109.00,
    'Men 400m Freestyle': 233.00,
    'Men 800m Freestyle': 485.00,
    'Men 1500m Freestyle': 935.00,
    'Men 50m Backstroke': 25.50,
    'Men 100m Backstroke': 55.20,
    'Men 200m Backstroke': 120.00,
    'Men 50m Breaststroke': 27.80,
    'Men 100m Breaststroke': 61.50,
    'Men 200m Breaststroke': 133.00,
    'Men 50m Butterfly': 24.00,
    'Men 100m Butterfly': 53.20,
    'Men 200m Butterfly': 118.00,
    'Men 200m Individual Medley': 122.00,
    'Men 400m Individual Medley': 260.00,
    # Women - Bronze medal pace
    'Women 50m Freestyle': 25.30,
    'Women 100m Freestyle': 55.00,
    'Women 200m Freestyle': 120.00,
    'Women 400m Freestyle': 252.00,
    'Women 800m Freestyle': 525.00,
    'Women 1500m Freestyle': 1005.00,
    'Women 50m Backstroke': 28.60,
    'Women 100m Backstroke': 61.80,
    'Women 200m Backstroke': 132.00,
    'Women 50m Breaststroke': 31.50,
    'Women 100m Breaststroke': 69.00,
    'Women 200m Breaststroke': 150.00,
    'Women 50m Butterfly': 26.50,
    'Women 100m Butterfly': 59.80,
    'Women 200m Butterfly': 131.50,
    'Women 200m Individual Medley': 135.00,
    'Women 400m Individual Medley': 290.00,
}


def get_entry_standards(competition: str = 'olympics') -> dict:
    """Get entry standards for a competition.

    Args:
        competition: 'olympics', 'worlds', 'asian_games_gold', 'asian_games_medal'

    Returns:
        Dictionary of entry standards in seconds
    """
    standards = {
        'olympics': LA_2028_OQT,
        'worlds': WORLD_CHAMPS_2027_ENTRY,
        'asian_games_gold': ASIAN_GAMES_2026_ENTRY,
        'asian_games_medal': ASIAN_GAMES_2026_MEDAL,
    }
    return standards.get(competition.lower(), LA_2028_OQT)


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS.ss or SS.ss time string."""
    if seconds is None or seconds <= 0:
        return "N/A"
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
    return f"{seconds:.2f}"


# =============================================================================

# Peak performance ages from research
PEAK_PERFORMANCE_AGES = {
    'male': {
        'sprint': 24.5,      # 50m, 100m
        'middle': 24.0,      # 200m, 400m
        'distance': 23.5,    # 800m, 1500m
        'average': 24.2,
        'std': 2.1
    },
    'female': {
        'sprint': 23.0,
        'middle': 22.5,
        'distance': 22.0,
        'average': 22.5,
        'std': 2.4
    }
}

# Elite benchmarks from research
ELITE_BENCHMARKS = {
    'fina_points_elite': 900,           # Points threshold for elite level
    'years_to_elite': 8,                # Average years of competition to reach elite
    'peak_window_years': 2.6,           # Years within 2% of career best
    'cv_elite_threshold': 1.3,          # Coefficient of variation for lap times (%)
    'heats_to_finals_improvement': 1.2, # Expected % improvement for medalists
    'final_100m_position': 3,           # Must be top 3 in final 100m to medal
}

# Competition Level Benchmarks - "What It Takes to Win" (WR% thresholds)
# Based on historical medal-winning times as % of World Record
COMPETITION_BENCHMARKS = {
    'Olympic Games': {
        'gold': 98.5,      # Gold medalists typically swim 98.5%+ of WR
        'medal': 97.5,     # Medal requires ~97.5% of WR
        'final': 96.0,     # Making finals requires ~96% of WR
        'level': 'Elite',
        'description': 'Highest level - World Record pace required'
    },
    'World Championships': {
        'gold': 98.0,
        'medal': 97.0,
        'final': 95.5,
        'level': 'Elite',
        'description': 'World-class competition'
    },
    'World Junior Championships': {
        'gold': 94.0,
        'medal': 92.5,
        'final': 90.0,
        'level': 'Junior Elite',
        'description': 'Top junior talent worldwide'
    },
    'Asian Games': {
        'gold': 96.0,
        'medal': 94.5,
        'final': 92.0,
        'level': 'Continental',
        'description': 'Asian continental championship'
    },
    'Asian Championships': {
        'gold': 95.0,
        'medal': 93.5,
        'final': 91.0,
        'level': 'Continental',
        'description': 'Asian swimming championship'
    },
    'GCC Championships': {
        'gold': 88.0,
        'medal': 85.0,
        'final': 82.0,
        'level': 'Regional',
        'description': 'Gulf Cooperation Council regional'
    },
    'Arab Championships': {
        'gold': 90.0,
        'medal': 87.0,
        'final': 84.0,
        'level': 'Regional',
        'description': 'Arab regional championship'
    },
    'National Championships': {
        'gold': 92.0,
        'medal': 89.0,
        'final': 85.0,
        'level': 'National',
        'description': 'Top national level (varies by country)'
    }
}

# Age Group Progression Benchmarks (WR% by age - typical elite trajectory)
AGE_PROGRESSION_BENCHMARKS = {
    'male': {
        14: {'target_wr_pct': 82.0, 'elite_wr_pct': 85.0, 'phase': 'Junior Development'},
        15: {'target_wr_pct': 84.0, 'elite_wr_pct': 87.0, 'phase': 'Junior Development'},
        16: {'target_wr_pct': 86.0, 'elite_wr_pct': 89.0, 'phase': 'Junior Transition'},
        17: {'target_wr_pct': 88.0, 'elite_wr_pct': 91.0, 'phase': 'Junior Transition'},
        18: {'target_wr_pct': 90.0, 'elite_wr_pct': 93.0, 'phase': 'Senior Development'},
        19: {'target_wr_pct': 91.5, 'elite_wr_pct': 94.0, 'phase': 'Senior Development'},
        20: {'target_wr_pct': 93.0, 'elite_wr_pct': 95.0, 'phase': 'Senior Development'},
        21: {'target_wr_pct': 94.0, 'elite_wr_pct': 96.0, 'phase': 'Senior Elite'},
        22: {'target_wr_pct': 95.0, 'elite_wr_pct': 97.0, 'phase': 'Senior Elite'},
        23: {'target_wr_pct': 95.5, 'elite_wr_pct': 97.5, 'phase': 'Peak Window'},
        24: {'target_wr_pct': 96.0, 'elite_wr_pct': 98.0, 'phase': 'Peak Window'},
        25: {'target_wr_pct': 96.0, 'elite_wr_pct': 98.0, 'phase': 'Peak Window'},
        26: {'target_wr_pct': 95.5, 'elite_wr_pct': 97.5, 'phase': 'Peak Window'},
        27: {'target_wr_pct': 95.0, 'elite_wr_pct': 97.0, 'phase': 'Maintenance'},
        28: {'target_wr_pct': 94.5, 'elite_wr_pct': 96.5, 'phase': 'Maintenance'},
        29: {'target_wr_pct': 94.0, 'elite_wr_pct': 96.0, 'phase': 'Maintenance'},
        30: {'target_wr_pct': 93.5, 'elite_wr_pct': 95.5, 'phase': 'Veteran'},
    },
    'female': {
        13: {'target_wr_pct': 82.0, 'elite_wr_pct': 85.0, 'phase': 'Junior Development'},
        14: {'target_wr_pct': 84.0, 'elite_wr_pct': 87.0, 'phase': 'Junior Development'},
        15: {'target_wr_pct': 86.5, 'elite_wr_pct': 89.5, 'phase': 'Junior Transition'},
        16: {'target_wr_pct': 88.5, 'elite_wr_pct': 91.5, 'phase': 'Junior Transition'},
        17: {'target_wr_pct': 90.5, 'elite_wr_pct': 93.5, 'phase': 'Senior Development'},
        18: {'target_wr_pct': 92.0, 'elite_wr_pct': 95.0, 'phase': 'Senior Development'},
        19: {'target_wr_pct': 93.5, 'elite_wr_pct': 96.0, 'phase': 'Senior Elite'},
        20: {'target_wr_pct': 94.5, 'elite_wr_pct': 97.0, 'phase': 'Senior Elite'},
        21: {'target_wr_pct': 95.0, 'elite_wr_pct': 97.5, 'phase': 'Peak Window'},
        22: {'target_wr_pct': 95.5, 'elite_wr_pct': 98.0, 'phase': 'Peak Window'},
        23: {'target_wr_pct': 95.5, 'elite_wr_pct': 98.0, 'phase': 'Peak Window'},
        24: {'target_wr_pct': 95.0, 'elite_wr_pct': 97.5, 'phase': 'Peak Window'},
        25: {'target_wr_pct': 94.5, 'elite_wr_pct': 97.0, 'phase': 'Maintenance'},
        26: {'target_wr_pct': 94.0, 'elite_wr_pct': 96.5, 'phase': 'Maintenance'},
        27: {'target_wr_pct': 93.5, 'elite_wr_pct': 96.0, 'phase': 'Maintenance'},
        28: {'target_wr_pct': 93.0, 'elite_wr_pct': 95.5, 'phase': 'Veteran'},
    }
}

# Annual improvement expectations by phase (% improvement per year)
IMPROVEMENT_EXPECTATIONS = {
    'Junior Development': {'typical': 3.0, 'elite': 4.5, 'description': 'Rapid improvement phase'},
    'Junior Transition': {'typical': 2.0, 'elite': 3.0, 'description': 'Critical transition period'},
    'Senior Development': {'typical': 1.0, 'elite': 2.0, 'description': 'Refinement phase'},
    'Senior Elite': {'typical': 0.5, 'elite': 1.0, 'description': 'Marginal gains focus'},
    'Peak Window': {'typical': 0.2, 'elite': 0.5, 'description': 'Maintaining peak'},
    'Maintenance': {'typical': 0.0, 'elite': 0.2, 'description': 'Holding performance'},
    'Veteran': {'typical': -0.5, 'elite': 0.0, 'description': 'Managing decline'},
}


class AdvancedPacingAnalyzer:
    """
    Advanced pacing strategy classification based on research.

    Strategies:
    - U-shape: Fast start, slower middle, fast finish
    - Inverted-J: Gradual acceleration throughout
    - Fast-start-even: Quick start, maintained pace
    - Positive: Gradual slowdown
    - Negative: Gradual speedup
    - Even: Consistent pace throughout
    """

    def __init__(self):
        self.analyzer = SplitTimeAnalyzer()

    def classify_pacing_strategy(self, lap_times: List[Dict]) -> Dict:
        """
        Classify pacing strategy using research-based methodology.

        Returns detailed pacing analysis including:
        - Strategy type (U-shape, Inverted-J, etc.)
        - Velocity profile
        - Position tracking potential
        """
        if not lap_times or len(lap_times) < 2:
            return {'strategy': 'Unknown', 'confidence': 0}

        # Extract lap time values
        times = [lt['lap_time_seconds'] for lt in lap_times]
        n_laps = len(times)

        # Calculate key metrics
        first_lap = times[0]
        last_lap = times[-1]
        middle_laps = times[1:-1] if n_laps > 2 else []

        first_quarter = times[:max(1, n_laps//4)]
        last_quarter = times[-max(1, n_laps//4):]
        middle_half = times[n_laps//4:3*n_laps//4] if n_laps >= 4 else times

        avg_first_q = np.mean(first_quarter)
        avg_last_q = np.mean(last_quarter)
        avg_middle = np.mean(middle_half) if middle_half else avg_first_q

        # Calculate velocity changes
        velocity_trend = []
        for i in range(1, len(times)):
            change = times[i] - times[i-1]
            velocity_trend.append(change)

        avg_trend = np.mean(velocity_trend) if velocity_trend else 0

        # Classify strategy
        strategy = self._determine_strategy(
            first_lap, last_lap, avg_first_q, avg_last_q,
            avg_middle, times, velocity_trend
        )

        # Calculate metrics
        cv = (np.std(times) / np.mean(times)) * 100 if times else 0

        return {
            'strategy': strategy,
            'lap_times': times,
            'first_quarter_avg': round(avg_first_q, 2),
            'middle_avg': round(avg_middle, 2),
            'last_quarter_avg': round(avg_last_q, 2),
            'coefficient_of_variation': round(cv, 2),
            'is_elite_consistency': cv < ELITE_BENCHMARKS['cv_elite_threshold'],
            'velocity_trend': 'accelerating' if avg_trend < 0 else 'decelerating',
            'fastest_lap': round(min(times), 2),
            'slowest_lap': round(max(times), 2),
            'lap_range': round(max(times) - min(times), 2)
        }

    def _determine_strategy(self, first_lap, last_lap, avg_first_q, avg_last_q,
                           avg_middle, times, velocity_trend) -> str:
        """Determine pacing strategy type."""

        threshold = 0.5  # seconds

        # U-shape: Fast start AND fast finish, slower middle
        if (avg_first_q < avg_middle - threshold and
            avg_last_q < avg_middle - threshold):
            return "U-shape"

        # Inverted-J: Gradual acceleration, especially in final quarter
        if (avg_last_q < avg_first_q - threshold and
            avg_last_q < avg_middle - threshold):
            return "Inverted-J"

        # Fast-start-even: Fast first quarter, then consistent
        if (avg_first_q < avg_middle - threshold and
            abs(avg_middle - avg_last_q) < threshold):
            return "Fast-start-even"

        # Positive split: Getting progressively slower
        if avg_last_q > avg_first_q + threshold:
            return "Positive"

        # Negative split: Getting progressively faster
        if avg_last_q < avg_first_q - threshold:
            return "Negative"

        # Even: Consistent throughout
        if abs(avg_last_q - avg_first_q) <= threshold:
            return "Even"

        return "Variable"

    def analyze_race_position_tracking(self, lap_times: List[Dict],
                                       competitor_laps: List[List[Dict]] = None) -> Dict:
        """
        Analyze position through race (critical for medal prediction).
        Research shows must be top 3 in final 100m to medal.
        """
        if not lap_times:
            return {}

        times = [lt['lap_time_seconds'] for lt in lap_times]
        cumulative = np.cumsum(times)

        # Identify final 100m (last 2-4 laps depending on distance)
        n_laps = len(times)
        final_segment_laps = min(4, max(2, n_laps // 4))
        final_segment_time = sum(times[-final_segment_laps:])

        return {
            'final_segment_time': round(final_segment_time, 2),
            'final_segment_laps': final_segment_laps,
            'cumulative_times': [round(t, 2) for t in cumulative],
            'finish_strength': 'strong' if times[-1] < np.mean(times) else 'fading'
        }


class TalentDevelopmentTracker:
    """
    Track athlete development trajectories based on research benchmarks.

    Key research findings:
    - ~8 years to reach elite level (>900 FINA points)
    - Peak ages: Males 24.2±2.1, Females 22.5±2.4
    - Critical transition: Males 16-19, Females 15-18
    - None of lower-performing juniors transitioned to high-performing seniors
    """

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def calculate_competition_age(self, athlete_name: str) -> Dict:
        """
        Calculate how long an athlete has been competing.
        Research: Elite swimmers average 8 years to reach >900 FINA points.
        """
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        athlete_data = athlete_data.copy()

        # Try to parse dates, fall back to year column
        athlete_data['date_parsed'] = pd.to_datetime(
            athlete_data['date_from'], errors='coerce'
        )

        first_comp = athlete_data['date_parsed'].min()
        latest_comp = athlete_data['date_parsed'].max()

        # If dates are not available, use year column as fallback
        if pd.isna(first_comp) or pd.isna(latest_comp):
            if 'year' in athlete_data.columns:
                first_year = athlete_data['year'].min()
                latest_year = athlete_data['year'].max()
                if pd.notna(first_year) and pd.notna(latest_year):
                    competition_years = float(latest_year - first_year)
                    first_comp_str = str(int(first_year))
                    latest_comp_str = str(int(latest_year))
                else:
                    return {'error': 'No date or year data available'}
            else:
                return {'error': 'No date or year data available'}
        else:
            competition_years = (latest_comp - first_comp).days / 365.25
            first_comp_str = first_comp.strftime('%Y-%m-%d')
            latest_comp_str = latest_comp.strftime('%Y-%m-%d')

        # Estimate years to elite based on research
        years_to_elite = ELITE_BENCHMARKS['years_to_elite']
        progress_to_elite = min(1.0, competition_years / years_to_elite)

        return {
            'athlete': athlete_name,
            'first_competition': first_comp_str,
            'latest_competition': latest_comp_str,
            'competition_years': round(competition_years, 1),
            'total_competitions': athlete_data['competition_name'].nunique(),
            'total_races': len(athlete_data),
            'years_to_elite_benchmark': years_to_elite,
            'progress_to_elite_pct': round(progress_to_elite * 100, 1),
            'on_track': competition_years >= years_to_elite * 0.5
        }

    def calculate_world_record_percentage(self, athlete_name: str,
                                          event: str = None) -> List[Dict]:
        """
        Calculate performance as percentage of world record.
        Enables cross-event comparison and benchmarking.
        """
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ]

        if event:
            athlete_data = athlete_data[
                athlete_data['discipline_name'].str.contains(event, case=False, na=False)
            ]

        if athlete_data.empty:
            return []

        results = []
        for _, row in athlete_data.iterrows():
            event_name = row.get('discipline_name', '')
            time_str = row.get('Time', '')

            if not event_name or not time_str:
                continue

            time_seconds = self.analyzer.time_to_seconds(time_str)
            if time_seconds <= 0:
                continue

            # Find matching world record
            wr_seconds = None
            for wr_event, wr_time in WORLD_RECORDS_LCM.items():
                if wr_event.lower() in event_name.lower() or event_name.lower() in wr_event.lower():
                    wr_seconds = wr_time
                    break

            if wr_seconds:
                wr_percentage = (wr_seconds / time_seconds) * 100
                gap_seconds = time_seconds - wr_seconds

                results.append({
                    'event': event_name,
                    'time': time_str,
                    'time_seconds': round(time_seconds, 2),
                    'world_record': wr_seconds,
                    'wr_percentage': round(wr_percentage, 2),
                    'gap_to_wr_seconds': round(gap_seconds, 2),
                    'date': row.get('date_from', ''),
                    'competition': row.get('competition_name', ''),
                    'is_elite_level': wr_percentage >= 95  # 95%+ of WR is elite
                })

        return sorted(results, key=lambda x: x['wr_percentage'], reverse=True)

    def analyze_age_progression(self, athlete_name: str,
                                current_age: int = None) -> Dict:
        """
        Analyze athlete's progression relative to peak performance age.

        Research findings:
        - Males peak at 24.2 ± 2.1 years
        - Females peak at 22.5 ± 2.4 years
        - Peak window lasts ~2.6 years
        """
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ]

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Determine gender from event names
        sample_event = athlete_data['discipline_name'].iloc[0] if 'discipline_name' in athlete_data.columns else ''
        gender = 'female' if 'Women' in sample_event or 'W ' in sample_event else 'male'

        # Get age data if available
        ages = athlete_data['AthleteResultAge'].dropna() if 'AthleteResultAge' in athlete_data.columns else pd.Series()

        if not ages.empty:
            current_age = int(ages.iloc[-1])
            first_age = int(ages.iloc[0])
        elif current_age is None:
            return {'error': 'Age data not available'}
        else:
            first_age = current_age

        peak_info = PEAK_PERFORMANCE_AGES[gender]

        # Determine event category
        event_categories = {
            'sprint': ['50m', '100m'],
            'middle': ['200m', '400m'],
            'distance': ['800m', '1500m']
        }

        primary_events = athlete_data['discipline_name'].value_counts().head(3).index.tolist()
        event_category = 'middle'  # default
        for cat, patterns in event_categories.items():
            for event in primary_events:
                if any(p in event for p in patterns):
                    event_category = cat
                    break

        peak_age = peak_info[event_category]
        years_to_peak = peak_age - current_age

        # Development phase
        if current_age < 16:
            phase = 'Junior Development'
        elif current_age < 19:
            phase = 'Junior-to-Senior Transition (Critical)'
        elif current_age < peak_age - peak_info['std']:
            phase = 'Senior Development'
        elif current_age <= peak_age + peak_info['std']:
            phase = 'Peak Performance Window'
        else:
            phase = 'Post-Peak'

        return {
            'athlete': athlete_name,
            'gender': gender,
            'current_age': current_age,
            'first_recorded_age': first_age,
            'years_competing': current_age - first_age,
            'primary_event_category': event_category,
            'expected_peak_age': peak_age,
            'years_to_peak': round(years_to_peak, 1),
            'development_phase': phase,
            'peak_window': f'{peak_age - peak_info["std"]:.1f} - {peak_age + peak_info["std"]:.1f} years',
            'in_peak_window': abs(current_age - peak_age) <= peak_info['std']
        }

    def calculate_annual_improvement_rate(self, athlete_name: str,
                                          event: str) -> Dict:
        """
        Calculate year-over-year improvement rate.
        Research: Improvement rate decreases with age, elite swimmers
        continue improving until ~21 years old.
        """
        athlete_data = self.results[
            (self.results['FullName'].str.contains(athlete_name, case=False, na=False)) &
            (self.results['discipline_name'].str.contains(event, case=False, na=False))
        ].copy()

        if len(athlete_data) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        # Convert times and dates
        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)
        athlete_data['date_parsed'] = pd.to_datetime(athlete_data['date_from'], errors='coerce')
        athlete_data['year'] = athlete_data['date_parsed'].dt.year

        # Get best time per year
        yearly_bests = athlete_data.groupby('year').agg({
            'time_seconds': 'min',
            'Time': 'first'
        }).reset_index()

        yearly_bests = yearly_bests.sort_values('year')

        if len(yearly_bests) < 2:
            return {'error': 'Need at least 2 years of data'}

        # Calculate year-over-year improvements
        improvements = []
        for i in range(1, len(yearly_bests)):
            prev_time = yearly_bests.iloc[i-1]['time_seconds']
            curr_time = yearly_bests.iloc[i]['time_seconds']
            year = yearly_bests.iloc[i]['year']

            improvement_seconds = prev_time - curr_time
            improvement_pct = (improvement_seconds / prev_time) * 100

            improvements.append({
                'year': int(year),
                'time': yearly_bests.iloc[i]['Time'],
                'improvement_seconds': round(improvement_seconds, 2),
                'improvement_pct': round(improvement_pct, 2)
            })

        avg_improvement = np.mean([i['improvement_pct'] for i in improvements])

        return {
            'athlete': athlete_name,
            'event': event,
            'yearly_improvements': improvements,
            'average_annual_improvement_pct': round(avg_improvement, 2),
            'total_improvement_pct': round(
                ((yearly_bests.iloc[0]['time_seconds'] - yearly_bests.iloc[-1]['time_seconds']) /
                 yearly_bests.iloc[0]['time_seconds']) * 100, 2
            ),
            'years_analyzed': len(yearly_bests),
            'trajectory': 'improving' if avg_improvement > 0 else 'declining'
        }


class RaceRoundAnalyzer:
    """
    Analyze performance across competition rounds (heats, semis, finals).

    Research finding: Medalists improve 1.0-1.4% from heats to finals,
    non-medalists show minimal or negative progression.
    """

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def analyze_heats_to_finals(self, athlete_name: str,
                                 competition_name: str = None,
                                 event: str = None) -> List[Dict]:
        """
        Track performance improvement from heats to finals.
        Critical metric for competition preparation.
        """
        query = self.results['FullName'].str.contains(athlete_name, case=False, na=False)

        if competition_name:
            query &= self.results['competition_name'].str.contains(
                competition_name, case=False, na=False
            )

        if event:
            query &= self.results['discipline_name'].str.contains(
                event, case=False, na=False
            )

        athlete_data = self.results[query].copy()

        if athlete_data.empty:
            return []

        # Convert times
        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)

        # Group by competition and event
        progressions = []

        for (comp, evt), group in athlete_data.groupby(['competition_name', 'discipline_name']):
            round_times = {}

            for _, row in group.iterrows():
                heat_cat = str(row.get('heat_category', '')).lower()
                time_sec = row['time_seconds']

                if 'final' in heat_cat and 'semi' not in heat_cat:
                    round_times['final'] = time_sec
                elif 'semi' in heat_cat:
                    round_times['semi'] = time_sec
                elif 'heat' in heat_cat:
                    if 'heat' not in round_times or time_sec < round_times['heat']:
                        round_times['heat'] = time_sec

            if 'heat' in round_times and len(round_times) > 1:
                heat_time = round_times['heat']

                progression = {
                    'competition': comp,
                    'event': evt,
                    'heat_time': round(heat_time, 2),
                    'rounds_swum': list(round_times.keys())
                }

                if 'semi' in round_times:
                    semi_time = round_times['semi']
                    progression['semi_time'] = round(semi_time, 2)
                    progression['heat_to_semi_improvement'] = round(
                        ((heat_time - semi_time) / heat_time) * 100, 2
                    )

                if 'final' in round_times:
                    final_time = round_times['final']
                    progression['final_time'] = round(final_time, 2)
                    progression['heat_to_final_improvement'] = round(
                        ((heat_time - final_time) / heat_time) * 100, 2
                    )

                    # Compare to elite benchmark
                    benchmark = ELITE_BENCHMARKS['heats_to_finals_improvement']
                    progression['meets_elite_progression'] = (
                        progression['heat_to_final_improvement'] >= benchmark
                    )

                progressions.append(progression)

        return progressions

    def analyze_competition_peaking(self, athlete_name: str) -> Dict:
        """
        Analyze if athlete peaks at major competitions.
        Track PB occurrences by competition tier.
        """
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ].copy()

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Convert times
        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)

        # Track PBs by event
        pb_analysis = {}

        for event, event_data in athlete_data.groupby('discipline_name'):
            event_data = event_data.sort_values('date_from')
            event_data['is_pb'] = event_data['time_seconds'].cummin() == event_data['time_seconds']

            pb_races = event_data[event_data['is_pb']]

            if not pb_races.empty:
                pb_analysis[event] = {
                    'total_pbs': len(pb_races),
                    'pb_competitions': pb_races['competition_name'].tolist(),
                    'current_pb': round(event_data['time_seconds'].min(), 2),
                    'current_pb_time': event_data.loc[
                        event_data['time_seconds'].idxmin(), 'Time'
                    ]
                }

        return {
            'athlete': athlete_name,
            'events_analyzed': len(pb_analysis),
            'pb_analysis': pb_analysis,
            'total_pbs_achieved': sum(p['total_pbs'] for p in pb_analysis.values())
        }


class CompetitorIntelligence:
    """
    Analyze competitors for tactical race planning.
    """

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.pacing_analyzer = AdvancedPacingAnalyzer()
        self.analyzer = SplitTimeAnalyzer()

    def build_competitor_profile(self, athlete_name: str,
                                  event: str = None) -> Dict:
        """
        Build tactical profile of a competitor.
        """
        query = self.results['FullName'].str.contains(athlete_name, case=False, na=False)

        if event:
            query &= self.results['discipline_name'].str.contains(event, case=False, na=False)

        competitor_data = self.results[query].copy()

        if competitor_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Convert times
        competitor_data['time_seconds'] = competitor_data['Time'].apply(
            self.analyzer.time_to_seconds
        )

        # Pacing analysis
        pacing_types = competitor_data['pacing_type'].value_counts().to_dict() if 'pacing_type' in competitor_data.columns else {}

        # Calculate advanced pacing for races with splits
        advanced_pacing = []
        for _, row in competitor_data[competitor_data['lap_times_json'].notna()].head(10).iterrows():
            try:
                lap_times = json.loads(row['lap_times_json'])
                analysis = self.pacing_analyzer.classify_pacing_strategy(lap_times)
                advanced_pacing.append(analysis['strategy'])
            except:
                continue

        # Determine preferred strategy
        if advanced_pacing:
            pacing_counts = pd.Series(advanced_pacing).value_counts()
            preferred_strategy = pacing_counts.index[0]
        elif pacing_types:
            preferred_strategy = max(pacing_types, key=pacing_types.get)
        else:
            preferred_strategy = 'Unknown'

        # Performance stats
        best_time = competitor_data['time_seconds'].min()
        avg_time = competitor_data['time_seconds'].mean()
        consistency = competitor_data['time_seconds'].std()

        # Finishing strength
        fast_finishers = competitor_data[
            competitor_data['pacing_type'].isin(['Negative Split', 'Even'])
        ] if 'pacing_type' in competitor_data.columns else pd.DataFrame()

        finish_strength_pct = len(fast_finishers) / len(competitor_data) * 100 if len(competitor_data) > 0 else 0

        return {
            'athlete': athlete_name,
            'country': competitor_data['NAT'].iloc[0] if 'NAT' in competitor_data.columns else 'Unknown',
            'races_analyzed': len(competitor_data),
            'best_time_seconds': round(best_time, 2),
            'avg_time_seconds': round(avg_time, 2),
            'consistency_std': round(consistency, 3),
            'preferred_pacing_strategy': preferred_strategy,
            'pacing_distribution': pacing_types,
            'finish_strength_pct': round(finish_strength_pct, 1),
            'tactical_summary': self._generate_tactical_summary(
                preferred_strategy, finish_strength_pct, consistency
            )
        }

    def _generate_tactical_summary(self, strategy: str, finish_pct: float,
                                   consistency: float) -> str:
        """Generate tactical insights for race planning."""
        insights = []

        if strategy in ['Negative Split', 'Inverted-J']:
            insights.append("Strong finisher - expect late-race surge")
        elif strategy in ['Positive', 'Fast-start-even']:
            insights.append("Fast starter - vulnerable in closing stages")
        elif strategy == 'U-shape':
            insights.append("Fast start and finish - manages middle well")

        if finish_pct > 60:
            insights.append("Highly reliable finisher")
        elif finish_pct < 30:
            insights.append("Tends to fade - can be caught late")

        if consistency < 1.0:
            insights.append("Very consistent performer")
        elif consistency > 2.0:
            insights.append("Variable - could have good or bad day")

        return "; ".join(insights) if insights else "Standard competitor"

    def compare_to_field(self, target_athlete: str, event: str,
                         competition: str = None) -> Dict:
        """
        Compare target athlete to competition field.
        """
        event_data = self.results[
            self.results['discipline_name'].str.contains(event, case=False, na=False)
        ].copy()

        if competition:
            event_data = event_data[
                event_data['competition_name'].str.contains(competition, case=False, na=False)
            ]

        if event_data.empty:
            return {'error': 'No data for this event'}

        # Get target athlete's best
        target_data = event_data[
            event_data['FullName'].str.contains(target_athlete, case=False, na=False)
        ]

        if target_data.empty:
            return {'error': f'{target_athlete} not found in this event'}

        target_data['time_seconds'] = target_data['Time'].apply(self.analyzer.time_to_seconds)
        target_best = target_data['time_seconds'].min()

        # Compare to field
        event_data['time_seconds'] = event_data['Time'].apply(self.analyzer.time_to_seconds)

        # Get best times per athlete
        athlete_bests = event_data.groupby('FullName')['time_seconds'].min().sort_values()

        # Find target's position
        position = (athlete_bests < target_best).sum() + 1

        # Gap analysis
        if position > 1:
            medal_time = athlete_bests.iloc[2] if len(athlete_bests) >= 3 else athlete_bests.iloc[0]
            gap_to_medal = target_best - medal_time
        else:
            gap_to_medal = 0

        return {
            'target_athlete': target_athlete,
            'event': event,
            'target_best': round(target_best, 2),
            'field_size': len(athlete_bests),
            'current_ranking': position,
            'gap_to_medal_seconds': round(gap_to_medal, 2),
            'top_5': [
                {'athlete': name, 'time': round(time, 2)}
                for name, time in athlete_bests.head(5).items()
            ],
            'medal_potential': position <= 6 and gap_to_medal < 2.0
        }


class PredictivePerformanceModel:
    """
    Predictive performance modeling based on elite swimming analytics research.

    Key capabilities:
    - Target time prediction based on split projections
    - Performance forecasting using historical trends
    - Optimal race strategy recommendations
    """

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def predict_target_time_from_splits(self, target_splits: List[float],
                                         event: str = None) -> Dict:
        """
        Calculate predicted finish time from target split times.
        Used for race planning and pacing strategy.
        """
        if not target_splits or len(target_splits) < 2:
            return {'error': 'Need at least 2 split times'}

        total_time = sum(target_splits)
        avg_split = np.mean(target_splits)
        cv = (np.std(target_splits) / avg_split) * 100

        # Determine pacing type
        if target_splits[-1] < target_splits[0] * 0.98:
            pacing = "Negative Split (Fast Finish)"
        elif target_splits[-1] > target_splits[0] * 1.02:
            pacing = "Positive Split (Fade)"
        else:
            pacing = "Even Pace"

        result = {
            'predicted_time_seconds': round(total_time, 2),
            'predicted_time_formatted': self._seconds_to_time_str(total_time),
            'target_splits': target_splits,
            'avg_split': round(avg_split, 2),
            'split_cv': round(cv, 2),
            'pacing_strategy': pacing,
            'is_elite_consistency': cv < ELITE_BENCHMARKS['cv_elite_threshold']
        }

        # Compare to world record if event specified
        if event:
            for wr_event, wr_time in WORLD_RECORDS_LCM.items():
                if event.lower() in wr_event.lower() or wr_event.lower() in event.lower():
                    result['wr_comparison'] = round((wr_time / total_time) * 100, 2)
                    result['gap_to_wr'] = round(total_time - wr_time, 2)
                    break

        return result

    def calculate_optimal_splits(self, target_time: float, num_laps: int,
                                  strategy: str = 'even') -> Dict:
        """
        Calculate optimal split times for a target finish time.

        Strategies:
        - even: Consistent pace throughout
        - negative: Build speed (recommended for distance)
        - u_shape: Fast start, ease middle, fast finish (400m specialists)
        - front_loaded: Fast early, maintain (sprint events)
        """
        if target_time <= 0 or num_laps <= 0:
            return {'error': 'Invalid target time or lap count'}

        base_split = target_time / num_laps
        splits = []

        if strategy == 'even':
            splits = [base_split] * num_laps

        elif strategy == 'negative':
            # Start 2% slower, finish 2% faster
            adjustment = 0.02 * base_split
            for i in range(num_laps):
                factor = 1 + adjustment * (1 - 2*i/(num_laps-1))
                splits.append(base_split * (1 + factor * 0.1))
            # Normalize to hit target
            splits = [s * target_time / sum(splits) for s in splits]

        elif strategy == 'u_shape':
            # Fast start, slower middle, fast finish
            for i in range(num_laps):
                mid = num_laps / 2
                distance_from_ends = min(i, num_laps - 1 - i)
                factor = 1 + (distance_from_ends / mid) * 0.02
                splits.append(base_split * factor)
            splits = [s * target_time / sum(splits) for s in splits]

        elif strategy == 'front_loaded':
            # Fast first quarter, maintain rest
            for i in range(num_laps):
                if i < num_laps // 4:
                    splits.append(base_split * 0.98)
                else:
                    splits.append(base_split * 1.007)
            splits = [s * target_time / sum(splits) for s in splits]

        return {
            'target_time': target_time,
            'target_time_formatted': self._seconds_to_time_str(target_time),
            'strategy': strategy,
            'num_laps': num_laps,
            'optimal_splits': [round(s, 2) for s in splits],
            'cumulative_times': [round(sum(splits[:i+1]), 2) for i in range(len(splits))],
            'cv': round((np.std(splits) / np.mean(splits)) * 100, 2)
        }

    def forecast_performance(self, athlete_name: str, event: str,
                             years_ahead: int = 1) -> Dict:
        """
        Forecast future performance based on historical improvement trends.
        Uses linear regression on best times by year.
        """
        athlete_data = self.results[
            (self.results['FullName'].str.contains(athlete_name, case=False, na=False)) &
            (self.results['discipline_name'].str.contains(event, case=False, na=False))
        ].copy()

        if len(athlete_data) < 3:
            return {'error': 'Insufficient data (need at least 3 performances)'}

        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)

        # Get best time per year
        if 'year' not in athlete_data.columns:
            athlete_data['year'] = pd.to_datetime(athlete_data['date_from'], errors='coerce').dt.year

        yearly_bests = athlete_data.groupby('year')['time_seconds'].min().reset_index()
        yearly_bests = yearly_bests.dropna().sort_values('year')

        if len(yearly_bests) < 2:
            return {'error': 'Need at least 2 years of data'}

        # Simple linear regression
        years = yearly_bests['year'].values
        times = yearly_bests['time_seconds'].values

        n = len(years)
        x_mean = np.mean(years)
        y_mean = np.mean(times)

        # Calculate slope and intercept
        numerator = sum((years - x_mean) * (times - y_mean))
        denominator = sum((years - x_mean) ** 2)

        if denominator == 0:
            return {'error': 'Cannot calculate trend'}

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = slope * years + intercept
        ss_res = sum((times - y_pred) ** 2)
        ss_tot = sum((times - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Forecast
        current_year = int(yearly_bests['year'].max())
        forecast_year = current_year + years_ahead
        forecast_time = slope * forecast_year + intercept

        # Calculate confidence based on R-squared and trend direction
        if slope > 0:
            trend = 'declining'
            forecast_time = max(forecast_time, yearly_bests['time_seconds'].min())
        else:
            trend = 'improving'
            # Don't forecast unrealistic times
            current_best = yearly_bests['time_seconds'].min()
            max_improvement = current_best * 0.05  # Cap at 5% improvement
            forecast_time = max(forecast_time, current_best - max_improvement)

        return {
            'athlete': athlete_name,
            'event': event,
            'current_best': round(yearly_bests['time_seconds'].min(), 2),
            'current_best_formatted': self._seconds_to_time_str(yearly_bests['time_seconds'].min()),
            'forecast_year': forecast_year,
            'forecast_time': round(forecast_time, 2),
            'forecast_time_formatted': self._seconds_to_time_str(forecast_time),
            'improvement_per_year': round(-slope, 3),
            'trend': trend,
            'confidence_r_squared': round(r_squared, 3),
            'confidence_level': 'High' if r_squared > 0.7 else 'Medium' if r_squared > 0.4 else 'Low',
            'yearly_data': [
                {'year': int(row['year']), 'best_time': round(row['time_seconds'], 2)}
                for _, row in yearly_bests.iterrows()
            ]
        }

    def _seconds_to_time_str(self, seconds: float) -> str:
        """Convert seconds to formatted time string."""
        if seconds <= 0:
            return "0:00.00"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes > 0:
            return f"{minutes}:{secs:05.2f}"
        return f"{secs:.2f}"


class AdvancedKPIAnalyzer:
    """
    Advanced KPI analysis based on elite swimming analytics.

    Metrics from professional systems (TritonWear, Dartfish):
    - Stroke rate and stroke count
    - Distance per stroke (DPS)
    - Turn efficiency
    - Underwater performance
    - Breakout distance
    """

    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.analyzer = SplitTimeAnalyzer()

    def analyze_race_efficiency(self, athlete_name: str, event: str = None) -> Dict:
        """
        Analyze race efficiency metrics from available data.
        """
        query = self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        if event:
            query &= self.results['discipline_name'].str.contains(event, case=False, na=False)

        athlete_data = self.results[query].copy()

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        athlete_data['time_seconds'] = athlete_data['Time'].apply(self.analyzer.time_to_seconds)

        # Analyze split consistency across races
        races_with_splits = athlete_data[athlete_data['lap_times_json'].notna()]

        split_analyses = []
        for _, row in races_with_splits.iterrows():
            try:
                lap_times = json.loads(row['lap_times_json'])
                times = [lt['lap_time_seconds'] for lt in lap_times]

                if len(times) >= 2:
                    cv = (np.std(times) / np.mean(times)) * 100
                    split_analyses.append({
                        'race': row.get('competition_name', 'Unknown')[:30],
                        'total_time': row['time_seconds'],
                        'num_laps': len(times),
                        'avg_lap': np.mean(times),
                        'cv': cv,
                        'fastest_lap': min(times),
                        'slowest_lap': max(times),
                        'lap_range': max(times) - min(times)
                    })
            except:
                continue

        if not split_analyses:
            return {
                'athlete': athlete_name,
                'total_races': len(athlete_data),
                'races_with_splits': 0,
                'message': 'No split data available for detailed analysis'
            }

        avg_cv = np.mean([s['cv'] for s in split_analyses])
        avg_lap_range = np.mean([s['lap_range'] for s in split_analyses])

        return {
            'athlete': athlete_name,
            'total_races': len(athlete_data),
            'races_with_splits': len(split_analyses),
            'avg_consistency_cv': round(avg_cv, 2),
            'is_elite_consistency': avg_cv < ELITE_BENCHMARKS['cv_elite_threshold'],
            'avg_lap_range': round(avg_lap_range, 2),
            'best_performance': min(split_analyses, key=lambda x: x['total_time']),
            'most_consistent': min(split_analyses, key=lambda x: x['cv']),
            'race_analyses': sorted(split_analyses, key=lambda x: x['total_time'])[:10]
        }

    def calculate_reaction_time_stats(self, athlete_name: str) -> Dict:
        """
        Analyze reaction time performance.
        Elite reaction time: < 0.65s
        """
        athlete_data = self.results[
            self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        ].copy()

        if athlete_data.empty:
            return {'error': f'No data for {athlete_name}'}

        # Get reaction time data
        if 'RT' not in athlete_data.columns:
            return {'error': 'No reaction time data available'}

        rt_data = athlete_data['RT'].apply(self.analyzer.time_to_seconds)
        rt_data = rt_data[(rt_data > 0) & (rt_data < 1.0)]  # Filter reasonable RTs

        if rt_data.empty:
            return {'error': 'No valid reaction time data'}

        avg_rt = rt_data.mean()
        best_rt = rt_data.min()
        worst_rt = rt_data.max()

        return {
            'athlete': athlete_name,
            'avg_reaction_time': round(avg_rt, 3),
            'best_reaction_time': round(best_rt, 3),
            'worst_reaction_time': round(worst_rt, 3),
            'rt_consistency': round(rt_data.std(), 3),
            'races_analyzed': len(rt_data),
            'is_elite_rt': avg_rt < 0.65,
            'improvement_potential': round(avg_rt - 0.60, 3) if avg_rt > 0.60 else 0,
            'rating': 'Elite' if avg_rt < 0.65 else 'Good' if avg_rt < 0.70 else 'Average' if avg_rt < 0.75 else 'Needs Work'
        }

    def analyze_lane_performance(self, athlete_name: str = None, event: str = None) -> Dict:
        """
        Analyze performance by lane assignment.
        Research shows slight advantages in center lanes.
        """
        query = pd.Series([True] * len(self.results))

        if athlete_name:
            query &= self.results['FullName'].str.contains(athlete_name, case=False, na=False)
        if event:
            query &= self.results['discipline_name'].str.contains(event, case=False, na=False)

        data = self.results[query].copy()

        if 'Lane' not in data.columns:
            return {'error': 'No lane data available'}

        data['time_seconds'] = data['Time'].apply(self.analyzer.time_to_seconds)
        data = data[data['time_seconds'] > 0]
        data = data[data['Lane'].notna()]

        if data.empty:
            return {'error': 'Insufficient data'}

        # Calculate average rank by lane
        lane_stats = data.groupby('Lane').agg({
            'Rank': ['mean', 'count'],
            'time_seconds': 'mean'
        }).reset_index()
        lane_stats.columns = ['lane', 'avg_rank', 'count', 'avg_time']

        return {
            'athlete': athlete_name if athlete_name else 'All Athletes',
            'event': event if event else 'All Events',
            'lane_analysis': lane_stats.to_dict('records'),
            'best_lane': int(lane_stats.loc[lane_stats['avg_rank'].idxmin(), 'lane']),
            'races_analyzed': int(lane_stats['count'].sum())
        }


class CoachingReportGenerator:
    """
    Generate formatted reports for coaching staff.
    """

    @staticmethod
    def athlete_development_report(tracker: TalentDevelopmentTracker,
                                   athlete_name: str) -> str:
        """Generate comprehensive development report."""

        report = []
        report.append("=" * 70)
        report.append(f"ATHLETE DEVELOPMENT REPORT: {athlete_name}")
        report.append("=" * 70)

        # Competition age
        comp_age = tracker.calculate_competition_age(athlete_name)
        if 'error' not in comp_age:
            report.append(f"\nCOMPETITION HISTORY:")
            report.append(f"  First Competition: {comp_age['first_competition']}")
            report.append(f"  Years Competing: {comp_age['competition_years']}")
            report.append(f"  Progress to Elite: {comp_age['progress_to_elite_pct']}%")
            report.append(f"  Total Races: {comp_age['total_races']}")

        # World record percentage
        wr_analysis = tracker.calculate_world_record_percentage(athlete_name)
        if wr_analysis:
            report.append(f"\nWORLD RECORD BENCHMARKING (Top 5 Events):")
            for wr in wr_analysis[:5]:
                elite_marker = "*ELITE*" if wr['is_elite_level'] else ""
                report.append(
                    f"  {wr['event'][:35]:35s} {wr['wr_percentage']:>6.2f}% of WR "
                    f"(Gap: {wr['gap_to_wr_seconds']:+.2f}s) {elite_marker}"
                )

        report.append("\n" + "=" * 70)
        return '\n'.join(report)

    @staticmethod
    def race_preparation_brief(competitor_intel: CompetitorIntelligence,
                              target_athlete: str,
                              competitors: List[str],
                              event: str) -> str:
        """Generate pre-race briefing document."""

        report = []
        report.append("=" * 70)
        report.append(f"RACE PREPARATION BRIEF")
        report.append(f"Event: {event}")
        report.append(f"Athlete: {target_athlete}")
        report.append("=" * 70)

        # Target athlete profile
        target_profile = competitor_intel.build_competitor_profile(target_athlete, event)
        if 'error' not in target_profile:
            report.append(f"\nYOUR PROFILE:")
            report.append(f"  Best Time: {target_profile['best_time_seconds']}s")
            report.append(f"  Pacing: {target_profile['preferred_pacing_strategy']}")
            report.append(f"  Consistency: σ = {target_profile['consistency_std']}")

        # Competitor analysis
        report.append(f"\nCOMPETITOR ANALYSIS:")
        for comp_name in competitors[:5]:
            comp_profile = competitor_intel.build_competitor_profile(comp_name, event)
            if 'error' not in comp_profile:
                report.append(f"\n  {comp_name} ({comp_profile.get('country', 'UNK')}):")
                report.append(f"    Best: {comp_profile['best_time_seconds']}s")
                report.append(f"    Strategy: {comp_profile['preferred_pacing_strategy']}")
                report.append(f"    Tactics: {comp_profile['tactical_summary']}")

        report.append("\n" + "=" * 70)
        return '\n'.join(report)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a consistent format."""
    column_mapping = {
        'DisciplineName': 'discipline_name',
        'Heat Category': 'heat_category',
        'Gender': 'gender',
    }

    df = df.rename(columns=column_mapping)

    # Ensure key columns exist
    if 'date_from' not in df.columns and 'year' in df.columns:
        # Create a proxy date from year
        df['date_from'] = df['year'].apply(lambda x: f"{int(x)}-06-01" if pd.notna(x) else None)

    if 'competition_name' not in df.columns:
        df['competition_name'] = 'Unknown Competition'

    if 'year' not in df.columns and 'date_from' in df.columns:
        df['year'] = pd.to_datetime(df['date_from'], errors='coerce').dt.year

    return df


def load_all_results(data_dir: str = "data") -> pd.DataFrame:
    """Load all available results data from Azure Blob, Azure SQL, or CSV files."""

    # Try Azure Blob Storage first (Parquet - fastest)
    if BLOB_AVAILABLE and blob_use_azure():
        try:
            print("Loading data from Azure Blob Storage...")
            df = blob_load_results()
            if not df.empty:
                df = normalize_columns(df)
                print(f"Loaded {len(df):,} results from Azure Blob")
                return df
        except Exception as e:
            print(f"Azure Blob failed: {e}")

    # Try Azure SQL as fallback
    if AZURE_SQL_AVAILABLE and _use_azure():
        try:
            print("Loading data from Azure SQL...")
            df = azure_load_results()
            if not df.empty:
                df = normalize_columns(df)
                print(f"Loaded {len(df):,} results from Azure SQL")
                return df
        except Exception as e:
            print(f"Azure SQL failed: {e}")

    # Fall back to local CSV files
    print("Loading from local CSV files...")
    data_path = Path(data_dir)
    all_dfs = []

    # Load root directory Results files (2025, 2026)
    root_files = list(Path(".").glob("Results_*.csv"))
    for f in root_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df = normalize_columns(df)
            all_dfs.append(df)
            print(f"Loaded {len(df):,} rows from {f.name}")
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    # Load enriched files (historical 2000-2024)
    enriched_files = list(data_path.glob("enriched_*.csv"))
    for f in enriched_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df = normalize_columns(df)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if enriched_files:
        print(f"Loaded {len(enriched_files)} enriched files")

    # Fall back to regular results if no enriched
    if not enriched_files:
        result_files = list(data_path.glob("results_*.csv"))
        for f in result_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df = normalize_columns(df)
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Total: {len(combined):,} results from CSV")
        return combined

    return pd.DataFrame()


def main():
    """Demo the coaching analytics module."""
    print("=" * 70)
    print("COACHING ANALYTICS MODULE")
    print("Evidence-based swimming performance analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    results = load_all_results()

    if results.empty:
        print("No data found. Please run the scraper first.")
        return

    print(f"Loaded {len(results):,} results")
    if 'year' in results.columns:
        print(f"Year range: {int(results['year'].dropna().min())} to {int(results['year'].dropna().max())}")
    print(f"Athletes: {results['FullName'].nunique():,}")

    # Demo: Talent Development
    print("\n" + "=" * 70)
    print("DEMO: Talent Development Tracking")
    print("=" * 70)

    tracker = TalentDevelopmentTracker(results)

    # Get a sample athlete
    sample_athlete = results['FullName'].dropna().iloc[0]

    print(f"\nAnalyzing: {sample_athlete}")

    comp_age = tracker.calculate_competition_age(sample_athlete)
    print(f"Competition years: {comp_age.get('competition_years', 'N/A')}")

    wr_pct = tracker.calculate_world_record_percentage(sample_athlete)
    if wr_pct:
        print(f"Best WR%: {wr_pct[0]['wr_percentage']}% ({wr_pct[0]['event']})")

    print("\n" + "=" * 70)
    print("Module ready for use.")
    print("\nUsage examples:")
    print("  from coaching_analytics import *")
    print("  results = load_all_results()")
    print("  tracker = TalentDevelopmentTracker(results)")
    print("  pacing = AdvancedPacingAnalyzer()")
    print("  intel = CompetitorIntelligence(results)")
    print("=" * 70)


if __name__ == "__main__":
    main()
