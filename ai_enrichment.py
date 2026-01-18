"""
AI-powered data enrichment using free OpenRouter models
"""

import requests
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_FREE_MODEL,
    FREE_MODELS
)

logger = logging.getLogger(__name__)


class AIEnricher:
    """Use free AI models for data enrichment and analysis"""

    def __init__(self, model: str = DEFAULT_FREE_MODEL):
        self.model = model
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")

    def chat_completion(self, messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
        """Send chat completion request to OpenRouter"""
        if not self.api_key:
            logger.error("API key not configured")
            return None

        url = f"{self.base_url}/chat/completions"

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://swimming-analysis.local',
            'X-Title': 'Swimming Performance Analysis'
        }

        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.3,  # Lower temperature for more factual responses
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None

    def classify_competition_tier(self, competition_name: str, official_name: str) -> str:
        """Classify competition tier (Olympics, World Championships, etc.)"""
        prompt = f"""Classify this swimming competition into one of these tiers:
- Olympics
- World Championships
- Continental Championships
- National Championships
- World Cup
- Grand Prix
- Other International
- Domestic

Competition Name: {competition_name}
Official Name: {official_name}

Respond with only the tier name, nothing else."""

        messages = [
            {"role": "system", "content": "You are a swimming competition expert. Classify competitions accurately and concisely."},
            {"role": "user", "content": prompt}
        ]

        result = self.chat_completion(messages, max_tokens=50)
        return result.strip() if result else "Other International"

    def analyze_performance_trend(self, athlete_times: List[float], dates: List[str]) -> Dict:
        """Analyze athlete performance trend over time"""
        if len(athlete_times) < 3:
            return {"trend": "insufficient_data", "analysis": "Need at least 3 performances"}

        times_str = ", ".join([f"{t:.2f}s" for t in athlete_times])
        dates_str = ", ".join(dates)

        prompt = f"""Analyze this swimmer's performance trend:

Times (chronological): {times_str}
Dates: {dates_str}

Provide a brief analysis covering:
1. Overall trend (improving/declining/stable)
2. Rate of improvement/decline
3. Notable patterns

Keep response under 100 words."""

        messages = [
            {"role": "system", "content": "You are a swimming performance analyst. Provide concise, data-driven insights."},
            {"role": "user", "content": prompt}
        ]

        analysis = self.chat_completion(messages, max_tokens=150)

        # Determine trend direction
        if len(athlete_times) >= 2:
            recent_avg = sum(athlete_times[-3:]) / min(3, len(athlete_times[-3:]))
            earlier_avg = sum(athlete_times[:3]) / min(3, len(athlete_times[:3]))

            if recent_avg < earlier_avg * 0.98:  # At least 2% improvement (lower time = better)
                trend = "improving"
            elif recent_avg > earlier_avg * 1.02:  # More than 2% slower
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return {
            "trend": trend,
            "analysis": analysis if analysis else "Unable to generate analysis",
            "latest_time": athlete_times[-1] if athlete_times else None,
            "best_time": min(athlete_times) if athlete_times else None,
            "worst_time": max(athlete_times) if athlete_times else None
        }

    def explain_split_strategy(self, lap_times: List[float], distance: int, stroke: str) -> str:
        """Explain pacing strategy based on split times"""
        if not lap_times or len(lap_times) < 2:
            return "Insufficient split data for analysis"

        laps_str = ", ".join([f"Lap {i+1}: {t:.2f}s" for i, t in enumerate(lap_times)])

        prompt = f"""Analyze this swimming race pacing strategy:

Event: {distance}m {stroke}
Split times: {laps_str}

Provide a brief tactical analysis (50 words max):
- Is this a good pacing strategy?
- Where did the swimmer gain/lose time?
- What should they focus on?"""

        messages = [
            {"role": "system", "content": "You are an elite swimming coach analyzing race tactics."},
            {"role": "user", "content": prompt}
        ]

        return self.chat_completion(messages, max_tokens=150) or "Unable to generate analysis"

    def validate_athlete_name(self, name1: str, name2: str) -> bool:
        """Check if two athlete names likely refer to the same person"""
        if not name1 or not name2:
            return False

        if name1.lower() == name2.lower():
            return True

        prompt = f"""Do these two names refer to the same person?

Name 1: {name1}
Name 2: {name2}

Respond with only 'yes' or 'no'."""

        messages = [
            {"role": "system", "content": "You are a name matching expert. Consider variations, nicknames, and cultural naming conventions."},
            {"role": "user", "content": prompt}
        ]

        result = self.chat_completion(messages, max_tokens=10)
        return result and result.strip().lower() == 'yes'

    def suggest_comparable_athletes(self, athlete_data: Dict) -> List[str]:
        """Suggest comparable athletes for benchmarking"""
        prompt = f"""Based on this swimmer's profile, suggest 3-5 comparable elite swimmers for benchmarking:

Name: {athlete_data.get('name', 'Unknown')}
Country: {athlete_data.get('country', 'Unknown')}
Best Event: {athlete_data.get('best_event', 'Unknown')}
Best Time: {athlete_data.get('best_time', 'Unknown')}
Age: {athlete_data.get('age', 'Unknown')}

List only the athlete names, one per line."""

        messages = [
            {"role": "system", "content": "You are a swimming expert with deep knowledge of elite swimmers worldwide."},
            {"role": "user", "content": prompt}
        ]

        result = self.chat_completion(messages, max_tokens=200)
        if result:
            return [name.strip() for name in result.split('\n') if name.strip()]
        return []

    def enrich_competition_batch(self, competitions_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich a batch of competitions with AI-generated classifications"""
        if competitions_df.empty:
            return competitions_df

        logger.info(f"Enriching {len(competitions_df)} competitions with AI classification")

        enriched = competitions_df.copy()
        tiers = []

        for idx, row in competitions_df.iterrows():
            name = row.get('name', '')
            official = row.get('officialName', '')

            # Use AI for ambiguous cases
            if any(keyword in official.lower() for keyword in ['olympic', 'world', 'championship', 'continental']):
                tier = self.classify_competition_tier(name, official)
            else:
                # Skip AI call for obviously lower-tier events
                tier = "Other International"

            tiers.append(tier)

            if idx % 10 == 0:  # Log progress
                logger.info(f"Processed {idx + 1}/{len(competitions_df)} competitions")

        enriched['competition_tier'] = tiers
        return enriched


class DataQualityChecker:
    """Check data quality and flag issues"""

    @staticmethod
    def validate_time(time_str: str, min_time: float = 10.0, max_time: float = 7200.0) -> Dict:
        """Validate if a race time is reasonable"""
        try:
            # Convert to seconds
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                total_seconds = minutes * 60 + seconds
            else:
                total_seconds = float(parts[0])

            valid = min_time <= total_seconds <= max_time

            return {
                'valid': valid,
                'seconds': total_seconds,
                'flag': None if valid else ('too_fast' if total_seconds < min_time else 'too_slow')
            }
        except (ValueError, IndexError):
            return {'valid': False, 'seconds': None, 'flag': 'invalid_format'}

    @staticmethod
    def check_split_consistency(splits: List[Dict], final_time: str) -> bool:
        """Check if split times are consistent with final time"""
        if not splits:
            return True  # No splits to check

        try:
            last_split = splits[-1].get('time', '')
            if not last_split:
                return True

            # Last split should match final time
            return abs(DataQualityChecker.validate_time(last_split)['seconds'] -
                      DataQualityChecker.validate_time(final_time)['seconds']) < 0.1
        except:
            return False

    @staticmethod
    def flag_anomalies(results_df: pd.DataFrame) -> pd.DataFrame:
        """Flag anomalous results in a dataframe"""
        flagged = results_df.copy()
        flags = []

        for idx, row in results_df.iterrows():
            row_flags = []

            # Check time validity
            if 'Time' in row and pd.notna(row['Time']):
                time_check = DataQualityChecker.validate_time(row['Time'])
                if not time_check['valid']:
                    row_flags.append(time_check['flag'])

            # Check for missing critical data
            if pd.isna(row.get('FullName')):
                row_flags.append('missing_athlete')

            if pd.isna(row.get('Rank')) and pd.isna(row.get('HeatRank')):
                row_flags.append('missing_rank')

            flags.append(','.join(row_flags) if row_flags else None)

        flagged['data_quality_flags'] = flags
        return flagged


def main():
    """Test AI enrichment functionality"""
    enricher = AIEnricher()

    # Test competition classification
    print("Testing competition classification...")
    tier = enricher.classify_competition_tier(
        "Paris 2024",
        "Games of the XXXIII Olympiad - Paris 2024"
    )
    print(f"Classification: {tier}")

    # Test split analysis
    print("\nTesting split analysis...")
    lap_times = [26.5, 27.2, 27.8, 28.1]
    analysis = enricher.explain_split_strategy(lap_times, 100, "Freestyle")
    print(f"Analysis: {analysis}")


if __name__ == "__main__":
    main()
