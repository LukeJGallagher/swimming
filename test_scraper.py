"""
Test script for enhanced swimming scraper
Demonstrates split time extraction and analysis
"""

from enhanced_swimming_scraper import EnhancedSwimmingScraper, SplitTimeAnalyzer
from ai_enrichment import AIEnricher, DataQualityChecker
import json
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def test_split_extraction():
    """Test split time extraction on a small sample"""
    print("="*70)
    print("TESTING SPLIT TIME EXTRACTION")
    print("="*70)

    scraper = EnhancedSwimmingScraper(output_dir="test_data")

    # Get a single competition to test
    print("\nFetching 2024 competitions...")
    competitions = scraper.api.get_competitions(2024)

    if competitions is None or competitions.empty:
        print("No competitions found")
        return

    print(f"Found {len(competitions)} competitions")

    # Get the first competition
    comp = competitions.iloc[0]
    comp_id = comp['id']
    comp_name = comp['name']

    print(f"\nTesting with: {comp_name} (ID: {comp_id})")

    # Get events
    print("Fetching events...")
    events = scraper.api.get_competition_events(comp_id)

    if not events:
        print("No events found")
        return

    print(f"Found {len(events)} swimming events")

    # Get results for first event
    event = events[0]
    event_id = event['Id']
    discipline = event['DisciplineName']

    print(f"\nFetching results for: {discipline} (Event ID: {event_id})")
    results = scraper.api.get_event_results(event_id)

    if not results:
        print("No results found")
        return

    print(f"Found {len(results)} result entries")

    # Analyze splits
    analyzer = SplitTimeAnalyzer()
    results_with_splits = 0

    print("\n" + "="*70)
    print("SPLIT TIME ANALYSIS EXAMPLES")
    print("="*70)

    for i, result in enumerate(results[:5]):  # Show first 5 results
        athlete = result.get('FullName', 'Unknown')
        time = result.get('Time', 'N/A')
        rank = result.get('Rank', 'N/A')
        splits_raw = result.get('Splits', [])

        print(f"\n{i+1}. {athlete} ({result.get('NAT', 'N/A')})")
        print(f"   Final Time: {time} | Rank: {rank}")

        if splits_raw:
            splits = analyzer.parse_splits(splits_raw)

            if splits:
                results_with_splits += 1
                print(f"   Splits ({len(splits)} laps):")

                lap_times = analyzer.calculate_lap_times(splits)
                for lap in lap_times:
                    print(f"      Lap {lap['lap_number']} ({lap['distance']}m): "
                          f"{lap['lap_time']} (cumulative: {lap['cumulative_time']})")

                # Pacing analysis
                pacing = analyzer.analyze_pacing(lap_times)
                if pacing:
                    print(f"   Pacing Analysis:")
                    print(f"      Strategy: {pacing['pacing_type']}")
                    print(f"      First Half Avg: {pacing['first_half_avg']}s")
                    print(f"      Second Half Avg: {pacing['second_half_avg']}s")
                    print(f"      Lap Variance: {pacing['lap_variance']}")
            else:
                print("   No splits available")
        else:
            print("   No splits data")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {results_with_splits}/{min(5, len(results))} results had split times")
    print(f"{'='*70}")


def test_ai_enrichment():
    """Test AI enrichment functionality"""
    print("\n" + "="*70)
    print("TESTING AI ENRICHMENT (if API key configured)")
    print("="*70)

    enricher = AIEnricher()

    # Test competition classification
    print("\n1. Testing Competition Classification:")
    tier = enricher.classify_competition_tier(
        competition_name="World Championships",
        official_name="World Aquatics Championships Doha 2024"
    )
    print(f"   Classification: {tier}")

    # Test split strategy analysis
    print("\n2. Testing Split Strategy Analysis:")
    lap_times = [25.3, 26.1, 26.8, 27.5]  # Gradually slowing
    analysis = enricher.explain_split_strategy(
        lap_times=lap_times,
        distance=100,
        stroke="Freestyle"
    )
    print(f"   Analysis: {analysis}")


def test_data_quality():
    """Test data quality checks"""
    print("\n" + "="*70)
    print("TESTING DATA QUALITY CHECKS")
    print("="*70)

    checker = DataQualityChecker()

    # Test various times
    test_times = ["47.52", "1:45.23", "15:32.41", "5.00", "99:99.99"]

    print("\nValidating sample times:")
    for time_str in test_times:
        result = checker.validate_time(time_str)
        status = "[OK] VALID" if result['valid'] else f"[X] INVALID ({result['flag']})"
        print(f"   {time_str:>10} -> {status} ({result['seconds']}s)")


def main():
    """Run all tests"""
    print("\n")
    print("="*70)
    print(" "*15 + "ENHANCED SWIMMING SCRAPER TEST")
    print("="*70)

    try:
        # Test 1: Split extraction
        test_split_extraction()

        # Test 2: Data quality
        test_data_quality()

        # Test 3: AI enrichment (optional)
        try:
            test_ai_enrichment()
        except Exception as e:
            print(f"\nAI enrichment test skipped: {e}")
            print("(This is optional - requires OPENROUTER_API_KEY)")

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("1. Run: python enhanced_swimming_scraper.py")
        print("2. Analyze: python quick_analysis.py")
        print("3. Check output in: ./data/ directory")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
