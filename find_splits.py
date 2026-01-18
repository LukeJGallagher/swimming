"""
Quick script to find competitions with split times
"""

from split_time_inspector import SplitTimeInspector, DedicatedSplitScraper
import pandas as pd

def main():
    print("=" * 80)
    print("SEARCHING FOR COMPETITIONS WITH SPLIT TIMES")
    print("=" * 80)
    print("")

    inspector = SplitTimeInspector()

    # Test multiple competition IDs from different years
    test_competitions = [
        (3433, "2024 World Aquatics Swimming Championships (25m)"),
        (3432, "2024 Competition"),
        (1, "Historical Competition"),
        (2, "Historical Competition 2"),
        (544, "2010 Competition"),
        (547, "2010 Competition 2"),
    ]

    results = []

    for comp_id, comp_name in test_competitions:
        print(f"\nChecking: {comp_name} (ID: {comp_id})")
        print("-" * 60)

        try:
            inspection = inspector.inspect_competition_splits(comp_id, sample_size=3)

            if 'error' not in inspection:
                has_splits = inspection.get('events_with_splits', 0) > 0
                coverage = inspection.get('split_coverage_pct', 0)

                results.append({
                    'id': comp_id,
                    'name': comp_name,
                    'has_splits': has_splits,
                    'events_with_splits': inspection.get('events_with_splits', 0),
                    'total_events': inspection.get('total_events', 0),
                    'coverage_pct': coverage
                })

                status = "[SPLITS FOUND]" if has_splits else "[NO SPLITS]"
                print(f"  Status: {status}")
                print(f"  Events with splits: {inspection.get('events_with_splits', 0)}/{inspection.get('total_events', 0)}")
                print(f"  Coverage: {coverage:.1f}%")

                # Show example if found
                if inspection.get('split_examples'):
                    example = inspection['split_examples'][0]
                    print(f"  Example: {example['event']}")
                    ex_data = example['example']
                    print(f"    Athlete: {ex_data.get('athlete')}")
                    print(f"    Splits: {len(ex_data.get('splits', []))} laps")
            else:
                print(f"  Error: {inspection['error']}")

        except Exception as e:
            print(f"  Exception: {e}")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        df = pd.DataFrame(results)
        print("\nCompetitions with splits:")
        print(df[df['has_splits'] == True])

        print("\nCompetitions without splits:")
        print(df[df['has_splits'] == False])
    else:
        print("No results collected")

if __name__ == "__main__":
    main()
