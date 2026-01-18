"""
Inspect actual API responses to understand split time structure
"""

import requests
import json

BASE_URL = "https://api.worldaquatics.com/fina"

def inspect_event_response(event_id):
    """Get and display raw API response for an event"""

    print(f"Fetching event {event_id}...")
    print("-" * 80)

    url = f"{BASE_URL}/events/{event_id}"
    response = requests.get(url, headers={'User-Agent': 'SwimmingAnalysis/1.0'})

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    data = response.json()

    print(f"Response keys: {list(data.keys())}")
    print("")

    # Check heats
    heats = data.get('Heats', [])
    print(f"Number of heats: {len(heats)}")
    print("")

    if heats:
        # Look at first heat
        first_heat = heats[0]
        print(f"First heat keys: {list(first_heat.keys())}")
        print(f"Heat name: {first_heat.get('Name')}")
        print("")

        # Look at first result
        results = first_heat.get('Results', [])
        print(f"Number of results in heat: {len(results)}")
        print("")

        if results:
            first_result = results[0]
            print(f"First result keys: {list(first_result.keys())}")
            print("")

            # Check splits specifically
            splits = first_result.get('Splits')
            print(f"Splits type: {type(splits)}")
            print(f"Splits value: {splits}")
            print("")

            if splits:
                print(f"Number of splits: {len(splits)}")
                if len(splits) > 0:
                    print(f"First split: {splits[0]}")
                    print(f"First split keys: {list(splits[0].keys()) if isinstance(splits[0], dict) else 'Not a dict'}")

            # Display full first result (formatted)
            print("\nFull first result:")
            print(json.dumps(first_result, indent=2)[:1000])  # First 1000 chars

    return data


def check_multiple_events():
    """Check several events from 2024 World Championships"""

    print("=" * 80)
    print("INSPECTING ACTUAL API RESPONSES - 2024 World Championships")
    print("=" * 80)
    print("")

    # These are event IDs from the 2024 World Championships (Short Course)
    # Event ID format appears to be UUID-style
    test_event_ids = [
        "12a6bafc-f1e3-4c0f-9d76-f366e17e9ecd",  # Women 50m Freestyle
        "e0d48b8d-7d98-4d7c-8e18-9f2c4a1b5d3e",  # Try another format
    ]

    for event_id in test_event_ids:
        try:
            print(f"\n{'=' * 80}")
            data = inspect_event_response(event_id)
            print("")
        except Exception as e:
            print(f"Error with event {event_id}: {e}")
            print("")


def list_competition_events(competition_id):
    """List all events in a competition to get valid event IDs"""

    print("=" * 80)
    print(f"LISTING EVENTS FOR COMPETITION {competition_id}")
    print("=" * 80)
    print("")

    url = f"{BASE_URL}/competitions/{competition_id}/events"
    response = requests.get(url, headers={'User-Agent': 'SwimmingAnalysis/1.0'})

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return []

    data = response.json()

    event_ids = []

    for sport in data.get('Sports', []):
        if sport.get('Name') == 'Swimming':
            disciplines = sport.get('DisciplineList', [])

            print(f"Found {len(disciplines)} swimming events")
            print("")

            for i, disc in enumerate(disciplines[:5], 1):  # First 5
                event_id = disc.get('Id')
                event_name = disc.get('DisciplineName')
                gender = disc.get('Gender')

                print(f"{i}. {event_name} ({gender})")
                print(f"   Event ID: {event_id}")
                event_ids.append(event_id)

    return event_ids


def main():
    # Step 1: Get actual event IDs from 2024 competition
    event_ids = list_competition_events(3433)  # 2024 World Championships

    if event_ids:
        # Step 2: Inspect first event in detail
        print("\n\n")
        print("=" * 80)
        print("DETAILED INSPECTION OF FIRST EVENT")
        print("=" * 80)
        print("")

        inspect_event_response(event_ids[0])


if __name__ == "__main__":
    main()
