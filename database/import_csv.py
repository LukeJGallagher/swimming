"""
CSV to SQL Import Script
Imports swimming results from CSV files into the database
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import (
    create_database, get_session, initialize_reference_data,
    Athlete, Competition, Event, Result
)


def time_to_seconds(time_str) -> float:
    """Convert time string to seconds."""
    if pd.isna(time_str) or time_str == '' or time_str is None:
        return None

    try:
        # Already a number
        if isinstance(time_str, (int, float)):
            return float(time_str)

        time_str = str(time_str).strip()

        # Format: MM:SS.ms
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

        # Just seconds
        return float(time_str)

    except (ValueError, TypeError):
        return None


def parse_discipline(discipline_name: str) -> dict:
    """Parse discipline name into components."""
    if not discipline_name or pd.isna(discipline_name):
        return {'gender': None, 'distance': None, 'stroke': None}

    # Pattern: "Men/Women XXXm Stroke"
    match = re.match(r'(Men|Women|Mixed)?\s*(\d+)m?\s*(.*)', str(discipline_name), re.IGNORECASE)

    if match:
        gender = match.group(1)
        distance = int(match.group(2)) if match.group(2) else None
        stroke = match.group(3).strip() if match.group(3) else None
        return {'gender': gender, 'distance': distance, 'stroke': stroke}

    return {'gender': None, 'distance': None, 'stroke': None}


# In-memory caches for faster lookups
_athlete_cache = {}
_competition_cache = {}
_event_cache = {}
_result_ids_seen = set()


def clear_caches():
    """Clear all caches - call before fresh import."""
    global _athlete_cache, _competition_cache, _event_cache, _result_ids_seen
    _athlete_cache = {}
    _competition_cache = {}
    _event_cache = {}
    _result_ids_seen = set()


def get_or_create_athlete(session, row) -> int:
    """Get existing athlete or create new one."""
    person_id = row.get('PersonId')

    # Check cache first
    if person_id and not pd.isna(person_id):
        cache_key = str(person_id)
        if cache_key in _athlete_cache:
            return _athlete_cache[cache_key]

        athlete = session.query(Athlete).filter_by(person_id=cache_key).first()
        if athlete:
            _athlete_cache[cache_key] = athlete.id
            return athlete.id

    # Create new athlete
    athlete = Athlete(
        person_id=str(person_id) if person_id and not pd.isna(person_id) else None,
        full_name=str(row.get('FullName', 'Unknown')),
        first_name=str(row.get('FirstName', '')) if row.get('FirstName') and not pd.isna(row.get('FirstName')) else None,
        last_name=str(row.get('LastName', '')) if row.get('LastName') and not pd.isna(row.get('LastName')) else None,
        nationality=str(row.get('NAT', '')) if row.get('NAT') and not pd.isna(row.get('NAT')) else None,
        nationality_name=str(row.get('NATName', '')) if row.get('NATName') and not pd.isna(row.get('NATName')) else None,
    )

    session.add(athlete)
    session.flush()

    # Cache the new athlete
    if person_id and not pd.isna(person_id):
        _athlete_cache[str(person_id)] = athlete.id

    return athlete.id


def get_or_create_competition(session, row) -> int:
    """Get existing competition or create new one."""
    comp_id = row.get('competition_id')

    # Check cache first
    if comp_id and not pd.isna(comp_id):
        cache_key = int(comp_id)
        if cache_key in _competition_cache:
            return _competition_cache[cache_key]

        competition = session.query(Competition).filter_by(competition_id=cache_key).first()
        if competition:
            _competition_cache[cache_key] = competition.id
            return competition.id

    # Parse date
    date_from = None
    if row.get('date_from') and not pd.isna(row.get('date_from')):
        try:
            date_from = pd.to_datetime(row['date_from']).date()
        except:
            pass

    # Create new competition
    competition = Competition(
        competition_id=int(comp_id) if comp_id and not pd.isna(comp_id) else None,
        competition_name=str(row.get('competition_name', 'Unknown Competition')),
        date_from=date_from,
        year=int(row.get('year')) if row.get('year') and not pd.isna(row.get('year')) else None,
    )

    session.add(competition)
    session.flush()

    # Cache the new competition
    if comp_id and not pd.isna(comp_id):
        _competition_cache[int(comp_id)] = competition.id

    return competition.id


def get_or_create_event(session, row) -> int:
    """Get existing event or create new one."""
    event_id = row.get('event_id')

    # Check cache first
    if event_id and not pd.isna(event_id):
        cache_key = str(event_id)
        if cache_key in _event_cache:
            return _event_cache[cache_key]

        event = session.query(Event).filter_by(event_id=cache_key).first()
        if event:
            _event_cache[cache_key] = event.id
            return event.id

    # Parse discipline
    discipline_name = row.get('discipline_name', '')
    parsed = parse_discipline(discipline_name)

    # Create new event
    event = Event(
        event_id=str(event_id) if event_id and not pd.isna(event_id) else None,
        discipline_name=str(discipline_name) if discipline_name and not pd.isna(discipline_name) else 'Unknown',
        gender=parsed['gender'],
        distance=parsed['distance'],
        stroke=parsed['stroke'],
    )

    session.add(event)
    session.flush()

    # Cache the new event
    if event_id and not pd.isna(event_id):
        _event_cache[str(event_id)] = event.id

    return event.id


def import_results_from_csv(csv_path: str, session, batch_size: int = 1000) -> int:
    """Import results from a CSV file."""
    print(f"Importing {csv_path}...")

    df = pd.read_csv(csv_path, low_memory=False)
    total_rows = len(df)
    imported = 0
    skipped = 0

    for i, row in df.iterrows():
        try:
            # Check if result already exists (using in-memory cache first)
            result_id = row.get('ResultId')
            if result_id and not pd.isna(result_id):
                result_key = str(result_id)
                if result_key in _result_ids_seen:
                    skipped += 1
                    continue
                existing = session.query(Result).filter_by(result_id=result_key).first()
                if existing:
                    _result_ids_seen.add(result_key)
                    skipped += 1
                    continue

            # Get or create related entities
            athlete_id = get_or_create_athlete(session, row)
            competition_id = get_or_create_competition(session, row)
            event_id = get_or_create_event(session, row)

            # Parse JSON fields
            splits_json = None
            if row.get('splits_json') and not pd.isna(row.get('splits_json')):
                try:
                    splits_json = json.loads(row['splits_json'])
                except:
                    splits_json = row['splits_json']

            lap_times_json = None
            if row.get('lap_times_json') and not pd.isna(row.get('lap_times_json')):
                try:
                    lap_times_json = json.loads(row['lap_times_json'])
                except:
                    lap_times_json = row['lap_times_json']

            # Create result
            result = Result(
                result_id=str(result_id) if result_id and not pd.isna(result_id) else None,
                athlete_id=athlete_id,
                competition_id=competition_id,
                event_id=event_id,
                heat_category=str(row.get('heat_category', '')) if row.get('heat_category') and not pd.isna(row.get('heat_category')) else None,
                heat_rank=int(row['HeatRank']) if row.get('HeatRank') and not pd.isna(row.get('HeatRank')) else None,
                final_rank=int(row['Rank']) if row.get('Rank') and not pd.isna(row.get('Rank')) else None,
                lane=int(row['Lane']) if row.get('Lane') and not pd.isna(row.get('Lane')) else None,
                time_raw=str(row.get('Time', '')) if row.get('Time') and not pd.isna(row.get('Time')) else None,
                time_seconds=time_to_seconds(row.get('Time')),
                reaction_time=time_to_seconds(row.get('RT')),
                time_behind=time_to_seconds(row.get('TimeBehind')),
                fina_points=int(row['Points']) if row.get('Points') and not pd.isna(row.get('Points')) else None,
                medal_tag=str(row.get('MedalTag', ''))[:1] if row.get('MedalTag') and not pd.isna(row.get('MedalTag')) else None,
                qualified=str(row.get('Qualified', '')) if row.get('Qualified') and not pd.isna(row.get('Qualified')) else None,
                record_type=str(row.get('RecordType', '')) if row.get('RecordType') and not pd.isna(row.get('RecordType')) else None,
                splits_json=splits_json,
                lap_times_json=lap_times_json,
                pacing_type=str(row.get('pacing_type', '')) if row.get('pacing_type') and not pd.isna(row.get('pacing_type')) else None,
                first_half_avg=time_to_seconds(row.get('first_half_avg')),
                second_half_avg=time_to_seconds(row.get('second_half_avg')),
                split_difference=time_to_seconds(row.get('split_difference')),
                fastest_lap=time_to_seconds(row.get('fastest_lap')),
                slowest_lap=time_to_seconds(row.get('slowest_lap')),
                lap_variance=time_to_seconds(row.get('lap_variance')),
                athlete_age=int(row['AthleteResultAge']) if row.get('AthleteResultAge') and not pd.isna(row.get('AthleteResultAge')) else None,
                year=int(row['year']) if row.get('year') and not pd.isna(row.get('year')) else None,
            )

            session.add(result)
            imported += 1

            # Cache the result ID
            if result_id and not pd.isna(result_id):
                _result_ids_seen.add(str(result_id))

            # Commit in batches
            if imported % batch_size == 0:
                session.commit()
                print(f"  Imported {imported:,} / {total_rows:,} ({imported/total_rows*100:.1f}%)")

        except Exception as e:
            print(f"  Error on row {i}: {e}")
            session.rollback()
            continue

    # Final commit
    session.commit()
    print(f"  Completed: {imported:,} imported, {skipped:,} skipped")

    return imported


def import_all_csv_files(data_dir: str = ".", session=None, clear_cache: bool = True) -> dict:
    """Import all CSV files from directory."""
    if clear_cache:
        clear_caches()

    if session is None:
        engine = create_database()
        session = get_session(engine)
        initialize_reference_data(session)

    data_path = Path(data_dir)
    stats = {'files': 0, 'records': 0}

    # Find all Results files
    csv_files = sorted(data_path.glob("Results_*.csv"))

    for csv_file in csv_files:
        imported = import_results_from_csv(str(csv_file), session)
        stats['files'] += 1
        stats['records'] += imported

    return stats


def sync_new_data(csv_path: str, db_url: str = None) -> dict:
    """
    Sync new scraped data to the database.
    This is the main function to call after scraping new results.

    Usage:
        from database import sync_new_data
        stats = sync_new_data('Results_2026.csv')
    """
    import os
    if db_url:
        os.environ['DATABASE_URL'] = db_url

    engine = create_database()
    session = get_session(engine)

    # Don't clear caches - we want to use existing data
    imported = import_results_from_csv(csv_path, session)

    stats = {
        'file': csv_path,
        'imported': imported,
        'athletes': session.query(Athlete).count(),
        'results': session.query(Result).count(),
    }

    session.close()
    return stats


def main():
    """Main import function."""
    import argparse

    parser = argparse.ArgumentParser(description='Import swimming results to database')
    parser.add_argument('--file', type=str, help='Specific CSV file to import')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing CSV files')
    parser.add_argument('--db', type=str, default='sqlite:///swimming_performance.db',
                       help='Database URL')
    args = parser.parse_args()

    print("=" * 60)
    print("Swimming Results Database Import")
    print("=" * 60)

    # Create database
    print(f"\nConnecting to: {args.db}")
    from database.models import create_database, get_session, initialize_reference_data
    import os
    os.environ['DATABASE_URL'] = args.db

    engine = create_database(args.db)
    session = get_session(engine)

    print("Initializing reference data...")
    initialize_reference_data(session)

    # Import
    if args.file:
        print(f"\nImporting single file: {args.file}")
        imported = import_results_from_csv(args.file, session)
        print(f"\nTotal imported: {imported:,}")
    else:
        print(f"\nImporting all files from: {args.dir}")
        stats = import_all_csv_files(args.dir, session)
        print(f"\nImport complete!")
        print(f"  Files processed: {stats['files']}")
        print(f"  Records imported: {stats['records']:,}")

    # Summary
    athlete_count = session.query(Athlete).count()
    competition_count = session.query(Competition).count()
    event_count = session.query(Event).count()
    result_count = session.query(Result).count()

    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    print(f"  Athletes:     {athlete_count:,}")
    print(f"  Competitions: {competition_count:,}")
    print(f"  Events:       {event_count:,}")
    print(f"  Results:      {result_count:,}")
    print("=" * 60)

    session.close()


if __name__ == "__main__":
    main()
