"""
Migrate Swimming CSV Data to Azure SQL
Uploads all Results_YYYY.csv files to Azure SQL database

Usage:
    python migrate_to_azure.py              # Migrate all years
    python migrate_to_azure.py 2024 2025    # Migrate specific years
    python migrate_to_azure.py --clear      # Clear database and re-migrate all
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_connection(max_retries=3):
    """Get Azure SQL connection with retry for serverless wake-up."""
    import pyodbc

    conn_str = os.getenv('AZURE_SQL_CONN')
    if not conn_str:
        raise ValueError("AZURE_SQL_CONN not found in .env file")

    # Handle driver compatibility
    conn_str = conn_str.replace('ODBC Driver 17 for SQL Server', 'SQL Server')
    conn_str = conn_str.replace('ODBC Driver 18 for SQL Server', 'SQL Server')

    for attempt in range(max_retries):
        try:
            return pyodbc.connect(conn_str, timeout=60)
        except pyodbc.Error as e:
            if 'not currently available' in str(e) and attempt < max_retries - 1:
                print(f"  Database paused, waking up... (attempt {attempt + 1}/{max_retries})")
                time.sleep(20)
            else:
                raise


def time_to_seconds(time_str):
    """Convert time string (MM:SS.ss or SS.ss) to seconds."""
    if pd.isna(time_str) or not time_str:
        return None
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return None


def prepare_dataframe(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prepare DataFrame for Azure SQL insertion."""

    # Standardize column names to match Azure schema
    # Handle both old format (2000-2024) and new format (2025+)
    column_mapping = {
        # Old format columns
        'Heat Category': 'heat_category',
        'DisciplineName': 'discipline_name',
        'Gender': 'gender',
        'event_id': 'event_id',
        'FullName': 'full_name',
        'FirstName': 'first_name',
        'LastName': 'last_name',
        'NAT': 'nationality',
        'NATName': 'nationality_name',
        'PersonId': 'person_id',
        'BiographyId': 'biography_id',
        'AthleteResultAge': 'athlete_age',
        'ResultId': 'result_id',
        'Lane': 'lane',
        'HeatRank': 'heat_rank',
        'Rank': 'final_rank',
        'Time': 'time_raw',
        'RT': 'reaction_time',
        'TimeBehind': 'time_behind',
        'Points': 'fina_points',
        'MedalTag': 'medal_tag',
        'Qualified': 'qualified',
        'RecordType': 'record_type',
        'Splits': 'splits_json',
    }

    # Handle 2025+ format: if splits_json already exists, don't rename Splits
    if 'splits_json' in df.columns and 'Splits' in column_mapping:
        del column_mapping['Splits']

    df = df.rename(columns=column_mapping)

    # Add year column
    df['year'] = year

    # Convert time to seconds
    if 'time_raw' in df.columns:
        df['time_seconds'] = df['time_raw'].apply(time_to_seconds)

    # Azure SQL schema columns
    schema_columns = [
        'heat_category', 'discipline_name', 'gender', 'event_id',
        'full_name', 'first_name', 'last_name', 'nationality', 'nationality_name',
        'person_id', 'biography_id', 'athlete_age', 'result_id',
        'lane', 'heat_rank', 'final_rank', 'time_raw', 'time_seconds',
        'reaction_time', 'time_behind', 'fina_points', 'medal_tag',
        'qualified', 'record_type', 'splits_json', 'year'
    ]

    # Add missing columns with None
    for col in schema_columns:
        if col not in df.columns:
            df[col] = None

    # Only keep schema columns
    df = df[schema_columns]

    # Truncate splits_json if too long (NVARCHAR(MAX) but be safe)
    def truncate_splits(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x)
        return s[:3900] if len(s) > 3900 else s

    if 'splits_json' in df.columns:
        df['splits_json'] = df['splits_json'].apply(truncate_splits)

    # Convert numeric columns safely
    numeric_int_cols = ['lane', 'heat_rank', 'final_rank', 'athlete_age', 'fina_points', 'year']
    for col in numeric_int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(pd.notna(df[col]), None)

    float_cols = ['time_seconds', 'reaction_time', 'time_behind']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(pd.notna(df[col]), None)

    # Ensure string columns are proper strings
    str_cols = ['heat_category', 'discipline_name', 'gender', 'full_name',
                'first_name', 'last_name', 'nationality', 'nationality_name',
                'person_id', 'biography_id', 'result_id', 'time_raw',
                'medal_tag', 'qualified', 'record_type', 'splits_json', 'event_id']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

    return df


def migrate_year(conn, year: int, base_path: Path, force: bool = False):
    """Migrate a single year's data to Azure SQL."""

    csv_file = base_path / f"Results_{year}.csv"
    if not csv_file.exists():
        print(f"  [SKIP] {csv_file.name} not found")
        return 0

    cursor = conn.cursor()

    # Check if already migrated
    if not force:
        cursor.execute("SELECT COUNT(*) FROM results_flat WHERE year = ?", (year,))
        existing = cursor.fetchone()[0]
        if existing > 0:
            print(f"  [SKIP] Year {year} already has {existing:,} rows")
            return 0

    # Load CSV
    print(f"  Loading {csv_file.name}...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} rows")

    # Prepare data
    df = prepare_dataframe(df, year)

    # Build INSERT statement
    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join(columns)
    sql = f"INSERT INTO results_flat ({column_names}) VALUES ({placeholders})"

    # Insert in batches
    batch_size = 200
    total_inserted = 0
    failed_rows = 0

    print(f"  Inserting...")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]

        # Convert to list of tuples, handling None properly
        rows = []
        for _, row in batch.iterrows():
            row_tuple = tuple(None if pd.isna(v) else v for v in row.values)
            rows.append(row_tuple)

        try:
            cursor.executemany(sql, rows)
            conn.commit()
            total_inserted += len(rows)
        except Exception as e:
            # Try row by row on failure
            for row in rows:
                try:
                    cursor.execute(sql, row)
                    conn.commit()
                    total_inserted += 1
                except Exception as row_e:
                    failed_rows += 1

        # Progress
        pct = min(100, (i + batch_size) / len(df) * 100)
        print(f"    {total_inserted:,}/{len(df):,} ({pct:.0f}%)", end='\r')

    status = f"{total_inserted:,} inserted"
    if failed_rows > 0:
        status += f", {failed_rows} failed"
    print(f"  [OK] {status}                    ")
    return total_inserted


def clear_database(conn):
    """Clear all data from results_flat table."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM results_flat")
    conn.commit()
    print("[OK] Database cleared")


def migrate_all(years: list = None, force: bool = False):
    """Migrate all or specified years to Azure SQL."""

    print("=" * 60)
    print("Swimming Data Migration to Azure SQL")
    print("=" * 60)

    base_path = Path(__file__).parent

    # Get list of years to migrate
    if years:
        years_to_migrate = years
    else:
        csv_files = sorted(base_path.glob("Results_*.csv"))
        years_to_migrate = []
        for f in csv_files:
            name = f.stem
            if 'checkpoint' not in str(f) and 'enriched' not in name and 'All_' not in name:
                try:
                    year = int(name.split('_')[-1])
                    years_to_migrate.append(year)
                except:
                    pass
        years_to_migrate = sorted(set(years_to_migrate))

    print(f"\nYears to migrate: {years_to_migrate}")
    print(f"Total: {len(years_to_migrate)} files")

    # Connect to Azure SQL
    print("\nConnecting to Azure SQL...")
    try:
        conn = get_connection()
        print("[OK] Connected")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    # Clear if force
    if force:
        print("\nClearing existing data...")
        clear_database(conn)

    # Migrate each year (reconnect for each year to avoid connection timeouts)
    total_migrated = 0
    start_time = datetime.now()
    conn.close()  # Close initial connection

    for year in years_to_migrate:
        print(f"\n[{year}]")
        try:
            # Fresh connection for each year
            conn = get_connection()
            migrated = migrate_year(conn, year, base_path, force)
            total_migrated += migrated
            conn.close()
        except Exception as e:
            print(f"  [ERROR] {e}")
            try:
                conn.close()
            except:
                pass

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Total rows migrated: {total_migrated:,}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    if elapsed > 0:
        print(f"Rate: {total_migrated / elapsed:.0f} rows/second")

    # Verify final count
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM results_flat")
        final_count = cursor.fetchone()[0]
        print(f"\nTotal rows in database: {final_count:,}")
        conn.close()
    except Exception as e:
        print(f"\n[WARN] Could not verify final count: {e}")


if __name__ == "__main__":
    force = '--clear' in sys.argv or '--force' in sys.argv

    # Remove flags from args
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if args:
        years = [int(y) for y in args]
        migrate_all(years, force)
    else:
        migrate_all(force=force)
