"""
Backup Swimming Data from Azure SQL to Local CSV
Downloads all data from Azure SQL and saves as CSV files

Usage:
    python backup_from_azure.py                    # Backup all data
    python backup_from_azure.py --by-year          # Backup split by year
    python backup_from_azure.py --year 2024        # Backup specific year
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Backup directory
BACKUP_DIR = Path(__file__).parent / "backups"


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


def backup_all_data():
    """Download all data from Azure SQL to a single CSV file."""
    print("=" * 60)
    print("Azure SQL Backup - Full Database")
    print("=" * 60)

    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)

    print("\nConnecting to Azure SQL...")
    conn = get_connection()
    print("[OK] Connected")

    # Get row count
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM results_flat")
    total_rows = cursor.fetchone()[0]
    print(f"\nTotal rows to backup: {total_rows:,}")

    if total_rows == 0:
        print("[WARN] Database is empty, nothing to backup")
        conn.close()
        return

    # Download data
    print("\nDownloading data...")
    start_time = datetime.now()

    df = pd.read_sql("SELECT * FROM results_flat ORDER BY year, discipline_name", conn)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[OK] Downloaded {len(df):,} rows in {elapsed:.1f} seconds")

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = BACKUP_DIR / f"swimming_backup_full_{timestamp}.csv"

    print(f"\nSaving to {filename}...")
    df.to_csv(filename, index=False)
    file_size = filename.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved ({file_size:.1f} MB)")

    conn.close()

    print("\n" + "=" * 60)
    print("BACKUP COMPLETE")
    print("=" * 60)
    print(f"File: {filename}")
    print(f"Rows: {len(df):,}")
    print(f"Size: {file_size:.1f} MB")


def backup_by_year():
    """Download data from Azure SQL, split by year into separate CSV files."""
    print("=" * 60)
    print("Azure SQL Backup - By Year")
    print("=" * 60)

    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)

    print("\nConnecting to Azure SQL...")
    conn = get_connection()
    print("[OK] Connected")

    # Get years in database
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT year FROM results_flat ORDER BY year")
    years = [row[0] for row in cursor.fetchall()]

    if not years:
        print("[WARN] No data found in database")
        conn.close()
        return

    print(f"\nYears found: {years}")
    print(f"Total: {len(years)} files to create")

    timestamp = datetime.now().strftime("%Y%m%d")
    total_rows = 0

    for year in years:
        print(f"\n[{year}] Downloading...")

        df = pd.read_sql(
            "SELECT * FROM results_flat WHERE year = ? ORDER BY discipline_name",
            conn,
            params=(year,)
        )

        filename = BACKUP_DIR / f"backup_Results_{year}_{timestamp}.csv"
        df.to_csv(filename, index=False)

        print(f"  [OK] {len(df):,} rows saved to {filename.name}")
        total_rows += len(df)

    conn.close()

    print("\n" + "=" * 60)
    print("BACKUP COMPLETE")
    print("=" * 60)
    print(f"Files created: {len(years)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Location: {BACKUP_DIR}")


def backup_specific_year(year: int):
    """Download a specific year's data from Azure SQL."""
    print("=" * 60)
    print(f"Azure SQL Backup - Year {year}")
    print("=" * 60)

    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)

    print("\nConnecting to Azure SQL...")
    conn = get_connection()
    print("[OK] Connected")

    print(f"\nDownloading year {year}...")
    df = pd.read_sql(
        "SELECT * FROM results_flat WHERE year = ? ORDER BY discipline_name",
        conn,
        params=(year,)
    )

    if len(df) == 0:
        print(f"[WARN] No data found for year {year}")
        conn.close()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = BACKUP_DIR / f"backup_Results_{year}_{timestamp}.csv"

    df.to_csv(filename, index=False)
    file_size = filename.stat().st_size / (1024 * 1024)

    conn.close()

    print("\n" + "=" * 60)
    print("BACKUP COMPLETE")
    print("=" * 60)
    print(f"File: {filename}")
    print(f"Rows: {len(df):,}")
    print(f"Size: {file_size:.2f} MB")


def show_database_stats():
    """Show database statistics without downloading."""
    print("=" * 60)
    print("Azure SQL Database Statistics")
    print("=" * 60)

    print("\nConnecting to Azure SQL...")
    conn = get_connection()
    print("[OK] Connected")

    cursor = conn.cursor()

    # Total rows
    cursor.execute("SELECT COUNT(*) FROM results_flat")
    total = cursor.fetchone()[0]
    print(f"\nTotal rows: {total:,}")

    # Rows by year
    cursor.execute("""
        SELECT year, COUNT(*) as cnt
        FROM results_flat
        GROUP BY year
        ORDER BY year
    """)
    rows = cursor.fetchall()

    if rows:
        print("\nRows by year:")
        for year, count in rows:
            print(f"  {year}: {count:,}")

    # Unique athletes
    cursor.execute("SELECT COUNT(DISTINCT full_name) FROM results_flat")
    athletes = cursor.fetchone()[0]
    print(f"\nUnique athletes: {athletes:,}")

    # Nationalities
    cursor.execute("SELECT COUNT(DISTINCT nationality) FROM results_flat")
    nations = cursor.fetchone()[0]
    print(f"Nationalities: {nations:,}")

    # Saudi athletes
    cursor.execute("SELECT COUNT(*) FROM results_flat WHERE nationality = 'KSA'")
    ksa = cursor.fetchone()[0]
    print(f"Saudi results: {ksa:,}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--by-year":
            backup_by_year()
        elif sys.argv[1] == "--year" and len(sys.argv) > 2:
            backup_specific_year(int(sys.argv[2]))
        elif sys.argv[1] == "--stats":
            show_database_stats()
        else:
            print("Usage:")
            print("  python backup_from_azure.py              # Full backup")
            print("  python backup_from_azure.py --by-year    # Backup split by year")
            print("  python backup_from_azure.py --year 2024  # Backup specific year")
            print("  python backup_from_azure.py --stats      # Show database stats")
    else:
        backup_all_data()
