"""
Azure SQL Database Connection Module
Swimming Performance Analysis - Team Saudi

Supports both local development (SQLite/CSV) and cloud deployment (Azure SQL).
Uses lazy-loading for Streamlit secrets compatibility.
"""

import os
import pandas as pd
from typing import Optional, List
from contextlib import contextmanager
from pathlib import Path

# Try to import pyodbc for Azure SQL
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    print("Note: pyodbc not installed. Azure SQL features disabled (using local CSV).")

# Lazy-loaded Azure SQL connection string
_AZURE_SQL_CONN = None

def _get_azure_conn_string() -> Optional[str]:
    """
    Get Azure SQL connection string from env or Streamlit secrets (lazy-loaded).
    This must be called at runtime, not at import time, for Streamlit Cloud compatibility.
    """
    global _AZURE_SQL_CONN

    if _AZURE_SQL_CONN is not None:
        return _AZURE_SQL_CONN

    # Try environment variable first (local development)
    _AZURE_SQL_CONN = os.getenv('AZURE_SQL_CONN')

    # Try Streamlit secrets if not in environment (cloud deployment)
    if not _AZURE_SQL_CONN:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_SQL_CONN' in st.secrets:
                _AZURE_SQL_CONN = st.secrets['AZURE_SQL_CONN']
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass

    # Handle Driver 17 vs 18 compatibility
    if _AZURE_SQL_CONN and 'ODBC Driver 18' in _AZURE_SQL_CONN:
        _AZURE_SQL_CONN = _AZURE_SQL_CONN.replace('ODBC Driver 18', 'ODBC Driver 17')

    return _AZURE_SQL_CONN


def _use_azure() -> bool:
    """Check if Azure SQL should be used."""
    # Check for FORCE_LOCAL environment variable to override
    if os.getenv('FORCE_LOCAL_DATA', '').lower() in ('true', '1', 'yes'):
        return False
    return bool(_get_azure_conn_string()) and PYODBC_AVAILABLE


def get_connection_mode() -> str:
    """Return current connection mode: 'azure' or 'local_csv'"""
    return 'azure' if _use_azure() else 'local_csv'


@contextmanager
def get_azure_connection():
    """Context manager for Azure SQL connections."""
    if not PYODBC_AVAILABLE:
        raise ImportError("pyodbc is required for Azure SQL connections")

    conn_str = _get_azure_conn_string()
    if not conn_str:
        raise ValueError("AZURE_SQL_CONN not found in environment or Streamlit secrets")

    conn = None
    try:
        # Try with driver substitution for compatibility
        if 'ODBC Driver 17' in conn_str or 'ODBC Driver 18' in conn_str:
            # Check if the driver exists, if not try SQL Server driver
            try:
                conn = pyodbc.connect(conn_str, timeout=60)
            except pyodbc.Error as e:
                if 'driver' in str(e).lower():
                    # Fall back to basic SQL Server driver
                    conn_str_fallback = conn_str.replace('ODBC Driver 17 for SQL Server', 'SQL Server')
                    conn_str_fallback = conn_str_fallback.replace('ODBC Driver 18 for SQL Server', 'SQL Server')
                    conn = pyodbc.connect(conn_str_fallback, timeout=60)
                else:
                    raise
        else:
            conn = pyodbc.connect(conn_str, timeout=60)
        yield conn
    finally:
        if conn:
            conn.close()


def query_azure(sql: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query against Azure SQL and return results as DataFrame.
    """
    with get_azure_connection() as conn:
        if params:
            return pd.read_sql(sql, conn, params=params)
        return pd.read_sql(sql, conn)


def load_results_from_azure(
    year: Optional[int] = None,
    nationality: Optional[str] = None,
    athlete_name: Optional[str] = None,
    discipline: Optional[str] = None,
    limit: int = 10000
) -> pd.DataFrame:
    """
    Load swimming results from Azure SQL with optional filters.

    Args:
        year: Filter by competition year
        nationality: Filter by athlete nationality (e.g., 'KSA')
        athlete_name: Filter by athlete name (partial match)
        discipline: Filter by discipline name (partial match)
        limit: Maximum rows to return

    Returns:
        DataFrame with results
    """
    conditions = []
    params = []

    if year:
        conditions.append("year = ?")
        params.append(year)
    if nationality:
        conditions.append("nationality = ?")
        params.append(nationality)
    if athlete_name:
        conditions.append("full_name LIKE ?")
        params.append(f"%{athlete_name}%")
    if discipline:
        conditions.append("discipline_name LIKE ?")
        params.append(f"%{discipline}%")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
        SELECT TOP {limit} *
        FROM results_flat
        WHERE {where_clause}
        ORDER BY year DESC, time_seconds ASC
    """

    return query_azure(sql, tuple(params) if params else None)


def load_results_local(
    year: Optional[int] = None,
    nationality: Optional[str] = None,
    athlete_name: Optional[str] = None,
    discipline: Optional[str] = None
) -> pd.DataFrame:
    """
    Load swimming results from local CSV files.

    Args:
        year: Filter by competition year (if None, loads all years)
        nationality: Filter by athlete nationality
        athlete_name: Filter by athlete name
        discipline: Filter by discipline name

    Returns:
        DataFrame with results
    """
    base_path = Path(__file__).parent

    # Determine which files to load
    if year:
        csv_files = [base_path / f"Results_{year}.csv"]
    else:
        csv_files = sorted(base_path.glob("Results_*.csv"))

    dfs = []
    for csv_file in csv_files:
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                # Extract year from filename
                file_year = int(csv_file.stem.split('_')[-1])
                df['year'] = file_year
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Standardize column names to match Azure schema
    column_mapping = {
        'Heat Category': 'heat_category',
        'DisciplineName': 'discipline_name',
        'Gender': 'gender',
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
    df = df.rename(columns=column_mapping)

    # Apply filters
    if nationality:
        df = df[df['nationality'] == nationality]
    if athlete_name:
        df = df[df['full_name'].str.contains(athlete_name, case=False, na=False)]
    if discipline:
        df = df[df['discipline_name'].str.contains(discipline, case=False, na=False)]

    return df


def load_results(
    year: Optional[int] = None,
    nationality: Optional[str] = None,
    athlete_name: Optional[str] = None,
    discipline: Optional[str] = None,
    limit: int = 10000
) -> pd.DataFrame:
    """
    Load swimming results using the best available source.
    Automatically uses Azure SQL if configured, otherwise local CSV files.

    Args:
        year: Filter by competition year
        nationality: Filter by athlete nationality (e.g., 'KSA')
        athlete_name: Filter by athlete name (partial match)
        discipline: Filter by discipline name (partial match)
        limit: Maximum rows to return (Azure only)

    Returns:
        DataFrame with results
    """
    if _use_azure():
        return load_results_from_azure(
            year=year,
            nationality=nationality,
            athlete_name=athlete_name,
            discipline=discipline,
            limit=limit
        )
    else:
        return load_results_local(
            year=year,
            nationality=nationality,
            athlete_name=athlete_name,
            discipline=discipline
        )


def test_connection() -> dict:
    """Test database connectivity and return diagnostic info."""
    result = {
        'mode': get_connection_mode(),
        'azure_configured': bool(_get_azure_conn_string()),
        'pyodbc_available': PYODBC_AVAILABLE,
        'connection_test': 'not_run',
        'error': None,
        'row_count': 0
    }

    if _use_azure():
        try:
            with get_azure_connection() as conn:
                df = pd.read_sql("SELECT COUNT(*) as cnt FROM results_flat", conn)
                result['connection_test'] = 'success'
                result['row_count'] = int(df['cnt'].iloc[0])
        except Exception as e:
            result['connection_test'] = 'failed'
            result['error'] = str(e)
    else:
        # Test local CSV loading
        try:
            df = load_results_local(year=2024)
            result['connection_test'] = 'success'
            result['row_count'] = len(df)
        except Exception as e:
            result['connection_test'] = 'failed'
            result['error'] = str(e)

    return result


# Convenience function to load all data (for compatibility with existing code)
def load_all_results() -> pd.DataFrame:
    """Load all swimming results from all available years."""
    return load_results()


if __name__ == "__main__":
    # Test the connection
    print("Testing Azure SQL connection...")
    result = test_connection()
    print(f"Mode: {result['mode']}")
    print(f"Connection test: {result['connection_test']}")
    print(f"Row count: {result['row_count']:,}")
    if result['error']:
        print(f"Error: {result['error']}")
