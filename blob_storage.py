"""
Azure Blob Storage Module for Swimming Analytics
Uses Parquet files stored in Azure Blob for efficient data access

Storage Structure:
  swimming/
    ├── master.parquet          # All results (main data file)
    ├── world_records.parquet   # Current world records
    └── backups/
        └── backup_YYYYMMDD.parquet
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from io import BytesIO

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Try to import Azure and DuckDB
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: azure-storage-blob not installed. Run: pip install azure-storage-blob")

# Azure Identity for AAD authentication (enterprise accounts)
try:
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False
    DefaultAzureCredential = None
    InteractiveBrowserCredential = None

# Try to import broker credential for VS Code integration
try:
    from azure.identity.broker import InteractiveBrowserBrokerCredential
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    InteractiveBrowserBrokerCredential = None

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: duckdb not installed. Run: pip install duckdb")

# Configuration
CONTAINER_NAME = "swimming-data"
FOLDER = ""  # Store at container root for simplicity
MASTER_FILE = "master.parquet"
WORLD_RECORDS_FILE = "world_records.parquet"

# Azure Storage Account URL (for AAD authentication)
# This is extracted from connection string if available, otherwise uses default
STORAGE_ACCOUNT_URL = "https://worldaquatics.blob.core.windows.net/"

# Connection string and SAS token (lazy-loaded)
_CONN_STRING = None
_SAS_TOKEN = None


def _get_connection_string() -> Optional[str]:
    """Get Azure Storage connection string from env or Streamlit secrets."""
    global _CONN_STRING

    if _CONN_STRING is not None:
        return _CONN_STRING

    # Try environment variable first
    _CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    # Try Streamlit secrets if not in environment
    if not _CONN_STRING:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
                _CONN_STRING = st.secrets['AZURE_STORAGE_CONNECTION_STRING']
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass

    return _CONN_STRING


def _get_sas_token() -> Optional[str]:
    """Get Azure Storage SAS token from env or Streamlit secrets."""
    global _SAS_TOKEN

    if _SAS_TOKEN is not None:
        return _SAS_TOKEN

    # Try environment variable first
    _SAS_TOKEN = os.getenv('AZURE_STORAGE_SAS_TOKEN')

    # Try Streamlit secrets if not in environment
    if not _SAS_TOKEN:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_STORAGE_SAS_TOKEN' in st.secrets:
                _SAS_TOKEN = st.secrets['AZURE_STORAGE_SAS_TOKEN']
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass

    return _SAS_TOKEN


def _use_azure() -> bool:
    """Check if Azure Blob Storage should be used."""
    if os.getenv('FORCE_LOCAL_DATA', '').lower() in ('true', '1', 'yes'):
        return False
    # Can use Azure with connection string, SAS token, OR AAD authentication
    has_conn_string = bool(_get_connection_string())
    has_sas_token = bool(_get_sas_token())
    has_aad = AZURE_IDENTITY_AVAILABLE and AZURE_AVAILABLE
    return (has_conn_string or has_sas_token or has_aad) and AZURE_AVAILABLE


def get_connection_mode() -> str:
    """Return current connection mode."""
    if _use_azure():
        if _get_connection_string():
            return 'azure_blob_connstring'
        elif _get_sas_token():
            return 'azure_blob_sas'
        elif AZURE_IDENTITY_AVAILABLE:
            return 'azure_blob_aad'
        return 'azure_blob'
    return 'local_csv'


def get_blob_service() -> Optional['BlobServiceClient']:
    """Get Azure Blob Service client.

    Supports authentication methods in order:
    1. Connection string (for GitHub Actions / automated pipelines)
    2. SAS token (simple, time-limited access)
    3. Service Principal env vars (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
    4. Interactive browser broker (Windows native auth - for local dev)
    5. Interactive browser (opens browser to sign in - for local dev)
    """
    if not AZURE_AVAILABLE:
        return None

    # Method 1: Connection string (best for GitHub Actions)
    conn_str = _get_connection_string()
    if conn_str and conn_str != "PASTE_YOUR_CONNECTION_STRING_HERE":
        print("Authenticating via connection string")
        return BlobServiceClient.from_connection_string(conn_str)

    # Method 2: SAS token (simple, no RBAC required)
    sas_token = _get_sas_token()
    if sas_token:
        # Remove leading ? if present
        sas_token = sas_token.lstrip('?')
        account_url_with_sas = f"{STORAGE_ACCOUNT_URL}?{sas_token}"
        print("Authenticating via SAS token")
        return BlobServiceClient(account_url=account_url_with_sas)

    # Method 2: Service Principal (for GitHub Actions with OIDC or secrets)
    # DefaultAzureCredential will pick up AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
    if os.getenv('AZURE_CLIENT_ID') and os.getenv('AZURE_TENANT_ID'):
        try:
            print("Authenticating via Service Principal...")
            credential = DefaultAzureCredential()
            client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
            client.get_account_information()
            print("Authenticated via Service Principal")
            return client
        except Exception as e:
            print(f"Service Principal auth failed: {e}")

    # Method 3 & 4: Interactive auth (for local development only)
    # Skip if running in CI/CD environment
    if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
        print("Running in CI - interactive auth not available")
        return None

    if AZURE_IDENTITY_AVAILABLE:
        # Try broker credential first (uses Windows native auth, integrates with VS Code)
        if BROKER_AVAILABLE:
            try:
                print("Attempting Windows broker authentication...")
                credential = InteractiveBrowserBrokerCredential(
                    parent_window_handle=0  # Use default window
                )
                client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
                client.get_account_information()
                print("Authenticated via Windows broker")
                return client
            except Exception as e:
                print(f"Broker auth failed: {e}")

        # Fall back to interactive browser authentication
        try:
            print("Opening browser for Azure authentication...")
            credential = InteractiveBrowserCredential()
            client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
            client.get_account_information()
            print("Authenticated via browser")
            return client
        except Exception as e:
            print(f"Interactive browser authentication failed: {e}")
            return None

    return None


def get_container_client(create_if_missing: bool = True) -> Optional['ContainerClient']:
    """Get container client for swimming data.

    Args:
        create_if_missing: If True, creates the container if it doesn't exist
    """
    blob_service = get_blob_service()
    if not blob_service:
        return None

    container = blob_service.get_container_client(CONTAINER_NAME)

    # Create container if it doesn't exist
    if create_if_missing:
        try:
            if not container.exists():
                container.create_container()
                print(f"Created container: {CONTAINER_NAME}")
        except Exception as e:
            # Container might already exist or we don't have permission
            pass

    return container


def download_parquet(blob_path: str) -> Optional[pd.DataFrame]:
    """Download a parquet file from Azure Blob Storage."""
    container = get_container_client()
    if not container:
        return None

    try:
        blob_client = container.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return pd.read_parquet(BytesIO(data))
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")
        return None


def upload_parquet(df: pd.DataFrame, blob_path: str, overwrite: bool = True) -> bool:
    """Upload a DataFrame as parquet to Azure Blob Storage."""
    container = get_container_client()
    if not container:
        return False

    try:
        buffer = BytesIO()
        # Use gzip compression for smaller file size (better for free tier)
        df.to_parquet(buffer, index=False, compression='gzip')
        buffer.seek(0)
        file_size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
        print(f"Parquet file size: {file_size_mb:.1f} MB")

        blob_client = container.get_blob_client(blob_path)
        # Use larger timeout for big files
        blob_client.upload_blob(
            buffer,
            overwrite=overwrite,
            max_concurrency=4,
            timeout=600  # 10 minute timeout
        )
        print(f"Uploaded {len(df):,} rows to {blob_path}")
        return True
    except Exception as e:
        print(f"Error uploading to {blob_path}: {e}")
        return False


def create_backup() -> Optional[str]:
    """Create a backup of the master parquet file."""
    container = get_container_client()
    if not container:
        return None

    try:
        # Check if master file exists
        blob_client = container.get_blob_client(MASTER_FILE)
        if not blob_client.exists():
            print("No master file to backup")
            return None

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{FOLDER}backups/backup_{timestamp}.parquet"

        # Copy blob
        source_url = blob_client.url
        backup_client = container.get_blob_client(backup_path)
        backup_client.start_copy_from_url(source_url)

        print(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Backup error: {e}")
        return None


def load_results() -> pd.DataFrame:
    """Load swimming results from Azure Blob or local CSV."""

    # Try Azure Blob first
    if _use_azure():
        print("Loading data from Azure Blob Storage...")
        df = download_parquet(MASTER_FILE)
        if df is not None and not df.empty:
            print(f"Loaded {len(df):,} results from Azure Blob")
            return df
        print("Azure Blob empty or failed, falling back to local...")

    # Fall back to local CSV files
    return _load_local_csv()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to consistent format."""
    # Column name mappings (old name -> new name)
    column_map = {
        'DisciplineName': 'discipline_name',
        'Heat Category': 'heat_category',
        'HeatCategory': 'heat_category',
    }

    # Rename columns that exist
    rename_dict = {old: new for old, new in column_map.items() if old in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


def _load_local_csv() -> pd.DataFrame:
    """Load results from local CSV files."""
    from pathlib import Path
    import re

    all_dfs = []

    # Load Results_*.csv files
    csv_files = sorted(Path('.').glob('Results_*.csv'))
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)

            # Extract year from filename (e.g., Results_2024.csv -> 2024)
            year_match = re.search(r'Results_(\d{4})\.csv', f.name)
            if year_match:
                file_year = int(year_match.group(1))
                # Only add year if column doesn't exist or is all null
                if 'year' not in df.columns or df['year'].isna().all():
                    df['year'] = file_year
                    print(f"  Added year={file_year} to {f.name}")

            # Normalize column names
            df = _normalize_columns(df)

            all_dfs.append(df)
            print(f"Loaded {len(df):,} rows from {f.name}")
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Total: {len(combined):,} results from CSV")
        return combined

    return pd.DataFrame()


def save_results(df: pd.DataFrame, append: bool = False) -> bool:
    """Save results to Azure Blob Storage as Parquet."""

    if not _use_azure():
        print("Azure not configured, saving locally")
        df.to_parquet('swimming_results.parquet', index=False)
        return True

    if append:
        # Load existing data and append
        existing = download_parquet(MASTER_FILE)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            # Remove duplicates based on key columns
            if 'result_id' in df.columns:
                df = df.drop_duplicates(subset=['result_id'], keep='last')
            elif 'ResultId' in df.columns:
                df = df.drop_duplicates(subset=['ResultId'], keep='last')

    return upload_parquet(df, MASTER_FILE)


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to ensure Parquet compatibility."""
    df = df.copy()

    # Convert object columns with mixed types to string
    for col in df.columns:
        if df[col].dtype == 'object':
            # Fill NaN with empty string and convert to string
            df[col] = df[col].fillna('').astype(str)
            # Replace 'nan' strings with empty
            df[col] = df[col].replace('nan', '')

    return df


def migrate_csv_to_parquet() -> bool:
    """Migrate all local CSV files to Azure Blob as single Parquet file."""

    print("=" * 60)
    print("MIGRATING CSV TO AZURE BLOB STORAGE")
    print("=" * 60)

    # Load all local data
    df = _load_local_csv()

    if df.empty:
        print("No data to migrate")
        return False

    # Clean data for Parquet compatibility
    print("Cleaning data for Parquet format...")
    df = _clean_dataframe_for_parquet(df)

    # Create backup first
    create_backup()

    # Upload to Azure
    success = upload_parquet(df, MASTER_FILE)

    if success:
        print(f"\nMigration complete: {len(df):,} rows uploaded to Azure Blob")

    return success


def test_connection() -> dict:
    """Test Azure Blob Storage connectivity."""
    conn_str = _get_connection_string()
    has_valid_conn_str = conn_str and conn_str != "PASTE_YOUR_CONNECTION_STRING_HERE"
    sas_token = _get_sas_token()
    has_sas_token = bool(sas_token)

    # Determine auth method
    if has_valid_conn_str:
        auth_method = 'connection_string'
    elif has_sas_token:
        auth_method = 'sas_token'
    elif AZURE_IDENTITY_AVAILABLE:
        auth_method = 'aad'
    else:
        auth_method = 'none'

    result = {
        'mode': get_connection_mode(),
        'azure_configured': has_valid_conn_str or has_sas_token,
        'has_connection_string': has_valid_conn_str,
        'has_sas_token': has_sas_token,
        'azure_identity_available': AZURE_IDENTITY_AVAILABLE,
        'azure_available': AZURE_AVAILABLE,
        'duckdb_available': DUCKDB_AVAILABLE,
        'auth_method': auth_method,
        'storage_account_url': STORAGE_ACCOUNT_URL,
        'container': CONTAINER_NAME,
        'connection_test': 'not_run',
        'row_count': 0,
        'error': None
    }

    if not _use_azure():
        result['connection_test'] = 'skipped'
        return result

    try:
        container = get_container_client()
        if container:
            # List blobs in swimming folder
            blobs = list(container.list_blobs(name_starts_with=FOLDER))
            result['connection_test'] = 'success'
            result['blobs_found'] = len(blobs)
            result['blob_names'] = [b.name for b in blobs[:10]]

            # Try to get row count from master file
            df = download_parquet(MASTER_FILE)
            if df is not None:
                result['row_count'] = len(df)
    except Exception as e:
        result['connection_test'] = 'failed'
        result['error'] = str(e)

    return result


# DuckDB direct query support (for advanced analytics)
_duckdb_conn = None
_duckdb_df_registered = False


def get_duckdb_connection():
    """Get or create a DuckDB connection with data loaded."""
    global _duckdb_conn, _duckdb_df_registered

    if not DUCKDB_AVAILABLE:
        print("DuckDB not available. Run: pip install duckdb")
        return None

    if _duckdb_conn is not None and _duckdb_df_registered:
        return _duckdb_conn

    try:
        _duckdb_conn = duckdb.connect(':memory:')

        # Load data from Azure and register as table
        print("Loading data into DuckDB...")
        df = load_results()

        if df.empty:
            print("No data available")
            return None

        # Register DataFrame as a table
        _duckdb_conn.register('swimming', df)
        _duckdb_df_registered = True

        print(f"DuckDB ready with {len(df):,} rows in 'swimming' table")
        return _duckdb_conn

    except Exception as e:
        print(f"DuckDB initialization error: {e}")
        return None


def query(sql: str) -> Optional[pd.DataFrame]:
    """Execute SQL query against swimming data using DuckDB.

    The data is available as the 'swimming' table.

    Examples:
        # Count results by year
        query("SELECT year, COUNT(*) as count FROM swimming GROUP BY year")

        # Find fastest times for an event
        query("SELECT FullName, NAT, Time FROM swimming WHERE discipline_name LIKE '%100m Freestyle%' ORDER BY Time LIMIT 10")

        # Athlete performance summary
        query("SELECT FullName, COUNT(*) as races, MIN(Time) as best FROM swimming WHERE NAT='KSA' GROUP BY FullName")
    """
    conn = get_duckdb_connection()
    if conn is None:
        return None

    try:
        return conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"Query error: {e}")
        return None


def query_with_duckdb(sql: str) -> Optional[pd.DataFrame]:
    """Execute SQL query against Azure Blob data using DuckDB.

    DEPRECATED: Use query() instead for better performance.
    This function is kept for backwards compatibility.
    """
    return query(sql)


def refresh_duckdb():
    """Reload data from Azure into DuckDB (use after scraper runs)."""
    global _duckdb_conn, _duckdb_df_registered

    if _duckdb_conn is not None:
        _duckdb_conn.close()
    _duckdb_conn = None
    _duckdb_df_registered = False

    return get_duckdb_connection()


# Convenience query functions
def get_athlete_results(name: str = None, nat: str = None) -> Optional[pd.DataFrame]:
    """Get results for a specific athlete or country.

    Args:
        name: Athlete name (partial match)
        nat: Country code (e.g., 'KSA', 'USA')
    """
    conditions = []
    if name:
        conditions.append(f"FullName ILIKE '%{name}%'")
    if nat:
        conditions.append(f"NAT = '{nat.upper()}'")

    if not conditions:
        return query("SELECT * FROM swimming LIMIT 1000")

    where = " AND ".join(conditions)
    return query(f"""
        SELECT year, competition_name, discipline_name, gender,
               FullName, NAT, Time, Rank, pacing_type, lap_times_json
        FROM swimming
        WHERE {where}
        ORDER BY year DESC, Time ASC
    """)


def get_event_rankings(event: str, year: int = None, gender: str = None, limit: int = 50) -> Optional[pd.DataFrame]:
    """Get fastest times for a specific event.

    Args:
        event: Event name (partial match, e.g., '100m Freestyle')
        year: Filter by year (optional)
        gender: 'Men' or 'Women' (optional)
        limit: Number of results to return
    """
    conditions = [f"discipline_name ILIKE '%{event}%'"]
    if year:
        conditions.append(f"year = {year}")
    if gender:
        conditions.append(f"gender ILIKE '%{gender}%'")

    where = " AND ".join(conditions)
    return query(f"""
        SELECT DISTINCT FullName, NAT, Time, year, competition_name, Rank, pacing_type
        FROM swimming
        WHERE {where} AND Time IS NOT NULL AND Time != ''
        ORDER BY Time ASC
        LIMIT {limit}
    """)


def get_yearly_summary() -> Optional[pd.DataFrame]:
    """Get summary statistics by year."""
    return query("""
        SELECT
            year,
            COUNT(*) as total_results,
            COUNT(DISTINCT FullName) as unique_athletes,
            COUNT(DISTINCT NAT) as countries,
            COUNT(DISTINCT competition_name) as competitions
        FROM swimming
        GROUP BY year
        ORDER BY year DESC
    """)


def get_country_summary(nat: str = None) -> Optional[pd.DataFrame]:
    """Get performance summary by country."""
    where = f"WHERE NAT = '{nat.upper()}'" if nat else ""
    return query(f"""
        SELECT
            NAT,
            COUNT(*) as total_results,
            COUNT(DISTINCT FullName) as athletes,
            SUM(CASE WHEN Rank = 1 THEN 1 ELSE 0 END) as gold,
            SUM(CASE WHEN Rank = 2 THEN 1 ELSE 0 END) as silver,
            SUM(CASE WHEN Rank = 3 THEN 1 ELSE 0 END) as bronze
        FROM swimming
        {where}
        GROUP BY NAT
        ORDER BY gold DESC, silver DESC, bronze DESC
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Azure Blob Storage Connection")
    print("=" * 60)
    result = test_connection()

    print(f"\nConnection Mode: {result['mode']}")
    print(f"Auth Method: {result['auth_method']}")
    print(f"Storage Account: {result['storage_account_url']}")
    print(f"Container: {result['container']}")
    print(f"\nAzure SDK Available: {result['azure_available']}")
    print(f"Azure Identity Available: {result['azure_identity_available']}")
    print(f"Connection String Configured: {result.get('has_connection_string', False)}")
    print(f"SAS Token Configured: {result.get('has_sas_token', False)}")
    print(f"DuckDB Available: {result['duckdb_available']}")
    print(f"\nConnection Test: {result['connection_test']}")

    if result.get('row_count'):
        print(f"Row Count: {result['row_count']:,}")

    if result.get('blob_names'):
        print(f"\nBlobs Found: {result['blobs_found']}")
        for name in result['blob_names']:
            print(f"  - {name}")

    if result.get('error'):
        print(f"\nError: {result['error']}")

    print("\n" + "=" * 60)
    if result['connection_test'] == 'success':
        print("SUCCESS: Ready to use Azure Blob Storage!")
    elif result['auth_method'] == 'sas_token':
        print("Using SAS token authentication.")
    elif result['auth_method'] == 'aad':
        print("Using Azure AD authentication.")
        print("Make sure you're logged in: az login")
    else:
        print("To configure, set one of these environment variables:")
        print("  - AZURE_STORAGE_CONNECTION_STRING (full access)")
        print("  - AZURE_STORAGE_SAS_TOKEN (time-limited access)")
        print("\nOr use interactive browser auth (will open browser)")
    print("=" * 60)
