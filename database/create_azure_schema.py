"""
Create Azure SQL Database Schema
Executes schema_azure.sql against the Azure SQL database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_schema():
    """Create database schema in Azure SQL."""
    print("=" * 60)
    print("Azure SQL Schema Creation")
    print("=" * 60)

    try:
        import pyodbc
    except ImportError:
        print("[ERROR] pyodbc not installed. Run: pip install pyodbc")
        return False

    # Get connection string
    conn_str = os.getenv('AZURE_SQL_CONN')
    if not conn_str:
        print("[ERROR] AZURE_SQL_CONN not found in environment")
        return False

    # Use SQL Server driver (works on this machine)
    conn_str = conn_str.replace('ODBC Driver 17 for SQL Server', 'SQL Server')

    # Read schema file
    schema_path = Path(__file__).parent / 'schema_azure.sql'
    if not schema_path.exists():
        print(f"[ERROR] Schema file not found: {schema_path}")
        return False

    print(f"[INFO] Reading schema from: {schema_path}")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # Split by GO statements (T-SQL batch separator)
    batches = [batch.strip() for batch in schema_sql.split('GO') if batch.strip()]
    print(f"[INFO] Found {len(batches)} SQL batches to execute")

    try:
        print("\n[CONNECT] Connecting to Azure SQL...")
        conn = pyodbc.connect(conn_str, timeout=60)
        cursor = conn.cursor()
        print("[OK] Connected successfully")

        # Execute each batch
        for i, batch in enumerate(batches, 1):
            if not batch or batch.startswith('--'):
                continue
            try:
                print(f"\n[BATCH {i}/{len(batches)}] Executing...")
                # Show first 60 chars of batch for context
                preview = batch.replace('\n', ' ')[:60]
                print(f"         {preview}...")
                cursor.execute(batch)
                conn.commit()
                print(f"         [OK]")
            except pyodbc.Error as e:
                error_msg = str(e)
                # Ignore "already exists" type errors
                if 'already exists' in error_msg.lower():
                    print(f"         [SKIP] Already exists")
                else:
                    print(f"         [WARN] {error_msg[:100]}")

        # Verify tables were created
        print("\n" + "=" * 60)
        print("Verifying schema...")
        print("=" * 60)

        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()

        if tables:
            print(f"\n[OK] Created {len(tables)} tables:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM [{table[0]}]")
                count = cursor.fetchone()[0]
                print(f"     - {table[0]}: {count:,} rows")
        else:
            print("[WARN] No tables found after schema creation")

        conn.close()
        print("\n" + "=" * 60)
        print("SCHEMA CREATION COMPLETED")
        print("=" * 60)
        return True

    except pyodbc.Error as e:
        print(f"[ERROR] Database error: {e}")
        if 'not currently available' in str(e):
            print("\n[INFO] Database may be paused (serverless).")
            print("       Go to Azure Portal and resume the database,")
            print("       or wait and retry in 30-60 seconds.")
        return False

if __name__ == "__main__":
    success = create_schema()
    sys.exit(0 if success else 1)
