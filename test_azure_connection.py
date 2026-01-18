"""
Azure SQL Connection Test Script
Tests connection to swimming-server-ksa.database.windows.net
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test Azure SQL Database connection."""
    print("=" * 60)
    print("Azure SQL Connection Test")
    print("=" * 60)

    # Check if pyodbc is available
    try:
        import pyodbc
        print(f"[OK] pyodbc version: {pyodbc.version}")
    except ImportError:
        print("[ERROR] pyodbc not installed. Run: pip install pyodbc")
        return False

    # Check for connection string
    conn_str = os.getenv('AZURE_SQL_CONN')
    if not conn_str:
        print("[ERROR] AZURE_SQL_CONN not found in environment")
        print("       Make sure .env file exists with AZURE_SQL_CONN=...")
        return False

    # Mask password for display
    masked = conn_str
    if 'Pwd=' in masked:
        start = masked.find('Pwd=') + 4
        end = masked.find(';', start)
        if end == -1:
            end = len(masked)
        masked = masked[:start] + '****' + masked[end:]

    print(f"[OK] Connection string found")
    print(f"     {masked[:80]}...")

    # Parse connection details
    parts = dict(item.split('=', 1) for item in conn_str.split(';') if '=' in item)
    server = parts.get('Server', 'Unknown')
    database = parts.get('Database', 'Unknown')
    driver = parts.get('Driver', 'Unknown')

    print(f"\n[INFO] Server: {server}")
    print(f"[INFO] Database: {database}")
    print(f"[INFO] Driver: {driver}")

    # Test connection
    print(f"\n[TEST] Attempting connection...")
    try:
        conn = pyodbc.connect(conn_str, timeout=30)
        print("[OK] Connection successful!")

        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        row = cursor.fetchone()
        print(f"\n[INFO] SQL Server Version:")
        print(f"       {row[0][:80]}...")

        # Check existing tables
        print(f"\n[INFO] Checking existing tables...")
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()
        if tables:
            print(f"[OK] Found {len(tables)} table(s):")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM [{table[0]}]")
                count = cursor.fetchone()[0]
                print(f"       - {table[0]}: {count:,} rows")
        else:
            print("[INFO] No tables found (database is empty)")

        conn.close()
        print("\n" + "=" * 60)
        print("CONNECTION TEST PASSED")
        print("=" * 60)
        return True

    except pyodbc.Error as e:
        print(f"[ERROR] Connection failed!")
        print(f"        {str(e)}")

        # Common error troubleshooting
        error_msg = str(e).lower()
        print("\n[TROUBLESHOOTING]")
        if 'login timeout' in error_msg or 'timeout' in error_msg:
            print("  - Check Azure SQL firewall rules")
            print("  - Add your IP to the firewall allowlist")
            print("  - Ensure 'Allow Azure services' is enabled")
        elif 'login failed' in error_msg:
            print("  - Check username and password in connection string")
            print("  - Verify the password doesn't have extra braces")
        elif 'driver' in error_msg:
            print("  - Install ODBC Driver 17 or 18 for SQL Server")
            print("  - Download from Microsoft website")
        elif 'network' in error_msg or 'tcp' in error_msg:
            print("  - Check internet connection")
            print("  - Verify server name is correct")

        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
