# Swimming Performance Database

SQL database for Team Saudi swimming performance analysis.

## Quick Start

### Option 1: SQLite (Local Development)

```bash
# Install dependencies
pip install sqlalchemy

# Create database and import data
cd database
python import_csv.py --dir ..

# Database file: swimming_performance.db
```

### Option 2: PostgreSQL (Production)

```bash
# Start with Docker
cd database
docker-compose up -d

# Import data
python import_csv.py --dir .. --db postgresql://swimming_admin:swimming_secret@localhost:5432/swimming_performance
```

### Option 3: Full Stack (Docker)

```bash
# Start database + dashboard
docker-compose --profile admin up -d

# Access:
# - Dashboard: http://localhost:8501
# - pgAdmin: http://localhost:8080
```

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `athletes` | Swimmer profiles (name, nationality, IDs) |
| `competitions` | Swimming meets/events |
| `events` | Disciplines (50m Freestyle, 100m Backstroke, etc.) |
| `results` | Race results with splits and pacing analysis |
| `world_records` | Current world records for benchmarking |
| `elite_benchmarks` | Research-based performance thresholds |

### Key Views

- `athlete_personal_bests` - PBs by athlete and event
- `medal_table` - Medal count by nationality
- `pacing_effectiveness` - Pacing strategy analysis
- `athlete_progression` - Year-over-year improvement

## Data Import

### Initial Import (All CSV Files)

```python
from database import create_database, get_session, import_all_csv_files

# Create and populate database
engine = create_database()
session = get_session(engine)
stats = import_all_csv_files('.', session)

print(f"Imported {stats['records']:,} results from {stats['files']} files")
```

### Incremental Sync (After New Scrape)

After scraping new data, sync it to the database:

```python
from database import sync_new_data

# Sync a newly scraped CSV file
stats = sync_new_data('Results_2026.csv')
print(f"Imported {stats['imported']} new results")
print(f"Total results in database: {stats['results']}")
```

Command line:
```bash
python database/import_csv.py --file Results_2026.csv
```

## Query Examples

```python
from database import create_database, get_session
from database.queries import SwimmingQueries

session = get_session()
q = SwimmingQueries(session)

# Get athlete personal bests
pbs = q.get_athlete_personal_bests(athlete_id=123)

# Get event rankings
rankings = q.get_event_rankings('100m Freestyle', year=2024)

# Get medal table
medals = q.get_medal_table(year=2024)

# Pacing effectiveness analysis
pacing = q.get_pacing_effectiveness('400m Freestyle')
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///swimming_performance.db` | Database connection string |
| `DB_PASSWORD` | `swimming_secret` | PostgreSQL password (Docker) |
| `PGADMIN_EMAIL` | `admin@teamsaudi.com` | pgAdmin login |
| `PGADMIN_PASSWORD` | `admin` | pgAdmin password |

## GitHub Actions Deployment

Add to `.github/workflows/deploy.yml`:

```yaml
name: Deploy Database

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r database/requirements-db.txt

      - name: Create database
        run: python database/import_csv.py --dir .

      - name: Upload database artifact
        uses: actions/upload-artifact@v4
        with:
          name: swimming-database
          path: swimming_performance.db
```

## Cloud Deployment Options

### Azure SQL (Recommended for Team Saudi)

1. **Create Azure SQL Database** in Azure Portal
2. **Configure Firewall** - Allow Azure services and GitHub Actions IPs
3. **Add GitHub Secret**:
   - Go to Repository Settings > Secrets and Variables > Actions
   - Add `AZURE_SQL_CONN` with your connection string:
   ```
   Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server.database.windows.net,1433;Database=swimming_db;Uid=your-username;Pwd=your-password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
   ```

4. **Push to main branch** - GitHub Actions will automatically:
   - Scrape latest swimming data
   - Sync to Azure SQL database
   - Run daily at 6 AM UTC (9 AM Saudi time)

5. **Manual Trigger**: Go to Actions > "Deploy to Azure SQL" > Run workflow

### Render.com (PostgreSQL)
1. Create PostgreSQL database
2. Set `DATABASE_URL` environment variable
3. Deploy from GitHub

### Railway
1. Add PostgreSQL plugin
2. Deploy from GitHub
3. Database auto-connects

### Supabase
1. Create project
2. Use connection string in `DATABASE_URL`
3. Run migrations

## Data Statistics

- **500,000+** race results
- **25,000+** athletes
- **27 years** of data (2000-2026)
- Split times and pacing analysis included
