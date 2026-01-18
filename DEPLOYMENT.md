# Streamlit Cloud Deployment Guide

## Team Saudi Swimming Analytics Dashboard

Complete guide for deploying the coaching dashboard to Streamlit Cloud with Azure Blob Storage backend.

---

## Architecture

```
Local Development               Streamlit Cloud
─────────────────               ──────────────────
CSV files (Results_*.csv)  →    Azure Blob Storage
.env file                       Streamlit Secrets

Storage Structure:
  personal-data/
    └── swimming/
        ├── master.parquet       # All results (480k+ rows)
        ├── world_records.parquet
        └── backups/
            └── backup_YYYYMMDD.parquet
```

**Why Azure Blob + Parquet?**
- Faster than Azure SQL (direct file access)
- Cheaper (~$0.02/GB/month vs SQL compute costs)
- Simpler (no database management)
- DuckDB can query directly for analytics

---

## Quick Start

### 1. Create Azure Storage Account

1. Go to [Azure Portal](https://portal.azure.com)
2. Create Storage Account:
   - **Name**: `teamsaudistorage` (must be unique)
   - **Region**: Choose closest to your users
   - **Performance**: Standard
   - **Redundancy**: LRS (cheapest)
3. Create container: `personal-data`
4. Get connection string: Access Keys → Connection string

### 2. Migrate Local Data to Azure Blob

```bash
# Set connection string
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."

# Migrate CSV files to Parquet in Azure Blob
python scraper_swimming.py --migrate
```

### 3. Push to GitHub

```bash
git add .
git commit -m "Add Azure Blob Storage integration"
git push origin main
```

### 4. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app → Select your repo
3. Settings:
   - **Branch**: `main`
   - **Main file**: `Swimming/coaching_dashboard.py`
4. Add secret in Settings → Secrets:

```toml
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=teamsaudistorage;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net"
```

---

## GitHub Actions Workflows

### Scheduled Scraper (`scraper.yml`)

Runs weekly on Sunday at 2 AM UTC:
- Scrapes World Aquatics API
- Saves to Azure Blob as Parquet
- Creates backup before updating

Manual trigger: Actions → Swimming Data Scraper → Run workflow

### CI/CD Pipeline (`streamlit.yml`)

Runs on every push:
- Validates Python syntax
- Tests imports
- Checks dashboard structure

---

## File Structure

```
Swimming/
├── coaching_dashboard.py      # Main Streamlit app
├── coaching_analytics.py      # Analytics module
├── blob_storage.py           # Azure Blob connection
├── scraper_swimming.py       # Data scraper
├── requirements.txt          # Dependencies
├── .streamlit/
│   ├── config.toml           # Theme (committed)
│   └── secrets.toml.example  # Template
├── .github/workflows/
│   ├── scraper.yml           # Weekly data sync
│   └── streamlit.yml         # CI/CD
└── DEPLOYMENT.md             # This file
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_STORAGE_CONNECTION_STRING` | Yes | Azure Blob connection string |
| `FORCE_LOCAL_DATA` | No | Set `true` to use local CSV instead |

---

## Commands

```bash
# Test Azure Blob connection
python scraper_swimming.py --test

# Migrate local CSVs to Azure Blob
python scraper_swimming.py --migrate

# Scrape current year
python scraper_swimming.py

# Scrape specific year
python scraper_swimming.py --year 2024

# Run dashboard locally
streamlit run coaching_dashboard.py
```

---

## Troubleshooting

### Connection Failed

**Error**: `Azure Blob connection failed`

**Fix**:
1. Verify connection string in Azure Portal → Storage Account → Access Keys
2. Check container `personal-data` exists
3. Ensure no firewall blocking access

### No Data Loaded

**Error**: `Loaded 0 results`

**Fix**:
1. Run migration first: `python scraper_swimming.py --migrate`
2. Check blob exists: Azure Portal → Storage Account → Containers → personal-data → swimming/

### Import Errors

**Error**: `ModuleNotFoundError: azure.storage.blob`

**Fix**: Install dependencies:
```bash
pip install azure-storage-blob duckdb pyarrow
```

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|--------------|
| Azure Blob Storage (1GB) | ~$0.02 |
| Streamlit Cloud (Free tier) | $0 |
| GitHub Actions (2000 min) | $0 |
| **Total** | **~$0.02/month** |

---

## Data Flow

```
World Aquatics API
       ↓
scraper_swimming.py (GitHub Actions - weekly)
       ↓
Azure Blob Storage (swimming/master.parquet)
       ↓
blob_storage.py (load_results)
       ↓
coaching_analytics.py (analysis)
       ↓
coaching_dashboard.py (Streamlit Cloud)
       ↓
User Browser
```

---

## GitHub Secrets Required

Add in GitHub → Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `AZURE_STORAGE_CONNECTION_STRING` | Full connection string from Azure |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard Overview | Key metrics, charts |
| Road to Nagoya 2026 | Asian Games tracking |
| Road to LA 2028 | Olympic qualification |
| Athlete Profiles | Individual analysis |
| Talent Development | Career progression |
| Pacing Analysis | Race strategy |
| Competition Intel | Competitor benchmarks |
| Advanced Analytics | KPIs, forecasting |

---

## Support

- **Data**: 480,000+ swimming results (2000-2026)
- **Updates**: Weekly via GitHub Actions
- **Storage**: Azure Blob (Parquet format)
