"""
Configuration file for swimming data scraper
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_KEY_2 = os.getenv('OPENROUTER_API_KEY_2')
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
BRIGHTDATA_API_KEY = os.getenv('BRIGHTDATA_API_KEY')

# Free models available via OpenRouter
FREE_MODELS = {
    'llama-3.2-3b': 'meta-llama/llama-3.2-3b-instruct:free',
    'gemini-flash': 'google/gemini-flash-1.5:free',
    'qwen-2-7b': 'qwen/qwen-2-7b-instruct:free',
    'gemini-flash-8b': 'google/gemini-flash-1.5-8b:free',
    'llama-3.1-8b': 'meta-llama/llama-3.1-8b-instruct:free',
}

# Default model for enrichment tasks
DEFAULT_FREE_MODEL = FREE_MODELS['gemini-flash']

# OpenRouter API endpoint
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

# World Aquatics API Configuration
WORLD_AQUATICS_BASE_URL = 'https://api.worldaquatics.com/fina'
RATE_LIMIT_DELAY = 1.5  # seconds between requests

# Scraper Configuration
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds

# Data Configuration
OUTPUT_DIR = 'data'
LOG_FILE = 'swimming_scraper.log'

# Swimming Event Categories
POOL_CONFIGURATIONS = ['LCM', 'SCM']  # Long Course, Short Course

DISTANCES = {
    'sprint': [50, 100],
    'middle': [200, 400],
    'distance': [800, 1500]
}

STROKES = [
    'FREESTYLE',
    'BACKSTROKE',
    'BREASTSTROKE',
    'BUTTERFLY',
    'FREESTYLE_RELAY',
    'MEDLEY_RELAY'
]

GENDERS = ['M', 'W', 'X']  # Male, Women, Mixed

# Data Quality Thresholds
MIN_VALID_TIME = 10.0  # Minimum reasonable race time in seconds
MAX_VALID_TIME = 7200.0  # Maximum reasonable race time (2 hours)

# Saudi Athletes of Interest (example)
SAUDI_ATHLETE_IDS = [
    1640444,
    1588718,
    1584963,
    1776309
]

# Competition Tiers (for classification)
MAJOR_COMPETITIONS = [
    'Olympic Games',
    'World Championships',
    'World Aquatics Championships',
    'FINA World Championships',
    'World Cup',
]

# Split Time Analysis Configuration
PACING_EVEN_THRESHOLD = 0.5  # seconds difference to be considered "even" pacing
MIN_SPLITS_FOR_ANALYSIS = 2
