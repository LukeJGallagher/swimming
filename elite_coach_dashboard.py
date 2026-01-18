"""
Elite Swimming Coach Dashboard - Team Saudi
World-class analytics platform for elite swim coaches and sports analysts
Features: Age percentiles, Race prediction, Regional analysis, Competition pathways
"""

try:
    import streamlit as st
except ImportError:
    print("Install: pip install streamlit plotly pandas numpy")
    exit(1)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import ast
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import swimming analytics module
try:
    from swimming_analytics import (
        calculate_fina_points, check_qualification_status,
        analyze_peak_performance_potential, analyze_race_segments,
        analyze_performance_trajectory, generate_athlete_report,
        QUALIFICATION_STANDARDS, BASE_TIMES_LCM_2024, normalize_event_name
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Team Saudi Swimming Analytics",
    page_icon="🇸🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elite dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #006C35 0%, #003D1F 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,108,53,0.3);
    }

    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #006C35;
    }

    .saudi-athlete {
        background: linear-gradient(135deg, #006C35 0%, #003D1F 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    .percentile-excellent { background-color: #22c55e; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
    .percentile-good { background-color: #84cc16; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
    .percentile-average { background-color: #eab308; color: black; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }
    .percentile-developing { background-color: #f97316; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; }

    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, white 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #006C35;
        margin: 1.5rem 0 1rem 0;
    }

    .region-gcc { border-left: 4px solid #006C35; }
    .region-asian { border-left: 4px solid #dc2626; }
    .region-world { border-left: 4px solid #2563eb; }
</style>
""", unsafe_allow_html=True)


# ===== CONSTANTS =====

GCC_COUNTRIES = ['KSA', 'SAU', 'UAE', 'QAT', 'KUW', 'BRN', 'OMA', 'OMN', 'BHR']
ASIAN_COUNTRIES = ['CHN', 'JPN', 'KOR', 'IND', 'HKG', 'SGP', 'MAS', 'THA', 'PHI', 'INA',
                   'VIE', 'TPE', 'KAZ', 'UZB', 'IRQ', 'IRI', 'SYR', 'JOR', 'LBN', 'PAK'] + GCC_COUNTRIES

AGE_GROUPS = {
    'Youth (13-14)': (13, 14),
    'Junior (15-17)': (15, 17),
    'Young Senior (18-22)': (18, 22),
    'Senior (23-28)': (23, 28),
    'Veteran (29+)': (29, 99)
}

MAJOR_COMPETITIONS = {
    'Asian Games 2026': {'date': '2026-09-19', 'location': 'Aichi-Nagoya, Japan'},
    'World Championships 2025': {'date': '2025-07-11', 'location': 'Singapore'},
    'LA Olympics 2028': {'date': '2028-07-14', 'location': 'Los Angeles, USA'},
    'Asian Championships 2025': {'date': '2025-10-01', 'location': 'TBD'}
}


# ===== DATA UTILITIES =====

def parse_splits_safe(splits_str):
    """Safely parse splits from string representation"""
    if not splits_str or pd.isna(splits_str) or splits_str == '[]':
        return []
    try:
        if isinstance(splits_str, str):
            if splits_str.startswith('['):
                try:
                    return json.loads(splits_str.replace("'", '"'))
                except:
                    return ast.literal_eval(splits_str)
        elif isinstance(splits_str, list):
            return splits_str
    except:
        pass
    return []


def time_to_seconds(time_str) -> Optional[float]:
    """Convert time string to seconds"""
    if not time_str or pd.isna(time_str):
        return None
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return None


def seconds_to_time(seconds: float) -> str:
    """Convert seconds to time string"""
    if pd.isna(seconds) or seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}:{secs:05.2f}"
    return f"{secs:.2f}"


@st.cache_data(ttl=3600)
def load_all_data():
    """Load all available swimming data"""
    current_dir = Path(".")
    data_dir = Path("data")

    all_files = []

    for pattern in ["enriched_*.csv", "all_results_enriched.csv", "All_Results*.csv", "Results_*.csv"]:
        all_files.extend(list(current_dir.glob(pattern)))
        if data_dir.exists():
            all_files.extend(list(data_dir.glob(pattern.lower())))
            all_files.extend(list(data_dir.glob(pattern)))

    all_files = list(set(all_files))

    if not all_files:
        return None, []

    # Priority: enriched combined
    enriched_combined = [f for f in all_files if 'all_results_enriched' in f.name.lower()]
    if enriched_combined:
        default = enriched_combined[0]
    else:
        default = sorted(all_files, key=lambda x: x.stat().st_size, reverse=True)[0]

    df = pd.read_csv(default, low_memory=False)

    # Normalize columns
    if 'DisciplineName' not in df.columns and 'discipline_name' in df.columns:
        df['DisciplineName'] = df['discipline_name']
    if 'Heat Category' in df.columns and 'heat_category' not in df.columns:
        df['heat_category'] = df['Heat Category']

    # Add time in seconds
    df['time_seconds'] = df['Time'].apply(time_to_seconds)

    return df, all_files


def get_region_data(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter data by region"""
    if 'NAT' not in df.columns:
        return df

    if region == 'GCC':
        return df[df['NAT'].isin(GCC_COUNTRIES)]
    elif region == 'Asian':
        return df[df['NAT'].isin(ASIAN_COUNTRIES)]
    elif region == 'World':
        return df
    return df


def get_age_group(age: float) -> str:
    """Determine age group"""
    if pd.isna(age):
        return 'Unknown'
    age = int(age)
    for group_name, (min_age, max_age) in AGE_GROUPS.items():
        if min_age <= age <= max_age:
            return group_name
    return 'Unknown'


def calculate_percentile(value: float, series: pd.Series) -> float:
    """Calculate percentile rank (lower time = better = higher percentile)"""
    if pd.isna(value) or series.empty:
        return None
    # For swimming, lower time is better, so we invert
    return 100 - (series < value).mean() * 100


# ===== AI INTEGRATION =====

def call_ai_analysis(prompt: str, system_prompt: str = None) -> str:
    """Call OpenRouter AI for analysis"""
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        return "AI analysis unavailable - set OPENROUTER_API_KEY in .env"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Free models to try
    free_models = [
        "google/gemini-flash-1.5:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "qwen/qwen-2-7b-instruct:free"
    ]

    for model in free_models:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1500
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
        except:
            continue

    return "AI analysis temporarily unavailable"


# ===== ANALYSIS FUNCTIONS =====

def calculate_age_percentiles(df: pd.DataFrame, event: str, age: float) -> Dict:
    """Calculate percentiles for an athlete's age group"""
    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    if disc_col not in df.columns or 'AthleteResultAge' not in df.columns:
        return {}

    event_df = df[df[disc_col] == event].copy()
    event_df = event_df.dropna(subset=['time_seconds', 'AthleteResultAge'])

    if event_df.empty:
        return {}

    age_group = get_age_group(age)

    # Filter to same age group
    event_df['age_group'] = event_df['AthleteResultAge'].apply(get_age_group)
    age_group_df = event_df[event_df['age_group'] == age_group]

    return {
        'age_group': age_group,
        'athletes_in_group': age_group_df['FullName'].nunique(),
        'percentile_10': age_group_df['time_seconds'].quantile(0.10),
        'percentile_25': age_group_df['time_seconds'].quantile(0.25),
        'percentile_50': age_group_df['time_seconds'].quantile(0.50),
        'percentile_75': age_group_df['time_seconds'].quantile(0.75),
        'percentile_90': age_group_df['time_seconds'].quantile(0.90),
        'world_percentile_10': event_df['time_seconds'].quantile(0.10),
        'world_percentile_50': event_df['time_seconds'].quantile(0.50)
    }


def predict_race_time(training_splits: List[float], race_distance: int) -> Dict:
    """Predict race time from training splits"""
    if not training_splits or len(training_splits) < 2:
        return {'error': 'Need at least 2 training splits'}

    # Calculate average split
    avg_split = np.mean(training_splits)
    std_split = np.std(training_splits)

    # Number of laps
    num_laps = race_distance // 50

    # Predictions with different fatigue models
    # Conservative: assume 2% fatigue per lap after first
    conservative_time = training_splits[0] * 0.98  # Dive advantage
    for i in range(1, num_laps):
        conservative_time += avg_split * (1 + 0.02 * i)

    # Optimistic: maintain pace
    optimistic_time = training_splits[0] * 0.98 + avg_split * (num_laps - 1)

    # Realistic: based on variance
    realistic_time = training_splits[0] * 0.98
    for i in range(1, num_laps):
        realistic_time += avg_split * (1 + 0.01 * i)

    return {
        'training_avg_split': avg_split,
        'training_consistency': std_split,
        'predicted_optimistic': optimistic_time,
        'predicted_realistic': realistic_time,
        'predicted_conservative': conservative_time,
        'confidence': 'High' if std_split < 0.5 else 'Medium' if std_split < 1.0 else 'Low'
    }


def analyze_progression_likelihood(df: pd.DataFrame, athlete_name: str, event: str) -> Dict:
    """Analyze likelihood of progression based on age and historical patterns"""
    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    athlete_df = df[(df['FullName'] == athlete_name) & (df[disc_col] == event)].copy()

    if athlete_df.empty or 'AthleteResultAge' not in athlete_df.columns:
        return {}

    current_age = athlete_df['AthleteResultAge'].max()
    current_best = athlete_df['time_seconds'].min()

    # Get historical improvement patterns for athletes at similar ages
    event_df = df[df[disc_col] == event].copy()
    event_df = event_df.dropna(subset=['AthleteResultAge', 'time_seconds'])

    # Group by athlete and calculate improvement rates
    improvements = []
    for name, group in event_df.groupby('FullName'):
        if len(group) >= 3:
            group = group.sort_values('AthleteResultAge')
            ages = group['AthleteResultAge'].values
            times = group['time_seconds'].values

            for i in range(1, len(times)):
                if ages[i] > ages[i-1]:
                    improvement_rate = (times[i-1] - times[i]) / (ages[i] - ages[i-1])
                    improvements.append({
                        'age': ages[i],
                        'improvement_per_year': improvement_rate
                    })

    if not improvements:
        return {}

    imp_df = pd.DataFrame(improvements)

    # Calculate expected improvement at current age
    age_range = (current_age - 1, current_age + 2)
    similar_age_improvements = imp_df[(imp_df['age'] >= age_range[0]) & (imp_df['age'] <= age_range[1])]

    if similar_age_improvements.empty:
        return {}

    avg_improvement = similar_age_improvements['improvement_per_year'].mean()
    std_improvement = similar_age_improvements['improvement_per_year'].std()

    return {
        'current_age': current_age,
        'current_best': seconds_to_time(current_best),
        'avg_improvement_at_age': avg_improvement,
        'projected_1_year': seconds_to_time(current_best - avg_improvement),
        'projected_2_year': seconds_to_time(current_best - avg_improvement * 2),
        'improvement_likelihood': 'High' if avg_improvement > 0.5 else 'Medium' if avg_improvement > 0 else 'Low',
        'athletes_analyzed': len(imp_df['age'].unique())
    }


def get_season_bests(df: pd.DataFrame, athlete_name: str, year: int = None) -> pd.DataFrame:
    """Get season bests for an athlete"""
    if year is None:
        year = datetime.now().year

    athlete_df = df[df['FullName'] == athlete_name].copy()

    if 'date_from' in athlete_df.columns:
        athlete_df['date'] = pd.to_datetime(athlete_df['date_from'], errors='coerce')
        athlete_df['year'] = athlete_df['date'].dt.year
    elif 'year' in athlete_df.columns:
        pass
    else:
        return pd.DataFrame()

    season_df = athlete_df[athlete_df['year'] == year]

    if season_df.empty:
        return pd.DataFrame()

    disc_col = 'DisciplineName' if 'DisciplineName' in season_df.columns else 'discipline_name'

    # Get best time per event
    season_bests = season_df.loc[season_df.groupby(disc_col)['time_seconds'].idxmin()]

    return season_bests[[disc_col, 'Time', 'time_seconds', 'competition_name', 'date_from']].sort_values('time_seconds')


def get_regional_competitors(df: pd.DataFrame, event: str, region: str, top_n: int = 20) -> pd.DataFrame:
    """Get top competitors from a region for an event"""
    region_df = get_region_data(df, region)
    disc_col = 'DisciplineName' if 'DisciplineName' in region_df.columns else 'discipline_name'

    if disc_col not in region_df.columns:
        return pd.DataFrame()

    event_df = region_df[region_df[disc_col] == event].copy()
    event_df = event_df.dropna(subset=['time_seconds'])

    # Best time per athlete
    best_times = event_df.loc[event_df.groupby('FullName')['time_seconds'].idxmin()]
    best_times = best_times.nsmallest(top_n, 'time_seconds')

    return best_times


# ===== DASHBOARD PAGES =====

def show_executive_overview(df: pd.DataFrame):
    """Executive overview for coaches"""
    st.markdown("""
    <div class="main-header">
        <h1>🇸🇦 Team Saudi Swimming Analytics</h1>
        <p>Elite Performance Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    gcc_df = get_region_data(df, 'GCC')
    asian_df = get_region_data(df, 'Asian')

    with col1:
        st.metric("📊 Total Results", f"{len(df):,}")
    with col2:
        st.metric("🏊 Athletes", df['FullName'].nunique() if 'FullName' in df.columns else 0)
    with col3:
        st.metric("🌍 Countries", df['NAT'].nunique() if 'NAT' in df.columns else 0)
    with col4:
        st.metric("🇸🇦 GCC Athletes", gcc_df['FullName'].nunique() if not gcc_df.empty else 0)
    with col5:
        st.metric("🌏 Asian Athletes", asian_df['FullName'].nunique() if not asian_df.empty else 0)

    st.markdown("---")

    # Regional breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌏 Regional Distribution")
        if 'NAT' in df.columns:
            df_copy = df.copy()
            df_copy['Region'] = df_copy['NAT'].apply(
                lambda x: 'GCC' if x in GCC_COUNTRIES else ('Asian' if x in ASIAN_COUNTRIES else 'Other')
            )
            region_counts = df_copy['Region'].value_counts()

            fig = px.pie(
                values=region_counts.values,
                names=region_counts.index,
                color=region_counts.index,
                color_discrete_map={'GCC': '#006C35', 'Asian': '#dc2626', 'Other': '#6b7280'},
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("📊 Age Group Distribution")
        if 'AthleteResultAge' in df.columns:
            df_copy = df.copy()
            df_copy['Age Group'] = df_copy['AthleteResultAge'].apply(get_age_group)
            age_counts = df_copy['Age Group'].value_counts()

            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                color=age_counts.index,
                color_discrete_sequence=px.colors.sequential.Greens_r
            )
            fig.update_layout(height=350, showlegend=False, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')

    # Upcoming competitions
    st.markdown("---")
    st.subheader("🎯 Road to Major Competitions")

    cols = st.columns(len(MAJOR_COMPETITIONS))
    for i, (comp_name, details) in enumerate(MAJOR_COMPETITIONS.items()):
        with cols[i]:
            comp_date = datetime.strptime(details['date'], '%Y-%m-%d')
            days_to_go = (comp_date - datetime.now()).days

            st.markdown(f"""
            <div class="metric-card">
                <h4>{comp_name}</h4>
                <p><strong>{days_to_go}</strong> days</p>
                <small>{details['location']}</small>
            </div>
            """, unsafe_allow_html=True)


def show_athlete_deep_dive(df: pd.DataFrame):
    """Deep dive analysis for individual athletes with percentiles"""
    st.header("👤 Athlete Deep Dive Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        search = st.text_input("🔍 Search Athlete", "")

    athletes = sorted(df['FullName'].dropna().unique()) if 'FullName' in df.columns else []

    if search:
        athletes = [a for a in athletes if search.lower() in a.lower()]

    selected_athlete = st.selectbox("Select Athlete", athletes[:200] if athletes else [])

    if not selected_athlete:
        st.info("Select an athlete to view detailed analysis")
        return

    athlete_df = df[df['FullName'] == selected_athlete].copy()

    # Profile header
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        country = athlete_df['NAT'].iloc[0] if 'NAT' in athlete_df.columns else 'Unknown'
        region = 'GCC' if country in GCC_COUNTRIES else ('Asian' if country in ASIAN_COUNTRIES else 'World')
        st.metric("🌍 Country", f"{country} ({region})")

    with col2:
        st.metric("🏊 Total Races", len(athlete_df))

    with col3:
        if 'AthleteResultAge' in athlete_df.columns:
            current_age = athlete_df['AthleteResultAge'].max()
            age_group = get_age_group(current_age)
            st.metric("📅 Age / Group", f"{int(current_age)} ({age_group})")

    with col4:
        if 'MedalTag' in athlete_df.columns:
            medals = athlete_df['MedalTag'].notna().sum()
            st.metric("🏅 Medals", medals)

    with col5:
        if 'pacing_type' in athlete_df.columns:
            pref = athlete_df['pacing_type'].mode()
            st.metric("⏱️ Pacing", pref.iloc[0] if len(pref) > 0 else "N/A")

    st.markdown("---")

    # Tabs for different analyses
    disc_col = 'DisciplineName' if 'DisciplineName' in athlete_df.columns else 'discipline_name'

    if disc_col in athlete_df.columns:
        events = sorted(athlete_df[disc_col].dropna().unique())

        tabs = st.tabs(["📈 Progression", "📊 Percentile Analysis", "🎯 Predictions", "📋 Season Bests", "🌏 Regional Ranking"])

        with tabs[0]:  # Progression
            selected_event = st.selectbox("Select Event", events, key="prog_event")

            if selected_event:
                event_df = athlete_df[athlete_df[disc_col] == selected_event].copy()

                if 'date_from' in event_df.columns:
                    event_df['date'] = pd.to_datetime(event_df['date_from'], errors='coerce')
                    event_df = event_df.sort_values('date')
                    x_axis = event_df['date'].tolist()
                    x_label = "Date"
                else:
                    x_axis = list(range(len(event_df)))
                    x_label = "Race Number"

                if not event_df.empty and len(event_df) > 0:
                    event_df['personal_best'] = event_df['time_seconds'].cummin()

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=event_df['time_seconds'].tolist(),
                        mode='markers+lines',
                        name='Race Time',
                        marker=dict(size=12, color='#006C35'),
                        line=dict(color='#006C35')
                    ))

                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=event_df['personal_best'].tolist(),
                        mode='lines',
                        name='Personal Best',
                        line=dict(dash='dash', color='#FFD700', width=3)
                    ))

                    fig.update_layout(
                        title=f"{selected_athlete} - {selected_event} Progression",
                        xaxis_title=x_label,
                        yaxis_title="Time (seconds)",
                        height=450,
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Stats row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Personal Best", seconds_to_time(event_df['time_seconds'].min()))
                    with col2:
                        st.metric("Average", seconds_to_time(event_df['time_seconds'].mean()))
                    with col3:
                        if len(event_df) > 1:
                            improvement = event_df['time_seconds'].iloc[0] - event_df['time_seconds'].min()
                            st.metric("Total Improvement", f"{improvement:.2f}s")
                    with col4:
                        st.metric("Races", len(event_df))

        with tabs[1]:  # Percentile Analysis
            selected_event_pct = st.selectbox("Select Event", events, key="pct_event")

            if selected_event_pct and 'AthleteResultAge' in athlete_df.columns:
                current_age = athlete_df['AthleteResultAge'].max()
                athlete_best = athlete_df[athlete_df[disc_col] == selected_event_pct]['time_seconds'].min()

                percentiles = calculate_age_percentiles(df, selected_event_pct, current_age)

                if percentiles:
                    st.subheader(f"Percentile Analysis - {percentiles['age_group']}")

                    # Calculate athlete's percentile
                    event_all = df[df[disc_col] == selected_event_pct]['time_seconds'].dropna()
                    athlete_percentile = calculate_percentile(athlete_best, event_all)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Your Position")
                        if athlete_percentile:
                            if athlete_percentile >= 90:
                                badge = "🏆 Elite (Top 10%)"
                                color = "percentile-excellent"
                            elif athlete_percentile >= 75:
                                badge = "🥇 Excellent (Top 25%)"
                                color = "percentile-good"
                            elif athlete_percentile >= 50:
                                badge = "🥈 Above Average"
                                color = "percentile-average"
                            else:
                                badge = "📈 Developing"
                                color = "percentile-developing"

                            st.markdown(f'<span class="{color}">{badge}</span>', unsafe_allow_html=True)
                            st.metric("World Percentile", f"{athlete_percentile:.1f}%")
                            st.metric("Your Best", seconds_to_time(athlete_best))

                    with col2:
                        st.markdown("### Standards to Target")
                        st.write(f"**Top 10% (Elite):** {seconds_to_time(percentiles['percentile_10'])}")
                        st.write(f"**Top 25% (Excellent):** {seconds_to_time(percentiles['percentile_25'])}")
                        st.write(f"**Top 50% (Median):** {seconds_to_time(percentiles['percentile_50'])}")

                        gap_to_top10 = athlete_best - percentiles['percentile_10']
                        st.metric("Gap to Top 10%", f"{gap_to_top10:.2f}s")

                    # Visual percentile chart
                    st.markdown("### Time Distribution")

                    event_times = df[df[disc_col] == selected_event_pct]['time_seconds'].dropna()

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=event_times, nbinsx=50, name='All Athletes', opacity=0.7))
                    fig.add_vline(x=athlete_best, line_dash="dash", line_color="red",
                                 annotation_text=f"Your Best: {seconds_to_time(athlete_best)}")
                    fig.add_vline(x=percentiles['percentile_10'], line_dash="dot", line_color="gold",
                                 annotation_text="Top 10%")

                    fig.update_layout(
                        title="Where You Stand",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')

        with tabs[2]:  # Predictions
            selected_event_pred = st.selectbox("Select Event", events, key="pred_event")

            if selected_event_pred:
                st.subheader("🎯 Performance Predictions")

                # Progression likelihood
                progression = analyze_progression_likelihood(df, selected_athlete, selected_event_pred)

                if progression:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Age-Based Projection")
                        st.write(f"**Current Age:** {progression.get('current_age', 'N/A')}")
                        st.write(f"**Current Best:** {progression.get('current_best', 'N/A')}")
                        st.write(f"**Avg Improvement at Age:** {progression.get('avg_improvement_at_age', 0):.2f}s/year")

                        likelihood = progression.get('improvement_likelihood', 'Unknown')
                        if likelihood == 'High':
                            st.success(f"📈 High likelihood of improvement")
                        elif likelihood == 'Medium':
                            st.warning(f"📊 Medium likelihood of improvement")
                        else:
                            st.info(f"📉 Focus on maintaining current level")

                    with col2:
                        st.markdown("### Projected Times")
                        st.metric("1 Year Projection", progression.get('projected_1_year', 'N/A'))
                        st.metric("2 Year Projection", progression.get('projected_2_year', 'N/A'))
                        st.write(f"*Based on {progression.get('athletes_analyzed', 0)} athletes analyzed*")

                # Training split predictor
                st.markdown("---")
                st.markdown("### 🏊 Race Time Predictor from Training")

                col1, col2 = st.columns(2)

                with col1:
                    training_input = st.text_input(
                        "Enter training 50m splits (comma-separated)",
                        placeholder="e.g., 28.5, 29.0, 29.2, 29.5"
                    )

                    race_distance = st.selectbox("Race Distance", [100, 200, 400, 800, 1500])

                with col2:
                    if training_input:
                        try:
                            splits = [float(x.strip()) for x in training_input.split(',')]
                            prediction = predict_race_time(splits, race_distance)

                            if 'error' not in prediction:
                                st.markdown("### Predicted Race Times")
                                st.metric("Optimistic", seconds_to_time(prediction['predicted_optimistic']))
                                st.metric("Realistic", seconds_to_time(prediction['predicted_realistic']))
                                st.metric("Conservative", seconds_to_time(prediction['predicted_conservative']))
                                st.write(f"**Confidence:** {prediction['confidence']}")
                                st.write(f"**Training Consistency:** {prediction['training_consistency']:.2f}s std dev")
                        except:
                            st.error("Invalid input format")

        with tabs[3]:  # Season Bests
            current_year = datetime.now().year
            selected_year = st.selectbox("Select Season", list(range(current_year, 2000, -1)))

            season_bests = get_season_bests(df, selected_athlete, selected_year)

            if not season_bests.empty:
                st.subheader(f"Season Bests - {selected_year}")

                # Add PB comparison
                all_time_bests = athlete_df.loc[athlete_df.groupby(disc_col)['time_seconds'].idxmin()]

                display_df = season_bests.copy()
                display_df['Season Best'] = display_df['Time']

                st.dataframe(
                    display_df[[disc_col, 'Season Best', 'competition_name']].rename(columns={
                        disc_col: 'Event',
                        'competition_name': 'Competition'
                    }),
                    width='stretch'
                )
            else:
                st.info(f"No results found for {selected_year}")

        with tabs[4]:  # Regional Ranking
            selected_event_reg = st.selectbox("Select Event", events, key="reg_event")
            selected_region = st.selectbox("Select Region", ['GCC', 'Asian', 'World'])

            if selected_event_reg:
                regional_top = get_regional_competitors(df, selected_event_reg, selected_region, 20)

                if not regional_top.empty:
                    st.subheader(f"Top {selected_region} Competitors - {selected_event_reg}")

                    # Find athlete's position
                    athlete_best = athlete_df[athlete_df[disc_col] == selected_event_reg]['time_seconds'].min()
                    athlete_rank = (regional_top['time_seconds'] < athlete_best).sum() + 1

                    st.info(f"**{selected_athlete}** ranks **#{athlete_rank}** in {selected_region} with {seconds_to_time(athlete_best)}")

                    display_df = regional_top[['FullName', 'NAT', 'Time']].reset_index(drop=True)
                    display_df.index = display_df.index + 1
                    display_df.index.name = 'Rank'

                    st.dataframe(display_df, width='stretch')


def show_race_analysis(df: pd.DataFrame):
    """Enhanced race analysis with stacked bar visualization"""
    st.header("🎬 Race Analysis & Visualization")

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    col1, col2 = st.columns(2)

    with col1:
        events = sorted(df[disc_col].dropna().unique()) if disc_col in df.columns else []
        selected_event = st.selectbox("Select Event", events)

    with col2:
        comp_col = 'competition_name' if 'competition_name' in df.columns else None
        if comp_col and comp_col in df.columns:
            event_df = df[df[disc_col] == selected_event] if selected_event else df
            competitions = sorted(event_df[comp_col].dropna().unique())
            selected_comp = st.selectbox("Select Competition (optional)", ['All'] + list(competitions))
        else:
            selected_comp = 'All'

    if not selected_event:
        return

    # Filter data
    race_df = df[df[disc_col] == selected_event].copy()
    if selected_comp != 'All' and comp_col:
        race_df = race_df[race_df[comp_col] == selected_comp]

    # Filter for finals if available
    if 'heat_category' in race_df.columns:
        finals = race_df[race_df['heat_category'].str.contains('Final', case=False, na=False)]
        if not finals.empty:
            race_df = finals

    # Get splits data
    splits_col = 'splits_json' if 'splits_json' in race_df.columns else 'Splits'

    if splits_col in race_df.columns:
        race_df = race_df[race_df[splits_col].notna() & (race_df[splits_col] != '[]')]

    race_df = race_df.dropna(subset=['time_seconds'])
    race_df = race_df.nsmallest(8, 'time_seconds')

    if race_df.empty:
        st.warning("No race data available")
        return

    st.subheader(f"Race Progression - {selected_event}")

    # Build split data for visualization
    all_splits = []

    for _, row in race_df.iterrows():
        athlete = row.get('FullName', 'Unknown')
        final_time = row.get('Time', 'N/A')

        if splits_col == 'splits_json':
            try:
                splits = json.loads(row[splits_col]) if row[splits_col] else []
            except:
                splits = []
        else:
            splits = parse_splits_safe(row[splits_col])

        if splits:
            prev_time = 0
            for split in splits:
                distance = int(str(split.get('distance', split.get('Distance', '0'))).replace('m', ''))
                cum_time = time_to_seconds(split.get('time', split.get('Time', '0')))
                diff_time = split.get('differential_time', split.get('DifferentialTime'))

                if diff_time:
                    lap_time = time_to_seconds(diff_time)
                else:
                    lap_time = cum_time - prev_time if cum_time else 0

                all_splits.append({
                    'Athlete': f"{athlete} ({final_time})",
                    'Distance': f"{distance}m",
                    'Lap Time': lap_time,
                    'Cumulative': cum_time,
                    'Order': distance
                })
                prev_time = cum_time if cum_time else prev_time

    if not all_splits:
        st.warning("No split data available for visualization")
        return

    splits_df = pd.DataFrame(all_splits)
    splits_df = splits_df.sort_values(['Athlete', 'Order'])

    # Remove duplicates before pivot (keep first occurrence per athlete/distance)
    splits_df = splits_df.drop_duplicates(subset=['Athlete', 'Distance'], keep='first')

    # Stacked horizontal bar chart (flipped)
    st.markdown("### 📊 Split Breakdown (Horizontal)")

    # Pivot for stacked bar - use pivot_table with aggfunc to handle any remaining duplicates
    pivot_df = splits_df.pivot_table(index='Athlete', columns='Distance', values='Lap Time', aggfunc='first')

    # Reorder columns by distance
    distance_order = sorted(pivot_df.columns, key=lambda x: int(x.replace('m', '')))
    pivot_df = pivot_df[distance_order]

    fig = go.Figure()

    colors = px.colors.sequential.Greens_r[:len(distance_order)]

    for i, dist in enumerate(distance_order):
        fig.add_trace(go.Bar(
            name=dist,
            y=pivot_df.index,
            x=pivot_df[dist],
            orientation='h',
            marker_color=colors[i % len(colors)],
            text=[f"{v:.2f}s" if pd.notna(v) else "" for v in pivot_df[dist]],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        title="Race Split Breakdown by Athlete",
        xaxis_title="Time (seconds)",
        yaxis_title="",
        height=400,
        legend_title="Split",
        template="plotly_white"
    )

    st.plotly_chart(fig, width='stretch')

    # Race progression line chart
    st.markdown("### 📈 Race Progression Over Distance")

    fig2 = px.line(
        splits_df,
        x='Order',
        y='Cumulative',
        color='Athlete',
        markers=True,
        labels={'Order': 'Distance (m)', 'Cumulative': 'Time (seconds)'}
    )

    fig2.update_layout(
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )

    st.plotly_chart(fig2, width='stretch')

    # Lap time comparison
    st.markdown("### ⏱️ Lap Time Comparison")

    fig3 = px.bar(
        splits_df,
        x='Distance',
        y='Lap Time',
        color='Athlete',
        barmode='group',
        labels={'Lap Time': 'Lap Time (seconds)'}
    )

    fig3.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig3, width='stretch')


def show_regional_analysis(df: pd.DataFrame):
    """Regional analysis - GCC, Asian, Youth, Senior perspectives"""
    st.header("🌏 Regional & Category Analysis")

    tabs = st.tabs(["🇸🇦 GCC Focus", "🌏 Asian Rankings", "👶 Youth Development", "🏆 Senior Elite"])

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    with tabs[0]:  # GCC
        st.subheader("GCC Swimming Performance")

        gcc_df = get_region_data(df, 'GCC')

        if gcc_df.empty:
            st.warning("No GCC data available")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("GCC Athletes", gcc_df['FullName'].nunique())
            with col2:
                st.metric("GCC Results", len(gcc_df))
            with col3:
                if 'MedalTag' in gcc_df.columns:
                    st.metric("GCC Medals", gcc_df['MedalTag'].notna().sum())

            # Country breakdown
            st.markdown("### Country Breakdown")
            country_counts = gcc_df.groupby('NAT').agg({
                'FullName': 'nunique',
                'Time': 'count'
            }).rename(columns={'FullName': 'Athletes', 'Time': 'Results'})

            st.dataframe(country_counts.sort_values('Athletes', ascending=False), width='stretch')

            # Top GCC performers by event
            if disc_col in gcc_df.columns:
                st.markdown("### Top GCC Performers")
                selected_event = st.selectbox("Select Event", sorted(gcc_df[disc_col].dropna().unique()), key="gcc_event")

                if selected_event:
                    top_gcc = get_regional_competitors(df, selected_event, 'GCC', 10)
                    if not top_gcc.empty:
                        display_df = top_gcc[['FullName', 'NAT', 'Time']].reset_index(drop=True)
                        display_df.index = display_df.index + 1
                        st.dataframe(display_df, width='stretch')

    with tabs[1]:  # Asian
        st.subheader("Asian Swimming Rankings")

        asian_df = get_region_data(df, 'Asian')

        if disc_col in asian_df.columns:
            selected_event_asian = st.selectbox("Select Event", sorted(asian_df[disc_col].dropna().unique()), key="asian_event")

            if selected_event_asian:
                top_asian = get_regional_competitors(df, selected_event_asian, 'Asian', 20)

                if not top_asian.empty:
                    # Add GCC flag
                    top_asian['Is GCC'] = top_asian['NAT'].isin(GCC_COUNTRIES)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### Asian Top 20")
                        display_df = top_asian[['FullName', 'NAT', 'Time', 'Is GCC']].reset_index(drop=True)
                        display_df.index = display_df.index + 1
                        st.dataframe(display_df, width='stretch')

                    with col2:
                        st.markdown("### GCC in Asian Rankings")
                        gcc_in_top = top_asian[top_asian['Is GCC']]
                        st.metric("GCC Athletes in Top 20", len(gcc_in_top))

                        if not gcc_in_top.empty:
                            for _, row in gcc_in_top.iterrows():
                                st.write(f"**{row['FullName']}** ({row['NAT']}): {row['Time']}")

    with tabs[2]:  # Youth
        st.subheader("Youth Development Analysis")

        if 'AthleteResultAge' in df.columns:
            youth_df = df[df['AthleteResultAge'].between(13, 17)].copy()

            if not youth_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Youth Athletes", youth_df['FullName'].nunique())

                    # Age distribution
                    fig = px.histogram(youth_df, x='AthleteResultAge', nbins=5,
                                      title="Youth Age Distribution")
                    st.plotly_chart(fig, width='stretch')

                with col2:
                    if disc_col in youth_df.columns:
                        st.markdown("### Top Youth by Event")
                        youth_event = st.selectbox("Select Event", sorted(youth_df[disc_col].dropna().unique()), key="youth_event")

                        if youth_event:
                            youth_top = youth_df[youth_df[disc_col] == youth_event].copy()
                            youth_top = youth_top.dropna(subset=['time_seconds', 'FullName'])
                            if not youth_top.empty:
                                # Get best time per athlete safely
                                idx = youth_top.groupby('FullName')['time_seconds'].idxmin()
                                idx = idx.dropna()  # Remove NaN indices
                                youth_top = youth_top.loc[idx]
                                youth_top = youth_top.nsmallest(10, 'time_seconds')

                                display_cols = ['FullName', 'NAT', 'Time', 'AthleteResultAge']
                                display_cols = [c for c in display_cols if c in youth_top.columns]
                                st.dataframe(youth_top[display_cols].reset_index(drop=True), width='stretch')
                            else:
                                st.info("No youth data for this event")

    with tabs[3]:  # Senior
        st.subheader("Senior Elite Analysis")

        if 'AthleteResultAge' in df.columns:
            senior_df = df[df['AthleteResultAge'].between(18, 35)].copy()

            if disc_col in senior_df.columns:
                senior_event = st.selectbox("Select Event", sorted(senior_df[disc_col].dropna().unique()), key="senior_event")

                if senior_event:
                    # World vs Asian vs GCC comparison
                    world_top = get_regional_competitors(df, senior_event, 'World', 10)
                    asian_top = get_regional_competitors(df, senior_event, 'Asian', 10)
                    gcc_top = get_regional_competitors(df, senior_event, 'GCC', 10)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### 🌍 World Top 10")
                        if not world_top.empty:
                            st.dataframe(world_top[['FullName', 'NAT', 'Time']].reset_index(drop=True), width='stretch')

                    with col2:
                        st.markdown("### 🌏 Asian Top 10")
                        if not asian_top.empty:
                            st.dataframe(asian_top[['FullName', 'NAT', 'Time']].reset_index(drop=True), width='stretch')

                    with col3:
                        st.markdown("### 🇸🇦 GCC Top 10")
                        if not gcc_top.empty:
                            st.dataframe(gcc_top[['FullName', 'NAT', 'Time']].reset_index(drop=True), width='stretch')


def show_event_report(df: pd.DataFrame):
    """Detailed per-event analysis report"""
    st.header("📋 Detailed Event Analysis Report")

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    if disc_col not in df.columns:
        st.error("Event data not available")
        return

    events = sorted(df[disc_col].dropna().unique())
    selected_event = st.selectbox("Select Event for Detailed Analysis", events)

    if not selected_event:
        return

    event_df = df[df[disc_col] == selected_event].copy()
    event_df = event_df.dropna(subset=['time_seconds'])

    st.markdown(f"## 📊 {selected_event} - Complete Analysis")

    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Results", len(event_df))
    with col2:
        st.metric("Athletes", event_df['FullName'].nunique())
    with col3:
        st.metric("Countries", event_df['NAT'].nunique() if 'NAT' in event_df.columns else 0)
    with col4:
        st.metric("Best Time", seconds_to_time(event_df['time_seconds'].min()))
    with col5:
        if 'RecordType' in event_df.columns:
            st.metric("Records", event_df['RecordType'].notna().sum())

    st.markdown("---")

    # Tabs for different analyses
    tabs = st.tabs(["🏆 Rankings", "📈 Time Standards", "🌏 Regional", "📊 Trends", "🤖 AI Analysis"])

    with tabs[0]:  # Rankings
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("All-Time Top 20")
            all_time = event_df.loc[event_df.groupby('FullName')['time_seconds'].idxmin()]
            all_time = all_time.nsmallest(20, 'time_seconds')

            display_df = all_time[['FullName', 'NAT', 'Time']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, width='stretch')

        with col2:
            st.subheader("Current Season Top 20")
            current_year = datetime.now().year

            if 'date_from' in event_df.columns:
                event_df['date'] = pd.to_datetime(event_df['date_from'], errors='coerce')
                event_df['year'] = event_df['date'].dt.year
            elif 'year' not in event_df.columns:
                event_df['year'] = current_year

            season_df = event_df[event_df['year'] == current_year]

            if not season_df.empty:
                season_top = season_df.loc[season_df.groupby('FullName')['time_seconds'].idxmin()]
                season_top = season_top.nsmallest(20, 'time_seconds')

                display_df = season_top[['FullName', 'NAT', 'Time']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, width='stretch')
            else:
                st.info(f"No {current_year} results yet")

    with tabs[1]:  # Time Standards
        st.subheader("Time Standards & Percentiles")

        percentiles = [5, 10, 25, 50, 75, 90, 95]
        standards = {}

        for p in percentiles:
            standards[f"Top {100-p}%"] = seconds_to_time(event_df['time_seconds'].quantile(p/100))

        standards_df = pd.DataFrame([standards]).T
        standards_df.columns = ['Time Standard']

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(standards_df, width='stretch')

        with col2:
            # Distribution chart
            fig = px.histogram(event_df, x='time_seconds', nbins=50,
                              title="Time Distribution")
            fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')

    with tabs[2]:  # Regional
        st.subheader("Regional Breakdown")

        col1, col2, col3 = st.columns(3)

        regions = [('World', 'World'), ('Asian', 'Asian'), ('GCC', 'GCC')]

        for i, (region_name, region_key) in enumerate(regions):
            with [col1, col2, col3][i]:
                st.markdown(f"### {region_name}")
                region_top = get_regional_competitors(df, selected_event, region_key, 5)

                if not region_top.empty:
                    st.write(f"**Best:** {seconds_to_time(region_top['time_seconds'].min())}")
                    st.write(f"**Athletes:** {region_top['FullName'].nunique()}")

                    for rank, (_, row) in enumerate(region_top.iterrows(), 1):
                        st.write(f"{rank}. {row['FullName']} ({row['NAT']}): {row['Time']}")

    with tabs[3]:  # Trends
        st.subheader("Performance Trends Over Time")

        if 'year' in event_df.columns or 'date_from' in event_df.columns:
            if 'year' not in event_df.columns:
                event_df['year'] = pd.to_datetime(event_df['date_from'], errors='coerce').dt.year

            yearly_best = event_df.groupby('year')['time_seconds'].min().reset_index()
            yearly_best.columns = ['Year', 'Best Time']

            fig = px.line(yearly_best, x='Year', y='Best Time', markers=True,
                         title="Yearly Best Times")
            fig.update_layout(yaxis_title="Time (seconds)")
            st.plotly_chart(fig, width='stretch')

    with tabs[4]:  # AI Analysis
        st.subheader("🤖 AI-Powered Analysis")

        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing..."):
                # Prepare data summary for AI
                top_5 = event_df.nsmallest(5, 'time_seconds')
                gcc_best = get_regional_competitors(df, selected_event, 'GCC', 1)
                asian_best = get_regional_competitors(df, selected_event, 'Asian', 1)

                prompt = f"""Analyze this swimming event data and provide insights for coaches:

Event: {selected_event}
Total athletes: {event_df['FullName'].nunique()}
Best time ever: {seconds_to_time(event_df['time_seconds'].min())}
World Top 5:
{top_5[['FullName', 'NAT', 'Time']].to_string()}

GCC Best: {gcc_best['Time'].iloc[0] if not gcc_best.empty else 'N/A'} ({gcc_best['FullName'].iloc[0] if not gcc_best.empty else 'N/A'})
Asian Best: {asian_best['Time'].iloc[0] if not asian_best.empty else 'N/A'}

Provide:
1. Analysis of competitive landscape
2. What times are needed to be competitive at Asian/World level
3. Key areas for GCC swimmers to focus on
4. Tactical recommendations for this event
"""
                system_prompt = "You are an elite swimming coach analyst. Provide concise, actionable insights."

                analysis = call_ai_analysis(prompt, system_prompt)
                st.markdown(analysis)


def show_competition_pathway(df: pd.DataFrame):
    """Road to major competitions - Asian Games, LA 2028"""
    st.header("🎯 Competition Pathway Analysis")

    tabs = st.tabs(["🏅 Asian Games 2026", "🇺🇸 LA 2028 Olympics", "🌍 World Championships"])

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    with tabs[0]:  # Asian Games
        st.subheader("Road to Asian Games 2026 - Aichi-Nagoya")

        days_to_go = (datetime(2026, 9, 19) - datetime.now()).days
        st.metric("Days Until Asian Games", days_to_go)

        if disc_col in df.columns:
            selected_event = st.selectbox("Select Event", sorted(df[disc_col].dropna().unique()), key="ag_event")

            if selected_event:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Asian Games Qualifying Standards")

                    # Get Asian top times as proxy for standards
                    asian_top = get_regional_competitors(df, selected_event, 'Asian', 16)

                    if not asian_top.empty:
                        st.write(f"**Estimated A Standard:** {seconds_to_time(asian_top['time_seconds'].iloc[7])}")
                        st.write(f"**Estimated B Standard:** {seconds_to_time(asian_top['time_seconds'].iloc[15])}")

                        st.markdown("### Current Asian Top 8")
                        display_df = asian_top.head(8)[['FullName', 'NAT', 'Time']].reset_index(drop=True)
                        display_df.index = display_df.index + 1
                        st.dataframe(display_df, width='stretch')

                with col2:
                    st.markdown("### GCC Qualification Status")

                    gcc_top = get_regional_competitors(df, selected_event, 'GCC', 10)

                    if not gcc_top.empty:
                        a_standard = asian_top['time_seconds'].iloc[7] if len(asian_top) >= 8 else None

                        for _, row in gcc_top.iterrows():
                            status = "✅ A Standard" if a_standard and row['time_seconds'] <= a_standard else "🔶 B Standard"
                            st.write(f"**{row['FullName']}** ({row['NAT']}): {row['Time']} - {status}")

    with tabs[1]:  # LA 2028
        st.subheader("Road to LA 2028 Olympics")

        days_to_la = (datetime(2028, 7, 14) - datetime.now()).days
        st.metric("Days Until LA 2028", days_to_la)

        if disc_col in df.columns:
            selected_event_la = st.selectbox("Select Event", sorted(df[disc_col].dropna().unique()), key="la_event")

            if selected_event_la:
                st.markdown("### Olympic Qualifying Analysis")

                world_top = get_regional_competitors(df, selected_event_la, 'World', 32)

                if not world_top.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Estimated OQT (Top 16):** {seconds_to_time(world_top['time_seconds'].iloc[15])}")
                        st.write(f"**Medal Contention:** {seconds_to_time(world_top['time_seconds'].iloc[2])}")
                        st.write(f"**Final Standard (Top 8):** {seconds_to_time(world_top['time_seconds'].iloc[7])}")

                    with col2:
                        gcc_best = get_regional_competitors(df, selected_event_la, 'GCC', 1)
                        if not gcc_best.empty:
                            gcc_time = gcc_best['time_seconds'].iloc[0]
                            oqt = world_top['time_seconds'].iloc[15]

                            gap = gcc_time - oqt
                            st.metric("GCC Best", seconds_to_time(gcc_time))
                            st.metric("Gap to OQT", f"{gap:.2f}s", delta_color="inverse")

                            # Years to close gap
                            if gap > 0:
                                improvement_needed = gap / 3.5  # years until 2028
                                st.write(f"**Required improvement:** {improvement_needed:.2f}s per year")

    with tabs[2]:  # World Champs
        st.subheader("World Championships Analysis")

        if disc_col in df.columns:
            selected_event_wc = st.selectbox("Select Event", sorted(df[disc_col].dropna().unique()), key="wc_event")

            if selected_event_wc:
                world_top = get_regional_competitors(df, selected_event_wc, 'World', 20)
                gcc_top = get_regional_competitors(df, selected_event_wc, 'GCC', 5)

                if not world_top.empty:
                    st.markdown("### World Top 20 vs GCC")

                    # Highlight GCC athletes in world ranking
                    world_top['Is GCC'] = world_top['NAT'].isin(GCC_COUNTRIES)

                    display_df = world_top[['FullName', 'NAT', 'Time', 'Is GCC']].reset_index(drop=True)
                    display_df.index = display_df.index + 1

                    def highlight_gcc(row):
                        if row['Is GCC']:
                            return ['background-color: #006C35; color: white'] * len(row)
                        return [''] * len(row)

                    st.dataframe(display_df.style.apply(highlight_gcc, axis=1), width='stretch')


# ===== MAIN APPLICATION =====

def main():
    # Load data
    df, available_files = load_all_data()

    if df is None or df.empty:
        st.error("No data found. Please run the scraper first.")
        st.code("python enhanced_swimming_scraper.py")
        return

    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Flag_of_Saudi_Arabia.svg", width=60)
    st.sidebar.title("🏊 Navigation")

    page = st.sidebar.radio("Select Module", [
        "📊 Executive Overview",
        "🇸🇦 Saudi Athletes",
        "👤 Athlete Deep Dive",
        "🎬 Race Analysis",
        "🌏 Regional Analysis",
        "📋 Event Reports",
        "🎯 Competition Pathway",
        "🔢 FINA Points Calculator",
        "📈 Performance Insights",
        "🔍 Data Explorer"
    ])

    # Data selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Source")

    selected_file = st.sidebar.selectbox(
        "Select Data",
        available_files,
        format_func=lambda x: x.name
    )

    if selected_file:
        df = pd.read_csv(selected_file, low_memory=False)
        if 'DisciplineName' not in df.columns and 'discipline_name' in df.columns:
            df['DisciplineName'] = df['discipline_name']
        df['time_seconds'] = df['Time'].apply(time_to_seconds)

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Records", f"{len(df):,}")
    st.sidebar.metric("Athletes", df['FullName'].nunique() if 'FullName' in df.columns else 0)

    has_enriched = 'pacing_type' in df.columns
    if has_enriched:
        st.sidebar.success("✅ Enriched Data")
    else:
        st.sidebar.warning("⚠️ Basic Data")

    # Route to pages
    if page == "📊 Executive Overview":
        show_executive_overview(df)
    elif page == "🇸🇦 Saudi Athletes":
        show_saudi_athletes(df)
    elif page == "👤 Athlete Deep Dive":
        show_athlete_deep_dive(df)
    elif page == "🎬 Race Analysis":
        show_race_analysis(df)
    elif page == "🌏 Regional Analysis":
        show_regional_analysis(df)
    elif page == "📋 Event Reports":
        show_event_report(df)
    elif page == "🎯 Competition Pathway":
        show_competition_pathway(df)
    elif page == "🔢 FINA Points Calculator":
        show_fina_calculator(df)
    elif page == "📈 Performance Insights":
        show_performance_insights(df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)


def show_fina_calculator(df: pd.DataFrame):
    """World Aquatics (FINA) Points Calculator"""
    st.header("🔢 World Aquatics Points Calculator")

    if not ANALYTICS_AVAILABLE:
        st.error("Analytics module not available. Please ensure swimming_analytics.py is in the same directory.")
        return

    st.markdown("""
    Calculate World Aquatics (FINA) points using the official formula:
    **P = 1000 × (B / T)³**

    Where:
    - **P** = Points (1000 = World Record pace)
    - **B** = Base time (current World Record)
    - **T** = Swimmer's time
    """)

    tabs = st.tabs(["🧮 Calculator", "📊 Batch Analysis", "📋 Standards Reference"])

    with tabs[0]:  # Calculator
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Calculate Points from Time")

            gender = st.selectbox("Gender", ["Men", "Women"], key="fina_gender")
            event = st.selectbox("Event", list(BASE_TIMES_LCM_2024.get(gender, {}).keys()), key="fina_event")

            time_input = st.text_input("Enter time (MM:SS.ss or SS.ss)", placeholder="e.g., 48.50 or 1:45.30")

            if time_input:
                try:
                    # Parse time
                    if ':' in time_input:
                        parts = time_input.split(':')
                        time_seconds = float(parts[0]) * 60 + float(parts[1])
                    else:
                        time_seconds = float(time_input)

                    points = calculate_fina_points(time_seconds, event, gender)
                    base_time = BASE_TIMES_LCM_2024.get(gender, {}).get(event, 0)

                    if points:
                        st.success(f"**{points:.0f} FINA Points**")

                        # Visual gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=points,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 1100]},
                                'bar': {'color': "#006C35"},
                                'steps': [
                                    {'range': [0, 500], 'color': "#fee2e2"},
                                    {'range': [500, 700], 'color': "#fef3c7"},
                                    {'range': [700, 850], 'color': "#d1fae5"},
                                    {'range': [850, 1000], 'color': "#a7f3d0"},
                                    {'range': [1000, 1100], 'color': "#10b981"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 1000
                                }
                            },
                            title={'text': "FINA Points"}
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width='stretch')

                        # Performance level
                        if points >= 1000:
                            st.markdown("🏆 **World Record Level**")
                        elif points >= 900:
                            st.markdown("🥇 **Olympic/World Finalist Level**")
                        elif points >= 800:
                            st.markdown("🌍 **World Class**")
                        elif points >= 700:
                            st.markdown("🌏 **Continental Elite**")
                        elif points >= 600:
                            st.markdown("🏊 **National Elite**")
                        else:
                            st.markdown("📈 **Developing**")

                        st.info(f"World Record: {base_time:.2f}s")
                except Exception as e:
                    st.error(f"Invalid time format: {e}")

        with col2:
            st.subheader("Calculate Time from Points")

            gender2 = st.selectbox("Gender", ["Men", "Women"], key="fina_gender2")
            event2 = st.selectbox("Event", list(BASE_TIMES_LCM_2024.get(gender2, {}).keys()), key="fina_event2")
            target_points = st.number_input("Target Points", min_value=100, max_value=1100, value=800)

            base_time = BASE_TIMES_LCM_2024.get(gender2, {}).get(event2, 0)
            if base_time:
                # Reverse formula: T = B / (P/1000)^(1/3)
                target_time = base_time / ((target_points / 1000) ** (1/3))

                minutes = int(target_time // 60)
                seconds = target_time % 60

                if minutes > 0:
                    time_str = f"{minutes}:{seconds:05.2f}"
                else:
                    time_str = f"{seconds:.2f}"

                st.success(f"**Target Time: {time_str}**")
                st.write(f"To achieve {target_points} points in {event2}")

    with tabs[1]:  # Batch Analysis
        st.subheader("Analyze Multiple Results")

        disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

        if st.button("Calculate FINA Points for All Data"):
            with st.spinner("Calculating points..."):
                df_with_points = df.copy()

                def calc_points_row(row):
                    time_val = row.get('time_seconds')
                    event = row.get(disc_col, '')
                    gender = row.get('Gender', '')
                    if pd.isna(time_val) or not event or not gender:
                        return None
                    return calculate_fina_points(time_val, event, gender)

                df_with_points['fina_points'] = df_with_points.apply(calc_points_row, axis=1)

                valid_points = df_with_points['fina_points'].dropna()

                if not valid_points.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Results with Points", len(valid_points))
                    with col2:
                        st.metric("Max Points", f"{valid_points.max():.0f}")
                    with col3:
                        st.metric("Average Points", f"{valid_points.mean():.0f}")
                    with col4:
                        st.metric("800+ Points", len(valid_points[valid_points >= 800]))

                    # Distribution
                    fig = px.histogram(df_with_points, x='fina_points', nbins=50,
                                      title="FINA Points Distribution")
                    fig.update_layout(xaxis_title="Points", yaxis_title="Count")
                    st.plotly_chart(fig, width='stretch')

                    # Top performers
                    st.subheader("Top Performances by Points")
                    top_df = df_with_points.nlargest(20, 'fina_points')[['FullName', 'NAT', disc_col, 'Time', 'fina_points']]
                    top_df.columns = ['Athlete', 'Country', 'Event', 'Time', 'FINA Points']
                    st.dataframe(top_df.reset_index(drop=True), width='stretch')

    with tabs[2]:  # Standards Reference
        st.subheader("Qualification Standards Reference")

        comp_select = st.selectbox("Competition", list(QUALIFICATION_STANDARDS.keys()))
        gender_select = st.selectbox("Gender", ["Men", "Women"], key="std_gender")

        standards = QUALIFICATION_STANDARDS.get(comp_select, {}).get(gender_select, {})

        if standards:
            standards_df = pd.DataFrame([
                {'Event': event, 'Standard (s)': time, 'Time': seconds_to_time(time)}
                for event, time in standards.items()
            ])
            st.dataframe(standards_df, width='stretch')


def show_performance_insights(df: pd.DataFrame):
    """Advanced performance insights and analytics"""
    st.header("📈 Performance Insights")

    if not ANALYTICS_AVAILABLE:
        st.error("Analytics module not available.")
        return

    tabs = st.tabs(["🎯 Peak Performance Age", "📊 Trajectory Analysis", "🔄 Competitor Comparison"])

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    with tabs[0]:  # Peak Performance Age
        st.subheader("Peak Performance Age Analysis")

        st.markdown("""
        Based on sports science research, swimmers reach peak performance at different ages
        depending on the event type:
        - **Sprint (50m, 100m)**: Men 23-26, Women 22-25
        - **Middle (200m, 400m)**: Men 22-26, Women 21-25
        - **Distance (800m, 1500m)**: Men 21-25, Women 19-24
        - **IM Events**: Men 22-26, Women 20-24
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Select athlete
            athletes = sorted(df['FullName'].dropna().unique())
            selected_athlete = st.selectbox("Select Athlete", athletes, key="ppa_athlete")

        with col2:
            athlete_df = df[df['FullName'] == selected_athlete]
            if disc_col in athlete_df.columns:
                events = sorted(athlete_df[disc_col].dropna().unique())
                selected_event = st.selectbox("Select Event", events, key="ppa_event")

        if selected_athlete and selected_event:
            athlete_event_df = df[(df['FullName'] == selected_athlete) & (df[disc_col] == selected_event)]

            if not athlete_event_df.empty and 'AthleteResultAge' in athlete_event_df.columns:
                current_age = athlete_event_df['AthleteResultAge'].max()
                current_best = athlete_event_df['time_seconds'].min()
                gender = athlete_event_df['Gender'].iloc[0] if 'Gender' in athlete_event_df.columns else 'Men'

                ppa = analyze_peak_performance_potential(current_age, selected_event, gender, current_best)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Current Age", f"{current_age:.0f}")
                    st.metric("Peak Window", ppa['peak_range'])

                with col2:
                    st.metric("Years to Peak", f"{ppa['years_to_peak']:.1f}")
                    st.metric("In Peak Window", "✅ Yes" if ppa['in_peak_window'] else "❌ No")

                with col3:
                    st.metric("Current Best", seconds_to_time(current_best))
                    st.metric("Projected Peak", seconds_to_time(ppa['projected_peak_time']))

                # Potential indicator
                potential = ppa['improvement_potential']
                if potential == 'High':
                    st.success(f"📈 **High Improvement Potential** - Expected {ppa['expected_annual_improvement']} per year")
                elif potential == 'Medium':
                    st.warning(f"📊 **Medium Improvement Potential** - Approaching peak years")
                else:
                    st.info(f"📉 **Maintenance Phase** - Focus on consistency and technique refinement")

    with tabs[1]:  # Trajectory Analysis
        st.subheader("Performance Trajectory Analysis")

        athletes = sorted(df['FullName'].dropna().unique())
        selected_athlete = st.selectbox("Select Athlete", athletes, key="traj_athlete")

        if selected_athlete:
            athlete_df = df[df['FullName'] == selected_athlete]

            if disc_col in athlete_df.columns:
                events = sorted(athlete_df[disc_col].dropna().unique())
                selected_event = st.selectbox("Select Event", events, key="traj_event")

                if selected_event:
                    trajectory = analyze_performance_trajectory(athlete_df, selected_event)

                    if trajectory:
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Career Best", seconds_to_time(trajectory['career_best']))
                            st.metric("Total Races", trajectory['total_races'])

                        with col2:
                            st.metric("Average Time", seconds_to_time(trajectory['average_time']))
                            st.metric("Time Range", f"{trajectory['time_range']:.2f}s")

                        with col3:
                            st.metric("Consistency", f"{trajectory['consistency_score']:.1f}%")
                            st.metric("PB Count", trajectory['pb_count'])

                        with col4:
                            trend = trajectory['recent_trend']
                            if trend == 'Improving':
                                st.metric("Recent Trend", "📈 Improving")
                            elif trend == 'Stable':
                                st.metric("Recent Trend", "➡️ Stable")
                            else:
                                st.metric("Recent Trend", "📉 Declining")
                            st.metric("Total Improvement", f"{trajectory['total_improvement']:.2f}s")

    with tabs[2]:  # Competitor Comparison
        st.subheader("Find Similar Competitors")

        athletes = sorted(df['FullName'].dropna().unique())
        selected_athlete = st.selectbox("Select Athlete", athletes, key="comp_athlete")

        if selected_athlete:
            athlete_df = df[df['FullName'] == selected_athlete]

            if disc_col in athlete_df.columns:
                events = sorted(athlete_df[disc_col].dropna().unique())
                selected_event = st.selectbox("Select Event", events, key="comp_event")

                time_window = st.slider("Time Window (seconds)", 0.5, 5.0, 2.0)

                if selected_event:
                    # Get athlete's best
                    athlete_event = athlete_df[athlete_df[disc_col] == selected_event]
                    athlete_best = athlete_event['time_seconds'].min()

                    st.info(f"**{selected_athlete}** best: {seconds_to_time(athlete_best)}")

                    # Find competitors
                    event_df = df[df[disc_col] == selected_event].copy()
                    event_df = event_df.dropna(subset=['time_seconds', 'FullName'])

                    # Best per athlete
                    idx = event_df.groupby('FullName')['time_seconds'].idxmin()
                    idx = idx.dropna()
                    best_times = event_df.loc[idx]

                    # Filter to window
                    similar = best_times[
                        (best_times['time_seconds'] >= athlete_best - time_window) &
                        (best_times['time_seconds'] <= athlete_best + time_window) &
                        (best_times['FullName'] != selected_athlete)
                    ].sort_values('time_seconds')

                    if not similar.empty:
                        st.write(f"Found {len(similar)} competitors within {time_window}s")

                        display_cols = ['FullName', 'NAT', 'Time', 'time_seconds']
                        display_cols = [c for c in display_cols if c in similar.columns]
                        similar_display = similar[display_cols].copy()
                        similar_display['Gap'] = similar_display['time_seconds'] - athlete_best
                        similar_display['Gap'] = similar_display['Gap'].apply(lambda x: f"{x:+.2f}s")

                        st.dataframe(similar_display.reset_index(drop=True), width='stretch')
                    else:
                        st.info("No competitors found in this time window")


def show_saudi_athletes(df: pd.DataFrame):
    """Saudi Athletes Tracker - dedicated page for Team Saudi"""
    st.markdown("""
    <div class="main-header">
        <h1>🇸🇦 Saudi Athletes Tracker</h1>
        <p>Complete performance tracking for Team Saudi swimmers</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter Saudi athletes
    saudi_df = df[df['NAT'].isin(['KSA', 'SAU'])].copy()

    if saudi_df.empty:
        st.warning("No Saudi athletes found in the database")
        return

    disc_col = 'DisciplineName' if 'DisciplineName' in saudi_df.columns else 'discipline_name'

    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🏊 Athletes", saudi_df['FullName'].nunique())
    with col2:
        st.metric("📊 Total Results", len(saudi_df))
    with col3:
        st.metric("🏆 Events", saudi_df[disc_col].nunique() if disc_col in saudi_df.columns else 0)
    with col4:
        if 'MedalTag' in saudi_df.columns:
            st.metric("🥇 Medals", saudi_df['MedalTag'].notna().sum())
    with col5:
        if 'Gender' in saudi_df.columns:
            female = saudi_df[saudi_df['Gender'] == 'Women']['FullName'].nunique()
            st.metric("👩 Female Athletes", female)

    st.markdown("---")

    tabs = st.tabs(["📋 Athlete Roster", "🏅 Personal Bests", "📈 Progression", "🎯 Qualification Status", "📊 Event Analysis"])

    with tabs[0]:  # Roster
        st.subheader("Saudi National Team Roster")

        # Build roster table
        roster_data = []
        for name in sorted(saudi_df['FullName'].dropna().unique()):
            athlete_data = saudi_df[saudi_df['FullName'] == name]
            events = athlete_data[disc_col].nunique() if disc_col in athlete_data.columns else 0
            results = len(athlete_data)
            gender = athlete_data['Gender'].iloc[0] if 'Gender' in athlete_data.columns else 'Unknown'

            # Age info
            if 'AthleteResultAge' in athlete_data.columns:
                ages = athlete_data['AthleteResultAge'].dropna()
                age = f"{int(ages.max())}" if not ages.empty else "N/A"
            else:
                age = "N/A"

            # Best event
            if disc_col in athlete_data.columns and 'time_seconds' in athlete_data.columns:
                best_idx = athlete_data['time_seconds'].idxmin()
                if pd.notna(best_idx):
                    best_event = athlete_data.loc[best_idx, disc_col]
                    best_time = athlete_data.loc[best_idx, 'Time']
                else:
                    best_event, best_time = "N/A", "N/A"
            else:
                best_event, best_time = "N/A", "N/A"

            roster_data.append({
                'Athlete': name,
                'Gender': gender,
                'Age': age,
                'Events': events,
                'Results': results,
                'Best Event': best_event,
                'Best Time': best_time
            })

        roster_df = pd.DataFrame(roster_data)
        st.dataframe(roster_df, width='stretch', height=500)

        # Download button
        csv = roster_df.to_csv(index=False)
        st.download_button(
            "📥 Download Roster CSV",
            csv,
            "saudi_athletes_roster.csv",
            "text/csv"
        )

    with tabs[1]:  # Personal Bests
        st.subheader("Personal Bests by Athlete")

        selected_athlete = st.selectbox(
            "Select Athlete",
            sorted(saudi_df['FullName'].dropna().unique()),
            key="saudi_pb_athlete"
        )

        if selected_athlete:
            athlete_df = saudi_df[saudi_df['FullName'] == selected_athlete]

            # Athlete profile card
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown(f"""
                <div class="saudi-athlete">
                    <h3>{selected_athlete}</h3>
                    <p>🇸🇦 Saudi Arabia</p>
                </div>
                """, unsafe_allow_html=True)

                if 'Gender' in athlete_df.columns:
                    st.write(f"**Gender:** {athlete_df['Gender'].iloc[0]}")
                if 'AthleteResultAge' in athlete_df.columns:
                    age = athlete_df['AthleteResultAge'].max()
                    st.write(f"**Current Age:** {int(age) if pd.notna(age) else 'N/A'}")
                st.write(f"**Total Results:** {len(athlete_df)}")

            with col2:
                # Personal bests table
                if disc_col in athlete_df.columns:
                    pb_data = []
                    for event in athlete_df[disc_col].dropna().unique():
                        event_df = athlete_df[athlete_df[disc_col] == event]
                        if 'time_seconds' in event_df.columns:
                            best_idx = event_df['time_seconds'].idxmin()
                            if pd.notna(best_idx):
                                best_row = event_df.loc[best_idx]
                                pb_data.append({
                                    'Event': event,
                                    'Personal Best': best_row['Time'],
                                    'Time (s)': round(best_row['time_seconds'], 2),
                                    'Competition': best_row.get('competition_name', 'N/A'),
                                    'Date': best_row.get('date_from', 'N/A')
                                })

                    if pb_data:
                        pb_df = pd.DataFrame(pb_data).sort_values('Time (s)')
                        st.dataframe(pb_df, width='stretch')

    with tabs[2]:  # Progression
        st.subheader("Performance Progression")

        col1, col2 = st.columns(2)

        with col1:
            prog_athlete = st.selectbox(
                "Select Athlete",
                sorted(saudi_df['FullName'].dropna().unique()),
                key="saudi_prog_athlete"
            )

        with col2:
            if prog_athlete:
                athlete_events = saudi_df[saudi_df['FullName'] == prog_athlete][disc_col].dropna().unique()
                prog_event = st.selectbox("Select Event", sorted(athlete_events), key="saudi_prog_event")

        if prog_athlete and prog_event:
            prog_df = saudi_df[(saudi_df['FullName'] == prog_athlete) & (saudi_df[disc_col] == prog_event)].copy()
            prog_df = prog_df.dropna(subset=['time_seconds'])

            if not prog_df.empty:
                # Sort by date or create index
                if 'date_from' in prog_df.columns:
                    prog_df['date'] = pd.to_datetime(prog_df['date_from'], errors='coerce')
                    prog_df = prog_df.sort_values('date')
                    x_data = prog_df['date'].tolist()
                    x_label = "Date"
                else:
                    x_data = list(range(len(prog_df)))
                    x_label = "Race #"

                # Running PB
                prog_df['pb'] = prog_df['time_seconds'].cummin()

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=prog_df['time_seconds'].tolist(),
                    mode='markers+lines',
                    name='Race Time',
                    marker=dict(size=12, color='#006C35'),
                    line=dict(color='#006C35')
                ))

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=prog_df['pb'].tolist(),
                    mode='lines',
                    name='Personal Best',
                    line=dict(dash='dash', color='#FFD700', width=3)
                ))

                fig.update_layout(
                    title=f"{prog_athlete} - {prog_event}",
                    xaxis_title=x_label,
                    yaxis_title="Time (seconds)",
                    height=400,
                    template="plotly_white"
                )

                st.plotly_chart(fig, width='stretch')

                # Stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Personal Best", seconds_to_time(prog_df['time_seconds'].min()))
                with col2:
                    st.metric("First Time", seconds_to_time(prog_df['time_seconds'].iloc[0]))
                with col3:
                    improvement = prog_df['time_seconds'].iloc[0] - prog_df['time_seconds'].min()
                    st.metric("Total Improvement", f"{improvement:.2f}s")
                with col4:
                    st.metric("Races", len(prog_df))

    with tabs[3]:  # Qualification Status
        st.subheader("Qualification Status for Major Competitions")

        if ANALYTICS_AVAILABLE:
            qual_athlete = st.selectbox(
                "Select Athlete",
                sorted(saudi_df['FullName'].dropna().unique()),
                key="saudi_qual_athlete"
            )

            if qual_athlete:
                athlete_df = saudi_df[saudi_df['FullName'] == qual_athlete]

                qual_results = []
                for event in athlete_df[disc_col].dropna().unique():
                    event_df = athlete_df[athlete_df[disc_col] == event]
                    if 'time_seconds' in event_df.columns:
                        best_time = event_df['time_seconds'].min()
                        gender = event_df['Gender'].iloc[0] if 'Gender' in event_df.columns else 'Men'

                        status = check_qualification_status(best_time, event, gender)

                        for comp, details in status.items():
                            qual_results.append({
                                'Event': event,
                                'PB': seconds_to_time(best_time),
                                'Competition': comp,
                                'Standard': seconds_to_time(details['standard']),
                                'Qualified': '✅' if details['qualified'] else '❌',
                                'Gap': f"{details['gap']:+.2f}s"
                            })

                if qual_results:
                    qual_df = pd.DataFrame(qual_results)
                    st.dataframe(qual_df, width='stretch')

                    # Summary
                    qualified_count = sum(1 for r in qual_results if r['Qualified'] == '✅')
                    st.info(f"**{qual_athlete}** has {qualified_count} qualifying times across all events/competitions")
        else:
            st.warning("Analytics module not available for qualification tracking")

    with tabs[4]:  # Event Analysis
        st.subheader("Saudi Performance by Event")

        if disc_col in saudi_df.columns:
            event_stats = []
            for event in saudi_df[disc_col].dropna().unique():
                event_df = saudi_df[saudi_df[disc_col] == event]
                if 'time_seconds' in event_df.columns:
                    event_stats.append({
                        'Event': event,
                        'Athletes': event_df['FullName'].nunique(),
                        'Results': len(event_df),
                        'Best Time': seconds_to_time(event_df['time_seconds'].min()),
                        'Best Athlete': event_df.loc[event_df['time_seconds'].idxmin(), 'FullName'] if not event_df['time_seconds'].isna().all() else 'N/A'
                    })

            if event_stats:
                event_df = pd.DataFrame(event_stats).sort_values('Athletes', ascending=False)
                st.dataframe(event_df, width='stretch')

                # Visualization
                fig = px.bar(
                    event_df.head(15),
                    x='Event',
                    y='Athletes',
                    color='Results',
                    title="Saudi Athletes by Event"
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, width='stretch')


def show_data_explorer(df: pd.DataFrame):
    """User-friendly data explorer"""
    st.header("🔍 Data Explorer")

    st.markdown("""
    Browse, search, and filter the swimming database with ease.
    """)

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    # Filters in sidebar-style columns
    st.subheader("🎛️ Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Athlete search
        athlete_search = st.text_input("🔎 Search Athlete", placeholder="Type name...")

    with col2:
        # Country filter
        countries = ['All'] + sorted(df['NAT'].dropna().unique().tolist()) if 'NAT' in df.columns else ['All']
        selected_country = st.selectbox("🌍 Country", countries)

    with col3:
        # Event filter
        events = ['All'] + sorted(df[disc_col].dropna().unique().tolist()) if disc_col in df.columns else ['All']
        selected_event = st.selectbox("🏊 Event", events)

    with col4:
        # Gender filter
        genders = ['All'] + sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else ['All']
        selected_gender = st.selectbox("👤 Gender", genders)

    # Additional filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Year filter
        if 'date_from' in df.columns:
            df['year'] = pd.to_datetime(df['date_from'], errors='coerce').dt.year
        if 'year' in df.columns:
            years = sorted(df['year'].dropna().unique())
            year_range = st.select_slider(
                "📅 Year Range",
                options=years,
                value=(min(years), max(years)) if years else (2000, 2024)
            )
        else:
            year_range = None

    with col2:
        # Age filter
        if 'AthleteResultAge' in df.columns:
            age_range = st.slider("🎂 Age Range", 10, 50, (10, 50))
        else:
            age_range = None

    with col3:
        # Finals only
        finals_only = st.checkbox("🏆 Finals Only", value=False)

    with col4:
        # Results limit
        limit = st.number_input("📊 Max Results", min_value=10, max_value=10000, value=500)

    # Apply filters
    filtered_df = df.copy()

    if athlete_search:
        filtered_df = filtered_df[filtered_df['FullName'].str.contains(athlete_search, case=False, na=False)]

    if selected_country != 'All' and 'NAT' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['NAT'] == selected_country]

    if selected_event != 'All' and disc_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[disc_col] == selected_event]

    if selected_gender != 'All' and 'Gender' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]

    if year_range and 'year' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]

    if age_range and 'AthleteResultAge' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['AthleteResultAge'] >= age_range[0]) & (filtered_df['AthleteResultAge'] <= age_range[1])]

    if finals_only and 'Heat Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Heat Category'].str.contains('Final', case=False, na=False)]

    # Sort and limit
    if 'time_seconds' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('time_seconds').head(limit)
    else:
        filtered_df = filtered_df.head(limit)

    # Show results
    st.markdown("---")
    st.subheader(f"📋 Results ({len(filtered_df):,} records)")

    # Select columns to display
    display_cols = ['FullName', 'NAT', 'Time', disc_col, 'Gender']
    optional_cols = ['AthleteResultAge', 'Heat Category', 'competition_name', 'date_from', 'Rank', 'pacing_type']

    for col in optional_cols:
        if col in filtered_df.columns:
            display_cols.append(col)

    display_cols = [c for c in display_cols if c in filtered_df.columns]

    # Rename for display
    rename_map = {
        'FullName': 'Athlete',
        'NAT': 'Country',
        disc_col: 'Event',
        'AthleteResultAge': 'Age',
        'Heat Category': 'Round',
        'competition_name': 'Competition',
        'date_from': 'Date',
        'pacing_type': 'Pacing'
    }

    display_df = filtered_df[display_cols].rename(columns=rename_map)

    st.dataframe(display_df.reset_index(drop=True), width='stretch', height=500)

    # Download filtered data
    col1, col2 = st.columns(2)

    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv,
            "swimming_data_filtered.csv",
            "text/csv"
        )

    with col2:
        # Quick stats on filtered data
        st.markdown("**Quick Stats:**")
        st.write(f"- Athletes: {filtered_df['FullName'].nunique()}")
        st.write(f"- Countries: {filtered_df['NAT'].nunique() if 'NAT' in filtered_df.columns else 'N/A'}")
        if 'time_seconds' in filtered_df.columns:
            st.write(f"- Best Time: {seconds_to_time(filtered_df['time_seconds'].min())}")


if __name__ == "__main__":
    main()
