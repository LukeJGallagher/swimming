"""
Elite Coaching Dashboard
Evidence-based analytics dashboard for Team Saudi coaching staff

Features:
- Talent Development Tracking
- Pacing Strategy Analysis
- World Record Benchmarking
- Heats-to-Finals Progression
- Competitor Intelligence
"""

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    print("Then run with: streamlit run coaching_dashboard.py")
    exit(1)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from coaching_analytics import (
    TalentDevelopmentTracker,
    AdvancedPacingAnalyzer,
    RaceRoundAnalyzer,
    CompetitorIntelligence,
    CoachingReportGenerator,
    PredictivePerformanceModel,
    AdvancedKPIAnalyzer,
    load_all_results,
    get_world_records,
    get_entry_standards,
    format_time,
    WORLD_RECORDS_LCM,
    WORLD_RECORDS_SCM,
    PEAK_PERFORMANCE_AGES,
    ELITE_BENCHMARKS,
    COMPETITION_BENCHMARKS,
    AGE_PROGRESSION_BENCHMARKS,
    IMPROVEMENT_EXPECTATIONS,
    LA_2028_OQT,
    WORLD_CHAMPS_2027_ENTRY,
    ASIAN_GAMES_2026_ENTRY,
    ASIAN_GAMES_2026_MEDAL,
)
from enhanced_swimming_scraper import SplitTimeAnalyzer

# Team Saudi Colors
TEAM_SAUDI_COLORS = {
    'primary_teal': '#007167',
    'gold_accent': '#a08e66',
    'dark_teal': '#005a51',
    'white': '#ffffff',
    'light_gray': '#f8f9fa'
}

# Page config
st.set_page_config(
    page_title="Team Saudi - Coaching Analytics",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Team Saudi branding
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%);
        color: {TEAM_SAUDI_COLORS['white']};
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }}
    .main-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }}
    .main-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%);
        color: {TEAM_SAUDI_COLORS['white']};
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {TEAM_SAUDI_COLORS['gold_accent']};
    }}
    .metric-label {{
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    .gold-highlight {{
        color: {TEAM_SAUDI_COLORS['gold_accent']};
        font-weight: bold;
    }}
    .elite-badge {{
        background-color: {TEAM_SAUDI_COLORS['gold_accent']};
        color: {TEAM_SAUDI_COLORS['white']};
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {TEAM_SAUDI_COLORS['light_gray']};
        border-radius: 4px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {TEAM_SAUDI_COLORS['primary_teal']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="Loading swimming data...")
def load_data():
    """Load all available swimming data. Cached for 1 hour."""
    import pandas as pd
    from pathlib import Path
    import os

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Try local cache first (fastest) - use absolute path
    cache_file = script_dir / ".cache" / "swimming_data.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                return df
        except Exception as e:
            st.warning(f"Cache read failed: {e}")

    # Change to script directory before loading
    original_dir = os.getcwd()
    try:
        os.chdir(script_dir)
        return load_all_results()
    finally:
        os.chdir(original_dir)


def create_team_saudi_chart_theme():
    """Return Plotly layout theme for Team Saudi."""
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Inter, sans-serif', 'color': '#333'},
        'colorway': [
            TEAM_SAUDI_COLORS['primary_teal'],
            TEAM_SAUDI_COLORS['gold_accent'],
            TEAM_SAUDI_COLORS['dark_teal'],
            '#FFB800',
            '#0077B6',
            '#6c757d'
        ]
    }


def export_data_summary(df):
    """Export data summary report."""
    report = []
    report.append("=" * 70)
    report.append("TEAM SAUDI - SWIMMING ANALYTICS DATA SUMMARY")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    report.append(f"\nTOTAL RECORDS: {len(df):,}")
    report.append(f"UNIQUE ATHLETES: {df['FullName'].nunique():,}")

    if 'year' in df.columns:
        report.append(f"YEAR RANGE: {int(df['year'].min())} - {int(df['year'].max())}")

    if 'NAT' in df.columns:
        report.append(f"COUNTRIES: {df['NAT'].nunique():,}")

    if 'discipline_name' in df.columns:
        report.append(f"EVENTS: {df['discipline_name'].nunique():,}")

    if 'competition_name' in df.columns:
        report.append(f"COMPETITIONS: {df['competition_name'].nunique():,}")

    # Medal summary
    if 'MedalTag' in df.columns:
        medals = df['MedalTag'].value_counts()
        report.append(f"\nMEDAL SUMMARY:")
        report.append(f"  Gold:   {medals.get('G', 0):,}")
        report.append(f"  Silver: {medals.get('S', 0):,}")
        report.append(f"  Bronze: {medals.get('B', 0):,}")

    # Pacing data summary
    if 'pacing_type' in df.columns:
        pacing = df['pacing_type'].value_counts()
        report.append(f"\nPACING STRATEGIES:")
        for strategy, count in pacing.head(5).items():
            report.append(f"  {strategy}: {count:,}")

    # Split data availability
    if 'lap_times_json' in df.columns:
        with_splits = df['lap_times_json'].notna().sum()
        report.append(f"\nSPLIT DATA:")
        report.append(f"  Results with splits: {with_splits:,} ({with_splits/len(df)*100:.1f}%)")

    report.append("\n" + "=" * 70)
    report_text = "\n".join(report)

    st.sidebar.download_button(
        "Download Report",
        report_text,
        file_name=f"swimming_data_summary_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )


def generate_athlete_report(df, athlete_name: str) -> str:
    """Generate comprehensive athlete report."""
    tracker = TalentDevelopmentTracker(df)
    pacing = AdvancedPacingAnalyzer()
    kpi = AdvancedKPIAnalyzer(df)

    report = []
    report.append("=" * 70)
    report.append(f"ATHLETE PERFORMANCE REPORT: {athlete_name}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    # Competition history
    comp_age = tracker.calculate_competition_age(athlete_name)
    if 'error' not in comp_age:
        report.append(f"\nCOMPETITION HISTORY:")
        report.append(f"  First Competition: {comp_age['first_competition']}")
        report.append(f"  Years Competing: {comp_age['competition_years']}")
        report.append(f"  Total Races: {comp_age['total_races']}")
        report.append(f"  Progress to Elite: {comp_age['progress_to_elite_pct']}%")

    # World record comparison
    wr_analysis = tracker.calculate_world_record_percentage(athlete_name)
    if wr_analysis:
        report.append(f"\nWORLD RECORD BENCHMARKING (Top 5):")
        for wr in wr_analysis[:5]:
            elite = "*ELITE*" if wr['is_elite_level'] else ""
            report.append(f"  {wr['event'][:35]:35s} {wr['wr_percentage']:>6.2f}% of WR {elite}")

    # Reaction time
    rt_stats = kpi.calculate_reaction_time_stats(athlete_name)
    if 'error' not in rt_stats:
        report.append(f"\nREACTION TIME:")
        report.append(f"  Average: {rt_stats['avg_reaction_time']:.3f}s")
        report.append(f"  Best: {rt_stats['best_reaction_time']:.3f}s")
        report.append(f"  Rating: {rt_stats['rating']}")

    # Race efficiency
    efficiency = kpi.analyze_race_efficiency(athlete_name)
    if 'avg_consistency_cv' in efficiency:
        report.append(f"\nRACE EFFICIENCY:")
        report.append(f"  Avg Consistency (CV): {efficiency['avg_consistency_cv']:.2f}%")
        report.append(f"  Elite Consistency: {'Yes' if efficiency['is_elite_consistency'] else 'No'}")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


def main():
    # Get script directory for asset paths (works on Streamlit Cloud)
    script_dir = Path(__file__).parent.resolve()

    # Header with logo
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        logo_path = script_dir / "Saudilogo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=100)
    with col_title:
        st.markdown("""
        <div class="main-header">
            <h1>Elite Coaching Analytics</h1>
            <p>Evidence-Based Performance Analysis for Team Saudi</p>
        </div>
        """, unsafe_allow_html=True)

    # Load data
    results_df = load_data()

    if results_df.empty:
        st.error("No data found. Please run the scraper first:")
        st.code("python enhanced_swimming_scraper.py")
        return

    # Sidebar navigation
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%);
                padding: 1rem; border-radius: 8px; color: white; text-align: center; margin-bottom: 1rem;">
        <strong>Team Saudi</strong><br>
        <span style="font-size: 0.8rem;">Coaching Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navigate", [
        "📊 Dashboard Overview",
        "🏯 Road to Nagoya 2026",
        "🏅 Road to LA 2028",
        "👤 Athlete Profiles",
        "📈 Talent Development",
        "🎯 Pacing Analysis",
        "🏅 Competition Intel",
        "📋 Race Preparation",
        "🚀 Advanced Analytics",
        "🏆 Performance Benchmarks",
        "🔬 Research Insights"
    ])

    # Course Type Filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Course Type**")

    # Detect course type from competition name - "(25m)" indicates short course
    if 'competition_name' in results_df.columns:
        results_df['is_short_course'] = results_df['competition_name'].str.contains(
            r'\(25m\)', case=False, na=False, regex=True
        )
        scm_count = results_df['is_short_course'].sum()
        lcm_count = len(results_df) - scm_count
    else:
        results_df['is_short_course'] = False
        scm_count = 0
        lcm_count = len(results_df)

    course_options = {
        f"All Courses ({len(results_df):,})": "all",
        f"Long Course 50m ({lcm_count:,})": "lcm",
        f"Short Course 25m ({scm_count:,})": "scm"
    }

    course_selection = st.sidebar.selectbox(
        "Select Course Type",
        options=list(course_options.keys()),
        index=0,
        help="LCM = Long Course Meters (50m pool), SCM = Short Course Meters (25m pool)"
    )

    # Filter data based on course selection
    selected_course = course_options[course_selection]
    if selected_course == "lcm":
        results_df = results_df[~results_df['is_short_course']].copy()
    elif selected_course == "scm":
        results_df = results_df[results_df['is_short_course']].copy()

    # Show current filter status
    if selected_course != "all":
        course_label = "Long Course (50m)" if selected_course == "lcm" else "Short Course (25m)"
        st.sidebar.info(f"Showing: {course_label}")

    # Data summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Summary**")
    st.sidebar.metric("Total Results", f"{len(results_df):,}")
    st.sidebar.metric("Athletes", f"{results_df['FullName'].nunique():,}")
    st.sidebar.metric("Years of Data", f"{results_df['year'].nunique() if 'year' in results_df.columns else 'N/A'}")

    # Export section
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Export Tools**")
    if st.sidebar.button("Export Data Summary"):
        export_data_summary(results_df)

    # Route to pages (pass course_type for WR benchmarking)
    if page == "📊 Dashboard Overview":
        show_overview(results_df, selected_course)
    elif page == "🏯 Road to Nagoya 2026":
        show_road_to_nagoya(results_df, selected_course)
    elif page == "🏅 Road to LA 2028":
        show_road_to_la(results_df, selected_course)
    elif page == "👤 Athlete Profiles":
        show_athlete_profiles(results_df, selected_course)
    elif page == "📈 Talent Development":
        show_talent_development(results_df, selected_course)
    elif page == "🎯 Pacing Analysis":
        show_pacing_analysis(results_df, selected_course)
    elif page == "🏅 Competition Intel":
        show_competition_intel(results_df, selected_course)
    elif page == "📋 Race Preparation":
        show_race_preparation(results_df, selected_course)
    elif page == "🚀 Advanced Analytics":
        show_advanced_analytics(results_df, selected_course)
    elif page == "🏆 Performance Benchmarks":
        show_performance_benchmarks(results_df, selected_course)
    elif page == "🔬 Research Insights":
        show_research_insights()


def show_overview(df, course_type='all'):
    """Dashboard overview with key metrics."""
    st.header("Dashboard Overview")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with_splits = df['splits_json'].notna().sum() if 'splits_json' in df.columns else 0
        st.metric("Results with Splits", f"{with_splits:,}")

    with col2:
        competitions = df['competition_name'].nunique() if 'competition_name' in df.columns else 0
        st.metric("Competitions", f"{competitions:,}")

    with col3:
        countries = df['NAT'].nunique() if 'NAT' in df.columns else 0
        st.metric("Countries", f"{countries:,}")

    with col4:
        medals = df['MedalTag'].notna().sum() if 'MedalTag' in df.columns else 0
        st.metric("Medal Performances", f"{medals:,}")

    with col5:
        events = df['discipline_name'].nunique() if 'discipline_name' in df.columns else 0
        st.metric("Events", f"{events:,}")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pacing Strategy Distribution")
        if 'pacing_type' in df.columns:
            pacing_data = df['pacing_type'].value_counts()
            fig = px.pie(
                values=pacing_data.values,
                names=pacing_data.index,
                color_discrete_sequence=[
                    TEAM_SAUDI_COLORS['primary_teal'],
                    TEAM_SAUDI_COLORS['gold_accent'],
                    TEAM_SAUDI_COLORS['dark_teal'],
                    '#FFB800',
                    '#6c757d'
                ]
            )
            fig.update_layout(**create_team_saudi_chart_theme())
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Results by Year")
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='count')
            fig = px.bar(
                yearly, x='year', y='count',
                color_discrete_sequence=[TEAM_SAUDI_COLORS['primary_teal']]
            )
            fig.update_layout(**create_team_saudi_chart_theme())
            st.plotly_chart(fig, use_container_width=True)

    # Top performing countries
    st.subheader("Medal Leaders")
    if 'MedalTag' in df.columns and 'NAT' in df.columns:
        medals_df = df[df['MedalTag'].notna()].copy()
        medal_counts = medals_df.groupby(['NAT', 'MedalTag']).size().unstack(fill_value=0)

        if not medal_counts.empty:
            medal_counts['Total'] = medal_counts.sum(axis=1)
            medal_counts = medal_counts.sort_values('Total', ascending=False).head(15)

            fig = go.Figure()

            if 'G' in medal_counts.columns:
                fig.add_trace(go.Bar(name='Gold', x=medal_counts.index, y=medal_counts['G'],
                                    marker_color='#FFD700'))
            if 'S' in medal_counts.columns:
                fig.add_trace(go.Bar(name='Silver', x=medal_counts.index, y=medal_counts['S'],
                                    marker_color='#C0C0C0'))
            if 'B' in medal_counts.columns:
                fig.add_trace(go.Bar(name='Bronze', x=medal_counts.index, y=medal_counts['B'],
                                    marker_color='#CD7F32'))

            fig.update_layout(barmode='stack', **create_team_saudi_chart_theme())
            st.plotly_chart(fig, use_container_width=True)


def show_road_to_nagoya(df, course_type='all'):
    """Road to Nagoya 2026 Asian Games - Swimming Qualification Tracker."""

    # Event details
    EVENT_DATE = datetime(2026, 9, 19)
    days_to_go = (EVENT_DATE - datetime.now()).days

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%);
                padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin: 0;">🏯 Road to Nagoya 2026</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            20th Asian Games • September 19 - October 4, 2026
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: {TEAM_SAUDI_COLORS['gold_accent']}; color: white; padding: 0.5rem 1.5rem;
                        border-radius: 20px; font-size: 1.5rem; font-weight: bold;">
                {days_to_go} Days to Go
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key info columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Host City</div>
            <div class="metric-value" style="font-size: 1.3rem;">Nagoya, Japan</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Venue</div>
            <div class="metric-value" style="font-size: 1.3rem;">Nippon Gaishi Hall</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pool Type</div>
            <div class="metric-value" style="font-size: 1.3rem;">50m LCM</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Swimming Events</div>
            <div class="metric-value" style="font-size: 1.3rem;">42 Events</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Saudi Athletes",
        "🏆 Asian Benchmarks",
        "📈 Qualification Standards",
        "🎯 Target Times"
    ])

    with tab1:
        st.subheader("Saudi Swimming Squad Analysis")

        # Filter Saudi athletes
        saudi_df = df[df['NAT'] == 'KSA'].copy() if 'NAT' in df.columns else pd.DataFrame()

        if saudi_df.empty:
            st.info("No Saudi (KSA) athletes found in the database. Try searching for specific athletes below.")

            # Manual athlete search
            all_athletes = sorted(df['FullName'].dropna().unique())
            selected_athletes = st.multiselect(
                "Select athletes to track for Nagoya 2026:",
                all_athletes,
                help="Select Team Saudi athletes to analyze"
            )

            if selected_athletes:
                saudi_df = df[df['FullName'].isin(selected_athletes)].copy()

        if not saudi_df.empty:
            # Recent form (last 2 years)
            if 'year' in saudi_df.columns:
                recent_df = saudi_df[saudi_df['year'] >= 2024]
            else:
                recent_df = saudi_df

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Athlete Performance Summary**")
                athlete_summary = recent_df.groupby('FullName').agg({
                    'discipline_name': 'nunique',
                    'competition_name': 'nunique'
                }).reset_index()
                athlete_summary.columns = ['Athlete', 'Events', 'Competitions']

                if 'Points' in recent_df.columns:
                    points_df = recent_df.groupby('FullName')['Points'].max().reset_index()
                    points_df.columns = ['Athlete', 'Best FINA Pts']
                    athlete_summary = athlete_summary.merge(points_df, on='Athlete', how='left')

                st.dataframe(athlete_summary, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("**Best Times by Event (Recent)**")
                if 'Time' in recent_df.columns and 'discipline_name' in recent_df.columns:
                    # Get best time with competition info
                    best_times_list = []
                    for (athlete, event), group in recent_df.groupby(['FullName', 'discipline_name']):
                        if group['Time'].notna().any():
                            # Sort by time to get best
                            sorted_group = group.sort_values('Time')
                            best_row = sorted_group.iloc[0]
                            comp_name = best_row.get('competition_name', '')[:25] if pd.notna(best_row.get('competition_name')) else ''
                            year = best_row.get('year', '')
                            best_times_list.append({
                                'Athlete': athlete,
                                'Event': event,
                                'Best Time': best_row['Time'],
                                'Competition': f"{comp_name} ({year})" if comp_name else str(year)
                            })
                    if best_times_list:
                        best_times = pd.DataFrame(best_times_list)
                        st.dataframe(best_times.head(20), use_container_width=True, hide_index=True)

            # Performance progression chart
            st.markdown("---")
            st.subheader("Performance Progression")

            athlete_list = saudi_df['FullName'].unique()
            selected_athlete = st.selectbox("Select Athlete", athlete_list, key="nagoya_athlete")

            if selected_athlete:
                athlete_data = saudi_df[saudi_df['FullName'] == selected_athlete]

                # Event selection
                events = athlete_data['discipline_name'].unique()
                selected_event = st.selectbox("Select Event", events, key="nagoya_event")

                if selected_event:
                    event_data = athlete_data[athlete_data['discipline_name'] == selected_event].copy()

                    if 'Time' in event_data.columns and 'year' in event_data.columns:
                        # Convert time to seconds for charting
                        def time_to_sec(t):
                            if pd.isna(t):
                                return None
                            try:
                                t = str(t)
                                if ':' in t:
                                    parts = t.split(':')
                                    return float(parts[0]) * 60 + float(parts[1])
                                return float(t)
                            except:
                                return None

                        event_data['time_seconds'] = event_data['Time'].apply(time_to_sec)
                        event_data = event_data.dropna(subset=['time_seconds'])

                        if not event_data.empty:
                            fig = px.scatter(
                                event_data,
                                x='year',
                                y='time_seconds',
                                color='competition_name' if 'competition_name' in event_data.columns else None,
                                title=f"{selected_athlete} - {selected_event} Progression",
                                labels={'time_seconds': 'Time (seconds)', 'year': 'Year'}
                            )

                            # Add best time line
                            best_time = event_data['time_seconds'].min()
                            fig.add_hline(
                                y=best_time,
                                line_dash="dash",
                                line_color=TEAM_SAUDI_COLORS['gold_accent'],
                                annotation_text=f"PB: {best_time:.2f}s"
                            )

                            fig.update_layout(**create_team_saudi_chart_theme())
                            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Asian Swimming Benchmarks")

        st.markdown("""
        Analyze top Asian swimmers to understand qualification pace and medal contention times.
        """)

        # Top Asian nations
        asian_nations = ['JPN', 'CHN', 'KOR', 'HKG', 'SGP', 'THA', 'MAS', 'IND', 'PHL', 'VIE', 'INA', 'KSA', 'UAE', 'QAT', 'KUW', 'BRN', 'OMA']

        asian_df = df[df['NAT'].isin(asian_nations)].copy() if 'NAT' in df.columns else pd.DataFrame()

        if not asian_df.empty:
            # Filter to recent data
            if 'year' in asian_df.columns:
                recent_asian = asian_df[asian_df['year'] >= 2023]
            else:
                recent_asian = asian_df

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Medal Leaders by Nation (Recent)**")
                if 'MedalTag' in recent_asian.columns:
                    medals = recent_asian[recent_asian['MedalTag'].notna()].groupby('NAT').size().reset_index(name='Medals')
                    medals = medals.sort_values('Medals', ascending=False).head(10)

                    fig = px.bar(
                        medals, x='NAT', y='Medals',
                        color_discrete_sequence=[TEAM_SAUDI_COLORS['primary_teal']]
                    )
                    fig.update_layout(**create_team_saudi_chart_theme(), title="Top Asian Nations - Recent Medals")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Top FINA Points by Event (Top 50)**")
                if 'Points' in recent_asian.columns and 'discipline_name' in recent_asian.columns:
                    # Get top FINA points with athlete, time, and competition info
                    top_points_list = []
                    for event, group in recent_asian.groupby('discipline_name'):
                        if group['Points'].notna().any():
                            # Get top performer for this event
                            best_row = group.loc[group['Points'].idxmax()]
                            top_points_list.append({
                                'Event': event,
                                'Athlete': best_row.get('FullName', 'N/A'),
                                'NAT': best_row.get('NAT', ''),
                                'Time': best_row.get('Time', 'N/A'),
                                'Points': int(best_row['Points']) if pd.notna(best_row['Points']) else 0
                            })
                    if top_points_list:
                        top_points = pd.DataFrame(top_points_list)
                        top_points = top_points.sort_values('Points', ascending=False).head(50)
                        st.dataframe(top_points, use_container_width=True, hide_index=True)

            # Event-by-event analysis
            st.markdown("---")
            st.subheader("Event Analysis - Asian Leaders")

            events = recent_asian['discipline_name'].dropna().unique()
            selected_event = st.selectbox("Select Event for Detailed Analysis", sorted(events), key="asian_event")

            if selected_event:
                event_df = recent_asian[recent_asian['discipline_name'] == selected_event]

                # Top performers
                if 'Time' in event_df.columns:
                    # Convert Time to numeric for sorting (handle MM:SS.ss format)
                    def time_to_seconds(t):
                        if pd.isna(t):
                            return float('inf')
                        try:
                            t = str(t)
                            if ':' in t:
                                parts = t.split(':')
                                return float(parts[0]) * 60 + float(parts[1])
                            return float(t)
                        except:
                            return float('inf')

                    event_df_sorted = event_df.copy()
                    event_df_sorted['time_numeric'] = event_df_sorted['Time'].apply(time_to_seconds)
                    event_df_sorted = event_df_sorted[event_df_sorted['time_numeric'] < float('inf')]

                    if not event_df_sorted.empty:
                        top_performers = event_df_sorted.nsmallest(10, 'time_numeric')[['FullName', 'NAT', 'Time', 'competition_name', 'year']].copy()
                        top_performers.columns = ['Athlete', 'Nation', 'Time', 'Competition', 'Year']
                        st.dataframe(top_performers, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Official Entry Standards Comparison")

        # Show warning if viewing short course data
        if course_type == 'scm':
            st.warning("⚠️ **Note:** Olympic Games, World Championships, and Asian Games are all **Long Course (50m pool)** competitions. The entry standards below apply to Long Course times only. Short course times cannot be directly compared to these standards.")
            st.markdown("---")

        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #007167;">
            <strong>What It Takes to Compete:</strong> Compare entry standards across major competitions.
            All major championships (Olympics, Worlds, Asian Games) are held in 50m pools (Long Course).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Gender and event selection
        col_gender, col_event = st.columns(2)

        with col_gender:
            qual_gender = st.selectbox("Gender", ["Men", "Women"], key="qual_gender")

        # Get events for selected gender
        gender_events = [e.replace(f"{qual_gender} ", "") for e in ASIAN_GAMES_2026_ENTRY.keys() if e.startswith(qual_gender)]

        with col_event:
            qual_event = st.selectbox("Select Event", sorted(gender_events), key="qual_event")

        event_key = f"{qual_gender} {qual_event}"

        # Get standards for this event
        asian_gold = ASIAN_GAMES_2026_ENTRY.get(event_key)
        asian_medal = ASIAN_GAMES_2026_MEDAL.get(event_key)
        worlds_entry = WORLD_CHAMPS_2027_ENTRY.get(event_key)
        olympic_oqt = LA_2028_OQT.get(event_key)
        world_record = WORLD_RECORDS_LCM.get(event_key)

        if asian_gold:
            st.markdown(f"### {event_key}")

            # Display standards comparison
            st.markdown("#### Entry Standards Comparison")

            standards_data = []
            if world_record:
                standards_data.append({
                    'Level': '🌍 World Record',
                    'Time': format_time(world_record),
                    'Seconds': world_record,
                    'Description': 'Current world best'
                })
            if olympic_oqt:
                standards_data.append({
                    'Level': '🏅 LA 2028 Olympic OQT',
                    'Time': format_time(olympic_oqt),
                    'Seconds': olympic_oqt,
                    'Description': 'Olympic Qualifying Time'
                })
            if worlds_entry:
                standards_data.append({
                    'Level': '🌐 World Champs 2027 Entry',
                    'Time': format_time(worlds_entry),
                    'Seconds': worlds_entry,
                    'Description': '"A" Standard entry'
                })
            if asian_gold:
                standards_data.append({
                    'Level': '🥇 Nagoya 2026 Gold Pace',
                    'Time': format_time(asian_gold),
                    'Seconds': asian_gold,
                    'Description': 'Expected gold medal time'
                })
            if asian_medal:
                standards_data.append({
                    'Level': '🥉 Nagoya 2026 Medal Pace',
                    'Time': format_time(asian_medal),
                    'Seconds': asian_medal,
                    'Description': 'Bronze medal contention'
                })

            # Display as colored cards
            cols = st.columns(len(standards_data))
            colors = ['#FFD700', '#007167', '#0077B6', '#007167', '#CD7F32']

            for i, (col, std) in enumerate(zip(cols, standards_data)):
                with col:
                    color = colors[i % len(colors)]
                    st.markdown(f"""
                    <div style="background: {color}; padding: 1rem; border-radius: 8px; text-align: center; color: white; height: 120px;">
                        <p style="margin: 0; font-size: 0.75rem; opacity: 0.9;">{std['Level']}</p>
                        <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold;">{std['Time']}</p>
                        <p style="margin: 0; font-size: 0.7rem; opacity: 0.8;">{std['Description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Gap analysis chart
            st.markdown("---")
            st.markdown("#### Standards Ladder")

            fig = go.Figure()

            # Add horizontal bar for each standard
            for i, std in enumerate(reversed(standards_data)):
                fig.add_trace(go.Bar(
                    y=[std['Level']],
                    x=[std['Seconds']],
                    orientation='h',
                    marker_color=colors[len(standards_data) - 1 - i],
                    text=std['Time'],
                    textposition='outside',
                    name=std['Level']
                ))

            fig.update_layout(
                title=f"Time Standards for {event_key}",
                xaxis_title="Time (seconds)",
                yaxis_title="",
                showlegend=False,
                height=300,
                **create_team_saudi_chart_theme()
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full comparison table
            st.markdown("#### Full Standards Table")

            # Build comparison for all events
            all_events_data = []
            for event in sorted(gender_events):
                full_event = f"{qual_gender} {event}"
                row = {
                    'Event': event,
                    'WR': format_time(WORLD_RECORDS_LCM.get(full_event)),
                    'Olympic OQT': format_time(LA_2028_OQT.get(full_event)),
                    'Worlds Entry': format_time(WORLD_CHAMPS_2027_ENTRY.get(full_event)),
                    'Asian Gold': format_time(ASIAN_GAMES_2026_ENTRY.get(full_event)),
                    'Asian Medal': format_time(ASIAN_GAMES_2026_MEDAL.get(full_event)),
                }
                all_events_data.append(row)

            all_events_df = pd.DataFrame(all_events_data)
            st.dataframe(all_events_df, use_container_width=True, hide_index=True)

        else:
            st.warning(f"No standards available for {event_key}")

    with tab4:
        st.subheader("Target Times & Gap Analysis")

        # Show warning if viewing short course data
        if course_type == 'scm':
            st.warning("⚠️ **Note:** Entry standards below are for Long Course (50m pool) competitions. Short course times cannot be directly compared to these standards.")

        st.markdown("""
        Analyze an athlete's current times vs. entry standards for major competitions.
        """)

        # Athlete selection
        all_athletes = sorted(df['FullName'].dropna().unique())
        target_athlete = st.selectbox("Select Athlete", all_athletes, key="target_athlete_nagoya")

        if target_athlete:
            athlete_data = df[df['FullName'].str.contains(target_athlete, case=False, na=False)].copy()

            if not athlete_data.empty:
                # Get unique events for this athlete
                athlete_events = athlete_data['discipline_name'].dropna().unique()

                st.markdown(f"### Gap Analysis for {target_athlete}")
                st.markdown("How close is this athlete to qualifying for each competition?")

                gap_data = []
                for event in athlete_events:
                    event_data = athlete_data[athlete_data['discipline_name'] == event]

                    # Get best time
                    def time_to_sec(t):
                        if pd.isna(t): return float('inf')
                        t = str(t)
                        if ':' in t:
                            parts = t.split(':')
                            return float(parts[0]) * 60 + float(parts[1])
                        try:
                            return float(t)
                        except:
                            return float('inf')

                    event_data = event_data.copy()
                    event_data['time_sec'] = event_data['Time'].apply(time_to_sec)
                    best_time = event_data['time_sec'].min()

                    if best_time < float('inf'):
                        # Find matching standards
                        # Try to match event name to standards
                        matched_event = None
                        for std_event in ASIAN_GAMES_2026_ENTRY.keys():
                            if event.lower() in std_event.lower() or std_event.lower() in event.lower():
                                matched_event = std_event
                                break

                        if matched_event:
                            asian_gold = ASIAN_GAMES_2026_ENTRY.get(matched_event)
                            asian_medal = ASIAN_GAMES_2026_MEDAL.get(matched_event)
                            worlds_entry = WORLD_CHAMPS_2027_ENTRY.get(matched_event)
                            olympic_oqt = LA_2028_OQT.get(matched_event)

                            gap_data.append({
                                'Event': event,
                                'PB': format_time(best_time),
                                'PB_sec': best_time,
                                'Asian Gold': format_time(asian_gold) if asian_gold else 'N/A',
                                'Gap to Gold': f"+{best_time - asian_gold:.2f}s" if asian_gold and best_time > asian_gold else ("✓ QUALIFIED" if asian_gold else 'N/A'),
                                'Asian Medal': format_time(asian_medal) if asian_medal else 'N/A',
                                'Gap to Medal': f"+{best_time - asian_medal:.2f}s" if asian_medal and best_time > asian_medal else ("✓ QUALIFIED" if asian_medal else 'N/A'),
                                'Olympic OQT': format_time(olympic_oqt) if olympic_oqt else 'N/A',
                                'Gap to OQT': f"+{best_time - olympic_oqt:.2f}s" if olympic_oqt and best_time > olympic_oqt else ("✓ QUALIFIED" if olympic_oqt else 'N/A'),
                            })

                if gap_data:
                    gap_df = pd.DataFrame(gap_data)
                    st.dataframe(gap_df, use_container_width=True, hide_index=True)

                    # Visual gap chart for best event
                    st.markdown("---")
                    st.markdown("#### Improvement Roadmap")

                    # Select event for detailed chart
                    chart_events = [g['Event'] for g in gap_data]
                    selected_chart_event = st.selectbox("Select Event for Roadmap", chart_events, key="roadmap_event")

                    selected_gap = next((g for g in gap_data if g['Event'] == selected_chart_event), None)

                    if selected_gap:
                        pb_sec = selected_gap['PB_sec']

                        # Find the matching standards
                        matched_std = None
                        for std_event in ASIAN_GAMES_2026_ENTRY.keys():
                            if selected_chart_event.lower() in std_event.lower() or std_event.lower() in selected_chart_event.lower():
                                matched_std = std_event
                                break

                        if matched_std:
                            fig = go.Figure()

                            targets = []
                            if ASIAN_GAMES_2026_MEDAL.get(matched_std):
                                targets.append(('Asian Medal', ASIAN_GAMES_2026_MEDAL[matched_std], '#CD7F32'))
                            if ASIAN_GAMES_2026_ENTRY.get(matched_std):
                                targets.append(('Asian Gold', ASIAN_GAMES_2026_ENTRY[matched_std], '#FFD700'))
                            if WORLD_CHAMPS_2027_ENTRY.get(matched_std):
                                targets.append(('Worlds Entry', WORLD_CHAMPS_2027_ENTRY[matched_std], '#0077B6'))
                            if LA_2028_OQT.get(matched_std):
                                targets.append(('Olympic OQT', LA_2028_OQT[matched_std], '#007167'))

                            # Current PB
                            fig.add_trace(go.Bar(
                                y=['Current PB'],
                                x=[pb_sec],
                                orientation='h',
                                marker_color='#dc3545',
                                text=format_time(pb_sec),
                                textposition='outside',
                                name='Current PB'
                            ))

                            # Target lines
                            for name, time_sec, color in targets:
                                fig.add_trace(go.Bar(
                                    y=[name],
                                    x=[time_sec],
                                    orientation='h',
                                    marker_color=color,
                                    text=format_time(time_sec),
                                    textposition='outside',
                                    name=name
                                ))

                            fig.update_layout(
                                title=f"Improvement Roadmap: {selected_chart_event}",
                                xaxis_title="Time (seconds)",
                                showlegend=False,
                                height=250,
                                **create_team_saudi_chart_theme()
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Improvement needed summary
                            st.markdown("#### Time to Drop")
                            cols = st.columns(len(targets))
                            for i, (name, time_sec, color) in enumerate(targets):
                                gap = pb_sec - time_sec
                                with cols[i]:
                                    if gap > 0:
                                        st.markdown(f"""
                                        <div style="background: {color}; padding: 0.8rem; border-radius: 8px; text-align: center; color: white;">
                                            <p style="margin: 0; font-size: 0.8rem;">{name}</p>
                                            <p style="margin: 0.3rem 0; font-size: 1.3rem; font-weight: bold;">-{gap:.2f}s</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style="background: #28a745; padding: 0.8rem; border-radius: 8px; text-align: center; color: white;">
                                            <p style="margin: 0; font-size: 0.8rem;">{name}</p>
                                            <p style="margin: 0.3rem 0; font-size: 1.3rem; font-weight: bold;">✓ QUALIFIED</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                else:
                    st.info("No matching events found for gap analysis. This athlete's events may not match standard event names.")


def show_road_to_la(df, course_type='all'):
    """Road to Los Angeles 2028 Olympics - Swimming Qualification Tracker."""

    # Event details
    EVENT_DATE = datetime(2028, 7, 14)
    days_to_go = (EVENT_DATE - datetime.now()).days

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%);
                padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;">
        <h1 style="color: white; margin: 0;">🏅 Road to Los Angeles 2028</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            XXXIV Olympic Games • July 14 - 30, 2028
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: {TEAM_SAUDI_COLORS['gold_accent']}; color: white; padding: 0.5rem 1.5rem;
                        border-radius: 20px; font-size: 1.5rem; font-weight: bold;">
                {days_to_go} Days to Go
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key info columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Host City</div>
            <div class="metric-value" style="font-size: 1.3rem;">Los Angeles, USA</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Venue</div>
            <div class="metric-value" style="font-size: 1.3rem;">LA84 Aquatic Center</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pool Type</div>
            <div class="metric-value" style="font-size: 1.3rem;">50m LCM</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Swimming Events</div>
            <div class="metric-value" style="font-size: 1.3rem;">35 Events</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Olympic Qualifying Times",
        "📊 Saudi Olympic Prospects",
        "🌍 Global Benchmarks",
        "📈 4-Year Pathway",
        "🔬 What It Takes"
    ])

    with tab1:
        st.subheader("Olympic Qualifying Times (OQT)")

        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #007167;">
            <strong>Qualification Pathway:</strong> Athletes can qualify through Olympic Qualifying Times (OQT),
            Olympic Consideration Times (OCT), or Universality places. OQT guarantees automatic qualification.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # OQT times (based on Paris 2024 standards - LA 2028 will be similar or slightly faster)
        oqt_times = {
            'Men': {
                '50m Freestyle': ('21.82', '22.17'),
                '100m Freestyle': ('47.61', '48.32'),
                '200m Freestyle': ('1:45.70', '1:47.00'),
                '400m Freestyle': ('3:44.95', '3:48.05'),
                '800m Freestyle': ('7:50.45', '7:58.00'),
                '1500m Freestyle': ('14:55.00', '15:10.00'),
                '100m Backstroke': ('53.41', '54.25'),
                '200m Backstroke': ('1:56.70', '1:58.50'),
                '100m Breaststroke': ('59.49', '1:00.50'),
                '200m Breaststroke': ('2:09.39', '2:11.50'),
                '100m Butterfly': ('51.00', '51.80'),
                '200m Butterfly': ('1:54.79', '1:56.50'),
                '200m Individual Medley': ('1:58.01', '1:59.80'),
                '400m Individual Medley': ('4:12.50', '4:16.50'),
            },
            'Women': {
                '50m Freestyle': ('24.69', '25.10'),
                '100m Freestyle': ('53.52', '54.35'),
                '200m Freestyle': ('1:56.32', '1:58.00'),
                '400m Freestyle': ('4:04.50', '4:08.50'),
                '800m Freestyle': ('8:24.00', '8:32.00'),
                '1500m Freestyle': ('16:05.00', '16:20.00'),
                '100m Backstroke': ('59.72', '1:00.70'),
                '200m Backstroke': ('2:09.00', '2:11.00'),
                '100m Breaststroke': ('1:06.29', '1:07.50'),
                '200m Breaststroke': ('2:23.00', '2:25.50'),
                '100m Butterfly': ('57.10', '58.00'),
                '200m Butterfly': ('2:07.50', '2:10.00'),
                '200m Individual Medley': ('2:10.50', '2:12.50'),
                '400m Individual Medley': ('4:38.53', '4:43.00'),
            }
        }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Men's Olympic Standards**")
            men_data = []
            for event, (oqt, oct) in oqt_times['Men'].items():
                men_data.append({'Event': event, 'OQT (Auto)': oqt, 'OCT (Consider)': oct})
            st.dataframe(pd.DataFrame(men_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Women's Olympic Standards**")
            women_data = []
            for event, (oqt, oct) in oqt_times['Women'].items():
                women_data.append({'Event': event, 'OQT (Auto)': oqt, 'OCT (Consider)': oct})
            st.dataframe(pd.DataFrame(women_data), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Saudi Olympic Swimming Prospects")

        # Filter Saudi athletes
        saudi_df = df[df['NAT'] == 'KSA'].copy() if 'NAT' in df.columns else pd.DataFrame()

        if saudi_df.empty:
            st.info("No Saudi (KSA) athletes found. Select athletes to track for Olympic qualification.")

            all_athletes = sorted(df['FullName'].dropna().unique())
            selected_athletes = st.multiselect(
                "Select athletes to analyze for LA 2028:",
                all_athletes,
                help="Select Team Saudi athletes"
            )

            if selected_athletes:
                saudi_df = df[df['FullName'].isin(selected_athletes)].copy()

        if not saudi_df.empty:
            tracker = TalentDevelopmentTracker(saudi_df)

            # Analyze each athlete's Olympic potential
            st.markdown("**World Record Percentage Analysis**")
            st.markdown("Athletes at 90%+ of World Record are competitive at Olympic level")

            for athlete in saudi_df['FullName'].unique():
                wr_analysis = tracker.calculate_world_record_percentage(athlete)

                if wr_analysis:
                    with st.expander(f"🏊 {athlete}"):
                        for wr in wr_analysis[:5]:
                            pct = wr['wr_percentage']
                            color = TEAM_SAUDI_COLORS['primary_teal'] if pct >= 90 else (
                                TEAM_SAUDI_COLORS['gold_accent'] if pct >= 85 else '#6c757d'
                            )
                            status = "🌟 Olympic Level" if pct >= 90 else (
                                "📈 Approaching" if pct >= 85 else "🎯 Developing"
                            )
                            time_display = wr.get('time', 'N/A')
                            comp_info = wr.get('competition', '')[:30] if wr.get('competition') else ''

                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 0.5rem;
                                        background: #f8f9fa; border-radius: 4px; margin-bottom: 0.5rem;
                                        border-left: 4px solid {color};">
                                <span><strong>{wr['event']}</strong> <small style="color: #666;">{comp_info}</small></span>
                                <span>{time_display} ({pct:.1f}% WR) {status}</span>
                            </div>
                            """, unsafe_allow_html=True)

    with tab3:
        st.subheader("Global Olympic Benchmarks")

        # Recent Olympic and World Championship data
        if 'competition_name' in df.columns:
            major_comps = df[
                df['competition_name'].str.contains('Olympic|World Championships|World Aquatics Championships',
                                                    case=False, na=False, regex=True)
            ].copy()

            if not major_comps.empty:
                st.markdown("**Recent Major Championship Standards**")

                # Medal times by event
                if 'MedalTag' in major_comps.columns:
                    medal_df = major_comps[major_comps['MedalTag'].isin(['G', 'S', 'B'])]

                    events = medal_df['discipline_name'].dropna().unique()
                    selected_event = st.selectbox("Select Event", sorted(events), key="olympic_event")

                    if selected_event:
                        event_medals = medal_df[medal_df['discipline_name'] == selected_event]
                        event_medals = event_medals[['FullName', 'NAT', 'Time', 'MedalTag', 'competition_name', 'year']]
                        event_medals.columns = ['Athlete', 'Nation', 'Time', 'Medal', 'Competition', 'Year']

                        # Sort by medal type
                        medal_order = {'G': 0, 'S': 1, 'B': 2}
                        event_medals['medal_rank'] = event_medals['Medal'].map(medal_order)
                        event_medals = event_medals.sort_values(['Year', 'medal_rank'], ascending=[False, True])

                        st.dataframe(event_medals.drop('medal_rank', axis=1).head(20),
                                   use_container_width=True, hide_index=True)
            else:
                st.info("No Olympic or World Championship data available. Data from major competitions helps establish benchmarks.")

    with tab4:
        st.subheader("4-Year Development Pathway to LA 2028")

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {TEAM_SAUDI_COLORS['primary_teal']}, {TEAM_SAUDI_COLORS['gold_accent']});
                    padding: 2px; border-radius: 10px; margin-bottom: 1rem;">
            <div style="background: white; padding: 1rem; border-radius: 8px;">
        """, unsafe_allow_html=True)

        # Timeline
        milestones = [
            ("2025", "Foundation Year", "Build aerobic base, technique refinement, compete at Asian Championships", "#007167"),
            ("2026", "Nagoya Asian Games", "Major continental test, qualification experience, benchmark against Asian elite", "#a08e66"),
            ("2027", "World Championships", "Global exposure, OQT qualification window opens, fine-tune race strategy", "#005a51"),
            ("2028", "Los Angeles Olympics", "Peak performance, taper and race execution, Olympic debut or medal push", "#FFB800"),
        ]

        for year, title, description, color in milestones:
            st.markdown(f"""
            <div style="display: flex; margin-bottom: 1rem;">
                <div style="background: {color}; color: white; padding: 0.5rem 1rem; border-radius: 8px;
                            min-width: 80px; text-align: center; font-weight: bold;">{year}</div>
                <div style="margin-left: 1rem; padding: 0.5rem;">
                    <strong>{title}</strong><br>
                    <span style="color: #666;">{description}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Annual improvement targets
        st.markdown("---")
        st.subheader("Required Annual Improvement")

        col1, col2 = st.columns(2)

        with col1:
            current_wr_pct = st.number_input("Current % of World Record", 80.0, 100.0, 85.0, 0.5)
            target_wr_pct = st.number_input("Target % of World Record (Olympic Level)", 88.0, 100.0, 92.0, 0.5)

        with col2:
            years_remaining = 4  # To 2028
            annual_improvement = (target_wr_pct - current_wr_pct) / years_remaining

            st.markdown(f"""
            <div style="background: {TEAM_SAUDI_COLORS['primary_teal']}; padding: 1.5rem;
                        border-radius: 10px; color: white; text-align: center;">
                <p style="margin: 0; font-size: 0.9rem;">REQUIRED ANNUAL IMPROVEMENT</p>
                <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;
                          color: {TEAM_SAUDI_COLORS['gold_accent']};">
                    +{annual_improvement:.2f}% per year
                </p>
                <p style="margin: 0; font-size: 0.85rem;">
                    From {current_wr_pct}% → {target_wr_pct}% of World Record
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab5:
        st.subheader("What It Takes to Make an Olympic Final")

        st.markdown("""
        Based on research from World Aquatics and Olympic performance data:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {TEAM_SAUDI_COLORS['primary_teal']};">
                <h4 style="color: {TEAM_SAUDI_COLORS['primary_teal']}; margin-top: 0;">Physical Benchmarks</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>World Record %:</strong> 95%+ for finals, 97%+ for medals</li>
                    <li><strong>FINA Points:</strong> 900+ consistently</li>
                    <li><strong>Pacing CV:</strong> < 1.3% lap variance</li>
                    <li><strong>Reaction Time:</strong> < 0.65s average</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {TEAM_SAUDI_COLORS['gold_accent']};">
                <h4 style="color: {TEAM_SAUDI_COLORS['gold_accent']}; margin-top: 0;">Development Timeline</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Years of Training:</strong> 8-10 years at elite level</li>
                    <li><strong>Peak Age (Men):</strong> 24-26 years</li>
                    <li><strong>Peak Age (Women):</strong> 22-24 years</li>
                    <li><strong>Competition Experience:</strong> 50+ international races</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.info("""
        **Key Success Factors from Olympic Medalists:**
        1. Start competing internationally by age 16-18
        2. Achieve 900+ FINA points by age 20-22
        3. Maintain 1-2% annual improvement through peak years
        4. Develop race IQ through major championship experience
        5. Master heats-to-finals improvement (negative split strategy)
        """)


def show_athlete_profiles(df, course_type='all'):
    """Comprehensive athlete profile analysis page."""
    st.header("👤 Athlete Profiles - Comprehensive Analysis")

    # Use appropriate world records based on course type
    world_records = get_world_records(course_type if course_type != 'all' else 'lcm')
    course_label = "Short Course (25m)" if course_type == 'scm' else "Long Course (50m)"

    st.markdown(f"""
    Complete athlete analysis with career overview, event breakdowns, progression tracking, and competition history.
    *Analysis based on {course_label} data*
    """)

    # Country and Athlete Selection
    st.markdown("### Select Athlete")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        countries = sorted(df['NAT'].dropna().unique())
        selected_country = st.selectbox(
            "Filter by Country",
            ["All Countries"] + list(countries),
            key="profile_country"
        )

    with col2:
        # Filter athletes by country
        if selected_country == "All Countries":
            filtered_df = df
        else:
            filtered_df = df[df['NAT'] == selected_country]

        athletes = sorted(filtered_df['FullName'].dropna().unique())
        selected_athlete = st.selectbox(
            "Select Athlete",
            athletes if athletes else ["No athletes found"],
            key="profile_athlete"
        )

    with col3:
        # Show athlete count
        st.metric("Athletes Available", f"{len(athletes):,}")

    if selected_athlete and selected_athlete != "No athletes found":
        # Get all athlete data
        athlete_data = df[df['FullName'].str.contains(selected_athlete, case=False, na=False)].copy()

        if athlete_data.empty:
            st.warning("No data found for this athlete")
            return

        # Time conversion
        from enhanced_swimming_scraper import SplitTimeAnalyzer
        time_analyzer = SplitTimeAnalyzer()
        athlete_data['time_seconds'] = athlete_data['Time'].apply(time_analyzer.time_to_seconds)
        athlete_data = athlete_data[athlete_data['time_seconds'] > 0]

        st.markdown("---")

        # Profile Header with key stats
        st.markdown(f"## {selected_athlete}")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            country = athlete_data['NAT'].mode().iloc[0] if not athlete_data['NAT'].mode().empty else "N/A"
            st.metric("Country", country)

        with col2:
            total_races = len(athlete_data)
            st.metric("Total Races", f"{total_races:,}")

        with col3:
            if 'year' in athlete_data.columns and athlete_data['year'].notna().any():
                valid_years = athlete_data['year'].dropna()
                years_active = valid_years.nunique()
                year_min = int(valid_years.min())
                year_max = int(valid_years.max())
                year_range = f"{year_min}-{year_max}"
                st.metric("Years Active", f"{years_active} ({year_range})")
            else:
                st.metric("Years Active", "N/A")

        with col4:
            events_count = athlete_data['discipline_name'].nunique()
            st.metric("Events", events_count)

        with col5:
            if 'MedalTag' in athlete_data.columns:
                medals = athlete_data['MedalTag'].value_counts()
                gold = medals.get('G', 0)
                silver = medals.get('S', 0)
                bronze = medals.get('B', 0)
                total_medals = gold + silver + bronze
                st.metric("Medals", f"🥇{gold} 🥈{silver} 🥉{bronze}")
            else:
                st.metric("Medals", "N/A")

        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Career Overview",
            "Event Analysis",
            "Progression",
            "Competition History",
            "World Record Comparison"
        ])

        with tab1:
            st.subheader("Career Overview")

            col_left, col_right = st.columns(2)

            with col_left:
                # Personal Bests by Event - simplified approach
                st.markdown("#### Personal Bests")
                pb_list = []
                for event in athlete_data['discipline_name'].dropna().unique():
                    event_rows = athlete_data[athlete_data['discipline_name'] == event]
                    if not event_rows.empty and event_rows['time_seconds'].notna().any():
                        best_idx = event_rows['time_seconds'].idxmin()
                        pb_list.append({
                            'Event': event,
                            'Time': event_rows.loc[best_idx, 'Time'],
                            'Competition': event_rows.loc[best_idx, 'competition_name'] if 'competition_name' in event_rows.columns else 'N/A'
                        })

                if pb_list:
                    pbs = pd.DataFrame(pb_list)
                    st.dataframe(
                        pbs,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No personal bests data available")

            with col_right:
                # Race distribution by event
                st.markdown("#### Races by Event")
                event_counts = athlete_data['discipline_name'].value_counts().head(10)
                fig = px.pie(
                    values=event_counts.values,
                    names=event_counts.index,
                    title="Race Distribution",
                    color_discrete_sequence=[
                        TEAM_SAUDI_COLORS['primary_teal'],
                        TEAM_SAUDI_COLORS['gold_accent'],
                        TEAM_SAUDI_COLORS['dark_teal'],
                        '#FFB800', '#0077B6', '#6c757d'
                    ]
                )
                fig.update_layout(**create_team_saudi_chart_theme())
                st.plotly_chart(fig, use_container_width=True)

            # Performance Timeline
            st.markdown("#### Performance Timeline")
            if 'year' in athlete_data.columns:
                yearly_stats = athlete_data.groupby('year').agg({
                    'time_seconds': ['count', 'min'],
                    'discipline_name': 'nunique'
                }).reset_index()
                yearly_stats.columns = ['Year', 'Races', 'Best Time', 'Events']

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=yearly_stats['Year'], y=yearly_stats['Races'],
                          name='Races', marker_color=TEAM_SAUDI_COLORS['primary_teal']),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Events'],
                              name='Events', mode='lines+markers',
                              line=dict(color=TEAM_SAUDI_COLORS['gold_accent'], width=3)),
                    secondary_y=True
                )
                fig.update_layout(title="Activity by Year", **create_team_saudi_chart_theme())
                fig.update_yaxes(title_text="Races", secondary_y=False)
                fig.update_yaxes(title_text="Events", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Event Analysis")

            # Event selector
            events = sorted(athlete_data['discipline_name'].dropna().unique())
            selected_event = st.selectbox("Select Event for Detailed Analysis", events, key="profile_event")

            if selected_event:
                event_data = athlete_data[athlete_data['discipline_name'] == selected_event].copy()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Races", len(event_data))
                    pb = event_data['time_seconds'].min()
                    pb_formatted = event_data.loc[event_data['time_seconds'].idxmin(), 'Time']
                    st.metric("Personal Best", pb_formatted)

                with col2:
                    avg_time = event_data['time_seconds'].mean()
                    st.metric("Average Time", f"{avg_time:.2f}s")
                    std_time = event_data['time_seconds'].std()
                    st.metric("Consistency (σ)", f"{std_time:.2f}s")

                with col3:
                    # Find WR for comparison
                    wr_time = None
                    for wr_event, wr in world_records.items():
                        if selected_event.lower() in wr_event.lower() or wr_event.lower() in selected_event.lower():
                            wr_time = wr
                            break
                    if wr_time:
                        wr_pct = (wr_time / pb) * 100
                        st.metric("WR %", f"{wr_pct:.1f}%")
                        st.metric("Gap to WR", f"+{pb - wr_time:.2f}s")

                with col4:
                    if 'Rank' in event_data.columns:
                        avg_rank = event_data['Rank'].mean()
                        best_rank = event_data['Rank'].min()
                        st.metric("Avg Rank", f"{avg_rank:.1f}")
                        st.metric("Best Rank", int(best_rank))

                # Performance chart for this event
                st.markdown("#### Performance History")
                if 'date_from' in event_data.columns or 'year' in event_data.columns:
                    event_data_sorted = event_data.sort_values('date_from' if 'date_from' in event_data.columns else 'year')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(event_data_sorted))),
                        y=event_data_sorted['time_seconds'],
                        mode='lines+markers',
                        name='Race Time',
                        line=dict(color=TEAM_SAUDI_COLORS['primary_teal']),
                        marker=dict(size=8)
                    ))
                    fig.add_hline(y=pb, line_dash="dash", line_color=TEAM_SAUDI_COLORS['gold_accent'],
                                 annotation_text=f"PB: {pb_formatted}")
                    fig.update_layout(
                        title=f"{selected_event} - Performance Progression",
                        xaxis_title="Race Number",
                        yaxis_title="Time (seconds)",
                        **create_team_saudi_chart_theme()
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recent results table
                st.markdown("#### Recent Results")
                recent = event_data.sort_values('date_from' if 'date_from' in event_data.columns else 'year', ascending=False).head(10)
                display_cols = ['Time', 'competition_name']
                if 'Rank' in recent.columns:
                    display_cols.append('Rank')
                if 'year' in recent.columns:
                    display_cols.append('year')
                st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Progression Analysis")

            if 'year' in athlete_data.columns and athlete_data['year'].nunique() > 1:
                # Select event for progression
                prog_events = sorted(athlete_data['discipline_name'].dropna().unique())
                prog_event = st.selectbox("Select Event", prog_events, key="prog_event")

                if prog_event:
                    event_data = athlete_data[athlete_data['discipline_name'] == prog_event]
                    yearly_pb = event_data.groupby('year')['time_seconds'].min().reset_index()
                    yearly_pb.columns = ['Year', 'Best Time']
                    yearly_pb = yearly_pb.sort_values('Year')

                    if len(yearly_pb) > 1:
                        # Calculate improvement
                        yearly_pb['Improvement'] = yearly_pb['Best Time'].diff() * -1
                        yearly_pb['Improvement %'] = (yearly_pb['Improvement'] / yearly_pb['Best Time'].shift(1)) * 100

                        col1, col2 = st.columns(2)

                        with col1:
                            # Progression chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=yearly_pb['Year'],
                                y=yearly_pb['Best Time'],
                                mode='lines+markers',
                                name='Best Time',
                                line=dict(color=TEAM_SAUDI_COLORS['primary_teal'], width=3),
                                marker=dict(size=10)
                            ))
                            fig.update_layout(
                                title=f"{prog_event} - Yearly Progression",
                                xaxis_title="Year",
                                yaxis_title="Best Time (seconds)",
                                **create_team_saudi_chart_theme()
                            )
                            # Invert y-axis so faster times are higher
                            fig.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Improvement chart
                            improvement_data = yearly_pb.dropna(subset=['Improvement'])
                            fig = go.Figure()
                            colors = [TEAM_SAUDI_COLORS['primary_teal'] if x > 0 else '#dc3545' for x in improvement_data['Improvement']]
                            fig.add_trace(go.Bar(
                                x=improvement_data['Year'],
                                y=improvement_data['Improvement'],
                                marker_color=colors,
                                name='Improvement (s)'
                            ))
                            fig.update_layout(
                                title="Year-over-Year Improvement",
                                xaxis_title="Year",
                                yaxis_title="Improvement (seconds)",
                                **create_team_saudi_chart_theme()
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Progression table
                        st.markdown("#### Progression Data")
                        st.dataframe(yearly_pb, use_container_width=True, hide_index=True)

                        # Summary stats
                        total_improvement = yearly_pb['Best Time'].iloc[0] - yearly_pb['Best Time'].iloc[-1]
                        avg_yearly_improvement = total_improvement / (len(yearly_pb) - 1)
                        st.info(f"**Total Improvement:** {total_improvement:.2f}s over {len(yearly_pb)-1} years | "
                               f"**Avg Annual:** {avg_yearly_improvement:.2f}s/year")
                    else:
                        st.info("Need at least 2 years of data for progression analysis")
            else:
                st.info("Insufficient year data for progression analysis")

        with tab4:
            st.subheader("Competition History")

            if 'competition_name' in athlete_data.columns:
                # Competition breakdown
                comp_stats = athlete_data.groupby('competition_name').agg({
                    'time_seconds': ['count', 'min'],
                    'discipline_name': 'nunique'
                }).reset_index()
                comp_stats.columns = ['Competition', 'Races', 'Best Time', 'Events']
                comp_stats = comp_stats.sort_values('Races', ascending=False)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("#### Competitions Attended")
                    st.dataframe(comp_stats.head(20), use_container_width=True, hide_index=True)

                with col2:
                    st.metric("Total Competitions", len(comp_stats))
                    st.metric("Total Races", comp_stats['Races'].sum())

                    # Medal breakdown by competition if available
                    if 'MedalTag' in athlete_data.columns:
                        medal_data = athlete_data[athlete_data['MedalTag'].notna()]
                        if not medal_data.empty:
                            st.markdown("#### Medal Competitions")
                            medal_comps = medal_data.groupby('competition_name')['MedalTag'].value_counts().unstack(fill_value=0)
                            st.dataframe(medal_comps.head(10), use_container_width=True)

        with tab5:
            st.subheader("World Record Comparison")

            st.markdown(f"*Comparing against {course_label} World Records*")

            # Calculate WR% for all events
            wr_comparisons = []
            for event in athlete_data['discipline_name'].unique():
                event_pb = athlete_data[athlete_data['discipline_name'] == event]['time_seconds'].min()
                event_pb_str = athlete_data.loc[
                    (athlete_data['discipline_name'] == event) &
                    (athlete_data['time_seconds'] == event_pb), 'Time'
                ].iloc[0]

                # Find matching WR
                for wr_event, wr_time in world_records.items():
                    if event.lower() in wr_event.lower() or wr_event.lower() in event.lower():
                        wr_pct = (wr_time / event_pb) * 100
                        gap = event_pb - wr_time
                        wr_comparisons.append({
                            'Event': event,
                            'PB': event_pb_str,
                            'PB (s)': round(event_pb, 2),
                            'WR': f"{wr_time:.2f}s",
                            'WR %': round(wr_pct, 1),
                            'Gap to WR': f"+{gap:.2f}s",
                            'Elite Level': '⭐' if wr_pct >= 95 else ''
                        })
                        break

            if wr_comparisons:
                wr_df = pd.DataFrame(wr_comparisons).sort_values('WR %', ascending=False)

                # Chart
                fig = px.bar(
                    wr_df,
                    x='Event',
                    y='WR %',
                    color='WR %',
                    color_continuous_scale=[[0, '#dc3545'], [0.5, TEAM_SAUDI_COLORS['primary_teal']], [1, TEAM_SAUDI_COLORS['gold_accent']]],
                    title="World Record Percentage by Event"
                )
                fig.add_hline(y=95, line_dash="dash", line_color=TEAM_SAUDI_COLORS['gold_accent'],
                             annotation_text="Elite Level (95%)")
                fig.update_layout(**create_team_saudi_chart_theme())
                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.dataframe(wr_df, use_container_width=True, hide_index=True)

                # Summary
                elite_events = len([x for x in wr_comparisons if x['WR %'] >= 95])
                avg_wr_pct = sum([x['WR %'] for x in wr_comparisons]) / len(wr_comparisons)
                st.success(f"**Elite Level Events (≥95%):** {elite_events} | **Average WR%:** {avg_wr_pct:.1f}%")
            else:
                st.info("No matching world records found for this athlete's events")


def show_talent_development(df, course_type='all'):
    """Talent development tracking page."""
    st.header("📈 Talent Development Tracking")

    st.info("""
    **Research-Based Benchmarks:**
    - Elite level: ~8 years of competition to reach >900 FINA points
    - Peak ages: Males 24.2 ± 2.1 years, Females 22.5 ± 2.4 years
    - Critical transition: Males 16-19, Females 15-18
    """)

    tracker = TalentDevelopmentTracker(df)

    # Country and Athlete selectors
    col1, col2 = st.columns([1, 3])

    with col1:
        countries = sorted(df['NAT'].dropna().unique())
        selected_country = st.selectbox("Filter by Country", ["All Countries"] + countries)

    with col2:
        # Filter athletes by country
        if selected_country == "All Countries":
            filtered_df = df
        else:
            filtered_df = df[df['NAT'] == selected_country]

        athletes = sorted(filtered_df['FullName'].dropna().unique())
        selected_athlete = st.selectbox("Select Athlete", athletes if athletes else ["No athletes found"])

    if selected_athlete:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Competition Age", "World Record %", "Age Progression", "Annual Improvement"
        ])

        with tab1:
            st.subheader("Competition History Analysis")
            comp_age = tracker.calculate_competition_age(selected_athlete)

            if 'error' not in comp_age:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Years Competing", comp_age['competition_years'])
                    st.metric("First Competition", comp_age['first_competition'])

                with col2:
                    progress = comp_age['progress_to_elite_pct']
                    st.metric("Progress to Elite", f"{progress}%")

                    # Progress bar
                    st.progress(min(progress / 100, 1.0))

                with col3:
                    st.metric("Total Races", comp_age['total_races'])
                    st.metric("Competitions", comp_age['total_competitions'])

                # Status
                if comp_age['on_track']:
                    st.success("✅ On track for elite development timeline")
                else:
                    st.warning("⚠️ Early in development - continue monitoring")
            else:
                st.error(comp_age['error'])

        with tab2:
            st.subheader("World Record Benchmarking")
            wr_analysis = tracker.calculate_world_record_percentage(selected_athlete)

            if wr_analysis:
                # Create table
                wr_df = pd.DataFrame(wr_analysis)
                wr_df['Elite'] = wr_df['is_elite_level'].apply(lambda x: '⭐ ELITE' if x else '')

                st.dataframe(
                    wr_df[['event', 'time', 'wr_percentage', 'gap_to_wr_seconds', 'Elite', 'competition']],
                    use_container_width=True,
                    column_config={
                        'wr_percentage': st.column_config.NumberColumn('WR %', format="%.2f%%"),
                        'gap_to_wr_seconds': st.column_config.NumberColumn('Gap to WR (s)', format="%.2f")
                    }
                )

                # Chart
                fig = px.bar(
                    wr_df.head(10), x='event', y='wr_percentage',
                    color='is_elite_level',
                    color_discrete_map={True: TEAM_SAUDI_COLORS['gold_accent'],
                                       False: TEAM_SAUDI_COLORS['primary_teal']},
                    title="Performance vs World Record"
                )
                fig.add_hline(y=95, line_dash="dash", line_color=TEAM_SAUDI_COLORS['gold_accent'],
                             annotation_text="Elite Level (95%)")
                fig.update_layout(**create_team_saudi_chart_theme())
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No world record comparisons available for this athlete's events")

        with tab3:
            st.subheader("Age Progression Analysis")

            # Try to get age from data
            athlete_data = df[df['FullName'].str.contains(selected_athlete, case=False, na=False)]
            if 'AthleteResultAge' in athlete_data.columns:
                current_age = int(athlete_data['AthleteResultAge'].dropna().iloc[-1]) if not athlete_data['AthleteResultAge'].dropna().empty else None
            else:
                current_age = st.number_input("Enter athlete's current age", min_value=10, max_value=50, value=20)

            if current_age:
                age_analysis = tracker.analyze_age_progression(selected_athlete, current_age)

                if 'error' not in age_analysis:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Current Age", age_analysis['current_age'])
                        st.metric("Expected Peak Age", age_analysis['expected_peak_age'])
                        st.metric("Years to Peak", age_analysis['years_to_peak'])

                    with col2:
                        st.metric("Development Phase", age_analysis['development_phase'])
                        st.metric("Peak Window", age_analysis['peak_window'])

                        if age_analysis['in_peak_window']:
                            st.success("🎯 Currently in peak performance window!")
                        elif age_analysis['years_to_peak'] > 0:
                            st.info(f"📈 {age_analysis['years_to_peak']} years until expected peak")

                    # Visual timeline
                    st.subheader("Development Timeline")

                    fig = go.Figure()

                    # Add phases
                    phases = [
                        ('Junior Dev', 12, 16, '#e3f2fd'),
                        ('Transition', 16, 19 if age_analysis['gender'] == 'male' else 18, '#fff3e0'),
                        ('Senior Dev', 19 if age_analysis['gender'] == 'male' else 18,
                         age_analysis['expected_peak_age'] - 2, '#e8f5e9'),
                        ('Peak Window', age_analysis['expected_peak_age'] - 2,
                         age_analysis['expected_peak_age'] + 2, TEAM_SAUDI_COLORS['gold_accent']),
                    ]

                    for name, start, end, color in phases:
                        fig.add_shape(
                            type="rect", x0=start, x1=end, y0=0, y1=1,
                            fillcolor=color, opacity=0.5, line_width=0
                        )
                        fig.add_annotation(x=(start+end)/2, y=0.5, text=name, showarrow=False)

                    # Add current age marker
                    fig.add_vline(x=current_age, line_color=TEAM_SAUDI_COLORS['primary_teal'],
                                 line_width=3, annotation_text="Current")

                    fig.update_layout(
                        xaxis_title="Age",
                        yaxis_visible=False,
                        height=200,
                        **create_team_saudi_chart_theme()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(age_analysis['error'])

        with tab4:
            st.subheader("Annual Improvement Rate")

            # Event selector
            events = df[df['FullName'].str.contains(selected_athlete, case=False, na=False)]['discipline_name'].dropna().unique()
            selected_event = st.selectbox("Select Event", events)

            if selected_event:
                improvement = tracker.calculate_annual_improvement_rate(selected_athlete, selected_event)

                if 'error' not in improvement:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Avg Annual Improvement", f"{improvement['average_annual_improvement_pct']}%")
                        st.metric("Total Improvement", f"{improvement['total_improvement_pct']}%")

                    with col2:
                        st.metric("Years Analyzed", improvement['years_analyzed'])
                        trajectory_color = "green" if improvement['trajectory'] == 'improving' else "red"
                        st.markdown(f"Trajectory: <span style='color:{trajectory_color}'>{improvement['trajectory'].upper()}</span>",
                                   unsafe_allow_html=True)

                    # Chart
                    yearly_df = pd.DataFrame(improvement['yearly_improvements'])
                    fig = px.bar(
                        yearly_df, x='year', y='improvement_pct',
                        color='improvement_pct',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title=f"Year-over-Year Improvement: {selected_event}"
                    )
                    fig.add_hline(y=0, line_dash="solid", line_color="gray")
                    fig.update_layout(**create_team_saudi_chart_theme())
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(improvement['error'])


def show_pacing_analysis(df, course_type='all'):
    """Advanced pacing analysis page."""
    st.header("🎯 Pacing Strategy Analysis")

    st.info("""
    **Research Findings:**
    - 400m medalists: Inverted-J strategy (98.9% of WR)
    - 800m medalists: U-shape strategy universally adopted
    - Elite CV (lap variance): < 1.3%
    - Critical: Must be top 3 in final 100m to medal
    """)

    pacing_analyzer = AdvancedPacingAnalyzer()

    # Filter for results with splits
    with_splits = df[df['lap_times_json'].notna()].copy() if 'lap_times_json' in df.columns else pd.DataFrame()

    if with_splits.empty:
        st.warning("No split time data available for advanced pacing analysis")
        return

    tab1, tab2 = st.tabs(["Event Analysis", "Athlete Analysis"])

    with tab1:
        st.subheader("Event Pacing Patterns")

        events = sorted(with_splits['discipline_name'].dropna().unique())
        selected_event = st.selectbox("Select Event", events)

        if selected_event:
            event_data = with_splits[with_splits['discipline_name'] == selected_event].copy()

            # Convert times to seconds for filtering
            from enhanced_swimming_scraper import SplitTimeAnalyzer
            time_analyzer = SplitTimeAnalyzer()
            event_data['time_seconds'] = event_data['Time'].apply(time_analyzer.time_to_seconds)

            # Target time filter
            st.markdown("**Filter by Target Time Range:**")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                min_time = event_data['time_seconds'].min()
                max_time = event_data['time_seconds'].max()

                # Format as MM:SS for display
                def format_time(secs):
                    if pd.isna(secs) or secs is None:
                        return "N/A"
                    mins = int(secs // 60)
                    secs_rem = secs % 60
                    return f"{mins}:{secs_rem:05.2f}" if mins > 0 else f"{secs_rem:.2f}"

                target_min = st.number_input("Min Time (seconds)", value=float(min_time) if pd.notna(min_time) else 0.0, step=1.0)

            with col2:
                target_max = st.number_input("Max Time (seconds)", value=float(max_time) if pd.notna(max_time) else 300.0, step=1.0)

            with col3:
                st.info(f"Range: {format_time(target_min)} to {format_time(target_max)}")

            # Filter by target time
            filtered_data = event_data[
                (event_data['time_seconds'] >= target_min) &
                (event_data['time_seconds'] <= target_max)
            ]

            st.write(f"**{len(filtered_data)} races in target time range**")

            # Analyze pacing for filtered results
            pacing_results = []
            for _, row in filtered_data.nsmallest(50, 'time_seconds').iterrows():
                try:
                    lap_times = json.loads(row['lap_times_json'])
                    analysis = pacing_analyzer.classify_pacing_strategy(lap_times)
                    analysis['athlete'] = row.get('FullName', 'Unknown')
                    analysis['rank'] = row.get('Rank', 'N/A')
                    analysis['time'] = row.get('Time', 'N/A')
                    pacing_results.append(analysis)
                except:
                    continue

            if pacing_results:
                # Strategy distribution
                strategies = [p['strategy'] for p in pacing_results]
                strategy_counts = pd.Series(strategies).value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        values=strategy_counts.values,
                        names=strategy_counts.index,
                        title=f"Pacing Strategies: {selected_event}",
                        color_discrete_sequence=[
                            TEAM_SAUDI_COLORS['primary_teal'],
                            TEAM_SAUDI_COLORS['gold_accent'],
                            TEAM_SAUDI_COLORS['dark_teal'],
                            '#FFB800', '#0077B6'
                        ]
                    )
                    fig.update_layout(**create_team_saudi_chart_theme())
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Most successful strategy
                    top_3_strategies = [p['strategy'] for p in pacing_results[:3]]
                    st.markdown("**Medalist Strategies (Top 3):**")
                    for i, strat in enumerate(top_3_strategies):
                        medal = ['🥇', '🥈', '🥉'][i]
                        st.write(f"{medal} {strat}")

                    avg_cv = np.mean([p['coefficient_of_variation'] for p in pacing_results])
                    st.metric("Avg Lap Variance (CV)", f"{avg_cv:.2f}%")

                    if avg_cv < ELITE_BENCHMARKS['cv_elite_threshold']:
                        st.success("Elite-level consistency")

                # Detailed table
                st.subheader("Detailed Pacing Analysis")
                pacing_df = pd.DataFrame(pacing_results)
                st.dataframe(
                    pacing_df[['rank', 'athlete', 'time', 'strategy', 'coefficient_of_variation',
                              'is_elite_consistency', 'fastest_lap', 'slowest_lap']],
                    use_container_width=True
                )

    with tab2:
        st.subheader("Individual Athlete Pacing - Race Comparison")

        # Country filter for athletes
        col1, col2 = st.columns([1, 3])

        with col1:
            countries = sorted(with_splits['NAT'].dropna().unique())
            pacing_country = st.selectbox("Filter by Country", ["All Countries"] + countries, key="pacing_country")

        with col2:
            if pacing_country == "All Countries":
                athlete_pool = with_splits
            else:
                athlete_pool = with_splits[with_splits['NAT'] == pacing_country]

            athletes = sorted(athlete_pool['FullName'].dropna().unique())
            selected_athlete = st.selectbox("Select Athlete", athletes if athletes else ["No athletes found"], key="pacing_athlete")

        if selected_athlete and selected_athlete != "No athletes found":
            athlete_splits = with_splits[
                with_splits['FullName'].str.contains(selected_athlete, case=False, na=False)
            ].copy()

            # Remove duplicate races (same competition, event, time, year)
            dedup_cols = ['competition_name', 'discipline_name', 'Time', 'year']
            available_dedup_cols = [c for c in dedup_cols if c in athlete_splits.columns]
            if available_dedup_cols:
                athlete_splits = athlete_splits.drop_duplicates(subset=available_dedup_cols, keep='first')

            # Event filter for this athlete
            athlete_events = sorted(athlete_splits['discipline_name'].dropna().unique())
            selected_pacing_event = st.selectbox("Filter by Event", ["All Events"] + athlete_events, key="pacing_event_filter")

            if selected_pacing_event != "All Events":
                athlete_splits = athlete_splits[athlete_splits['discipline_name'] == selected_pacing_event]

            # Number of races to show
            num_races = st.slider("Number of races to display", 1, min(20, len(athlete_splits)), 5)

            st.markdown(f"**Showing {num_races} of {len(athlete_splits)} races with split data**")
            st.markdown("---")

            for idx, (_, row) in enumerate(athlete_splits.head(num_races).iterrows()):
                race_title = f"Race {idx+1}: {row.get('discipline_name', 'Event')} - {row.get('Time', 'N/A')}"
                comp_info = f"{row.get('competition_name', '')[:50]} ({row.get('year', 'N/A')})"

                with st.expander(f"{race_title} | {comp_info}", expanded=(idx == 0)):
                    try:
                        # Validate lap_times_json before parsing
                        lap_json = row.get('lap_times_json', '')
                        if pd.isna(lap_json) or not lap_json or lap_json in ('', '[]', 'null', 'None'):
                            st.info("No split data available for this race")
                            continue

                        lap_times = json.loads(lap_json)
                        if not lap_times or len(lap_times) < 2:
                            st.info("Insufficient split data for pacing analysis")
                            continue

                        analysis = pacing_analyzer.classify_pacing_strategy(lap_times)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategy", analysis['strategy'])
                        with col2:
                            st.metric("CV", f"{analysis['coefficient_of_variation']:.2f}%")
                        with col3:
                            st.metric("Rank", row.get('Rank', 'N/A'))
                        with col4:
                            if analysis['is_elite_consistency']:
                                st.success("Elite Consistency")
                            else:
                                st.warning("Room to improve")

                        # Lap time chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=analysis['lap_times'],
                            mode='lines+markers',
                            name='Lap Times',
                            line=dict(color=TEAM_SAUDI_COLORS['primary_teal'], width=3),
                            marker=dict(size=10)
                        ))
                        fig.add_hline(y=np.mean(analysis['lap_times']), line_dash="dash",
                                     line_color=TEAM_SAUDI_COLORS['gold_accent'],
                                     annotation_text="Average")

                        # Mark fastest and slowest laps
                        fastest_idx = np.argmin(analysis['lap_times'])
                        slowest_idx = np.argmax(analysis['lap_times'])

                        fig.add_annotation(x=fastest_idx, y=analysis['lap_times'][fastest_idx],
                                          text="Fastest", showarrow=True, arrowhead=2,
                                          font=dict(color="green"))
                        fig.add_annotation(x=slowest_idx, y=analysis['lap_times'][slowest_idx],
                                          text="Slowest", showarrow=True, arrowhead=2,
                                          font=dict(color="red"))

                        fig.update_layout(
                            xaxis_title="Lap Number",
                            yaxis_title="Time (seconds)",
                            **create_team_saudi_chart_theme()
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Lap breakdown table
                        lap_df = pd.DataFrame({
                            'Lap': range(1, len(analysis['lap_times']) + 1),
                            'Time (s)': [f"{t:.2f}" for t in analysis['lap_times']],
                            'Diff from Avg': [f"{t - np.mean(analysis['lap_times']):+.2f}" for t in analysis['lap_times']]
                        })
                        st.dataframe(lap_df, use_container_width=True, hide_index=True)

                    except Exception as e:
                        st.error(f"Could not analyze pacing: {e}")


def show_competition_intel(df, course_type='all'):
    """Competitor intelligence page."""
    st.header("🏅 Competition Intelligence")

    intel = CompetitorIntelligence(df)

    tab1, tab2 = st.tabs(["Competitor Profile", "Field Comparison"])

    with tab1:
        st.subheader("Build Competitor Profile")

        # Country and Athlete selectors
        col1, col2 = st.columns([1, 3])

        with col1:
            countries = sorted(df['NAT'].dropna().unique())
            selected_country = st.selectbox("Filter by Country", ["All Countries"] + countries, key="intel_country")

        with col2:
            # Filter athletes by country
            if selected_country == "All Countries":
                filtered_df = df
            else:
                filtered_df = df[df['NAT'] == selected_country]

            athletes = sorted(filtered_df['FullName'].dropna().unique())
            selected_competitor = st.selectbox("Select Competitor", athletes if athletes else ["No athletes found"], key="intel_athlete")

        events = df['discipline_name'].dropna().unique()
        selected_event = st.selectbox("Filter by Event (optional)", ['All Events'] + sorted(events))

        if selected_competitor and selected_competitor != "No athletes found":
            event_filter = None if selected_event == 'All Events' else selected_event
            profile = intel.build_competitor_profile(selected_competitor, event_filter)

            if 'error' not in profile:
                # Main metrics
                st.markdown("### Overview")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Country", profile['country'])
                    st.metric("Races Analyzed", profile['races_analyzed'])

                with col2:
                    st.metric("Best Time", f"{profile['best_time_seconds']}s")
                    st.metric("Avg Time", f"{profile['avg_time_seconds']}s")

                with col3:
                    st.metric("Preferred Strategy", profile['preferred_pacing_strategy'])
                    st.metric("Consistency (σ)", profile['consistency_std'])

                with col4:
                    # Calculate medals if available
                    athlete_data = df[df['FullName'].str.contains(selected_competitor, case=False, na=False)]
                    if 'MedalTag' in athlete_data.columns:
                        medals = athlete_data['MedalTag'].value_counts()
                        gold = medals.get('G', 0)
                        silver = medals.get('S', 0)
                        bronze = medals.get('B', 0)
                        st.metric("Medals", f"🥇{gold} 🥈{silver} 🥉{bronze}")
                    years_active = athlete_data['year'].nunique() if 'year' in athlete_data.columns else 0
                    st.metric("Years Active", years_active)

                st.markdown("---")

                # Tactical Assessment
                st.markdown("### Tactical Assessment")
                st.info(f"**{profile['tactical_summary']}**")

                # Create two columns for pacing and race history
                col_left, col_right = st.columns(2)

                with col_left:
                    # Pacing distribution
                    if profile['pacing_distribution']:
                        st.markdown("### Pacing Pattern Distribution")
                        pacing_df = pd.DataFrame([
                            {'Strategy': k, 'Count': v}
                            for k, v in profile['pacing_distribution'].items()
                        ])
                        fig = px.pie(
                            pacing_df, values='Count', names='Strategy',
                            color_discrete_sequence=[
                                TEAM_SAUDI_COLORS['primary_teal'],
                                TEAM_SAUDI_COLORS['gold_accent'],
                                TEAM_SAUDI_COLORS['dark_teal'],
                                '#FFB800', '#0077B6'
                            ]
                        )
                        fig.update_layout(**create_team_saudi_chart_theme())
                        st.plotly_chart(fig, use_container_width=True)

                with col_right:
                    # Events competed in
                    st.markdown("### Events Competed")
                    event_counts = athlete_data['discipline_name'].value_counts().head(10)
                    event_df = pd.DataFrame({'Event': event_counts.index, 'Races': event_counts.values})
                    st.dataframe(event_df, use_container_width=True, hide_index=True)

                # Detailed Race History
                st.markdown("---")
                st.markdown("### Race History by Event")

                # Group races by event
                if event_filter:
                    events_to_show = [event_filter]
                else:
                    events_to_show = athlete_data['discipline_name'].dropna().unique()

                num_events_to_show = st.slider("Events to display", 1, min(10, len(events_to_show)), 3, key="num_events")

                for event_name in list(events_to_show)[:num_events_to_show]:
                    event_races = athlete_data[athlete_data['discipline_name'] == event_name].copy()
                    if event_races.empty:
                        continue

                    # Remove duplicate races (same competition, time, year)
                    dedup_cols = ['competition_name', 'Time', 'year']
                    available_dedup = [c for c in dedup_cols if c in event_races.columns]
                    if available_dedup:
                        event_races = event_races.drop_duplicates(subset=available_dedup, keep='first')

                    # Convert times
                    from enhanced_swimming_scraper import SplitTimeAnalyzer
                    time_analyzer = SplitTimeAnalyzer()
                    event_races['time_seconds'] = event_races['Time'].apply(time_analyzer.time_to_seconds)
                    event_races = event_races.sort_values('time_seconds')

                    with st.expander(f"📊 {event_name} ({len(event_races)} races)", expanded=True):
                        # Best time highlight
                        best_race = event_races.iloc[0] if len(event_races) > 0 else None
                        if best_race is not None:
                            st.success(f"**PB: {best_race.get('Time', 'N/A')}** at {best_race.get('competition_name', 'Unknown')[:50]} ({best_race.get('year', 'N/A')})")

                        # Race table
                        display_cols = ['Time', 'Rank', 'competition_name', 'year', 'pacing_type']
                        available_cols = [c for c in display_cols if c in event_races.columns]
                        race_display = event_races[available_cols].head(10).copy()
                        race_display.columns = ['Time', 'Rank', 'Competition', 'Year', 'Pacing'][:len(available_cols)]
                        st.dataframe(race_display, use_container_width=True, hide_index=True)

                        # Progression chart if multiple races
                        if len(event_races) > 1 and 'year' in event_races.columns:
                            yearly_best = event_races.groupby('year')['time_seconds'].min().reset_index()
                            fig = px.line(
                                yearly_best, x='year', y='time_seconds',
                                markers=True,
                                title=f"Progression - {event_name}"
                            )
                            fig.update_traces(line_color=TEAM_SAUDI_COLORS['primary_teal'], marker_size=10)
                            fig.update_layout(
                                xaxis_title="Year",
                                yaxis_title="Time (seconds)",
                                **create_team_saudi_chart_theme()
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(profile['error'])

    with tab2:
        st.subheader("Compare to Competition Field")

        st.markdown("Compare your athlete's performance against the competition field in their events.")

        # Country filter for athlete selection
        col1, col2 = st.columns([1, 3])

        with col1:
            compare_countries = sorted(df['NAT'].dropna().unique())
            compare_country = st.selectbox("Filter by Country", ["All Countries"] + compare_countries, key="compare_country")

        with col2:
            if compare_country == "All Countries":
                athlete_pool = df
            else:
                athlete_pool = df[df['NAT'] == compare_country]

            athletes = sorted(athlete_pool['FullName'].dropna().unique())
            target = st.selectbox("Your Athlete", athletes if athletes else ["No athletes found"], key="target")

        # Only show events this athlete has competed in
        if target and target != "No athletes found":
            athlete_data = df[df['FullName'].str.contains(target, case=False, na=False)]
            athlete_events = sorted(athlete_data['discipline_name'].dropna().unique())

            if athlete_events:
                event = st.selectbox(f"Event ({len(athlete_events)} events for this athlete)", athlete_events, key="event_compare")

                # Show athlete's stats in this event
                event_data = athlete_data[athlete_data['discipline_name'] == event]
                st.info(f"**{target}** has **{len(event_data)} races** in {event}")

                comparison = intel.compare_to_field(target, event)

                if 'error' not in comparison:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Your Best", f"{comparison['target_best']}s")
                        st.metric("Current Ranking", f"#{comparison['current_ranking']} of {comparison['field_size']}")

                    with col2:
                        gap = comparison['gap_to_medal_seconds']
                        st.metric("Gap to Medal", f"{gap:+.2f}s")

                    with col3:
                        if comparison['medal_potential']:
                            st.success("🏅 Medal Potential: YES")
                        else:
                            st.warning("📈 Work needed for medal contention")

                    st.subheader("Top 5 in Field")
                    top5_df = pd.DataFrame(comparison['top_5'])
                    st.dataframe(top5_df, use_container_width=True)

                    # Show where athlete ranks in the field
                    st.subheader("Field Position Analysis")
                    col_a, col_b = st.columns(2)

                    with col_a:
                        # Calculate percentile
                        ranking = comparison['current_ranking']
                        field_size = comparison['field_size']
                        percentile = ((field_size - ranking) / field_size) * 100
                        st.metric("Percentile", f"Top {100-percentile:.1f}%")

                    with col_b:
                        # Gap to leader
                        if comparison['top_5']:
                            leader_time = comparison['top_5'][0]['best_time']
                            gap_to_leader = comparison['target_best'] - leader_time
                            st.metric("Gap to Leader", f"{gap_to_leader:+.2f}s")
                else:
                    st.error(comparison['error'])
            else:
                st.warning(f"No event data found for {target}")


def show_race_preparation(df, course_type='all'):
    """Race preparation briefing page."""
    st.header("📋 Race Preparation")

    round_analyzer = RaceRoundAnalyzer(df)
    intel = CompetitorIntelligence(df)

    tab1, tab2 = st.tabs(["Heats-to-Finals Progression", "Race Brief Generator"])

    with tab1:
        st.subheader("Round-by-Round Performance")

        st.info(f"""
        **Research Benchmark:**
        Medalists improve {ELITE_BENCHMARKS['heats_to_finals_improvement']}% from heats to finals.
        Non-medalists show minimal or negative progression.
        """)

        # Country and Athlete selectors
        col1, col2 = st.columns([1, 3])

        with col1:
            round_countries = sorted(df['NAT'].dropna().unique())
            round_country = st.selectbox("Filter by Country", ["All Countries"] + round_countries, key="round_country")

        with col2:
            if round_country == "All Countries":
                athlete_pool = df
            else:
                athlete_pool = df[df['NAT'] == round_country]

            athletes = sorted(athlete_pool['FullName'].dropna().unique())
            selected_athlete = st.selectbox("Select Athlete", athletes if athletes else ["No athletes found"], key="round_athlete")

        if selected_athlete and selected_athlete != "No athletes found":
            progressions = round_analyzer.analyze_heats_to_finals(selected_athlete)

            if progressions:
                st.markdown(f"**Found {len(progressions)} multi-round competitions**")
                st.markdown("*Times shown are best times from each round. Expand for full race details.*")

                for prog in progressions:
                    # Get all races for this competition/event combo
                    athlete_data = df[
                        (df['FullName'].str.contains(selected_athlete, case=False, na=False)) &
                        (df['competition_name'] == prog['competition']) &
                        (df['discipline_name'] == prog['event'])
                    ].copy()

                    rounds_count = len(athlete_data)

                    with st.expander(f"📊 {prog['event']} - {prog['competition'][:50]} ({rounds_count} races)"):
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Heat (Best)", f"{prog['heat_time']}s")

                        with col2:
                            if 'semi_time' in prog:
                                heat_to_semi = prog.get('heat_to_semi_improvement', 0)
                                st.metric("Semi (Best)", f"{prog['semi_time']}s", delta=f"{heat_to_semi:+.2f}%")
                            else:
                                st.metric("Semi", "N/A")

                        with col3:
                            if 'final_time' in prog:
                                improvement = prog.get('heat_to_final_improvement', 0)
                                st.metric("Final (Best)", f"{prog['final_time']}s", delta=f"{improvement:+.2f}%")
                            else:
                                st.metric("Final", "N/A")

                        with col4:
                            if 'heat_to_final_improvement' in prog:
                                if prog.get('meets_elite_progression'):
                                    st.success("✅ Elite Progression")
                                else:
                                    st.warning("⚠️ Below Benchmark")

                        st.markdown("---")
                        st.markdown("**All Races in this Competition:**")

                        # Show all individual races
                        from enhanced_swimming_scraper import SplitTimeAnalyzer
                        time_analyzer = SplitTimeAnalyzer()
                        athlete_data['time_seconds'] = athlete_data['Time'].apply(time_analyzer.time_to_seconds)

                        # Sort by round order (Heats -> Semi -> Final)
                        def round_order(x):
                            cat = str(x).lower()
                            if 'final' in cat and 'semi' not in cat:
                                return 3
                            elif 'semi' in cat:
                                return 2
                            elif 'heat' in cat:
                                return 1
                            return 0

                        athlete_data['round_order'] = athlete_data['heat_category'].apply(round_order)
                        athlete_data = athlete_data.sort_values('round_order')

                        # Display table with all races
                        display_cols = ['heat_category', 'Time', 'Rank', 'Lane', 'RT', 'pacing_type']
                        available_cols = [c for c in display_cols if c in athlete_data.columns]
                        race_table = athlete_data[available_cols].copy()
                        race_table.columns = ['Round', 'Time', 'Rank', 'Lane', 'Reaction', 'Pacing'][:len(available_cols)]
                        st.dataframe(race_table, use_container_width=True, hide_index=True)

                        # Progression chart
                        if len(athlete_data) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=athlete_data['heat_category'],
                                y=athlete_data['time_seconds'],
                                mode='lines+markers+text',
                                text=[f"{t:.2f}s" for t in athlete_data['time_seconds']],
                                textposition='top center',
                                line=dict(color=TEAM_SAUDI_COLORS['primary_teal'], width=3),
                                marker=dict(size=12)
                            ))
                            fig.update_layout(
                                title="Round-by-Round Progression",
                                xaxis_title="Round",
                                yaxis_title="Time (seconds)",
                                **create_team_saudi_chart_theme()
                            )
                            # Invert y-axis so faster times are higher
                            fig.update_yaxes(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No multi-round competitions found for this athlete. This analysis requires competitions with heats, semis, or finals.")

    with tab2:
        st.subheader("Generate Race Brief")

        target = st.selectbox("Your Athlete", sorted(df['FullName'].dropna().unique()), key="brief_target")
        event = st.selectbox("Event", sorted(df['discipline_name'].dropna().unique()), key="brief_event")

        # Get potential competitors
        event_data = df[df['discipline_name'].str.contains(event, case=False, na=False)]
        potential_competitors = [a for a in event_data['FullName'].dropna().unique() if a != target]

        competitors = st.multiselect("Key Competitors", potential_competitors[:50])

        if st.button("Generate Brief", type="primary"):
            if target and event and competitors:
                brief = CoachingReportGenerator.race_preparation_brief(
                    intel, target, competitors, event
                )
                st.text(brief)

                # Download button
                st.download_button(
                    "Download Brief",
                    brief,
                    file_name=f"race_brief_{target.replace(' ', '_')}_{event.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Please select athlete, event, and at least one competitor")


def show_advanced_analytics(df, course_type='all'):
    """Advanced analytics page with predictive modeling and KPI analysis."""
    st.header("🚀 Advanced Analytics")

    # Use appropriate world records based on course type
    world_records = get_world_records(course_type if course_type != 'all' else 'lcm')

    st.info("""
    **Elite-Level Analytics Features:**
    - Target Time Calculator with split projections
    - Performance Forecasting using trend analysis
    - Advanced KPI Analysis (reaction times, consistency metrics)
    - Race Strategy Optimizer
    """)

    predictor = PredictivePerformanceModel(df)
    kpi_analyzer = AdvancedKPIAnalyzer(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Time Calculator",
        "Performance Forecast",
        "KPI Analysis",
        "Strategy Optimizer"
    ])

    with tab1:
        st.subheader("Target Time Calculator")
        st.markdown("Calculate optimal splits to achieve your target time.")

        col1, col2 = st.columns(2)

        with col1:
            # Event selection for context
            events = sorted(df['discipline_name'].dropna().unique())
            selected_event = st.selectbox("Select Event", events, key="target_event")

            # Determine default laps based on event
            default_laps = 4
            if '50m' in selected_event:
                default_laps = 1
            elif '100m' in selected_event:
                default_laps = 2
            elif '200m' in selected_event:
                default_laps = 4
            elif '400m' in selected_event:
                default_laps = 8
            elif '800m' in selected_event:
                default_laps = 16
            elif '1500m' in selected_event:
                default_laps = 30

            num_laps = st.number_input("Number of Laps (50m each)", min_value=1, max_value=30, value=default_laps)

            # Target time input
            st.markdown("**Enter Target Time:**")
            col_min, col_sec = st.columns(2)
            with col_min:
                target_mins = st.number_input("Minutes", min_value=0, max_value=30, value=1)
            with col_sec:
                target_secs = st.number_input("Seconds", min_value=0.0, max_value=59.99, value=0.0, format="%.2f")

            target_time = target_mins * 60 + target_secs

            # Strategy selection
            strategy = st.selectbox("Pacing Strategy", [
                ("even", "Even Pace - Consistent throughout"),
                ("negative", "Negative Split - Build speed (distance events)"),
                ("u_shape", "U-Shape - Fast start/finish (400m specialists)"),
                ("front_loaded", "Front Loaded - Fast early, maintain (sprints)")
            ], format_func=lambda x: x[1])[0]

        with col2:
            if target_time > 0:
                splits = predictor.calculate_optimal_splits(target_time, num_laps, strategy)

                if 'error' not in splits:
                    st.markdown(f"### Target: **{splits['target_time_formatted']}**")
                    st.metric("Split Consistency (CV)", f"{splits['cv']:.2f}%",
                             help="Elite swimmers: < 1.3% CV")

                    # Compare to world record (use dynamic world_records based on course type)
                    for wr_event, wr_time in world_records.items():
                        if selected_event.lower() in wr_event.lower() or wr_event.lower() in selected_event.lower():
                            wr_pct = (wr_time / target_time) * 100
                            st.metric("World Record %", f"{wr_pct:.1f}%",
                                     delta=f"{target_time - wr_time:+.2f}s from WR")
                            break

                    # Display splits
                    st.markdown("### Optimal Split Times")
                    split_df = pd.DataFrame({
                        'Lap': range(1, num_laps + 1),
                        'Split (s)': splits['optimal_splits'],
                        'Cumulative': splits['cumulative_times']
                    })
                    st.dataframe(split_df, use_container_width=True, hide_index=True)

                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=list(range(1, num_laps + 1)),
                        y=splits['optimal_splits'],
                        marker_color=TEAM_SAUDI_COLORS['primary_teal'],
                        name='Target Splits'
                    ))
                    fig.add_hline(y=np.mean(splits['optimal_splits']), line_dash="dash",
                                 line_color=TEAM_SAUDI_COLORS['gold_accent'],
                                 annotation_text=f"Avg: {np.mean(splits['optimal_splits']):.2f}s")
                    fig.update_layout(
                        xaxis_title="Lap",
                        yaxis_title="Time (seconds)",
                        title=f"Target Split Profile - {strategy.replace('_', ' ').title()}",
                        **create_team_saudi_chart_theme()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(splits['error'])

    with tab2:
        st.subheader("Performance Forecast")
        st.markdown("Predict future performance based on historical improvement trends.")

        # Athlete and event selection
        col1, col2 = st.columns([1, 2])

        with col1:
            countries = sorted(df['NAT'].dropna().unique())
            forecast_country = st.selectbox("Filter by Country", ["All Countries"] + countries, key="forecast_country")

        with col2:
            if forecast_country == "All Countries":
                athlete_pool = df
            else:
                athlete_pool = df[df['NAT'] == forecast_country]

            athletes = sorted(athlete_pool['FullName'].dropna().unique())
            forecast_athlete = st.selectbox("Select Athlete", athletes if athletes else ["No athletes found"], key="forecast_athlete")

        if forecast_athlete and forecast_athlete != "No athletes found":
            # Get events for this athlete
            athlete_data = df[df['FullName'].str.contains(forecast_athlete, case=False, na=False)]
            athlete_events = sorted(athlete_data['discipline_name'].dropna().unique())

            forecast_event = st.selectbox("Select Event", athlete_events, key="forecast_event")

            col1, col2 = st.columns(2)
            with col1:
                years_ahead = st.slider("Forecast Years Ahead", 1, 3, 1)

            if st.button("Generate Forecast", type="primary"):
                forecast = predictor.forecast_performance(forecast_athlete, forecast_event, years_ahead)

                if 'error' not in forecast:
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Best", forecast['current_best_formatted'])
                        st.metric("Improvement/Year", f"{forecast['improvement_per_year']:.2f}s")

                    with col2:
                        st.metric(f"Forecast ({forecast['forecast_year']})",
                                 forecast['forecast_time_formatted'],
                                 delta=f"{forecast['current_best'] - forecast['forecast_time']:.2f}s improvement" if forecast['trend'] == 'improving' else None)

                    with col3:
                        confidence_color = "green" if forecast['confidence_level'] == 'High' else "orange" if forecast['confidence_level'] == 'Medium' else "red"
                        st.metric("Confidence", forecast['confidence_level'])
                        st.metric("R-squared", f"{forecast['confidence_r_squared']:.3f}")
                        trend_icon = "📈" if forecast['trend'] == 'improving' else "📉"
                        st.write(f"Trend: {trend_icon} {forecast['trend'].title()}")

                    # Progression chart
                    yearly_df = pd.DataFrame(forecast['yearly_data'])
                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=yearly_df['year'],
                        y=yearly_df['best_time'],
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color=TEAM_SAUDI_COLORS['primary_teal'], width=3),
                        marker=dict(size=10)
                    ))

                    # Forecast point
                    fig.add_trace(go.Scatter(
                        x=[forecast['forecast_year']],
                        y=[forecast['forecast_time']],
                        mode='markers',
                        name='Forecast',
                        marker=dict(color=TEAM_SAUDI_COLORS['gold_accent'], size=15, symbol='star')
                    ))

                    # Trend line
                    all_years = list(yearly_df['year']) + [forecast['forecast_year']]
                    trend_line = [forecast['current_best'] - forecast['improvement_per_year'] * (y - yearly_df['year'].max())
                                 for y in all_years]
                    fig.add_trace(go.Scatter(
                        x=all_years,
                        y=trend_line,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color=TEAM_SAUDI_COLORS['gold_accent'], dash='dash', width=2)
                    ))

                    fig.update_layout(
                        title=f"Performance Forecast - {forecast_athlete}",
                        xaxis_title="Year",
                        yaxis_title="Time (seconds)",
                        **create_team_saudi_chart_theme()
                    )
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(forecast['error'])

    with tab3:
        st.subheader("Advanced KPI Analysis")
        st.markdown("Analyze key performance indicators used by elite coaching systems.")

        col1, col2 = st.columns([1, 2])

        with col1:
            kpi_countries = sorted(df['NAT'].dropna().unique())
            kpi_country = st.selectbox("Filter by Country", ["All Countries"] + kpi_countries, key="kpi_country")

        with col2:
            if kpi_country == "All Countries":
                kpi_pool = df
            else:
                kpi_pool = df[df['NAT'] == kpi_country]

            kpi_athletes = sorted(kpi_pool['FullName'].dropna().unique())
            kpi_athlete = st.selectbox("Select Athlete", kpi_athletes if kpi_athletes else ["No athletes found"], key="kpi_athlete")

        if kpi_athlete and kpi_athlete != "No athletes found":
            kpi_tab1, kpi_tab2, kpi_tab3 = st.tabs(["Race Efficiency", "Reaction Time", "Lane Analysis"])

            with kpi_tab1:
                st.markdown("### Race Efficiency Metrics")

                # Get athlete events
                athlete_data = df[df['FullName'].str.contains(kpi_athlete, case=False, na=False)]
                kpi_events = ["All Events"] + sorted(athlete_data['discipline_name'].dropna().unique())
                kpi_event = st.selectbox("Filter by Event", kpi_events, key="kpi_efficiency_event")

                event_filter = None if kpi_event == "All Events" else kpi_event
                efficiency = kpi_analyzer.analyze_race_efficiency(kpi_athlete, event_filter)

                if 'error' not in efficiency:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Races", efficiency['total_races'])
                        st.metric("Races with Splits", efficiency['races_with_splits'])

                    with col2:
                        if 'avg_consistency_cv' in efficiency:
                            cv = efficiency['avg_consistency_cv']
                            st.metric("Avg Consistency (CV)", f"{cv:.2f}%")
                            if efficiency['is_elite_consistency']:
                                st.success("Elite-level consistency")
                            else:
                                st.warning(f"Target: < {ELITE_BENCHMARKS['cv_elite_threshold']}%")

                    with col3:
                        if 'avg_lap_range' in efficiency:
                            st.metric("Avg Lap Range", f"{efficiency['avg_lap_range']:.2f}s")

                    # Race analysis table
                    if 'race_analyses' in efficiency and efficiency['race_analyses']:
                        st.markdown("### Race-by-Race Analysis")
                        race_df = pd.DataFrame(efficiency['race_analyses'])
                        race_df['cv'] = race_df['cv'].round(2)
                        race_df['avg_lap'] = race_df['avg_lap'].round(2)
                        race_df['total_time'] = race_df['total_time'].round(2)
                        st.dataframe(race_df[['race', 'total_time', 'num_laps', 'avg_lap', 'cv', 'fastest_lap', 'slowest_lap']],
                                    use_container_width=True, hide_index=True)

                        # CV distribution chart
                        fig = px.histogram(race_df, x='cv', nbins=10,
                                          title="Consistency Distribution (CV %)",
                                          color_discrete_sequence=[TEAM_SAUDI_COLORS['primary_teal']])
                        fig.add_vline(x=ELITE_BENCHMARKS['cv_elite_threshold'], line_dash="dash",
                                     line_color=TEAM_SAUDI_COLORS['gold_accent'],
                                     annotation_text="Elite Threshold")
                        fig.update_layout(**create_team_saudi_chart_theme())
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(efficiency.get('message', efficiency.get('error', 'No data available')))

            with kpi_tab2:
                st.markdown("### Reaction Time Analysis")
                st.info("Elite benchmark: < 0.65 seconds")

                rt_analysis = kpi_analyzer.calculate_reaction_time_stats(kpi_athlete)

                if 'error' not in rt_analysis:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Average RT", f"{rt_analysis['avg_reaction_time']:.3f}s")

                    with col2:
                        st.metric("Best RT", f"{rt_analysis['best_reaction_time']:.3f}s")

                    with col3:
                        st.metric("Worst RT", f"{rt_analysis['worst_reaction_time']:.3f}s")

                    with col4:
                        rating_color = "green" if rt_analysis['rating'] == 'Elite' else "orange" if rt_analysis['rating'] == 'Good' else "red"
                        st.metric("Rating", rt_analysis['rating'])

                    # Improvement potential
                    if rt_analysis['improvement_potential'] > 0:
                        st.warning(f"Improvement Potential: {rt_analysis['improvement_potential']:.3f}s to reach elite level")
                    else:
                        st.success("Already at elite reaction time level")

                    # Consistency metric
                    st.metric("RT Consistency (std)", f"{rt_analysis['rt_consistency']:.3f}s",
                             help="Lower is better - shows consistency across races")
                else:
                    st.error(rt_analysis['error'])

            with kpi_tab3:
                st.markdown("### Lane Performance Analysis")
                st.info("Research shows slight advantages in center lanes (4-5)")

                lane_analysis = kpi_analyzer.analyze_lane_performance(kpi_athlete)

                if 'error' not in lane_analysis:
                    st.metric("Best Lane", f"Lane {lane_analysis['best_lane']}")
                    st.metric("Races Analyzed", lane_analysis['races_analyzed'])

                    if lane_analysis['lane_analysis']:
                        lane_df = pd.DataFrame(lane_analysis['lane_analysis'])
                        lane_df['avg_rank'] = lane_df['avg_rank'].round(2)
                        lane_df['avg_time'] = lane_df['avg_time'].round(2)

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=lane_df['lane'],
                            y=lane_df['avg_rank'],
                            marker_color=[TEAM_SAUDI_COLORS['gold_accent'] if l == lane_analysis['best_lane']
                                         else TEAM_SAUDI_COLORS['primary_teal'] for l in lane_df['lane']],
                            text=lane_df['count'],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title="Average Finish Position by Lane",
                            xaxis_title="Lane",
                            yaxis_title="Average Rank (lower is better)",
                            **create_team_saudi_chart_theme()
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(lane_df, use_container_width=True, hide_index=True)
                else:
                    st.error(lane_analysis['error'])

    with tab4:
        st.subheader("Race Strategy Optimizer")
        st.markdown("Input your current splits to analyze and optimize your race strategy.")

        st.markdown("**Enter Your Split Times (in seconds):**")

        # Dynamic split input
        num_splits = st.number_input("Number of Splits", min_value=2, max_value=30, value=4)

        split_cols = st.columns(min(num_splits, 8))
        splits = []
        for i in range(num_splits):
            col_idx = i % 8
            with split_cols[col_idx]:
                split = st.number_input(f"Lap {i+1}", min_value=0.0, max_value=120.0, value=30.0, format="%.2f", key=f"split_{i}")
                splits.append(split)

        # Optional event for WR comparison
        optimizer_events = sorted(df['discipline_name'].dropna().unique())
        optimizer_event = st.selectbox("Event (for WR comparison)", ["None"] + optimizer_events, key="optimizer_event")

        if st.button("Analyze Strategy", type="primary"):
            event_for_analysis = None if optimizer_event == "None" else optimizer_event
            analysis = predictor.predict_target_time_from_splits(splits, event_for_analysis)

            if 'error' not in analysis:
                st.markdown("---")
                st.markdown("### Analysis Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicted Time", analysis['predicted_time_formatted'])
                    st.metric("Average Split", f"{analysis['avg_split']:.2f}s")

                with col2:
                    st.metric("Pacing Strategy", analysis['pacing_strategy'])
                    cv = analysis['split_cv']
                    st.metric("Split Consistency (CV)", f"{cv:.2f}%")

                with col3:
                    if analysis['is_elite_consistency']:
                        st.success("Elite-level consistency")
                    else:
                        st.warning(f"Target CV: < {ELITE_BENCHMARKS['cv_elite_threshold']}%")

                    if 'wr_comparison' in analysis:
                        st.metric("World Record %", f"{analysis['wr_comparison']:.1f}%")
                        st.metric("Gap to WR", f"+{analysis['gap_to_wr']:.2f}s")

                # Split visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(range(1, len(splits) + 1)),
                    y=splits,
                    marker_color=TEAM_SAUDI_COLORS['primary_teal'],
                    name='Your Splits'
                ))
                fig.add_hline(y=analysis['avg_split'], line_dash="dash",
                             line_color=TEAM_SAUDI_COLORS['gold_accent'],
                             annotation_text=f"Avg: {analysis['avg_split']:.2f}s")

                # Mark fastest and slowest
                fastest_idx = splits.index(min(splits))
                slowest_idx = splits.index(max(splits))

                fig.add_annotation(x=fastest_idx + 1, y=splits[fastest_idx],
                                  text="Fastest", showarrow=True, arrowhead=2,
                                  font=dict(color="green"))
                fig.add_annotation(x=slowest_idx + 1, y=splits[slowest_idx],
                                  text="Slowest", showarrow=True, arrowhead=2,
                                  font=dict(color="red"))

                fig.update_layout(
                    title="Your Split Profile",
                    xaxis_title="Lap",
                    yaxis_title="Time (seconds)",
                    **create_team_saudi_chart_theme()
                )
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations
                st.markdown("### Recommendations")

                if analysis['pacing_strategy'] == "Positive Split (Fade)":
                    st.warning("""
                    **Pacing Issue Detected:** Your splits show a positive split (fading) pattern.

                    **Recommendations:**
                    - Work on endurance to maintain pace in later laps
                    - Consider a slightly more conservative start
                    - Focus on maintaining technique when fatigued
                    """)
                elif analysis['pacing_strategy'] == "Negative Split (Fast Finish)":
                    st.success("""
                    **Strong Finish Pattern:** You're finishing faster than you started.

                    **Considerations:**
                    - You may have room to push harder early
                    - This pattern works well for distance events
                    - Maintain this ability for championship racing
                    """)
                else:
                    st.success("""
                    **Even Pacing:** Good consistent pacing throughout the race.

                    **Tips:**
                    - This is optimal for most middle-distance events
                    - Focus on maintaining this consistency at faster overall pace
                    """)

                if not analysis['is_elite_consistency']:
                    st.info(f"""
                    **Consistency Improvement:** Your CV of {cv:.2f}% is above the elite threshold of {ELITE_BENCHMARKS['cv_elite_threshold']}%.

                    Focus on reducing the gap between your fastest and slowest laps ({max(splits) - min(splits):.2f}s range).
                    """)
            else:
                st.error(analysis['error'])


def show_performance_benchmarks(df, course_type='all'):
    """Performance Benchmarks - What It Takes to Win at each level."""
    st.header("🏆 Performance Benchmarks")

    # Use appropriate world records based on course type
    world_records = get_world_records(course_type if course_type != 'all' else 'lcm')
    course_label = "Short Course (25m)" if course_type == 'scm' else "Long Course (50m)"

    st.markdown(f"""
    **What It Takes to Win** - Performance thresholds at different competition levels,
    age-based progression targets, and pathway to elite status.

    *Benchmarks based on {course_label} World Records*
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Competition Levels",
        "Age Progression",
        "Target Calculator",
        "Athlete Assessment"
    ])

    with tab1:
        st.subheader("What It Takes to Win - By Competition Level")

        st.markdown("""
        **Actual times** required to medal at different competition levels.
        Select an event to see specific target times based on World Records.
        """)

        # Event selection
        col_gender, col_event = st.columns(2)

        # Get world records based on course type
        wr_dict = get_world_records(course_type)

        with col_gender:
            gender = st.selectbox("Gender", ["Men", "Women"], key="benchmark_gender")

        # Filter events by gender
        gender_events = [e for e in wr_dict.keys() if e.startswith(gender)]
        event_display = [e.replace(f"{gender} ", "") for e in gender_events]

        with col_event:
            selected_event_display = st.selectbox("Select Event", event_display, key="benchmark_event")

        selected_event = f"{gender} {selected_event_display}"

        # Get world record time
        wr_time = wr_dict.get(selected_event)

        if wr_time:
            # Helper to format seconds to time string
            def seconds_to_time(secs):
                if secs >= 60:
                    mins = int(secs // 60)
                    remaining = secs % 60
                    return f"{mins}:{remaining:05.2f}"
                return f"{secs:.2f}"

            # Calculate target times for each competition level
            st.markdown(f"### {selected_event}")
            st.info(f"**World Record: {seconds_to_time(wr_time)}**")

            # Build target times table
            competitions = ['Olympic Games', 'World Championships', 'Asian Games', 'Asian Championships', 'GCC Championships', 'Arab Championships']
            target_data = []

            for comp in competitions:
                benchmarks = COMPETITION_BENCHMARKS.get(comp, {})
                if benchmarks:
                    gold_pct = benchmarks.get('gold', 98) / 100
                    medal_pct = benchmarks.get('medal', 97) / 100
                    final_pct = benchmarks.get('final', 95) / 100

                    gold_time = wr_time / gold_pct
                    medal_time = wr_time / medal_pct
                    final_time = wr_time / final_pct

                    target_data.append({
                        'Competition': comp,
                        'Gold': seconds_to_time(gold_time),
                        'Medal': seconds_to_time(medal_time),
                        'Finals': seconds_to_time(final_time),
                        'Gold_sec': gold_time,
                        'Medal_sec': medal_time,
                        'Finals_sec': final_time
                    })

            target_df = pd.DataFrame(target_data)

            # Display as formatted table
            st.markdown("### Target Times by Competition")
            display_df = target_df[['Competition', 'Gold', 'Medal', 'Finals']].copy()
            display_df.columns = ['Competition', '🥇 Gold', '🥈🥉 Medal', '🏊 Finals']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Bar chart showing actual times (in seconds)
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='🥇 Gold',
                x=target_df['Competition'],
                y=target_df['Gold_sec'],
                marker_color='#FFD700',
                text=target_df['Gold'],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='🥈🥉 Medal',
                x=target_df['Competition'],
                y=target_df['Medal_sec'],
                marker_color='#C0C0C0',
                text=target_df['Medal'],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='🏊 Finals',
                x=target_df['Competition'],
                y=target_df['Finals_sec'],
                marker_color='#CD7F32',
                text=target_df['Finals'],
                textposition='outside'
            ))

            # Add World Record line
            fig.add_hline(y=wr_time, line_dash="dash", line_color=TEAM_SAUDI_COLORS['primary_teal'],
                         annotation_text=f"WR: {seconds_to_time(wr_time)}", annotation_position="right")

            fig.update_layout(
                barmode='group',
                title=f"Target Times for {selected_event}",
                yaxis_title="Time (seconds)",
                xaxis_title="Competition",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                **create_team_saudi_chart_theme()
            )
            st.plotly_chart(fig, use_container_width=True)

            # Quick reference cards
            st.markdown("### Quick Reference - Key Targets")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1rem; border-radius: 8px; text-align: center;">
                    <p style="color: #333; margin: 0; font-size: 0.9rem;">🥇 Olympic/World Gold</p>
                    <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{target_data[0]['Gold']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                asian_gold = next((t['Gold'] for t in target_data if t['Competition'] == 'Asian Games'), 'N/A')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {TEAM_SAUDI_COLORS['primary_teal']} 0%, {TEAM_SAUDI_COLORS['dark_teal']} 100%); padding: 1rem; border-radius: 8px; text-align: center;">
                    <p style="color: white; margin: 0; font-size: 0.9rem;">🥇 Asian Games Gold</p>
                    <p style="color: {TEAM_SAUDI_COLORS['gold_accent']}; margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{asian_gold}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                gcc_gold = next((t['Gold'] for t in target_data if t['Competition'] == 'GCC Championships'), 'N/A')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #6c757d 0%, #495057 100%); padding: 1rem; border-radius: 8px; text-align: center;">
                    <p style="color: white; margin: 0; font-size: 0.9rem;">🥇 GCC Gold</p>
                    <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: bold;">{gcc_gold}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning(f"World Record not available for {selected_event}")

    with tab2:
        st.subheader("Age-Based Progression Targets")

        st.markdown("""
        Expected WR% progression by age for athletes on the elite pathway.
        **Target** = good national level, **Elite** = international medal potential.
        """)

        gender = st.radio("Select Gender", ["Male", "Female"], horizontal=True)
        gender_key = gender.lower()

        # Get age data
        age_data = AGE_PROGRESSION_BENCHMARKS[gender_key]

        ages = sorted(age_data.keys())
        targets = [age_data[a]['target_wr_pct'] for a in ages]
        elite = [age_data[a]['elite_wr_pct'] for a in ages]
        phases = [age_data[a]['phase'] for a in ages]

        # Create progression chart
        fig = go.Figure()

        # Elite zone (shaded area)
        fig.add_trace(go.Scatter(
            x=ages + ages[::-1],
            y=elite + targets[::-1],
            fill='toself',
            fillcolor='rgba(0, 113, 103, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Target Zone',
            showlegend=True
        ))

        # Elite line
        fig.add_trace(go.Scatter(
            x=ages,
            y=elite,
            mode='lines+markers',
            name='Elite Trajectory',
            line=dict(color=TEAM_SAUDI_COLORS['gold_accent'], width=3),
            marker=dict(size=10)
        ))

        # Target line
        fig.add_trace(go.Scatter(
            x=ages,
            y=targets,
            mode='lines+markers',
            name='Target Trajectory',
            line=dict(color=TEAM_SAUDI_COLORS['primary_teal'], width=3),
            marker=dict(size=10)
        ))

        # Add phase annotations
        phase_colors = {
            'Junior Development': '#e3f2fd',
            'Junior Transition': '#fff3e0',
            'Senior Development': '#e8f5e9',
            'Senior Elite': '#fce4ec',
            'Peak Window': TEAM_SAUDI_COLORS['gold_accent'],
            'Maintenance': '#f5f5f5',
            'Veteran': '#eceff1'
        }

        fig.update_layout(
            title=f"{gender} Age Progression Benchmarks",
            xaxis_title="Age",
            yaxis_title="% of World Record",
            yaxis=dict(range=[80, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            **create_team_saudi_chart_theme()
        )
        st.plotly_chart(fig, use_container_width=True)

        # Phase breakdown table
        st.markdown("### Development Phases")

        phase_data = []
        current_phase = None
        for age in ages:
            phase = age_data[age]['phase']
            if phase != current_phase:
                current_phase = phase
                improvement = IMPROVEMENT_EXPECTATIONS.get(phase, {})
                phase_data.append({
                    'Phase': phase,
                    'Start Age': age,
                    'Target WR%': age_data[age]['target_wr_pct'],
                    'Elite WR%': age_data[age]['elite_wr_pct'],
                    'Typical Improvement/Year': f"{improvement.get('typical', 'N/A')}%",
                    'Elite Improvement/Year': f"{improvement.get('elite', 'N/A')}%"
                })

        phase_df = pd.DataFrame(phase_data)
        st.dataframe(phase_df, use_container_width=True, hide_index=True)

        # Improvement expectations chart
        st.markdown("### Expected Annual Improvement by Phase")

        imp_phases = list(IMPROVEMENT_EXPECTATIONS.keys())
        typical_imp = [IMPROVEMENT_EXPECTATIONS[p]['typical'] for p in imp_phases]
        elite_imp = [IMPROVEMENT_EXPECTATIONS[p]['elite'] for p in imp_phases]

        imp_fig = go.Figure()
        imp_fig.add_trace(go.Bar(
            name='Typical Athlete',
            x=imp_phases,
            y=typical_imp,
            marker_color=TEAM_SAUDI_COLORS['primary_teal']
        ))
        imp_fig.add_trace(go.Bar(
            name='Elite Athlete',
            x=imp_phases,
            y=elite_imp,
            marker_color=TEAM_SAUDI_COLORS['gold_accent']
        ))
        imp_fig.add_hline(y=0, line_dash="solid", line_color="gray")

        imp_fig.update_layout(
            barmode='group',
            title="Expected Annual Improvement (% per year)",
            xaxis_title="Development Phase",
            yaxis_title="Improvement %",
            **create_team_saudi_chart_theme()
        )
        st.plotly_chart(imp_fig, use_container_width=True)

    with tab3:
        st.subheader("Target Time Calculator")

        st.markdown("""
        Calculate the target time needed to achieve specific goals based on World Records.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Event selection - use dynamic world records based on course type
            events = sorted([e for e in world_records.keys()])
            selected_event = st.selectbox("Select Event", events, key="bench_event")

            # Competition level
            competitions = list(COMPETITION_BENCHMARKS.keys())
            selected_comp = st.selectbox("Target Competition", competitions, key="bench_comp")

            # Goal
            goal = st.selectbox("Goal", ["Gold Medal", "Any Medal", "Make Finals"])

        with col2:
            if selected_event and selected_comp:
                wr_time = world_records[selected_event]
                benchmarks = COMPETITION_BENCHMARKS[selected_comp]

                goal_key = {'Gold Medal': 'gold', 'Any Medal': 'medal', 'Make Finals': 'final'}[goal]
                target_pct = benchmarks[goal_key]

                # Calculate target time
                target_time = wr_time / (target_pct / 100)

                def format_time(secs):
                    mins = int(secs // 60)
                    s = secs % 60
                    return f"{mins}:{s:05.2f}" if mins > 0 else f"{s:.2f}"

                st.markdown("### Target Analysis")

                st.metric("World Record", format_time(wr_time))
                st.metric("Required WR%", f"{target_pct}%")
                st.metric(f"Target Time ({goal})", format_time(target_time),
                         delta=f"+{target_time - wr_time:.2f}s from WR")

                # Show all goals for this event/competition
                st.markdown("---")
                st.markdown(f"### All Standards for {selected_event}")
                st.markdown(f"**{selected_comp}** ({benchmarks['level']})")

                for goal_name, key in [('Gold', 'gold'), ('Medal', 'medal'), ('Finals', 'final')]:
                    pct = benchmarks[key]
                    time_needed = wr_time / (pct / 100)
                    st.write(f"**{goal_name}:** {format_time(time_needed)} ({pct}% WR)")

    with tab4:
        st.subheader("Athlete Assessment")

        st.markdown("""
        Assess where an athlete stands relative to benchmarks and what competitions they're ready for.
        """)

        # Athlete selector
        col1, col2 = st.columns([1, 2])

        with col1:
            countries = sorted(df['NAT'].dropna().unique())
            assess_country = st.selectbox("Filter by Country", ["All Countries"] + countries, key="assess_country")

        with col2:
            if assess_country == "All Countries":
                athlete_pool = df
            else:
                athlete_pool = df[df['NAT'] == assess_country]

            athletes = sorted(athlete_pool['FullName'].dropna().unique())
            assess_athlete = st.selectbox("Select Athlete", athletes if athletes else ["No athletes found"], key="assess_athlete")

        if assess_athlete and assess_athlete != "No athletes found":
            # Get athlete's events
            athlete_data = df[df['FullName'].str.contains(assess_athlete, case=False, na=False)]
            athlete_events = sorted(athlete_data['discipline_name'].dropna().unique())

            assess_event = st.selectbox("Select Event", athlete_events, key="assess_event")

            if assess_event:
                # Get athlete's best time in this event
                from enhanced_swimming_scraper import SplitTimeAnalyzer
                time_analyzer = SplitTimeAnalyzer()

                event_data = athlete_data[athlete_data['discipline_name'] == assess_event].copy()
                event_data['time_seconds'] = event_data['Time'].apply(time_analyzer.time_to_seconds)
                event_data = event_data[event_data['time_seconds'] > 0]

                if not event_data.empty:
                    best_time = event_data['time_seconds'].min()
                    best_time_str = event_data.loc[event_data['time_seconds'].idxmin(), 'Time']

                    # Find matching world record (use dynamic world_records based on course type)
                    wr_time = None
                    for wr_event, wr in world_records.items():
                        if assess_event.lower() in wr_event.lower() or wr_event.lower() in assess_event.lower():
                            wr_time = wr
                            break

                    if wr_time:
                        athlete_wr_pct = (wr_time / best_time) * 100

                        st.markdown("---")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Personal Best", best_time_str)
                            st.metric("PB (seconds)", f"{best_time:.2f}s")

                        with col2:
                            st.metric("World Record", f"{wr_time:.2f}s")
                            st.metric("Current WR%", f"{athlete_wr_pct:.1f}%")

                        with col3:
                            st.metric("Gap to WR", f"+{best_time - wr_time:.2f}s")

                        # Competition readiness
                        st.markdown("### Competition Readiness")

                        ready_for = []
                        developing_for = []
                        not_ready = []

                        for comp, benchmarks in COMPETITION_BENCHMARKS.items():
                            if athlete_wr_pct >= benchmarks['gold']:
                                ready_for.append((comp, 'Gold Contender', benchmarks['gold']))
                            elif athlete_wr_pct >= benchmarks['medal']:
                                ready_for.append((comp, 'Medal Contender', benchmarks['medal']))
                            elif athlete_wr_pct >= benchmarks['final']:
                                developing_for.append((comp, 'Finals Potential', benchmarks['final']))
                            else:
                                not_ready.append((comp, 'Development Needed', benchmarks['final']))

                        col_ready, col_dev, col_target = st.columns(3)

                        with col_ready:
                            st.markdown("**Ready to Compete**")
                            if ready_for:
                                for comp, status, threshold in ready_for:
                                    st.success(f"{comp}: {status}")
                            else:
                                st.info("Keep developing!")

                        with col_dev:
                            st.markdown("**Close to Ready**")
                            for comp, status, threshold in developing_for[:4]:
                                gap = threshold - athlete_wr_pct
                                st.warning(f"{comp}: {gap:.1f}% to finals")

                        with col_target:
                            st.markdown("**Next Targets**")
                            for comp, status, threshold in not_ready[:4]:
                                gap = threshold - athlete_wr_pct
                                st.info(f"{comp}: {gap:.1f}% needed")

                        # Visual comparison
                        st.markdown("### Visual Comparison to Standards")

                        comp_names = list(COMPETITION_BENCHMARKS.keys())
                        gold_standards = [COMPETITION_BENCHMARKS[c]['gold'] for c in comp_names]
                        medal_standards = [COMPETITION_BENCHMARKS[c]['medal'] for c in comp_names]
                        final_standards = [COMPETITION_BENCHMARKS[c]['final'] for c in comp_names]

                        assess_fig = go.Figure()

                        # Standards
                        assess_fig.add_trace(go.Bar(
                            name='Gold Standard',
                            x=comp_names,
                            y=gold_standards,
                            marker_color='#FFD700',
                            opacity=0.7
                        ))

                        # Athlete's level (horizontal line)
                        assess_fig.add_hline(
                            y=athlete_wr_pct,
                            line_dash="dash",
                            line_color=TEAM_SAUDI_COLORS['primary_teal'],
                            line_width=3,
                            annotation_text=f"{assess_athlete}: {athlete_wr_pct:.1f}%"
                        )

                        assess_fig.update_layout(
                            title=f"Athlete Performance vs Competition Standards",
                            yaxis_title="% of World Record",
                            yaxis=dict(range=[75, 100]),
                            **create_team_saudi_chart_theme()
                        )
                        st.plotly_chart(assess_fig, use_container_width=True)

                        # Age-based assessment if available
                        if 'AthleteResultAge' in event_data.columns:
                            latest_age = event_data['AthleteResultAge'].dropna()
                            if not latest_age.empty:
                                age = int(latest_age.iloc[-1])
                                gender = 'female' if 'Women' in assess_event else 'male'

                                if age in AGE_PROGRESSION_BENCHMARKS[gender]:
                                    age_bench = AGE_PROGRESSION_BENCHMARKS[gender][age]
                                    st.markdown(f"### Age-Based Assessment (Age {age})")

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Current WR%", f"{athlete_wr_pct:.1f}%")
                                    with col2:
                                        target = age_bench['target_wr_pct']
                                        diff = athlete_wr_pct - target
                                        st.metric("Age Target", f"{target}%",
                                                 delta=f"{diff:+.1f}% {'ahead' if diff > 0 else 'behind'}")
                                    with col3:
                                        elite = age_bench['elite_wr_pct']
                                        diff = athlete_wr_pct - elite
                                        st.metric("Elite Standard", f"{elite}%",
                                                 delta=f"{diff:+.1f}% {'ahead' if diff > 0 else 'behind'}")

                                    st.info(f"**Development Phase:** {age_bench['phase']}")
                    else:
                        st.warning("No matching world record found for this event")
                else:
                    st.warning("No valid times found for this event")


def show_research_insights():
    """Display research-backed insights."""
    st.header("🔬 Research Insights")

    st.markdown("""
    This dashboard is built on peer-reviewed sports science research and industry best practices.
    Below are the key findings that inform our analytics.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Pacing Research", "Development Research", "Performance Benchmarks", "Elite Analytics Tools"])

    with tab1:
        st.subheader("Pacing Strategy Research")

        st.markdown("""
        ### Key Findings from World Championships (2017-2024)

        **Source:** [Frontiers in Sports and Active Living, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11557356/)

        #### 400m Freestyle
        - **Medalist Strategy:** Inverted-J pacing (98.91% of WR)
        - **Worst Strategy:** U-shape for non-medalists
        - Males transition from fast-start-even (heats) to U-shape (finals)

        #### 800m Freestyle
        - **Universal Strategy:** U-shape adopted by all top performers
        - Consistent across genders and rounds

        #### Critical Finding
        > "Swimmers must be in top 3 position in the final 100m to medal.
        > Medalists occupied top 3 positions more than 90% of the time in the final sprint."

        #### Performance Progression
        - Medalists improve **1.0-1.4%** from heats to finals
        - Non-medalists show minimal or negative progression
        """)

    with tab2:
        st.subheader("Talent Development Research")

        st.markdown("""
        ### Junior-to-Senior Transition

        **Source:** [Frontiers in Physiology, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10446966/)

        #### Key Finding
        > "None of the lower-performing juniors transitioned to the high-performing senior group."

        This highlights the critical importance of early development and monitoring.

        #### Critical Ages
        - **Males:** 16-19 years (junior-to-senior transition)
        - **Females:** 15-18 years (junior-to-senior transition)

        #### What Differentiates Elite Juniors
        - Higher maximal swimming velocity
        - Better turn times and lower body power (males)
        - Superior stroke efficiency (females)
        - Advanced anthropometrics

        ---

        ### Peak Performance Ages

        **Source:** [PLOS ONE, 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332306)
        """)

        # Peak age table
        peak_data = {
            'Category': ['Sprint', 'Middle Distance', 'Distance', 'Overall'],
            'Male Peak Age': [24.5, 24.0, 23.5, 24.2],
            'Female Peak Age': [23.0, 22.5, 22.0, 22.5]
        }
        st.table(pd.DataFrame(peak_data))

        st.markdown("""
        #### Time to Elite
        - Average: **~8 years** of competition to reach >900 FINA points
        - Peak window duration: **~2.6 years** within 2% of career best
        """)

    with tab3:
        st.subheader("Elite Performance Benchmarks")

        benchmarks = {
            'Metric': [
                'FINA Points (Elite)',
                'Years to Elite',
                'Peak Window Duration',
                'Lap Variance (CV) - Elite',
                'Heats-to-Finals Improvement',
                'Final 100m Position for Medal'
            ],
            'Benchmark': [
                '> 900 points',
                '~8 years',
                '~2.6 years',
                '< 1.3%',
                '> 1.2%',
                'Top 3'
            ],
            'Source': [
                'FINA/World Aquatics',
                'Career Trajectory Studies',
                'PLOS ONE 2024',
                'Frontiers 2024',
                'World Championships Analysis',
                'Medal Analysis 2017-2024'
            ]
        }

        st.table(pd.DataFrame(benchmarks))

        st.markdown("---")

        st.subheader("World Records Reference (LCM)")

        wr_df = pd.DataFrame([
            {'Event': event, 'World Record': f"{time:.2f}s"}
            for event, time in sorted(WORLD_RECORDS_LCM.items())
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Men's Events**")
            st.dataframe(wr_df[wr_df['Event'].str.startswith('Men')], use_container_width=True)
        with col2:
            st.markdown("**Women's Events**")
            st.dataframe(wr_df[wr_df['Event'].str.startswith('Women')], use_container_width=True)

    with tab4:
        st.subheader("Elite Analytics Tools & Techniques")

        st.markdown("""
        ### What Elite Teams Are Using

        Based on analysis of world-leading swimming programs (British Swimming, USA Swimming, Australian Institute of Sport),
        here are the key analytics tools and techniques used at the elite level:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Professional Analysis Software

            **TritonWear** (AI-Powered)
            - Real-time stroke analytics
            - Turn time optimization
            - Underwater distance tracking
            - Personalized training recommendations

            **Dartfish** (Video + Data)
            - Video overlay comparisons
            - Kinematic analysis
            - Race comparison tools
            - Performance trending

            **Race Analyzer** (Comprehensive)
            - Split time analysis
            - Velocity curves
            - Race simulation
            - Competition benchmarking
            """)

        with col2:
            st.markdown("""
            #### Key Metrics Tracked

            **Stroke Mechanics**
            - Stroke count per lap
            - Distance per stroke (DPS)
            - Stroke rate (cycles/min)
            - Stroke index efficiency

            **Turn Performance**
            - Wall contact time
            - Push-off velocity
            - Underwater distance (15m rule)
            - Breakout time

            **Start Performance**
            - Reaction time (< 0.65s elite)
            - Block time
            - Flight distance
            - Entry angle
            """)

        st.markdown("---")

        st.markdown("""
        ### Machine Learning in Swimming Analytics

        Recent research (2024-2025) has applied ML to swimming performance prediction:

        | Technique | Application | Accuracy |
        |-----------|-------------|----------|
        | Linear Regression | Performance forecasting | R² > 0.7 |
        | Random Forest | Medal prediction | 85%+ accuracy |
        | Neural Networks | Split optimization | MAE < 1% |
        | PCA Analysis | Performance factors | 3-5 key components |

        **Key Predictors Identified:**
        1. Previous best times (strongest predictor)
        2. Year-over-year improvement rate
        3. Heats-to-finals progression
        4. Lap consistency (CV < 1.3%)
        5. Reaction time
        """)

        st.markdown("---")

        st.markdown("""
        ### British Swimming / Intel Partnership

        At the elite level (Olympic preparation), systems track:
        - **1000+ data points per second** from starting blocks
        - Force plate analysis for start optimization
        - AI-powered video analysis for technique refinement
        - Real-time race simulation for tactical planning

        > "The integration of AI and data analytics has become essential for marginal gains
        > at the Olympic level." - British Swimming Performance Director
        """)

        st.markdown("---")

        st.info("""
        **This Dashboard's Capabilities:**

        While we don't have wearable sensor data, this dashboard provides:
        - Split time analysis and pacing strategy classification
        - Performance forecasting using trend analysis
        - Reaction time tracking and benchmarking
        - Consistency metrics (CV analysis)
        - World record benchmarking
        - Competition intelligence and field comparison

        These analytics cover the key metrics that research shows are most predictive of success.
        """)


if __name__ == "__main__":
    main()
