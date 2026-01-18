"""
Enhanced Swimming Performance Dashboard
Interactive visualization dashboard using Streamlit with advanced features
"""

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    print("Then run with: streamlit run dashboard_enhanced.py")
    exit(1)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import ast
from datetime import datetime
from enhanced_swimming_scraper import SplitTimeAnalyzer

# Page config
st.set_page_config(
    page_title="Swimming Performance Dashboard",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .split-table {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(ttl=3600)
def load_data():
    """Load available swimming data from current directory or data folder"""
    current_dir = Path(".")
    data_dir = Path("data")

    result_files = []

    # Check current directory first
    result_files.extend(list(current_dir.glob("Results_*.csv")))
    result_files.extend(list(current_dir.glob("enriched_Results_*.csv")))

    # Check data directory if exists
    if data_dir.exists():
        result_files.extend(list(data_dir.glob("results_*.csv")))
        result_files.extend(list(data_dir.glob("enriched_*.csv")))

    # Check for combined files
    combined_files = list(current_dir.glob("All_Results*.csv")) + list(current_dir.glob("all_results*.csv"))
    if data_dir.exists():
        combined_files.extend(list(data_dir.glob("all_results*.csv")))

    # Remove duplicates and sort
    all_files = list(set(result_files + combined_files))
    all_files = sorted(all_files, key=lambda x: x.name, reverse=True)

    if not all_files:
        return None, []

    # Prefer enriched files
    enriched = [f for f in all_files if 'enriched' in f.name.lower()]
    if enriched:
        default_file = enriched[0]
    else:
        default_file = all_files[0]

    df = pd.read_csv(default_file)

    return df, all_files


def time_to_seconds(time_str):
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


def main():
    st.markdown('<div class="main-header">🏊 Swimming Performance Dashboard</div>', unsafe_allow_html=True)

    # Load data
    results_df, available_files = load_data()

    if results_df is None or results_df.empty:
        st.error("No data found. Please run the scraper first or place CSV files in current directory.")
        st.code("python enhanced_swimming_scraper.py")
        return

    # Sidebar navigation
    st.sidebar.title("🏊 Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Overview",
        "👤 Athlete Profile",
        "🏆 Competition Analysis",
        "⏱️ Split Time Analysis",
        "🔄 Head-to-Head",
        "🎬 Race Replay",
        "🌍 Country Performance",
        "📈 Rankings & Records"
    ])

    # Data file selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Selection")
    selected_file = st.sidebar.selectbox(
        "Select Data File",
        available_files,
        format_func=lambda x: x.name
    )

    if selected_file:
        results_df = pd.read_csv(selected_file)

    # Normalize column names (handle both old and new formats)
    col_mapping = {
        'discipline_name': 'DisciplineName',
        'competition_name': 'competition_name',
        'Heat Category': 'heat_category'
    }
    for old, new in col_mapping.items():
        if old in results_df.columns and new not in results_df.columns:
            results_df[new] = results_df[old]

    # Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Stats")
    st.sidebar.metric("Total Results", f"{len(results_df):,}")
    st.sidebar.metric("Athletes", results_df['FullName'].nunique() if 'FullName' in results_df.columns else 0)
    st.sidebar.metric("Countries", results_df['NAT'].nunique() if 'NAT' in results_df.columns else 0)

    # Check for enriched columns
    has_enriched = 'pacing_type' in results_df.columns or 'splits_json' in results_df.columns
    if has_enriched:
        st.sidebar.success("✅ Enriched data loaded")
    else:
        st.sidebar.warning("⚠️ Run process_splits.py for full analysis")

    # Route to pages
    if page == "📊 Overview":
        show_overview(results_df)
    elif page == "👤 Athlete Profile":
        show_athlete_profile(results_df)
    elif page == "🏆 Competition Analysis":
        show_competition_analysis(results_df)
    elif page == "⏱️ Split Time Analysis":
        show_split_analysis(results_df)
    elif page == "🔄 Head-to-Head":
        show_head_to_head(results_df)
    elif page == "🎬 Race Replay":
        show_race_replay(results_df)
    elif page == "🌍 Country Performance":
        show_country_performance(results_df)
    elif page == "📈 Rankings & Records":
        show_rankings_records(results_df)


def show_overview(df):
    """Overview dashboard with key metrics and charts"""
    st.header("📊 Dashboard Overview")

    # Get column names
    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'
    comp_col = 'competition_name' if 'competition_name' in df.columns else None

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if comp_col and comp_col in df.columns:
            st.metric("Competitions", df[comp_col].nunique())
        else:
            st.metric("Competitions", "N/A")

    with col2:
        if disc_col in df.columns:
            st.metric("Events", df[disc_col].nunique())

    with col3:
        medals = df['MedalTag'].notna().sum() if 'MedalTag' in df.columns else 0
        st.metric("Medal Performances", medals)

    with col4:
        records = df['RecordType'].notna().sum() if 'RecordType' in df.columns else 0
        st.metric("Records Set", records)

    with col5:
        if 'pacing_type' in df.columns:
            with_splits = df['pacing_type'].notna().sum()
            st.metric("With Split Analysis", with_splits)

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Events by Participation")
        if disc_col in df.columns:
            event_counts = df[disc_col].value_counts().head(12)
            fig = px.bar(
                x=event_counts.values,
                y=event_counts.index,
                orientation='h',
                labels={'x': 'Number of Results', 'y': 'Event'},
                color=event_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Countries")
        if 'NAT' in df.columns:
            country_counts = df['NAT'].value_counts().head(15)
            fig = px.pie(
                values=country_counts.values,
                names=country_counts.index,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='label+percent')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Pacing distribution
    if 'pacing_type' in df.columns:
        st.markdown("---")
        st.subheader("Pacing Strategy Distribution")

        pacing_data = df['pacing_type'].value_counts()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Color mapping for pacing types
            colors = {'Positive Split': '#ef5350', 'Even': '#66bb6a', 'Negative Split': '#42a5f5'}
            fig = px.bar(
                x=pacing_data.index,
                y=pacing_data.values,
                labels={'x': 'Pacing Type', 'y': 'Count'},
                color=pacing_data.index,
                color_discrete_map=colors
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def show_athlete_profile(df):
    """Athlete profiling with progression charts"""
    st.header("👤 Athlete Profile Analysis")

    if 'FullName' not in df.columns:
        st.error("FullName column not found in data")
        return

    # Athlete selector with search
    athletes = sorted(df['FullName'].dropna().unique())

    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("🔍 Search athlete", "")
        if search:
            filtered_athletes = [a for a in athletes if search.lower() in a.lower()]
        else:
            filtered_athletes = athletes[:100]  # Limit initial display

        selected_athlete = st.selectbox("Select Athlete", filtered_athletes)

    if not selected_athlete:
        return

    # Get athlete data
    athlete_df = df[df['FullName'] == selected_athlete].copy()

    # Profile header
    st.markdown(f"### {selected_athlete}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        country = athlete_df['NAT'].iloc[0] if 'NAT' in athlete_df.columns else 'N/A'
        st.metric("🌍 Country", country)

    with col2:
        st.metric("🏊 Total Races", len(athlete_df))

    with col3:
        if 'MedalTag' in athlete_df.columns:
            medals = athlete_df['MedalTag'].notna().sum()
            gold = (athlete_df['MedalTag'] == 'G').sum()
            silver = (athlete_df['MedalTag'] == 'S').sum()
            bronze = (athlete_df['MedalTag'] == 'B').sum()
            st.metric("🏅 Medals", f"🥇{gold} 🥈{silver} 🥉{bronze}")

    with col4:
        if 'pacing_type' in athlete_df.columns:
            pacing_pref = athlete_df['pacing_type'].mode()
            if len(pacing_pref) > 0:
                st.metric("⏱️ Pacing Style", pacing_pref.iloc[0])

    st.markdown("---")

    # Events competed
    disc_col = 'DisciplineName' if 'DisciplineName' in athlete_df.columns else 'discipline_name'

    if disc_col in athlete_df.columns:
        events = sorted(athlete_df[disc_col].dropna().unique())
        selected_event = st.selectbox("Select Event for Analysis", events)

        if selected_event:
            event_df = athlete_df[athlete_df[disc_col] == selected_event].copy()

            # Convert times to seconds
            event_df['time_seconds'] = event_df['Time'].apply(time_to_seconds)
            event_df = event_df.dropna(subset=['time_seconds'])

            if not event_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Performance Progression")

                    # Sort by date
                    if 'date_from' in event_df.columns:
                        event_df['date'] = pd.to_datetime(event_df['date_from'], errors='coerce')
                        event_df = event_df.sort_values('date')

                    # Personal best line
                    event_df['personal_best'] = event_df['time_seconds'].cummin()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=event_df['date'] if 'date' in event_df.columns else range(len(event_df)),
                        y=event_df['time_seconds'],
                        mode='markers+lines',
                        name='Race Time',
                        marker=dict(size=10, color='#1f77b4')
                    ))
                    fig.add_trace(go.Scatter(
                        x=event_df['date'] if 'date' in event_df.columns else range(len(event_df)),
                        y=event_df['personal_best'],
                        mode='lines',
                        name='Personal Best',
                        line=dict(dash='dash', color='#2ca02c')
                    ))
                    fig.update_layout(
                        yaxis_title="Time (seconds)",
                        xaxis_title="Date",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Performance Stats")
                    best_time = event_df['time_seconds'].min()
                    avg_time = event_df['time_seconds'].mean()
                    latest = event_df['time_seconds'].iloc[-1] if len(event_df) > 0 else None

                    analyzer = SplitTimeAnalyzer()
                    st.metric("Best Time", analyzer.seconds_to_time(best_time))
                    st.metric("Average Time", analyzer.seconds_to_time(avg_time))
                    if latest:
                        st.metric("Latest Time", analyzer.seconds_to_time(latest))

                    # Pacing distribution for this event
                    if 'pacing_type' in event_df.columns:
                        st.markdown("**Pacing Distribution:**")
                        pacing = event_df['pacing_type'].value_counts()
                        for pt, count in pacing.items():
                            if pd.notna(pt):
                                st.write(f"  {pt}: {count}")


def show_competition_analysis(df):
    """Competition analysis page"""
    st.header("🏆 Competition Analysis")

    comp_col = 'competition_name' if 'competition_name' in df.columns else None

    if not comp_col or comp_col not in df.columns:
        st.warning("Competition data not available")
        return

    competitions = sorted(df[comp_col].dropna().unique())
    selected_comp = st.selectbox("Select Competition", competitions)

    if selected_comp:
        comp_df = df[df[comp_col] == selected_comp]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Athletes", comp_df['FullName'].nunique())
        with col2:
            st.metric("Countries", comp_df['NAT'].nunique())
        with col3:
            disc_col = 'DisciplineName' if 'DisciplineName' in comp_df.columns else 'discipline_name'
            st.metric("Events", comp_df[disc_col].nunique() if disc_col in comp_df.columns else 0)
        with col4:
            records = comp_df['RecordType'].notna().sum() if 'RecordType' in comp_df.columns else 0
            st.metric("Records", records)

        st.markdown("---")

        # Medal table
        if 'MedalTag' in comp_df.columns:
            st.subheader("Medal Table")

            medal_df = comp_df[comp_df['MedalTag'].notna()].copy()
            if not medal_df.empty:
                medal_table = medal_df.groupby('NAT')['MedalTag'].value_counts().unstack(fill_value=0)

                for col in ['G', 'S', 'B']:
                    if col not in medal_table.columns:
                        medal_table[col] = 0

                medal_table = medal_table[['G', 'S', 'B']]
                medal_table.columns = ['🥇 Gold', '🥈 Silver', '🥉 Bronze']
                medal_table['Total'] = medal_table.sum(axis=1)
                medal_table = medal_table.sort_values(['🥇 Gold', '🥈 Silver', '🥉 Bronze'], ascending=False)

                st.dataframe(medal_table.head(20), use_container_width=True)


def show_split_analysis(df):
    """Enhanced split time analysis"""
    st.header("⏱️ Split Time Analysis")

    # Check for splits data
    splits_col = 'splits_json' if 'splits_json' in df.columns else 'Splits'

    if splits_col not in df.columns:
        st.warning("No split data found. Run process_splits.py first.")
        return

    # Filter for results with splits
    if splits_col == 'splits_json':
        with_splits = df[df[splits_col].notna()].copy()
    else:
        with_splits = df[df[splits_col].apply(lambda x: x != '[]' and pd.notna(x))].copy()

    if with_splits.empty:
        st.warning("No split time data available")
        return

    st.metric("Results with Split Times", len(with_splits))

    # Event selector
    disc_col = 'DisciplineName' if 'DisciplineName' in with_splits.columns else 'discipline_name'
    events = sorted(with_splits[disc_col].dropna().unique())

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_event = st.selectbox("Select Event", events)
    with col2:
        top_n = st.slider("Show Top N", 5, 30, 10)

    if selected_event:
        event_df = with_splits[with_splits[disc_col] == selected_event].copy()

        # Sort by time (need to parse)
        event_df['time_seconds'] = event_df['Time'].apply(time_to_seconds)
        event_df = event_df.dropna(subset=['time_seconds'])
        event_df = event_df.nsmallest(top_n, 'time_seconds')

        st.subheader(f"Top {top_n} Performers - {selected_event}")

        analyzer = SplitTimeAnalyzer()

        # Build comparison data for chart
        all_splits_data = []

        for idx, row in event_df.iterrows():
            athlete = row.get('FullName', 'Unknown')
            time = row.get('Time', 'N/A')
            rank = row.get('Rank', 'N/A')

            # Parse splits
            if splits_col == 'splits_json' and row[splits_col]:
                try:
                    splits = json.loads(row[splits_col])
                except:
                    splits = []
            else:
                splits = parse_splits_safe(row[splits_col])

            lap_times = analyzer.calculate_lap_times(splits)

            for lt in lap_times:
                all_splits_data.append({
                    'Athlete': athlete,
                    'Distance': lt['distance'],
                    'Lap Time': lt['lap_time_seconds'],
                    'Cumulative': lt['cumulative_seconds']
                })

            with st.expander(f"#{rank} - {athlete} - {time}"):
                if lap_times:
                    lap_df = pd.DataFrame(lap_times)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.dataframe(
                            lap_df[['lap_number', 'distance', 'lap_time', 'cumulative_time']],
                            use_container_width=True
                        )

                    with col2:
                        if 'pacing_type' in row and pd.notna(row['pacing_type']):
                            st.metric("Pacing", row['pacing_type'])
                        if 'lap_variance' in row and pd.notna(row['lap_variance']):
                            st.metric("Consistency", f"{row['lap_variance']:.3f}")
                        if 'fastest_lap' in row and pd.notna(row['fastest_lap']):
                            st.metric("Fastest Lap", f"{row['fastest_lap']:.2f}s")

        # Comparative split chart
        if all_splits_data:
            st.subheader("Split Comparison")

            splits_df = pd.DataFrame(all_splits_data)

            fig = px.line(
                splits_df,
                x='Distance',
                y='Cumulative',
                color='Athlete',
                markers=True,
                title="Race Progression Comparison"
            )
            fig.update_layout(
                xaxis_title="Distance (m)",
                yaxis_title="Time (seconds)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)


def show_head_to_head(df):
    """Head-to-head athlete comparison"""
    st.header("🔄 Head-to-Head Comparison")

    if 'FullName' not in df.columns:
        st.error("Athlete data not available")
        return

    athletes = sorted(df['FullName'].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        search1 = st.text_input("Search Athlete 1", "", key="search1")
        filtered1 = [a for a in athletes if search1.lower() in a.lower()] if search1 else athletes[:50]
        athlete1 = st.selectbox("Select Athlete 1", filtered1, key="athlete1")

    with col2:
        search2 = st.text_input("Search Athlete 2", "", key="search2")
        filtered2 = [a for a in athletes if search2.lower() in a.lower()] if search2 else athletes[:50]
        athlete2 = st.selectbox("Select Athlete 2", filtered2, key="athlete2")

    if athlete1 and athlete2 and athlete1 != athlete2:
        df1 = df[df['FullName'] == athlete1]
        df2 = df[df['FullName'] == athlete2]

        # Find common events
        disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'
        events1 = set(df1[disc_col].dropna().unique())
        events2 = set(df2[disc_col].dropna().unique())
        common_events = sorted(events1.intersection(events2))

        if common_events:
            selected_event = st.selectbox("Select Common Event", common_events)

            if selected_event:
                e1 = df1[df1[disc_col] == selected_event].copy()
                e2 = df2[df2[disc_col] == selected_event].copy()

                e1['time_seconds'] = e1['Time'].apply(time_to_seconds)
                e2['time_seconds'] = e2['Time'].apply(time_to_seconds)

                col1, col2 = st.columns(2)

                analyzer = SplitTimeAnalyzer()

                with col1:
                    st.subheader(athlete1)
                    best1 = e1['time_seconds'].min()
                    st.metric("Best Time", analyzer.seconds_to_time(best1) if pd.notna(best1) else "N/A")
                    st.metric("Races", len(e1))
                    if 'pacing_type' in e1.columns:
                        mode = e1['pacing_type'].mode()
                        if len(mode) > 0:
                            st.metric("Preferred Pacing", mode.iloc[0])

                with col2:
                    st.subheader(athlete2)
                    best2 = e2['time_seconds'].min()
                    st.metric("Best Time", analyzer.seconds_to_time(best2) if pd.notna(best2) else "N/A")
                    st.metric("Races", len(e2))
                    if 'pacing_type' in e2.columns:
                        mode = e2['pacing_type'].mode()
                        if len(mode) > 0:
                            st.metric("Preferred Pacing", mode.iloc[0])

                # Head to head result
                if pd.notna(best1) and pd.notna(best2):
                    diff = best1 - best2
                    if diff < 0:
                        st.success(f"✅ {athlete1} is faster by {abs(diff):.2f} seconds")
                    elif diff > 0:
                        st.success(f"✅ {athlete2} is faster by {abs(diff):.2f} seconds")
                    else:
                        st.info("It's a tie!")

                # Progression comparison
                st.subheader("Performance Comparison")

                fig = go.Figure()

                if 'date_from' in e1.columns:
                    e1['date'] = pd.to_datetime(e1['date_from'], errors='coerce')
                    e1 = e1.sort_values('date')
                    fig.add_trace(go.Scatter(
                        x=e1['date'], y=e1['time_seconds'],
                        mode='markers+lines', name=athlete1
                    ))

                if 'date_from' in e2.columns:
                    e2['date'] = pd.to_datetime(e2['date_from'], errors='coerce')
                    e2 = e2.sort_values('date')
                    fig.add_trace(go.Scatter(
                        x=e2['date'], y=e2['time_seconds'],
                        mode='markers+lines', name=athlete2
                    ))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Time (seconds)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No common events found between these athletes")


def show_race_replay(df):
    """Animated race replay visualization"""
    st.header("🎬 Race Replay Visualization")

    # Need splits data
    splits_col = 'splits_json' if 'splits_json' in df.columns else 'Splits'

    if splits_col not in df.columns:
        st.warning("Split data required for race replay. Run process_splits.py first.")
        return

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'
    comp_col = 'competition_name' if 'competition_name' in df.columns else None

    col1, col2 = st.columns(2)

    with col1:
        events = sorted(df[disc_col].dropna().unique())
        selected_event = st.selectbox("Select Event", events, key="replay_event")

    with col2:
        if comp_col and comp_col in df.columns:
            event_df = df[df[disc_col] == selected_event]
            competitions = sorted(event_df[comp_col].dropna().unique())
            selected_comp = st.selectbox("Select Competition", competitions, key="replay_comp")
        else:
            selected_comp = None

    if selected_event:
        # Filter data
        race_df = df[df[disc_col] == selected_event].copy()
        if selected_comp and comp_col:
            race_df = race_df[race_df[comp_col] == selected_comp]

        # Filter for finals only if possible
        if 'heat_category' in race_df.columns:
            finals = race_df[race_df['heat_category'].str.contains('Final', case=False, na=False)]
            if not finals.empty:
                race_df = finals
        elif 'Heat Category' in race_df.columns:
            finals = race_df[race_df['Heat Category'].str.contains('Final', case=False, na=False)]
            if not finals.empty:
                race_df = finals

        # Get results with splits
        if splits_col == 'splits_json':
            race_df = race_df[race_df[splits_col].notna()]
        else:
            race_df = race_df[race_df[splits_col].apply(lambda x: x != '[]' and pd.notna(x))]

        # Convert times and sort
        race_df['time_seconds'] = race_df['Time'].apply(time_to_seconds)
        race_df = race_df.dropna(subset=['time_seconds'])
        race_df = race_df.nsmallest(8, 'time_seconds')  # Top 8 for final

        if race_df.empty:
            st.warning("No race data with splits found")
            return

        st.subheader(f"Race Replay: {selected_event}")

        analyzer = SplitTimeAnalyzer()

        # Build race data
        race_data = []
        max_distance = 0

        for _, row in race_df.iterrows():
            athlete = row.get('FullName', 'Unknown')

            if splits_col == 'splits_json' and row[splits_col]:
                try:
                    splits = json.loads(row[splits_col])
                except:
                    splits = []
            else:
                splits = parse_splits_safe(row[splits_col])

            lap_times = analyzer.calculate_lap_times(splits)

            for lt in lap_times:
                race_data.append({
                    'Athlete': athlete,
                    'Distance': lt['distance'],
                    'Time': lt['cumulative_seconds']
                })
                max_distance = max(max_distance, lt['distance'])

        if not race_data:
            st.warning("Could not parse split data")
            return

        race_df_plot = pd.DataFrame(race_data)

        # Create animated chart
        fig = px.line(
            race_df_plot,
            x='Time',
            y='Distance',
            color='Athlete',
            markers=True,
            title="Race Progression (Time vs Distance)",
            labels={'Time': 'Time (seconds)', 'Distance': 'Distance (m)'}
        )

        fig.update_layout(
            height=500,
            yaxis=dict(range=[0, max_distance + 10]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show final results table
        st.subheader("Final Results")
        results_table = race_df[['Rank', 'FullName', 'NAT', 'Time', 'pacing_type']].copy() if 'pacing_type' in race_df.columns else race_df[['Rank', 'FullName', 'NAT', 'Time']].copy()
        st.dataframe(results_table, use_container_width=True)


def show_country_performance(df):
    """Country performance analysis"""
    st.header("🌍 Country Performance Analysis")

    if 'NAT' not in df.columns:
        st.error("Country data not available")
        return

    countries = sorted(df['NAT'].dropna().unique())
    selected_country = st.selectbox("Select Country", countries)

    if selected_country:
        country_df = df[df['NAT'] == selected_country]

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Entries", len(country_df))
        with col2:
            st.metric("Athletes", country_df['FullName'].nunique())
        with col3:
            disc_col = 'DisciplineName' if 'DisciplineName' in country_df.columns else 'discipline_name'
            st.metric("Events", country_df[disc_col].nunique() if disc_col in country_df.columns else 0)
        with col4:
            if 'MedalTag' in country_df.columns:
                gold = (country_df['MedalTag'] == 'G').sum()
                silver = (country_df['MedalTag'] == 'S').sum()
                bronze = (country_df['MedalTag'] == 'B').sum()
                st.metric("Medals", f"🥇{gold} 🥈{silver} 🥉{bronze}")

        st.markdown("---")

        # Top athletes
        st.subheader("Top Athletes")
        athlete_results = country_df.groupby('FullName').agg({
            'Time': 'count',
            'MedalTag': lambda x: x.notna().sum() if 'MedalTag' in country_df.columns else 0
        }).rename(columns={'Time': 'Races', 'MedalTag': 'Medals'})
        athlete_results = athlete_results.sort_values('Medals', ascending=False).head(10)
        st.dataframe(athlete_results, use_container_width=True)


def show_rankings_records(df):
    """Rankings and records page"""
    st.header("📈 Rankings & Records")

    # Records section
    if 'RecordType' in df.columns:
        records = df[df['RecordType'].notna()].copy()

        if not records.empty:
            st.subheader("World Records & National Records")

            record_types = records['RecordType'].unique()
            selected_types = st.multiselect("Filter by Record Type", record_types, default=list(record_types))

            filtered = records[records['RecordType'].isin(selected_types)]

            cols = ['FullName', 'NAT', 'DisciplineName' if 'DisciplineName' in filtered.columns else 'discipline_name',
                    'Time', 'RecordType']
            if 'competition_name' in filtered.columns:
                cols.append('competition_name')

            display_cols = [c for c in cols if c in filtered.columns]
            st.dataframe(filtered[display_cols].head(50), use_container_width=True)

    st.markdown("---")

    # Best times by event
    st.subheader("Best Times by Event")

    disc_col = 'DisciplineName' if 'DisciplineName' in df.columns else 'discipline_name'

    if disc_col in df.columns:
        events = sorted(df[disc_col].dropna().unique())
        selected_event = st.selectbox("Select Event", events, key="rankings_event")

        if selected_event:
            event_df = df[df[disc_col] == selected_event].copy()
            event_df['time_seconds'] = event_df['Time'].apply(time_to_seconds)
            event_df = event_df.dropna(subset=['time_seconds'])

            # Remove duplicates (keep best time per athlete)
            best_times = event_df.loc[event_df.groupby('FullName')['time_seconds'].idxmin()]
            best_times = best_times.nsmallest(20, 'time_seconds')

            display_cols = ['FullName', 'NAT', 'Time']
            if 'competition_name' in best_times.columns:
                display_cols.append('competition_name')
            if 'date_from' in best_times.columns:
                display_cols.append('date_from')

            st.dataframe(best_times[display_cols].reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
