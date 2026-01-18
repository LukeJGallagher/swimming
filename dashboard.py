"""
Swimming Performance Dashboard
Interactive visualization dashboard using Streamlit
"""

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    print("Then run with: streamlit run dashboard.py")
    exit(1)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
from enhanced_swimming_scraper import SplitTimeAnalyzer
from performance_analyst_tools import AthleteProfiler, ProgressionTracker, CompetitionAnalyzer

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
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load available swimming data"""
    data_dir = Path("data")

    if not data_dir.exists():
        return None, []

    # Find all result files
    result_files = list(data_dir.glob("results_*.csv"))

    if not result_files:
        return None, []

    # Load most recent
    latest_file = sorted(result_files)[-1]
    df = pd.read_csv(latest_file)

    return df, result_files


def main():
    # Header
    st.markdown('<div class="main-header">🏊 Swimming Performance Dashboard</div>', unsafe_allow_html=True)

    # Load data
    results_df, available_files = load_data()

    if results_df is None or results_df.empty:
        st.error("No data found. Please run the scraper first:")
        st.code("python enhanced_swimming_scraper.py")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Overview",
        "Athlete Profile",
        "Competition Analysis",
        "Split Time Analysis",
        "Country Performance",
        "Rankings & Records"
    ])

    # Data file selector
    st.sidebar.title("Data Selection")
    selected_file = st.sidebar.selectbox(
        "Select Data File",
        available_files,
        format_func=lambda x: x.name
    )

    if selected_file:
        results_df = pd.read_csv(selected_file)

    st.sidebar.metric("Total Results", len(results_df))
    st.sidebar.metric("Unique Athletes", results_df['FullName'].nunique() if 'FullName' in results_df.columns else 0)
    st.sidebar.metric("Countries", results_df['NAT'].nunique() if 'NAT' in results_df.columns else 0)

    # Page routing
    if page == "Overview":
        show_overview(results_df)
    elif page == "Athlete Profile":
        show_athlete_profile(results_df)
    elif page == "Competition Analysis":
        show_competition_analysis(results_df)
    elif page == "Split Time Analysis":
        show_split_analysis(results_df)
    elif page == "Country Performance":
        show_country_performance(results_df)
    elif page == "Rankings & Records":
        show_rankings_records(results_df)


def show_overview(df):
    """Overview dashboard"""
    st.header("Dashboard Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Competitions", df['competition_name'].nunique() if 'competition_name' in df.columns else 0)

    with col2:
        st.metric("Total Events", df['discipline_name'].nunique() if 'discipline_name' in df.columns else 0)

    with col3:
        medals = df['MedalTag'].notna().sum() if 'MedalTag' in df.columns else 0
        st.metric("Medal Performances", medals)

    with col4:
        records = df['RecordType'].notna().sum() if 'RecordType' in df.columns else 0
        st.metric("Records Set", records)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results by Event Type")
        if 'discipline_name' in df.columns:
            event_counts = df['discipline_name'].value_counts().head(10)
            fig = px.bar(x=event_counts.index, y=event_counts.values,
                        labels={'x': 'Event', 'y': 'Number of Results'},
                        title="Top 10 Events by Result Count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Participation by Country")
        if 'NAT' in df.columns:
            country_counts = df['NAT'].value_counts().head(15)
            fig = px.pie(values=country_counts.values, names=country_counts.index,
                        title="Top 15 Countries by Participation")
            st.plotly_chart(fig, use_container_width=True)

    # Pacing distribution
    if 'pacing_type' in df.columns:
        st.subheader("Pacing Strategy Distribution")
        pacing_data = df['pacing_type'].value_counts()
        fig = px.bar(x=pacing_data.index, y=pacing_data.values,
                    labels={'x': 'Pacing Type', 'y': 'Count'},
                    color=pacing_data.index)
        st.plotly_chart(fig, use_container_width=True)


def show_athlete_profile(df):
    """Athlete profiling page"""
    st.header("Athlete Profile Analysis")

    # Athlete selector
    if 'FullName' in df.columns:
        athletes = sorted(df['FullName'].dropna().unique())
        selected_athlete = st.selectbox("Select Athlete", athletes)

        if selected_athlete:
            profiler = AthleteProfiler(df)
            profile = profiler.create_profile(selected_athlete)

            if 'error' not in profile:
                # Display profile
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Country", profile.get('country', 'N/A'))
                    st.metric("Total Races", profile.get('total_races', 0))

                with col2:
                    medals = profile.get('medals', {})
                    st.metric("Total Medals", medals.get('total', 0))
                    st.write(f"🥇 {medals.get('gold', 0)} 🥈 {medals.get('silver', 0)} 🥉 {medals.get('bronze', 0)}")

                with col3:
                    if 'recent_form' in profile:
                        st.metric("Recent Races (90d)", profile['recent_form']['races'])
                        st.metric("Best Recent Time", f"{profile['recent_form']['best_recent']}s")

                # Best times table
                st.subheader("Personal Bests")
                if 'best_times' in profile:
                    best_times_df = pd.DataFrame(profile['best_times'])
                    st.dataframe(
                        best_times_df[['discipline_name', 'Time', 'Rank', 'competition_name']].head(10),
                        use_container_width=True
                    )

                # Progression chart
                st.subheader("Performance Progression")
                event = st.selectbox("Select Event for Progression",
                                    [bt['discipline_name'] for bt in profile.get('best_times', [])])

                if event:
                    tracker = ProgressionTracker(df)
                    progression = tracker.calculate_progression(selected_athlete, event)

                    if not progression.empty:
                        fig = px.line(progression, x='date_from', y='time_seconds',
                                     title=f"{selected_athlete} - {event} Progression",
                                     labels={'date_from': 'Date', 'time_seconds': 'Time (seconds)'})

                        # Add PB line
                        fig.add_scatter(x=progression['date_from'], y=progression['personal_best'],
                                       mode='lines', name='Personal Best',
                                       line=dict(dash='dash', color='red'))

                        st.plotly_chart(fig, use_container_width=True)

                        # Breakthroughs
                        breakthroughs = tracker.identify_breakthroughs(progression)
                        if breakthroughs:
                            st.subheader("Breakthrough Performances")
                            st.dataframe(pd.DataFrame(breakthroughs), use_container_width=True)

            else:
                st.error(profile['error'])


def show_competition_analysis(df):
    """Competition analysis page"""
    st.header("Competition Analysis")

    if 'competition_name' in df.columns:
        competitions = sorted(df['competition_name'].dropna().unique())
        selected_comp = st.selectbox("Select Competition", competitions)

        if selected_comp:
            analyzer = CompetitionAnalyzer(df)
            summary = analyzer.competition_summary(competition_name=selected_comp)

            if 'error' not in summary:
                # Display summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Athletes", summary['unique_athletes'])
                    st.metric("Countries", summary['unique_countries'])

                with col2:
                    st.metric("Events", summary['events'])
                    st.metric("Total Results", summary['total_results'])

                with col3:
                    st.metric("Location", summary['location'])
                    st.metric("Dates", summary['dates'])

                # Medal table
                if 'medal_table' in summary:
                    st.subheader("Medal Table")
                    medal_df = pd.DataFrame(summary['medal_table'])
                    st.dataframe(medal_df, use_container_width=True)

                # Event breakdown
                st.subheader("Event Results Distribution")
                comp_data = df[df['competition_name'] == selected_comp]
                event_counts = comp_data['discipline_name'].value_counts()

                fig = px.bar(x=event_counts.index, y=event_counts.values,
                            labels={'x': 'Event', 'y': 'Results'},
                            title="Results by Event")
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)


def show_split_analysis(df):
    """Split time analysis page"""
    st.header("Split Time Analysis")

    # Filter for results with splits
    if 'splits_json' in df.columns:
        with_splits = df[df['splits_json'].notna()]

        if with_splits.empty:
            st.warning("No split time data available in current dataset")
            return

        st.metric("Results with Split Times", len(with_splits))

        # Event selector
        events = sorted(with_splits['discipline_name'].dropna().unique())
        selected_event = st.selectbox("Select Event", events)

        if selected_event:
            event_data = with_splits[with_splits['discipline_name'] == selected_event]

            # Top performers
            st.subheader(f"Top Performers - {selected_event}")

            top_n = st.slider("Show Top N", 3, 20, 10)
            top_performers = event_data.nsmallest(top_n, 'Rank') if 'Rank' in event_data.columns else event_data.head(top_n)

            # Display results
            for idx, row in top_performers.iterrows():
                with st.expander(f"#{row.get('Rank', 'N/A')} - {row.get('FullName', 'Unknown')} - {row.get('Time', 'N/A')}"):
                    if row['splits_json']:
                        try:
                            splits = json.loads(row['splits_json'])
                            analyzer = SplitTimeAnalyzer()
                            lap_times = analyzer.calculate_lap_times(splits)

                            if lap_times:
                                lap_df = pd.DataFrame(lap_times)
                                st.dataframe(lap_df[['lap_number', 'distance', 'lap_time', 'cumulative_time']])

                                # Pacing info
                                if 'pacing_type' in row and pd.notna(row['pacing_type']):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Pacing", row['pacing_type'])
                                    with col2:
                                        if 'lap_variance' in row:
                                            st.metric("Lap Variance", f"{row['lap_variance']:.3f}")
                                    with col3:
                                        if 'fastest_lap' in row:
                                            st.metric("Fastest Lap", f"{row['fastest_lap']:.2f}s")
                        except:
                            st.write("Could not parse split data")


def show_country_performance(df):
    """Country performance page"""
    st.header("Country Performance Analysis")

    if 'NAT' in df.columns:
        countries = sorted(df['NAT'].dropna().unique())
        selected_country = st.selectbox("Select Country", countries)

        if selected_country:
            analyzer = CompetitionAnalyzer(df)
            perf = analyzer.country_performance(selected_country)

            if 'error' not in perf:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Entries", perf['total_entries'])
                    st.metric("Unique Athletes", perf['unique_athletes'])

                with col2:
                    st.metric("Events Competed", perf['events_competed'])
                    if 'finals_reached' in perf:
                        st.metric("Finals Reached", perf['finals_reached'])

                with col3:
                    if 'medals' in perf:
                        m = perf['medals']
                        st.metric("Total Medals", m['total'])
                        st.write(f"🥇 {m['gold']} 🥈 {m['silver']} 🥉 {m['bronze']}")

                # Top performances
                if 'top_performances' in perf:
                    st.subheader("Top Performances")
                    top_perf_df = pd.DataFrame(perf['top_performances'])
                    st.dataframe(top_perf_df, use_container_width=True)


def show_rankings_records(df):
    """Rankings and records page"""
    st.header("Rankings & Records")

    # Records
    if 'RecordType' in df.columns:
        records = df[df['RecordType'].notna()].copy()

        if not records.empty:
            st.subheader("Record Performances")

            record_type_filter = st.multiselect(
                "Filter by Record Type",
                records['RecordType'].unique(),
                default=records['RecordType'].unique()
            )

            filtered_records = records[records['RecordType'].isin(record_type_filter)]

            st.dataframe(
                filtered_records[['FullName', 'NAT', 'discipline_name', 'Time',
                                 'RecordType', 'competition_name', 'date_from']],
                use_container_width=True
            )

    # Best times by event
    st.subheader("Best Times by Event")

    if 'discipline_name' in df.columns:
        event = st.selectbox("Select Event",
                            sorted(df['discipline_name'].dropna().unique()),
                            key='rankings_event')

        if event:
            event_data = df[df['discipline_name'] == event].copy()

            # Convert to seconds for proper sorting
            analyzer = SplitTimeAnalyzer()
            event_data['time_seconds'] = event_data['Time'].apply(analyzer.time_to_seconds)

            top_times = event_data.nsmallest(20, 'time_seconds')

            st.dataframe(
                top_times[['Rank', 'FullName', 'NAT', 'Time', 'competition_name', 'date_from']],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
