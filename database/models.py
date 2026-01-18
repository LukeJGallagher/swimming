"""
SQLAlchemy Database Models for Swimming Performance Database
Compatible with PostgreSQL and SQLite
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime,
    ForeignKey, Text, Index, UniqueConstraint, Numeric
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON
from datetime import datetime
import os

Base = declarative_base()


class Athlete(Base):
    """Athletes table - normalized swimmer information"""
    __tablename__ = 'athletes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(50), unique=True, index=True)
    full_name = Column(String(200), nullable=False, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    nationality = Column(String(10), index=True)
    nationality_name = Column(String(100))
    biography_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    results = relationship("Result", back_populates="athlete")

    def __repr__(self):
        return f"<Athlete(id={self.id}, name='{self.full_name}', nat='{self.nationality}')>"


class Competition(Base):
    """Competitions table - swimming meets/events"""
    __tablename__ = 'competitions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    competition_id = Column(Integer, unique=True, index=True)
    competition_name = Column(String(300), nullable=False, index=True)
    date_from = Column(Date)
    date_to = Column(Date)
    year = Column(Integer, index=True)
    host_city = Column(String(100))
    host_country = Column(String(100))
    pool_type = Column(String(10))  # LCM, SCM
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    results = relationship("Result", back_populates="competition")

    def __repr__(self):
        return f"<Competition(id={self.id}, name='{self.competition_name[:50]}', year={self.year})>"


class Event(Base):
    """Events table - swimming disciplines"""
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), unique=True, index=True)
    discipline_name = Column(String(100), nullable=False, index=True)
    gender = Column(String(10))  # Men, Women, Mixed
    distance = Column(Integer)  # in meters
    stroke = Column(String(50))  # Freestyle, Backstroke, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    results = relationship("Result", back_populates="event")

    def __repr__(self):
        return f"<Event(id={self.id}, name='{self.discipline_name}')>"


class Result(Base):
    """Results table - main fact table for race results"""
    __tablename__ = 'results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(50), unique=True, index=True)

    # Foreign keys
    athlete_id = Column(Integer, ForeignKey('athletes.id'), index=True)
    competition_id = Column(Integer, ForeignKey('competitions.id'), index=True)
    event_id = Column(Integer, ForeignKey('events.id'), index=True)

    # Race details
    heat_category = Column(String(50))
    heat_rank = Column(Integer)
    final_rank = Column(Integer, index=True)
    lane = Column(Integer)

    # Timing
    time_raw = Column(String(20))
    time_seconds = Column(Numeric(10, 3), index=True)
    reaction_time = Column(Numeric(5, 3))
    time_behind = Column(Numeric(10, 3))

    # Scoring
    fina_points = Column(Integer)
    medal_tag = Column(String(1), index=True)  # G, S, B
    qualified = Column(String(10))
    record_type = Column(String(20))

    # Split data (JSON)
    splits_json = Column(JSON)
    lap_times_json = Column(JSON)

    # Pacing analysis
    pacing_type = Column(String(30), index=True)
    first_half_avg = Column(Numeric(10, 3))
    second_half_avg = Column(Numeric(10, 3))
    split_difference = Column(Numeric(10, 3))
    fastest_lap = Column(Numeric(10, 3))
    slowest_lap = Column(Numeric(10, 3))
    lap_variance = Column(Numeric(10, 5))

    # Metadata
    athlete_age = Column(Integer)
    year = Column(Integer, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    athlete = relationship("Athlete", back_populates="results")
    competition = relationship("Competition", back_populates="results")
    event = relationship("Event", back_populates="results")

    def __repr__(self):
        return f"<Result(id={self.id}, athlete_id={self.athlete_id}, time={self.time_seconds})>"


class WorldRecord(Base):
    """World Records reference table"""
    __tablename__ = 'world_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    discipline_name = Column(String(100), unique=True, nullable=False)
    record_time = Column(Numeric(10, 3), nullable=False)
    record_holder = Column(String(200))
    record_date = Column(Date)
    pool_type = Column(String(10), default='LCM')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<WorldRecord(event='{self.discipline_name}', time={self.record_time})>"


class EliteBenchmark(Base):
    """Elite benchmarks from research"""
    __tablename__ = 'elite_benchmarks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), unique=True, nullable=False)
    metric_value = Column(Numeric(10, 3))
    description = Column(Text)
    source = Column(String(200))

    def __repr__(self):
        return f"<EliteBenchmark(metric='{self.metric_name}', value={self.metric_value})>"


# Database connection utilities
def get_database_url(db_type: str = 'sqlite') -> str:
    """
    Get database URL from environment or default.

    Supports:
    - sqlite: Local SQLite file (default)
    - postgresql: PostgreSQL server
    - azure: Azure SQL Database (via ODBC)

    Environment Variables:
    - AZURE_SQL_CONN: Azure SQL connection string (for azure type)
    - DATABASE_URL: Generic database URL (for sqlite/postgresql)
    """
    # Check for Azure SQL connection first
    azure_conn = os.getenv('AZURE_SQL_CONN')
    if azure_conn:
        # Azure SQL uses pyodbc with ODBC driver
        return f"mssql+pyodbc:///?odbc_connect={azure_conn}"

    if db_type == 'azure':
        azure_conn = os.getenv('AZURE_SQL_CONN')
        if azure_conn:
            return f"mssql+pyodbc:///?odbc_connect={azure_conn}"
        raise ValueError("AZURE_SQL_CONN environment variable not set")
    elif db_type == 'postgresql':
        return os.getenv(
            'DATABASE_URL',
            'postgresql://localhost/swimming_db'
        )
    else:
        return os.getenv(
            'DATABASE_URL',
            'sqlite:///swimming_performance.db'
        )


def create_database(db_url: str = None):
    """Create database engine and tables."""
    if db_url is None:
        db_url = get_database_url()

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)

    return engine


def get_session(engine=None):
    """Get database session."""
    if engine is None:
        engine = create_database()

    Session = sessionmaker(bind=engine)
    return Session()


# World records data for initialization
WORLD_RECORDS_LCM = {
    'Men 50m Freestyle': 20.91,
    'Men 100m Freestyle': 46.40,
    'Men 200m Freestyle': 102.00,
    'Men 400m Freestyle': 220.07,
    'Men 800m Freestyle': 452.12,
    'Men 1500m Freestyle': 871.02,
    'Men 50m Backstroke': 23.71,
    'Men 100m Backstroke': 51.60,
    'Men 200m Backstroke': 111.92,
    'Men 50m Breaststroke': 25.95,
    'Men 100m Breaststroke': 56.88,
    'Men 200m Breaststroke': 125.48,
    'Men 50m Butterfly': 22.27,
    'Men 100m Butterfly': 49.45,
    'Men 200m Butterfly': 110.34,
    'Men 200m Individual Medley': 114.00,
    'Men 400m Individual Medley': 243.84,
    'Women 50m Freestyle': 23.61,
    'Women 100m Freestyle': 51.71,
    'Women 200m Freestyle': 112.98,
    'Women 400m Freestyle': 235.82,
    'Women 800m Freestyle': 493.04,
    'Women 1500m Freestyle': 940.34,
    'Women 50m Backstroke': 26.98,
    'Women 100m Backstroke': 57.33,
    'Women 200m Backstroke': 123.35,
    'Women 50m Breaststroke': 29.16,
    'Women 100m Breaststroke': 64.13,
    'Women 200m Breaststroke': 139.11,
    'Women 50m Butterfly': 24.43,
    'Women 100m Butterfly': 55.18,
    'Women 200m Butterfly': 121.81,
    'Women 200m Individual Medley': 126.12,
    'Women 400m Individual Medley': 266.36,
}

ELITE_BENCHMARKS = {
    'fina_points_elite': (900, 'FINA points threshold for elite level', 'World Aquatics'),
    'years_to_elite': (8, 'Average years of competition to reach elite', 'Career trajectory studies'),
    'peak_window_years': (2.6, 'Years within 2% of career best', 'PLOS ONE 2024'),
    'cv_elite_threshold': (1.3, 'Coefficient of variation for lap times (%)', 'Frontiers 2024'),
    'heats_to_finals_improvement': (1.2, 'Expected % improvement for medalists', 'World Championships'),
    'male_peak_age': (24.2, 'Average peak performance age for males', 'PLOS ONE 2024'),
    'female_peak_age': (22.5, 'Average peak performance age for females', 'PLOS ONE 2024'),
}


def initialize_reference_data(session):
    """Initialize world records and benchmarks tables."""
    # World records
    for event, time in WORLD_RECORDS_LCM.items():
        existing = session.query(WorldRecord).filter_by(discipline_name=event).first()
        if not existing:
            session.add(WorldRecord(discipline_name=event, record_time=time))

    # Elite benchmarks
    for metric, (value, desc, source) in ELITE_BENCHMARKS.items():
        existing = session.query(EliteBenchmark).filter_by(metric_name=metric).first()
        if not existing:
            session.add(EliteBenchmark(
                metric_name=metric,
                metric_value=value,
                description=desc,
                source=source
            ))

    session.commit()


if __name__ == "__main__":
    # Create database and initialize
    print("Creating database...")
    engine = create_database()
    session = get_session(engine)

    print("Initializing reference data...")
    initialize_reference_data(session)

    print("Database setup complete!")
    print(f"Tables created: {list(Base.metadata.tables.keys())}")
