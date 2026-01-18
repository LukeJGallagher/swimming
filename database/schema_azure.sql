-- Swimming Performance Database Schema for Azure SQL
-- Team Saudi Performance Analysis
-- Converted from PostgreSQL to T-SQL syntax

-- ============================================================
-- CORE TABLES
-- ============================================================

-- Athletes table (normalized from results)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'athletes')
BEGIN
    CREATE TABLE athletes (
        id INT IDENTITY(1,1) PRIMARY KEY,
        person_id VARCHAR(50) UNIQUE,
        full_name NVARCHAR(200) NOT NULL,
        first_name NVARCHAR(100),
        last_name NVARCHAR(100),
        nationality VARCHAR(10),
        nationality_name NVARCHAR(100),
        biography_id VARCHAR(50),
        created_at DATETIME2 DEFAULT GETDATE(),
        updated_at DATETIME2 DEFAULT GETDATE()
    );

    CREATE INDEX idx_athletes_name ON athletes(full_name);
    CREATE INDEX idx_athletes_nat ON athletes(nationality);
END;
GO

-- Competitions table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'competitions')
BEGIN
    CREATE TABLE competitions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        competition_id INT UNIQUE,
        competition_name NVARCHAR(300) NOT NULL,
        date_from DATE,
        date_to DATE,
        year INT,
        host_city NVARCHAR(100),
        host_country NVARCHAR(100),
        pool_type VARCHAR(10), -- LCM, SCM
        created_at DATETIME2 DEFAULT GETDATE()
    );

    CREATE INDEX idx_competitions_year ON competitions(year);
    CREATE INDEX idx_competitions_name ON competitions(competition_name);
END;
GO

-- Events/Disciplines table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'events')
BEGIN
    CREATE TABLE events (
        id INT IDENTITY(1,1) PRIMARY KEY,
        event_id VARCHAR(50) UNIQUE,
        discipline_name NVARCHAR(100) NOT NULL,
        gender VARCHAR(10), -- Men, Women, Mixed
        distance INT, -- in meters
        stroke NVARCHAR(50), -- Freestyle, Backstroke, Breaststroke, Butterfly, Individual Medley
        created_at DATETIME2 DEFAULT GETDATE()
    );

    CREATE INDEX idx_events_discipline ON events(discipline_name);
    CREATE INDEX idx_events_gender ON events(gender);
END;
GO

-- ============================================================
-- RESULTS TABLE (Main fact table)
-- ============================================================

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'results')
BEGIN
    CREATE TABLE results (
        id INT IDENTITY(1,1) PRIMARY KEY,
        result_id VARCHAR(50) UNIQUE,

        -- Foreign keys
        athlete_id INT REFERENCES athletes(id),
        competition_id INT REFERENCES competitions(id),
        event_id INT REFERENCES events(id),

        -- Race details
        heat_category VARCHAR(50), -- Finals, Heats, Semi-Finals, etc.
        heat_rank INT,
        final_rank INT,
        lane INT,

        -- Timing
        time_raw VARCHAR(20), -- Original time string
        time_seconds DECIMAL(10,3), -- Converted to seconds
        reaction_time DECIMAL(5,3),
        time_behind DECIMAL(10,3),

        -- Scoring
        fina_points INT,
        medal_tag CHAR(1), -- G, S, B, NULL
        qualified VARCHAR(10),
        record_type VARCHAR(20), -- WR, OR, CR, NR, etc.

        -- Split data (JSON)
        splits_json NVARCHAR(MAX),
        lap_times_json NVARCHAR(MAX),

        -- Pacing analysis
        pacing_type VARCHAR(30), -- Even, Positive Split, Negative Split, U-shape, Inverted-J
        first_half_avg DECIMAL(10,3),
        second_half_avg DECIMAL(10,3),
        split_difference DECIMAL(10,3),
        fastest_lap DECIMAL(10,3),
        slowest_lap DECIMAL(10,3),
        lap_variance DECIMAL(10,5),

        -- Metadata
        athlete_age INT,
        year INT,
        created_at DATETIME2 DEFAULT GETDATE()
    );

    CREATE INDEX idx_results_athlete ON results(athlete_id);
    CREATE INDEX idx_results_competition ON results(competition_id);
    CREATE INDEX idx_results_event ON results(event_id);
    CREATE INDEX idx_results_year ON results(year);
    CREATE INDEX idx_results_time ON results(time_seconds);
    CREATE INDEX idx_results_rank ON results(final_rank);
    CREATE INDEX idx_results_medal ON results(medal_tag);
    CREATE INDEX idx_results_pacing ON results(pacing_type);
END;
GO

-- ============================================================
-- WORLD RECORDS REFERENCE TABLE
-- ============================================================

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'world_records')
BEGIN
    CREATE TABLE world_records (
        id INT IDENTITY(1,1) PRIMARY KEY,
        discipline_name NVARCHAR(100) UNIQUE NOT NULL,
        record_time DECIMAL(10,3) NOT NULL,
        record_holder NVARCHAR(200),
        record_date DATE,
        pool_type VARCHAR(10) DEFAULT 'LCM',
        updated_at DATETIME2 DEFAULT GETDATE()
    );
END;
GO

-- Insert current world records (LCM) - using MERGE for upsert
MERGE INTO world_records AS target
USING (VALUES
    ('Men 50m Freestyle', 20.91),
    ('Men 100m Freestyle', 46.40),
    ('Men 200m Freestyle', 102.00),
    ('Men 400m Freestyle', 220.07),
    ('Men 800m Freestyle', 452.12),
    ('Men 1500m Freestyle', 871.02),
    ('Men 50m Backstroke', 23.71),
    ('Men 100m Backstroke', 51.60),
    ('Men 200m Backstroke', 111.92),
    ('Men 50m Breaststroke', 25.95),
    ('Men 100m Breaststroke', 56.88),
    ('Men 200m Breaststroke', 125.48),
    ('Men 50m Butterfly', 22.27),
    ('Men 100m Butterfly', 49.45),
    ('Men 200m Butterfly', 110.34),
    ('Men 200m Individual Medley', 114.00),
    ('Men 400m Individual Medley', 243.84),
    ('Women 50m Freestyle', 23.61),
    ('Women 100m Freestyle', 51.71),
    ('Women 200m Freestyle', 112.98),
    ('Women 400m Freestyle', 235.82),
    ('Women 800m Freestyle', 493.04),
    ('Women 1500m Freestyle', 940.34),
    ('Women 50m Backstroke', 26.98),
    ('Women 100m Backstroke', 57.33),
    ('Women 200m Backstroke', 123.35),
    ('Women 50m Breaststroke', 29.16),
    ('Women 100m Breaststroke', 64.13),
    ('Women 200m Breaststroke', 139.11),
    ('Women 50m Butterfly', 24.43),
    ('Women 100m Butterfly', 55.18),
    ('Women 200m Butterfly', 121.81),
    ('Women 200m Individual Medley', 126.12),
    ('Women 400m Individual Medley', 266.36)
) AS source (discipline_name, record_time)
ON target.discipline_name = source.discipline_name
WHEN MATCHED THEN UPDATE SET record_time = source.record_time
WHEN NOT MATCHED THEN INSERT (discipline_name, record_time) VALUES (source.discipline_name, source.record_time);
GO

-- ============================================================
-- ELITE BENCHMARKS TABLE
-- ============================================================

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'elite_benchmarks')
BEGIN
    CREATE TABLE elite_benchmarks (
        id INT IDENTITY(1,1) PRIMARY KEY,
        metric_name VARCHAR(100) UNIQUE NOT NULL,
        metric_value DECIMAL(10,3),
        description NVARCHAR(MAX),
        source NVARCHAR(200)
    );
END;
GO

MERGE INTO elite_benchmarks AS target
USING (VALUES
    ('fina_points_elite', 900, 'FINA points threshold for elite level', 'World Aquatics'),
    ('years_to_elite', 8, 'Average years of competition to reach elite', 'Career trajectory studies'),
    ('peak_window_years', 2.6, 'Years within 2% of career best', 'PLOS ONE 2024'),
    ('cv_elite_threshold', 1.3, 'Coefficient of variation for lap times (%)', 'Frontiers 2024'),
    ('heats_to_finals_improvement', 1.2, 'Expected % improvement for medalists', 'World Championships analysis'),
    ('male_peak_age', 24.2, 'Average peak performance age for males', 'PLOS ONE 2024'),
    ('female_peak_age', 22.5, 'Average peak performance age for females', 'PLOS ONE 2024')
) AS source (metric_name, metric_value, description, source_ref)
ON target.metric_name = source.metric_name
WHEN NOT MATCHED THEN INSERT (metric_name, metric_value, description, source)
VALUES (source.metric_name, source.metric_value, source.description, source.source_ref);
GO

-- ============================================================
-- FLAT RESULTS TABLE (Simplified - matches CSV structure)
-- For easier data import and querying
-- ============================================================

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'results_flat')
BEGIN
    CREATE TABLE results_flat (
        id INT IDENTITY(1,1) PRIMARY KEY,

        -- Competition info
        competition_name NVARCHAR(300),
        year INT,

        -- Event info
        heat_category VARCHAR(50),
        discipline_name NVARCHAR(100),
        gender VARCHAR(10),
        event_id VARCHAR(50),

        -- Athlete info
        full_name NVARCHAR(200),
        first_name NVARCHAR(100),
        last_name NVARCHAR(100),
        nationality VARCHAR(10),
        nationality_name NVARCHAR(100),
        person_id VARCHAR(50),
        biography_id VARCHAR(50),
        athlete_age INT,

        -- Race result
        result_id VARCHAR(50),
        lane INT,
        heat_rank INT,
        final_rank INT,
        time_raw VARCHAR(20),
        time_seconds DECIMAL(10,3),
        reaction_time DECIMAL(5,3),
        time_behind DECIMAL(10,3),
        fina_points INT,
        medal_tag CHAR(1),
        qualified VARCHAR(10),
        record_type VARCHAR(20),

        -- Split data
        splits_json NVARCHAR(MAX),
        lap_times_json NVARCHAR(MAX),
        pacing_type VARCHAR(30),
        lap_variance DECIMAL(10,5),

        -- Metadata
        created_at DATETIME2 DEFAULT GETDATE()
    );

    CREATE INDEX idx_results_flat_athlete ON results_flat(full_name);
    CREATE INDEX idx_results_flat_nat ON results_flat(nationality);
    CREATE INDEX idx_results_flat_year ON results_flat(year);
    CREATE INDEX idx_results_flat_event ON results_flat(discipline_name);
    CREATE INDEX idx_results_flat_time ON results_flat(time_seconds);
END;
GO

PRINT 'Schema creation completed successfully!';
