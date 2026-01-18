-- Swimming Performance Database Schema
-- Team Saudi Performance Analysis
-- Designed for PostgreSQL (compatible with SQLite with minor modifications)

-- ============================================================
-- CORE TABLES
-- ============================================================

-- Athletes table (normalized from results)
CREATE TABLE IF NOT EXISTS athletes (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50) UNIQUE,
    full_name VARCHAR(200) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    nationality VARCHAR(10),
    nationality_name VARCHAR(100),
    biography_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_athletes_name ON athletes(full_name);
CREATE INDEX idx_athletes_nat ON athletes(nationality);

-- Competitions table
CREATE TABLE IF NOT EXISTS competitions (
    id SERIAL PRIMARY KEY,
    competition_id INTEGER UNIQUE,
    competition_name VARCHAR(300) NOT NULL,
    date_from DATE,
    date_to DATE,
    year INTEGER,
    host_city VARCHAR(100),
    host_country VARCHAR(100),
    pool_type VARCHAR(10), -- LCM, SCM
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_competitions_year ON competitions(year);
CREATE INDEX idx_competitions_name ON competitions(competition_name);

-- Events/Disciplines table
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(50) UNIQUE,
    discipline_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10), -- Men, Women, Mixed
    distance INTEGER, -- in meters
    stroke VARCHAR(50), -- Freestyle, Backstroke, Breaststroke, Butterfly, Individual Medley
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_discipline ON events(discipline_name);
CREATE INDEX idx_events_gender ON events(gender);

-- ============================================================
-- RESULTS TABLE (Main fact table)
-- ============================================================

CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    result_id VARCHAR(50) UNIQUE,

    -- Foreign keys
    athlete_id INTEGER REFERENCES athletes(id),
    competition_id INTEGER REFERENCES competitions(id),
    event_id INTEGER REFERENCES events(id),

    -- Race details
    heat_category VARCHAR(50), -- Finals, Heats, Semi-Finals, etc.
    heat_rank INTEGER,
    final_rank INTEGER,
    lane INTEGER,

    -- Timing
    time_raw VARCHAR(20), -- Original time string
    time_seconds DECIMAL(10,3), -- Converted to seconds
    reaction_time DECIMAL(5,3),
    time_behind DECIMAL(10,3),

    -- Scoring
    fina_points INTEGER,
    medal_tag CHAR(1), -- G, S, B, NULL
    qualified VARCHAR(10),
    record_type VARCHAR(20), -- WR, OR, CR, NR, etc.

    -- Split data (JSON for flexibility)
    splits_json JSONB,
    lap_times_json JSONB,

    -- Pacing analysis
    pacing_type VARCHAR(30), -- Even, Positive Split, Negative Split, U-shape, Inverted-J
    first_half_avg DECIMAL(10,3),
    second_half_avg DECIMAL(10,3),
    split_difference DECIMAL(10,3),
    fastest_lap DECIMAL(10,3),
    slowest_lap DECIMAL(10,3),
    lap_variance DECIMAL(10,5),

    -- Metadata
    athlete_age INTEGER,
    year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_results_athlete ON results(athlete_id);
CREATE INDEX idx_results_competition ON results(competition_id);
CREATE INDEX idx_results_event ON results(event_id);
CREATE INDEX idx_results_year ON results(year);
CREATE INDEX idx_results_time ON results(time_seconds);
CREATE INDEX idx_results_rank ON results(final_rank);
CREATE INDEX idx_results_medal ON results(medal_tag);
CREATE INDEX idx_results_pacing ON results(pacing_type);

-- ============================================================
-- ANALYTICS VIEWS
-- ============================================================

-- Personal bests view
CREATE OR REPLACE VIEW athlete_personal_bests AS
SELECT
    a.id AS athlete_id,
    a.full_name,
    a.nationality,
    e.discipline_name,
    MIN(r.time_seconds) AS personal_best,
    COUNT(*) AS race_count,
    MAX(r.year) AS last_raced
FROM results r
JOIN athletes a ON r.athlete_id = a.id
JOIN events e ON r.event_id = e.id
WHERE r.time_seconds > 0
GROUP BY a.id, a.full_name, a.nationality, e.discipline_name;

-- Medal table view
CREATE OR REPLACE VIEW medal_table AS
SELECT
    a.nationality,
    COUNT(CASE WHEN r.medal_tag = 'G' THEN 1 END) AS gold,
    COUNT(CASE WHEN r.medal_tag = 'S' THEN 1 END) AS silver,
    COUNT(CASE WHEN r.medal_tag = 'B' THEN 1 END) AS bronze,
    COUNT(r.medal_tag) AS total_medals
FROM results r
JOIN athletes a ON r.athlete_id = a.id
WHERE r.medal_tag IS NOT NULL
GROUP BY a.nationality
ORDER BY gold DESC, silver DESC, bronze DESC;

-- Pacing strategy effectiveness view
CREATE OR REPLACE VIEW pacing_effectiveness AS
SELECT
    e.discipline_name,
    r.pacing_type,
    COUNT(*) AS race_count,
    AVG(r.final_rank) AS avg_rank,
    COUNT(CASE WHEN r.medal_tag IS NOT NULL THEN 1 END) AS medals_won,
    AVG(r.lap_variance) AS avg_lap_variance
FROM results r
JOIN events e ON r.event_id = e.id
WHERE r.pacing_type IS NOT NULL
GROUP BY e.discipline_name, r.pacing_type
ORDER BY e.discipline_name, avg_rank;

-- Athlete progression view
CREATE OR REPLACE VIEW athlete_progression AS
SELECT
    a.id AS athlete_id,
    a.full_name,
    e.discipline_name,
    r.year,
    MIN(r.time_seconds) AS best_time_that_year,
    AVG(r.time_seconds) AS avg_time_that_year,
    COUNT(*) AS races_that_year
FROM results r
JOIN athletes a ON r.athlete_id = a.id
JOIN events e ON r.event_id = e.id
WHERE r.time_seconds > 0
GROUP BY a.id, a.full_name, e.discipline_name, r.year
ORDER BY a.full_name, e.discipline_name, r.year;

-- ============================================================
-- WORLD RECORDS REFERENCE TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS world_records (
    id SERIAL PRIMARY KEY,
    discipline_name VARCHAR(100) UNIQUE NOT NULL,
    record_time DECIMAL(10,3) NOT NULL,
    record_holder VARCHAR(200),
    record_date DATE,
    pool_type VARCHAR(10) DEFAULT 'LCM',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert current world records (LCM)
INSERT INTO world_records (discipline_name, record_time) VALUES
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
ON CONFLICT (discipline_name) DO UPDATE SET record_time = EXCLUDED.record_time;

-- ============================================================
-- ELITE BENCHMARKS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS elite_benchmarks (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) UNIQUE NOT NULL,
    metric_value DECIMAL(10,3),
    description TEXT,
    source VARCHAR(200)
);

INSERT INTO elite_benchmarks (metric_name, metric_value, description, source) VALUES
    ('fina_points_elite', 900, 'FINA points threshold for elite level', 'World Aquatics'),
    ('years_to_elite', 8, 'Average years of competition to reach elite', 'Career trajectory studies'),
    ('peak_window_years', 2.6, 'Years within 2% of career best', 'PLOS ONE 2024'),
    ('cv_elite_threshold', 1.3, 'Coefficient of variation for lap times (%)', 'Frontiers 2024'),
    ('heats_to_finals_improvement', 1.2, 'Expected % improvement for medalists', 'World Championships analysis'),
    ('male_peak_age', 24.2, 'Average peak performance age for males', 'PLOS ONE 2024'),
    ('female_peak_age', 22.5, 'Average peak performance age for females', 'PLOS ONE 2024')
ON CONFLICT (metric_name) DO NOTHING;
