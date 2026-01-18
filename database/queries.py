"""
Database Query Utilities
Common queries for swimming performance analysis
"""

from sqlalchemy import func, desc, and_, or_
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from .models import Athlete, Competition, Event, Result, WorldRecord, EliteBenchmark


class SwimmingQueries:
    """Collection of common swimming database queries."""

    def __init__(self, session: Session):
        self.session = session

    # ============================================================
    # ATHLETE QUERIES
    # ============================================================

    def get_athlete_by_name(self, name: str) -> Optional[Athlete]:
        """Find athlete by name (case-insensitive partial match)."""
        return self.session.query(Athlete).filter(
            Athlete.full_name.ilike(f'%{name}%')
        ).first()

    def get_athlete_results(self, athlete_id: int,
                           event: str = None,
                           limit: int = None) -> List[Result]:
        """Get all results for an athlete."""
        query = self.session.query(Result).filter(Result.athlete_id == athlete_id)

        if event:
            query = query.join(Event).filter(Event.discipline_name.ilike(f'%{event}%'))

        query = query.order_by(desc(Result.year), Result.time_seconds)

        if limit:
            query = query.limit(limit)

        return query.all()

    def get_athlete_personal_bests(self, athlete_id: int) -> Dict[str, Dict]:
        """Get personal bests for an athlete by event."""
        results = self.session.query(
            Event.discipline_name,
            func.min(Result.time_seconds).label('pb'),
            func.count(Result.id).label('race_count')
        ).join(Event).filter(
            Result.athlete_id == athlete_id,
            Result.time_seconds > 0
        ).group_by(Event.discipline_name).all()

        return {
            r.discipline_name: {'pb': float(r.pb), 'race_count': r.race_count}
            for r in results
        }

    def search_athletes(self, name: str = None, nationality: str = None,
                       limit: int = 100) -> List[Athlete]:
        """Search athletes with filters."""
        query = self.session.query(Athlete)

        if name:
            query = query.filter(Athlete.full_name.ilike(f'%{name}%'))
        if nationality:
            query = query.filter(Athlete.nationality == nationality.upper())

        return query.limit(limit).all()

    # ============================================================
    # COMPETITION QUERIES
    # ============================================================

    def get_competition_results(self, competition_id: int,
                                event: str = None) -> List[Result]:
        """Get all results from a competition."""
        query = self.session.query(Result).join(Competition).filter(
            Competition.competition_id == competition_id
        )

        if event:
            query = query.join(Event).filter(Event.discipline_name.ilike(f'%{event}%'))

        return query.order_by(Result.final_rank).all()

    def get_medal_table(self, competition_id: int = None,
                       year: int = None) -> List[Dict]:
        """Get medal table by nationality."""
        query = self.session.query(
            Athlete.nationality,
            func.count(func.nullif(Result.medal_tag, '')).filter(
                Result.medal_tag == 'G'
            ).label('gold'),
            func.count(func.nullif(Result.medal_tag, '')).filter(
                Result.medal_tag == 'S'
            ).label('silver'),
            func.count(func.nullif(Result.medal_tag, '')).filter(
                Result.medal_tag == 'B'
            ).label('bronze'),
        ).join(Athlete)

        if competition_id:
            query = query.join(Competition).filter(
                Competition.competition_id == competition_id
            )
        if year:
            query = query.filter(Result.year == year)

        results = query.filter(
            Result.medal_tag.in_(['G', 'S', 'B'])
        ).group_by(Athlete.nationality).all()

        return sorted([
            {
                'nationality': r.nationality,
                'gold': r.gold,
                'silver': r.silver,
                'bronze': r.bronze,
                'total': r.gold + r.silver + r.bronze
            }
            for r in results
        ], key=lambda x: (-x['gold'], -x['silver'], -x['bronze']))

    # ============================================================
    # EVENT/PERFORMANCE QUERIES
    # ============================================================

    def get_event_rankings(self, discipline: str, year: int = None,
                          limit: int = 100) -> List[Dict]:
        """Get top performers in an event."""
        query = self.session.query(
            Athlete.full_name,
            Athlete.nationality,
            func.min(Result.time_seconds).label('best_time'),
            func.count(Result.id).label('race_count')
        ).join(Athlete).join(Event).filter(
            Event.discipline_name.ilike(f'%{discipline}%'),
            Result.time_seconds > 0
        )

        if year:
            query = query.filter(Result.year == year)

        results = query.group_by(
            Athlete.id, Athlete.full_name, Athlete.nationality
        ).order_by('best_time').limit(limit).all()

        return [
            {
                'athlete': r.full_name,
                'nationality': r.nationality,
                'best_time': float(r.best_time),
                'race_count': r.race_count
            }
            for r in results
        ]

    def get_world_record_percentage(self, result_id: int) -> Optional[float]:
        """Calculate WR percentage for a result."""
        result = self.session.query(Result).filter(Result.id == result_id).first()
        if not result or not result.time_seconds:
            return None

        event = self.session.query(Event).filter(Event.id == result.event_id).first()
        if not event:
            return None

        wr = self.session.query(WorldRecord).filter(
            WorldRecord.discipline_name == event.discipline_name
        ).first()

        if not wr:
            return None

        return (wr.record_time / result.time_seconds) * 100

    # ============================================================
    # PACING ANALYSIS QUERIES
    # ============================================================

    def get_pacing_effectiveness(self, discipline: str = None) -> List[Dict]:
        """Analyze pacing strategy effectiveness."""
        query = self.session.query(
            Result.pacing_type,
            func.avg(Result.final_rank).label('avg_rank'),
            func.count(Result.id).label('count'),
            func.count(Result.medal_tag).label('medals'),
            func.avg(Result.lap_variance).label('avg_variance')
        )

        if discipline:
            query = query.join(Event).filter(
                Event.discipline_name.ilike(f'%{discipline}%')
            )

        results = query.filter(
            Result.pacing_type.isnot(None),
            Result.pacing_type != ''
        ).group_by(Result.pacing_type).all()

        return [
            {
                'pacing_type': r.pacing_type,
                'avg_rank': float(r.avg_rank) if r.avg_rank else None,
                'count': r.count,
                'medals': r.medals,
                'avg_variance': float(r.avg_variance) if r.avg_variance else None
            }
            for r in results
        ]

    def get_athlete_pacing_profile(self, athlete_id: int) -> Dict:
        """Get pacing profile for an athlete."""
        results = self.session.query(
            Result.pacing_type,
            func.count(Result.id).label('count')
        ).filter(
            Result.athlete_id == athlete_id,
            Result.pacing_type.isnot(None)
        ).group_by(Result.pacing_type).all()

        total = sum(r.count for r in results)
        return {
            r.pacing_type: {'count': r.count, 'percentage': r.count / total * 100}
            for r in results
        } if total > 0 else {}

    # ============================================================
    # PROGRESSION QUERIES
    # ============================================================

    def get_athlete_progression(self, athlete_id: int,
                                discipline: str) -> List[Dict]:
        """Track athlete's progression over time in an event."""
        results = self.session.query(
            Result.year,
            func.min(Result.time_seconds).label('best_time'),
            func.avg(Result.time_seconds).label('avg_time'),
            func.count(Result.id).label('races')
        ).join(Event).filter(
            Result.athlete_id == athlete_id,
            Event.discipline_name.ilike(f'%{discipline}%'),
            Result.time_seconds > 0
        ).group_by(Result.year).order_by(Result.year).all()

        progression = []
        prev_best = None
        for r in results:
            improvement = None
            if prev_best:
                improvement = prev_best - r.best_time

            progression.append({
                'year': r.year,
                'best_time': float(r.best_time),
                'avg_time': float(r.avg_time),
                'races': r.races,
                'improvement': improvement
            })
            prev_best = r.best_time

        return progression

    def get_heats_to_finals_analysis(self, athlete_id: int) -> List[Dict]:
        """Analyze heats to finals improvement."""
        # Get results grouped by competition and event
        results = self.session.query(Result).filter(
            Result.athlete_id == athlete_id,
            Result.time_seconds > 0
        ).order_by(Result.competition_id, Result.event_id).all()

        # Group by competition/event
        grouped = {}
        for r in results:
            key = (r.competition_id, r.event_id)
            if key not in grouped:
                grouped[key] = {'heats': [], 'finals': []}

            if r.heat_category and 'final' in r.heat_category.lower():
                grouped[key]['finals'].append(r.time_seconds)
            elif r.heat_category and 'heat' in r.heat_category.lower():
                grouped[key]['heats'].append(r.time_seconds)

        # Calculate improvements
        improvements = []
        for (comp_id, event_id), data in grouped.items():
            if data['heats'] and data['finals']:
                heat_best = min(data['heats'])
                final_best = min(data['finals'])
                improvement_pct = ((heat_best - final_best) / heat_best) * 100

                improvements.append({
                    'competition_id': comp_id,
                    'event_id': event_id,
                    'heat_time': float(heat_best),
                    'final_time': float(final_best),
                    'improvement_pct': improvement_pct
                })

        return improvements

    # ============================================================
    # STATISTICS QUERIES
    # ============================================================

    def get_database_stats(self) -> Dict:
        """Get overall database statistics."""
        return {
            'athletes': self.session.query(Athlete).count(),
            'competitions': self.session.query(Competition).count(),
            'events': self.session.query(Event).count(),
            'results': self.session.query(Result).count(),
            'results_with_splits': self.session.query(Result).filter(
                Result.splits_json.isnot(None)
            ).count(),
            'year_range': {
                'min': self.session.query(func.min(Result.year)).scalar(),
                'max': self.session.query(func.max(Result.year)).scalar()
            }
        }

    def get_nationality_stats(self, nationality: str) -> Dict:
        """Get statistics for a specific country."""
        athletes = self.session.query(Athlete).filter(
            Athlete.nationality == nationality.upper()
        ).count()

        results = self.session.query(Result).join(Athlete).filter(
            Athlete.nationality == nationality.upper()
        )

        medals = results.filter(Result.medal_tag.in_(['G', 'S', 'B'])).count()
        finals = results.filter(
            Result.heat_category.ilike('%final%')
        ).count()

        return {
            'nationality': nationality.upper(),
            'athletes': athletes,
            'total_results': results.count(),
            'medals': medals,
            'finals': finals
        }


def query_to_dataframe(results: list) -> pd.DataFrame:
    """Convert query results to pandas DataFrame."""
    if not results:
        return pd.DataFrame()

    if isinstance(results[0], dict):
        return pd.DataFrame(results)

    # SQLAlchemy objects
    return pd.DataFrame([
        {c.name: getattr(r, c.name) for c in r.__table__.columns}
        for r in results
    ])


if __name__ == "__main__":
    # Demo queries
    from .models import create_database, get_session

    engine = create_database()
    session = get_session(engine)
    queries = SwimmingQueries(session)

    print("Database Statistics:")
    stats = queries.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
