"""
Swimming Performance Database Package
"""

from .models import (
    Base,
    Athlete,
    Competition,
    Event,
    Result,
    WorldRecord,
    EliteBenchmark,
    create_database,
    get_session,
    get_database_url,
    initialize_reference_data,
    WORLD_RECORDS_LCM,
    ELITE_BENCHMARKS,
)

from .import_csv import (
    import_results_from_csv,
    import_all_csv_files,
    time_to_seconds,
    clear_caches,
    sync_new_data,
)

__all__ = [
    'Base',
    'Athlete',
    'Competition',
    'Event',
    'Result',
    'WorldRecord',
    'EliteBenchmark',
    'create_database',
    'get_session',
    'get_database_url',
    'initialize_reference_data',
    'import_results_from_csv',
    'import_all_csv_files',
    'time_to_seconds',
    'clear_caches',
    'sync_new_data',
    'WORLD_RECORDS_LCM',
    'ELITE_BENCHMARKS',
]
