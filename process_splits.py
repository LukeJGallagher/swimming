"""
Process existing swimming data to add split analysis columns
Enriches CSV files with pacing type, lap times, and other split metrics
"""

import pandas as pd
import json
import ast
from pathlib import Path
from tqdm import tqdm
from enhanced_swimming_scraper import SplitTimeAnalyzer


def parse_splits_safe(splits_str):
    """Safely parse splits from string representation"""
    if not splits_str or pd.isna(splits_str) or splits_str == '[]':
        return []

    try:
        # Try JSON first
        if isinstance(splits_str, str):
            # Handle string representation of list
            if splits_str.startswith('['):
                try:
                    return json.loads(splits_str.replace("'", '"'))
                except:
                    return ast.literal_eval(splits_str)
        elif isinstance(splits_str, list):
            return splits_str
    except (json.JSONDecodeError, ValueError, SyntaxError):
        pass

    return []


def process_file(filepath: Path, output_dir: Path = None) -> pd.DataFrame:
    """Process a single CSV file to add split analysis"""
    print(f"\nProcessing: {filepath.name}")

    df = pd.read_csv(filepath)
    original_len = len(df)

    # Check for Splits column
    if 'Splits' not in df.columns:
        print(f"  No Splits column found in {filepath.name}")
        return df

    analyzer = SplitTimeAnalyzer()

    # Initialize new columns
    new_columns = {
        'splits_json': [],
        'lap_times_json': [],
        'pacing_type': [],
        'first_half_avg': [],
        'second_half_avg': [],
        'split_difference': [],
        'fastest_lap': [],
        'slowest_lap': [],
        'lap_variance': [],
        'num_splits': []
    }

    splits_found = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Analyzing splits"):
        splits_raw = row.get('Splits', '[]')
        splits = parse_splits_safe(splits_raw)

        if splits:
            splits_found += 1
            parsed = analyzer.parse_splits(splits)
            lap_times = analyzer.calculate_lap_times(parsed)
            pacing = analyzer.analyze_pacing(lap_times)

            new_columns['splits_json'].append(json.dumps(parsed))
            new_columns['lap_times_json'].append(json.dumps(lap_times))
            new_columns['num_splits'].append(len(parsed))

            if pacing:
                new_columns['pacing_type'].append(pacing['pacing_type'])
                new_columns['first_half_avg'].append(pacing['first_half_avg'])
                new_columns['second_half_avg'].append(pacing['second_half_avg'])
                new_columns['split_difference'].append(pacing['split_difference'])
                new_columns['fastest_lap'].append(pacing['fastest_lap'])
                new_columns['slowest_lap'].append(pacing['slowest_lap'])
                new_columns['lap_variance'].append(pacing['lap_variance'])
            else:
                for key in ['pacing_type', 'first_half_avg', 'second_half_avg',
                           'split_difference', 'fastest_lap', 'slowest_lap', 'lap_variance']:
                    new_columns[key].append(None)
        else:
            for key in new_columns:
                new_columns[key].append(None)

    # Add new columns to dataframe
    for col, values in new_columns.items():
        df[col] = values

    print(f"  Total rows: {original_len}")
    print(f"  Rows with splits: {splits_found} ({splits_found/original_len*100:.1f}%)")

    # Pacing distribution
    if 'pacing_type' in df.columns:
        pacing_counts = df['pacing_type'].value_counts()
        print(f"  Pacing distribution:")
        for ptype, count in pacing_counts.items():
            if pd.notna(ptype):
                print(f"    {ptype}: {count}")

    # Save enriched file
    if output_dir:
        output_path = output_dir / f"enriched_{filepath.name}"
    else:
        output_path = filepath.parent / f"enriched_{filepath.name}"

    df.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df


def process_all_files(data_dir: str = ".", output_dir: str = "data"):
    """Process all result files in directory"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all result files
    result_files = list(data_path.glob("Results_*.csv"))
    result_files = [f for f in result_files if not f.name.startswith('enriched_')]

    print(f"Found {len(result_files)} result files to process")

    all_results = []

    for filepath in sorted(result_files):
        df = process_file(filepath, output_path)
        all_results.append(df)

    # Create combined enriched file
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = output_path / "all_results_enriched.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined enriched file saved to: {combined_path}")
        print(f"Total rows: {len(combined)}")

    return all_results


def process_single_year(year: int, data_dir: str = "."):
    """Process a single year's data"""
    filepath = Path(data_dir) / f"Results_{year}.csv"
    if filepath.exists():
        return process_file(filepath)
    else:
        print(f"File not found: {filepath}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process swimming data for split analysis")
    parser.add_argument("--year", type=int, help="Process specific year")
    parser.add_argument("--all", action="store_true", help="Process all years")
    parser.add_argument("--file", type=str, help="Process specific file")

    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file))
    elif args.year:
        process_single_year(args.year)
    elif args.all:
        process_all_files()
    else:
        # Default: process 2024 only
        print("Processing 2024 data (use --all for all years)")
        process_single_year(2024)
