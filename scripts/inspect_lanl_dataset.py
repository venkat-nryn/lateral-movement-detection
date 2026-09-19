#!/usr/bin/env python3
"""
LANL Dataset Inspection Script
Performs careful structural inspection of auth.txt.gz and redteam.txt.gz
without loading the full dataset into memory.
"""

import gzip
import os
import sys
from pathlib import Path
from collections import Counter
from itertools import islice
import time


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    while current != current.parent:
        if (current / "data" / "raw" / "lanl").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root with data/raw/lanl/")


def inspect_file_structure(filepath: Path, max_lines: int = 1000, sample_size: int = 10) -> dict:
    """Inspect a compressed file's structure using streaming."""
    result = {
        "file_exists": filepath.exists(),
        "file_size_bytes": filepath.stat().st_size if filepath.exists() else 0,
        "compression": "gzip",
        "readable": False,
        "num_lines": 0,
        "num_columns": 0,
        "delimiter": None,
        "has_header": False,
        "sample_records": [],
        "field_analysis": {},
        "timestamp_format": None,
        "user_representation": None,
        "host_representation": None,
        "auth_representation": None,
        "success_representation": None,
    }

    if not filepath.exists():
        return result

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            result["readable"] = True
            
            # Read first few lines for analysis
            first_lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                first_lines.append(line.strip())
            
            result["num_lines"] = len(first_lines)
            
            if first_lines:
                # Analyze delimiter
                first_line = first_lines[0]
                if ',' in first_line:
                    result["delimiter"] = ','
                    parts = first_line.split(',')
                elif '\t' in first_line:
                    result["delimiter"] = '\t'
                    parts = first_line.split('\t')
                else:
                    result["delimiter"] = 'unknown'
                    parts = [first_line]
                
                result["num_columns"] = len(parts)
                
                # Check if first line is a header
                # LANL data doesn't have headers - first line is data
                result["has_header"] = False
                
                # Sample records
                result["sample_records"] = first_lines[:sample_size]
                
                # Field analysis
                result["field_analysis"] = analyze_fields(first_lines, result["delimiter"])
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


def analyze_fields(lines: list, delimiter: str) -> dict:
    """Analyze field structure from sample lines."""
    if not lines or not delimiter:
        return {}
    
    field_counts = Counter()
    field_samples = {}
    
    for line in lines:
        parts = line.split(delimiter)
        field_counts[len(parts)] += 1
        if len(parts) not in field_samples:
            field_samples[len(parts)] = parts
    
    # Use the most common column count
    if field_counts:
        most_common_cols = field_counts.most_common(1)[0][0]
        sample = field_samples.get(most_common_cols, [])
        
        return {
            "column_count_distribution": dict(field_counts),
            "most_common_columns": most_common_cols,
            "sample_fields": sample,
        }
    
    return {}


def count_total_lines(filepath: Path) -> int:
    """Count total lines in a gzipped file using streaming."""
    count = 0
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def analyze_auth_file(filepath: Path, max_sample: int = 10000) -> dict:
    """Deep analysis of auth.txt.gz with bounded memory."""
    print(f"Analyzing {filepath.name}...")
    start = time.time()
    
    result = {
        "total_records": 0,
        "unique_users": set(),
        "unique_source_hosts": set(),
        "unique_dest_hosts": set(),
        "unique_auth_types": Counter(),
        "unique_logon_types": Counter(),
        "unique_orientations": Counter(),
        "success_counts": Counter(),
        "time_min": None,
        "time_max": None,
        "malformed_lines": 0,
        "empty_lines": 0,
    }
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                result["empty_lines"] += 1
                continue
            
            parts = line.split(',')
            if len(parts) != 9:
                result["malformed_lines"] += 1
                continue
            
            result["total_records"] += 1
            
            # Time (first column)
            try:
                t = int(parts[0])
                if result["time_min"] is None or t < result["time_min"]:
                    result["time_min"] = t
                if result["time_max"] is None or t > result["time_max"]:
                    result["time_max"] = t
            except ValueError:
                pass
            
            # User (second column: source user@domain)
            result["unique_users"].add(parts[1])
            
            # Source host (from user or 4th column)
            # The user format is USER@DOMAIN, but source computer is column 4
            src_host = parts[3]  # source computer
            result["unique_source_hosts"].add(src_host)
            
            # Destination host (column 5)
            dest_host = parts[4]
            result["unique_dest_hosts"].add(dest_host)
            
            # Auth type (column 6)
            result["unique_auth_types"][parts[5]] += 1
            
            # Logon type (column 7)
            result["unique_logon_types"][parts[6]] += 1
            
            # Orientation/Action (column 8)
            result["unique_orientations"][parts[7]] += 1
            
            # Success/Failure (column 9)
            result["success_counts"][parts[8]] += 1
            
            # Limit for reasonable runtime
            if i >= max_sample:
                break
    
    result["unique_users_count"] = len(result["unique_users"])
    result["unique_source_hosts_count"] = len(result["unique_source_hosts"])
    result["unique_dest_hosts_count"] = len(result["unique_dest_hosts"])
    
    # Convert sets to sorted lists for JSON serialization
    result["unique_users"] = sorted(result["unique_users"])[:20]
    result["unique_source_hosts"] = sorted(result["unique_source_hosts"])[:20]
    result["unique_dest_hosts"] = sorted(result["unique_dest_hosts"])[:20]
    
    # Convert counters
    result["unique_auth_types"] = dict(result["unique_auth_types"])
    result["unique_logon_types"] = dict(result["unique_logon_types"])
    result["unique_orientations"] = dict(result["unique_orientations"])
    result["success_counts"] = dict(result["success_counts"])
    
    elapsed = time.time() - start
    print(f"  Analyzed {result['total_records']:,} records in {elapsed:.1f}s")
    
    return result


def analyze_redteam_file(filepath: Path) -> dict:
    """Deep analysis of redteam.txt.gz."""
    print(f"Analyzing {filepath.name}...")
    start = time.time()
    
    result = {
        "total_records": 0,
        "unique_users": set(),
        "unique_source_hosts": set(),
        "unique_dest_hosts": set(),
        "time_min": None,
        "time_max": None,
        "malformed_lines": 0,
        "empty_lines": 0,
    }
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                result["empty_lines"] += 1
                continue
            
            parts = line.split(',')
            if len(parts) != 4:
                result["malformed_lines"] += 1
                continue
            
            result["total_records"] += 1
            
            # Time (first column)
            try:
                t = int(parts[0])
                if result["time_min"] is None or t < result["time_min"]:
                    result["time_min"] = t
                if result["time_max"] is None or t > result["time_max"]:
                    result["time_max"] = t
            except ValueError:
                pass
            
            # User (second column)
            result["unique_users"].add(parts[1])
            
            # Source host (third column)
            result["unique_source_hosts"].add(parts[2])
            
            # Destination host (fourth column)
            result["unique_dest_hosts"].add(parts[3])
    
    result["unique_users_count"] = len(result["unique_users"])
    result["unique_source_hosts_count"] = len(result["unique_source_hosts"])
    result["unique_dest_hosts_count"] = len(result["unique_dest_hosts"])
    
    result["unique_users"] = sorted(result["unique_users"])
    result["unique_source_hosts"] = sorted(result["unique_source_hosts"])
    result["unique_dest_hosts"] = sorted(result["unique_dest_hosts"])
    
    elapsed = time.time() - start
    print(f"  Analyzed {result['total_records']:,} records in {elapsed:.1f}s")
    
    return result


def check_duplicates(filepath: Path, sample_size: int = 100000) -> dict:
    """Check for duplicate records using streaming with bounded memory."""
    seen = set()
    duplicates = 0
    total = 0
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if line in seen:
                duplicates += 1
            else:
                seen.add(line)
            if total >= sample_size:
                break
    
    return {
        "sample_size": total,
        "unique_in_sample": len(seen),
        "duplicates_in_sample": duplicates,
    }


def main(output_file: Path | None = None):
    root = get_project_root()
    lanl_dir = root / "data" / "raw" / "lanl"
    
    auth_file = lanl_dir / "auth.txt.gz"
    redteam_file = lanl_dir / "redteam.txt.gz"
    
    print("=" * 60)
    print("LANL DATASET INSPECTION")
    print("=" * 60)
    
    # Basic file info
    print("\n1. FILE EXISTENCE AND SIZES")
    print("-" * 40)
    
    for f in [auth_file, redteam_file]:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.2f} MB ({f.stat().st_size:,} bytes)")
        else:
            print(f"  {f.name}: NOT FOUND")
    
    # Structure inspection
    print("\n2. STRUCTURE INSPECTION (first 1000 lines)")
    print("-" * 40)
    
    auth_struct = inspect_file_structure(auth_file, max_lines=1000)
    redteam_struct = inspect_file_structure(redteam_file, max_lines=1000)
    
    print(f"\n  auth.txt.gz:")
    print(f"    Readable: {auth_struct['readable']}")
    print(f"    Columns: {auth_struct['num_columns']}")
    print(f"    Delimiter: '{auth_struct['delimiter']}'")
    print(f"    Has header: {auth_struct['has_header']}")
    print(f"    Sample records (first 5):")
    for r in auth_struct['sample_records'][:5]:
        print(f"      {r}")
    
    print(f"\n  redteam.txt.gz:")
    print(f"    Readable: {redteam_struct['readable']}")
    print(f"    Columns: {redteam_struct['num_columns']}")
    print(f"    Delimiter: '{redteam_struct['delimiter']}'")
    print(f"    Has header: {redteam_struct['has_header']}")
    print(f"    Sample records (first 5):")
    for r in redteam_struct['sample_records'][:5]:
        print(f"      {r}")
    
    # Deep analysis (bounded)
    print("\n3. DEEP ANALYSIS (bounded sample)")
    print("-" * 40)
    
    auth_analysis = analyze_auth_file(auth_file, max_sample=100000)
    redteam_analysis = analyze_redteam_file(redteam_file)
    
    print(f"\n  auth.txt.gz (first 100k records):")
    print(f"    Total records scanned: {auth_analysis['total_records']:,}")
    print(f"    Unique users (sample): {auth_analysis['unique_users_count']:,}")
    print(f"    Unique source hosts (sample): {auth_analysis['unique_source_hosts_count']:,}")
    print(f"    Unique dest hosts (sample): {auth_analysis['unique_dest_hosts_count']:,}")
    print(f"    Time range: {auth_analysis['time_min']} - {auth_analysis['time_max']}")
    print(f"    Auth types: {auth_analysis['unique_auth_types']}")
    print(f"    Logon types: {auth_analysis['unique_logon_types']}")
    print(f"    Orientations: {auth_analysis['unique_orientations']}")
    print(f"    Success/Failure: {auth_analysis['success_counts']}")
    print(f"    Malformed lines: {auth_analysis['malformed_lines']}")
    print(f"    Empty lines: {auth_analysis['empty_lines']}")
    
    print(f"\n  redteam.txt.gz (all {redteam_analysis['total_records']} records):")
    print(f"    Total records: {redteam_analysis['total_records']:,}")
    print(f"    Unique users: {redteam_analysis['unique_users_count']:,}")
    print(f"    Unique source hosts: {redteam_analysis['unique_source_hosts_count']:,}")
    print(f"    Unique dest hosts: {redteam_analysis['unique_dest_hosts_count']:,}")
    print(f"    Time range: {redteam_analysis['time_min']} - {redteam_analysis['time_max']}")
    print(f"    Malformed lines: {redteam_analysis['malformed_lines']}")
    print(f"    Empty lines: {redteam_analysis['empty_lines']}")
    
    # Duplicate check
    print("\n4. DUPLICATE CHECK (first 100k records)")
    print("-" * 40)
    auth_dups = check_duplicates(auth_file, sample_size=100000)
    redteam_dups = check_duplicates(redteam_file, sample_size=100000)
    
    print(f"  auth.txt.gz: {auth_dups['duplicates_in_sample']} duplicates in {auth_dups['sample_size']:,} records")
    print(f"  redteam.txt.gz: {redteam_dups['duplicates_in_sample']} duplicates in {redteam_dups['sample_size']:,} records")
    
    # Cross-reference potential
    print("\n5. CROSS-REFERENCE POTENTIAL")
    print("-" * 40)
    
    auth_users = set(auth_analysis["unique_users"])
    redteam_users = set(redteam_analysis["unique_users"])
    common_users = auth_users & redteam_users
    
    auth_src_hosts = set(auth_analysis["unique_source_hosts"])
    redteam_src_hosts = set(redteam_analysis["unique_source_hosts"])
    common_src_hosts = auth_src_hosts & redteam_src_hosts
    
    auth_dest_hosts = set(auth_analysis["unique_dest_hosts"])
    redteam_dest_hosts = set(redteam_analysis["unique_dest_hosts"])
    common_dest_hosts = auth_dest_hosts & redteam_dest_hosts
    
    print(f"  Common users: {len(common_users)}")
    print(f"  Common source hosts: {len(common_src_hosts)}")
    print(f"  Common dest hosts: {len(common_dest_hosts)}")
    print(f"  Time overlap: auth({auth_analysis['time_min']}-{auth_analysis['time_max']}) vs redteam({redteam_analysis['time_min']}-{redteam_analysis['time_max']})")
    
    # Save results for report generation
    import json
    results = {
        "auth_structure": auth_struct,
        "redteam_structure": redteam_struct,
        "auth_analysis": auth_analysis,
        "redteam_analysis": redteam_analysis,
        "auth_duplicates": auth_dups,
        "redteam_duplicates": redteam_dups,
        "cross_reference": {
            "common_users": len(common_users),
            "common_source_hosts": len(common_src_hosts),
            "common_dest_hosts": len(common_dest_hosts),
        }
    }
    
    if output_file is None:
        output_file = root / "docs" / "lanl_inspection_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON
        def convert(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nInspection complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bounded LANL dataset inspection")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the JSON results "
        "(default: docs/lanl_inspection_results.json)",
    )
    main(parser.parse_args().output)