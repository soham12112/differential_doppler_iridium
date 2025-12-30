#!/usr/bin/env python3
"""
Orchestrates the full matching pipeline and reports confidence levels.
"""

import argparse
import shutil
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Define the pipeline steps
SCRIPTS_IN_ORDER = [
    "unique_matching.py",           # 1. Global Optimization (Hungarian Algo)
    "time_aligned_matching.py",     # 2. Physics-based Verification (Pearson/RMSE)
]

def run_script(script: Path, cwd: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Required script not found: {script}")
    print(f"\n{'='*80}")
    print(f"Running {script.name}")
    print('='*80)
    subprocess.run([sys.executable, script.name], cwd=cwd, check=True)

def print_final_report(target_dir: Path):
    results_file = target_dir / "time_aligned_matches.csv"
    
    print("\n" + "="*80)
    print("FINAL MATCHING CONFIDENCE REPORT")
    print("="*80)
    
    if not results_file.exists():
        print("⚠️  No results file found (time_aligned_matches.csv). Check script errors.")
        return

    df = pd.read_csv(results_file)
    
    # Define table headers
    print(f"{'Measured ID':<15} {'Identified As':<20} {'UE':<5} {'Inv?':<6} {'Confidence':<12} {'Status':<10}")
    print("-" * 75)
    
    for _, row in df.iterrows():
        # Determine Confidence Level based on Pearson Correlation
        score = row['confidence_score']
        if score >= 0.90:
            status = "✅ HIGH"
        elif score >= 0.70:
            status = "⚠️ MEDIUM"
        else:
            status = "❌ LOW"
            
        inv_str = "YES" if row['inverted'] else "NO"
        
        print(f"{row['measured_sat']:<15} {row['identified_sat']:<20} {row['ue']:<5} {inv_str:<6} {score:<12.4f} {status:<10}")

    print("\n" + "-"*75)
    print("Metrics Guide:")
    print("  * Confidence: Pearson Correlation (1.0 = Perfect Shape Match)")
    print("  * High: > 0.90 | Medium: > 0.70 | Low: < 0.70")
    print("="*80)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Iridium matching pipeline.")
    parser.add_argument("folder", nargs="?", default=".", help="Target folder")
    args = parser.parse_args()

    target_dir = Path(args.folder).resolve()
    if not target_dir.is_dir():
        parser.error(f"Folder does not exist: {target_dir}")

    # Verify input files exist
    if not list(target_dir.glob("ue*_doppler_data*.csv")):
        parser.error(f"Missing UE doppler data in {target_dir}")
    if not list(target_dir.glob("sat_*_ira_doppler.csv")):
        parser.error(f"Missing measured satellite data in {target_dir}")

    # Copy scripts if missing
    script_source_dir = Path(__file__).parent
    for script_name in SCRIPTS_IN_ORDER:
        source = script_source_dir / script_name
        target = target_dir / script_name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)

    # Run Pipeline
    try:
        for script_name in SCRIPTS_IN_ORDER:
            run_script(target_dir / script_name, cwd=target_dir)
            
        # PRINT THE REPORT
        print_final_report(target_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed at step: {e.cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()