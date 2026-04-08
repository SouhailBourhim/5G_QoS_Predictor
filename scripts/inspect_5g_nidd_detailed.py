# scripts/inspect_5g_nidd_detailed.py

"""
Detailed 5G-NIDD Inspector for Intrusion Detection Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

NIDD_DIR = Path("data/raw/5g_nidd")


def inspect_csv_files(directory: Path):
    """Inspect all CSV files in a directory."""
    csv_files = list(directory.rglob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {directory}")
        return
    
    print(f"\n📊 Found {len(csv_files)} CSV file(s) in {directory.name}")
    print("=" * 70)
    
    for i, csv_file in enumerate(csv_files[:5], 1):  # Show first 5
        print(f"\n[{i}] {csv_file.name}")
        print("-" * 70)
        
        try:
            # Read just the first few rows to inspect structure
            df = pd.read_csv(csv_file, nrows=100)
            
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns (showing first 100 rows)")
            print(f"   Columns ({len(df.columns)}):")
            
            for j, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
                print(f"      {j:2d}. {col:35s} | {str(dtype):12s} | Sample: {sample_val}")
            
            # Show first 3 rows
            print(f"\n   First 3 rows:")
            print(df.head(3).to_string(max_colwidth=30))
            
            return df  # Return first found file for further analysis
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if len(csv_files) > 5:
        print(f"\n   ... and {len(csv_files) - 5} more files")


def main():
    print("\n" + "=" * 70)
    print("5G-NIDD DETAILED DATASET INSPECTOR")
    print("=" * 70)
    
    # Check what directories exist
    subdirs = [d for d in NIDD_DIR.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("\n⚠️  No extracted directories found.")
        print("   Run the extraction script first:")
        print("   bash scripts/extract_5g_nidd.sh")
        return
    
    print(f"\n📁 Found {len(subdirs)} extracted director{'y' if len(subdirs) == 1 else 'ies'}:")
    for d in subdirs:
        csv_count = len(list(d.rglob("*.csv")))
        print(f"   • {d.name:30s} ({csv_count} CSV files)")
    
    # Inspect each directory
    for subdir in subdirs:
        df = inspect_csv_files(subdir)
        if df is not None:
            # Analyze this file more deeply
            print("\n" + "=" * 70)
            print("DEEP ANALYSIS OF FIRST FILE")
            print("=" * 70)
            
            # Look for 5G-specific columns
            print("\n🔍 Searching for 5G/Network KPI columns:")
            
            kpi_keywords = {
                "Throughput/Bandwidth": ['byte', 'bit', 'throughput', 'rate', 'bandwidth'],
                "Timing/Latency": ['time', 'dur', 'delay', 'latency', 'rtt'],
                "Packets": ['packet', 'pkts', 'frame'],
                "Protocol": ['proto', 'protocol', 'tcp', 'udp', 'ip'],
                "Addresses": ['addr', 'ip', 'src', 'dst', 'port'],
                "Radio/Signal": ['rsrp', 'rsrq', 'sinr', 'signal', 'quality'],
            }
            
            found_cols = {category: [] for category in kpi_keywords}
            
            for col in df.columns:
                col_lower = col.lower()
                for category, keywords in kpi_keywords.items():
                    if any(kw in col_lower for kw in keywords):
                        found_cols[category].append(col)
            
            for category, cols in found_cols.items():
                if cols:
                    print(f"\n   {category}:")
                    for col in cols:
                        print(f"      • {col}")
            
            # Statistics on numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n📈 Statistics (first 10 numeric columns):")
                print(df[numeric_cols[:10]].describe().to_string())
            
            break  # Stop after first valid file
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Based on the columns found above, identify which are useful KPIs
2. Check if there's a "label" or "attack_type" column (normal vs attack)
3. Filter for NORMAL traffic only to get baseline 5G behavior
4. Extract statistics from normal traffic for synthetic data calibration

Suggested approach:
- Use NORMAL traffic data to understand typical 5G KPI ranges
- Ignore the attack-related rows for calibration
- Focus on: throughput, latency, packet rates, timing features
""")


if __name__ == "__main__":
    main()