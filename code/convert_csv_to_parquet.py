#!/usr/bin/env python3
"""
convert_csv_to_parquet.py

Converts the existing CSV embeddings file to Parquet format,
split by island (one file per island).

Usage:
    python convert_csv_to_parquet.py --csv ./out/channel_islands_embeddings.csv --outdir ./out/parquet
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet, split by island")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--outdir", default=None, help="Output directory for Parquet files (default: same as CSV)")
    args = parser.parse_args()
    
    # Determine output directory
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.csv) or '.', 'parquet')
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load CSV
    print(f"Loading {os.path.basename(args.csv)}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows")
    
    # Get unique islands
    islands = df['island'].unique()
    print(f"Found {len(islands)} islands: {list(islands)}")
    
    # Export each island to a separate Parquet file
    for island in islands:
        island_df = df[df['island'] == island]
        
        # Create safe filename
        safe_name = island.replace(' ', '_').lower()
        parquet_path = os.path.join(args.outdir, f"{safe_name}.parquet")
        
        # Save to Parquet with compression
        island_df.to_parquet(parquet_path, index=False, compression='snappy')
        print(f"[SAVED] {parquet_path} ({len(island_df):,} pixels)")
    
    # Also save a combined file
    combined_path = os.path.join(args.outdir, "all_islands.parquet")
    df.to_parquet(combined_path, index=False, compression='snappy')
    print(f"[SAVED] {combined_path} ({len(df):,} pixels)")
    
    print("\nDone!")
    print(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()

