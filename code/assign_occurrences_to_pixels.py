#!/usr/bin/env python3
"""
assign_occurrences_to_pixels.py

Assign each filtered GBIF occurrence to the nearest TESSERA pixel based on
lat/lon coordinates.

TESSERA Coordinate System:
    - Tile size: 0.1° × 0.1° (approximately 11km × 11km at equator)
    - Tile naming: By center coordinates (e.g., grid_0.15_52.05)
    - Resolution: 10m per pixel
    - Each pixel has a center coordinate; we snap occurrences to nearest center

Workflow:
    1. Load filtered GBIF occurrences (with decimalLongitude, decimalLatitude)
    2. Load TESSERA parquet data and convert pixel coords to WGS84 (lon/lat)
    3. Build KDTree spatial index on pixel centers
    4. For each occurrence, find nearest pixel center
    5. Join occurrence with pixel's 128-dim embedding
    6. Output combined dataset

Usage:
    python assign_occurrences_to_pixels.py \
        --occurrences ./out/species/quercus_tomentella_channel_islands.csv \
        --parquet ./out/parquet/all_islands.parquet \
        --outdir ./out/species
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assign GBIF occurrences to nearest TESSERA pixels."
    )
    
    parser.add_argument(
        "--occurrences",
        type=str,
        default="./out/species/quercus_tomentella_channel_islands.csv",
        help="Path to filtered occurrence CSV"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="./out/parquet/all_islands.parquet",
        help="Path to TESSERA embeddings parquet"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./out/species",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.occurrences):
        parser.error(f"Occurrences file not found: {args.occurrences}")
    if not os.path.exists(args.parquet):
        parser.error(f"Parquet file not found: {args.parquet}")
    
    return args


def load_occurrences(path):
    """Load filtered GBIF occurrences."""
    print(f"\n{'='*60}")
    print("LOADING OCCURRENCES")
    print(f"{'='*60}")
    print(f"[INFO] Loading: {os.path.basename(path)}")
    
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} occurrences")
    
    # Verify required columns
    required = ['decimalLongitude', 'decimalLatitude']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)
    
    return df


def load_tessera_with_lonlat(parquet_path):
    """
    Load TESSERA embeddings and convert pixel coordinates to WGS84.
    
    The parquet contains:
        - x, y: pixel coordinates in tile CRS (usually UTM)
        - crs: the coordinate reference system (e.g., EPSG:32611)
        - tile_lon, tile_lat: tile center in WGS84
        - band_0 to band_127: embedding values
    
    We need to convert x, y to lon, lat for distance comparisons.
    """
    print(f"\n{'='*60}")
    print("LOADING TESSERA EMBEDDINGS")
    print(f"{'='*60}")
    print(f"[INFO] Loading: {os.path.basename(parquet_path)}")
    
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df):,} pixels")
    print(f"[INFO] Islands: {df['island'].unique().tolist()}")
    
    # Convert x, y (UTM) to lon, lat (WGS84)
    print("[INFO] Converting pixel coordinates to WGS84...")
    
    lons = np.zeros(len(df))
    lats = np.zeros(len(df))
    
    # Group by CRS for efficient batch transformation
    for crs in df['crs'].unique():
        mask = df['crs'] == crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        x_vals = df.loc[mask, 'x'].values
        y_vals = df.loc[mask, 'y'].values
        
        lon_vals, lat_vals = transformer.transform(x_vals, y_vals)
        
        lons[mask.values] = lon_vals
        lats[mask.values] = lat_vals
    
    df['pixel_lon'] = lons
    df['pixel_lat'] = lats
    
    print(f"[INFO] Pixel coordinate ranges:")
    print(f"       Longitude: [{df['pixel_lon'].min():.5f}, {df['pixel_lon'].max():.5f}]")
    print(f"       Latitude:  [{df['pixel_lat'].min():.5f}, {df['pixel_lat'].max():.5f}]")
    
    return df


def build_spatial_index(tessera_df):
    """
    Build KDTree spatial index for fast nearest-neighbor queries.
    
    Uses lon/lat coordinates. For the Channel Islands region,
    Euclidean distance in degrees is a reasonable approximation
    since the area is relatively small.
    """
    print(f"\n{'='*60}")
    print("BUILDING SPATIAL INDEX")
    print(f"{'='*60}")
    
    coords = np.column_stack([
        tessera_df['pixel_lon'].values,
        tessera_df['pixel_lat'].values
    ])
    
    tree = cKDTree(coords)
    print(f"[INFO] KDTree built with {len(coords):,} pixel centers")
    
    return tree


def assign_to_nearest_pixels(occurrences_df, tessera_df, tree):
    """
    Assign each occurrence to the nearest TESSERA pixel center.
    
    Returns a combined DataFrame with occurrence info + pixel embeddings.
    """
    print(f"\n{'='*60}")
    print("ASSIGNING OCCURRENCES TO PIXELS")
    print(f"{'='*60}")
    
    # Query coordinates from occurrences
    query_coords = np.column_stack([
        occurrences_df['decimalLongitude'].values,
        occurrences_df['decimalLatitude'].values
    ])
    
    print(f"[INFO] Finding nearest pixels for {len(query_coords):,} occurrences...")
    
    # Find nearest neighbor for each occurrence
    distances, indices = tree.query(query_coords, k=1)
    
    # Convert distances from degrees to approximate meters
    # At ~34°N latitude: 1° lat ≈ 111km, 1° lon ≈ 92km
    # Use average: 1° ≈ 100km = 100,000m
    distances_m = distances * 100000
    
    print(f"[INFO] Distance statistics (to nearest pixel center):")
    print(f"       Min:    {distances_m.min():.1f} m")
    print(f"       Max:    {distances_m.max():.1f} m")
    print(f"       Mean:   {distances_m.mean():.1f} m")
    print(f"       Median: {np.median(distances_m):.1f} m")
    
    # Get matched pixel data
    matched_pixels = tessera_df.iloc[indices].reset_index(drop=True)
    
    # Build result DataFrame
    result = occurrences_df.copy().reset_index(drop=True)
    
    # Add snap distance
    result['snap_distance_m'] = distances_m
    
    # Add pixel location info
    result['pixel_lon'] = matched_pixels['pixel_lon'].values
    result['pixel_lat'] = matched_pixels['pixel_lat'].values
    result['pixel_x'] = matched_pixels['x'].values
    result['pixel_y'] = matched_pixels['y'].values
    result['pixel_crs'] = matched_pixels['crs'].values
    result['pixel_island'] = matched_pixels['island'].values
    
    # Add tile info
    result['tile_id'] = matched_pixels['tile_id'].values
    result['tile_lon'] = matched_pixels['tile_lon'].values
    result['tile_lat'] = matched_pixels['tile_lat'].values
    result['pixel_row'] = matched_pixels['pixel_row'].values
    result['pixel_col'] = matched_pixels['pixel_col'].values
    
    # Add all embedding bands
    band_cols = [c for c in matched_pixels.columns if c.startswith('band_')]
    for col in band_cols:
        result[col] = matched_pixels[col].values
    
    print(f"\n[INFO] Successfully assigned {len(result):,} occurrences to pixels")
    print(f"[INFO] Embedding dimensions: {len(band_cols)}")
    
    # Check if occurrence island matches pixel island
    if 'island' in result.columns:
        matches = result['island'] == result['pixel_island']
        print(f"\n[INFO] Island assignment verification:")
        print(f"       Occurrence island == Pixel island: {matches.sum():,} ({100*matches.mean():.1f}%)")
        if not matches.all():
            mismatches = result[~matches][['island', 'pixel_island']].drop_duplicates()
            print(f"       Mismatches found (occurrence → pixel):")
            for _, row in mismatches.iterrows():
                print(f"         {row['island']} → {row['pixel_island']}")
    
    return result


def save_results(result_df, outdir):
    """Save assigned occurrences with embeddings."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    os.makedirs(outdir, exist_ok=True)
    
    # Save combined file
    combined_path = os.path.join(outdir, "quercus_tomentella_with_embeddings.parquet")
    result_df.to_parquet(combined_path, index=False)
    print(f"[SAVED] {combined_path} ({len(result_df):,} records)")
    
    # Also save as CSV (without embeddings for readability)
    csv_cols = [c for c in result_df.columns if not c.startswith('band_')]
    csv_path = os.path.join(outdir, "quercus_tomentella_with_embeddings_metadata.csv")
    result_df[csv_cols].to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path} (metadata only, no embeddings)")
    
    # Save per-island files
    island_col = 'pixel_island'
    for island in sorted(result_df[island_col].unique()):
        island_df = result_df[result_df[island_col] == island]
        island_clean = str(island).replace(' ', '_').lower()
        island_path = os.path.join(outdir, f"quercus_tomentella_{island_clean}_with_embeddings.parquet")
        island_df.to_parquet(island_path, index=False)
        print(f"[SAVED] {island_path} ({len(island_df):,} records)")
    
    return combined_path


def print_summary(result_df):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nTotal occurrences with embeddings: {len(result_df):,}")
    
    # By island
    print(f"\nBy island:")
    island_col = 'pixel_island'
    for island in sorted(result_df[island_col].unique()):
        count = (result_df[island_col] == island).sum()
        print(f"  {island}: {count:,}")
    
    # By basisOfRecord
    if 'basisOfRecord' in result_df.columns:
        print(f"\nBy observation type:")
        for basis in sorted(result_df['basisOfRecord'].unique()):
            count = (result_df['basisOfRecord'] == basis).sum()
            print(f"  {basis}: {count:,}")
    
    # Cross-tabulation
    if 'basisOfRecord' in result_df.columns:
        print(f"\nIsland × Observation Type:")
        pivot = pd.crosstab(result_df[island_col], result_df['basisOfRecord'])
        print(pivot.to_string())


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("ASSIGN OCCURRENCES TO TESSERA PIXELS")
    print("="*60)
    
    args = parse_arguments()
    
    # 1. Load occurrences
    occurrences_df = load_occurrences(args.occurrences)
    
    # 2. Load TESSERA embeddings with lon/lat
    tessera_df = load_tessera_with_lonlat(args.parquet)
    
    # 3. Build spatial index
    tree = build_spatial_index(tessera_df)
    
    # 4. Assign occurrences to nearest pixels
    result_df = assign_to_nearest_pixels(occurrences_df, tessera_df, tree)
    
    # 5. Save results
    save_results(result_df, args.outdir)
    
    # 6. Print summary
    print_summary(result_df)
    
    print(f"\n{'='*60}")
    print("ASSIGNMENT COMPLETE")
    print(f"{'='*60}")
    print("\nDone!")


if __name__ == "__main__":
    main()
