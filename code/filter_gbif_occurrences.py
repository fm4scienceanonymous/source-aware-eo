#!/usr/bin/env python3
"""
filter_gbif_occurrences.py

Filter GBIF occurrence data to keep only records that:
    1. Have valid coordinates (decimalLatitude and decimalLongitude)
    2. Fall within the Channel Islands study area mask

Uses the same GeoPackage as extract_channel_islands_tessera.py for consistency.

Usage:
    python filter_gbif_occurrences.py \
        --gbif <path_to_gbif_occurrence_file> \
        --gpkg <path_to_study_area_geopackage> \
        --outdir ./out/species
"""

import argparse
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter GBIF occurrences to Channel Islands study area."
    )
    
    parser.add_argument(
        "--gbif",
        type=str,
        default=None,
        required=True,
        help="Path to GBIF occurrence file (tab-separated txt)"
    )
    parser.add_argument(
        "--gpkg",
        type=str,
        default=None,
        required=True,
        help="Path to GeoPackage with Channel Islands study area polygons"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./out/species",
        help="Output directory for filtered data"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gbif):
        parser.error(f"GBIF file not found: {args.gbif}")
    if not os.path.exists(args.gpkg):
        parser.error(f"GeoPackage not found: {args.gpkg}")
    
    return args


def load_gbif_data(gbif_path):
    """Load GBIF occurrence data."""
    print(f"\n{'='*60}")
    print("LOADING GBIF DATA")
    print(f"{'='*60}")
    print(f"[INFO] Loading: {os.path.basename(gbif_path)}")
    
    df = pd.read_csv(gbif_path, sep='\t', low_memory=False)
    print(f"[INFO] Total records: {len(df):,}")
    
    # Show basisOfRecord breakdown
    print(f"\n[INFO] basisOfRecord breakdown (all records):")
    for basis, count in df['basisOfRecord'].value_counts().items():
        print(f"       - {basis}: {count:,}")
    
    return df


def filter_with_coordinates(df):
    """Filter to records with valid coordinates."""
    print(f"\n{'='*60}")
    print("FILTERING: VALID COORDINATES")
    print(f"{'='*60}")
    
    has_coords = df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
    df_coords = df[has_coords].copy()
    
    print(f"[INFO] Records with coordinates: {len(df_coords):,}")
    print(f"[INFO] Records without coordinates: {(~has_coords).sum():,} (dropped)")
    
    return df_coords


def load_study_area(gpkg_path):
    """Load Channel Islands study area polygons."""
    print(f"\n{'='*60}")
    print("LOADING STUDY AREA")
    print(f"{'='*60}")
    print(f"[INFO] Loading: {os.path.basename(gpkg_path)}")
    
    gdf = gpd.read_file(gpkg_path)
    print(f"[INFO] Loaded {len(gdf)} island polygon(s)")
    
    # Ensure WGS84
    if gdf.crs is None:
        print("[WARNING] No CRS defined, assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        print(f"[INFO] Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
    
    # Find island name column
    name_cols = [c for c in gdf.columns if c.lower() in ['name', 'island', 'island_name', 'label']]
    island_col = name_cols[0] if name_cols else None
    
    if island_col:
        print(f"[INFO] Islands: {gdf[island_col].tolist()}")
    
    return gdf, island_col


def filter_to_study_area(df, island_gdf, island_col):
    """Filter occurrences to those within the island polygons."""
    print(f"\n{'='*60}")
    print("FILTERING: WITHIN STUDY AREA")
    print(f"{'='*60}")
    
    # Create GeoDataFrame from occurrence points
    print("[INFO] Creating point geometries...")
    geometry = [Point(lon, lat) for lon, lat in 
                zip(df['decimalLongitude'], df['decimalLatitude'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join to find points within islands
    print("[INFO] Performing spatial join (point-in-polygon)...")
    joined = gpd.sjoin(gdf_points, island_gdf, how='inner', predicate='within')
    
    print(f"[INFO] Records within Channel Islands: {len(joined):,}")
    print(f"[INFO] Records outside (dropped): {len(df) - len(joined):,}")
    
    # Extract island name
    if island_col:
        # Handle potential column name conflicts from sjoin
        if island_col in joined.columns:
            joined['island'] = joined[island_col]
        elif f'{island_col}_right' in joined.columns:
            joined['island'] = joined[f'{island_col}_right']
        else:
            joined['island'] = 'unknown'
    else:
        joined['island'] = 'unknown'
    
    # Show per-island breakdown
    print(f"\n[INFO] Records per island:")
    for island, count in joined['island'].value_counts().items():
        print(f"       - {island}: {count:,}")
    
    # Show basisOfRecord breakdown
    print(f"\n[INFO] basisOfRecord breakdown (filtered):")
    for basis, count in joined['basisOfRecord'].value_counts().items():
        print(f"       - {basis}: {count:,}")
    
    # Convert back to DataFrame (drop geometry for CSV output)
    result = pd.DataFrame(joined.drop(columns=['geometry', 'index_right'], errors='ignore'))
    
    return result


def save_filtered_data(df, outdir):
    """Save filtered data to CSV and parquet."""
    print(f"\n{'='*60}")
    print("SAVING FILTERED DATA")
    print(f"{'='*60}")
    
    os.makedirs(outdir, exist_ok=True)
    
    # Select key columns to keep (anonymized - no collector/institution info)
    key_cols = [
        'gbifID', 'basisOfRecord', 
        'decimalLatitude', 'decimalLongitude',
        'coordinateUncertaintyInMeters',
        'eventDate', 'year', 'month', 'day',
        'island',
        'species', 'scientificName'
    ]
    
    # Keep only columns that exist
    keep_cols = [c for c in key_cols if c in df.columns]
    df_output = df[keep_cols].copy()
    
    # Save as CSV
    csv_path = os.path.join(outdir, "quercus_tomentella_channel_islands.csv")
    df_output.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path} ({len(df_output):,} records)")
    
    # Save as parquet
    parquet_path = os.path.join(outdir, "quercus_tomentella_channel_islands.parquet")
    df_output.to_parquet(parquet_path, index=False)
    print(f"[SAVED] {parquet_path}")
    
    return csv_path, parquet_path


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("GBIF OCCURRENCE FILTER - CHANNEL ISLANDS")
    print("="*60)
    
    args = parse_arguments()
    
    # 1. Load GBIF data
    df = load_gbif_data(args.gbif)
    
    # 2. Filter to records with coordinates
    df_coords = filter_with_coordinates(df)
    
    if len(df_coords) == 0:
        print("[ERROR] No records with coordinates!")
        sys.exit(1)
    
    # 3. Load study area
    island_gdf, island_col = load_study_area(args.gpkg)
    
    # 4. Filter to study area
    df_filtered = filter_to_study_area(df_coords, island_gdf, island_col)
    
    if len(df_filtered) == 0:
        print("[ERROR] No records within Channel Islands!")
        sys.exit(1)
    
    # 5. Save filtered data
    save_filtered_data(df_filtered, args.outdir)
    
    print(f"\n{'='*60}")
    print("FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"[INFO] Final count: {len(df_filtered):,} occurrences")
    print("\nDone!")


if __name__ == "__main__":
    main()
