#!/usr/bin/env python3
"""
extract_channel_islands_tessera.py

A simple, well-commented script to extract GeoTessera (TESSERA) embedding tiles
for the four largest Channel Islands. Ocean pixels are masked out using a
study-area polygon.

This script uses **Method 2** from the GeoTessera API:
  1) Compute a bounding box from the study area
  2) Query the registry for tiles intersecting the bounding box
  3) Fetch embeddings via fetch_embeddings()
  4) Loop over returned tiles, mask ocean, and save outputs

OUTPUT FORMATS:
  - NPY: Masked 3D arrays (bands × height × width) per tile + JSON metadata
  - CSV: One row per LAND pixel with columns:
         tile_id, year, tile_lon, tile_lat, pixel_row, pixel_col, x, y, crs,
         island, band_0, band_1, ..., band_127
  - BOTH: Both NPY and CSV outputs (default)

Usage:
    python extract_channel_islands_tessera.py \
        --gpkg /path/to/channel_islands_4_study_area.gpkg \
        --year 2024 \
        --outdir ./out

Optional arguments:
    --layer <layer_name>       Layer name in the GeoPackage (default: first layer)
    --bands "0,1,2"            Comma-separated band indices to keep (default: all 128)
    --mask-value nan|zero      Value for masked ocean pixels (default: nan)
    --max-tiles N              Maximum number of tiles to process (for debugging)
    --output-format npy|csv|both   Output format (default: both)
    --island-column <col_name> Column in GeoPackage with island names (auto-detects if not set)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

# GeoTessera library for accessing precomputed embeddings
from geotessera import GeoTessera


# =============================================================================
# BLOCK 0: ARGUMENT PARSING
# =============================================================================
# Parse command-line arguments to configure the script's behavior.
# This makes the script flexible and reusable for different study areas/years.

def parse_arguments():
    """
    Parse and validate command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Extract GeoTessera embeddings for Channel Islands with ocean masking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with defaults
    python extract_channel_islands_tessera.py --gpkg islands.gpkg --year 2024 --outdir ./out
    
    # With band subsetting and zero masking
    python extract_channel_islands_tessera.py --gpkg islands.gpkg --year 2024 --outdir ./out \\
        --bands "0,1,2,3" --mask-value zero
    
    # Debug mode with tile limit
    python extract_channel_islands_tessera.py --gpkg islands.gpkg --year 2024 --outdir ./out \\
        --max-tiles 2
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--gpkg",
        type=str,
        required=True,
        help="Path to the GeoPackage (.gpkg) containing the study area polygons."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year for which to fetch GeoTessera embeddings (e.g., 2024)."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory where .npy files and metadata will be saved."
    )
    
    # Optional arguments
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name within the GeoPackage. If not specified, uses the first layer."
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help="Comma-separated band indices to extract (e.g., '0,1,2'). Default: all 128 bands."
    )
    parser.add_argument(
        "--mask-value",
        type=str,
        choices=["nan", "zero"],
        default="nan",
        help="Value to assign to masked (ocean) pixels. Default: nan."
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximum number of tiles to process (for debugging). Default: no limit."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["npy", "csv", "both"],
        default="both",
        help="Output format: 'npy' (arrays), 'csv' (spreadsheet), or 'both'. Default: both."
    )
    parser.add_argument(
        "--island-column",
        type=str,
        default=None,
        help="Column name in GeoPackage containing island labels (e.g., 'name', 'island_id'). "
             "If not specified, will auto-detect or use A, B, C, D based on feature order."
    )
    
    args = parser.parse_args()
    
    # Validate that the input file exists
    if not os.path.exists(args.gpkg):
        parser.error(f"Input GeoPackage not found: {args.gpkg}")
    
    # Parse band indices if provided
    if args.bands is not None:
        try:
            args.band_indices = [int(b.strip()) for b in args.bands.split(",")]
            print(f"[INFO] Will extract bands: {args.band_indices}")
        except ValueError:
            parser.error(f"Invalid band specification: {args.bands}. Use comma-separated integers.")
    else:
        args.band_indices = None  # Will use all bands
        print("[INFO] Will extract all 128 bands.")
    
    return args


# =============================================================================
# BLOCK 1: LOAD STUDY AREA
# =============================================================================
# Load the study area polygons from a GeoPackage file.
# The polygons define which land areas we want embeddings for (i.e., the islands).

def load_study_area(gpkg_path, layer_name=None, island_column=None):
    """
    Load study area polygons from a GeoPackage.
    
    This function:
    - Reads the GeoPackage using GeoPandas
    - Validates that a CRS is defined (raises error if not)
    - Reprojects to EPSG:4326 (WGS84) if necessary (required for GeoTessera)
    - Assigns island labels (A, B, C, D) based on column or auto-generates them
    - Dissolves all polygons into a single geometry for efficient intersection tests
    
    Args:
        gpkg_path (str): Path to the GeoPackage file.
        layer_name (str, optional): Specific layer to read. If None, reads first layer.
        island_column (str, optional): Column containing island names/labels.
    
    Returns:
        tuple: (dissolved_geometry, original_gdf, island_labels)
            - dissolved_geometry: A single shapely geometry (union of all polygons)
            - original_gdf: The original GeoDataFrame in EPSG:4326
            - island_labels: List of island label strings in feature order
    
    Raises:
        ValueError: If the GeoPackage has no CRS defined.
    """
    print(f"\n{'='*60}")
    print("BLOCK 1: LOAD STUDY AREA")
    print(f"{'='*60}")
    print(f"[INFO] Loading GeoPackage: {os.path.basename(gpkg_path)}")
    
    # Read the GeoPackage
    # If layer_name is None, GeoPandas reads the first layer by default
    if layer_name:
        print(f"[INFO] Reading layer: {layer_name}")
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
    else:
        print("[INFO] Reading first layer (no layer specified).")
        gdf = gpd.read_file(gpkg_path)
    
    print(f"[INFO] Loaded {len(gdf)} feature(s).")
    print(f"[INFO] Columns available: {list(gdf.columns)}")
    
    # CRITICAL: Check that the GeoPackage has a CRS defined
    # Without a CRS, we cannot correctly compute bounding boxes or reproject
    if gdf.crs is None:
        raise ValueError(
            "ERROR: The GeoPackage has no CRS defined!\n"
            "Please assign a CRS to the file before running this script.\n"
            "You can do this in QGIS: Layer > Set CRS of Layer(s).\n"
            "For Channel Islands, the CRS is likely EPSG:4326 (WGS84) or a UTM zone."
        )
    
    print(f"[INFO] Original CRS: {gdf.crs}")
    
    # Reproject to EPSG:4326 (WGS84) if necessary
    # GeoTessera uses WGS84 coordinates (lat/lon in degrees)
    if gdf.crs.to_epsg() != 4326:
        print(f"[INFO] Reprojecting from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)
        print("[INFO] Reprojection complete.")
    else:
        print("[INFO] Data is already in EPSG:4326.")
    
    # Determine island labels
    # Priority: 1) user-specified column, 2) auto-detect 'name'/'island' column, 3) generate A,B,C,D
    island_labels = []
    if island_column and island_column in gdf.columns:
        island_labels = gdf[island_column].tolist()
        print(f"[INFO] Using island labels from column '{island_column}': {island_labels}")
    else:
        # Try to auto-detect a name column
        name_columns = [c for c in gdf.columns if c.lower() in ['name', 'island', 'island_name', 'label']]
        if name_columns:
            island_labels = gdf[name_columns[0]].tolist()
            print(f"[INFO] Auto-detected island labels from column '{name_columns[0]}': {island_labels}")
        else:
            # Generate labels A, B, C, D based on feature order
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:len(gdf)]
            island_labels = labels
            print(f"[INFO] No label column found. Assigning labels by order: {island_labels}")
    
    # Store labels in the GeoDataFrame for later use
    gdf['_island_label'] = island_labels
    
    # Dissolve all polygons into a single geometry
    # This simplifies intersection tests - we only need to check against one geometry
    # instead of multiple polygons per island
    print("[INFO] Dissolving polygons into single geometry...")
    dissolved = gdf.dissolve()
    dissolved_geometry = dissolved.geometry.iloc[0]
    
    print(f"[INFO] Study area geometry type: {dissolved_geometry.geom_type}")
    print(f"[INFO] Study area bounds: {dissolved_geometry.bounds}")
    
    return dissolved_geometry, gdf, island_labels


# =============================================================================
# BLOCK 2: COMPUTE BOUNDING BOX
# =============================================================================
# Compute the bounding box of the study area.
# This is used to query the GeoTessera registry for candidate tiles.

def compute_bounding_box(geometry):
    """
    Compute the bounding box of a geometry in (min_lon, min_lat, max_lon, max_lat) format.
    
    GeoTessera expects bounding boxes in this specific order for Method 2.
    
    Args:
        geometry: A shapely geometry object.
    
    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat) bounding box.
    """
    print(f"\n{'='*60}")
    print("BLOCK 2: COMPUTE BOUNDING BOX")
    print(f"{'='*60}")
    
    # Shapely's bounds property returns (minx, miny, maxx, maxy)
    # For geographic coordinates: (min_lon, min_lat, max_lon, max_lat)
    minx, miny, maxx, maxy = geometry.bounds
    bbox = (minx, miny, maxx, maxy)
    
    print(f"[INFO] Bounding box (min_lon, min_lat, max_lon, max_lat):")
    print(f"       {bbox}")
    print(f"[INFO] Longitude range: {minx:.4f}° to {maxx:.4f}°")
    print(f"[INFO] Latitude range:  {miny:.4f}° to {maxy:.4f}°")
    
    return bbox


# =============================================================================
# BLOCK 3: QUERY GEOTESSERA REGISTRY (METHOD 2)
# =============================================================================
# Query the GeoTessera registry to find all tiles that fall within the bounding box.
# This is the first step of Method 2 from the GeoTessera API.

def query_registry(bbox, year):
    """
    Query the GeoTessera registry for tiles intersecting the bounding box.
    
    This uses Method 2 from the GeoTessera API:
    - Initialize the GeoTessera client
    - Use registry.load_blocks_for_region() to find candidate tiles
    
    The registry returns tiles based only on the bounding box, so we'll need
    to further filter by actual polygon intersection in the next block.
    
    Args:
        bbox (tuple): (min_lon, min_lat, max_lon, max_lat) bounding box.
        year (int): Year for which to fetch tiles.
    
    Returns:
        tuple: (gt_client, tiles_to_fetch)
            - gt_client: The GeoTessera client instance (reused for fetching)
            - tiles_to_fetch: Object containing tile information from registry
    """
    print(f"\n{'='*60}")
    print("BLOCK 3: QUERY GEOTESSERA REGISTRY (METHOD 2)")
    print(f"{'='*60}")
    
    # Initialize the GeoTessera client
    # This connects to the GeoTessera service and prepares for data access
    print("[INFO] Initializing GeoTessera client...")
    gt = GeoTessera()
    print("[INFO] Client initialized successfully.")
    
    # Query the registry for tiles in our bounding box
    # The registry knows which tiles are available for each year
    print(f"[INFO] Querying registry for tiles in bbox for year {year}...")
    tiles_to_fetch = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
    
    # Handle the returned object defensively
    # The registry may return a DataFrame, list, or other iterable
    if hasattr(tiles_to_fetch, '__len__'):
        num_tiles = len(tiles_to_fetch)
    else:
        # If it's a generator or similar, we can't get length without consuming it
        # In this case, we'll count during processing
        num_tiles = "unknown"
    
    print(f"[INFO] Registry returned {num_tiles} candidate tile(s).")
    
    # IMPORTANT: Convert to list so we can iterate multiple times
    # (once for filtering, once for fetching)
    # If it's already a DataFrame, this is fine; if it's a generator, this prevents exhaustion
    if hasattr(tiles_to_fetch, 'iterrows'):
        # It's a DataFrame - keep as is (can be iterated multiple times)
        tiles_list = tiles_to_fetch
    else:
        # Convert generator/iterator to list
        tiles_list = list(tiles_to_fetch)
        print(f"[INFO] Converted to list: {len(tiles_list)} tiles")
    
    return gt, tiles_list


# =============================================================================
# BLOCK 4: FILTER TILES BY POLYGON INTERSECTION
# =============================================================================
# Filter the candidate tiles to keep only those that actually intersect the
# island polygons. The bounding box query may include tiles that are entirely
# ocean - we want to skip those.

def create_tile_footprint(tile_lon, tile_lat):
    """
    Create a shapely box representing a GeoTessera tile footprint.
    
    GeoTessera tiles are on a 0.1° × 0.1° grid, with the tile center at (lon, lat).
    The tile covers:
        lon ∈ [lon − 0.05, lon + 0.05]
        lat ∈ [lat − 0.05, lat + 0.05]
    
    Args:
        tile_lon (float): Tile center longitude.
        tile_lat (float): Tile center latitude.
    
    Returns:
        shapely.geometry.Polygon: The tile footprint as a box.
    """
    # GeoTessera uses 0.1° tiles, so half-width is 0.05°
    half_size = 0.05
    
    return box(
        tile_lon - half_size,  # min_lon
        tile_lat - half_size,  # min_lat
        tile_lon + half_size,  # max_lon
        tile_lat + half_size   # max_lat
    )


def filter_tiles_by_intersection(tiles_to_fetch, study_geometry, max_tiles=None):
    """
    Filter tiles to keep only those that intersect the study area polygons.
    
    The registry query returns all tiles in the bounding box, but many may be
    entirely ocean. This function checks each tile's footprint against the
    dissolved island geometry.
    
    Args:
        tiles_to_fetch: Tile information from the registry (may be DataFrame or list).
        study_geometry: Dissolved shapely geometry of the study area.
        max_tiles (int, optional): Maximum number of tiles to keep (for debugging).
    
    Returns:
        list: List of tiles that intersect the study area.
              Each tile is a dict with 'year', 'lon', 'lat', and 'original' keys.
    """
    print(f"\n{'='*60}")
    print("BLOCK 4: FILTER TILES BY POLYGON INTERSECTION")
    print(f"{'='*60}")
    
    intersecting_tiles = []
    total_checked = 0
    
    # Handle different return types from the registry
    # It may be a pandas DataFrame or a list-like object
    if hasattr(tiles_to_fetch, 'iterrows'):
        # It's a DataFrame - iterate over rows
        print("[INFO] Processing tiles from DataFrame...")
        for idx, row in tiles_to_fetch.iterrows():
            # Extract tile coordinates from DataFrame columns
            # Column names may vary; try common conventions
            if 'lon' in row and 'lat' in row:
                tile_lon, tile_lat = row['lon'], row['lat']
            elif 'longitude' in row and 'latitude' in row:
                tile_lon, tile_lat = row['longitude'], row['latitude']
            elif 'tile_lon' in row and 'tile_lat' in row:
                tile_lon, tile_lat = row['tile_lon'], row['tile_lat']
            else:
                # Try positional access if column names don't match
                tile_lon, tile_lat = row.iloc[0], row.iloc[1]
            
            year = row.get('year', row.iloc[2] if len(row) > 2 else None)
            
            total_checked += 1
            tile_footprint = create_tile_footprint(tile_lon, tile_lat)
            
            if tile_footprint.intersects(study_geometry):
                intersecting_tiles.append({
                    'year': year,
                    'lon': tile_lon,
                    'lat': tile_lat,
                    'original': row
                })
                print(f"  [✓] Tile ({tile_lat:.2f}, {tile_lon:.2f}) intersects study area")
            
            # Apply max_tiles limit if specified
            if max_tiles and len(intersecting_tiles) >= max_tiles:
                print(f"[INFO] Reached max_tiles limit ({max_tiles}). Stopping filter.")
                break
    else:
        # Assume it's a list-like or iterable
        print("[INFO] Processing tiles from list/iterable...")
        for tile_info in tiles_to_fetch:
            # Extract coordinates based on the structure
            if hasattr(tile_info, '__getitem__'):
                # It might be a tuple or list: (year, lon, lat) or similar
                if len(tile_info) >= 3:
                    # Assume (lon, lat, year) or (year, lon, lat) format
                    # GeoTessera typically uses (lon, lat) ordering
                    year, tile_lon, tile_lat = tile_info[0], tile_info[1], tile_info[2]
                else:
                    tile_lon, tile_lat = tile_info[0], tile_info[1]
                    year = None
            elif hasattr(tile_info, 'lon') and hasattr(tile_info, 'lat'):
                tile_lon, tile_lat = tile_info.lon, tile_info.lat
                year = getattr(tile_info, 'year', None)
            else:
                print(f"  [WARNING] Unknown tile format: {type(tile_info)}")
                continue
            
            total_checked += 1
            tile_footprint = create_tile_footprint(tile_lon, tile_lat)
            
            if tile_footprint.intersects(study_geometry):
                intersecting_tiles.append({
                    'year': year,
                    'lon': tile_lon,
                    'lat': tile_lat,
                    'original': tile_info
                })
                print(f"  [✓] Tile ({tile_lat:.2f}, {tile_lon:.2f}) intersects study area")
            
            # Apply max_tiles limit if specified
            if max_tiles and len(intersecting_tiles) >= max_tiles:
                print(f"[INFO] Reached max_tiles limit ({max_tiles}). Stopping filter.")
                break
    
    print(f"\n[INFO] Checked {total_checked} tiles, {len(intersecting_tiles)} intersect the study area.")
    
    if len(intersecting_tiles) == 0:
        print("[WARNING] No tiles intersect the study area! Check your input polygon.")
    
    return intersecting_tiles


# =============================================================================
# BLOCK 5 & 6: FETCH EMBEDDINGS + RASTERIZE POLYGON + MASK OCEAN PIXELS
# =============================================================================
# For each tile, rasterize the study area polygon to create a mask,
# then apply the mask to set ocean pixels to NaN or zero.

def create_land_mask(geometry, gdf_wgs84, tile_crs, tile_transform, tile_shape):
    """
    Create a binary land mask for a tile by rasterizing the study area polygon.
    
    This function:
    1. Reprojects the study area polygons from WGS84 to the tile's CRS
    2. Rasterizes the polygons to match the tile's grid
    3. Returns a boolean mask where True = land, False = ocean
    
    Args:
        geometry: The dissolved study area geometry (in WGS84).
        gdf_wgs84: The original GeoDataFrame in WGS84.
        tile_crs: The CRS of the embedding tile.
        tile_transform: The affine transform of the embedding tile.
        tile_shape: The (height, width) shape of the embedding tile.
    
    Returns:
        numpy.ndarray: Boolean mask with shape (height, width).
                       True where land exists, False for ocean.
    """
    # Create a GeoDataFrame with the dissolved geometry for reprojection
    dissolved_gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    
    # Reproject to the tile's CRS
    # This is necessary because the rasterization must match the tile's coordinate system
    dissolved_reprojected = dissolved_gdf.to_crs(tile_crs)
    
    # Extract the reprojected geometry
    reprojected_geometry = dissolved_reprojected.geometry.iloc[0]
    
    # Rasterize the polygon
    # rasterio.features.rasterize creates a 2D array where pixels inside the geometry
    # have value 1, and pixels outside have value 0
    mask = rasterize(
        shapes=[reprojected_geometry],
        out_shape=tile_shape,
        transform=tile_transform,
        fill=0,      # Ocean pixels get 0
        default_value=1,  # Land pixels get 1
        dtype=np.uint8
    )
    
    # Convert to boolean mask (True = land, False = ocean)
    land_mask = mask.astype(bool)
    
    return land_mask


def create_island_label_raster(gdf_wgs84, island_labels, tile_crs, tile_transform, tile_shape):
    """
    Create a raster where each pixel contains an island label index (1, 2, 3, 4).
    
    This allows us to identify which island each pixel belongs to.
    Ocean pixels get value 0 (no island).
    
    Args:
        gdf_wgs84: GeoDataFrame with individual island polygons in WGS84.
        island_labels: List of island label strings (e.g., ['A', 'B', 'C', 'D']).
        tile_crs: The CRS of the embedding tile.
        tile_transform: The affine transform of the embedding tile.
        tile_shape: The (height, width) shape of the embedding tile.
    
    Returns:
        tuple: (label_raster, label_mapping)
            - label_raster: 2D array where each pixel has island index (0 = ocean, 1-4 = islands)
            - label_mapping: Dict mapping index to label string {1: 'A', 2: 'B', ...}
    """
    # Reproject all island polygons to the tile's CRS
    gdf_reprojected = gdf_wgs84.to_crs(tile_crs)
    
    # Create shapes list for rasterization: (geometry, value) pairs
    # Each island gets a unique integer value (1, 2, 3, 4...)
    shapes = []
    label_mapping = {0: "ocean"}  # 0 is reserved for ocean/no-data
    
    for idx, (_, row) in enumerate(gdf_reprojected.iterrows()):
        island_idx = idx + 1  # Start from 1 (0 is ocean)
        shapes.append((row.geometry, island_idx))
        label_mapping[island_idx] = island_labels[idx] if idx < len(island_labels) else f"island_{idx}"
    
    # Rasterize with island indices
    # If polygons overlap, the later one wins (shouldn't happen for distinct islands)
    label_raster = rasterize(
        shapes=shapes,
        out_shape=tile_shape,
        transform=tile_transform,
        fill=0,  # Ocean pixels get 0
        dtype=np.uint8
    )
    
    return label_raster, label_mapping


def apply_mask_to_embedding(embedding_array, land_mask, mask_value="nan", band_indices=None):
    """
    Apply the land mask to an embedding array, setting ocean pixels to NaN or zero.
    
    Embedding arrays have shape (bands, height, width) or (height, width, bands).
    This function handles both cases and applies the mask across all bands.
    
    Args:
        embedding_array: The embedding numpy array.
        land_mask: Boolean mask with True for land pixels.
        mask_value: "nan" or "zero" - what to set ocean pixels to.
        band_indices: Optional list of band indices to extract. If None, keeps all.
    
    Returns:
        numpy.ndarray: Masked embedding array with selected bands.
    """
    # Determine array shape and band dimension
    # GeoTessera typically returns (bands, height, width) format
    if len(embedding_array.shape) == 3:
        if embedding_array.shape[0] == 128 or embedding_array.shape[0] < embedding_array.shape[1]:
            # Shape is (bands, height, width)
            bands_first = True
            n_bands, height, width = embedding_array.shape
        else:
            # Shape might be (height, width, bands)
            bands_first = False
            height, width, n_bands = embedding_array.shape
    else:
        raise ValueError(f"Unexpected embedding shape: {embedding_array.shape}")
    
    # Convert to float32 for NaN support if using NaN masking
    if mask_value == "nan":
        masked_array = embedding_array.astype(np.float32)
    else:
        masked_array = embedding_array.copy()
    
    # Apply band subsetting if specified
    if band_indices is not None:
        if bands_first:
            masked_array = masked_array[band_indices, :, :]
        else:
            masked_array = masked_array[:, :, band_indices]
    
    # Apply the mask to set ocean pixels to the specified value
    # We need to broadcast the 2D mask across all bands
    if bands_first:
        # Shape is (bands, height, width)
        # Expand mask to (1, height, width) for broadcasting
        mask_3d = ~land_mask[np.newaxis, :, :]  # Invert: True where ocean
        if mask_value == "nan":
            masked_array = np.where(mask_3d, np.nan, masked_array)
        else:
            masked_array = np.where(mask_3d, 0, masked_array)
    else:
        # Shape is (height, width, bands)
        mask_3d = ~land_mask[:, :, np.newaxis]  # Invert: True where ocean
        if mask_value == "nan":
            masked_array = np.where(mask_3d, np.nan, masked_array)
        else:
            masked_array = np.where(mask_3d, 0, masked_array)
    
    return masked_array


def pixel_to_coords(row, col, transform):
    """
    Convert pixel row/col to coordinates in the tile's CRS.
    
    Uses rasterio's transform.xy() to compute the CENTER of each pixel.
    
    Note: The returned (x, y) are in the tile's CRS (likely UTM or similar),
    NOT necessarily lon/lat. The CRS is stored in the CSV for reference.
    
    Args:
        row: Pixel row index (0-based).
        col: Pixel column index (0-based).
        transform: Affine transform from the tile.
    
    Returns:
        tuple: (x, y) coordinates in the tile's CRS.
    """
    # Use rasterio's standard method for pixel center coordinates
    # This is equivalent to: x = a*(col+0.5) + b*(row+0.5) + c
    #                        y = d*(col+0.5) + e*(row+0.5) + f
    x, y = rasterio.transform.xy(transform, row, col, offset='center')
    return x, y


def tile_to_csv_rows(embedding_array, label_raster, label_mapping, tile_id, year,
                     tile_lon, tile_lat, transform, tile_crs, band_indices=None):
    """
    Convert a tile's embedding data to CSV rows (one row per LAND pixel).
    
    This flattens the 3D embedding array into a list of dictionaries,
    where each dictionary represents one pixel with its coordinates,
    island label, and all band values.
    
    Args:
        embedding_array: The embedding array (bands, height, width).
        label_raster: 2D array with island indices (0 = ocean).
        label_mapping: Dict mapping index to island label.
        tile_id: Unique identifier for this tile.
        year: Year of the embedding.
        tile_lon, tile_lat: Tile center coordinates.
        transform: Affine transform for coordinate conversion.
        tile_crs: CRS of the tile.
        band_indices: Optional list of band indices used.
    
    Returns:
        list: List of dicts, each representing one land pixel.
    """
    rows = []
    
    # Determine array shape
    if embedding_array.shape[0] == 128 or embedding_array.shape[0] < embedding_array.shape[1]:
        bands_first = True
        n_bands, height, width = embedding_array.shape
    else:
        bands_first = False
        height, width, n_bands = embedding_array.shape
    
    # Determine band names
    if band_indices is not None:
        band_names = [f"band_{i}" for i in band_indices]
    else:
        band_names = [f"band_{i}" for i in range(n_bands)]
    
    # Iterate over all pixels
    for row_idx in range(height):
        for col_idx in range(width):
            # Get island label (0 = ocean, skip these)
            island_idx = label_raster[row_idx, col_idx]
            if island_idx == 0:
                continue  # Skip ocean pixels
            
            island_label = label_mapping.get(island_idx, f"unknown_{island_idx}")
            
            # Get pixel coordinates in tile CRS
            x, y = pixel_to_coords(row_idx, col_idx, transform)
            
            # Get embedding values for this pixel
            if bands_first:
                pixel_values = embedding_array[:, row_idx, col_idx]
            else:
                pixel_values = embedding_array[row_idx, col_idx, :]
            
            # Build row dict
            pixel_row = {
                "tile_id": tile_id,
                "year": year,
                "tile_lon": tile_lon,
                "tile_lat": tile_lat,
                "pixel_row": row_idx,
                "pixel_col": col_idx,
                "x": x,
                "y": y,
                "crs": str(tile_crs),
                "island": island_label
            }
            
            # Add band values
            for band_name, value in zip(band_names, pixel_values):
                pixel_row[band_name] = float(value)
            
            rows.append(pixel_row)
    
    return rows


# =============================================================================
# BLOCK 7: SAVE OUTPUTS
# =============================================================================
# Save the masked embeddings as .npy files with accompanying .json metadata.

def save_tile_outputs(masked_embedding, year, tile_lon, tile_lat, crs, transform,
                      outdir, band_indices, mask_value):
    """
    Save a masked embedding tile to disk with metadata.
    
    Creates two files:
    - {tile_id}.npy: The masked embedding array
    - {tile_id}.json: Metadata about the tile
    
    Args:
        masked_embedding: The masked numpy array to save.
        year: Year of the embedding.
        tile_lon, tile_lat: Tile center coordinates.
        crs: CRS of the tile.
        transform: Affine transform of the tile.
        outdir: Output directory path.
        band_indices: List of band indices used, or None if all bands.
        mask_value: "nan" or "zero" - the masking policy used.
    
    Returns:
        dict: Information about the saved files for the index.
    """
    # Create a unique tile ID based on coordinates
    # Use a format that sorts nicely: lat_lon with sign indicators
    lat_str = f"{'N' if tile_lat >= 0 else 'S'}{abs(tile_lat):.2f}"
    lon_str = f"{'E' if tile_lon >= 0 else 'W'}{abs(tile_lon):.2f}"
    tile_id = f"tile_{year}_{lat_str}_{lon_str}"
    
    # Sanitize the tile_id for filesystem (replace dots with underscores)
    tile_id = tile_id.replace(".", "_")
    
    # File paths
    npy_path = os.path.join(outdir, f"{tile_id}.npy")
    json_path = os.path.join(outdir, f"{tile_id}.json")
    
    # Save the numpy array
    np.save(npy_path, masked_embedding)
    
    # Compute tile bounds in WGS84
    tile_bounds_wgs84 = {
        "min_lon": tile_lon - 0.05,
        "max_lon": tile_lon + 0.05,
        "min_lat": tile_lat - 0.05,
        "max_lat": tile_lat + 0.05
    }
    
    # Convert affine transform to a serializable format
    # Affine transform has: a, b, c, d, e, f (or named: a, b, xoff, d, e, yoff)
    if hasattr(transform, 'to_gdal'):
        transform_params = list(transform.to_gdal())
    else:
        # Manual extraction
        transform_params = [transform.a, transform.b, transform.c,
                           transform.d, transform.e, transform.f]
    
    # Create metadata dictionary
    metadata = {
        "tile_id": tile_id,
        "year": int(year) if year is not None else None,
        "tile_center": {
            "lon": float(tile_lon),
            "lat": float(tile_lat)
        },
        "tile_bounds_wgs84": tile_bounds_wgs84,
        "embedding_shape": list(masked_embedding.shape),
        "embedding_dtype": str(masked_embedding.dtype),
        "crs": str(crs),
        "affine_transform": transform_params,
        "bands_used": band_indices if band_indices else "all_128",
        "masking_policy": mask_value,
        "npy_file": f"{tile_id}.npy"
    }
    
    # Save metadata JSON
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  [SAVED] {npy_path}")
    print(f"  [SAVED] {json_path}")
    
    return {
        "tile_id": tile_id,
        "npy_file": f"{tile_id}.npy",
        "json_file": f"{tile_id}.json",
        "tile_center": {"lon": float(tile_lon), "lat": float(tile_lat)},
        "year": int(year) if year is not None else None
    }


def save_index_json(saved_tiles, outdir, args):
    """
    Save a top-level index.json listing all processed tiles.
    
    This file provides a convenient overview of all outputs and can be used
    to programmatically load the tiles later.
    
    Args:
        saved_tiles: List of dicts with information about each saved tile.
        outdir: Output directory path.
        args: The original command-line arguments for reference.
    """
    index_path = os.path.join(outdir, "index.json")
    
    index_data = {
        "description": "GeoTessera embeddings for Channel Islands study area",
        "input_gpkg": os.path.basename(args.gpkg),
        "year": args.year,
        "bands_used": args.band_indices if args.band_indices else "all_128",
        "masking_policy": args.mask_value,
        "total_tiles": len(saved_tiles),
        "tiles": saved_tiles
    }
    
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"\n[SAVED] Index file: {index_path}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """
    Main entry point for the script.
    
    Orchestrates the entire workflow:
    1. Parse arguments
    2. Load study area
    3. Compute bounding box
    4. Query registry
    5. Filter tiles
    6. Fetch and process embeddings
    7. Save outputs
    """
    print("\n" + "="*60)
    print("CHANNEL ISLANDS GEOTESSERA EMBEDDING EXTRACTOR")
    print("="*60)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[INFO] Output directory: {args.outdir}")
    print(f"[INFO] Output format: {args.output_format}")
    
    # BLOCK 1: Load study area from GeoPackage
    study_geometry, gdf_wgs84, island_labels = load_study_area(
        args.gpkg, args.layer, args.island_column
    )
    
    # BLOCK 2: Compute bounding box for registry query
    bbox = compute_bounding_box(study_geometry)
    
    # BLOCK 3: Query GeoTessera registry for tiles in the bounding box
    gt_client, tiles_to_fetch = query_registry(bbox, args.year)
    
    # BLOCK 4: Filter tiles to keep only those intersecting the study area
    intersecting_tiles = filter_tiles_by_intersection(
        tiles_to_fetch, 
        study_geometry,
        max_tiles=args.max_tiles
    )
    
    if len(intersecting_tiles) == 0:
        print("\n[ERROR] No tiles to process. Exiting.")
        sys.exit(1)
    
    # BLOCK 5 & 6 & 7: Fetch embeddings, mask, and save
    # We process tiles as they come from the generator to minimize memory usage
    print(f"\n{'='*60}")
    print("BLOCKS 5-7: FETCH, MASK, AND SAVE EMBEDDINGS")
    print(f"{'='*60}")
    
    # Create a set of (lon, lat) for tiles we want to keep
    wanted_tiles = {(t['lon'], t['lat']) for t in intersecting_tiles}
    
    saved_tiles = []
    all_csv_rows = []  # Accumulate CSV rows across all tiles
    processed_count = 0
    
    # Fetch embeddings - this returns a generator
    embeddings = gt_client.fetch_embeddings(tiles_to_fetch)
    
    for year, tile_lon, tile_lat, embedding_array, crs, transform in embeddings:
        # Check if this tile is in our wanted set
        if (tile_lon, tile_lat) not in wanted_tiles:
            # Skip tiles that don't intersect the study area
            continue
        
        processed_count += 1
        print(f"\n[PROCESSING] Tile {processed_count}/{len(intersecting_tiles)}: "
              f"({tile_lat:.2f}, {tile_lon:.2f})")
        print(f"  Embedding shape: {embedding_array.shape}")
        print(f"  CRS: {crs}")
        
        # Determine tile dimensions for masking
        # Embedding shape is typically (bands, height, width)
        if embedding_array.shape[0] == 128 or embedding_array.shape[0] < embedding_array.shape[1]:
            height, width = embedding_array.shape[1], embedding_array.shape[2]
        else:
            height, width = embedding_array.shape[0], embedding_array.shape[1]
        
        tile_shape = (height, width)
        print(f"  Tile dimensions: {height} x {width} pixels")
        
        # Create tile ID for this tile
        lat_str = f"{'N' if tile_lat >= 0 else 'S'}{abs(tile_lat):.2f}"
        lon_str = f"{'E' if tile_lon >= 0 else 'W'}{abs(tile_lon):.2f}"
        tile_id = f"tile_{year}_{lat_str}_{lon_str}".replace(".", "_")
        
        # BLOCK 6: Create land mask and island label raster
        print("  Creating land mask...")
        land_mask = create_land_mask(
            study_geometry, 
            gdf_wgs84, 
            crs, 
            transform, 
            tile_shape
        )
        
        land_pixels = np.sum(land_mask)
        total_pixels = land_mask.size
        land_percent = 100 * land_pixels / total_pixels
        print(f"  Land pixels: {land_pixels}/{total_pixels} ({land_percent:.1f}%)")
        
        # Create island label raster (needed for CSV output)
        print("  Creating island label raster...")
        label_raster, label_mapping = create_island_label_raster(
            gdf_wgs84, island_labels, crs, transform, tile_shape
        )
        print(f"  Island labels found: {[v for k, v in label_mapping.items() if k > 0]}")
        
        # Apply mask to embedding (for NPY output)
        print(f"  Applying mask (ocean → {args.mask_value})...")
        masked_embedding = apply_mask_to_embedding(
            embedding_array,
            land_mask,
            mask_value=args.mask_value,
            band_indices=args.band_indices
        )
        print(f"  Masked embedding shape: {masked_embedding.shape}")
        
        # BLOCK 7: Save outputs based on format
        tile_info = None
        
        # Save NPY format if requested
        if args.output_format in ["npy", "both"]:
            print("  Saving NPY outputs...")
            tile_info = save_tile_outputs(
                masked_embedding=masked_embedding,
                year=year,
                tile_lon=tile_lon,
                tile_lat=tile_lat,
                crs=crs,
                transform=transform,
                outdir=args.outdir,
                band_indices=args.band_indices,
                mask_value=args.mask_value
            )
            saved_tiles.append(tile_info)
        
        # Generate CSV rows if requested
        if args.output_format in ["csv", "both"]:
            print("  Generating CSV rows for land pixels...")
            csv_rows = tile_to_csv_rows(
                embedding_array=embedding_array,
                label_raster=label_raster,
                label_mapping=label_mapping,
                tile_id=tile_id,
                year=year,
                tile_lon=tile_lon,
                tile_lat=tile_lat,
                transform=transform,
                tile_crs=crs,
                band_indices=args.band_indices
            )
            all_csv_rows.extend(csv_rows)
            print(f"  Generated {len(csv_rows)} CSV rows (land pixels only)")
        
        # Check if we've reached max_tiles
        if args.max_tiles and processed_count >= args.max_tiles:
            print(f"\n[INFO] Reached max_tiles limit ({args.max_tiles}). Stopping.")
            break
    
    # Save the index file (for NPY output)
    if args.output_format in ["npy", "both"]:
        save_index_json(saved_tiles, args.outdir, args)
    
    # Save the combined CSV file
    if args.output_format in ["csv", "both"] and all_csv_rows:
        csv_path = os.path.join(args.outdir, "channel_islands_embeddings.csv")
        print(f"\n[INFO] Writing CSV file with {len(all_csv_rows)} total land pixels...")
        
        # Get column names from first row
        fieldnames = list(all_csv_rows[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        
        print(f"[SAVED] {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"[INFO] Total tiles processed: {processed_count}")
    print(f"[INFO] Output directory: {args.outdir}")
    print(f"[INFO] Files created:")
    
    if args.output_format in ["npy", "both"]:
        print(f"       - {len(saved_tiles)} .npy embedding files")
        print(f"       - {len(saved_tiles)} .json metadata files")
        print(f"       - 1 index.json")
    
    if args.output_format in ["csv", "both"]:
        print(f"       - 1 channel_islands_embeddings.csv ({len(all_csv_rows)} rows)")
        
        # Print island breakdown
        if all_csv_rows:
            island_counts = {}
            for row in all_csv_rows:
                island = row.get('island', 'unknown')
                island_counts[island] = island_counts.get(island, 0) + 1
            print(f"\n[INFO] Pixels per island:")
            for island, count in sorted(island_counts.items()):
                print(f"       - {island}: {count:,} pixels")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
