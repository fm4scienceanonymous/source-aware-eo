#!/usr/bin/env python3
"""
create_study_area_map.py

Create a clean map showing the Channel Islands in relation to California coast.
Highlights the 4 study islands.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# Paths - configure these for your environment
# Set via environment variables or modify defaults below
LAND_GPKG = os.environ.get('LAND_GPKG', './data/land_polygons.gpkg')
ISLANDS_GPKG = os.environ.get('ISLANDS_GPKG', './data/study_area.gpkg')
OUTPUT_DIR = './out/figures/study_area'

# Bounding box for the map (Southern California + Channel Islands)
# [min_lon, max_lon, min_lat, max_lat]
CALIFORNIA_BOUNDS = [-121.5, -117.0, 32.5, 35.0]
ISLANDS_BOUNDS = [-120.8, -118.2, 32.7, 34.2]

# Island color scheme (consistent with figures)
ISLAND_COLORS = {
    'san_clemente': '#B11226',   # Deep red
    'santa_catalina': '#2C4E9C', # Indigo/blue
    'santa_cruz': '#1F7A6D',     # Teal/green
    'santa_rosa': '#6A3D9A',     # Purple
}


def main():
    print("=" * 60)
    print("CREATING STUDY AREA MAP")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("Loading geographic data...")
    land = gpd.read_file(LAND_GPKG)
    islands = gpd.read_file(ISLANDS_GPKG)
    
    # Ensure CRS is WGS84
    if land.crs is None:
        land = land.set_crs(epsg=4326)
    else:
        land = land.to_crs(epsg=4326)
    
    if islands.crs is None:
        islands = islands.set_crs(epsg=4326)
    else:
        islands = islands.to_crs(epsg=4326)
    
    # Clip land to California region
    from shapely.geometry import box
    ca_box = box(*CALIFORNIA_BOUNDS)
    land_clipped = land.clip(ca_box)
    
    print(f"Study islands: {islands.shape[0]}")
    
    # ========================================
    # FIGURE 1: Overview Map (California + Islands)
    # ========================================
    print("Creating overview map...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot mainland California (light gray)
    land_clipped.plot(ax=ax, color='#d4d4d4', edgecolor='#888888', linewidth=0.5)
    
    # Plot study islands (highlighted in warm color)
    islands.plot(ax=ax, color='#e85d04', edgecolor='#9d0208', linewidth=1.5)
    
    # Set bounds
    ax.set_xlim(CALIFORNIA_BOUNDS[0], CALIFORNIA_BOUNDS[1])
    ax.set_ylim(CALIFORNIA_BOUNDS[2], CALIFORNIA_BOUNDS[3])
    
    # Clean styling - no axes, no labels
    ax.set_axis_off()
    
    # Add subtle box around islands region
    rect = mpatches.Rectangle(
        (ISLANDS_BOUNDS[0], ISLANDS_BOUNDS[2]),
        ISLANDS_BOUNDS[1] - ISLANDS_BOUNDS[0],
        ISLANDS_BOUNDS[3] - ISLANDS_BOUNDS[2],
        fill=False, edgecolor='#333333', linewidth=1.5, linestyle='--'
    )
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overview_map.png'), dpi=300, 
                bbox_inches='tight', facecolor='white', transparent=False, metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'overview_map.pdf'), 
                bbox_inches='tight', facecolor='white', metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'overview_map_transparent.png'), dpi=300,
                bbox_inches='tight', transparent=True, metadata={})
    plt.close()
    print("  Saved overview_map.png/pdf")
    
    # ========================================
    # FIGURE 1b: Overview Map WITH LABELS
    # ========================================
    print("Creating labeled overview map...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot mainland California (light gray)
    land_clipped.plot(ax=ax, color='#d4d4d4', edgecolor='#888888', linewidth=0.5)
    
    # Plot each island with its own color
    for island_name, color in ISLAND_COLORS.items():
        island_geom = islands[islands['island_name'] == island_name]
        if len(island_geom) > 0:
            # Darken the edge color
            edge_color = '#' + ''.join([hex(max(0, int(color[i:i+2], 16) - 40))[2:].zfill(2) for i in (1, 3, 5)])
            island_geom.plot(ax=ax, color=color, edgecolor=edge_color, linewidth=1.5)
    
    # Set bounds
    ax.set_xlim(CALIFORNIA_BOUNDS[0], CALIFORNIA_BOUNDS[1])
    ax.set_ylim(CALIFORNIA_BOUNDS[2], CALIFORNIA_BOUNDS[3])
    
    # Clean styling - no axes
    ax.set_axis_off()
    
    # Add subtle box around islands region
    rect = mpatches.Rectangle(
        (ISLANDS_BOUNDS[0], ISLANDS_BOUNDS[2]),
        ISLANDS_BOUNDS[1] - ISLANDS_BOUNDS[0],
        ISLANDS_BOUNDS[3] - ISLANDS_BOUNDS[2],
        fill=False, edgecolor='#333333', linewidth=1.5, linestyle='--'
    )
    ax.add_patch(rect)
    
    # Island label positions - lines point to island centers
    # All labels inside the bounding box
    # Format: (island_center_lon, island_center_lat, label_lon, label_lat, name)
    island_labels = [
        (-118.48, 32.90, -118.72, 32.76, 'San Clemente'),    # center of San Clemente
        (-118.42, 33.39, -118.68, 33.22, 'Santa Catalina'),  # center of Santa Catalina
        (-119.75, 33.98, -119.50, 33.82, 'Santa Cruz'),      # center of Santa Cruz
        (-120.08, 33.96, -120.35, 33.80, 'Santa Rosa'),      # center of Santa Rosa
    ]
    
    for island_lon, island_lat, label_lon, label_lat, name in island_labels:
        # Draw thin line from label to island
        ax.plot([label_lon, island_lon], [label_lat, island_lat], 
                color='#666666', linewidth=0.8, linestyle='-', zorder=1)
        # Add label with subtle background
        ax.text(label_lon, label_lat, name, fontsize=8, fontweight='medium',
                ha='center', va='center', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#cccccc', linewidth=0.5, alpha=0.9),
                zorder=2)
    
    # Add minimal compass (North arrow) in ocean area (bottom-left)
    compass_x = CALIFORNIA_BOUNDS[0] + 0.4
    compass_y = CALIFORNIA_BOUNDS[2] + 0.5
    arrow_len = 0.3
    
    # Arrow pointing north
    ax.annotate('', xy=(compass_x, compass_y + arrow_len), 
                xytext=(compass_x, compass_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.text(compass_x, compass_y + arrow_len + 0.08, 'N', 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    
    # Add location context label (bottom, left of land)
    ax.text(-118.3, CALIFORNIA_BOUNDS[2] + 0.08, 
            'California Channel Islands, USA', 
            ha='right', va='bottom', fontsize=7, fontstyle='italic', 
            color='#666666')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overview_map_labeled.png'), dpi=300, 
                bbox_inches='tight', facecolor='white', transparent=False, metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'overview_map_labeled.pdf'), 
                bbox_inches='tight', facecolor='white', metadata={})
    plt.close()
    print("  Saved overview_map_labeled.png/pdf")
    
    # ========================================
    # FIGURE 2: Islands Detail (zoomed in)
    # ========================================
    print("Creating islands detail map...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Clip land to islands region for context
    islands_box = box(*ISLANDS_BOUNDS)
    land_islands = land.clip(islands_box)
    
    # Plot any mainland visible in the islands view (light gray)
    land_islands.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa', linewidth=0.5)
    
    # Plot study islands with distinct color
    islands.plot(ax=ax, color='#e85d04', edgecolor='#9d0208', linewidth=2)
    
    # Set bounds
    ax.set_xlim(ISLANDS_BOUNDS[0], ISLANDS_BOUNDS[1])
    ax.set_ylim(ISLANDS_BOUNDS[2], ISLANDS_BOUNDS[3])
    
    # Clean styling
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'islands_detail.png'), dpi=300,
                bbox_inches='tight', facecolor='white', transparent=False, metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'islands_detail.pdf'),
                bbox_inches='tight', facecolor='white', metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'islands_detail_transparent.png'), dpi=300,
                bbox_inches='tight', transparent=True, metadata={})
    plt.close()
    print("  Saved islands_detail.png/pdf")
    
    # ========================================
    # FIGURE 3: Minimal Overview (just shapes)
    # ========================================
    print("Creating minimal overview...")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot mainland (very light)
    land_clipped.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
    
    # Plot study islands (bold)
    islands.plot(ax=ax, color='#1a1a1a', edgecolor='none')
    
    ax.set_xlim(CALIFORNIA_BOUNDS[0], CALIFORNIA_BOUNDS[1])
    ax.set_ylim(CALIFORNIA_BOUNDS[2], CALIFORNIA_BOUNDS[3])
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'minimal_overview.png'), dpi=300,
                bbox_inches='tight', facecolor='white', transparent=False, metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'minimal_overview_transparent.png'), dpi=300,
                bbox_inches='tight', transparent=True, metadata={})
    plt.close()
    print("  Saved minimal_overview.png")
    
    # ========================================
    # FIGURE 4: High contrast for presentation
    # ========================================
    print("Creating high-contrast version...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # White background, dark land
    land_clipped.plot(ax=ax, color='#2d2d2d', edgecolor='#1a1a1a', linewidth=0.5)
    
    # Bright orange islands
    islands.plot(ax=ax, color='#ff6b35', edgecolor='#ffffff', linewidth=2)
    
    ax.set_xlim(CALIFORNIA_BOUNDS[0], CALIFORNIA_BOUNDS[1])
    ax.set_ylim(CALIFORNIA_BOUNDS[2], CALIFORNIA_BOUNDS[3])
    ax.set_axis_off()
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'high_contrast.png'), dpi=300,
                bbox_inches='tight', facecolor='white', metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'high_contrast_transparent.png'), dpi=300,
                bbox_inches='tight', transparent=True, metadata={})
    plt.close()
    print("  Saved high_contrast.png")
    
    print()
    print("=" * 60)
    print("MAP CREATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
