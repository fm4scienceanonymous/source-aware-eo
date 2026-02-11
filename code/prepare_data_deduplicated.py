#!/usr/bin/env python3
"""
prepare_data_deduplicated.py

Prepare training and test data matching the paper's Methods section:

    1. Deduplicate occurrences to one row per unique pixel
    2. Create two binary labels: y_human (HO), y_herb (PS)
    3. For each island AND each data source INDEPENDENTLY:
       - 70/30 train-test split on that data source's positive pixels
    4. Negatives sampled from P_i \ L_i^{Data Source}:
       - HO model negatives: all island pixels EXCEPT HO-positives
       - PS model negatives: all island pixels EXCEPT PS-positives
       (A PS-positive pixel CAN be a negative for the HO model, and vice versa)
    5. |N_i| = 10 * |L_i,train| (10:1 negative:positive ratio)

Usage:
    python prepare_data_deduplicated.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyproj import Transformer

SEED = 42
TRAIN_RATIO = 0.70
NEGATIVE_RATIO = 10  # 10× negatives per positive

# Paths
SPECIES_DIR = './out/species'
PARQUET_DIR = './out/parquet'
OUTPUT_DIR = './out/species'

ISLANDS = ['san_clemente', 'santa_catalina', 'santa_cruz', 'santa_rosa']


def create_pixel_id(row):
    """Create unique pixel identifier."""
    return f"{row['tile_id']}_{row['pixel_row']}_{row['pixel_col']}"


def deduplicate_and_label(island):
    """
    Load occurrences for an island, deduplicate by pixel, create y_human and y_herb labels.
    
    Returns DataFrame with one row per unique pixel and both labels.
    """
    filepath = os.path.join(SPECIES_DIR, f'quercus_tomentella_{island}_with_embeddings.parquet')
    df = pd.read_parquet(filepath)
    
    # Create pixel_id
    df['pixel_id'] = df.apply(create_pixel_id, axis=1)
    
    # Group by pixel and aggregate labels
    # y_human = 1 if any HUMAN_OBSERVATION, else 0
    # y_herb = 1 if any PRESERVED_SPECIMEN, else 0
    
    pixel_groups = df.groupby('pixel_id').agg({
        'basisOfRecord': lambda x: list(x),
        # Keep first value for other columns
        'tile_id': 'first',
        'pixel_row': 'first',
        'pixel_col': 'first',
        'pixel_lon': 'first',
        'pixel_lat': 'first',
        'pixel_x': 'first',
        'pixel_y': 'first',
        'pixel_crs': 'first',
        'tile_lon': 'first',
        'tile_lat': 'first',
        **{f'band_{i}': 'first' for i in range(128)}
    }).reset_index()
    
    # Create binary labels
    pixel_groups['y_human'] = pixel_groups['basisOfRecord'].apply(
        lambda x: 1 if 'HUMAN_OBSERVATION' in x else 0
    )
    pixel_groups['y_herb'] = pixel_groups['basisOfRecord'].apply(
        lambda x: 1 if 'PRESERVED_SPECIMEN' in x else 0
    )
    
    # Drop the aggregated basisOfRecord list
    pixel_groups = pixel_groups.drop(columns=['basisOfRecord'])
    
    return pixel_groups


def load_all_island_pixels(island):
    """
    Load ALL TESSERA pixels for an island (P_i in paper notation).
    """
    parquet_path = os.path.join(PARQUET_DIR, f'{island}.parquet')
    df = pd.read_parquet(parquet_path)
    
    # Create pixel IDs
    df['pixel_id'] = df.apply(create_pixel_id, axis=1)
    
    # Add lon/lat if needed
    if 'pixel_lon' not in df.columns:
        lons = np.zeros(len(df))
        lats = np.zeros(len(df))
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
    
    # Rename columns to match positive data
    df = df.rename(columns={
        'x': 'pixel_x',
        'y': 'pixel_y',
        'crs': 'pixel_crs'
    })
    
    return df


def main():
    print("=" * 70)
    print("PREPARE DATA (Methods Section Implementation)")
    print("=" * 70)
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Negative ratio: {NEGATIVE_RATIO}×")
    print(f"Seed: {SEED}")
    print()
    print("Key: Negatives for HO model = P_i \\ L_i^HO")
    print("     Negatives for PS model = P_i \\ L_i^PS")
    print("     (PS-positive pixels CAN be HO negatives, and vice versa)")
    print()
    
    # Create output directories
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    summary = []
    
    for island in ISLANDS:
        print(f"\n{'='*70}")
        print(f"ISLAND: {island.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Deduplicate and create labels for occurrence pixels
        print("\n[Step 1] Deduplicating occurrences by pixel...")
        positive_df = deduplicate_and_label(island)
        
        n_pixels = len(positive_df)
        n_human = positive_df['y_human'].sum()
        n_herb = positive_df['y_herb'].sum()
        n_both = ((positive_df['y_human'] == 1) & (positive_df['y_herb'] == 1)).sum()
        
        print(f"  Unique occurrence pixels: {n_pixels}")
        print(f"  L_i^HO (y_human=1): {n_human}")
        print(f"  L_i^PS (y_herb=1): {n_herb}")
        print(f"  Both=1: {n_both}")
        
        # Step 2: Load ALL island pixels (P_i)
        print("\n[Step 2] Loading all island pixels (P_i)...")
        all_pixels_df = load_all_island_pixels(island)
        print(f"  Total island pixels |P_i|: {len(all_pixels_df):,}")
        
        # Create sets for quick lookup
        ho_positive_pixels = set(positive_df[positive_df['y_human'] == 1]['pixel_id'])
        ps_positive_pixels = set(positive_df[positive_df['y_herb'] == 1]['pixel_id'])
        
        band_cols = [f'band_{i}' for i in range(128)]
        common_cols = ['pixel_id', 'tile_id', 'pixel_row', 'pixel_col',
                       'pixel_lon', 'pixel_lat', 'pixel_x', 'pixel_y', 'pixel_crs',
                       'tile_lon', 'tile_lat'] + band_cols
        
        # Step 3: For each data source INDEPENDENTLY, split and sample
        print("\n[Step 3] Processing each data source independently...")
        
        for label_name, label_col, positive_pixel_set in [
            ('human', 'y_human', ho_positive_pixels), 
            ('herb', 'y_herb', ps_positive_pixels)
        ]:
            print(f"\n  --- {label_name.upper()} ({label_col}) ---")
            
            # Get positive pixels for this label type
            label_positive_df = positive_df[positive_df[label_col] == 1].copy()
            n_total_pos = len(label_positive_df)
            print(f"    |L_i^{label_name.upper()}|: {n_total_pos}")
            
            # INDEPENDENT 70/30 split for this data source
            label_seed = SEED + ISLANDS.index(island) * 100 + (['human', 'herb'].index(label_name)) * 10
            
            train_pos_label, test_pos_label = train_test_split(
                label_positive_df,
                train_size=TRAIN_RATIO,
                random_state=label_seed
            )
            
            n_train_pos = len(train_pos_label)
            n_test_pos = len(test_pos_label)
            actual_ratio = n_train_pos / n_total_pos if n_total_pos > 0 else 0
            
            print(f"    L_i,train^{label_name.upper()}: {n_train_pos} ({actual_ratio*100:.1f}%)")
            print(f"    L_i,test^{label_name.upper()}: {n_test_pos} ({(1-actual_ratio)*100:.1f}%)")
            
            # Negative sampling: N_i = P_i \ L_i^{Data Source}
            # For HO model: exclude HO-positive pixels (PS-positives CAN be negatives)
            # For PS model: exclude PS-positive pixels (HO-positives CAN be negatives)
            negative_pool_df = all_pixels_df[~all_pixels_df['pixel_id'].isin(positive_pixel_set)].copy()
            print(f"    |P_i \\ L_i^{label_name.upper()}| (negative pool): {len(negative_pool_df):,}")
            
            # Sample 10x negatives
            n_negatives = n_train_pos * NEGATIVE_RATIO
            print(f"    Negatives to sample (10 × {n_train_pos}): {n_negatives}")
            
            seed_offset = ISLANDS.index(island) * 10 + (['human', 'herb'].index(label_name))
            if len(negative_pool_df) >= n_negatives:
                negatives = negative_pool_df.sample(n=n_negatives, random_state=SEED + seed_offset)
            else:
                print(f"    [WARNING] Only {len(negative_pool_df)} available, using all")
                negatives = negative_pool_df.sample(n=len(negative_pool_df), random_state=SEED + seed_offset)
            
            print(f"    Negatives sampled: {len(negatives)}")
            
            # Check how many negatives are positive for the OTHER label
            neg_pixel_ids = set(negatives['pixel_id'])
            if label_name == 'human':
                other_pos_in_neg = len(neg_pixel_ids & ps_positive_pixels)
                print(f"    (Of these, {other_pos_in_neg} are PS-positive)")
            else:
                other_pos_in_neg = len(neg_pixel_ids & ho_positive_pixels)
                print(f"    (Of these, {other_pos_in_neg} are HO-positive)")
            
            # Verify no train/test overlap
            train_pixels = set(train_pos_label['pixel_id'])
            test_pixels = set(test_pos_label['pixel_id'])
            overlap = train_pixels & test_pixels
            if len(overlap) > 0:
                print(f"    [ERROR] Train/Test overlap: {len(overlap)} pixels!")
            else:
                print(f"    [OK] No train/test pixel overlap ✓")
            
            # Prepare training data
            train_pos_label = train_pos_label.copy()
            train_pos_label['label'] = 1
            neg_df = negatives.copy()
            neg_df['label'] = 0
            
            # Select common columns
            train_cols = [c for c in common_cols if c in train_pos_label.columns]
            neg_cols = [c for c in common_cols if c in neg_df.columns]
            final_cols = list(set(train_cols) & set(neg_cols))
            
            # Combine and shuffle
            train_data = pd.concat([
                train_pos_label[final_cols + ['label']],
                neg_df[final_cols + ['label']]
            ], ignore_index=True)
            train_data = train_data.sample(frac=1, random_state=SEED).reset_index(drop=True)
            
            # Save training data
            train_filename = f'quercus_tomentella_{island}_{label_name}_train.parquet'
            train_data.to_parquet(os.path.join(train_dir, train_filename), index=False)
            print(f"    [SAVED] train/{train_filename} ({len(train_data)} rows)")
            
            # Save test data (positives only)
            test_pos_label = test_pos_label.copy()
            test_pos_label['label'] = 1
            test_cols = [c for c in final_cols if c in test_pos_label.columns] + ['label']
            test_data = test_pos_label[test_cols].copy()
            
            test_filename = f'quercus_tomentella_{island}_{label_name}_test.parquet'
            test_data.to_parquet(os.path.join(test_dir, test_filename), index=False)
            print(f"    [SAVED] test/{test_filename} ({len(test_data)} rows)")
            
            summary.append({
                'island': island,
                'label': label_name,
                'total_pos': n_total_pos,
                'train_pos': n_train_pos,
                'train_neg': len(negatives),
                'test_pos': n_test_pos,
                'train_ratio': f"{actual_ratio*100:.1f}%"
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    # Final verification
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION")
    print(f"{'='*70}")
    
    all_ok = True
    print(f"\n{'Island':<18} {'Label':<8} {'Train':>8} {'Test':>8} {'Total':>8} {'Ratio':>10}")
    print("-" * 60)
    
    for island in ISLANDS:
        for label_name in ['human', 'herb']:
            train_file = os.path.join(train_dir, f'quercus_tomentella_{island}_{label_name}_train.parquet')
            test_file = os.path.join(test_dir, f'quercus_tomentella_{island}_{label_name}_test.parquet')
            
            train_df = pd.read_parquet(train_file)
            test_df = pd.read_parquet(test_file)
            
            n_train_pos = (train_df['label'] == 1).sum()
            n_test_pos = len(test_df)
            n_total = n_train_pos + n_test_pos
            ratio = n_train_pos / n_total if n_total > 0 else 0
            
            is_ok = abs(ratio - TRAIN_RATIO) <= 0.03
            if not is_ok:
                all_ok = False
            
            island_name = island.replace('_', ' ').title()
            status = "✓" if is_ok else "⚠"
            print(f"{island_name:<18} {label_name:<8} {n_train_pos:>8} {n_test_pos:>8} {n_total:>8} {ratio*100:>9.1f}% {status}")
    
    # Leakage check
    print(f"\n{'='*70}")
    print("LEAKAGE CHECK")
    print(f"{'='*70}")
    
    leakage_ok = True
    for island in ISLANDS:
        for label_name in ['human', 'herb']:
            train_file = os.path.join(train_dir, f'quercus_tomentella_{island}_{label_name}_train.parquet')
            test_file = os.path.join(test_dir, f'quercus_tomentella_{island}_{label_name}_test.parquet')
            
            train_df = pd.read_parquet(train_file)
            test_df = pd.read_parquet(test_file)
            
            train_pixels = set(train_df['pixel_id'])
            test_pixels = set(test_df['pixel_id'])
            overlap = train_pixels & test_pixels
            
            status = '✓ OK' if len(overlap) == 0 else f'✗ OVERLAP: {len(overlap)}'
            if len(overlap) > 0:
                leakage_ok = False
            print(f"{island} {label_name}: {status}")
    
    print(f"\n{'='*70}")
    print(f"DATA PREPARATION {'COMPLETE ✓' if (all_ok and leakage_ok) else 'ISSUES DETECTED'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
