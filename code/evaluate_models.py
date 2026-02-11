#!/usr/bin/env python3
"""
evaluate_models.py

Evaluate 8 logistic regression models using Recall@k% area metric.

Two evaluation modes:
1. Same-Type: HO models tested on HO positives, PS models on PS positives
2. Combined: All 8 models tested on HO + PS positives
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
MODEL_DIR = './out/models'
TRAIN_DIR = './out/species/train'
TEST_DIR = './out/species/test'
TESSERA_DIR = './out/parquet'
OUTPUT_DIR = './out/evaluation'

ISLANDS = ['san_clemente', 'santa_catalina', 'santa_cruz', 'santa_rosa']
ISLAND_NAMES = {
    'san_clemente': 'San Clemente',
    'santa_catalina': 'Santa Catalina', 
    'santa_cruz': 'Santa Cruz',
    'santa_rosa': 'Santa Rosa'
}

K_VALUES = [1, 2, 5, 10, 20, 50]  # Recall@k%


def load_models():
    """Load all 8 trained models."""
    models = {}
    for island in ISLANDS:
        for label_type in ['human', 'herb']:
            model_file = f'logreg_{island}_{label_type}.pkl'
            model_path = os.path.join(MODEL_DIR, model_file)
            with open(model_path, 'rb') as f:
                models[(island, label_type)] = pickle.load(f)
    return models


def load_training_pixels():
    """Load pixel IDs used in each training set."""
    train_pixels = {}
    for island in ISLANDS:
        for label_type in ['human', 'herb']:
            filename = f'quercus_tomentella_{island}_{label_type}_train.parquet'
            filepath = os.path.join(TRAIN_DIR, filename)
            df = pd.read_parquet(filepath)
            train_pixels[(island, label_type)] = set(df['pixel_id'].unique())
    return train_pixels


def load_test_positives():
    """Load positive pixel IDs from each test set."""
    test_positives = {}
    for island in ISLANDS:
        for label_type in ['human', 'herb']:
            filename = f'quercus_tomentella_{island}_{label_type}_test.parquet'
            filepath = os.path.join(TEST_DIR, filename)
            df = pd.read_parquet(filepath)
            # Only keep positives (label=1)
            pos_df = df[df['label'] == 1]
            test_positives[(island, label_type)] = set(pos_df['pixel_id'].unique())
    return test_positives


def load_island_pixels(island):
    """Load all TESSERA pixels for an island."""
    filename = f'{island}.parquet'
    filepath = os.path.join(TESSERA_DIR, filename)
    df = pd.read_parquet(filepath)
    
    # Create pixel_id if not present
    if 'pixel_id' not in df.columns:
        df['pixel_id'] = df['tile_id'] + '_' + df['pixel_row'].astype(str) + '_' + df['pixel_col'].astype(str)
    
    return df


def calculate_recall_at_k(scores, is_positive, k_percent):
    """
    Calculate Recall@k% area.
    
    Args:
        scores: Array of model scores for each pixel
        is_positive: Boolean array indicating positive pixels
        k_percent: The k% threshold
    
    Returns:
        Recall@k% value
    """
    n_total = len(scores)
    n_positives = is_positive.sum()
    
    if n_positives == 0:
        return 0.0
    
    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_positives = is_positive[sorted_indices]
    
    # Top k% of pixels
    k_count = max(1, int(np.ceil(n_total * k_percent / 100)))
    
    # Count positives in top k%
    positives_in_top_k = sorted_positives[:k_count].sum()
    
    recall = positives_in_top_k / n_positives
    return recall


def evaluate_model_on_island(model_data, island_df, exclude_pixels, positive_pixels):
    """
    Evaluate a model on an island.
    
    Args:
        model_data: Dict containing model and metadata
        island_df: DataFrame with all island pixels
        exclude_pixels: Set of pixel_ids to exclude
        positive_pixels: Set of pixel_ids that are positives
    
    Returns:
        Dict with recall@k% for each k value
    """
    model = model_data['model']
    band_cols = model_data['feature_names']
    
    # Filter out excluded pixels
    eval_df = island_df[~island_df['pixel_id'].isin(exclude_pixels)].copy()
    
    if len(eval_df) == 0:
        return {k: 0.0 for k in K_VALUES}, 0, 0
    
    # Get features
    X = eval_df[band_cols].values
    
    # Get model scores (probability of positive class)
    scores = model.predict_proba(X)[:, 1]
    
    # Mark positives
    is_positive = eval_df['pixel_id'].isin(positive_pixels).values
    
    # Calculate recall at each k%
    results = {}
    for k in K_VALUES:
        results[k] = calculate_recall_at_k(scores, is_positive, k)
    
    n_eval = len(eval_df)
    n_pos = is_positive.sum()
    
    return results, n_eval, n_pos


def run_evaluation():
    """Run full evaluation."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load everything
    print("\nLoading models...")
    models = load_models()
    print(f"  Loaded {len(models)} models")
    
    print("\nLoading training pixel IDs...")
    train_pixels = load_training_pixels()
    
    print("\nLoading test positive pixel IDs...")
    test_positives = load_test_positives()
    
    # Print test positive counts
    print("\n  Test positives per island:")
    for island in ISLANDS:
        ho = len(test_positives[(island, 'human')])
        ps = len(test_positives[(island, 'herb')])
        print(f"    {ISLAND_NAMES[island]}: {ho} HO, {ps} PS, {ho + ps} combined")
    
    # Load island pixels
    print("\nLoading TESSERA pixels...")
    island_dfs = {}
    for island in ISLANDS:
        island_dfs[island] = load_island_pixels(island)
        print(f"  {ISLAND_NAMES[island]}: {len(island_dfs[island]):,} pixels")
    
    # ========================================
    # LEAKAGE CHECK
    # ========================================
    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)
    
    all_ok = True
    
    for island in ISLANDS:
        for label_type in ['human', 'herb']:
            train_pix = train_pixels[(island, label_type)]
            test_pix = test_positives[(island, label_type)]
            overlap = train_pix & test_pix
            
            if len(overlap) > 0:
                print(f"  [ERROR] {island}/{label_type}: {len(overlap)} pixels in both train and test!")
                all_ok = False
            else:
                print(f"  [OK] {island}/{label_type}: No overlap between train and test positives")
    
    if not all_ok:
        print("\n[FATAL] Leakage detected! Aborting.")
        return
    
    print("\n✓ No leakage detected")
    
    # ========================================
    # MODE 1: SAME-TYPE EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("MODE 1: SAME-TYPE EVALUATION")
    print("=" * 70)
    
    # Results storage
    mode1_ho_results = {}  # (train_island, test_island) -> {k: recall}
    mode1_ps_results = {}
    
    # HO models on HO test positives
    print("\n--- Human Observation Models ---")
    for train_island in ISLANDS:
        model_data = models[(train_island, 'human')]
        print(f"\n  Model: {ISLAND_NAMES[train_island]} HO")
        
        for test_island in ISLANDS:
            # Exclude training pixels for the evaluated HO model (train_island)
            exclude = train_pixels[(train_island, 'human')]
            # Positives are HO test positives
            positives = test_positives[(test_island, 'human')]
            
            results, n_eval, n_pos = evaluate_model_on_island(
                model_data, island_dfs[test_island], exclude, positives
            )
            
            mode1_ho_results[(train_island, test_island)] = results
            print(f"    Test {ISLAND_NAMES[test_island]}: {n_pos} positives in {n_eval:,} pixels, R@10%={results[10]:.3f}")
    
    # PS models on PS test positives
    print("\n--- Preserved Specimen Models ---")
    for train_island in ISLANDS:
        model_data = models[(train_island, 'herb')]
        print(f"\n  Model: {ISLAND_NAMES[train_island]} PS")
        
        for test_island in ISLANDS:
            # Exclude training pixels for the evaluated PS model (train_island)
            exclude = train_pixels[(train_island, 'herb')]
            # Positives are PS test positives
            positives = test_positives[(test_island, 'herb')]
            
            results, n_eval, n_pos = evaluate_model_on_island(
                model_data, island_dfs[test_island], exclude, positives
            )
            
            mode1_ps_results[(train_island, test_island)] = results
            print(f"    Test {ISLAND_NAMES[test_island]}: {n_pos} positives in {n_eval:,} pixels, R@10%={results[10]:.3f}")
    
    # ========================================
    # MODE 2: COMBINED EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("MODE 2: COMBINED EVALUATION (HO + PS positives)")
    print("=" * 70)
    
    mode2_results = {}  # (train_island, label_type, test_island) -> {k: recall}
    
    for train_island in ISLANDS:
        for label_type in ['human', 'herb']:
            model_data = models[(train_island, label_type)]
            type_name = 'HO' if label_type == 'human' else 'PS'
            print(f"\n  Model: {ISLAND_NAMES[train_island]} {type_name}")
            
            for test_island in ISLANDS:
                # Exclude training pixels for the evaluated model (train_island, label_type)
                exclude = train_pixels[(train_island, label_type)]
                
                # Positives are BOTH HO and PS test positives combined
                positives = test_positives[(test_island, 'human')] | test_positives[(test_island, 'herb')]
                
                results, n_eval, n_pos = evaluate_model_on_island(
                    model_data, island_dfs[test_island], exclude, positives
                )
                
                mode2_results[(train_island, label_type, test_island)] = results
                print(f"    Test {ISLAND_NAMES[test_island]}: {n_pos} positives in {n_eval:,} pixels, R@10%={results[10]:.3f}")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save as CSVs
    save_results_to_csv(mode1_ho_results, mode1_ps_results, mode2_results)
    
    # ========================================
    # VISUALIZATIONS
    # ========================================
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_visualizations(mode1_ho_results, mode1_ps_results, mode2_results)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


def save_results_to_csv(mode1_ho, mode1_ps, mode2):
    """Save results to CSV files."""
    
    # Mode 1 HO (4x4)
    for k in K_VALUES:
        rows = []
        for train_island in ISLANDS:
            row = {'train_island': ISLAND_NAMES[train_island]}
            for test_island in ISLANDS:
                row[ISLAND_NAMES[test_island]] = mode1_ho[(train_island, test_island)][k]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f'mode1_HO_recall_at_{k}pct.csv'), index=False)
    
    # Mode 1 PS (4x4)
    for k in K_VALUES:
        rows = []
        for train_island in ISLANDS:
            row = {'train_island': ISLAND_NAMES[train_island]}
            for test_island in ISLANDS:
                row[ISLAND_NAMES[test_island]] = mode1_ps[(train_island, test_island)][k]
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f'mode1_PS_recall_at_{k}pct.csv'), index=False)
    
    # Mode 2 Combined (8x4)
    for k in K_VALUES:
        rows = []
        for train_island in ISLANDS:
            for label_type in ['human', 'herb']:
                type_name = 'HO' if label_type == 'human' else 'PS'
                row = {'model': f'{ISLAND_NAMES[train_island]} {type_name}'}
                for test_island in ISLANDS:
                    row[ISLAND_NAMES[test_island]] = mode2[(train_island, label_type, test_island)][k]
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f'mode2_combined_recall_at_{k}pct.csv'), index=False)
    
    print(f"  Saved CSV files to {OUTPUT_DIR}")


def create_visualizations(mode1_ho, mode1_ps, mode2):
    """Create visualization plots."""
    
    # Color scheme
    colors = {
        'san_clemente': '#E74C3C',
        'santa_catalina': '#3498DB',
        'santa_cruz': '#2ECC71',
        'santa_rosa': '#9B59B6'
    }
    
    # ========================================
    # FIGURE 1: Mode 1 HO Heatmaps
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mode 1: HO→HO Recall@k%', fontsize=14, fontweight='bold')
    
    for idx, k in enumerate(K_VALUES):
        ax = axes[idx // 3, idx % 3]
        
        # Build matrix
        matrix = np.zeros((4, 4))
        for i, train_island in enumerate(ISLANDS):
            for j, test_island in enumerate(ISLANDS):
                matrix[i, j] = mode1_ho[(train_island, test_island)][k]
        
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([ISLAND_NAMES[i][:8] for i in ISLANDS], rotation=45, ha='right')
        ax.set_yticklabels([ISLAND_NAMES[i][:8] for i in ISLANDS])
        ax.set_xlabel('Test Island')
        ax.set_ylabel('Train Island')
        ax.set_title(f'Recall@{k}%')
        
        # Add values
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mode1_HO_heatmaps.png'), dpi=150, bbox_inches='tight', metadata={})
    plt.close()
    print("  Saved mode1_HO_heatmaps.png")
    
    # ========================================
    # FIGURE 2: Mode 1 PS Heatmaps
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mode 1: PS→PS Recall@k%', fontsize=14, fontweight='bold')
    
    for idx, k in enumerate(K_VALUES):
        ax = axes[idx // 3, idx % 3]
        
        # Build matrix
        matrix = np.zeros((4, 4))
        for i, train_island in enumerate(ISLANDS):
            for j, test_island in enumerate(ISLANDS):
                matrix[i, j] = mode1_ps[(train_island, test_island)][k]
        
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([ISLAND_NAMES[i][:8] for i in ISLANDS], rotation=45, ha='right')
        ax.set_yticklabels([ISLAND_NAMES[i][:8] for i in ISLANDS])
        ax.set_xlabel('Test Island')
        ax.set_ylabel('Train Island')
        ax.set_title(f'Recall@{k}%')
        
        # Add values
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mode1_PS_heatmaps.png'), dpi=150, bbox_inches='tight', metadata={})
    plt.close()
    print("  Saved mode1_PS_heatmaps.png")
    
    # ========================================
    # FIGURE 3: Mode 2 Combined Heatmap (8x4 for R@10%)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Build 8x4 matrix
    matrix = np.zeros((8, 4))
    row_labels = []
    for i, train_island in enumerate(ISLANDS):
        for j, label_type in enumerate(['human', 'herb']):
            row_idx = i * 2 + j
            type_name = 'HO' if label_type == 'human' else 'PS'
            row_labels.append(f'{ISLAND_NAMES[train_island][:8]} {type_name}')
            for col, test_island in enumerate(ISLANDS):
                matrix[row_idx, col] = mode2[(train_island, label_type, test_island)][10]
    
    im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(8))
    ax.set_xticklabels([ISLAND_NAMES[i][:8] for i in ISLANDS], rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Test Island')
    ax.set_ylabel('Model')
    ax.set_title('Mode 2: Combined (HO+PS) Recall@10%', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=9, color='white')
    
    plt.colorbar(im, ax=ax, label='Recall@10%')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mode2_combined_heatmap.png'), dpi=150, bbox_inches='tight', metadata={})
    plt.close()
    print("  Saved mode2_combined_heatmap.png")
    
    # ========================================
    # FIGURE 4: Recall Curves - All Models Combined
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Recall@k% Curves by Test Island (Mode 2: Combined Positives)', fontsize=14, fontweight='bold')
    
    for idx, test_island in enumerate(ISLANDS):
        ax = axes[idx // 2, idx % 2]
        
        for train_island in ISLANDS:
            for label_type in ['human', 'herb']:
                type_name = 'HO' if label_type == 'human' else 'PS'
                linestyle = '-' if label_type == 'human' else '--'
                
                recalls = [mode2[(train_island, label_type, test_island)][k] for k in K_VALUES]
                ax.plot(K_VALUES, recalls, 
                       color=colors[train_island], 
                       linestyle=linestyle,
                       marker='o',
                       label=f'{ISLAND_NAMES[train_island][:8]} {type_name}',
                       linewidth=2)
        
        ax.set_xlabel('k% of Island Area')
        ax.set_ylabel('Recall')
        ax.set_title(f'Test Island: {ISLAND_NAMES[test_island]}')
        ax.set_xlim(0, 55)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mode2_recall_curves.png'), dpi=150, bbox_inches='tight', metadata={})
    plt.close()
    print("  Saved mode2_recall_curves.png")
    
    # ========================================
    # FIGURE 5: Same-type diagonal comparison
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # HO diagonal
    ax = axes[0]
    for train_island in ISLANDS:
        recalls = [mode1_ho[(train_island, train_island)][k] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[train_island], marker='o', 
               label=ISLAND_NAMES[train_island], linewidth=2)
    ax.set_xlabel('k% of Island Area')
    ax.set_ylabel('Recall')
    ax.set_title('HO→HO: Same-Island Performance (Diagonal)')
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # PS diagonal
    ax = axes[1]
    for train_island in ISLANDS:
        recalls = [mode1_ps[(train_island, train_island)][k] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[train_island], marker='o',
               label=ISLAND_NAMES[train_island], linewidth=2)
    ax.set_xlabel('k% of Island Area')
    ax.set_ylabel('Recall')
    ax.set_title('PS→PS: Same-Island Performance (Diagonal)')
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'same_island_performance.png'), dpi=150, bbox_inches='tight', metadata={})
    plt.close()
    print("  Saved same_island_performance.png")


if __name__ == "__main__":
    run_evaluation()
