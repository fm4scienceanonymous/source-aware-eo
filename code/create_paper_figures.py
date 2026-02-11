#!/usr/bin/env python3
"""
create_paper_figures.py

Uses inferno colormap and clean styling.

Output: out/visualizations/
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality settings with Noto Sans
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Noto Sans', 'DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.compression': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Island color scheme (consistent across all figures)
ISLAND_COLORS = {
    'san_clemente': '#B11226',   # Deep red
    'santa_catalina': '#2C4E9C', # Indigo/blue
    'santa_cruz': '#1F7A6D',     # Teal/green
    'santa_rosa': '#6A3D9A',     # Purple
}

# Paths
EVAL_DIR = './out/evaluation'
# Base directory for visualizations; each run writes into a timestamped subfolder
OUTPUT_BASE_DIR = './out/visualizations'
# Will be set to a timestamped subdirectory inside main()
OUTPUT_DIR = OUTPUT_BASE_DIR

ISLANDS = ['san_clemente', 'santa_catalina', 'santa_cruz', 'santa_rosa']
ISLAND_LABELS = ['San Clemente', 'Santa Catalina', 'Santa Cruz', 'Santa Rosa']
ISLAND_LABELS_SHORT = ['S. Clem.', 'S. Cat.', 'S. Cruz', 'S. Rosa']  # Abbreviated for heatmaps
K_VALUES = [1, 2, 5, 10, 20, 50]


def load_mode1_results():
    """Load Mode 1 results from CSVs."""
    ho_results = {}
    ps_results = {}
    
    for k in K_VALUES:
        ho_df = pd.read_csv(os.path.join(EVAL_DIR, f'mode1_HO_recall_at_{k}pct.csv'))
        ps_df = pd.read_csv(os.path.join(EVAL_DIR, f'mode1_PS_recall_at_{k}pct.csv'))
        
        ho_matrix = ho_df[ISLAND_LABELS].values
        ps_matrix = ps_df[ISLAND_LABELS].values
        
        ho_results[k] = ho_matrix
        ps_results[k] = ps_matrix
    
    return ho_results, ps_results


def load_mode2_results():
    """Load Mode 2 results from CSVs and split into HO and PS."""
    ho_results = {}
    ps_results = {}
    
    for k in K_VALUES:
        df = pd.read_csv(os.path.join(EVAL_DIR, f'mode2_combined_recall_at_{k}pct.csv'))
        
        # Split: rows 0,2,4,6 are HO models; rows 1,3,5,7 are PS models
        ho_matrix = df.iloc[0::2][ISLAND_LABELS].values
        ps_matrix = df.iloc[1::2][ISLAND_LABELS].values
        
        ho_results[k] = ho_matrix
        ps_results[k] = ps_matrix
    
    return ho_results, ps_results


def create_heatmap(ax, matrix, title, show_xlabel=True, show_ylabel=True, vmin=0, vmax=1, use_short_labels=True):
    """Create a single heatmap with inferno colormap."""
    im = ax.imshow(matrix, cmap='inferno', vmin=vmin, vmax=vmax, aspect='equal')
    
    labels = ISLAND_LABELS_SHORT if use_short_labels else ISLAND_LABELS
    
    # Ticks
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    
    if show_xlabel:
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Test Island', fontweight='medium', fontsize=9)
    else:
        ax.set_xticklabels([])
    
    if show_ylabel:
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel('Train Island', fontweight='medium', fontsize=9)
    else:
        ax.set_yticklabels([])
    
    ax.set_title(title, fontweight='bold', pad=6, fontsize=10)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            # Use white text for dark cells, black for light
            color = 'white' if val < 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=8, color=color, fontweight='medium')
    
    return im


def figure1_mode1_heatmaps():
    """Mode 1: Same-type evaluation heatmaps (HO→HO and PS→PS)."""
    print("Creating Figure 1: Mode 1 heatmaps...")
    
    ho_results, ps_results = load_mode1_results()
    
    # Create figure with 2 rows (HO, PS) x 6 columns (k values)
    fig, axes = plt.subplots(2, 6, figsize=(14, 6))
    fig.suptitle('Same-Type Evaluation: Recall@k% Area', fontsize=14, fontweight='bold', y=1.02)
    
    for idx, k in enumerate(K_VALUES):
        # HO row
        im = create_heatmap(axes[0, idx], ho_results[k], f'k={k}%',
                           show_xlabel=False, show_ylabel=(idx == 0))
        
        # PS row
        im = create_heatmap(axes[1, idx], ps_results[k], '',
                           show_xlabel=True, show_ylabel=(idx == 0))
    
    # Row labels
    axes[0, 0].annotate('HO to HO', xy=(-0.6, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    axes[1, 0].annotate('PS to PS', xy=(-0.6, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Recall', fontweight='medium')
    
    plt.tight_layout(rect=[0.05, 0, 0.9, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_mode1_heatmaps.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_mode1_heatmaps.pdf'), metadata={})
    plt.close()
    print("  Saved fig1_mode1_heatmaps.png/pdf")


def figure2_mode2_heatmaps():
    """Mode 2: Combined evaluation heatmaps (split into 2 4x4)."""
    print("Creating Figure 2: Mode 2 heatmaps...")
    
    ho_results, ps_results = load_mode2_results()
    
    # Create figure with 2 rows (HO models, PS models) x 6 columns (k values)
    fig, axes = plt.subplots(2, 6, figsize=(14, 6))
    fig.suptitle('Combined Evaluation (HO+PS Positives): Recall@k% Area', 
                fontsize=14, fontweight='bold', y=1.02)
    
    for idx, k in enumerate(K_VALUES):
        # HO models row
        im = create_heatmap(axes[0, idx], ho_results[k], f'k={k}%',
                           show_xlabel=False, show_ylabel=(idx == 0))
        
        # PS models row
        im = create_heatmap(axes[1, idx], ps_results[k], '',
                           show_xlabel=True, show_ylabel=(idx == 0))
    
    # Row labels
    axes[0, 0].annotate('HO Models', xy=(-0.6, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    axes[1, 0].annotate('PS Models', xy=(-0.6, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', rotation=90, va='center')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Recall', fontweight='medium')
    
    plt.tight_layout(rect=[0.05, 0, 0.9, 0.98])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_mode2_heatmaps.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_mode2_heatmaps.pdf'), metadata={})
    plt.close()
    print("  Saved fig2_mode2_heatmaps.png/pdf")


def figure3_recall_at_10():
    """Single figure showing R@10% for all three evaluation modes."""
    print("Creating Figure 3: R@10% comparison...")
    
    ho_same, ps_same = load_mode1_results()
    ho_comb, ps_comb = load_mode2_results()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle('Recall@10% Area Across Evaluation Modes', fontsize=14, fontweight='bold', y=0.98)
    
    # (0,0) HO to HO (Same-type)
    im1 = create_heatmap(axes[0, 0], ho_same[10], 'HO to HO (Same-Type)')
    
    # (0,1) PS to PS (Same-type)
    im2 = create_heatmap(axes[0, 1], ps_same[10], 'PS to PS (Same-Type)', show_ylabel=False)
    
    # (1,0) HO models on combined
    im3 = create_heatmap(axes[1, 0], ho_comb[10], 'HO Models to Combined')
    
    # (1,1) PS models on combined
    im4 = create_heatmap(axes[1, 1], ps_comb[10], 'PS Models to Combined', show_ylabel=False)
    
    # Single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Recall@10%', fontweight='medium')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_recall_at_10_comparison.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_recall_at_10_comparison.pdf'), metadata={})
    plt.close()
    print("  Saved fig3_recall_at_10_comparison.png/pdf")


def figure4_recall_curves():
    """Recall curves showing performance across k values."""
    print("Creating Figure 4: Recall curves...")
    
    ho_same, ps_same = load_mode1_results()
    
    # Colors for islands (consistent color scheme)
    colors = [ISLAND_COLORS[island] for island in ISLANDS]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Same-Island Performance Across k% Thresholds', fontsize=12, fontweight='bold')
    
    # HO models (diagonal = same-island)
    ax = axes[0]
    for i, island in enumerate(ISLAND_LABELS):
        recalls = [ho_same[k][i, i] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[i], marker='o', markersize=5,
               linewidth=2, label=island)
    
    ax.set_xlabel('k% of Island Area', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Human Observation Models', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # PS models (diagonal = same-island)
    ax = axes[1]
    for i, island in enumerate(ISLAND_LABELS):
        recalls = [ps_same[k][i, i] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[i], marker='s', markersize=5,
               linewidth=2, label=island)
    
    ax.set_xlabel('k% of Island Area', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Preserved Specimen Models', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_recall_curves.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_recall_curves.pdf'), metadata={})
    plt.close()
    print("  Saved fig4_recall_curves.png/pdf")


def figure5_transfer_analysis():
    """Analyze transfer performance (off-diagonal vs diagonal)."""
    print("Creating Figure 5: Transfer analysis...")
    
    ho_same, ps_same = load_mode1_results()
    
    # Calculate diagonal (same-island) vs off-diagonal (transfer) averages
    k_vals = [5, 10, 20]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Same-Island vs. Transfer Performance', fontsize=12, fontweight='bold')
    
    colors = [ISLAND_COLORS[island] for island in ISLANDS]
    
    for ax_idx, (results, title) in enumerate([(ho_same, 'HO Models'), (ps_same, 'PS Models')]):
        ax = axes[ax_idx]
        
        x = np.arange(len(k_vals))
        width = 0.18
        
        for i, island in enumerate(ISLAND_LABELS):
            # Same-island (diagonal)
            same_vals = [results[k][i, i] for k in k_vals]
            
            # Transfer (average of off-diagonal for this row)
            transfer_vals = []
            for k in k_vals:
                off_diag = [results[k][i, j] for j in range(4) if j != i]
                transfer_vals.append(np.mean(off_diag))
            
            # Plot grouped bars
            offset = (i - 1.5) * width
            bars1 = ax.bar(x + offset - width/2, same_vals, width * 0.9, 
                          color=colors[i], alpha=0.9, label=f'{island} (same)')
            bars2 = ax.bar(x + offset + width/2, transfer_vals, width * 0.9,
                          color=colors[i], alpha=0.4, hatch='//')
        
        ax.set_xlabel('k% Threshold')
        ax.set_ylabel('Recall')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}%' for k in k_vals])
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Custom legend
        if ax_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gray', alpha=0.9, label='Same-Island'),
                Patch(facecolor='gray', alpha=0.4, hatch='//', label='Transfer (avg)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_transfer_analysis.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_transfer_analysis.pdf'), metadata={})
    plt.close()
    print("  Saved fig5_transfer_analysis.png/pdf")


def figure6_recall_curves_combined():
    """Recall curves for combined test data (Mode 2) - same-island performance."""
    print("Creating Figure 6: Recall curves (Combined test data)...")
    
    ho_comb, ps_comb = load_mode2_results()
    
    # Colors for islands (consistent color scheme)
    colors = [ISLAND_COLORS[island] for island in ISLANDS]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Same-Island Performance on Combined Test Data (HO+PS Positives)', fontsize=12, fontweight='bold')
    
    # HO models on combined test data (diagonal = same-island)
    ax = axes[0]
    for i, island in enumerate(ISLAND_LABELS):
        recalls = [ho_comb[k][i, i] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[i], marker='o', markersize=5,
               linewidth=2, label=island)
    
    ax.set_xlabel('k% of Island Area', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Human Observation Models', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # PS models on combined test data (diagonal = same-island)
    ax = axes[1]
    for i, island in enumerate(ISLAND_LABELS):
        recalls = [ps_comb[k][i, i] for k in K_VALUES]
        ax.plot(K_VALUES, recalls, color=colors[i], marker='s', markersize=5,
               linewidth=2, label=island)
    
    ax.set_xlabel('k% of Island Area', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Preserved Specimen Models', fontweight='bold', fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_recall_curves_combined.png'), metadata={})
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_recall_curves_combined.pdf'), metadata={})
    plt.close()
    print("  Saved fig6_recall_curves_combined.png/pdf")


def main():
    global OUTPUT_DIR
    
    # Create a unique timestamped output directory so we don't overwrite
    # figures from previous runs.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"run_{run_stamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("CREATING PUBLICATION FIGURES")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Colormap: inferno")
    print()
    figure1_mode1_heatmaps()
    figure2_mode2_heatmaps()
    figure3_recall_at_10()
    figure4_recall_curves()
    figure5_transfer_analysis()
    figure6_recall_curves_combined()
    
    print()
    print("=" * 60)
    print("ALL FIGURES CREATED")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
