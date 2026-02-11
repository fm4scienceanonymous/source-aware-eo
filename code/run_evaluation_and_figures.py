#!/usr/bin/env python3
"""
run_evaluation_and_figures.py

Combined script to:
1. Run model evaluation (Recall@k% metrics)
2. Generate figures
"""

import os
import sys

# Add code directory to path
sys.path.insert(0, './code')

print("=" * 70)
print("STEP 1: RUNNING MODEL EVALUATION")
print("=" * 70)

# Import and run evaluation
from evaluate_models import run_evaluation
run_evaluation()

print("\n" + "=" * 70)
print("STEP 2: GENERATING PUBLICATION FIGURES")
print("=" * 70)

# Import and run figure generation
from create_paper_figures import main as create_figures
create_figures()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("\nOutputs:")
print("  - Evaluation CSVs: ./out/evaluation/")
print("  - Figures (PNG/PDF): ./out/visualizations/")
