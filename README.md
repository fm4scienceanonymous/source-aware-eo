# Anonymous ICLR 2026 Submission

Source-Aware Evaluation of Earth Observation Embeddings for Sparse Occurrence Data

This repository contains code accompanying the anonymous ICLR 2026 submission.

---

## Overview

This project evaluates pretrained Earth Observation (EO) embeddings for habitat similarity
ranking under sparse species occurrence data.

Using Quercus tomentella across four California Channel Islands, we:

1. Obtain and filter GBIF occurrence records
2. Assign occurrences to 10m TESSERA pixel embeddings
3. Construct source-specific labels:
   - Human Observation (HO)
   - Preserved Specimen (PS)
4. Perform independent 70/30 train–test splits per island and source
5. Train L2-regularized logistic regression models on 128-dimensional embeddings
6. Evaluate performance using Recall@k% land area under:
   - Same-source evaluation
   - Combined-positives evaluation

The experimental protocol corresponds to Sections 2.1–2.3 of the paper.

---

## Repository Structure

code/
  filter_gbif_occurrences.py
  assign_occurrences_to_pixels.py
  extract_channel_islands_tessera.py
  prepare_data_deduplicated.py
  train_models.py
  evaluate_models.py
  create_paper_figures.py
  create_study_area_map.py
  run_evaluation_and_figures.py

data/
  (input occurrence records and study area geometries; not included in this archive)

out/
  (generated models, metrics, and figures produced by the pipeline)

---

## Requirements

Python 3.10+

Core libraries:
- pandas
- geopandas
- numpy
- scikit-learn
- matplotlib
- shapely
- pyarrow

Install dependencies using your preferred package manager (e.g., pip or conda).

---

## Reproducing Results

### Inputs

Running the full pipeline requires:
- GBIF occurrence download (filtered to the study area and basis-of-record types)
- Study area polygons (island geometries)
- TESSERA embeddings for the study area

Paths/filenames are provided via script arguments or environment variables as indicated in each script.

### 1. Preprocess Occurrence Data

    python code/filter_gbif_occurrences.py
    python code/assign_occurrences_to_pixels.py
    python code/extract_channel_islands_tessera.py

### 2. Prepare Train/Test Splits

    python code/prepare_data_deduplicated.py

This performs:
- Independent 70/30 train–test splits per island and source
- Negative sampling at a 10:1 ratio (absences to presences)

### 3. Train Models

    python code/train_models.py

Two models are trained per island:
- HO-trained
- PS-trained

Each model uses L2-regularized logistic regression on fixed embeddings.

### 4. Evaluate Models

    python code/evaluate_models.py

Computes Recall@k% land area for:
- Same-source evaluation
- Combined-positives evaluation

Training positives and sampled negatives are excluded from ranking pools
to prevent leakage, as described in the paper.

### 5. Generate Figures

    python code/create_paper_figures.py

Produces heatmaps and recall curves corresponding to the Results section.

---

## Evaluation Metric

Recall@k% land area is defined as:

    (Number of evaluation positives ranked within top k% of pixels)
    ----------------------------------------------------------------
    (Number of evaluation positives in ranking pool)

Pixels used as training positives or sampled negatives are excluded from ranking.

---

## Reproducibility

- Random seeds are fixed where applicable.
- The results in the submission were produced using this codebase and the described pipeline.
- Logistic regression is used as an interpretable baseline.

---

This repository is released for anonymous peer review.
