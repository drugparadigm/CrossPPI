# CrossPPI

CrossPPI — a cross-attention based deep learning method for protein-protein binding affinity (pKD) prediction.

Table of contents
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Requirements & environment](#requirements--environment)
- [Data preparation](#data-preparation)
- [Generate contact maps & embeddings (ESM-2)](#generate-contact-maps--embeddings-esm-2)
- [Training](#training)
- [Important files & scripts](#important-files--scripts)
- [Architecture](#architecture)
---

## Key features
- Uses ESM-2 protein language model for sequence embeddings.
- Extracts structural features via contact maps and graph-based encoders.
- Cross-fusion module with cross-attention between ligand and receptor features.
- Trained for pKD (protein-protein binding affinity) prediction with cross-validation.

## Repository structure
- `data/` — data processing scripts and place to store generated contact maps, embeddings, and processed dataset folders.
  - `PPB-Affinity-Modified(pkd).xlsx` — raw pKD dataset.
  - `protein_dataset.py`, `protein_processor.py`, `process_ligands.py`, `process_receptors.py` — data handling & processing.
- `model/` — model modules
  - `gnn_model_protein.py` — graph-based protein feature extraction
  - `cross_attention.py` — cross-attention and cross-fusion modules
- `dataset/` — processed dataset produced by training or preprocessing steps (not included)
- Training & evaluation
  - `main_cv.py` — training script using cross-validation
- Preprocessing
  - `contact_map.py` — contact map generation using ESM-2
  - `embedding.py` — sequence embedding generation using ESM-2

## Requirements & environment
Recommended: conda for reproducibility.

Create conda environment from provided environment.yml:
```bash
conda env create -f environment.yml
conda activate CrossPPI
```

If you don't use conda, create a Python 3.8+ environment and install dependencies in `environment.yml`. Ensure PyTorch (with CUDA if using GPU), numpy, pandas, scikit-learn, networkx, and the ESM package (or fair-esm) are installed.

ESM-2 dependency:
- ESM-2 (Facebook Research) is required for sequence embeddings and optionally for contact map features. See https://github.com/facebookresearch/esm for installation instructions. Installation via pip (example):
```bash
pip install fair-esm
```
or follow official ESM install notes.

## Data preparation

1. Put your raw dataset `PPB-Affinity-Modified(pkd).xlsx` in `data/` (if it's not already).
2. Generate contact maps and embeddings (required before training):
```bash
python contact_map.py
python embedding.py
```

Notes:
- Generating embeddings/contact maps may be compute-intensive. Use a GPU and sufficient RAM/disk.
- The exact arguments to `contact_map.py` and `embedding.py` can be inspected in their header (e.g., input/output paths, sequence length truncation, batching). Adjust as needed for your setup.

## Generate contact maps & embeddings (ESM-2)
- contact_map.py: loads sequences and runs ESM-2 to compute contact map predictions (or pairwise distance/contact scores).
- embedding.py: uses ESM-2 to produce per-residue embeddings and aggregated embeddings.

Typical flow:
1. Ensure `data/PPB-Affinity-Modified(pkd).xlsx` is present and formatted correctly.
2. Run `python contact_map.py` — this will create contact map files in `data/contact_maps_*`.
3. Run `python embedding.py` — this will create embedding files in `data/embeddings_*`.

## Training
Train the CrossPPI model using cross-validation:
```bash
python main_cv.py
```


Check each prediction script for exact CLI options and required input formats.

## Important files & scripts
- `main_cv.py` — Training with cross-validation and checkpoint saving.
- `contact_map.py` — Generates contact maps using ESM-2.
- `embedding.py` — Generates protein embeddings using ESM-2.
- `model/gnn_model_protein.py` — Protein graph feature extraction.
- `model/cross_attention.py` — Cross-attention and cross-fusion implementation.
- `data/protein_dataset.py` — Dataset class used by training.
- `data/protein_processor.py` — Processor used for inference/prediction.

## Architecture

Below is the CrossPPI architecture diagram.

![CrossPPI architecture](Picture1.jpg)


