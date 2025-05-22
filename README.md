# Protein Function Prediction Pipeline

This project provides a pipeline for protein function prediction using deep learning and protein language model embeddings.

## Quick Start

1. **Prepare Data:**
   - Use `prepare_data.py` to process your raw FASTA and TSV files into training-ready data.

2. **Extract Embeddings:**
   - Generate embeddings for your protein sequences (see `plm.py` or `cluster_embed/`).

3. **Train Model:**
   - Run the main training pipeline:
     ```bash
     python train_script.py
     ```
   - The script uses configs in `configs/` and saves results in `outputs/` or `runs/`.

## Structure
- `train_script.py`: Main entry for training
- `prepare_data.py`: Data preparation
- `configs/`: Experiment/model configs
- `Network/`, `models/`: Model code
- `utils/`: Utilities

