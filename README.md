# Protein Function Prediction (PFP)

This repository provides a framework for protein function prediction using a combination of sequence, structure, and text-based features, leveraging deep learning and protein language models.

## Features
- **Data Preparation**: Scripts to process protein sequences, annotations, and generate training/test datasets.
- **Model Training**: Deep neural network models for multi-label protein function prediction, including support for various embedding types (e.g., ESM, T5, structure-based, and fused models).
- **Ontology Tools**: Utilities for handling Gene Ontology (GO) terms and relationships.
- **Text Embedding**: Generate protein-related text embeddings using transformer models (e.g., BiomedBERT).
- **Structure Processing**: Tools for building and processing protein structure graphs.

## Directory Structure
- `prepare_data.py`: Prepare datasets from raw sequence and annotation files.
- `train_InterLabelGO.py`: Main training script for protein function prediction models.
- `AP_align_fuse.py`: Model for aligning and fusing sequence and text embeddings.
- `prot2text.py`: Generate text embeddings for protein descriptions.
- `structure.py`: Utilities for processing protein structures and building graph representations.
- `Network/`: Contains model architectures and training utilities.
- `utils/`: Ontology and miscellaneous utility scripts.
- `cluster/`: Scripts for HPC.
