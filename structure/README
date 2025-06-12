# Structure-Based GO Prediction Pipeline

This repository implements **E(3)-Equivariant Graph Neural Networks (EGNN)** to predict Gene Ontology (GO) terms directly from AlphaFold protein structures.

---

## Overview

1. **PDB processing** – extract sequence + Cα coordinates from AlphaFold PDB files  
2. **ESM embedding generation** – align per-residue ESM embeddings with PDB sequences  
3. **Graph construction** – build *k*-NN or radius graphs from 3-D structures  
4. **EGNN model** – learn structure-aware representations with equivariant GNNs  
5. **GO classification** – predict GO terms from the learned graph embeddings  

---

## Directory structure

```text
/PFP/structure/
├── pdb_graph_utils.py      # PDB processing & graph construction
├── esm_pdb_embeddings.py   # Generate ESM embeddings for PDB sequences
├── egnn_model.py           # EGNN implementation
├── train_structure.py      # Training script
├── configs/
│   └── structure_config.yaml  # Default configuration
└── experiments/            # Output directory for runs
