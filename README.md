# MMFP: MultiModal Fusion for Protein Function Prediction

A PyTorch framework for multimodal protein function prediction that integrates four modalities: **sequence**, **text**, **structure**, and **protein-protein interaction (PPI)** embeddings.

## Overview

MMFP predicts Gene Ontology (GO) term annotations for proteins by combining information from multiple modalities using advanced fusion techniques. The framework supports:

- **Three GO aspects**: Biological Process (BPO), Cellular Component (CCO), Molecular Function (MFO)
- **Multiple fusion methods**: Concatenation, Bilinear Gated Fusion
- **Flexible modality handling**: Gracefully handles missing modalities via learned masking
- **CAFA-compliant evaluation**: Standard evaluation metrics including F-max, weighted F-max, and S-min

## Architecture

```
                    ┌─────────────────┐
                    │   Input Protein │
                    └────────┬────────┘
                             │
         ┌───────────┬───────┴───────┬───────────┐
         ▼           ▼               ▼           ▼
    ┌─────────┐ ┌─────────┐   ┌─────────┐ ┌─────────┐
    │   Seq   │ │  Text   │   │ Struct  │ │   PPI   │
    │ ProtT5  │ │PubMedBERT│  │  ESM-IF │ │  Graph  │
    │  1024d  │ │   768d  │   │  512d   │ │  512d   │
    └────┬────┘ └────┬────┘   └────┬────┘ └────┬────┘
         │           │             │           │
         ▼           ▼             ▼           ▼
    ┌─────────────────────────────────────────────┐
    │          Modality Encoders                  │
    │     (Project to common hidden space)        │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │            Fusion Module                     │
    │       (Concat / Bilinear Gated)              │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │           Classifier Head                   │
    │         (GO Term Predictions)               │
    └─────────────────────────────────────────────┘
```

## Fusion Methods

### 1. Concatenation Fusion (Baseline)
Simple concatenation of encoded modality embeddings followed by projection.

### 2. Bilinear Gated Fusion
Combines unary (per-modality) and pairwise (between-modality) scores to learn which modality combinations are most informative:
- Learns compatibility matrix between modality pairs
- Captures synergistic relationships (e.g., sequence + structure)
- Adaptive weighting based on input

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MMFP.git
cd MMFP

# Create conda environment
conda create -n mmfp python=3.9
conda activate mmfp

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

You have **two options** for preparing data:

---

### Option A: Download Precomputed Embeddings (Recommended)

Download our precomputed CAFA3 embeddings to directly reproduce results:

```bash
# Download precomputed data (~2GB compressed)
wget https://zenodo.org/record/XXXXXXX/files/mmfp_cafa3_data.tar.gz

# Extract to data directory
tar -xzf mmfp_cafa3_data.tar.gz -C ./data
```

**Included files:**

| File Type | Description | Coverage |
|-----------|-------------|----------|
| `embedding_cache/prott5/` | ProtT5-XL sequence embeddings (1024-D) | 69,811 proteins |
| `embedding_cache/exp_text_embeddings/` | PubMedBERT text embeddings (768-D) | 69,519 proteins (~99.6%) |
| `embedding_cache/IF1/` | ESM-IF1 structure embeddings (512-D) | 67,948 proteins (~97.3%) |
| `embedding_cache/ppi/` | STRING PPI network embeddings (512-D) | 58,294 proteins (~83.5%) |
| `{BPO,CCO,MFO}_*.npy/.npz/.json` | Train/valid/test splits and GO term labels | 3 aspects |

**Data directory structure:**
```
data/
├── embedding_cache/
│   ├── prott5/              # ProtT5 sequence embeddings (1024-D)
│   │   ├── P12345.npy       # Shape: (1024,)
│   │   └── ...
│   ├── exp_text_embeddings/ # PubMedBERT text embeddings (768-D)
│   ├── IF1/                 # ESM-IF1 structure embeddings (512-D)
│   └── ppi/                 # STRING PPI embeddings (512-D)
├── BPO_train_names.npy      # Protein IDs for training
├── BPO_train_labels.npz     # GO term labels (sparse matrix)
├── BPO_valid_names.npy
├── BPO_valid_labels.npz
├── BPO_test_names.npy
├── BPO_test_labels.npz
├── BPO_go_terms.json        # List of GO terms
└── ... (same for CCO, MFO)
```

After downloading, you can directly run training:
```bash
python train.py --data-dir ./data --aspects BPO CCO MFO
```

---

### Option B: Extract Features from Scratch

If you want to extract embeddings for your own dataset or reproduce the extraction pipeline:

#### Required External Resources

| Modality | Required Resources |
|----------|-------------------|
| **ProtT5** | FASTA file with protein sequences |
| **Text** | Internet access (UniProt API) |
| **Structure** | Internet access (AlphaFold DB) + `fair-esm` package |
| **PPI** | STRING database files (`protein.network.embeddings.v12.0.h5`, `protein.aliases.v12.0.txt`) |

#### Step 1: Extract ProtT5 Sequence Embeddings
```bash
python scripts/extract_prott5_embeddings.py \
    --fasta proteins.fasta \
    --output_dir data/embedding_cache/prott5 \
    --batch_size 8
```

#### Step 2: Extract Text Embeddings
```bash
# First, fetch UniProt descriptions
python scripts/extract_uniprot_text.py \
    --protein_ids proteins.txt \
    --output_dir data/uniprot_text

# Then, embed with PubMedBERT
python scripts/embed_uniprot_descriptions.py \
    --text_dir data/uniprot_text \
    --output_dir data/embedding_cache/exp_text_embeddings
```

#### Step 3: Extract Structure Embeddings
```bash
# First, download AlphaFold structures
python scripts/check_alphafold_coverage.py \
    --protein_ids proteins.txt \
    --output_dir data/alphafold_structures

# Then, extract ESM-IF1 embeddings
python scripts/extract_esm_if1_embeddings.py \
    --pdb_dir data/alphafold_structures \
    --output_dir data/embedding_cache/IF1 \
    --pooling mean
```

#### Step 4: Extract PPI Embeddings
```bash
python scripts/extract_ppi_embeddings.py \
    --protein_ids proteins.txt \
    --output_dir data/embedding_cache/ppi
```

> **Note:** See `scripts/README.md` for detailed documentation on each extraction script, including required dependencies and configuration options.

## Training

### Basic Training

```bash
# Train with bilinear gated fusion on all aspects
python train.py --seq-model prott5 --aspects BPO CCO MFO --fusion-types gated_bilinear

# Train all fusion types for comparison
python train.py --seq-model prott5 --aspects BPO CCO MFO --fusion-types concat gated_bilinear
```

### Advanced Options

```bash
python train.py \
    --seq-model prott5 \           # Sequence model: prott5 or esm
    --aspects BPO CCO MFO \        # GO aspects to train
    --fusion-types gated_bilinear \ # Fusion method
    --modality-dropout 0.1 \       # Dropout rate for non-sequence modalities
    --use-late-fusion \            # Enable auxiliary heads + hybrid fusion
    --aux-loss-weight 0.8 \        # Weight for auxiliary supervision
    --output-base ./results        # Output directory
```

### Configuration

Key hyperparameters (in `train.py`):
- `hidden_dim`: 512 (common embedding dimension)
- `dropout`: 0.4
- `lr`: 1e-3
- `batch_size`: 32
- `max_epochs`: 50
- `patience`: 5 (early stopping)

## Evaluation

The framework provides CAFA-compliant evaluation:

```bash
# Training automatically runs evaluation
# Results are saved to: results/fusion_comparison/{seq_model}/{aspect}/{fusion_type}/

# View results
cat results/fusion_comparison/prott5/BPO/gated_bilinear/results.json
```

### Metrics

- **F-max**: Maximum F1-score across thresholds
- **Weighted F-max**: F-max weighted by Information Accretion (IA)
- **S-min**: Minimum semantic distance
- **Precision/Recall**: At optimal threshold

## Results Structure

```
results/
└── fusion_comparison/
    └── prott5/
        ├── BPO/
        │   ├── concat/
        │   │   ├── best_model.pt
        │   │   ├── results.json
        │   │   ├── history.csv
        │   │   └── cafa_eval/
        │   └── gated_bilinear/
        ├── CCO/
        ├── MFO/
        └── summary.csv
```

## Model API

```python
from mmfp.models import MultiModalFusionModel, create_model

# Create model
model = create_model(
    fusion_type='gated_bilinear',  # or 'concat'
    seq_dim=1024,                  # ProtT5: 1024, ESM: 1280
    text_dim=768,
    struct_dim=512,
    ppi_dim=512,
    hidden_dim=512,
    num_go_terms=1000,
    dropout=0.3,
    modality_dropout=0.1,
    use_late_fusion=False
)

# Forward pass
logits, fusion_weights, aux_outputs = model(
    seq, seq_mask,       # [B, 1024], [B, 1]
    text, text_mask,     # [B, 768], [B, 1]
    struct, struct_mask, # [B, 512], [B, 1]
    ppi, ppi_mask        # [B, 512], [B, 1]
)
# logits: [B, num_go_terms]
# fusion_weights: [B, 4] - weights for [seq, text, struct, ppi]
```

## Project Structure

```
MMFP/
├── mmfp/
│   ├── __init__.py
│   ├── models.py              # Fusion models and architectures
│   ├── dataset.py             # Dataset and data loading
│   └── evaluation.py          # CAFA evaluation utilities
├── scripts/
│   ├── README.md              # Detailed script documentation
│   ├── extract_prott5_embeddings.py    # ProtT5 sequence embeddings
│   ├── extract_uniprot_text.py         # Fetch UniProt descriptions
│   ├── embed_uniprot_descriptions.py   # PubMedBERT text embeddings
│   ├── check_alphafold_coverage.py     # Download AlphaFold structures
│   ├── extract_esm_if1_embeddings.py   # ESM-IF1 structure embeddings
│   ├── extract_ppi_embeddings.py       # STRING PPI embeddings
│   └── prepare_cafa3_data.py           # Prepare CAFA3 benchmark splits
├── train.py                   # Main training script
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mmfp2024,
  title={Multimodal Fusion for Protein Function Prediction},
  author={Your Name},
  journal={},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- [ProtT5](https://github.com/agemagician/ProtTrans) for protein language model embeddings
- [ESM](https://github.com/facebookresearch/esm) for sequence and structure embeddings
- [cafaeval](https://github.com/BioComputingUP/CAFA-evaluator) for CAFA-compliant evaluation
