# Hybrid Gated Fusion: A Multimodal Deep Learning Framework for Protein Functional Annotation

A PyTorch framework for multimodal protein function prediction that integrates four modalities: **sequence**, **text**, **structure**, and **protein-protein interaction (PPI)** embeddings using a hybrid gated bilinear fusion architecture.

## Overview

MMFP predicts Gene Ontology (GO) term annotations for proteins by combining information from multiple modalities using advanced fusion techniques. The framework supports:

- **Three GO aspects**: Biological Process (BPO), Cellular Component (CCO), Molecular Function (MFO)
- **Multiple fusion methods**: Concatenation, Bilinear Gated Fusion, Hybrid (bilinear + auxiliary heads)
- **Flexible modality handling**: Gracefully handles missing modalities via learned masking
- **CAFA-compliant evaluation**: Standard evaluation metrics including F-max, weighted F-max, and S-min
- **Temporal decontamination**: Historical UniProt text (pre-CAFA3 cutoff) via UniSave to prevent data leakage

## Temporal Decontamination

To prevent data leakage, test-set text embeddings use **historical UniProt records** retrieved from UniSave with a cutoff date of `2016-02-17` (before the CAFA3 assessment period). Train/validation text embeddings use current UniProt descriptions. This ensures the model cannot exploit post-assessment functional annotations at test time.

## Installation

```bash
git clone https://github.com/psipred/PFP.git
cd PFP/MMFP
pip install -r requirements.txt
```

## Data Preparation

### Option A: Download Precomputed Data (Recommended)

Download precomputed CAFA3 embeddings and data splits:

```bash
# Download precomputed data from Zenodo: https://zenodo.org/records/19498341
wget https://zenodo.org/records/19498341/files/mmfp_cafa3_data.tar.gz

# Extract to data directory
tar -xzf mmfp_cafa3_data.tar.gz -C ./data
```

**Required data:**

| Directory | Description | Size |
|-----------|-------------|------|
| `embedding_cache/prott5/` | ProtT5-XL sequence embeddings (1024-D) | ~550 MB |
| `embedding_cache/exp_text_embeddings_temporal/` | Temporal PubMedBERT text embeddings (768-D) | ~280 MB |
| `embedding_cache/IF1/` | ESM-IF1 structure embeddings (512-D) | ~270 MB |
| `embedding_cache/ppi/` | STRING PPI network embeddings (512-D) | ~230 MB |
| `{BPO,CCO,MFO}_*.npy/.npz/.json` | Train/valid/test splits and GO term labels | - |
| `{BPO,CCO,MFO}_ia.txt` | Information Accretion weights | - |
| `go.obo` | Gene Ontology structure (place in parent dir) | - |

### Option B: Extract from Scratch

See `scripts/README.md` for detailed extraction instructions.

## Reproducing Paper Results

| Paper Table | Description | Command |
|------------|-------------|---------|
| Table 1 | CAFA3 Comparison (Full Model) | `python scripts/reproduce_full_model.py` |
| Table 2 | Modality Contribution | `python scripts/reproduce_modality_contribution.py` |
| Table 3 | Ablation Study | `python scripts/run_ablation.py` |
| Table 4 | Masking Evaluation | `python scripts/reproduce_masking_eval.py` |

### Training from Scratch

```bash
# Full model (Table 1)
python train.py \
  --seq-model prott5 \
  --fusion-types gated_bilinear \
  --aspects BPO CCO MFO \
  --use-late-fusion \
  --text-embedding-dir data/embedding_cache/exp_text_embeddings_temporal \
  --output-base results/full_model \
  --seed 42

# Evaluation with CAFA metrics
python scripts/reproduce_full_model.py
```

### Expected Results (Table 1)

| Aspect | Fmax | wFmax |
|--------|------|-------|
| BPO | 0.601 | 0.515 |
| MFO | 0.702 | 0.605 |
| CCO | 0.706 | 0.566 |

## Project Structure

```
MMFP/
├── mmfp/
│   ├── models.py              # Fusion models and architectures
│   ├── dataset.py             # Dataset and data loading
│   └── evaluation.py          # CAFA evaluation utilities
├── scripts/
│   ├── reproduce_full_model.py          # Table 1 reproduction
│   ├── reproduce_modality_contribution.py  # Table 2 reproduction
│   ├── run_ablation.py                  # Table 3 reproduction
│   ├── reproduce_masking_eval.py        # Table 4 reproduction
│   ├── extract_uniprot_text.py          # Text extraction + temporal bundle
│   ├── embed_uniprot_descriptions.py    # PubMedBERT embedding
│   ├── extract_prott5_embeddings.py     # ProtT5 sequence embeddings
│   ├── extract_esm_if1_embeddings.py    # ESM-IF1 structure embeddings
│   ├── extract_ppi_embeddings.py        # STRING PPI embeddings
│   ├── extract_historical_uniprot_text.py  # Historical UniSave extraction
│   ├── check_alphafold_coverage.py      # AlphaFold PDB download
│   └── prepare_cafa3_data.py           # CAFA3 benchmark splits
├── train.py                   # Main training script
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mmfp2025,
  title={Hybrid Gated Fusion: A Multimodal Deep Learning Framework for Protein Functional Annotation},
  author={Zijian Zhou and Daniel W. A. Buchan},
  journal={TODO},
  year={2025}
}
```

## License

MIT License

