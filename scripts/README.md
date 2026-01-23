# MMFP Feature Extraction Scripts

This directory contains scripts for extracting multi-modal protein embeddings for the MMFP (Multi-Modal Fusion for Protein function prediction) framework.

## Overview

| Modality | Script(s) | Output Dimension | Description |
|----------|-----------|------------------|-------------|
| **PPI** | `extract_ppi_embeddings.py` | 512-D | STRING protein-protein interaction network embeddings |
| **Text** | `extract_uniprot_text.py` + `embed_uniprot_descriptions.py` | 768-D | PubMedBERT embeddings from UniProt descriptions |
| **Structure** | `check_alphafold_coverage.py` + `extract_esm_if1_embeddings.py` | 512-D | ESM-IF1 structure embeddings from AlphaFold PDB files |
| **ProtT5** | `extract_prott5_embeddings.py` | 1024-D | ProtT5-XL sequence embeddings |

---

## Dependencies

### Python Packages
```bash
pip install torch transformers numpy tqdm requests h5py fair-esm
```

### External Resources

| Resource | Required By | Download |
|----------|-------------|----------|
| CAFA Assessment Tool | PPI, Text, Structure | https://github.com/ashleyzhou972/CAFA_assessment_tool |
| STRING Network Files | PPI | https://string-db.org/cgi/download |
| AlphaFold API | Structure | Internet access required |
| UniProt API | Text | Internet access required |

---

## Script Details

### 1. PPI Embeddings (`extract_ppi_embeddings.py`)

Extracts 512-D protein-protein interaction embeddings from STRING database.

**Required Files:**
- STRING embeddings: `protein.network.embeddings.v12.0.h5`
- STRING aliases: `protein.aliases.v12.0.txt`
- CAFA Assessment Tool (for ID mapping)

**Configuration (edit in script):**
```python
STRING_H5_FILE = "/path/to/protein.network.embeddings.v12.0.h5"
STRING_ALIAS_FILE = "/path/to/protein.aliases.v12.0.txt"
CAFA_ASSESSMENT_DIR = "/path/to/CAFA_assessment_tool"
```

**Usage:**
```bash
cd MMFP
python scripts/extract_ppi_embeddings.py
```

**Output:** `data/embedding_cache/ppi/{protein_id}.npy`

---

### 2. Text Embeddings

#### Step 1: Extract Text (`extract_uniprot_text.py`)

Fetches protein descriptions from UniProt REST API.

**Required:**
- Internet access
- CAFA Assessment Tool (for CAFA3 ID → UniProt mapping)

**Configuration (edit in script):**
```python
cafa_assessment_dir = Path("/path/to/CAFA_assessment_tool")
```

**Usage:**
```bash
cd MMFP
python scripts/extract_uniprot_text.py
```

**Output:** `data/embedding_cache/uniprot_text/protein_descriptions.tsv`

#### Step 2: Generate Embeddings (`embed_uniprot_descriptions.py`)

Generates PubMedBERT embeddings from extracted text descriptions.

**Required:**
- Output from Step 1
- HuggingFace transformers (model auto-downloads)

**Usage:**
```bash
python scripts/embed_uniprot_descriptions.py --data-dir data
```

**Output:** `data/embedding_cache/exp_text_embeddings/{protein_id}.npy`

---

### 3. Structure Embeddings

#### Step 1: Download PDB Files (`check_alphafold_coverage.py`)

Checks AlphaFold availability and downloads PDB structures.

**Required:**
- Internet access (AlphaFold API)
- CAFA Assessment Tool

**Configuration (edit in script):**
```python
cafa_assessment_dir = Path("/path/to/CAFA_assessment_tool")
pdb_output_dir = Path("data/alphafold_structures")
```

**Usage:**
```bash
cd MMFP
python scripts/check_alphafold_coverage.py
```

**Output:** `data/alphafold_structures/{protein_id}.pdb`

#### Step 2: Generate ESM-IF1 Embeddings (`extract_esm_if1_embeddings.py`)

Extracts 512-D structure embeddings using ESM-IF1 inverse folding model.

**Required:**
- PDB files from Step 1
- `fair-esm` package

**Usage:**
```bash
python scripts/extract_esm_if1_embeddings.py \
    --pdb_dir data/alphafold_structures \
    --output_dir data/embedding_cache/structure \
    --pooling mean \
    --device cuda
```

**Output:** `data/embedding_cache/structure/{protein_id}.npy`

---

### 4. ProtT5 Embeddings (`extract_prott5_embeddings.py`)

Generates 1024-D sequence embeddings using ProtT5-XL model.

**Required:**
- FASTA file with protein sequences
- HuggingFace transformers

**Usage:**
```bash
python scripts/extract_prott5_embeddings.py \
    --fasta_file data/proteins.fasta \
    --output_dir data/embedding_cache/prott5 \
    --batch_size 8
```

**Output:** `data/embedding_cache/prott5/{protein_id}.npy`

---

## Data Preparation (`prepare_cafa3_data.py`)

Prepares CAFA3 benchmark data splits.

**Usage:**
```bash
python scripts/prepare_cafa3_data.py
```

**Output:**
- `data/{aspect}_{split}_names.npy` - Protein IDs
- `data/{aspect}_{split}_labels.npz` - GO term labels
- `data/{aspect}_{split}_sequences.json` - Protein sequences

---

## Expected Coverage

Based on CAFA3 benchmark data:

| Modality | Train | Valid | Test |
|----------|-------|-------|------|
| Text | 100% | 100% | 100% |
| Structure | ~98% | ~98% | ~90% |
| PPI | ~83% | ~83% | ~87% |
| ProtT5/ESM | 100% | 100% | 100% |

---

## Troubleshooting

### Common Issues

1. **CAFA Assessment Tool not found**
   - Download from: https://github.com/ashleyzhou972/CAFA_assessment_tool
   - Update path in script configuration

2. **STRING files not found**
   - Download from STRING database: https://string-db.org/cgi/download
   - Required files: `protein.network.embeddings.v12.0.h5`, `protein.aliases.v12.0.txt`

3. **API rate limiting (429 errors)**
   - Scripts include built-in rate limiting
   - For text extraction, reduce `max_workers` if needed

4. **GPU out of memory**
   - Reduce `--batch_size` parameter
   - Use `--device cpu` for CPU-only execution
