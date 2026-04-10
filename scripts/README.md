# MMFP Reproduction Scripts

This directory contains scripts to rebuild MMFP inputs and reproduce all paper results.

## Reproduction Scripts

| Paper Table | Script | Description |
|------------|--------|-------------|
| Table 1 | `reproduce_full_model.py` | Full model train + eval with temporal text embeddings |
| Table 2 | `reproduce_modality_contribution.py` | Retrain with modality subsets (seq_only, seq_text, seq_struct, seq_ppi) |
| Table 3 | `run_ablation.py` | Ablation over fusion variants (hybrid, concat, bilinear, aux_only) |
| Table 4 | `reproduce_masking_eval.py` | Single-modality masking evaluation using trained checkpoints |

## Core Reproduction Paths

### 1. Full Model (Table 1)

Prepare the canonical mixed text bundle:

```bash
python scripts/extract_uniprot_text.py prepare-temporal-text
```

This produces:
- `data/embedding_cache/uniprot_text/protein_descriptions.tsv`
- `data/embedding_cache/uniprot_text/protein_descriptions_historical.tsv`
- `data/embedding_cache/uniprot_text/temporal_recipe/protein_descriptions_historical_punct_v1_test.tsv`
- `data/embedding_cache/uniprot_text/temporal_recipe/protein_descriptions_mixed.tsv`

Train from scratch with the temporal text cache:

```bash
python train.py \
  --seq-model prott5 \
  --fusion-types gated_bilinear \
  --aspects BPO CCO MFO \
  --use-late-fusion \
  --text-embedding-dir data/embedding_cache/exp_text_embeddings_temporal \
  --output-base results/full_model \
  --num-workers 0 \
  --seed 42
```

Verify against the kept target metrics:

```bash
python scripts/reproduce_full_model.py
```

### 2. Modality Contribution (Table 2)

```bash
python scripts/reproduce_modality_contribution.py
```

Retrains 4 variants: seq_only, seq_text, seq_struct, seq_ppi. Results in `results/modality_contribution/`.

### 3. Ablation Study (Table 3)

```bash
python scripts/run_ablation.py
```

Retrains fusion variants: hybrid_gated, concat_only, concat_plus_aux, bilinear_gated, aux_only. Results in `results/ablation/`.

### 4. Masking Evaluation (Table 4)

```bash
python scripts/reproduce_masking_eval.py
```

Uses full model checkpoints, evaluates with single modalities masked. Results in `results/masking_eval/`.

## Embedding Extraction Scripts

| Modality | Script(s) | Output Dimension | Description |
|----------|-----------|------------------|-------------|
| **PPI** | `extract_ppi_embeddings.py` | 512-D | STRING protein-protein interaction network embeddings |
| **Text** | `extract_uniprot_text.py` + `embed_uniprot_descriptions.py` | 768-D | Current + historical UniProt text and PubMedBERT embeddings |
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

### Environment Variables

Scripts that previously had hardcoded paths now use CLI arguments or environment variables:

| Variable | Used By | Description |
|----------|---------|-------------|
| `CAFA_ASSESSMENT_DIR` | `extract_uniprot_text.py`, `extract_ppi_embeddings.py`, `check_alphafold_coverage.py` | Path to CAFA assessment tool |
| `CAFA3_RAW_DIR` | `prepare_cafa3_data.py` | Path to raw CAFA3 CSV files |
| `STRING_H5_FILE` | `extract_ppi_embeddings.py` | Path to STRING embeddings h5 file |
| `STRING_ALIAS_FILE` | `extract_ppi_embeddings.py` | Path to STRING alias file |

All scripts also accept these as CLI arguments (run with `--help` to see options).

---

## Script Details

### 1. PPI Embeddings (`extract_ppi_embeddings.py`)

Extracts 512-D protein-protein interaction embeddings from STRING database.

**Usage:**
```bash
python scripts/extract_ppi_embeddings.py \
  --string-h5 /path/to/protein.network.embeddings.v12.0.h5 \
  --string-alias /path/to/protein.aliases.v12.0.txt \
  --cafa-assessment-dir /path/to/CAFA_assessment_tool
```

**Output:** `data/embedding_cache/ppi/{protein_id}.npy`

---

### 2. Text Embeddings

#### Step 1: Extract Text (`extract_uniprot_text.py`)

Supports:
- current UniProt text extraction
- historical UniSave extraction for the test split with a `2016-02-17` cutoff
- canonical temporal text assembly

**Usage:**
```bash
python scripts/extract_uniprot_text.py extract-current
python scripts/extract_uniprot_text.py extract-historical
python scripts/extract_uniprot_text.py prepare-temporal-text
```

**Outputs:**
- `data/embedding_cache/uniprot_text/protein_descriptions.tsv`
- `data/embedding_cache/uniprot_text/protein_descriptions_historical.tsv`
- `data/embedding_cache/uniprot_text/temporal_recipe/*.tsv`

#### Step 2: Generate Embeddings (`embed_uniprot_descriptions.py`)

Generates PubMedBERT embeddings from extracted text descriptions.

**Usage:**
```bash
python scripts/embed_uniprot_descriptions.py --data-dir data
```

**Primary outputs:**
- `data/embedding_cache/exp_text_embeddings/{protein_id}.npy`
- `data/embedding_cache/exp_text_embeddings_temporal/{protein_id}.npy`

---

### 3. Structure Embeddings

#### Step 1: Download PDB Files (`check_alphafold_coverage.py`)

```bash
python scripts/check_alphafold_coverage.py \
  --cafa-assessment-dir /path/to/CAFA_assessment_tool
```

**Output:** `data/alphafold_structures/{protein_id}.pdb`

#### Step 2: Generate ESM-IF1 Embeddings (`extract_esm_if1_embeddings.py`)

```bash
python scripts/extract_esm_if1_embeddings.py \
    --pdb_dir data/alphafold_structures \
    --output_dir data/embedding_cache/IF1 \
    --pooling mean \
    --device cuda
```

**Output:** `data/embedding_cache/IF1/{protein_id}.npy`

---

### 4. ProtT5 Embeddings (`extract_prott5_embeddings.py`)

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

```bash
python scripts/prepare_cafa3_data.py --cafa3-dir /path/to/cafa3
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
