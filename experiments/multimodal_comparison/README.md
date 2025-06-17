# Multi-Modal Gene Ontology Prediction: Comparative Analysis

Experimental pipeline for systematically evaluating different feature modalities and their combinations for protein function prediction using Gene Ontology terms.

## Overview

The experiment compares:
- **Single Modalities**: ESM embeddings, Text embeddings, Structure-based features
- **Multi-Modal Combinations**: Various fusion strategies for combining features
- **Graph Representations**: Different methods for encoding protein structures

## Experimental Design

### Model Groups

#### Group A: Baseline Models (Single Modality)
- **Model A**: ESM-only baseline
- **Model B**: Text-only baseline

#### Group B: Structure-Only Models  
Radius or KNN methods for graph construction, one-hot encoding or esm for node features in the graph representations.
- **Model C1**: Radius Graph + One-Hot encoding 
- **Model C2**: Radius Graph + ESM features
- **Model C3**: k-NN Graph + One-Hot encoding
- **Model C4**: k-NN Graph + ESM features

#### Group C: Multi-Modal Combinations
- **Model D**: ESM + Text (Concatenate fusion)
- **Model E**: ESM + Best Structure (Concatenate fusion)
- **Model F**: ESM + Text + Structure (Full model concatenate fusion)

## Setup Instructions

### 1. Environment Setup

```bash
# Activate the training environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train
```

### 2. Create Aligned Datasets

First, create datasets where all proteins have all required modalities:

```bash
cd /SAN/bioinf/PFP/PFP/experiments/multimodal_comparison
python experiment_pipeline.py --action align
```

This will:
- Find proteins with ESM, Text, and Structure data available
- Create aligned train/validation sets
- Save alignment statistics

### 3. Generate Experiment Configurations

```bash
python experiment_pipeline.py --action generate \
    --base-config /SAN/bioinf/PFP/PFP/configs/base_multimodal.yaml
```

This creates:
- Individual configuration files for each experiment
- Cluster submission scripts
- Monitoring scripts

### 4. Submit Experiments

Submit experiments in phases:

```bash
# Submit baseline and structure experiments
./scripts/submit_all.sh

# Monitor progress
./scripts/monitor.sh

# After structure experiments complete, determine best config and submit multi-modal
./scripts/submit_multimodal.sh
```

### 5. Analyze Results

Once experiments complete:

```bash
python experiment_pipeline.py --action analyze
```

This generates:
- Comprehensive analysis report
- Publication-ready figures
- Performance comparison tables

## Directory Structure

```
/SAN/bioinf/PFP/PFP/experiments/multimodal_comparison/
├── experiment_pipeline.py      # Main pipeline script
├── train_unified.py           # Unified training script
├── aligned_data/              # Aligned datasets
│   ├── BPO_aligned_*.npy
│   ├── CCO_aligned_*.npy
│   └── MFO_aligned_*.npy
├── configs/                   # Experiment configurations
│   ├── A_ESM_only_*.yaml
│   ├── B_Text_only_*.yaml
│   └── ...
├── scripts/                   # Submission scripts
│   ├── submit_all.sh
│   ├── submit_multimodal.sh
│   └── monitor.sh
├── logs/                      # Cluster job logs
├── results/                   # Experiment results
│   └── {experiment_name}/
│       ├── final_model.pt
│       ├── final_metrics.json
│       └── training_history.csv
└── analysis/                  # Analysis outputs
    ├── experiment_report.md
    ├── all_results.csv
    └── figures/

```

## Key Metrics

The experiments track:
- **F-max**: Maximum F-score across thresholds (protein-centric)
- **AUPR**: Area Under Precision-Recall curve
- **AUROC**: Area Under ROC curve
- **Coverage**: Fraction of true terms predicted
- **mAP**: Mean Average Precision

## Expected Outcomes

1. **Performance Ranking**: Determine which feature combinations work best
2. **Structure Analysis**: Identify optimal graph construction methods
3. **Fusion Strategy**: Compare different multi-modal fusion approaches
4. **GO Aspect Differences**: Understand performance variations across MF/BP/CC

## Computational Requirements

- **GPU**: Required for all experiments
- **Memory**: 48GB recommended
- **Time**: ~24 hours per experiment
- **Storage**: ~50GB for all results

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Check alignment report for coverage statistics
   - Verify paths in configuration files

2. **Out of Memory**
   - Reduce batch size in config
   - Enable gradient accumulation

3. **Failed Jobs**
   - Check logs in `logs/` directory
   - Verify GPU allocation
