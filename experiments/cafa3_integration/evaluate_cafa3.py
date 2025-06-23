#!/usr/bin/env python3
"""
CAFA3 Evaluation Script
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/evaluate_cafa3.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add cafaeval to path if needed
sys.path.append('/SAN/bioinf/PFP/PFP')

import cafaeval
from cafaeval.evaluation import cafa_eval, write_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cafa_evaluation(aspect: str, experiment_dir: str):
    """Run CAFA evaluation for a specific aspect."""
    
    experiment_dir = Path(experiment_dir)
    
    # Paths
    ontology_file = "/SAN/bioinf/PFP/dataset/zenodo/go.obo"
    predictions_dir = experiment_dir / "predictions"
    data_dir = experiment_dir / "data"
    output_dir = experiment_dir / "evaluation" / aspect
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare ground truth file
    logger.info(f"Preparing ground truth for {aspect}...")
    
    test_names = np.load(data_dir / f"{aspect}_test_names.npy", allow_pickle=True)
    test_labels_sparse = cafaeval.ssp.load_npz(data_dir / f"{aspect}_test_labels.npz")
    
    import json
    with open(data_dir / f"{aspect}_go_terms.json", 'r') as f:
        go_terms = json.load(f)
    
    # Create ground truth file
    gt_file = output_dir / "ground_truth.tsv"
    with open(gt_file, 'w') as f:
        for i, protein_id in enumerate(test_names):
            row = test_labels_sparse.getrow(i)
            for j in row.indices:
                if row.data[j] > 0:  # Only positive annotations
                    f.write(f"{protein_id}\t{go_terms[j]}\n")
    
    logger.info(f"Running CAFA evaluation for {aspect}...")
    
    # Run evaluation
    try:
        results, best_scores = cafa_eval(
            ontology_file,
            str(predictions_dir),
            str(gt_file),
            norm='cafa',
            prop='max',
            th_step=0.01
        )
        
        # Write results
        write_results(results, best_scores, str(output_dir))
        
        logger.info(f"Evaluation complete for {aspect}")
        
        # Print summary
        print(f"\n{aspect} Results Summary:")
        print("=" * 50)
        
        for metric, df in best_scores.items():
            print(f"\nBest {metric}:")
            print(df.head())
            
    except Exception as e:
        logger.error(f"Error evaluating {aspect}: {e}")
        raise


def create_evaluation_report(experiment_dir: str):
    """Create comprehensive evaluation report."""
    
    experiment_dir = Path(experiment_dir)
    eval_dir = experiment_dir / "evaluation"
    
    report_lines = ["# CAFA3 Evaluation Report\n\n"]
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_dir = eval_dir / aspect
        if not aspect_dir.exists():
            continue
            
        report_lines.append(f"## {aspect} Results\n\n")
        
        # Load best F-measure results
        fmax_file = aspect_dir / "evaluation_best_f.tsv"
        if fmax_file.exists():
            df = pd.read_csv(fmax_file, sep='\t')
            report_lines.append("### Top 5 Models by F-max\n\n")
            report_lines.append(df.head().to_markdown())
            report_lines.append("\n\n")
    
    # Save report
    report_file = eval_dir / "evaluation_report.md"
    with open(report_file, 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"Evaluation report saved to {report_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/cafa3_integration',
                       help='Experiment directory')
    parser.add_argument('--aspect', type=str, choices=['BPO', 'CCO', 'MFO'],
                       help='Evaluate specific aspect only')
    
    args = parser.parse_args()
    
    if args.aspect:
        run_cafa_evaluation(args.aspect, args.experiment_dir)
    else:
        # Evaluate all aspects
        for aspect in ['BPO', 'CCO', 'MFO']:
            logger.info(f"\nEvaluating {aspect}...")
            run_cafa_evaluation(aspect, args.experiment_dir)
    
    # Create summary report
    create_evaluation_report(args.experiment_dir)


if __name__ == "__main__":
    main()