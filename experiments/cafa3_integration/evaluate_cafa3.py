#!/usr/bin/env python3
"""
Updated CAFA3 Evaluation Script with Information Accretion support and proper metric mapping
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/evaluate_cafa3_with_ia.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import scipy.sparse as ssp
import json

# Add cafaeval to path if needed
sys.path.append('/SAN/bioinf/PFP/PFP')

import cafaeval
from cafaeval.evaluation import cafa_eval, write_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metric name mapping for better readability
METRIC_NAME_MAPPING = {
    'f': 'F-measure',
    's': 'S-measure (Semantic Distance)',
    'pr': 'Precision',
    'rc': 'Recall',
    'cov': 'Coverage',
    'mi': 'Misinformation',
    'ru': 'Remaining Uncertainty',
    'pr_micro': 'Precision (Micro)',
    'rc_micro': 'Recall (Micro)',
    'f_micro': 'F-measure (Micro)',
    'f_w': 'F-measure (IA-weighted)',
    's_w': 'S-measure (IA-weighted)',
    'pr_w': 'Precision (IA-weighted)',
    'rc_w': 'Recall (IA-weighted)',
    'cov_w': 'Coverage (IA-weighted)',
    'mi_w': 'Misinformation (IA-weighted)',
    'ru_w': 'Remaining Uncertainty (IA-weighted)',
    'pr_micro_w': 'Precision (Micro, IA-weighted)',
    'rc_micro_w': 'Recall (Micro, IA-weighted)',
    'f_micro_w': 'F-measure (Micro, IA-weighted)',
    'tau': 'Threshold',
    'n': 'Number of Predictions',
    'tp': 'True Positives',
    'fp': 'False Positives',
    'fn': 'False Negatives',
    'n_w': 'Number of Predictions (Weighted)',
    'tp_w': 'True Positives (Weighted)',
    'fp_w': 'False Positives (Weighted)',
    'fn_w': 'False Negatives (Weighted)',
    'cov_max': 'Maximum Coverage'
}


def format_metric_name(metric_key):
    """Convert metric key to readable name."""
    return METRIC_NAME_MAPPING.get(metric_key, metric_key)


def extract_best_metrics_from_results(results_df, aspect):
    """Extract best performance metrics from cafaeval results."""
    
    best_metrics = {}
    results_df = results_df.reset_index()     # <── this line fixes the KeyError

    # Find best F-measure
    if 'f' in results_df.columns:
        best_f_idx = results_df['f'].idxmax()
        best_f_row = results_df.loc[best_f_idx]

        best_metrics['f_max'] = {
            'value': float(best_f_row['f']),
            'threshold': float(best_f_row['tau']),
            'precision': float(best_f_row['pr']),
            'recall': float(best_f_row['rc']),
            'coverage': float(best_f_row['cov']),
            'model': best_f_row['filename']
        }
    
    # Find best weighted F-measure
    if 'f_w' in results_df.columns:
        best_fw_idx = results_df['f_w'].idxmax()
        best_fw_row = results_df.loc[best_fw_idx]
        best_metrics['f_max_weighted'] = {
            'value': float(best_fw_row['f_w']),
            'threshold': float(best_fw_row['tau']),
            'precision': float(best_fw_row['pr_w']),
            'recall': float(best_fw_row['rc_w']),
            'coverage': float(best_fw_row['cov_w']),
            'model': best_fw_row['filename']
        }
    
    # Find best S-measure
    if 's' in results_df.columns:
        best_s_idx = results_df['s'].idxmin()  # S-measure should be minimized
        best_s_row = results_df.loc[best_s_idx]
        best_metrics['s_min'] = {
            'value': float(best_s_row['s']),
            'threshold': float(best_s_row['tau']),
            'remaining_uncertainty': float(best_s_row['ru']),
            'misinformation': float(best_s_row['mi']),
            'model': best_s_row['filename']
        }
    
    # Find best weighted S-measure
    if 's_w' in results_df.columns:
        best_sw_idx = results_df['s_w'].idxmin()
        best_sw_row = results_df.loc[best_sw_idx]
        best_metrics['s_min_weighted'] = {
            'value': float(best_sw_row['s_w']),
            'threshold': float(best_sw_row['tau']),
            'remaining_uncertainty': float(best_sw_row['ru_w']),
            'misinformation': float(best_sw_row['mi_w']),
            'model': best_sw_row['filename']
        }
    
    return best_metrics


def run_cafa_evaluation_with_ia(aspect: str, experiment_dir: str, ia_dir: str):
    """Run CAFA evaluation with Information Accretion for a specific aspect."""
    
    experiment_dir = Path(experiment_dir)
    ia_dir = Path(ia_dir)
    
    # Paths
    ontology_file = "/SAN/bioinf/PFP/dataset/zenodo/go.obo"
    predictions_dir = experiment_dir / "predictions"
    data_dir = experiment_dir / "data"
    output_dir = experiment_dir / "evaluation_with_ia" / aspect
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # IA file path
    ia_file = ia_dir / f"{aspect}_ia.txt"
    if not ia_file.exists():
        logger.warning(f"IA file not found: {ia_file}")
        logger.info("Generating IA file...")
        from generate_ia import InformationAccretionGenerator
        generator = InformationAccretionGenerator(
            cafa3_dir="/SAN/bioinf/PFP/dataset/zenodo",
            output_dir=str(ia_dir),
            aspect=aspect
        )
        ia_file, _ = generator.generate_ia_file()
    
    # Validate IA file format
    from generate_ia import validate_ia_file
    if not validate_ia_file(str(ia_file)):
        raise ValueError(f"IA file validation failed: {ia_file}")
    
    # Prepare ground truth file
    logger.info(f"Preparing ground truth for {aspect}...")
    
    test_names = np.load(data_dir / f"{aspect}_test_names.npy", allow_pickle=True)
    test_labels_sparse = ssp.load_npz(data_dir / f"{aspect}_test_labels.npz")
    
    with open(data_dir / f"{aspect}_go_terms.json", 'r') as f:
        go_terms = json.load(f)
    
    # Create ground truth file
    gt_file = output_dir / "ground_truth.tsv"
    with open(gt_file, 'w') as f:
        for i, protein_id in enumerate(test_names):
            row = test_labels_sparse.getrow(i)
            for col_id, val in zip(row.indices, row.data):
                if val > 0:
                    f.write(f"{protein_id}\t{go_terms[col_id]}\n")
    
    logger.info(f"Running CAFA evaluation with IA for {aspect}...")
    logger.info(f"Using IA file: {ia_file}")
    
    # Run evaluation with IA
    try:
        results = cafa_eval(
            ontology_file,
            str(predictions_dir),
            str(gt_file),
            norm='cafa',
            prop='max',
            th_step=0.01,
            ia=str(ia_file)
        )
        
        # Process results
        if isinstance(results, tuple):
            results_data, best_scores = results
        else:
            results_data = results
            best_scores = {}
        
        # Save raw results
        raw_results_file = output_dir / "raw_results.json"
        with open(raw_results_file, 'w') as f:
            if isinstance(results_data, pd.DataFrame):
                json.dump(results_data.to_dict('records'), f, indent=2)
            else:
                json.dump(results_data, f, indent=2)
        
        # Extract and save best metrics
        if isinstance(results_data, pd.DataFrame):
            best_metrics = extract_best_metrics_from_results(results_data, aspect)
            
            best_metrics_file = output_dir / "best_metrics.json"
            with open(best_metrics_file, 'w') as f:
                json.dump(best_metrics, f, indent=2)
            
            # Save formatted results
            save_formatted_results(results_data, output_dir, aspect)
        
        logger.info(f"Evaluation with IA complete for {aspect}")
        
        # Save comparison results
        save_ia_comparison(aspect, experiment_dir, output_dir, best_metrics if 'best_metrics' in locals() else {})
            
    except Exception as e:
        logger.error(f"Error evaluating {aspect} with IA: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_formatted_results(results_df, output_dir, aspect):
    """Save formatted evaluation results."""
    results_df= results_df.reset_index() 
    # Best F-measure results
    if 'f' in results_df.columns:
        f_results = results_df.nlargest(10, 'f')[['filename', 'tau', 'f', 'pr', 'rc', 'cov', 'n']]
        f_results.columns = ['Model', 'Threshold', 'F-measure', 'Precision', 'Recall', 'Coverage', 'Predictions']
        f_results.to_csv(output_dir / "best_f_measure.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best weighted F-measure results
    if 'f_w' in results_df.columns:
        fw_results = results_df.nlargest(10, 'f_w')[['filename', 'tau', 'f_w', 'pr_w', 'rc_w', 'cov_w', 'n_w']]
        fw_results.columns = ['Model', 'Threshold', 'F-measure (Weighted)', 'Precision (Weighted)', 
                             'Recall (Weighted)', 'Coverage (Weighted)', 'Predictions (Weighted)']
        fw_results.to_csv(output_dir / "best_f_measure_weighted.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best S-measure results (lower is better)
    if 's' in results_df.columns:
        s_results = results_df.nsmallest(10, 's')[['filename', 'tau', 's', 'ru', 'mi']]
        s_results.columns = ['Model', 'Threshold', 'S-measure', 'Remaining Uncertainty', 'Misinformation']
        s_results.to_csv(output_dir / "best_s_measure.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best weighted S-measure results
    if 's_w' in results_df.columns:
        sw_results = results_df.nsmallest(10, 's_w')[['filename', 'tau', 's_w', 'ru_w', 'mi_w']]
        sw_results.columns = ['Model', 'Threshold', 'S-measure (Weighted)', 
                              'Remaining Uncertainty (Weighted)', 'Misinformation (Weighted)']
        sw_results.to_csv(output_dir / "best_s_measure_weighted.tsv", sep='\t', index=False, float_format='%.4f')


def save_ia_comparison(aspect: str, experiment_dir: Path, ia_output_dir: Path, best_metrics: dict):
    """Compare results with and without IA."""
    
    comparison_file = ia_output_dir / "ia_comparison.json"
    comparison = {'aspect': aspect}
    
    # Add best metrics from IA evaluation
    if best_metrics:
        comparison['with_ia'] = best_metrics
    
    # Load results without IA (if available)
    standard_eval_dir = experiment_dir / "evaluation" / aspect
    if standard_eval_dir.exists():
        # Try to load previous evaluation results
        standard_results_file = standard_eval_dir / "best_metrics.json"
        if standard_results_file.exists():
            with open(standard_results_file, 'r') as f:
                comparison['without_ia'] = json.load(f)
        else:
            # Try to extract from TSV files
            fmax_file = standard_eval_dir / "evaluation_best_f.tsv"
            if fmax_file.exists():
                df_no_ia = pd.read_csv(fmax_file, sep='\t')
                if len(df_no_ia) > 0:
                    comparison['without_ia'] = {
                        'f_max': {
                            'value': float(df_no_ia.iloc[0].get('Fmax', df_no_ia.iloc[0].get('f', 0))),
                            'model': df_no_ia.iloc[0].get('filename', 'unknown')
                        }
                    }
    
    # Calculate impact
    if 'with_ia' in comparison and 'without_ia' in comparison:
        if 'f_max' in comparison['with_ia'] and 'f_max' in comparison['without_ia']:
            f_with = comparison['with_ia']['f_max']['value']
            f_without = comparison['without_ia']['f_max']['value']
            comparison['impact'] = {
                'f_max_difference': f_with - f_without,
                'f_max_percent_change': ((f_with - f_without) / f_without * 100) if f_without > 0 else 0
            }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
        
    logger.info(f"IA comparison saved to {comparison_file}")


def create_comprehensive_evaluation_report(experiment_dir: str):
    """Create comprehensive evaluation report including IA metrics."""
    
    experiment_dir = Path(experiment_dir)
    eval_dir = experiment_dir / "evaluation_with_ia"
    
    report_lines = ["# CAFA3 Evaluation Report with Information Accretion\n\n"]
    report_lines.append("## Executive Summary\n\n")
    
    # Collect summary data
    summary_data = []
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_dir = eval_dir / aspect
        if not aspect_dir.exists():
            continue
            
        # Load best metrics
        best_metrics_file = aspect_dir / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file, 'r') as f:
                metrics = json.load(f)
                
            row = {'Aspect': aspect}
            
            # Standard metrics
            if 'f_max' in metrics:
                row['F-max'] = f"{metrics['f_max']['value']:.4f}"
                row['Best Model (F)'] = metrics['f_max']['model'].replace('.tsv', '')
                row['Threshold (F)'] = f"{metrics['f_max']['threshold']:.2f}"
                
            # IA-weighted metrics
            if 'f_max_weighted' in metrics:
                row['F-max (IA-weighted)'] = f"{metrics['f_max_weighted']['value']:.4f}"
                row['Best Model (F-IA)'] = metrics['f_max_weighted']['model'].replace('.tsv', '')
                
            # S-measure
            if 's_min' in metrics:
                row['S-min'] = f"{metrics['s_min']['value']:.4f}"
                
            if 's_min_weighted' in metrics:
                row['S-min (IA-weighted)'] = f"{metrics['s_min_weighted']['value']:.4f}"
                
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_lines.append(summary_df.to_markdown(index=False))
        report_lines.append("\n\n")
    
    # IA Impact Analysis
    report_lines.append("## Information Accretion Impact Analysis\n\n")
    
    impact_data = []
    for aspect in ['BPO', 'CCO', 'MFO']:
        comparison_file = eval_dir / aspect / "ia_comparison.json"
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                comp = json.load(f)
                
            if 'impact' in comp:
                impact_data.append({
                    'Aspect': aspect,
                    'F-max Change': f"{comp['impact']['f_max_difference']:.4f}",
                    'F-max Change (%)': f"{comp['impact']['f_max_percent_change']:.2f}%"
                })
    
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        report_lines.append(impact_df.to_markdown(index=False))
        report_lines.append("\n\n")
    
    # IA Statistics
    report_lines.append("## Information Accretion Statistics\n\n")
    
    ia_stats_data = []
    for aspect in ['BPO', 'CCO', 'MFO']:
        stats_file = experiment_dir / "ia_files" / f"{aspect}_ia_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                ia_stats_data.append({
                    'Aspect': aspect,
                    'GO Terms': stats['n_terms'],
                    'Mean IC': f"{stats['mean_ic']:.3f}",
                    'Std IC': f"{stats['std_ic']:.3f}",
                    'Min IC': f"{stats['min_ic']:.3f}",
                    'Max IC': f"{stats['max_ic']:.3f}"
                })
    
    if ia_stats_data:
        ia_stats_df = pd.DataFrame(ia_stats_data)
        report_lines.append(ia_stats_df.to_markdown(index=False))
        report_lines.append("\n\n")
    
    # Detailed results for each aspect
    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_dir = eval_dir / aspect
        if not aspect_dir.exists():
            continue
            
        report_lines.append(f"## {aspect} Detailed Results\n\n")
        
        # Best F-measure
        f_file = aspect_dir / "best_f_measure.tsv"
        if f_file.exists():
            df = pd.read_csv(f_file, sep='\t')
            report_lines.append("### Top 5 Models by F-measure\n\n")
            report_lines.append(df.head().to_markdown(index=False))
            report_lines.append("\n\n")
        
        # Best weighted F-measure
        fw_file = aspect_dir / "best_f_measure_weighted.tsv"
        if fw_file.exists():
            df = pd.read_csv(fw_file, sep='\t')
            report_lines.append("### Top 5 Models by F-measure (IA-weighted)\n\n")
            report_lines.append(df.head().to_markdown(index=False))
            report_lines.append("\n\n")
    
    # Metric Descriptions
    report_lines.append("## Metric Descriptions\n\n")
    report_lines.append("- **F-measure**: Harmonic mean of precision and recall\n")
    report_lines.append("- **S-measure**: Semantic distance-based measure (lower is better)\n")
    report_lines.append("- **IA-weighted**: Metrics weighted by Information Accretion (term specificity)\n")
    report_lines.append("- **Coverage**: Fraction of proteins with at least one prediction\n")
    report_lines.append("- **Remaining Uncertainty**: Information content not captured by predictions\n")
    report_lines.append("- **Misinformation**: Information content of incorrect predictions\n")
    
    # Save report
    report_file = eval_dir / "evaluation_report_with_ia.md"
    with open(report_file, 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"Comprehensive evaluation report saved to {report_file}")
    
    # Also create a simplified summary report
    create_simplified_summary(experiment_dir)


def create_simplified_summary(experiment_dir: Path):
    """Create a simplified summary for quick reference."""
    
    eval_dir = experiment_dir / "evaluation_with_ia"
    summary_file = eval_dir / "summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("CAFA3 Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            best_metrics_file = eval_dir / aspect / "best_metrics.json"
            if best_metrics_file.exists():
                with open(best_metrics_file, 'r') as mf:
                    metrics = json.load(mf)
                    
                f.write(f"{aspect}:\n")
                if 'f_max' in metrics:
                    f.write(f"  F-max: {metrics['f_max']['value']:.4f} ({metrics['f_max']['model']})\n")
                if 'f_max_weighted' in metrics:
                    f.write(f"  F-max (IA): {metrics['f_max_weighted']['value']:.4f} ({metrics['f_max_weighted']['model']})\n")
                f.write("\n")
    
    logger.info(f"Summary saved to {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/cafa3_integration',
                       help='Experiment directory')
    parser.add_argument('--ia-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/ia_files',
                       help='Directory containing IA files')
    parser.add_argument('--aspect', type=str, choices=['BPO', 'CCO', 'MFO'],
                       help='Evaluate specific aspect only')
    parser.add_argument('--generate-ia', action='store_true',
                       help='Generate IA files before evaluation')
    
    args = parser.parse_args()
    
    # Generate IA files if requested
    if args.generate_ia:
        from generate_ia import generate_all_ia_files
        logger.info("Generating Information Accretion files...")
        generate_all_ia_files(
            cafa3_dir="/SAN/bioinf/PFP/dataset/zenodo",
            output_dir=args.ia_dir
        )
    
    if args.aspect:
        run_cafa_evaluation_with_ia(args.aspect, args.experiment_dir, args.ia_dir)
    else:
        # Evaluate all aspects
        for aspect in ['BPO', 'CCO', 'MFO']:
            logger.info(f"\nEvaluating {aspect} with IA...")
            run_cafa_evaluation_with_ia(aspect, args.experiment_dir, args.ia_dir)
    
    # Create summary report
    create_comprehensive_evaluation_report(args.experiment_dir)


if __name__ == "__main__":
    main()