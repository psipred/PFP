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
from collections import defaultdict

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


def extract_model_best_performance(results_df, model_name):
    """Extract best performance metrics for a specific model across all thresholds."""
    model_results = results_df[results_df['filename'] == model_name].copy()
    
    if model_results.empty:
        return None
    
    best_metrics = {}

    
    
    # Find best F-measure
    if 'f' in model_results.columns:
        best_f_idx = model_results['f'].idxmax()
        best_f_row = model_results.loc[best_f_idx]
        best_metrics['f_max'] = {
            'value': float(best_f_row['f']),
            'threshold': float(best_f_row['tau']),
            'precision': float(best_f_row['pr']),
            'recall': float(best_f_row['rc']),
            'coverage': float(best_f_row['cov'])
        }
    
    # Find best weighted F-measure
    if 'f_w' in model_results.columns:
        best_fw_idx = model_results['f_w'].idxmax()
        best_fw_row = model_results.loc[best_fw_idx]
        best_metrics['f_max_weighted'] = {
            'value': float(best_fw_row['f_w']),
            'threshold': float(best_fw_row['tau']),
            'precision': float(best_fw_row['pr_w']),
            'recall': float(best_fw_row['rc_w']),
            'coverage': float(best_fw_row['cov_w'])
        }
    
    # Find best S-measure (minimize)
    if 's' in model_results.columns:
        best_s_idx = model_results['s'].idxmin()
        best_s_row = model_results.loc[best_s_idx]
        best_metrics['s_min'] = {
            'value': float(best_s_row['s']),
            'threshold': float(best_s_row['tau']),
            'remaining_uncertainty': float(best_s_row['ru']),
            'misinformation': float(best_s_row['mi'])
        }
    
    # Find best weighted S-measure
    if 's_w' in model_results.columns:
        best_sw_idx = model_results['s_w'].idxmin()
        best_sw_row = model_results.loc[best_sw_idx]
        best_metrics['s_min_weighted'] = {
            'value': float(best_sw_row['s_w']),
            'threshold': float(best_sw_row['tau']),
            'remaining_uncertainty': float(best_sw_row['ru_w']),
            'misinformation': float(best_sw_row['mi_w'])
        }
    
    return best_metrics


def aggregate_best_metrics_across_models(results_df):
    """Get best performance for each model and rank them."""
    results_df = results_df.reset_index()
    
    # Get unique models
    models = results_df['filename'].unique()
    
    # Collect best metrics for each model
    model_best_metrics = {}
    for model in models:
        best_metrics = extract_model_best_performance(results_df, model)
        if best_metrics:
            model_best_metrics[model] = best_metrics
    
    # Create ranking tables
    rankings = {
        'f_max': [],
        'f_max_weighted': [],
        's_min': [],
        's_min_weighted': []
    }
    
    # Rank by F-measure
    for model, metrics in model_best_metrics.items():
        if 'f_max' in metrics:
            rankings['f_max'].append({
                'model': model,
                'value': metrics['f_max']['value'],
                'threshold': metrics['f_max']['threshold'],
                'precision': metrics['f_max']['precision'],
                'recall': metrics['f_max']['recall'],
                'coverage': metrics['f_max']['coverage']
            })
    
    # Rank by weighted F-measure
    for model, metrics in model_best_metrics.items():
        if 'f_max_weighted' in metrics:
            rankings['f_max_weighted'].append({
                'model': model,
                'value': metrics['f_max_weighted']['value'],
                'threshold': metrics['f_max_weighted']['threshold'],
                'precision': metrics['f_max_weighted']['precision'],
                'recall': metrics['f_max_weighted']['recall'],
                'coverage': metrics['f_max_weighted']['coverage']
            })
    
    # Rank by S-measure
    for model, metrics in model_best_metrics.items():
        if 's_min' in metrics:
            rankings['s_min'].append({
                'model': model,
                'value': metrics['s_min']['value'],
                'threshold': metrics['s_min']['threshold'],
                'remaining_uncertainty': metrics['s_min']['remaining_uncertainty'],
                'misinformation': metrics['s_min']['misinformation']
            })
    
    # Rank by weighted S-measure
    for model, metrics in model_best_metrics.items():
        if 's_min_weighted' in metrics:
            rankings['s_min_weighted'].append({
                'model': model,
                'value': metrics['s_min_weighted']['value'],
                'threshold': metrics['s_min_weighted']['threshold'],
                'remaining_uncertainty': metrics['s_min_weighted']['remaining_uncertainty'],
                'misinformation': metrics['s_min_weighted']['misinformation']
            })
    
    # Sort rankings
    rankings['f_max'].sort(key=lambda x: x['value'], reverse=True)
    rankings['f_max_weighted'].sort(key=lambda x: x['value'], reverse=True)
    rankings['s_min'].sort(key=lambda x: x['value'])
    rankings['s_min_weighted'].sort(key=lambda x: x['value'])
    
    return rankings, model_best_metrics


def extract_best_metrics_from_results(results_df, aspect):
    """Extract best performance metrics from cafaeval results."""
    
    rankings, model_best_metrics = aggregate_best_metrics_across_models(results_df)
    
    best_metrics = {}
    
    # Best F-measure across all models
    if rankings['f_max']:
        best_f = rankings['f_max'][0]
        best_metrics['f_max'] = {
            'value': best_f['value'],
            'threshold': best_f['threshold'],
            'precision': best_f['precision'],
            'recall': best_f['recall'],
            'coverage': best_f['coverage'],
            'model': best_f['model']
        }
    
    # Best weighted F-measure
    if rankings['f_max_weighted']:
        best_fw = rankings['f_max_weighted'][0]
        best_metrics['f_max_weighted'] = {
            'value': best_fw['value'],
            'threshold': best_fw['threshold'],
            'precision': best_fw['precision'],
            'recall': best_fw['recall'],
            'coverage': best_fw['coverage'],
            'model': best_fw['model']
        }
    
    # Best S-measure
    if rankings['s_min']:
        best_s = rankings['s_min'][0]
        best_metrics['s_min'] = {
            'value': best_s['value'],
            'threshold': best_s['threshold'],
            'remaining_uncertainty': best_s['remaining_uncertainty'],
            'misinformation': best_s['misinformation'],
            'model': best_s['model']
        }
    
    # Best weighted S-measure
    if rankings['s_min_weighted']:
        best_sw = rankings['s_min_weighted'][0]
        best_metrics['s_min_weighted'] = {
            'value': best_sw['value'],
            'threshold': best_sw['threshold'],
            'remaining_uncertainty': best_sw['remaining_uncertainty'],
            'misinformation': best_sw['misinformation'],
            'model': best_sw['model']
        }
    
    # Store full rankings
    best_metrics['rankings'] = rankings
    best_metrics['model_best_metrics'] = model_best_metrics
    
    return best_metrics


def save_formatted_results(results_df, output_dir, aspect, rankings):
    """Save formatted evaluation results with each model's best performance."""
    
    # Best F-measure results
    if rankings['f_max']:
        f_data = []
        for i, entry in enumerate(rankings['f_max'][:20]):  # Top 20 models
            f_data.append({
                'Rank': i + 1,
                'Model': entry['model'].replace('.tsv', ''),
                'F-measure': entry['value'],
                'Precision': entry['precision'],
                'Recall': entry['recall'],
                'Coverage': entry['coverage'],
                'Threshold': entry['threshold']
            })
        f_df = pd.DataFrame(f_data)
        f_df.to_csv(output_dir / "best_f_measure.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best weighted F-measure results
    if rankings['f_max_weighted']:
        fw_data = []
        for i, entry in enumerate(rankings['f_max_weighted'][:20]):
            fw_data.append({
                'Rank': i + 1,
                'Model': entry['model'].replace('.tsv', ''),
                'F-measure (Weighted)': entry['value'],
                'Precision (Weighted)': entry['precision'],
                'Recall (Weighted)': entry['recall'],
                'Coverage (Weighted)': entry['coverage'],
                'Threshold': entry['threshold']
            })
        fw_df = pd.DataFrame(fw_data)
        fw_df.to_csv(output_dir / "best_f_measure_weighted.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best S-measure results
    if rankings['s_min']:
        s_data = []
        for i, entry in enumerate(rankings['s_min'][:20]):
            s_data.append({
                'Rank': i + 1,
                'Model': entry['model'].replace('.tsv', ''),
                'S-measure': entry['value'],
                'Remaining Uncertainty': entry['remaining_uncertainty'],
                'Misinformation': entry['misinformation'],
                'Threshold': entry['threshold']
            })
        s_df = pd.DataFrame(s_data)
        s_df.to_csv(output_dir / "best_s_measure.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Best weighted S-measure results
    if rankings['s_min_weighted']:
        sw_data = []
        for i, entry in enumerate(rankings['s_min_weighted'][:20]):
            sw_data.append({
                'Rank': i + 1,
                'Model': entry['model'].replace('.tsv', ''),
                'S-measure (Weighted)': entry['value'],
                'Remaining Uncertainty (Weighted)': entry['remaining_uncertainty'],
                'Misinformation (Weighted)': entry['misinformation'],
                'Threshold': entry['threshold']
            })
        sw_df = pd.DataFrame(sw_data)
        sw_df.to_csv(output_dir / "best_s_measure_weighted.tsv", sep='\t', index=False, float_format='%.4f')
    
    # Save comprehensive model comparison
    save_model_comparison_table(output_dir, rankings)


def save_model_comparison_table(output_dir, rankings):
    """Create a comprehensive comparison table showing each model's best performance across metrics."""
    
    # Collect all models
    all_models = set()
    for metric_rankings in rankings.values():
        for entry in metric_rankings:
            all_models.add(entry['model'])
    
    # Build comparison data
    comparison_data = []
    for model in sorted(all_models):
        row = {'Model': model.replace('.tsv', '')}
        
        # Find model's performance in each metric
        for entry in rankings['f_max']:
            if entry['model'] == model:
                row['F-max'] = entry['value']
                row['F-max Rank'] = rankings['f_max'].index(entry) + 1
                break
        
        for entry in rankings['f_max_weighted']:
            if entry['model'] == model:
                row['F-max (IA)'] = entry['value']
                row['F-max (IA) Rank'] = rankings['f_max_weighted'].index(entry) + 1
                break
        
        for entry in rankings['s_min']:
            if entry['model'] == model:
                row['S-min'] = entry['value']
                row['S-min Rank'] = rankings['s_min'].index(entry) + 1
                break
        
        for entry in rankings['s_min_weighted']:
            if entry['model'] == model:
                row['S-min (IA)'] = entry['value']
                row['S-min (IA) Rank'] = rankings['s_min_weighted'].index(entry) + 1
                break
        
        comparison_data.append(row)
    
    # Sort by average rank
    for row in comparison_data:
        ranks = []
        for col in ['F-max Rank', 'F-max (IA) Rank', 'S-min Rank', 'S-min (IA) Rank']:
            if col in row:
                ranks.append(row[col])
        row['Avg Rank'] = np.mean(ranks) if ranks else 999
    
    comparison_data.sort(key=lambda x: x['Avg Rank'])
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "model_comparison.tsv", sep='\t', index=False, float_format='%.4f')


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
                # Don't save the full model metrics to keep file size reasonable
                metrics_to_save = {k: v for k, v in best_metrics.items() 
                                 if k not in ['rankings', 'model_best_metrics']}
                json.dump(metrics_to_save, f, indent=2)
            
            # Save formatted results with rankings
            save_formatted_results(results_data, output_dir, aspect, best_metrics['rankings'])
            
            # Save detailed model metrics
            model_metrics_file = output_dir / "model_best_metrics.json"
            with open(model_metrics_file, 'w') as f:
                json.dump(best_metrics['model_best_metrics'], f, indent=2)
        
        logger.info(f"Evaluation with IA complete for {aspect}")
        
        # Save comparison results
        save_ia_comparison(aspect, experiment_dir, output_dir, best_metrics if 'best_metrics' in locals() else {})
            
    except Exception as e:
        logger.error(f"Error evaluating {aspect} with IA: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_ia_comparison(aspect: str, experiment_dir: Path, ia_output_dir: Path, best_metrics: dict):
    """Compare results with and without IA."""
    
    comparison_file = ia_output_dir / "ia_comparison.json"
    comparison = {'aspect': aspect}
    
    # Add best metrics from IA evaluation
    if best_metrics:
        comparison['with_ia'] = {k: v for k, v in best_metrics.items() 
                               if k not in ['rankings', 'model_best_metrics']}
    
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
        comparison['impact'] = {}
        
        # F-measure impact
        if 'f_max' in comparison['with_ia'] and 'f_max' in comparison['without_ia']:
            f_with = comparison['with_ia']['f_max']['value']
            f_without = comparison['without_ia']['f_max']['value']
            comparison['impact']['f_max_difference'] = f_with - f_without
            comparison['impact']['f_max_percent_change'] = ((f_with - f_without) / f_without * 100) if f_without > 0 else 0
        
        # Check if IA changes which model performs best
        if 'f_max' in comparison['with_ia'] and 'f_max' in comparison['without_ia']:
            if comparison['with_ia']['f_max']['model'] != comparison['without_ia']['f_max']['model']:
                comparison['impact']['best_model_changed'] = True
                comparison['impact']['best_model_with_ia'] = comparison['with_ia']['f_max']['model']
                comparison['impact']['best_model_without_ia'] = comparison['without_ia']['f_max']['model']
    
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
                row = {
                    'Aspect': aspect,
                    'F-max Change': f"{comp['impact'].get('f_max_difference', 0):.4f}",
                    'F-max Change (%)': f"{comp['impact'].get('f_max_percent_change', 0):.2f}%"
                }
                
                if comp['impact'].get('best_model_changed', False):
                    row['Model Change'] = 'Yes'
                    row['Best w/o IA'] = comp['impact']['best_model_without_ia'].replace('.tsv', '')
                    row['Best w/ IA'] = comp['impact']['best_model_with_ia'].replace('.tsv', '')
                else:
                    row['Model Change'] = 'No'
                    
                impact_data.append(row)
    
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        report_lines.append(impact_df.to_markdown(index=False))
        report_lines.append("\n\n")
    
    # Model Performance Summary
    report_lines.append("## Model Performance Summary\n\n")
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_dir = eval_dir / aspect
        if not aspect_dir.exists():
            continue
            
        report_lines.append(f"### {aspect}\n\n")
        
        # Load model comparison
        comparison_file = aspect_dir / "model_comparison.tsv"
        if comparison_file.exists():
            df = pd.read_csv(comparison_file, sep='\t')
            # Show top 10 models
            report_lines.append("**Top 10 Models by Average Rank**\n\n")
            report_lines.append(df.head(10).to_markdown(index=False))
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
            report_lines.append("- **Threshold**: Confidence threshold used for predictions\n")
            report_lines.append("- **Rank**: Model's position in the ranking for each metric\n")
            report_lines.append("- **Avg Rank**: Average rank across all metrics (lower is better)\n")
            
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
       f.write("CAFA3 Evaluation Summary with Information Accretion\n")
       f.write("=" * 60 + "\n\n")
       
       # Overall best performers
       f.write("OVERALL BEST PERFORMERS\n")
       f.write("-" * 30 + "\n\n")
       
       best_models = defaultdict(lambda: defaultdict(dict))
       
       for aspect in ['BPO', 'CCO', 'MFO']:
           best_metrics_file = eval_dir / aspect / "best_metrics.json"
           if best_metrics_file.exists():
               with open(best_metrics_file, 'r') as mf:
                   metrics = json.load(mf)
                   
               f.write(f"{aspect}:\n")
               if 'f_max' in metrics:
                   model = metrics['f_max']['model'].replace('.tsv', '')
                   value = metrics['f_max']['value']
                   threshold = metrics['f_max']['threshold']
                   f.write(f"  Best F-max: {value:.4f} ({model} @ τ={threshold:.2f})\n")
                   best_models[model][aspect]['f_max'] = value
                   
               if 'f_max_weighted' in metrics:
                   model = metrics['f_max_weighted']['model'].replace('.tsv', '')
                   value = metrics['f_max_weighted']['value']
                   threshold = metrics['f_max_weighted']['threshold']
                   f.write(f"  Best F-max (IA): {value:.4f} ({model} @ τ={threshold:.2f})\n")
                   best_models[model][aspect]['f_max_weighted'] = value
                   
               if 's_min' in metrics:
                   model = metrics['s_min']['model'].replace('.tsv', '')
                   value = metrics['s_min']['value']
                   f.write(f"  Best S-min: {value:.4f} ({model})\n")
                   best_models[model][aspect]['s_min'] = value
                   
               if 's_min_weighted' in metrics:
                   model = metrics['s_min_weighted']['model'].replace('.tsv', '')
                   value = metrics['s_min_weighted']['value']
                   f.write(f"  Best S-min (IA): {value:.4f} ({model})\n")
                   best_models[model][aspect]['s_min_weighted'] = value
                   
               f.write("\n")
       
       # Models that appear most frequently as best
       f.write("\nMOST FREQUENT BEST PERFORMERS\n")
       f.write("-" * 30 + "\n")
       
       model_counts = defaultdict(int)
       for model, aspects in best_models.items():
           for aspect, metrics in aspects.items():
               model_counts[model] += len(metrics)
       
       sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
       for model, count in sorted_models[:5]:
           f.write(f"  {model}: {count} best scores\n")
       
       f.write("\n")
       
       # IA Impact Summary
       f.write("INFORMATION ACCRETION IMPACT\n")
       f.write("-" * 30 + "\n")
       
       for aspect in ['BPO', 'CCO', 'MFO']:
           comparison_file = eval_dir / aspect / "ia_comparison.json"
           if comparison_file.exists():
               with open(comparison_file, 'r') as cf:
                   comp = json.load(cf)
                   
               if 'impact' in comp:
                   f.write(f"{aspect}:\n")
                   if 'f_max_difference' in comp['impact']:
                       f.write(f"  F-max change: {comp['impact']['f_max_difference']:+.4f} ")
                       f.write(f"({comp['impact']['f_max_percent_change']:+.2f}%)\n")
                   
                   if comp['impact'].get('best_model_changed', False):
                       f.write(f"  Best model changed: ")
                       f.write(f"{comp['impact']['best_model_without_ia']} → ")
                       f.write(f"{comp['impact']['best_model_with_ia']}\n")
                   
                   f.write("\n")
   
   logger.info(f"Summary saved to {summary_file}")
   
   # Create a CSV summary for easy import into spreadsheets
   create_csv_summary(eval_dir)


def create_csv_summary(eval_dir: Path):
   """Create CSV summaries for easy analysis."""
   
   # Collect all best metrics across aspects
   all_metrics = []
   
   for aspect in ['BPO', 'CCO', 'MFO']:
       best_metrics_file = eval_dir / aspect / "best_metrics.json"
       if best_metrics_file.exists():
           with open(best_metrics_file, 'r') as f:
               metrics = json.load(f)
           
           # Standard metrics
           if 'f_max' in metrics:
               all_metrics.append({
                   'Aspect': aspect,
                   'Metric': 'F-max',
                   'Value': metrics['f_max']['value'],
                   'Model': metrics['f_max']['model'].replace('.tsv', ''),
                   'Threshold': metrics['f_max']['threshold'],
                   'Weighted': 'No'
               })
           
           if 'f_max_weighted' in metrics:
               all_metrics.append({
                   'Aspect': aspect,
                   'Metric': 'F-max',
                   'Value': metrics['f_max_weighted']['value'],
                   'Model': metrics['f_max_weighted']['model'].replace('.tsv', ''),
                   'Threshold': metrics['f_max_weighted']['threshold'],
                   'Weighted': 'Yes (IA)'
               })
           
           if 's_min' in metrics:
               all_metrics.append({
                   'Aspect': aspect,
                   'Metric': 'S-min',
                   'Value': metrics['s_min']['value'],
                   'Model': metrics['s_min']['model'].replace('.tsv', ''),
                   'Threshold': metrics['s_min']['threshold'],
                   'Weighted': 'No'
               })
           
           if 's_min_weighted' in metrics:
               all_metrics.append({
                   'Aspect': aspect,
                   'Metric': 'S-min',
                   'Value': metrics['s_min_weighted']['value'],
                   'Model': metrics['s_min_weighted']['model'].replace('.tsv', ''),
                   'Threshold': metrics['s_min_weighted']['threshold'],
                   'Weighted': 'Yes (IA)'
               })
   
   # Save CSV
   if all_metrics:
       metrics_df = pd.DataFrame(all_metrics)
       metrics_df.to_csv(eval_dir / "all_best_metrics.csv", index=False)
       
       # Create pivot table
       pivot_df = metrics_df.pivot_table(
           values='Value',
           index=['Model', 'Weighted'],
           columns=['Aspect', 'Metric'],
           aggfunc='first'
       )
       pivot_df.to_csv(eval_dir / "metrics_pivot.csv")
   
   logger.info("CSV summaries created")


def analyze_threshold_patterns(experiment_dir: Path):
   """Analyze optimal threshold patterns across models and metrics."""
   
   eval_dir = experiment_dir / "evaluation_with_ia"
   threshold_analysis = []
   
   for aspect in ['BPO', 'CCO', 'MFO']:
       model_metrics_file = eval_dir / aspect / "model_best_metrics.json"
       if model_metrics_file.exists():
           with open(model_metrics_file, 'r') as f:
               model_metrics = json.load(f)
           
           for model, metrics in model_metrics.items():
               for metric_name, metric_data in metrics.items():
                   if 'threshold' in metric_data:
                       threshold_analysis.append({
                           'Aspect': aspect,
                           'Model': model.replace('.tsv', ''),
                           'Metric': metric_name,
                           'Optimal_Threshold': metric_data['threshold'],
                           'Performance': metric_data['value']
                       })
   
   if threshold_analysis:
       threshold_df = pd.DataFrame(threshold_analysis)
       
       # Analyze threshold statistics
       threshold_stats = threshold_df.groupby(['Aspect', 'Metric'])['Optimal_Threshold'].agg([
           'mean', 'std', 'min', 'max'
       ]).round(3)
       
       threshold_stats.to_csv(eval_dir / "threshold_statistics.csv")
       
       # Find models with consistent thresholds
       model_threshold_consistency = threshold_df.groupby('Model')['Optimal_Threshold'].agg([
           'mean', 'std'
       ]).round(3)
       model_threshold_consistency['consistency'] = 1 / (1 + model_threshold_consistency['std'])
       model_threshold_consistency = model_threshold_consistency.sort_values('consistency', ascending=False)
       
       model_threshold_consistency.to_csv(eval_dir / "model_threshold_consistency.csv")
       
       logger.info("Threshold pattern analysis completed")


def main():
   import argparse
   
   parser = argparse.ArgumentParser(description="CAFA3 Evaluation with Information Accretion")
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
   parser.add_argument('--analyze-thresholds', action='store_true',
                      help='Perform threshold pattern analysis')
   
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
       # Evaluate single aspect
       run_cafa_evaluation_with_ia(args.aspect, args.experiment_dir, args.ia_dir)
   else:
       # Evaluate all aspects
       for aspect in ['BPO', 'CCO', 'MFO']:
           logger.info(f"\nEvaluating {aspect} with IA...")
           try:
               run_cafa_evaluation_with_ia(aspect, args.experiment_dir, args.ia_dir)
           except Exception as e:
               logger.error(f"Failed to evaluate {aspect}: {e}")
               continue
   
   # Create comprehensive report
   logger.info("\nCreating evaluation reports...")
   create_comprehensive_evaluation_report(args.experiment_dir)
   
   # Analyze threshold patterns if requested
   if args.analyze_thresholds:
       logger.info("\nAnalyzing threshold patterns...")
       analyze_threshold_patterns(Path(args.experiment_dir))
   
   logger.info("\nEvaluation complete!")


if __name__ == "__main__":
   main()