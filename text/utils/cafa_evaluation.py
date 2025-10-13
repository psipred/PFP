"""CAFA-style evaluation using cafaeval package."""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import shutil


def save_predictions_cafa_format(predictions, protein_ids, go_terms, output_file):
    """
    Save predictions in CAFA format (protein_id, term_id, score).
    
    Args:
        predictions: numpy array [n_proteins, n_terms] with probabilities
        protein_ids: list of protein IDs
        go_terms: list of GO term IDs
        output_file: path to output file
    """
    with open(output_file, 'w') as f:
        for i, protein_id in enumerate(protein_ids):
            for j, go_term in enumerate(go_terms):
                score = predictions[i, j]
                if score > 0:  # Only write non-zero predictions
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")


def save_ground_truth_cafa_format(labels, protein_ids, go_terms, output_file):
    """
    Save ground truth in CAFA format (protein_id, term_id).
    
    Args:
        labels: numpy array [n_proteins, n_terms] with binary labels
        protein_ids: list of protein IDs
        go_terms: list of GO term IDs
        output_file: path to output file
    """
    with open(output_file, 'w') as f:
        for i, protein_id in enumerate(protein_ids):
            for j, go_term in enumerate(go_terms):
                if labels[i, j] == 1:
                    f.write(f"{protein_id}\t{go_term}\n")


def run_cafa_evaluator(obo_file, pred_file, truth_file, output_dir, ia_file=None, threads=4):
    """
    Run CAFA evaluator using cafaeval package.
    
    Args:
        obo_file: path to GO obo file
        pred_file: path to predictions file (or directory)
        truth_file: path to ground truth file
        output_dir: path to output directory
        ia_file: optional information accretion file
        threads: number of threads
    
    Returns:
        Path to results directory
    """
    try:
        import cafaeval
        from cafaeval.evaluation import cafa_eval, write_results
        
        print(f"Running CAFA evaluation...")
        print(f"  OBO file: {obo_file}")
        print(f"  Predictions: {pred_file}")
        print(f"  Ground truth: {truth_file}")
        
        # Run evaluation
        kwargs = {
            'out_dir': str(output_dir),
            'threads': threads,
            'norm': 'cafa',
            'prop': 'max',
            'th_step': 0.01
        }
        
        if ia_file:
            kwargs['ia'] = str(ia_file)
        
        results = cafa_eval(
            str(obo_file),
            str(pred_file),
            str(truth_file),
            **kwargs
        )
        
        # Write results
        write_results(*results, out_dir=str(output_dir))
        
        print(f"✓ CAFA evaluation complete. Results saved to: {output_dir}")
        return output_dir
        
    except ImportError:
        print("cafaeval package not found. Attempting to install...")
        subprocess.run(['pip', 'install', 'cafaeval'], check=True)
        print("Please re-run the evaluation.")
        return None


def evaluate_with_cafa(model, loader, device, protein_ids, go_terms, 
                       obo_file, output_dir, model_type='text', model_name='model'):
    """
    Evaluate model predictions using CAFA evaluator.
    
    Args:
        model: trained PyTorch model
        loader: test data loader
        device: torch device
        protein_ids: list of protein IDs in test set
        go_terms: list of GO term IDs
        obo_file: path to GO obo file
        output_dir: directory to save results
        model_type: 'text', 'concat', 'esm', or 'function'
        model_name: name for the prediction file
    
    Returns:
        dict with CAFA metrics
    """
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create temporary directory for CAFA files
    temp_dir = output_dir / 'cafa_temp'
    temp_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = temp_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating predictions for CAFA evaluation...")
    
    # Get predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if model_type in ['text', 'concat']:
                inputs = [h.to(device) for h in batch['hidden_states']]
                logits = model(inputs)
            else:  # esm or function
                inputs = batch['embeddings'].to(device)
                logits = model(inputs)
            
            labels = batch['labels']
            
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    
    # Save in CAFA format
    pred_file = pred_dir / f"{model_name}.tsv"
    truth_file = temp_dir / "ground_truth.tsv"
    
    print("Saving predictions in CAFA format...")
    save_predictions_cafa_format(predictions, protein_ids, go_terms, pred_file)
    save_ground_truth_cafa_format(labels, protein_ids, go_terms, truth_file)
    
    # Run CAFA evaluator
    cafa_results_dir = output_dir / 'cafa_results'
    cafa_results_dir.mkdir(exist_ok=True, parents=True)
    
    result_path = run_cafa_evaluator(
        obo_file=obo_file,
        pred_file=str(pred_dir),
        truth_file=str(truth_file),
        output_dir=str(cafa_results_dir),
        threads=4
    )
    
    if result_path:
        # Parse results
        all_results_file = Path(result_path) / "evaluation_all.tsv"
        if all_results_file.exists():
            df = pd.read_csv(all_results_file, sep='\t')
            
            # Extract best metrics
            best_f = df.loc[df['f'].idxmax()]
            
            metrics = {
                'cafa_fmax': float(best_f['f']),
                'cafa_threshold': float(best_f['tau']),
                'cafa_precision': float(best_f['pr']),
                'cafa_recall': float(best_f['rc']),
                'cafa_coverage': float(best_f['cov']) if 'cov' in best_f else None,
                'cafa_pr_micro': float(best_f['pr_micro']) if 'pr_micro' in best_f else None,
                'cafa_rc_micro': float(best_f['rc_micro']) if 'rc_micro' in best_f else None,
                'cafa_f_micro': float(best_f['f_micro']) if 'f_micro' in best_f else None
            }
            
            # Check for weighted metrics
            best_wf_file = Path(result_path) / "evaluation_best_wf.tsv"
            if best_wf_file.exists():
                df_wf = pd.read_csv(best_wf_file, sep='\t')
                if len(df_wf) > 0:
                    best_wf = df_wf.iloc[0]
                    metrics['cafa_wfmax'] = float(best_wf['wf'])
                    metrics['cafa_wpr'] = float(best_wf['wpr'])
                    metrics['cafa_wrc'] = float(best_wf['wrc'])
            
            # Check for S metric
            best_s_file = Path(result_path) / "evaluation_best_s.tsv"
            if best_s_file.exists():
                df_s = pd.read_csv(best_s_file, sep='\t')
                if len(df_s) > 0:
                    best_s = df_s.iloc[0]
                    metrics['cafa_smin'] = float(best_s['s'])
            
            print("\nCAFA Evaluation Results:")
            print(f"  Fmax: {metrics['cafa_fmax']:.4f} (threshold={metrics['cafa_threshold']:.2f})")
            print(f"  Precision: {metrics['cafa_precision']:.4f}, Recall: {metrics['cafa_recall']:.4f}")
            if metrics.get('cafa_f_micro'):
                print(f"  Micro F-score: {metrics['cafa_f_micro']:.4f}")
            if metrics.get('cafa_wfmax'):
                print(f"  Weighted Fmax: {metrics['cafa_wfmax']:.4f}")
            if metrics.get('cafa_smin'):
                print(f"  Smin: {metrics['cafa_smin']:.4f}")
            
            return metrics
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {}