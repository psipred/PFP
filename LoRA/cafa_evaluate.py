#!/usr/bin/env python3
"""
Standalone CAFA Evaluation Script
Run this after training to evaluate your model with CAFA metrics
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add your training script directory to path if needed
# sys.path.insert(0, '/path/to/your/training/script')

def load_trained_model(checkpoint_path, num_go_terms, config):
    """Load trained model from checkpoint"""
    from esm import ESM2MLPClassifier  # Adjust import name
    
    model = ESM2MLPClassifier(config, num_go_terms=num_go_terms)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
    
    return model


def generate_predictions_and_evaluate(aspect='mf'):
    """
    Generate CAFA predictions and run evaluation
    
    Usage:
        python cafa_evaluate.py --aspect mf
    """
    
    # Import your config and classes
    from esm import Config, CAFA3DataLoader, CAFA3MLPDataset
    
    config = Config(go_aspect=aspect, debug_mode=False)
    
    # Load data
    print(f"📂 Loading {aspect.upper()} data...")
    data_loader = CAFA3DataLoader(config)
    (_, _, _, _, test_proteins, test_labels, go_terms) = data_loader.prepare_data()
    
    # Load best modelLoRA/cafa3_mlp_experiments/mf/checkpoints/conditional_best_mf.pt
    checkpoint_path = config.CHECKPOINT_DIR / f"conditional_best_{aspect}.pt"
    model = load_trained_model(checkpoint_path, len(go_terms), config)
    
    # Create test dataset
    test_dataset = CAFA3MLPDataset(test_proteins, test_labels, config.ESM_EMBEDDINGS_DIR)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Generate predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    all_preds = []
    print(f"🔮 Generating predictions...")
    
    with torch.no_grad():
        for embeddings, _ in tqdm(test_loader):
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
    
    all_preds = np.vstack(all_preds)
    
    # Create CAFA evaluation directory
    eval_dir = config.RESULTS_DIR / "cafa_evaluation"
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    # Write prediction file
    pred_file = eval_dir / f"predictions_{aspect}.tsv"
    print(f"📝 Writing predictions to {pred_file}...")
    
    with open(pred_file, 'w') as f:
        for i, protein_id in enumerate(test_proteins):
            for j, go_term in enumerate(go_terms):
                score = all_preds[i, j]
                if score > 0.01:  # Filter low scores
                    f.write(f"{protein_id}\t{go_term}\t{score:.4f}\n")
    
    # Write ground truth file
    gt_file = eval_dir / f"ground_truth_{aspect}.tsv"
    print(f"📝 Writing ground truth to {gt_file}...")
    
    with open(gt_file, 'w') as f:
        for i, protein_id in enumerate(test_proteins):
            for j, go_term in enumerate(go_terms):
                if test_labels[i, j] == 1:
                    f.write(f"{protein_id}\t{go_term}\n")
    
    # Run CAFA evaluation
    print(f"\n🎯 Running CAFA evaluation...")
    
    try:
        from cafaeval.evaluation import cafa_eval, write_results
        
        # Create prediction folder
        pred_folder = eval_dir / "predictions"
        pred_folder.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(pred_file, pred_folder / pred_file.name)
        
        # Run evaluation
        obo_file = "/home/zijianzhou/Datasets/cafa3/go.obo"
        results_dir = eval_dir / f"cafa_results_{aspect}"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        results, best_scores = cafa_eval(
            str(obo_file),
            str(pred_folder),
            str(gt_file)
        )
        
        # Save results
        write_results(results, best_scores, str(results_dir))
        
        print(f"\n✅ Evaluation complete!")
        print(f"📊 Results saved to: {results_dir}")
        
        # Print summary
        if 'f' in best_scores:
            print(f"\n📈 Best F-measure:")
            print(best_scores['f'])
        
        if 'f_micro' in best_scores:
            print(f"\n📈 Best Micro F-measure:")
            print(best_scores['f_micro'])
        
        # Save summary
        summary_file = results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"CAFA Evaluation Summary - {aspect.upper()}\n")
            f.write("=" * 60 + "\n\n")
            if 'f' in best_scores:
                f.write("Best F-measure:\n")
                f.write(best_scores['f'].to_string() + "\n\n")
            if 'f_micro' in best_scores:
                f.write("Best Micro F-measure:\n")
                f.write(best_scores['f_micro'].to_string() + "\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        
    except ImportError:
        print("\n❌ cafaeval package not installed!")
        print("Install with: pip install cafaeval")
        print("\nPredictions and ground truth files are ready:")
        print(f"  Predictions: {pred_file}")
        print(f"  Ground truth: {gt_file}")
        print("\nYou can run CAFA evaluation manually later.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CAFA evaluation on trained model")
    parser.add_argument("--aspect", type=str, default="mf", 
                       choices=["mf", "bp", "cc"],
                       help="GO aspect to evaluate")
    
    args = parser.parse_args()
    
    generate_predictions_and_evaluate(aspect=args.aspect)