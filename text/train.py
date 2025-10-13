"""Main training script for text and ESM models."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.preprocessing import load_data
from data.dataset import TextDataset, ESMDataset, FunctionDataset, text_collate_fn, esm_collate_fn, function_collate_fn
from models.text_model import TextFusionModel
from models.simple_function_model import SimpleFunctionModel

from models.esm_model import ESMClassifier
from utils.metrics import compute_fmax


def train_epoch(model, loader, optimizer, criterion, device, model_type='text'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        if model_type == 'text':
            inputs = [h.to(device) for h in batch['hidden_states']]
            logits = model(inputs)
        else:  # esm
            inputs = batch['embeddings'].to(device)
            logits = model(inputs)
        
        labels = batch['labels'].to(device)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, model_type='text'):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if model_type == 'text':
                inputs = [h.to(device) for h in batch['hidden_states']]
                logits = model(inputs)
            else:  # esm
                inputs = batch['embeddings'].to(device)
                logits = model(inputs)
            
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall

def train_model(config, data_dict, model_type='text'):
    """
    Train a model (text, esm, or function).
    
    Args:
        config: Configuration object
        data_dict: Dictionary containing train/val/test data
        model_type: 'text', 'esm', or 'function'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining {model_type.upper()} model on {device}")
    
    # Create datasets
    if model_type == 'text':
        train_dataset = TextDataset(
            data_dict['train'][0], 
            data_dict['train'][1],
            config.cache_dir
        )
        val_dataset = TextDataset(
            data_dict['val'][0], 
            data_dict['val'][1],
            config.cache_dir
        )
        test_dataset = TextDataset(
            data_dict['test'][0], 
            data_dict['test'][1],
            config.cache_dir
        )
        collate_fn = text_collate_fn
    elif model_type == 'function':
        # Function-only model uses same data as ESM
        train_dataset = FunctionDataset(
            data_dict['train'][0], 
            data_dict['train'][1],
            config.cache_dir
        )
        val_dataset = FunctionDataset(
            data_dict['val'][0], 
            data_dict['val'][1],
            config.cache_dir
        )
        test_dataset = FunctionDataset(
            data_dict['test'][0], 
            data_dict['test'][1],
            config.cache_dir
        )
        collate_fn = function_collate_fn
    else:  # esm
        train_dataset = ESMDataset(
            data_dict['train'][0], 
            data_dict['train'][1],
            config.cache_dir
        )
        val_dataset = ESMDataset(
            data_dict['val'][0], 
            data_dict['val'][1],
            config.cache_dir
        )
        test_dataset = ESMDataset(
            data_dict['test'][0], 
            data_dict['test'][1],
            config.cache_dir
        )
        collate_fn = esm_collate_fn
    
    # Create dataloaders (unchanged)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    num_go_terms = data_dict['num_go_terms']
    if model_type == 'text':
        model = TextFusionModel(num_go_terms).to(device)
    elif model_type == 'function':
        model = SimpleFunctionModel(num_go_terms).to(device)
    else:  # esm
        model = ESMClassifier(num_go_terms).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_fmax = 0
    best_threshold = 0.5
    history = []
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_loss, val_fmax, val_threshold, val_prec, val_recall = evaluate(
            model, val_loader, criterion, device, model_type
        )
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Fmax: {val_fmax:.3f} (t={val_threshold:.2f})")
        print(f"  Precision: {val_prec:.3f}, Recall: {val_recall:.3f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fmax': val_fmax,
            'threshold': val_threshold,
            'precision': val_prec,
            'recall': val_recall
        })
        
        # Save best model
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_threshold = val_threshold
            torch.save(
                model.state_dict(), 
                config.checkpoint_dir / f"best_{model_type}.pt"
            )
            print(f"  ✓ New best: {best_val_fmax:.3f}")
    
    # Test evaluation
    print("\nTest evaluation...")
    model.load_state_dict(torch.load(config.checkpoint_dir / f"best_{model_type}.pt"))
    test_loss, test_fmax, test_threshold, test_prec, test_recall = evaluate(
        model, test_loader, criterion, device, model_type
    )
    
    print(f"\n{model_type.upper()} Results:")
    print(f"  Test Fmax: {test_fmax:.3f} (t={test_threshold:.2f})")
    print(f"  Precision: {test_prec:.3f}, Recall: {test_recall:.3f}")
    
    # Save results
    results = {
        'model_type': model_type,
        'aspect': config.aspect,
        'similarity_threshold': config.similarity_threshold,
        'num_go_terms': num_go_terms,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall)
        },
        'best_val_fmax': float(best_val_fmax),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    with open(config.results_dir / f"results_{model_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(
        config.results_dir / f"history_{model_type}.csv", 
        index=False
    )
    
    return results


def main():
    """Run training experiments."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=30,
                       help="Similarity threshold (30/50/70/95)")
    parser.add_argument("--aspect", type=str, default='BP',
                       choices=['BP', 'MF', 'CC'],
                       help="GO aspect")
    parser.add_argument("--model", type=str, default='both',
                       choices=['text', 'esm', 'function', 'both', 'all'],
                       help="Which model to train")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (small dataset)")
    
    args = parser.parse_args()
    
    # Setup config
    config = Config(
        similarity_threshold=args.threshold,
        aspect=args.aspect,
        debug_mode=args.debug
    )
    
    print("="*70)
    print(f"Protein Function Prediction")
    print(f"Aspect: {args.aspect}, Threshold: {args.threshold}%")
    print("="*70)
    
    # Load data
    data_dict = load_data(config)
    
    # Train models
    results = {}
    
    if args.model in ['esm', 'both', 'all']:
        print("\n" + "="*70)
        print("ESM BASELINE")
        print("="*70)
        results['esm'] = train_model(config, data_dict, model_type='esm')
    
    if args.model in ['function', 'all']:
        print("\n" + "="*70)
        print("FUNCTION-ONLY MODEL")
        print("="*70)
        results['function'] = train_model(config, data_dict, model_type='function')
    
    if args.model in ['text', 'both', 'all']:
        print("\n" + "="*70)
        print("TEXT FUSION MODEL")
        print("="*70)
        results['text'] = train_model(config, data_dict, model_type='text')
    
    # Comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        for model_name, model_results in results.items():
            fmax = model_results['test_metrics']['fmax']
            print(f"{model_name.upper()}: Fmax = {fmax:.3f}")
        
        # Save comparison
        if 'esm' in results:
            esm_fmax = results['esm']['test_metrics']['fmax']
            comparison = {
                'aspect': config.aspect,
                'similarity_threshold': config.similarity_threshold,
                'esm': results['esm']['test_metrics']
            }
            
            if 'function' in results:
                func_fmax = results['function']['test_metrics']['fmax']
                comparison['function'] = results['function']['test_metrics']
                comparison['function_vs_esm'] = {
                    'absolute': float(func_fmax - esm_fmax),
                    'relative_percent': float((func_fmax/esm_fmax - 1)*100)
                }
                print(f"\nFunction vs ESM: {func_fmax - esm_fmax:+.3f} ({(func_fmax/esm_fmax - 1)*100:+.1f}%)")
            
            if 'text' in results:
                text_fmax = results['text']['test_metrics']['fmax']
                comparison['text'] = results['text']['test_metrics']
                comparison['text_vs_esm'] = {
                    'absolute': float(text_fmax - esm_fmax),
                    'relative_percent': float((text_fmax/esm_fmax - 1)*100)
                }
                print(f"Text vs ESM: {text_fmax - esm_fmax:+.3f} ({(text_fmax/esm_fmax - 1)*100:+.1f}%)")
                
                if 'function' in results:
                    comparison['text_vs_function'] = {
                        'absolute': float(text_fmax - func_fmax),
                        'relative_percent': float((text_fmax/func_fmax - 1)*100)
                    }
                    print(f"Text vs Function: {text_fmax - func_fmax:+.3f} ({(text_fmax/func_fmax - 1)*100:+.1f}%)")
            
            with open(config.results_dir / "comparison.json", 'w') as f:
                json.dump(comparison, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()