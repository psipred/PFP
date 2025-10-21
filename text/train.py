"""Main training script for text and PLM models."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from config import Config
from data.preprocessing import load_data
from data.dataset import (
    TextDataset, text_collate_fn,
    FunctionDataset, function_collate_fn,
    MultiModalDataset, multimodal_collate_fn
)
from models.text_model import TextFusionModel
from models.concat_model import ConcatModel
from models.simple_function_model import SimpleFunctionModel
from utils.metrics import compute_fmax, compute_auprc
from utils.cafa_evaluation import evaluate_with_cafa


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def train_epoch(model, loader, optimizer, criterion, device, model_type='text'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if model_type == 'plm':
            inputs = batch['embeddings'].to(device)
            logits = model(inputs)
        elif model_type in ['cross_attn', 'gated', 'gated_cross']:
            text_inputs = [h.to(device) for h in batch['hidden_states']]
            esm_inputs = batch['esm_embeddings'].to(device)
            logits = model(text_inputs, esm_inputs)
        elif model_type in ['text', 'concat']:
            inputs = [h.to(device) for h in batch['hidden_states']]
            logits = model(inputs)
        elif model_type == 'function':
            inputs = batch['embeddings'].to(device)
            logits = model(inputs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        labels = batch['labels'].to(device)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, model_type='text'):
    """Evaluate model with Fmax and AUPRC metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if model_type == 'plm':
                inputs = batch['embeddings'].to(device)
                logits = model(inputs)
            elif model_type in ['cross_attn', 'gated', 'gated_cross']:
                text_inputs = [h.to(device) for h in batch['hidden_states']]
                esm_inputs = batch['esm_embeddings'].to(device)
                logits = model(text_inputs, esm_inputs)
            elif model_type in ['text', 'concat']:
                inputs = [h.to(device) for h in batch['hidden_states']]
                logits = model(inputs)
            elif model_type == 'function':
                inputs = batch['embeddings'].to(device)
                logits = model(inputs)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    # Compute Fmax
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    
    # Compute AUPRC
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall, micro_auprc, macro_auprc


def train_model(config, data_dict, model_type='text', plm_type='esm', sequences_dict=None):
    """
    Train a model with early stopping.
    
    Args:
        config: Configuration object
        data_dict: Dictionary with train/val/test data
        model_type: 'text', 'concat', 'plm', 'function', 'cross_attn', 'gated', 'gated_cross'
        plm_type: 'esm', 'ankh', 'prott5', 'prostt5' (only used if model_type='plm')
        sequences_dict: Dict of protein sequences (not used in current implementation)
    """
    device = config.device
    print(f"\nTraining {model_type.upper()} model on {device}")
    if model_type == 'plm':
        print(f"  Using PLM: {plm_type.upper()}")
    
    # Create datasets
    if model_type == 'plm':
        from data.dataset import PLMDataset, plm_collate_fn
        
        print(f"  Creating {plm_type.upper()} datasets...")
        train_dataset = PLMDataset(
            data_dict['train'][0], 
            data_dict['train'][1], 
            config.cache_dir, 
            plm_type
        )
        val_dataset = PLMDataset(
            data_dict['val'][0], 
            data_dict['val'][1], 
            config.cache_dir, 
            plm_type
        )
        test_dataset = PLMDataset(
            data_dict['test'][0], 
            data_dict['test'][1], 
            config.cache_dir, 
            plm_type
        )
        collate_fn = plm_collate_fn
        
        # Verify dataset
        print(f"  Verifying dataset...")
        try:
            sample = train_dataset[0]
            print(f"    ✓ Sample embedding shape: {sample['embedding'].shape}")
        except Exception as e:
            print(f"    ❌ Error loading sample: {e}")
            raise
    
    elif model_type in ['text', 'concat']:
        train_dataset = TextDataset(data_dict['train'][0], data_dict['train'][1], config.cache_dir)
        val_dataset = TextDataset(data_dict['val'][0], data_dict['val'][1], config.cache_dir)
        test_dataset = TextDataset(data_dict['test'][0], data_dict['test'][1], config.cache_dir)
        collate_fn = text_collate_fn
    
    elif model_type == 'function':
        train_dataset = FunctionDataset(data_dict['train'][0], data_dict['train'][1], config.cache_dir)
        val_dataset = FunctionDataset(data_dict['val'][0], data_dict['val'][1], config.cache_dir)
        test_dataset = FunctionDataset(data_dict['test'][0], data_dict['test'][1], config.cache_dir)
        collate_fn = function_collate_fn
    
    elif model_type in ['cross_attn', 'gated', 'gated_cross']:
        train_dataset = MultiModalDataset(data_dict['train'][0], data_dict['train'][1], config.cache_dir)
        val_dataset = MultiModalDataset(data_dict['val'][0], data_dict['val'][1], config.cache_dir)
        test_dataset = MultiModalDataset(data_dict['test'][0], data_dict['test'][1], config.cache_dir)
        collate_fn = multimodal_collate_fn
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True if 'cuda' in device else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True if 'cuda' in device else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True if 'cuda' in device else False
    )
    
    # Create model
    num_go_terms = data_dict['num_go_terms']
    
    if model_type == 'plm':
        from models.plm_classifier import PLMClassifier
        from utils.embeddings import get_actual_plm_dim
        
        # Get actual dimension from cached embeddings
        print(f"\n  Detecting embedding dimension for {plm_type.upper()}...")
        plm_dim = get_actual_plm_dim(
            config.cache_dir, 
            data_dict['train'][0],
            plm_type
        )
        print(f"  ✓ Detected {plm_type.upper()} embedding dimension: {plm_dim}")
        
        model = PLMClassifier(num_go_terms, plm_dim)
        print(f"  ✓ Created PLMClassifier with input_dim={plm_dim}, output_dim={num_go_terms}")
    
    elif model_type == 'text':
        model = TextFusionModel(num_go_terms)
    
    elif model_type == 'concat':
        model = ConcatModel(num_go_terms)
    
    elif model_type == 'function':
        model = SimpleFunctionModel(num_go_terms)
    
    elif model_type == 'cross_attn':
        from models.cross_attention_model import CrossModalAttentionFusion
        model = CrossModalAttentionFusion(num_go_terms)
    
    elif model_type == 'gated':
        from models.gated_fusion_model import GatedFusionModel
        model = GatedFusionModel(num_go_terms)
    
    elif model_type == 'gated_cross':
        from models.gated_cross_attention_model import GatedCrossAttentionFusion
        model = GatedCrossAttentionFusion(num_go_terms)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Handle DataParallel
    if config.use_ddp:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if config.use_ddp:
        print(f"  Using DataParallel across {config.n_gpus} GPUs")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.min_delta,
        mode='max'
    )
    
    # Training loop
    best_val_fmax = 0
    best_threshold = 0.5
    best_epoch = 0
    history = []
    
    # Determine model name for saving
    if model_type == 'plm':
        model_save_name = plm_type
    else:
        model_save_name = model_type
    
    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc = evaluate(
            model, val_loader, criterion, device, model_type
        )
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Fmax: {val_fmax:.3f} (t={val_threshold:.2f})")
        print(f"  Precision: {val_prec:.3f}, Recall: {val_recall:.3f}")
        print(f"  Micro-AUPRC: {val_micro_auprc:.3f}, Macro-AUPRC: {val_macro_auprc:.3f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fmax': val_fmax,
            'threshold': val_threshold,
            'precision': val_prec,
            'recall': val_recall,
            'micro_auprc': val_micro_auprc,
            'macro_auprc': val_macro_auprc
        })
        
        # Save best model
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_threshold = val_threshold
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.checkpoint_dir / f"best_{model_save_name}.pt")
            print(f"  ✓ New best: {best_val_fmax:.3f}")
        
        # Early stopping check
        if early_stopping(val_fmax):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation Fmax: {best_val_fmax:.3f} at epoch {best_epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    model.load_state_dict(torch.load(config.checkpoint_dir / f"best_{model_save_name}.pt"))
    test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc = evaluate(
        model, test_loader, criterion, device, model_type
    )
    
    print(f"\n{model_save_name.upper()} Test Results:")
    print(f"  Fmax: {test_fmax:.4f} (threshold={test_threshold:.3f})")
    print(f"  Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
    print(f"  Micro-AUPRC: {test_micro_auprc:.4f}, Macro-AUPRC: {test_macro_auprc:.4f}")
    
    # CAFA evaluation
    from pathlib import Path
    obo_file = Path("/home/zijianzhou/Datasets/protad/go_annotations/go-basic.obo")
    if obo_file.exists():
        print("\n" + "="*70)
        print("CAFA-STYLE EVALUATION")
        print("="*70)
        
        cafa_metrics = evaluate_with_cafa(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=data_dict['test'][0],
            go_terms=data_dict['go_terms'],
            obo_file=obo_file,
            output_dir=config.results_dir / 'cafa_eval',
            model_type=model_type,
            model_name=f"{model_save_name}_{config.aspect}_{config.similarity_threshold}"
        )
    else:
        print(f"\nWarning: OBO file not found at {obo_file}. Skipping CAFA evaluation.")
        cafa_metrics = {}
    
    # Save results
    results = {
        'model_type': model_save_name,
        'aspect': config.aspect,
        'similarity_threshold': config.similarity_threshold,
        'num_go_terms': num_go_terms,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'micro_auprc': float(test_micro_auprc),
            'macro_auprc': float(test_macro_auprc),
            **cafa_metrics
        },
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    with open(config.results_dir / f"results_{model_save_name}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(config.results_dir / f"history_{model_save_name}.csv", index=False)
    
    return results


def main():
    """Run training experiments."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--aspect", type=str, default='BP', choices=['BP', 'MF', 'CC'])
    parser.add_argument("--model", type=str, default='plm', 
                       choices=['text', 'concat', 'plm', 'function', 'all'])
    parser.add_argument("--plm", type=str, default='esm',
                       choices=['esm', 'ankh', 'prott5', 'prostt5', 'all'])
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Custom output directory (default: ./experiments)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Setup config with custom output dir if provided
    if args.output_dir:
        config = Config(
            similarity_threshold=args.threshold, 
            aspect=args.aspect, 
            debug_mode=args.debug,
            output_dir=Path(args.output_dir)
        )
    else:
        config = Config(
            similarity_threshold=args.threshold, 
            aspect=args.aspect, 
            debug_mode=args.debug
        )
    
    print("="*70)
    print(f"Protein Function Prediction")
    print(f"Aspect: {args.aspect}, Threshold: {args.threshold}%")
    print(f"Model: {args.model}")
    if args.model in ['plm', 'all']:
        print(f"PLM(s): {args.plm}")
    print("="*70)
    
    # Determine which PLMs to use
    if args.plm == 'all':
        plm_types = ['esm', 'ankh', 'prott5', 'prostt5']
    else:
        plm_types = [args.plm]
    
    results = {}
    
    # Train PLM models
    if args.model in ['plm', 'all']:
        for plm_type in plm_types:
            print("\n" + "="*70)
            print(f"{plm_type.upper()} MODEL")
            print("="*70)
            
            # Load data specifically for this PLM
            print(f"\nPreparing data for {plm_type.upper()}...")
            data_dict = load_data(config, plm_types=[plm_type])
            
            # Train the model
            results[plm_type] = train_model(
                config, 
                data_dict, 
                model_type='plm', 
                plm_type=plm_type
            )
    
    # Train text-based models (if requested)
    if args.model in ['text', 'all']:
        print("\n" + "="*70)
        print("TEXT FUSION MODEL")
        print("="*70)
        data_dict = load_data(config, plm_types=[])
        results['text'] = train_model(config, data_dict, model_type='text')
    
    if args.model in ['concat', 'all']:
        print("\n" + "="*70)
        print("CONCAT BASELINE MODEL")
        print("="*70)
        if 'data_dict' not in locals():
            data_dict = load_data(config, plm_types=[])
        results['concat'] = train_model(config, data_dict, model_type='concat')
    
    if args.model in ['function', 'all']:
        print("\n" + "="*70)
        print("FUNCTION-ONLY MODEL")
        print("="*70)
        if 'data_dict' not in locals():
            data_dict = load_data(config, plm_types=[])
        results['function'] = train_model(config, data_dict, model_type='function')
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        
        for model_name, model_results in results.items():
            metrics = model_results['test_metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  Fmax: {metrics['fmax']:.4f}")
            print(f"  Micro-AUPRC: {metrics['micro_auprc']:.4f}")
            print(f"  Macro-AUPRC: {metrics['macro_auprc']:.4f}")
            if 'cafa_fmax' in metrics:
                print(f"  CAFA Fmax: {metrics['cafa_fmax']:.4f}")
    
    print("\n✓ Training complete!")
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()