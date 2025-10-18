"""Main training script for text and ESM models."""

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
from data.dataset import TextDataset, ESMDataset, FunctionDataset, text_collate_fn, esm_collate_fn, function_collate_fn
from models.text_model import TextFusionModel
from models.concat_model import ConcatModel
from models.simple_function_model import SimpleFunctionModel
from models.esm_model import ESMClassifier
from utils.metrics import compute_fmax, compute_auprc
from utils.cafa_evaluation import evaluate_with_cafa
from models.cross_attention_model import CrossModalAttentionFusion
from models.gated_fusion_model import GatedFusionModel
from models.gated_cross_attention_model import GatedCrossAttentionFusion
from data.dataset import MultiModalDataset, multimodal_collate_fn
from models.esm_sequence_branch import ESMSequenceBranch
from data.dataset import ESMRawDataset, esm_raw_collate_fn

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


        if model_type == 'esm_seq':
            # NEW: Pass input_ids and attention_mask directly
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
        elif model_type in ['cross_attn', 'gated', 'gated_cross']:
            # Multi-modal models
            text_inputs = [h.to(device) for h in batch['hidden_states']]
            esm_inputs = batch['esm_embeddings'].to(device)
            logits = model(text_inputs, esm_inputs)
        elif model_type in ['text', 'concat']:
            inputs = [h.to(device) for h in batch['hidden_states']]
            logits = model(inputs)
        else:  # esm or function
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
    """Evaluate model with Fmax and AUPRC metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):


            if model_type == 'esm_seq':
                # NEW: Pass input_ids and attention_mask directly
                logits = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
            elif model_type in ['cross_attn', 'gated', 'gated_cross']:
                text_inputs = [h.to(device) for h in batch['hidden_states']]
                esm_inputs = batch['esm_embeddings'].to(device)
                logits = model(text_inputs, esm_inputs)
            elif model_type in ['text', 'concat']:
                inputs = [h.to(device) for h in batch['hidden_states']]
                logits = model(inputs)
            else:  # esm or function
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
    
    # Compute Fmax
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    
    # Compute AUPRC
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall, micro_auprc, macro_auprc


def train_model(config, data_dict, model_type='text', sequences_dict=None):
    """
    Train a model with early stopping.
    Supports: text, concat, esm, function, cross_attn, gated, gated_cross
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining {model_type.upper()} model on {device}")



        # Create datasets
    if model_type == 'esm_seq':
        # NEW: Use raw sequences instead of precomputed embeddings
        if sequences_dict is None:
            raise ValueError("sequences_dict required for esm_seq model")
        
        train_dataset = ESMRawDataset(
            data_dict['train'][0], 
            data_dict['train'][1], 
            sequences_dict,
            config.esm_model
        )
        val_dataset = ESMRawDataset(
            data_dict['val'][0], 
            data_dict['val'][1], 
            sequences_dict,
            config.esm_model
        )
        test_dataset = ESMRawDataset(
            data_dict['test'][0], 
            data_dict['test'][1], 
            sequences_dict,
            config.esm_model
        )
        collate_fn = esm_raw_collate_fn
    
    # Create datasets
    elif model_type in ['cross_attn', 'gated', 'gated_cross']:
        # Multi-modal models need both text and ESM
        train_dataset = MultiModalDataset(data_dict['train'][0], data_dict['train'][1], config.cache_dir)
        val_dataset = MultiModalDataset(data_dict['val'][0], data_dict['val'][1], config.cache_dir)
        test_dataset = MultiModalDataset(data_dict['test'][0], data_dict['test'][1], config.cache_dir)
        collate_fn = multimodal_collate_fn
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
    else:  # esm
        train_dataset = ESMDataset(data_dict['train'][0], data_dict['train'][1], config.cache_dir)
        val_dataset = ESMDataset(data_dict['val'][0], data_dict['val'][1], config.cache_dir)
        test_dataset = ESMDataset(data_dict['test'][0], data_dict['test'][1], config.cache_dir)
        collate_fn = esm_collate_fn
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True if device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    num_go_terms = data_dict['num_go_terms']
    if model_type == 'esm_seq':
        model = ESMSequenceBranch(num_go_terms, config.esm_model).to(device)
    elif model_type == 'text':
        model = TextFusionModel(num_go_terms).to(device)
    elif model_type == 'concat':
        model = ConcatModel(num_go_terms).to(device)
    elif model_type == 'function':
        model = SimpleFunctionModel(num_go_terms).to(device)
    elif model_type == 'esm':
        model = ESMClassifier(num_go_terms).to(device)
    elif model_type == 'cross_attn':
        model = CrossModalAttentionFusion(num_go_terms).to(device)
    elif model_type == 'gated':
        model = GatedFusionModel(num_go_terms).to(device)
    elif model_type == 'gated_cross':
        model = GatedCrossAttentionFusion(num_go_terms).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            torch.save(model.state_dict(), config.checkpoint_dir / f"best_{model_type}.pt")
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
    model.load_state_dict(torch.load(config.checkpoint_dir / f"best_{model_type}.pt"))
    test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc = evaluate(
        model, test_loader, criterion, device, model_type
    )
    
    print(f"\n{model_type.upper()} Test Results:")
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
            model_name=f"{model_type}_{config.aspect}_{config.similarity_threshold}"
        )
    else:
        print(f"\nWarning: OBO file not found at {obo_file}. Skipping CAFA evaluation.")
        cafa_metrics = {}
    
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
    
    with open(config.results_dir / f"results_{model_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(config.results_dir / f"history_{model_type}.csv", index=False)
    
    return results


def main():
    """Run training experiments."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--aspect", type=str, default='BP', choices=['BP', 'MF', 'CC'])
    parser.add_argument("--model", type=str, default='both', 
                       choices=['text', 'concat', 'esm', 'function', 
                               'cross_attn', 'gated', 'gated_cross', 
                               'esm_seq', 'both', 'all'])  # Added 'esm_seq'
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Setup config
    config = Config(similarity_threshold=args.threshold, aspect=args.aspect, debug_mode=args.debug)
    
    print("="*70)
    print(f"Protein Function Prediction")
    print(f"Aspect: {args.aspect}, Threshold: {args.threshold}%")
    print(f"Model: {args.model}")
    print("="*70)
    
    # Load data
    data_dict = load_data(config)


    # Load sequences for esm_seq model
    sequences_dict = None
    if args.model in ['esm_seq', 'all']:
        print("\nLoading protein sequences...")
        protad_df = pd.read_csv(config.protad_path, sep='\t')
        sequences_dict = protad_df.set_index('Entry')['Sequence'].to_dict()
    
    
    # Train models
    results = {}
    


        # NEW: Train ESM sequence branch model
    if args.model in ['esm_seq', 'all']:
        print("\n" + "="*70)
        print("ESM SEQUENCE BRANCH MODEL")
        print("="*70)
        results['esm_seq'] = train_model(
            config, data_dict, 
            model_type='esm_seq',
            sequences_dict=sequences_dict
        )
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
    
    if args.model in ['concat', 'all']:
        print("\n" + "="*70)
        print("CONCAT BASELINE MODEL")
        print("="*70)
        results['concat'] = train_model(config, data_dict, model_type='concat')
    
    if args.model in ['text', 'both', 'all']:
        print("\n" + "="*70)
        print("TEXT FUSION MODEL")
        print("="*70)
        results['text'] = train_model(config, data_dict, model_type='text')
    
    # NEW: Train cross-attention model
    if args.model in ['cross_attn', 'all']:
        print("\n" + "="*70)
        print("CROSS-ATTENTION FUSION MODEL")
        print("="*70)
        results['cross_attn'] = train_model(config, data_dict, model_type='cross_attn')
    
    # NEW: Train gated fusion model
    if args.model in ['gated', 'all']:
        print("\n" + "="*70)
        print("GATED FUSION MODEL")
        print("="*70)
        results['gated'] = train_model(config, data_dict, model_type='gated')
    
    # NEW: Train gated cross-attention model
    if args.model in ['gated_cross', 'all']:
        print("\n" + "="*70)
        print("GATED CROSS-ATTENTION FUSION MODEL")
        print("="*70)
        results['gated_cross'] = train_model(config, data_dict, model_type='gated_cross')
    
    # # Comparison
    # if len(results) > 1:
    #     print("\n" + "="*70)
    #     print("FINAL COMPARISON")
    #     print("="*70)
        
    #     for model_name, model_results in results.items():
    #         metrics = model_results['test_metrics']
    #         print(f"\n{model_name.upper()}:")
    #         print(f"  Fmax: {metrics['fmax']:.4f}")
    #         print(f"  Micro-AUPRC: {metrics['micro_auprc']:.4f}")
    #         print(f"  Macro-AUPRC: {metrics['macro_auprc']:.4f}")
    #         if 'cafa_fmax' in metrics:
    #             print(f"  CAFA Fmax: {metrics['cafa_fmax']:.4f}")
        
    #     # Save comparison
    #     if 'esm' in results:
    #         esm_metrics = results['esm']['test_metrics']
    #         comparison = {
    #             'aspect': config.aspect,
    #             'similarity_threshold': config.similarity_threshold,
    #             'esm': esm_metrics
    #         }
            
    #         # Compare all models against ESM baseline
    #         for model_name in ['function', 'concat', 'text', 'cross_attn', 'gated', 'gated_cross']:
    #             if model_name in results:
    #                 model_metrics = results[model_name]['test_metrics']
    #                 comparison[model_name] = model_metrics
    #                 comparison[f'{model_name}_improvement'] = {
    #                     'fmax_absolute': float(model_metrics['fmax'] - esm_metrics['fmax']),
    #                     'fmax_relative_percent': float((model_metrics['fmax']/esm_metrics['fmax'] - 1)*100),
    #                     'micro_auprc_absolute': float(model_metrics['micro_auprc'] - esm_metrics['micro_auprc']),
    #                     'macro_auprc_absolute': float(model_metrics['macro_auprc'] - esm_metrics['macro_auprc'])
    #                 }
    #                 print(f"\n{model_name.upper()} vs ESM:")
    #                 print(f"  Fmax: {model_metrics['fmax'] - esm_metrics['fmax']:+.4f} "
    #                       f"({(model_metrics['fmax']/esm_metrics['fmax'] - 1)*100:+.2f}%)")
    #                 print(f"  Micro-AUPRC: {model_metrics['micro_auprc'] - esm_metrics['micro_auprc']:+.4f}")
            
    #         with open(config.results_dir / "comparison.json", 'w') as f:
    #             json.dump(comparison, f, indent=2)
        
    #     # Also compare fusion models against text-only baseline
    #     if 'text' in results:
    #         print("\n" + "="*70)
    #         print("FUSION MODELS vs TEXT BASELINE")
    #         print("="*70)
    #         text_metrics = results['text']['test_metrics']
            
    #         for model_name in ['cross_attn', 'gated', 'gated_cross']:
    #             if model_name in results:
    #                 model_metrics = results[model_name]['test_metrics']
    #                 improvement = model_metrics['fmax'] - text_metrics['fmax']
    #                 improvement_pct = (model_metrics['fmax']/text_metrics['fmax'] - 1) * 100
    #                 print(f"\n{model_name.upper()} vs TEXT:")
    #                 print(f"  Fmax: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    #                 print(f"  Micro-AUPRC: {model_metrics['micro_auprc'] - text_metrics['micro_auprc']:+.4f}")
    
    print("\n✓ Training complete!")
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()