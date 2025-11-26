"""Training script for comparing fusion techniques.

Compares: concatenation, average, gated, transformer fusion
Uses CAFA evaluation for test set benchmarking.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fusion_models import MultiModalFusionModel
from fusion_dataset import MultiModalDataset, collate_fn


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize worker with unique seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_fmax(y_true, y_pred, thresholds=None):
    """Compute micro-averaged Fmax."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    best_fmax = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return best_fmax, best_threshold, best_precision, best_recall


def compute_auprc(y_true, y_pred):
    """Compute micro and macro AUPRC."""
    from sklearn.metrics import average_precision_score
    
    micro_auprc = average_precision_score(y_true.ravel(), y_pred.ravel())
    
    term_auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            term_auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            term_auprcs.append(term_auprc)
    
    macro_auprc = np.mean(term_auprcs) if term_auprcs else 0.0
    
    return micro_auprc, macro_auprc


def train_epoch(model, loader, optimizer, criterion, scheduler, device):
    """Training epoch."""
    model.train()
    total_loss = 0
    weight_stats = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        seq = batch['seq'].to(device)
        seq_mask = batch['seq_mask'].to(device)
        text = batch['text'].to(device)
        text_mask = batch['text_mask'].to(device)
        struct = batch['struct'].to(device)
        struct_mask = batch['struct_mask'].to(device)
        ppi = batch['ppi'].to(device)
        ppi_mask = batch['ppi_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits, fusion_weights = model(
            seq, seq_mask, text, text_mask,
            struct, struct_mask, ppi, ppi_mask
        )
        
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        weight_stats.append(fusion_weights.mean(0).detach().cpu().numpy())
    
    avg_weights = np.array(weight_stats).mean(0)
    
    return {
        'loss': total_loss / len(loader),
        'weight_seq': avg_weights[0],
        'weight_text': avg_weights[1],
        'weight_struct': avg_weights[2],
        'weight_ppi': avg_weights[3]
    }


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    weight_stats = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            seq = batch['seq'].to(device)
            seq_mask = batch['seq_mask'].to(device)
            text = batch['text'].to(device)
            text_mask = batch['text_mask'].to(device)
            struct = batch['struct'].to(device)
            struct_mask = batch['struct_mask'].to(device)
            ppi = batch['ppi'].to(device)
            ppi_mask = batch['ppi_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, fusion_weights = model(
                seq, seq_mask, text, text_mask,
                struct, struct_mask, ppi, ppi_mask
            )
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            weight_stats.append(fusion_weights.mean(0).detach().cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    avg_weights = np.array(weight_stats).mean(0)
    
    return {
        'loss': total_loss / len(loader),
        'fmax': fmax,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'micro_auprc': micro_auprc,
        'macro_auprc': macro_auprc,
        'weight_seq': avg_weights[0],
        'weight_text': avg_weights[1],
        'weight_struct': avg_weights[2],
        'weight_ppi': avg_weights[3]
    }


def evaluate_with_cafa(model, loader, device, protein_ids, go_terms, obo_file, 
                       output_dir, seq_model, fusion_type):
    """CAFA evaluation for test set."""
    try:
        sys.path.append("/home/zijianzhou/project/PFP")
        from text.utils.cafa_evaluation import evaluate_with_cafa as cafa_eval
        
        seq_dim = 1024 if seq_model == 'prott5' else 1280
        
        class MultimodalWrapper:
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for batch in self.loader:
                    embeddings = torch.cat([
                        batch['seq'], batch['seq_mask'],
                        batch['text'], batch['text_mask'],
                        batch['struct'], batch['struct_mask'],
                        batch['ppi'], batch['ppi_mask']
                    ], dim=-1)
                    yield {'embeddings': embeddings, 'labels': batch['labels']}
            
            def __len__(self):
                return len(self.loader)
        
        class ModelWrapper(nn.Module):
            def __init__(self, multimodal_model, seq_dim):
                super().__init__()
                self.model = multimodal_model
                self.seq_dim = seq_dim
            
            def forward(self, embeddings):
                seq = embeddings[:, :self.seq_dim]
                seq_mask = embeddings[:, self.seq_dim:self.seq_dim+1]
                
                offset = self.seq_dim + 1
                text = embeddings[:, offset:offset+768]
                text_mask = embeddings[:, offset+768:offset+769]
                
                offset = offset + 769
                struct = embeddings[:, offset:offset+512]
                struct_mask = embeddings[:, offset+512:offset+513]
                
                offset = offset + 513
                ppi = embeddings[:, offset:offset+512]
                ppi_mask = embeddings[:, offset+512:offset+513]
                
                logits, _ = self.model(
                    seq, seq_mask, text, text_mask,
                    struct, struct_mask, ppi, ppi_mask
                )
                return logits
        
        wrapped_model = ModelWrapper(model, seq_dim).to(device)
        wrapped_loader = MultimodalWrapper(loader)
        
        return cafa_eval(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type=f'{fusion_type}_fusion',
            model_name=f"{seq_model}_{fusion_type}"
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_fusion_model(seq_model, aspect, fusion_type, modality_dropout=0.1, output_base='.'):
    """Train a model with specified fusion type."""
    seed = 42
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'hidden_dim': 512,
        'dropout': 0.4,
        'modality_dropout': modality_dropout,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'batch_size': 32,
        'max_epochs': 50,
        'patience': 5,
        'warmup_ratio': 0.1,
        'min_delta_fmax': 1e-4,
        'min_delta_loss': 1e-4,
        # Transformer-specific
        'n_heads': 4,
        'n_layers': 2,
    }
    
    embedding_dirs = {
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/exp_text_embeddings',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/prott5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/esm',
        'struct': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/IF1',
        'ppi': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/ppi'
    }
    
    data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
    obo_file = Path("/home/zijianzhou/project/PFP/go.obo")
    
    output_dir = Path(output_base) / 'fusion_comparison' / seq_model / aspect / fusion_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training Fusion Model")
    print(f"Fusion Type: {fusion_type.upper()}")
    print(f"Seq Model: {seq_model.upper()}, Aspect: {aspect}")
    print(f"Modality Dropout: {config['modality_dropout']}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MultiModalDataset(
        data_dir, embedding_dirs, seq_model, aspect, 'train',
        normalize='standard'
    )
    
    val_dataset = MultiModalDataset(
        data_dir, embedding_dirs, seq_model, aspect, 'valid',
        normalize='standard', norm_stats=train_dataset.norm_stats
    )
    
    test_dataset = MultiModalDataset(
        data_dir, embedding_dirs, seq_model, aspect, 'test',
        normalize='standard', norm_stats=train_dataset.norm_stats
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=8, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    
    # Load GO terms for CAFA evaluation
    go_terms_file = data_dir / f"{aspect}_go_terms.json"
    with open(go_terms_file, 'r') as f:
        go_terms = json.load(f)
    
    test_protein_ids = test_dataset.protein_ids.tolist()
    
    # Create model
    num_go_terms = train_dataset.labels.shape[1]
    seq_dim = 1024 if seq_model == 'prott5' else 1280
    
    model = MultiModalFusionModel(
        seq_dim=seq_dim,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=config['hidden_dim'],
        num_go_terms=num_go_terms,
        dropout=config['dropout'],
        fusion_type=fusion_type,
        modality_dropout=config['modality_dropout'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    
    num_training_steps = len(train_loader) * config['max_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr'], total_steps=num_training_steps,
        pct_start=config['warmup_ratio'], anneal_strategy='cos'
    )
    
    # Training loop
    best_val_fmax = 0.0
    loss_at_best_fmax = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(1, config['max_epochs'] + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Fmax: {val_metrics['fmax']:.4f}")
        print(f"  Weights: seq={val_metrics['weight_seq']:.3f}, text={val_metrics['weight_text']:.3f}, "
              f"struct={val_metrics['weight_struct']:.3f}, ppi={val_metrics['weight_ppi']:.3f}")
        
        history.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': scheduler.get_last_lr()[0]
        })
        
        current_fmax = val_metrics['fmax']
        current_loss = val_metrics['loss']
        
        fmax_better = current_fmax > best_val_fmax + config['min_delta_fmax']
        fmax_similar = abs(current_fmax - best_val_fmax) <= config['min_delta_fmax']
        loss_better = current_loss < loss_at_best_fmax - config['min_delta_loss']
        
        if fmax_better or (fmax_similar and loss_better):
            best_val_fmax = current_fmax
            loss_at_best_fmax = current_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Best model saved (Fmax: {best_val_fmax:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_metrics['fmax']:.4f}")
    print(f"  Micro-AUPRC: {test_metrics['micro_auprc']:.4f}")
    print(f"  Macro-AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # CAFA evaluation
    print("\n" + "="*70)
    print("CAFA EVALUATION")
    print("="*70)
    
    cafa_metrics = {}
    if obo_file.exists():
        cafa_metrics = evaluate_with_cafa(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir / 'cafa_eval',
            seq_model=seq_model,
            fusion_type=fusion_type
        )
        
        if cafa_metrics:
            print("\nCAFA Metrics:")
            for key, value in cafa_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'seq_model': seq_model,
        'aspect': aspect,
        'fusion_type': fusion_type,
        'modality_dropout': config['modality_dropout'],
        'num_go_terms': num_go_terms,
        'num_parameters': n_params,
        'seed': seed,
        'config': config,
        'test_fmax': float(test_metrics['fmax']),
        'test_micro_auprc': float(test_metrics['micro_auprc']),
        'test_macro_auprc': float(test_metrics['macro_auprc']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_threshold': float(test_metrics['threshold']),
        'weight_seq': float(test_metrics['weight_seq']),
        'weight_text': float(test_metrics['weight_text']),
        'weight_struct': float(test_metrics['weight_struct']),
        'weight_ppi': float(test_metrics['weight_ppi']),
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': int(best_epoch),
        'total_epochs': epoch,
    }
    
    if cafa_metrics:
        for key, value in cafa_metrics.items():
            if isinstance(value, (int, float)):
                results[f'cafa_{key}'] = float(value)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return results


def run_all_experiments(seq_model='prott5', aspects=None, fusion_types=None, 
                        modality_dropout=0.1, output_base='.'):
    """Run all fusion comparison experiments."""
    if aspects is None:
        aspects = ['BPO', 'CCO', 'MFO']
    if fusion_types is None:
        fusion_types = ['concat', 'average', 'gated', 'transformer']
    
    all_results = []
    
    for aspect in aspects:
        for fusion_type in fusion_types:
            print(f"\n{'#'*70}")
            print(f"# {seq_model.upper()} - {aspect} - {fusion_type.upper()}")
            print(f"{'#'*70}")
            
            try:
                results = train_fusion_model(
                    seq_model=seq_model,
                    aspect=aspect,
                    fusion_type=fusion_type,
                    modality_dropout=modality_dropout,
                    output_base=output_base
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError training {fusion_type} for {aspect}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Fusion Method Comparison")
    print("="*80)
    
    summary_df = pd.DataFrame(all_results)
    summary_file = Path(output_base) / 'fusion_comparison' / seq_model / 'summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Print formatted summary
    print(f"\n{'Aspect':<6} {'Fusion':<12} {'Test Fmax':<10} {'CAFA Fmax':<10} {'AUPRC':<10} {'Params':<12}")
    print("-" * 70)
    
    for r in all_results:
        cafa_fmax = r.get('cafa_fmax', '-')
        cafa_str = f"{cafa_fmax:.4f}" if isinstance(cafa_fmax, float) else cafa_fmax
        print(f"{r['aspect']:<6} {r['fusion_type']:<12} {r['test_fmax']:<10.4f} "
              f"{cafa_str:<10} {r['test_micro_auprc']:<10.4f} {r['num_parameters']:<12,}")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fusion Technique Comparison')
    parser.add_argument('--seq-model', type=str, default='prott5', choices=['prott5', 'esm'])
    parser.add_argument('--aspects', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'])
    parser.add_argument('--fusion-types', type=str, nargs='+', 
                        default=['concat', 'average', 'gated', 'transformer', 'seq_anchored'],
                        choices=['concat', 'average', 'gated', 'transformer', 'seq_anchored'])
    parser.add_argument('--modality-dropout', type=float, default=0.1,
                        help='Dropout rate for struct/ppi modalities during training')
    parser.add_argument('--output-base', type=str, default='.')
    parser.add_argument('--single', action='store_true', 
                        help='Run single experiment (first aspect and fusion type)')
    
    args = parser.parse_args()
    
    if args.single:
        # Single experiment mode
        train_fusion_model(
            seq_model=args.seq_model,
            aspect=args.aspects[0],
            fusion_type=args.fusion_types[0],
            modality_dropout=args.modality_dropout,
            output_base=args.output_base
        )
    else:
        # Run all experiments
        run_all_experiments(
            seq_model=args.seq_model,
            aspects=args.aspects,
            fusion_types=args.fusion_types,
            modality_dropout=args.modality_dropout,
            output_base=args.output_base
        )