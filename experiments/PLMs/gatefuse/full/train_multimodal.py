"""Training script for 4-modality fusion model - simplified version."""

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

from multimodal_model import FourModalityFusion
from multimodal_dataset import MultiModalDatasetNormalized as MultiModalDataset, collate_fn


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Initialize each worker with a different seed."""
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


def train_epoch(model, loader, optimizer, criterion, scheduler, device, 
                aux_weight=0.1, use_aux_heads=True):
    """Training epoch with auxiliary heads."""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    gate_stats = []
    
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
        
        # Forward
        logits, aux_logits, gate_weights = model(
            seq, seq_mask, text, text_mask, 
            struct, struct_mask, ppi, ppi_mask
        )
        
        # Main loss
        main_loss = criterion(logits, labels)
        
        # Auxiliary losses (only for present modalities)
        aux_loss = 0
        if use_aux_heads and model.use_aux_heads:
            if seq_mask.sum() > 0:
                aux_loss += criterion(aux_logits['seq'][seq_mask.squeeze() > 0], 
                                      labels[seq_mask.squeeze() > 0])
            if text_mask.sum() > 0:
                aux_loss += criterion(aux_logits['text'][text_mask.squeeze() > 0], 
                                      labels[text_mask.squeeze() > 0])
            if struct_mask.sum() > 0:
                aux_loss += criterion(aux_logits['struct'][struct_mask.squeeze() > 0], 
                                      labels[struct_mask.squeeze() > 0])
            if ppi_mask.sum() > 0:
                aux_loss += criterion(aux_logits['ppi'][ppi_mask.squeeze() > 0], 
                                      labels[ppi_mask.squeeze() > 0])
            aux_loss = aux_loss / 4  # Average over modalities
        
        # Total loss
        loss = main_loss + aux_weight * aux_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        
        # Track gate statistics
        gate_stats.append(gate_weights.mean(0).detach().cpu().numpy())
    
    avg_gates = np.array(gate_stats).mean(0)
    
    return {
        'total_loss': total_loss / len(loader),
        'main_loss': total_main_loss / len(loader),
        'aux_loss': total_aux_loss / len(loader),
        'gate_seq': avg_gates[0],
        'gate_text': avg_gates[1],
        'gate_struct': avg_gates[2],
        'gate_ppi': avg_gates[3]
    }


def evaluate(model, loader, criterion, device, use_aux_heads=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    gate_stats = []
    
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
            
            logits, aux_logits, gate_weights = model(
                seq, seq_mask, text, text_mask,
                struct, struct_mask, ppi, ppi_mask
            )
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            gate_stats.append(gate_weights.mean(0).detach().cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    avg_gates = np.array(gate_stats).mean(0)
    
    return {
        'loss': total_loss / len(loader),
        'fmax': fmax,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'micro_auprc': micro_auprc,
        'macro_auprc': macro_auprc,
        'gate_seq': avg_gates[0],
        'gate_text': avg_gates[1],
        'gate_struct': avg_gates[2],
        'gate_ppi': avg_gates[3]
    }


def evaluate_with_cafa_multimodal(model, loader, device, protein_ids, go_terms, obo_file, output_dir, seq_model):
    """CAFA evaluation wrapper for multimodal model."""
    try:
        sys.path.append("/home/zijianzhou/project/PFP")
        from text.utils.cafa_evaluation import evaluate_with_cafa
        
        # Determine sequence dimension
        seq_dim = 1024 if seq_model == 'prott5' else 1280
        
        class MultimodalToSingleBatch:
            """Convert multimodal batches to single embedding format for CAFA eval."""
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for batch in self.loader:
                    # Concatenate all modalities and masks
                    embeddings = torch.cat([
                        batch['seq'],           # [B, seq_dim]
                        batch['seq_mask'],      # [B, 1]
                        batch['text'],          # [B, 768]
                        batch['text_mask'],     # [B, 1]
                        batch['struct'],        # [B, 512]
                        batch['struct_mask'],   # [B, 1]
                        batch['ppi'],           # [B, 512]
                        batch['ppi_mask']       # [B, 1]
                    ], dim=-1)
                    
                    yield {
                        'embeddings': embeddings,
                        'labels': batch['labels']
                    }
            
            def __len__(self):
                return len(self.loader)
        
        class ModelWrapper(nn.Module):
            """Wrapper that splits concatenated embeddings back into modalities."""
            def __init__(self, multimodal_model, seq_dim):
                super().__init__()
                self.model = multimodal_model
                self.seq_dim = seq_dim
            
            def forward(self, embeddings):
                # Split concatenated embeddings back into modalities
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
                
                # Forward through model
                logits, _, _ = self.model(
                    seq, seq_mask, text, text_mask,
                    struct, struct_mask, ppi, ppi_mask
                )
                
                return logits
        
        wrapped_model = ModelWrapper(model, seq_dim).to(device)
        wrapped_loader = MultimodalToSingleBatch(loader)
        
        return evaluate_with_cafa(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type='4_modality_fusion',
            model_name=f"{seq_model}_multimodal"
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_multimodal(seq_model, aspect, use_aux_heads=True, modality_dropout=0.1, 
                    aux_weight=1.0, output_base=None):
    """Main training function."""
    seed = 42
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    config = {
        'hidden_dim': 512,
        'dropout': 0.4,
        'modality_dropout': modality_dropout,
        'aux_weight': aux_weight,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'batch_size': 32,
        'max_epochs': 50,
        'patience': 5,
        'warmup_ratio': 0.1,
        'min_delta_fmax': 1e-4,
        'min_delta_loss': 1e-4,
    }
    
    # Embedding directories
    embedding_dirs = {
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/exp_text_embeddings',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/prott5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/esm',
        'struct': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/IF1',
        'ppi': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/ppi'
    }
    
    data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
    
    # Output directory
    if output_base is None:
        output_base = Path("/home/zijianzhou/project/PFP/experiments/PLMs/full/results_multimodal")
    else:
        output_base = Path(output_base)
    
    exp_name = f"mdrop{modality_dropout}_aux{aux_weight}_heads{use_aux_heads}"
    output_dir = output_base / seq_model / aspect / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training 4-Modality Fusion Model")
    print(f"Seq Model: {seq_model.upper()}, Aspect: {aspect}")
    print(f"Modality Dropout: {modality_dropout}, Aux Weight: {aux_weight}")
    print(f"Aux Heads: {use_aux_heads}")
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
    
    # Dataloaders
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
    
    # Create model
    num_go_terms = train_dataset.labels.shape[1]
    seq_dim = 1024 if seq_model == 'prott5' else 1280
    
    model = FourModalityFusion(
        seq_dim=seq_dim,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=config['hidden_dim'],
        num_go_terms=num_go_terms,
        dropout=config['dropout'],
        use_aux_heads=use_aux_heads,
        modality_dropout=config['modality_dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
    
    num_training_steps = len(train_loader) * config['max_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr'], total_steps=num_training_steps,
        pct_start=config['warmup_ratio'], anneal_strategy='cos'
    )
    
    # Training loop
    best_val_fmax = 0.0
    loss_at_best_fmax = float('inf')
    best_epoch_fmax = 0
    best_val_loss_early = float('inf')
    early_stop_epoch = 0
    patience_counter = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(1, config['max_epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device,
            aux_weight=aux_weight, use_aux_heads=use_aux_heads
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, use_aux_heads)
        
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Fmax: {val_metrics['fmax']:.4f}")
        print(f"  Gates: seq={val_metrics['gate_seq']:.3f}, text={val_metrics['gate_text']:.3f}, "
              f"struct={val_metrics['gate_struct']:.3f}, ppi={val_metrics['gate_ppi']:.3f}")
        
        history.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Checkpoint on Fmax
        current_fmax = val_metrics['fmax']
        current_loss = val_metrics['loss']
        
        fmax_better = current_fmax > best_val_fmax + config['min_delta_fmax']
        fmax_similar = abs(current_fmax - best_val_fmax) <= config['min_delta_fmax']
        loss_better = current_loss < loss_at_best_fmax - config['min_delta_loss']
        
        if fmax_better or (fmax_similar and loss_better):
            best_val_fmax = current_fmax
            loss_at_best_fmax = current_loss
            best_epoch_fmax = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Best model saved (Fmax: {best_val_fmax:.4f})")
        
        # Early stopping on loss
        if current_loss < best_val_loss_early - config['min_delta_loss']:
            best_val_loss_early = current_loss
            early_stop_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device, use_aux_heads)
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_metrics['fmax']:.4f}")
    print(f"  Micro-AUPRC: {test_metrics['micro_auprc']:.4f}")
    print(f"  Macro-AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"  Gates: seq={test_metrics['gate_seq']:.3f}, text={test_metrics['gate_text']:.3f}, "
          f"struct={test_metrics['gate_struct']:.3f}, ppi={test_metrics['gate_ppi']:.3f}")
    
    # CAFA evaluation
    print("\n" + "="*70)
    print("CAFA EVALUATION")
    print("="*70)
    
    obo_file = Path("/home/zijianzhou/project/PFP/go.obo")
    cafa_metrics = {}
    
    if obo_file.exists():
        go_terms_file = data_dir / f"{aspect}_go_terms.json"
        with open(go_terms_file, 'r') as f:
            go_terms = json.load(f)
        
        test_protein_ids = test_dataset.protein_ids.tolist()
        
        cafa_metrics = evaluate_with_cafa_multimodal(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir / 'cafa_eval',
            seq_model=seq_model
        )
        
        if cafa_metrics:
            print("\nCAFA Metrics:")
            for key, value in cafa_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    else:
        print(f"Warning: GO OBO file not found at {obo_file}")
        print("Skipping CAFA evaluation")
    
    # Save results
    results = {
        'seq_model': seq_model,
        'aspect': aspect,
        'modality_dropout': modality_dropout,
        'aux_weight': aux_weight,
        'use_aux_heads': use_aux_heads,
        'num_go_terms': num_go_terms,
        'num_parameters': n_params,
        'seed': seed,
        'config': config,
        'test_fmax': float(test_metrics['fmax']),
        'test_micro_auprc': float(test_metrics['micro_auprc']),
        'test_macro_auprc': float(test_metrics['macro_auprc']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'gate_seq': float(test_metrics['gate_seq']),
        'gate_text': float(test_metrics['gate_text']),
        'gate_struct': float(test_metrics['gate_struct']),
        'gate_ppi': float(test_metrics['gate_ppi']),
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': int(best_epoch_fmax),
        'total_epochs': epoch,
    }
    
    # Add CAFA metrics if available - CRITICAL for evaluation
    if cafa_metrics:
        print(f"\nSaving CAFA metrics: {list(cafa_metrics.keys())}")
        for key, value in cafa_metrics.items():
            if isinstance(value, (int, float)):
                results[f'cafa_{key}'] = float(value)
        print(f"✓ CAFA metrics saved: {[k for k in results.keys() if k.startswith('cafa_')]}")
    else:
        print("\nWarning: No CAFA metrics to save")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-model', type=str, default='prott5', choices=['prott5', 'esm'])
    parser.add_argument('--aspects', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'],
                        help='GO aspects to train (default: all three)')
    parser.add_argument('--modality-dropout', type=float, default=0.1)
    parser.add_argument('--aux-weight', type=float, default=1.0)
    parser.add_argument('--no-aux-heads', action='store_true')
    parser.add_argument('--output-base', type=str, default=None)
    
    args = parser.parse_args()
    
    # Run for all specified aspects
    for aspect in args.aspects:
        train_multimodal(
            seq_model=args.seq_model,
            aspect=aspect,
            use_aux_heads=not args.no_aux_heads,
            modality_dropout=args.modality_dropout,
            aux_weight=args.aux_weight,
            output_base=args.output_base
        )