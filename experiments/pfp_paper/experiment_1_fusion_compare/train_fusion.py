"""Training script for comparing fusion techniques.

Compares: concat, gated_bilinear, multihead_attn
Uses CAFA evaluation for test set benchmarking.
"""
# python ./train_fusion.py --fusion-types gated_bilinear
# python ./train_fusion.py --fusion-types multihead_attn
# python ./train_fusion.py --fusion-types concat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from typing import List
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from fusion_models import (
    MultiModalFusionModel,
    FUSION_REGISTRY,
    SPECIAL_MODEL_TYPES,
    count_fusion_parameters,
    create_model
)
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


def compute_weight_statistics(weight_stats):
    """Compute detailed statistics on fusion weights."""
    weights_array = np.array(weight_stats)
    
    stats = {
        'mean': weights_array.mean(axis=0),
        'std': weights_array.std(axis=0),
        'min': weights_array.min(axis=0),
        'max': weights_array.max(axis=0),
        'median': np.median(weights_array, axis=0),
    }
    
    return stats


def train_epoch(model, loader, optimizer, criterion, scheduler, device, aux_loss_weight=0.0):
    """Training epoch.

    Args:
        aux_loss_weight: Weight η for auxiliary loss (only used if model.use_late_fusion=True)
    """
    model.train()
    total_loss = 0
    total_aux_loss = 0
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

        logits, fusion_weights, aux_outputs = model(
            seq, seq_mask, text, text_mask,
            struct, struct_mask, ppi, ppi_mask
        )

        # Main loss
        loss = criterion(logits, labels)

        # Auxiliary loss (if late fusion enabled)
        if aux_outputs is not None and aux_loss_weight > 0:
            aux_loss = 0
            for name, aux_logits in aux_outputs['aux_logits'].items():
                aux_loss = aux_loss + criterion(aux_logits, labels)
            aux_loss = aux_loss_weight * aux_loss
            loss = loss + aux_loss
            total_aux_loss += aux_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        weight_stats.append(fusion_weights.mean(0).detach().cpu().numpy())

    avg_weights = np.array(weight_stats).mean(0)

    result = {
        'loss': total_loss / len(loader),
        'weight_seq': avg_weights[0],
        'weight_text': avg_weights[1],
        'weight_struct': avg_weights[2],
        'weight_ppi': avg_weights[3]
    }

    if total_aux_loss > 0:
        result['aux_loss'] = total_aux_loss / len(loader)

    return result








def evaluate(model, loader, criterion, device, compute_weight_stats_detailed=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    weight_stats = []
    lambda_values = []

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

            logits, fusion_weights, aux_outputs = model(
                seq, seq_mask, text, text_mask,
                struct, struct_mask, ppi, ppi_mask
            )

            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            weight_stats.append(fusion_weights.detach().cpu().numpy())

            if aux_outputs is not None:
                lambda_values.append(aux_outputs['lambda'].item())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    
    # Compute weight statistics
    all_weights = np.vstack(weight_stats)
    avg_weights = all_weights.mean(0)
    
    result = {
        'loss': total_loss / len(loader),
        'fmax': fmax,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'weight_seq': avg_weights[0],
        'weight_text': avg_weights[1],
        'weight_struct': avg_weights[2],
        'weight_ppi': avg_weights[3],
    }

    # Add late fusion lambda if available
    if lambda_values:
        result['late_fusion_lambda'] = np.mean(lambda_values)

    # Detailed weight statistics for analysis
    if compute_weight_stats_detailed:
        weight_detailed = compute_weight_statistics(all_weights)
        result['weight_stats_detailed'] = {
            'mean': weight_detailed['mean'].tolist(),
            'std': weight_detailed['std'].tolist(),
            'min': weight_detailed['min'].tolist(),
            'max': weight_detailed['max'].tolist(),
            'median': weight_detailed['median'].tolist(),
        }

    return result


def evaluate_with_cafa(model, loader, device, protein_ids, go_terms, obo_file,
                       output_dir, seq_model, fusion_type, train_labels=None):
    """CAFA evaluation for test set with optional IA-weighted metrics."""
    try:
        from cafa_evaluation import evaluate_with_cafa as cafa_eval

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

                logits, _, _ = self.model(
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
            model_name=f"{seq_model}_{fusion_type}",
            train_labels=train_labels
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_fusion_model(seq_model, aspect, fusion_type, modality_dropout=0.1, output_base='.',
                       use_late_fusion=False, aux_loss_weight=0.1):
    """Train a model with specified fusion type.

    Args:
        use_late_fusion: Enable auxiliary heads + hybrid gated late fusion
        aux_loss_weight: Weight η for auxiliary supervision loss (Eq. in paper)
    """

    # Check if experiment already finished
    output_dir = Path(output_base) / 'fusion_comparison' / seq_model / aspect / fusion_type
    results_file = output_dir / "results.json"

    if results_file.exists():
        print(f"\n{'='*70}")
        print(f"SKIPPING: {fusion_type} for {aspect}")
        print(f"Results already exist: {results_file}")
        print(f"{'='*70}\n")

        # Load and return existing results
        with open(results_file, 'r') as f:
            return json.load(f)

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
        # Transformer-specific (kept for backward compatibility)
        'n_heads': 4,
        'n_layers': 2,
        # Late fusion
        'use_late_fusion': use_late_fusion,
        'aux_loss_weight': aux_loss_weight,
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

    fusion_params = count_fusion_parameters(fusion_type, config['hidden_dim'])

    print(f"\n{'='*70}")
    print(f"Training Fusion Model")
    print(f"Fusion Type: {fusion_type.upper()}")
    print(f"Seq Model: {seq_model.upper()}, Aspect: {aspect}")
    print(f"Modality Dropout: {config['modality_dropout']}")
    print(f"Fusion Module Parameters: {fusion_params:,}")
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
    
    # Create model using factory function
    num_go_terms = train_dataset.labels.shape[1]
    seq_dim = 1024 if seq_model == 'prott5' else 1280
    
    model = create_model(
        fusion_type=fusion_type,
        seq_dim=seq_dim,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=config['hidden_dim'],
        num_go_terms=num_go_terms,
        dropout=config['dropout'],
        modality_dropout=config['modality_dropout'],
        use_late_fusion=config['use_late_fusion'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {n_params:,}")
    print(f"Fusion module parameters: {fusion_params:,}")
    if config['use_late_fusion']:
        print(f"Late Fusion: ENABLED (aux_loss_weight={config['aux_loss_weight']})")
    
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
            model, train_loader, optimizer, criterion, scheduler, device,
            aux_loss_weight=config['aux_loss_weight'] if config['use_late_fusion'] else 0.0
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}", end="")
        if 'aux_loss' in train_metrics:
            print(f" (aux: {train_metrics['aux_loss']:.4f})", end="")
        print()
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Fmax: {val_metrics['fmax']:.4f}")
        print(f"  Weights: seq={val_metrics['weight_seq']:.3f}, text={val_metrics['weight_text']:.3f}, "
              f"struct={val_metrics['weight_struct']:.3f}, ppi={val_metrics['weight_ppi']:.3f}", end="")
        if 'late_fusion_lambda' in val_metrics:
            print(f", λ={val_metrics['late_fusion_lambda']:.3f}", end="")
        print()
        
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
    
    # Test evaluation with detailed weight statistics
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    test_metrics = evaluate(model, test_loader, criterion, device, compute_weight_stats_detailed=True)
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_metrics['fmax']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    
    # Print detailed weight statistics
    if 'weight_stats_detailed' in test_metrics:
        print(f"\nFusion Weight Statistics:")
        modality_names = ['seq', 'text', 'struct', 'ppi']
        stats = test_metrics['weight_stats_detailed']
        print(f"  {'Modality':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*42}")
        for i, name in enumerate(modality_names):
            print(f"  {name:<10} {stats['mean'][i]:>8.4f} {stats['std'][i]:>8.4f} "
                  f"{stats['min'][i]:>8.4f} {stats['max'][i]:>8.4f}")
    
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
            fusion_type=fusion_type,
            train_labels=train_dataset.labels
        )
    
    # Save results
    results = {
        'seq_model': seq_model,
        'aspect': aspect,
        'fusion_type': fusion_type,
        'modality_dropout': config['modality_dropout'],
        'num_go_terms': num_go_terms,
        'num_parameters': n_params,
        'fusion_parameters': fusion_params,
        'seed': seed,
        'config': config,
        'test_fmax': float(test_metrics['fmax']),
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
    
    # Add detailed weight statistics
    if 'weight_stats_detailed' in test_metrics:
        results['weight_stats_detailed'] = test_metrics['weight_stats_detailed']

    # Add late fusion lambda if available
    if 'late_fusion_lambda' in test_metrics:
        results['late_fusion_lambda'] = float(test_metrics['late_fusion_lambda'])

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
                        modality_dropout=0.1, output_base='.',
                        use_late_fusion=False, aux_loss_weight=0.1):
    """Run all fusion comparison experiments."""
    if aspects is None:
        aspects = ['MFO', 'CCO','BPO']
    if fusion_types is None:
        fusion_types = ['concat', 'gated_bilinear', 'multihead_attn']

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
                    output_base=output_base,
                    use_late_fusion=use_late_fusion,
                    aux_loss_weight=aux_loss_weight,
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError training {fusion_type} for {aspect}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY: Fusion Method Comparison")
    print("="*90)

    summary_df = pd.DataFrame(all_results)
    summary_file = Path(output_base) / 'fusion_comparison' / seq_model / 'summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Print formatted summary
    print(f"\n{'Aspect':<6} {'Fusion':<18} {'Test Fmax':<10} {'CAFA Fmax':<10} {'Fusion Params':<14}")
    print("-" * 70)

    for r in all_results:
        cafa_fmax = r.get('cafa_fmax', '-')
        cafa_str = f"{cafa_fmax:.4f}" if isinstance(cafa_fmax, float) else cafa_fmax
        fusion_params = r.get('fusion_parameters', 0)
        print(f"{r['aspect']:<6} {r['fusion_type']:<18} {r['test_fmax']:<10.4f} "
              f"{cafa_str:<10} {fusion_params:<14,}")

    print(f"\n✓ Summary saved to: {summary_file}")

    return all_results


class EnsembleWrapper(nn.Module):
    """
    Wraps multiple trained models to average their predictions.
    Behaves like a standard model in the evaluation loop.
    """
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = next(models[0].parameters()).device
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
            
    def forward(self, *args, **kwargs):
        """
        Forward pass averages the PROBABILITIES (sigmoid(logits)) of member models.
        Returns averaged logits (logit(avg_probs)) for compatibility with BCEWithLogitsLoss.
        """
        all_probs = []
        fusion_weights_sum = None

        with torch.no_grad():
            for i, model in enumerate(self.models):
                logits, fusion_w, _ = model(*args, **kwargs)
                probs = torch.sigmoid(logits)
                all_probs.append(probs * self.weights[i])

                # Accumulate fusion weights for visualization (averaged)
                if fusion_weights_sum is None:
                    fusion_weights_sum = fusion_w * self.weights[i]
                else:
                    fusion_weights_sum += fusion_w * self.weights[i]

        # Sum weighted probabilities
        avg_probs = torch.stack(all_probs).sum(dim=0)

        # Convert back to logits for compatibility with existing eval code
        # Clip to avoid log(0) or log(1)
        eps = 1e-6
        avg_probs = torch.clamp(avg_probs, eps, 1 - eps)
        avg_logits = torch.log(avg_probs / (1 - avg_probs))

        return avg_logits, fusion_weights_sum, None

def run_ensemble_evaluation(model_paths: List[Path], test_loader, device, 
                            seq_model, aspect):
    """
    Loads multiple checkpoints and evaluates them as an ensemble.
    """
    print(f"\n{'='*70}")
    print(f"Running Ensemble Evaluation on {len(model_paths)} models")
    print(f"{'='*70}")

    models = []
    # Load config from the first result to rebuild models
    # (Assuming all models in ensemble have compatible architectures/shapes)
    
    # Simple loader logic - assumes you saved args/config alongside model
    # If not, you might need to hardcode dimensions or pass config
    print("Loading models...")
    for path in model_paths:
        # Reconstruct model structure (You need to know the parameters)
        # This is a placeholder - you need to know which fusion type matches the path
        # If your filenames contain the fusion type, parse it:
        f_type = path.parent.name # Assuming folder structure .../fusion_type/best_model.pt
        
        # Re-create model instance
        model = create_model(
            fusion_type=f_type,
            seq_dim=1024 if seq_model == 'prott5' else 1280,
            text_dim=768, struct_dim=512, ppi_dim=512,
            hidden_dim=512, num_go_terms=100, # NOTE: Ensure this matches training!
        ).to(device)
        
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        print(f"  Loaded: {f_type} from {path}")

    # Create Ensemble
    ensemble = EnsembleWrapper(models).to(device)
    
    # Reuse your existing evaluate function
    criterion = nn.BCEWithLogitsLoss().to(device)
    metrics = evaluate(ensemble, test_loader, criterion, device)
    
    print("\nEnsemble Results:")
    print(f"  Fmax: {metrics['fmax']:.4f}")

    return metrics

if __name__ == "__main__":
    import argparse

    available_fusion_types = list(FUSION_REGISTRY.keys())

    parser = argparse.ArgumentParser(description='Fusion Technique Comparison')
    parser.add_argument('--seq-model', type=str, default='prott5', choices=['prott5', 'esm'])
    parser.add_argument('--aspects', type=str, nargs='+', default=['CCO', 'MFO', 'BPO'])
    parser.add_argument('--fusion-types', type=str, nargs='+',
                        default=['concat', 'gated_bilinear', 'multihead_attn'],
                        choices=available_fusion_types,
                        help=f'Fusion types to compare. Available: {available_fusion_types}')
    parser.add_argument('--modality-dropout', type=float, default=0.1,
                        help='Dropout rate for struct/ppi modalities during training')
    parser.add_argument('--output-base', type=str, default='/home/zijianzhou/project/PFP/experiments/pfp_paper/experiment_1_fusion_compare/alldrop')
    parser.add_argument('--single', action='store_true',
                        help='Run single experiment (first aspect and fusion type)')
    parser.add_argument('--use-late-fusion', action='store_true',
                        help='Enable auxiliary heads + hybrid gated late fusion')
    parser.add_argument('--aux-loss-weight', type=float, default=0.1,
                        help='Weight η for auxiliary supervision loss')

    args = parser.parse_args()

    if args.single:
        train_fusion_model(
            seq_model=args.seq_model,
            aspect=args.aspects[0],
            fusion_type=args.fusion_types[0],
            modality_dropout=args.modality_dropout,
            output_base=args.output_base,
            use_late_fusion=args.use_late_fusion,
            aux_loss_weight=args.aux_loss_weight,
        )
    else:
        run_all_experiments(
            seq_model=args.seq_model,
            aspects=args.aspects,
            fusion_types=args.fusion_types,
            modality_dropout=args.modality_dropout,
            output_base=args.output_base,
            use_late_fusion=args.use_late_fusion,
            aux_loss_weight=args.aux_loss_weight,
        )
