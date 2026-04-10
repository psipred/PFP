"""Training script for multimodal protein function prediction.

Compares fusion techniques: concat, gated_bilinear
Uses CAFA evaluation for test set benchmarking.

Usage:
    python train.py --fusion-types gated_bilinear
    python train.py --aspects BPO CCO MFO --fusion-types concat gated_bilinear
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import argparse
from typing import List, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mmfp.models import (
    MultiModalFusionModel,
    FUSION_REGISTRY,
    count_fusion_parameters,
    create_model
)
from mmfp.dataset import MultiModalDataset, collate_fn
from mmfp.evaluation import evaluate_with_cafa, compute_fmax


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with unique seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device: str,
    aux_loss_weight: float = 0.0
) -> Dict:
    """Training epoch.

    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler
        device: Device to use
        aux_loss_weight: Weight for auxiliary loss (late fusion only)

    Returns:
        Dict with training metrics
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

        loss = criterion(logits, labels)

        # Auxiliary loss (if late fusion enabled)
        if aux_outputs is not None and aux_loss_weight > 0:
            aux_loss = 0
            for _, aux_logits in aux_outputs['aux_logits'].items():
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
        'loss': total_loss / max(1, len(loader)),
        'weight_seq': float(avg_weights[0]),
        'weight_text': float(avg_weights[1]),
        'weight_struct': float(avg_weights[2]),
        'weight_ppi': float(avg_weights[3]),
    }
    if total_aux_loss > 0:
        result['aux_loss'] = total_aux_loss / max(1, len(loader))

    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    compute_weight_stats_detailed: bool = False
) -> Dict:
    """Evaluate model.

    Args:
        model: Model to evaluate
        loader: Evaluation data loader
        criterion: Loss function
        device: Device to use
        compute_weight_stats_detailed: If True, compute detailed weight statistics

    Returns:
        Dict with evaluation metrics
    """
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

            if aux_outputs is not None and 'lambda' in aux_outputs:
                lambda_values.append(float(aux_outputs['lambda'].item()))

    y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, 0), dtype=np.float32)
    y_true = np.vstack(all_labels) if all_labels else np.zeros((0, 0), dtype=np.float32)

    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)

    try:
        micro_auprc = average_precision_score(y_true.ravel(), y_pred.ravel())
    except Exception:
        micro_auprc = float('nan')

    macro_scores = []
    for i in range(y_true.shape[1] if y_true.ndim == 2 else 0):
        if np.any(y_true[:, i] == 1):
            try:
                macro_scores.append(average_precision_score(y_true[:, i], y_pred[:, i]))
            except Exception:
                pass
    macro_auprc = float(np.mean(macro_scores)) if macro_scores else float('nan')

    all_weights = np.vstack(weight_stats) if weight_stats else np.zeros((0, 4), dtype=np.float32)
    avg_weights = all_weights.mean(0) if len(all_weights) else np.zeros(4, dtype=np.float32)

    result = {
        'loss': total_loss / max(1, len(loader)),
        'fmax': float(fmax),
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'micro_auprc': float(micro_auprc),
        'macro_auprc': float(macro_auprc),
        'weight_seq': float(avg_weights[0]),
        'weight_text': float(avg_weights[1]),
        'weight_struct': float(avg_weights[2]),
        'weight_ppi': float(avg_weights[3]),
    }

    if lambda_values:
        result['late_fusion_lambda'] = float(np.mean(lambda_values))

    if compute_weight_stats_detailed and len(all_weights):
        weights_array = all_weights
        result['weight_stats_detailed'] = {
            'mean': weights_array.mean(axis=0).tolist(),
            'std': weights_array.std(axis=0).tolist(),
            'min': weights_array.min(axis=0).tolist(),
            'max': weights_array.max(axis=0).tolist(),
            'median': np.median(weights_array, axis=0).tolist(),
        }

    return result


def train_fusion_model(
    seq_model: str,
    aspect: str,
    fusion_type: str,
    data_dir: str,
    embedding_dirs: Dict[str, str],
    obo_file: str,
    train_embedding_dirs: Optional[Dict[str, str]] = None,
    val_embedding_dirs: Optional[Dict[str, str]] = None,
    test_embedding_dirs: Optional[Dict[str, str]] = None,
    modality_dropout: float = 0.1,
    output_base: str = '.',
    use_late_fusion: bool = False,
    late_output_mode: str = 'hybrid',
    aux_loss_weight: float = 0.1,
    num_workers: int = 8,
    ia_file: Optional[str] = None,
    seed: int = 42,
    ablation_name: Optional[str] = None,
    config_label: Optional[str] = None,
) -> Dict:
    """Train a model with specified fusion type.

    Args:
        seq_model: Sequence model type ('prott5' or 'esm')
        aspect: GO aspect ('BPO', 'CCO', or 'MFO')
        fusion_type: Fusion method to use
        data_dir: Path to data directory
        embedding_dirs: Default dict mapping modality to embedding directory
        train_embedding_dirs: Optional train split embedding dirs override
        val_embedding_dirs: Optional validation split embedding dirs override
        test_embedding_dirs: Optional test split embedding dirs override
        obo_file: Path to GO ontology file
        modality_dropout: Dropout rate for modalities during training
        output_base: Base output directory
        use_late_fusion: Whether to use late fusion with auxiliary heads
        late_output_mode: 'hybrid' or 'aux_only' when late fusion is enabled
        aux_loss_weight: Weight for auxiliary loss
        num_workers: Number of DataLoader workers
        ia_file: Optional path to precomputed IA file
        seed: Random seed for training
        ablation_name: Optional human-readable ablation label
        config_label: Optional technical configuration label

    Returns:
        Dict with training results
    """
    output_dir = Path(output_base) / 'fusion_comparison' / seq_model / aspect / fusion_type
    results_file = output_dir / "results.json"

    # Skip if already trained
    if results_file.exists():
        print(f"\n{'='*70}")
        print(f"SKIPPING: {fusion_type} for {aspect}")
        print(f"Results already exist: {results_file}")
        print(f"{'='*70}\n")
        with open(results_file, 'r') as f:
            return json.load(f)

    set_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration
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
        'use_late_fusion': use_late_fusion,
        'late_output_mode': late_output_mode,
        'aux_loss_weight': aux_loss_weight,
    }

    data_dir = Path(data_dir)
    obo_file = Path(obo_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_embedding_dirs = train_embedding_dirs or dict(embedding_dirs)
    val_embedding_dirs = val_embedding_dirs or dict(embedding_dirs)
    test_embedding_dirs = test_embedding_dirs or dict(embedding_dirs)

    fusion_params = count_fusion_parameters(fusion_type, config['hidden_dim'])

    print(f"\n{'='*70}")
    print(f"Training Fusion Model")
    print(f"Fusion Type: {fusion_type.upper()}")
    print(f"Seq Model: {seq_model.upper()}, Aspect: {aspect}")
    print(f"Seed: {seed}")
    print(f"Modality Dropout: {config['modality_dropout']}")
    print(f"Fusion Module Parameters: {fusion_params:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = MultiModalDataset(
        data_dir, train_embedding_dirs, seq_model, aspect, 'train',
        normalize='standard'
    )
    val_dataset = MultiModalDataset(
        data_dir, val_embedding_dirs, seq_model, aspect, 'valid',
        normalize='standard', norm_stats=train_dataset.norm_stats
    )
    test_dataset = MultiModalDataset(
        data_dir, test_embedding_dirs, seq_model, aspect, 'test',
        normalize='standard', norm_stats=train_dataset.norm_stats
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device == 'cuda'),
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=(device == 'cuda')
    )

    # Load GO terms for CAFA evaluation
    go_terms_file = data_dir / f"{aspect}_go_terms.json"
    with open(go_terms_file, 'r') as f:
        go_terms = json.load(f)

    test_protein_ids = test_dataset.protein_ids.tolist()

    # Create model
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
        late_output_mode=config['late_output_mode'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {n_params:,}")
    print(f"Fusion module parameters: {fusion_params:,}")
    if config['use_late_fusion']:
        print(
            f"Late Fusion: ENABLED "
            f"(mode={config['late_output_mode']}, aux_loss_weight={config['aux_loss_weight']})"
        )

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
            print(f", lambda={val_metrics['late_fusion_lambda']:.3f}", end="")
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
            print(f"  * Best model saved (Fmax: {best_val_fmax:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)

    test_metrics = evaluate(model, test_loader, criterion, device, compute_weight_stats_detailed=True)

    print(f"\nTest Results:")
    print(f"  Fmax: {test_metrics['fmax']:.4f}")
    print(f"  Micro-AUPRC: {test_metrics['micro_auprc']:.4f}")
    print(f"  Macro-AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

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
    cafa_metrics = {}
    if obo_file.exists():
        print("\n" + "="*70)
        print("CAFA EVALUATION")
        print("="*70)

        cafa_metrics = evaluate_with_cafa(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=str(obo_file),
            output_dir=str(output_dir / 'cafa_eval'),
            model_name=f"{seq_model}_{fusion_type}",
            train_labels=train_dataset.labels,
            ia_file=ia_file
        )

    # Save results
    results = {
        'seq_model': seq_model,
        'aspect': aspect,
        'fusion_type': fusion_type,
        'ablation_name': ablation_name,
        'config_label': config_label,
        'use_late_fusion': bool(config['use_late_fusion']),
        'late_output_mode': config['late_output_mode'],
        'text_embedding_dirs_by_split': {
            'train': str(train_embedding_dirs['text']),
            'valid': str(val_embedding_dirs['text']),
            'test': str(test_embedding_dirs['text']),
        },
        'modality_dropout': float(config['modality_dropout']),
        'num_go_terms': int(num_go_terms),
        'num_parameters': int(n_params),
        'fusion_parameters': int(fusion_params),
        'seed': int(seed),
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
        'total_epochs': int(epoch),
    }

    if 'weight_stats_detailed' in test_metrics:
        results['weight_stats_detailed'] = test_metrics['weight_stats_detailed']
    if 'late_fusion_lambda' in test_metrics:
        results['late_fusion_lambda'] = float(test_metrics['late_fusion_lambda'])

    if cafa_metrics:
        for key, value in cafa_metrics.items():
            if isinstance(value, (int, float)):
                results[f'cafa_{key}'] = float(value)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    print(f"\n* Results saved to: {output_dir}")

    return results


def run_all_experiments(
    seq_model: str = 'prott5',
    aspects: Optional[List[str]] = None,
    fusion_types: Optional[List[str]] = None,
    data_dir: str = './data',
    embedding_dirs: Optional[Dict[str, str]] = None,
    obo_file: str = './go.obo',
    modality_dropout: float = 0.1,
    output_base: str = '.',
    use_late_fusion: bool = False,
    late_output_mode: str = 'hybrid',
    aux_loss_weight: float = 0.1,
    num_workers: int = 8,
    ia_file_dir: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """Run all fusion comparison experiments.

    Args:
        seq_model: Sequence model type
        aspects: List of GO aspects to train
        fusion_types: List of fusion methods to compare
        data_dir: Path to data directory
        embedding_dirs: Dict mapping modality to embedding directory
        obo_file: Path to GO ontology file
        modality_dropout: Dropout rate for modalities
        output_base: Base output directory
        use_late_fusion: Whether to use late fusion
        late_output_mode: 'hybrid' or 'aux_only' when late fusion is enabled
        aux_loss_weight: Weight for auxiliary loss
        num_workers: Number of DataLoader workers
        ia_file_dir: Optional directory containing <ASPECT>_ia.txt files
        seed: Random seed for training

    Returns:
        List of result dicts for each experiment
    """
    if aspects is None:
        aspects = ['MFO', 'CCO', 'BPO']
    if fusion_types is None:
        fusion_types = ['concat', 'gated_bilinear']

    # Default embedding directories
    if embedding_dirs is None:
        embedding_dirs = {
            'text': f'{data_dir}/embedding_cache/exp_text_embeddings',
            'prott5': f'{data_dir}/embedding_cache/prott5',
            'esm': f'{data_dir}/embedding_cache/esm',
            'struct': f'{data_dir}/embedding_cache/IF1',
            'ppi': f'{data_dir}/embedding_cache/ppi'
        }

    all_results = []

    for aspect in aspects:
        for fusion_type in fusion_types:
            print(f"\n{'#'*70}")
            print(f"# {seq_model.upper()} - {aspect} - {fusion_type.upper()}")
            print(f"{'#'*70}")

            try:
                ia_file = None
                if ia_file_dir:
                    candidate = Path(ia_file_dir) / f"{aspect}_ia.txt"
                    if candidate.exists():
                        ia_file = str(candidate)
                results = train_fusion_model(
                    seq_model=seq_model,
                    aspect=aspect,
                    fusion_type=fusion_type,
                    data_dir=data_dir,
                    embedding_dirs=embedding_dirs,
                    obo_file=obo_file,
                    modality_dropout=modality_dropout,
                    output_base=output_base,
                    use_late_fusion=use_late_fusion,
                    late_output_mode=late_output_mode,
                    aux_loss_weight=aux_loss_weight,
                    num_workers=num_workers,
                    ia_file=ia_file,
                    seed=seed,
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError training {fusion_type} for {aspect}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print summary
    print("\n" + "="*90)
    print("SUMMARY: Fusion Method Comparison")
    print("="*90)

    summary_df = pd.DataFrame(all_results)
    summary_file = Path(output_base) / 'fusion_comparison' / seq_model / 'summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"\n{'Aspect':<6} {'Fusion':<18} {'Test Fmax':<10} {'CAFA Fmax':<10} {'Fusion Params':<14}")
    print("-" * 70)

    for r in all_results:
        cafa_fmax = r.get('cafa_fmax', '-')
        cafa_str = f"{cafa_fmax:.4f}" if isinstance(cafa_fmax, float) else cafa_fmax
        fusion_params = r.get('fusion_parameters', 0)
        print(f"{r['aspect']:<6} {r['fusion_type']:<18} {r['test_fmax']:<10.4f} "
              f"{cafa_str:<10} {fusion_params:<14,}")

    print(f"\n* Summary saved to: {summary_file}")
    return all_results


if __name__ == "__main__":
    available_fusion_types = list(FUSION_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description='MMFP: Multimodal Fusion for Protein Function Prediction'
    )
    parser.add_argument('--seq-model', type=str, default='prott5',
                        choices=['prott5', 'esm'],
                        help='Sequence model type')
    parser.add_argument('--aspects', type=str, nargs='+',
                        default=['BPO', 'MFO', 'CCO'],
                        help='GO aspects to train')
    parser.add_argument('--fusion-types', type=str, nargs='+',
                        default=['concat', 'gated_bilinear'],
                        choices=available_fusion_types,
                        help=f'Fusion types to compare. Available: {available_fusion_types}')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--obo-file', type=str, default='./go.obo',
                        help='Path to GO ontology file')
    parser.add_argument('--modality-dropout', type=float, default=0.1,
                        help='Dropout rate for non-sequence modalities during training')
    parser.add_argument('--output-base', type=str, default='./results',
                        help='Base output directory')
    parser.add_argument('--single', action='store_true',
                        help='Run single experiment (first aspect and fusion type)')
    parser.add_argument('--use-late-fusion', action='store_true',
                        help='Enable auxiliary heads + hybrid gated late fusion')
    parser.add_argument('--late-output-mode', type=str, default='hybrid',
                        choices=['hybrid', 'aux_only'],
                        help='Final prediction path when late fusion is enabled')
    parser.add_argument('--aux-loss-weight', type=float, default=0.8,
                        help='Weight for auxiliary supervision loss')
    parser.add_argument('--text-embedding-dir', type=str, default=None,
                        help='Override text embedding directory (for historical text experiments)')
    parser.add_argument('--train-text-embedding-dir', type=str, default=None,
                        help='Override train split text embedding directory')
    parser.add_argument('--valid-text-embedding-dir', type=str, default=None,
                        help='Override validation split text embedding directory')
    parser.add_argument('--test-text-embedding-dir', type=str, default=None,
                        help='Override test split text embedding directory')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--ia-file-dir', type=str, default=None,
                        help='Directory containing precomputed <ASPECT>_ia.txt files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training')
    parser.add_argument('--no-struct', action='store_true',
                        help='Disable struct modality (for seq+text experiments)')
    parser.add_argument('--no-ppi', action='store_true',
                        help='Disable PPI modality (for seq+text experiments)')

    args = parser.parse_args()

    # Default embedding directories
    embedding_dirs = {
        'text': f'{args.data_dir}/embedding_cache/exp_text_embeddings',
        'prott5': f'{args.data_dir}/embedding_cache/prott5',
        'esm': f'{args.data_dir}/embedding_cache/esm',
        'struct': f'{args.data_dir}/embedding_cache/IF1',
        'ppi': f'{args.data_dir}/embedding_cache/ppi'
    }

    if args.text_embedding_dir:
        embedding_dirs['text'] = args.text_embedding_dir

    train_embedding_dirs = dict(embedding_dirs)
    val_embedding_dirs = dict(embedding_dirs)
    test_embedding_dirs = dict(embedding_dirs)

    if args.train_text_embedding_dir:
        train_embedding_dirs['text'] = args.train_text_embedding_dir
    if args.valid_text_embedding_dir:
        val_embedding_dirs['text'] = args.valid_text_embedding_dir
    if args.test_text_embedding_dir:
        test_embedding_dirs['text'] = args.test_text_embedding_dir

    # For seq+text experiments, point to empty dirs so masks = 0
    if args.no_struct or args.no_ppi:
        empty_dir = Path(args.output_base) / '_empty_modality'
        empty_dir.mkdir(parents=True, exist_ok=True)
        if args.no_struct:
            embedding_dirs['struct'] = str(empty_dir)
            train_embedding_dirs['struct'] = str(empty_dir)
            val_embedding_dirs['struct'] = str(empty_dir)
            test_embedding_dirs['struct'] = str(empty_dir)
        if args.no_ppi:
            embedding_dirs['ppi'] = str(empty_dir)
            train_embedding_dirs['ppi'] = str(empty_dir)
            val_embedding_dirs['ppi'] = str(empty_dir)
            test_embedding_dirs['ppi'] = str(empty_dir)

    if args.single:
        train_fusion_model(
            seq_model=args.seq_model,
            aspect=args.aspects[0],
            fusion_type=args.fusion_types[0],
            data_dir=args.data_dir,
            embedding_dirs=embedding_dirs,
            train_embedding_dirs=train_embedding_dirs,
            val_embedding_dirs=val_embedding_dirs,
            test_embedding_dirs=test_embedding_dirs,
            obo_file=args.obo_file,
            modality_dropout=args.modality_dropout,
            output_base=args.output_base,
            use_late_fusion=args.use_late_fusion,
            late_output_mode=args.late_output_mode,
            aux_loss_weight=args.aux_loss_weight,
            num_workers=args.num_workers,
            ia_file=(
                str(Path(args.ia_file_dir) / f"{args.aspects[0]}_ia.txt")
                if args.ia_file_dir and (Path(args.ia_file_dir) / f"{args.aspects[0]}_ia.txt").exists()
                else None
            ),
            seed=args.seed,
        )
    else:
        run_all_experiments(
            seq_model=args.seq_model,
            aspects=args.aspects,
            fusion_types=args.fusion_types,
            data_dir=args.data_dir,
            embedding_dirs=embedding_dirs,
            obo_file=args.obo_file,
            modality_dropout=args.modality_dropout,
            output_base=args.output_base,
            use_late_fusion=args.use_late_fusion,
            late_output_mode=args.late_output_mode,
            aux_loss_weight=args.aux_loss_weight,
            num_workers=args.num_workers,
            ia_file_dir=args.ia_file_dir,
            seed=args.seed,
        )
