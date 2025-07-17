#!/usr/bin/env python3
"""
Simplified CAFA3 training script with unified fusion support
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as ssp
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append('/SAN/bioinf/PFP/PFP')

from Network.base_go_classifier import BaseGOClassifier
from Network.model_utils import EarlyStop
from fusion_models import (
    ConcatFusion,
    GatedMultimodalFusion, 
    AdaptiveMoEFusion,
    MultimodalTransformerFusion,
    ContrastiveMultimodalFusion,
    EnhancedGatedFusion, GateDiagnostics, train_epoch_with_diagnostics, staged_training
)


def collate_batch(batch):
    """Simple collate function for batching."""
    names, features, labels = zip(*batch)
    
    # Stack features for each modality
    features_dict = {}
    for feat_name in features[0].keys():
        features_dict[feat_name] = torch.stack([f[feat_name] for f in features])
    
    labels = torch.stack(labels)
    
    return names, features_dict, labels


class CAFA3Dataset(Dataset):
    """Unified dataset for CAFA3 supporting all embeddings."""
    
    def __init__(self, names_file, labels_file, features, embeddings_dir):
        self.names = np.load(names_file, allow_pickle=True)
        self.labels = torch.from_numpy(ssp.load_npz(labels_file).toarray()).float()
        self.features = features
        self.embeddings_dir = embeddings_dir
        
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]
        
        features_dict = {}
        for feat in self.features:
            emb_file = Path(self.embeddings_dir[feat]) / f"{name}.npy"
            if emb_file.exists():
                data = np.load(emb_file, allow_pickle=True).item()
                emb = data['embedding'] if isinstance(data, dict) else data
                if emb.ndim == 2:
                    emb = emb.mean(axis=0)
                features_dict[feat] = torch.from_numpy(emb).float()
            else:
                # Use zero embedding if file not found
                dim = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat]
                features_dict[feat] = torch.zeros(dim)
                
        return name, features_dict, label


def create_model(cfg, device):
    """Create model based on configuration."""
    features = cfg['dataset']['features']
    output_dim = cfg['model']['output_dim']
    
    if len(features) == 1:
        # Single modality
        feat = features[0]
        input_dim = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat]
        return BaseGOClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            projection_dim=1024,
            hidden_dim=512
        ).to(device)
    
    # Dual modality fusion
    feat1, feat2 = features
    dim1 = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat1]
    dim2 = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat2]
    
    fusion_type = cfg['model'].get('fusion_type', 'concat')
    hidden_dim = cfg['model'].get('hidden_dim', 512)
    print(fusion_type)
    fusion_models = {
        'concat': ConcatFusion,
        'gated': GatedMultimodalFusion,
        'moe': lambda: AdaptiveMoEFusion(
            esm_dim=dim1, text_dim=dim2, hidden_dim=hidden_dim, 
            output_dim=output_dim, num_experts_per_modality=cfg['model'].get('num_experts', 3)
        ),
        'transformer': lambda: MultimodalTransformerFusion(
            esm_dim=dim1, text_dim=dim2, hidden_dim=hidden_dim,
            output_dim=output_dim, num_layers=cfg['model'].get('num_layers', 4)
        ),
        'contrastive': lambda: ContrastiveMultimodalFusion(
            esm_dim=dim1, text_dim=dim2, hidden_dim=hidden_dim,
            output_dim=output_dim, temperature=cfg['model'].get('temperature', 0.07)
        )
    }
    
    if fusion_type in fusion_models:
        if fusion_type in ['concat', 'gated']:
            return fusion_models[fusion_type](dim1, dim2, hidden_dim, output_dim).to(device)
        else:
            return fusion_models[fusion_type]().to(device)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def train_epoch(model, loader, optimizer, criterion, device, cfg, epoch):
    """Training epoch for all model types."""
    model.train()
    total_loss = 0
    features = cfg['dataset']['features']
    
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        names, features_batch, labels = batch
        labels = labels.to(device)
        
        # Get predictions
        if len(features) == 1:
            # Single modality
            feat = features[0]
            predictions = model(features_batch[feat].to(device))
            aux_outputs = {}
        else:
            # Dual modality
            feat1, feat2 = features
            output = model(features_batch[feat1].to(device), features_batch[feat2].to(device))
            if isinstance(output, tuple):
                predictions, aux_outputs = output
            else:
                predictions, aux_outputs = output, {}
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Add auxiliary losses if present
        if 'contrastive_loss' in aux_outputs and aux_outputs['contrastive_loss'] is not None:
            loss += cfg['model'].get('contrastive_weight', 0.5) * aux_outputs['contrastive_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if cfg['optim'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optim']['gradient_clip'])
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, cfg):
    """Validation for all model types."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    features = cfg['dataset']['features']
    
    with torch.no_grad():
        for batch in loader:
            names, features_batch, labels = batch
            labels = labels.to(device)
            
            # Get predictions
            if len(features) == 1:
                feat = features[0]
                predictions = model(features_batch[feat].to(device))
            else:
                feat1, feat2 = features
                output = model(features_batch[feat1].to(device), features_batch[feat2].to(device))
                predictions = output[0] if isinstance(output, tuple) else output
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(predictions).cpu())
            all_labels.append(labels.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Simple F-max calculation
    fmax = calculate_fmax(all_preds, all_labels)
    
    return {
        'loss': total_loss / len(loader),
        'Fmax_protein': fmax,
        'macro_AP': calculate_map(all_preds, all_labels)
    }


def calculate_fmax(preds, labels, thresholds=np.arange(0.01, 1.0, 0.01)):
    """Calculate protein-centric F-max."""
    fmax = 0
    for t in thresholds:
        pred_binary = (preds >= t).float()
        tp = (pred_binary * labels).sum(dim=1)
        fp = (pred_binary * (1 - labels)).sum(dim=1)
        fn = ((1 - pred_binary) * labels).sum(dim=1)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        fmax = max(fmax, f1.mean().item())
    
    return fmax


def calculate_map(preds, labels):
    """Calculate mean average precision."""
    n_classes = labels.shape[1]
    aps = []
    
    for i in range(n_classes):
        if labels[:, i].sum() > 0:
            ap = average_precision_score(labels[:, i].numpy(), preds[:, i].numpy())
            aps.append(ap)
    
    return np.mean(aps) if aps else 0


def average_precision_score(y_true, y_score):
    """Simple average precision calculation."""
    indices = np.argsort(-y_score)
    y_true = y_true[indices]
    y_score = y_score[indices]
    
    tp = y_true.cumsum()
    fp = (1 - y_true).cumsum()
    
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    
    # Calculate AP
    ap = 0
    prev_recall = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            ap += precision[i] * (recall[i] - prev_recall)
            prev_recall = recall[i]
    
    return ap


def generate_predictions(model, cfg, device, output_dir):
    """Generate test predictions."""
    logger.info("Generating test predictions...")
    
    aspect = cfg['experiment_name'].split('_')[-1]
    data_dir = Path(cfg['dataset']['train_names']).parent
    features = cfg['dataset']['features']
    
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'prott5': '/SAN/bioinf/PFP/embeddings/cafa3/prott5',
        'prostt5': '/SAN/bioinf/PFP/embeddings/cafa3/prostt5',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text'
    }
    
    # Create test dataset
    test_dataset = CAFA3Dataset(
        names_file=str(data_dir / f"{aspect}_test_names.npy"),
        labels_file=str(data_dir / f"{aspect}_test_labels.npz"),
        features=features,
        embeddings_dir=embeddings_dir
    )
    
    test_loader = DataLoader(test_dataset, batch_size=cfg['dataset'].get('batch_size', 32), 
                           shuffle=False, collate_fn=collate_batch)
    
    # Generate predictions
    model.eval()
    all_preds = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            names, features_batch, _ = batch
            
            # Get predictions
            if len(features) == 1:
                feat = features[0]
                predictions = model(features_batch[feat].to(device))
            else:
                feat1, feat2 = features
                output = model(features_batch[feat1].to(device), features_batch[feat2].to(device))
                predictions = output[0] if isinstance(output, tuple) else output
            
            all_preds.append(torch.sigmoid(predictions).cpu().numpy())
            all_names.extend(names)
    
    # Save predictions
    all_preds = np.vstack(all_preds)
    
    with open(data_dir / f"{aspect}_go_terms.json", 'r') as f:
        go_terms = json.load(f)
    
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    model_name = cfg['experiment_name'].replace('CAFA3_', '')
    pred_file = pred_dir / f"{model_name}.tsv"
    
    with open(pred_file, 'w') as f:
        for i, protein_id in enumerate(all_names):
            for j, go_term in enumerate(go_terms):
                score = all_preds[i, j]
                if score > 0.01:
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")
    
    logger.info(f"Predictions saved to {pred_file}")


def train_cafa3_model(config_path):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup output directory
    output_dir = Path(cfg['log']['out_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting: {cfg['experiment_name']}")
    logger.info(f"Features: {cfg['dataset']['features']}")
    
    # Extract aspect
    aspect = cfg['experiment_name'].split('_')[-1]
    data_dir = Path(cfg['dataset']['train_names']).parent
    
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'prott5': '/SAN/bioinf/PFP/embeddings/cafa3/prott5',
        'prostt5': '/SAN/bioinf/PFP/embeddings/cafa3/prostt5',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text'
    }
    
    # Create datasets
    train_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['train_names'],
        labels_file=cfg['dataset']['train_labels'],
        features=cfg['dataset']['features'],
        embeddings_dir=embeddings_dir
    )
    
    valid_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['valid_names'],
        labels_file=cfg['dataset']['valid_labels'],
        features=cfg['dataset']['features'],
        embeddings_dir=embeddings_dir
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset'].get('batch_size', 32), 
                            shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['dataset'].get('batch_size', 32), 
                            shuffle=False, collate_fn=collate_batch)
    
    # Create model
    
    model = create_model(cfg, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stop = EarlyStop(patience=cfg['optim']['patience'], min_epochs=cfg['optim'].get('min_epochs', 10))
    
    # Training loop
    best_fmax = 0.0
    history = []
    
    for epoch in range(1, cfg['optim']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['optim']['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg, epoch)
        val_metrics = validate(model, valid_loader, criterion, device, cfg)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_metrics['loss']:.4f}, F-max: {val_metrics['Fmax_protein']:.4f}")
        
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})
        
        # Early stopping
        current_fmax = val_metrics['Fmax_protein']
        early_stop(-current_fmax, current_fmax, model)
        
        if current_fmax > best_fmax:
            best_fmax = current_fmax
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_fmax': best_fmax,
                'config': cfg
            }, output_dir / 'best_model.pt')
        
        if early_stop.stop():
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save results
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({
            'best_Fmax_protein': best_fmax,
            'final_epoch': epoch,
            'experiment': cfg['experiment_name'],
            'features': cfg['dataset']['features']
        }, f, indent=2)
    
    import pandas as pd
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    
    # Generate test predictions
    generate_predictions(model, cfg, device, output_dir)
    
    logger.info(f"Training complete! Best F-max: {best_fmax:.4f}")

def create_diagnostic_model(cfg, device):
    """Create enhanced model with diagnostics."""
    features = cfg['dataset']['features']
    output_dim = cfg['model']['output_dim']
    
    if len(features) == 2 and cfg['model'].get('fusion_type') == 'gated':
        feat1, feat2 = features
        dim1 = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat1]
        dim2 = {'esm': 1280, 'prott5': 1024, 'prostt5': 1024, 'text': 768}[feat2]
        
        return EnhancedGatedFusion(
            esm_dim=dim1,
            text_dim=dim2,
            hidden_dim=cfg['model'].get('hidden_dim', 512),
            output_dim=output_dim,
            freeze_gates=cfg['model'].get('freeze_gates', False),
            gate_temperature=cfg['model'].get('gate_temperature', 1.0)
        ).to(device)
    else:
        # Fall back to original create_model
        return create_model(cfg, device)


def visualize_gate_patterns(diagnostics, output_dir, epoch):
    """Visualize gate behavior patterns."""
    patterns = diagnostics.analyze_patterns()
    
    # Plot 1: Gate activation distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ESM vs Text gate means
    esm_gates = [p['esm_gate'] for p in diagnostics.protein_gate_patterns.values()]
    text_gates = [p['text_gate'] for p in diagnostics.protein_gate_patterns.values()]
    
    axes[0, 0].hist(esm_gates, bins=50, alpha=0.7, label='ESM Gates')
    axes[0, 0].hist(text_gates, bins=50, alpha=0.7, label='Text Gates')
    axes[0, 0].set_xlabel('Gate Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Gate Value Distribution')
    axes[0, 0].legend()
    
    # Gate difference distribution
    differences = [p['difference'] for p in diagnostics.protein_gate_patterns.values()]
    axes[0, 1].hist(differences, bins=50, color='purple', alpha=0.7)
    axes[0, 1].set_xlabel('ESM Gate - Text Gate')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Gate Difference Distribution')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    
    # GO term preferences
    if patterns['go_term_preferences']:
        go_terms = list(patterns['go_term_preferences'].keys())[:20]  # Top 20
        esm_prefs = [patterns['go_term_preferences'][go]['esm_preference'] for go in go_terms]
        text_prefs = [patterns['go_term_preferences'][go]['text_preference'] for go in go_terms]
        
        x = np.arange(len(go_terms))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, esm_prefs, width, label='ESM')
        axes[1, 0].bar(x + width/2, text_prefs, width, label='Text')
        axes[1, 0].set_xlabel('GO Terms')
        axes[1, 0].set_ylabel('Gate Activation')
        axes[1, 0].set_title('GO Term Modality Preferences')
        axes[1, 0].legend()
        axes[1, 0].set_xticks([])
    
    # Gate evolution over training
    if len(diagnostics.gate_history['esm_mean']) > 0:
        axes[1, 1].plot(diagnostics.gate_history['esm_mean'], label='ESM Mean')
        axes[1, 1].plot(diagnostics.gate_history['text_mean'], label='Text Mean')
        axes[1, 1].fill_between(
            range(len(diagnostics.gate_history['esm_mean'])),
            np.array(diagnostics.gate_history['esm_mean']) - np.array(diagnostics.gate_history['esm_std']),
            np.array(diagnostics.gate_history['esm_mean']) + np.array(diagnostics.gate_history['esm_std']),
            alpha=0.3
        )
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Gate Value')
        axes[1, 1].set_title('Gate Evolution During Training')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'gate_analysis_epoch_{epoch}.png', dpi=300)
    plt.close()
    
    # Save detailed patterns
    with open(output_dir / f'gate_patterns_epoch_{epoch}.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    return patterns


def train_with_diagnostics(config_path):
    """Enhanced training with full diagnostics."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(cfg['log']['out_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger.info(f"Starting diagnostic training: {cfg['experiment_name']}")
    
    # Load GO terms
    aspect = cfg['experiment_name'].split('_')[-1]
    data_dir = Path(cfg['dataset']['train_names']).parent
    with open(data_dir / f"{aspect}_go_terms.json", 'r') as f:
        go_terms = json.load(f)
    
    # Create datasets (same as before)
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'prott5': '/SAN/bioinf/PFP/embeddings/cafa3/prott5',
        'prostt5': '/SAN/bioinf/PFP/embeddings/cafa3/prostt5',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text'
    }
    
    train_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['train_names'],
        labels_file=cfg['dataset']['train_labels'],
        features=cfg['dataset']['features'],
        embeddings_dir=embeddings_dir
    )
    
    valid_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['valid_names'],
        labels_file=cfg['dataset']['valid_labels'],
        features=cfg['dataset']['features'],
        embeddings_dir=embeddings_dir
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset'].get('batch_size', 32), 
                            shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['dataset'].get('batch_size', 32), 
                            shuffle=False, collate_fn=collate_batch)
    
    # Create enhanced model
    model = create_diagnostic_model(cfg, device)
    
    # Initialize diagnostics
    diagnostics = GateDiagnostics()
    
    # Use staged training if configured
    if cfg.get('staged_training', False):
        # optimizer = staged_training(model, train_loader, valid_loader, cfg, device, diagnostics, go_terms)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], 
                                    weight_decay=cfg['optim']['weight_decay'])
    
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], 
                                    weight_decay=cfg['optim']['weight_decay'])
    
    criterion = torch.nn.BCEWithLogitsLoss()
    best_fmax = 0.0
    
    # Training loop with diagnostics
    for epoch in range(1, cfg['optim']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['optim']['epochs']}")
        
        # Train with diagnostics
        train_loss = train_epoch_with_diagnostics(
            model, train_loader, optimizer, criterion, device, cfg, epoch, diagnostics, go_terms
        )
        
        # Validate
        val_metrics = validate_with_gate_analysis(model, valid_loader, criterion, device, cfg)
        # Check if fusion is working
        if val_metrics['text_gate_mean'] < 0.1:
            print("WARNING: Text modality being ignored!")
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_metrics['loss']:.4f}, F-max: {val_metrics['Fmax_protein']:.4f}")
        logger.info(f"ESM Gate Mean: {val_metrics['esm_gate_mean']:.3f}, Text Gate Mean: {val_metrics['text_gate_mean']:.3f}")
        
        # Visualize patterns every 10 epochs
        if epoch % 10 == 0:
            patterns = visualize_gate_patterns(diagnostics, output_dir, epoch)
            
            # Log interesting patterns
            logger.info(f"Modality usage - ESM dominance: {patterns['global_stats']['esm_dominance']:.2%}")
            logger.info(f"Proteins with strong ESM preference: {len(patterns['extreme_cases']['strong_esm'])}")
            logger.info(f"Proteins with strong Text preference: {len(patterns['extreme_cases']['strong_text'])}")
            logger.info(f"Proteins with balanced gates: {len(patterns['extreme_cases']['balanced'])}")
        
        # Save checkpoint with diagnostics
        if val_metrics['Fmax_protein'] > best_fmax:
            best_fmax = val_metrics['Fmax_protein']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_fmax': best_fmax,
                'diagnostics': diagnostics.protein_gate_patterns,
                'config': cfg
            }, output_dir / 'best_model_with_diagnostics.pt')
    
    # Final analysis
    final_patterns = diagnostics.analyze_patterns()
    
    # Save comprehensive report
    with open(output_dir / 'final_diagnostic_report.json', 'w') as f:
        json.dump({
            'experiment': cfg['experiment_name'],
            'best_fmax': best_fmax,
            'gate_patterns': final_patterns,
            'modality_usage': {
                'esm_dominant_proteins': len(final_patterns['extreme_cases']['strong_esm']),
                'text_dominant_proteins': len(final_patterns['extreme_cases']['strong_text']),
                'balanced_proteins': len(final_patterns['extreme_cases']['balanced'])
            },
            'go_term_analysis': {
                go: pref for go, pref in list(final_patterns['go_term_preferences'].items())[:50]
            }
        }, f, indent=2)

    generate_predictions(model, cfg, device, output_dir)

    
    logger.info("Diagnostic training complete!")


def validate_with_gate_analysis(model, loader, criterion, device, cfg):
    """Validation with gate statistics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_esm_gates = []
    all_text_gates = []
    
    with torch.no_grad():
        for batch in loader:
            names, features_batch, labels = batch
            labels = labels.to(device)
            
            feat1, feat2 = cfg['dataset']['features']

            output = model(features_batch[feat1].to(device), features_batch[feat2].to(device))
            
            if isinstance(output, tuple):
                predictions, gate_stats = output
                all_esm_gates.append(gate_stats['esm_gate_mean'].cpu())
                all_text_gates.append(gate_stats['text_gate_mean'].cpu())
            else:
                predictions = output
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(predictions).cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = {
        'loss': total_loss / len(loader),
        'Fmax_protein': calculate_fmax(all_preds, all_labels),
        'macro_AP': calculate_map(all_preds, all_labels)
    }
    
    if all_esm_gates:
        all_esm_gates = torch.cat(all_esm_gates)
        all_text_gates = torch.cat(all_text_gates)
        metrics['esm_gate_mean'] = all_esm_gates.mean().item()
        metrics['text_gate_mean'] = all_text_gates.mean().item()
        metrics['gate_std'] = (all_esm_gates.std() + all_text_gates.std()).item() / 2
    
    return metrics
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    
    args = parser.parse_args()

    # train_cafa3_model(args.config)
    train_with_diagnostics(args.config)     
