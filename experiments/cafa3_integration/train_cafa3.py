#!/usr/bin/env python3
"""
Training script for CAFA3 experiments with complete multi-modal support
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/train_cafa3.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
import logging
from typing import Dict, Optional, List, Any
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as ssp
from tqdm.auto import tqdm
import json
from fusion_models import (
    GatedMultimodalFusion,
    AdaptiveMoEFusion,
    MultimodalTransformerFusion,
    ContrastiveMultimodalFusion,
    ImprovedGatedFusion
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add project root
sys.path.append('/SAN/bioinf/PFP/PFP')

# Import from multimodal comparison
from experiments.multimodal_comparison.train_unified import (
    MultiModalFusionModel, EnhancedMetricTracker,
    collate_multimodal, train_epoch, validate
)

# Import structure components
from structure.pdb_graph_utils import StructureGraphDataset
from structure.egnn_model import collate_graph_batch

# Import base components
from Network.base_go_classifier import BaseGOClassifier
from Network.model_utils import EarlyStop


class CAFA3MultiModalDataset(Dataset):
    """Unified dataset for CAFA3 supporting all modalities."""
    
    def __init__(self,
                 names_file: str,
                 labels_file: str,
                 sequences_file: str,
                 features: List[str],
                 embeddings_dir: Dict[str, str],
                 graph_config: Optional[Dict] = None,
                 use_cache: bool = True):
        
        self.names = np.load(names_file, allow_pickle=True)
        self.labels = torch.from_numpy(ssp.load_npz(labels_file).toarray()).float()
        self.features = features
        self.use_cache = use_cache
        self.graph_config = graph_config
        
        # Load sequences
        with open(sequences_file, 'r') as f:
            self.sequences = json.load(f)
            
        # Set embedding directories
        self.esm_dir = Path(embeddings_dir.get('esm', ''))
        self.text_dir = Path(embeddings_dir.get('text', ''))
        self.struct_dir = Path(embeddings_dir.get('structure', ''))
        
        # Initialize structure dataset if needed
        if 'structure' in features and graph_config:
            self.struct_dataset = StructureGraphDataset(
                pdb_dir=embeddings_dir.get('structure', ''),
                esm_embedding_dir=str(self.esm_dir),
                names_npy=names_file,
                labels_npy=labels_file,
                **graph_config
            )
            self.struct_idx_map = {name: i for i, name in enumerate(self.struct_dataset.valid_names)}
        
        # Cache
        self._cache = {} if use_cache else None
        
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]
        
        # Check cache
        if self.use_cache and idx in self._cache:
            return name, self._cache[idx], label
            
        features_dict = {}
        
        # Load ESM embeddings
        if 'esm' in self.features:
            esm_file = self.esm_dir / f"{name}.npy"
            if esm_file.exists():
                data = np.load(esm_file, allow_pickle=True).item()
                emb = data['embedding']
                if emb.ndim == 2:
                    emb = emb.mean(axis=0)
                features_dict['esm'] = torch.from_numpy(emb).float()
            else:
                # Use zeros as fallback
                features_dict['esm'] = torch.zeros(1280)
                
        # Load text embeddings
        if 'text' in self.features:
            text_file = self.text_dir / f"{name}.npy"
            if text_file.exists():
                data = np.load(text_file, allow_pickle=True).item()
                emb = data.get('embedding', data)
                if isinstance(emb, np.ndarray) and emb.ndim == 2:
                    emb = emb.mean(axis=0)

                features_dict['text'] = torch.from_numpy(emb).float()
            else:
                exit(name)
                features_dict['text'] = torch.zeros(768)
                
        # Load structure data
        if 'structure' in self.features:
            if name in self.struct_idx_map:
                struct_idx = self.struct_idx_map[name]
                _, graph_data = self.struct_dataset[struct_idx]
                features_dict['structure'] = graph_data
            else:
                features_dict['structure'] = None
                
        # Cache if enabled
        if self.use_cache:

            self._cache[idx] = features_dict
            
        return name, features_dict, label


def create_cafa3_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Create appropriate model based on configuration."""
    
    features = cfg['dataset']['features']
    output_dim = cfg['model']['output_dim']
    
    if len(features) == 1:
        # Single modality model
        feature = features[0]
        if feature == 'esm':
            input_dim = 1280
        elif feature == 'text':
            input_dim = 768
        elif feature == 'structure':
            # Use structure-specific model
            from structure.egnn_model import StructureGOClassifier
            
            egnn_config = {
                'input_dim': 1280 if cfg.get('graph', {}).get('use_esm_features', True) else 20,
                'hidden_dim': 256,
                'output_dim': 512,
                'n_layers': 4,
                'edge_dim': 4,
                'dropout': 0.3,
                'update_pos': False,
                'pool': 'mean'
            }
            
            classifier_config = {
                'input_dim': 512,
                'output_dim': output_dim,
                'hidden_dim': 512,
                'projection_dim': 512
            }
            
            return StructureGOClassifier(
                egnn_config=egnn_config,
                classifier_config=classifier_config,
                use_mmstie_fusion=False
            ).to(device)
        else:
            raise ValueError(f"Unknown feature: {feature}")
            
        # Standard classifier for ESM/text
        return BaseGOClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            projection_dim=1024,
            hidden_dim=512
        ).to(device)
        
    else:
        # Multi-modal model
        return MultiModalFusionModel(
            features=features,
            fusion_method=cfg['model'].get('fusion_method', 'concat'),
            output_dim=output_dim,
            graph_config=cfg.get('graph', None),
            device=device
        ).to(device)



def create_model_from_config(cfg, device):
    """Create model based on configuration."""
    features = cfg['dataset']['features']
    output_dim = cfg['model']['output_dim']
    
    if len(features) == 1:
        # Single modality
        feature = features[0]
        input_dim = 1280 if feature == 'esm' else 768
        
        return BaseGOClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            projection_dim=1024,
            hidden_dim=512
        ).to(device)
    
    elif 'esm' in features and 'text' in features and len(features) == 2:
        # ESM + Text fusion
        fusion_type = cfg['model'].get('fusion_type', 'concat')
        
        if fusion_type == 'concat':
            # Your existing MultiModalFusionModel
            return MultiModalFusionModel(
                features=features,
                fusion_method='concat',
                output_dim=output_dim,
                device=device
            ).to(device)
            
        elif fusion_type == 'gated':
            return GatedMultimodalFusion(
                esm_dim=1280,
                text_dim=768,
                hidden_dim=cfg['model'].get('hidden_dim', 512),
                output_dim=output_dim
            ).to(device)
            
        elif fusion_type == 'moe':
            return AdaptiveMoEFusion(
                esm_dim=1280,
                text_dim=768,
                hidden_dim=cfg['model'].get('hidden_dim', 512),
                output_dim=output_dim,
                num_experts_per_modality=cfg['model'].get('num_experts', 3)
            ).to(device)
            
        elif fusion_type == 'transformer':
            return MultimodalTransformerFusion(
                esm_dim=1280,
                text_dim=768,
                hidden_dim=cfg['model'].get('hidden_dim', 512),
                output_dim=output_dim,
                num_layers=cfg['model'].get('num_layers', 4),
                num_heads=cfg['model'].get('num_heads', 8)
            ).to(device)
            
        elif fusion_type == 'contrastive':
            return ContrastiveMultimodalFusion(
                esm_dim=1280,
                text_dim=768,
                hidden_dim=cfg['model'].get('hidden_dim', 512),
                output_dim=output_dim,
                temperature=cfg['model'].get('temperature', 0.07)
            ).to(device)
    
    else:
        # Full multimodal or other configurations
        return MultiModalFusionModel(
            features=features,
            fusion_method=cfg['model'].get('fusion_method', 'concat'),
            output_dim=output_dim,
            graph_config=cfg.get('graph', None),
            device=device
        ).to(device)


# 4. Update the training loop to handle auxiliary outputs:

def train_epoch_with_fusion(model, train_loader, optimizer, criterion, 
                           device, cfg, epoch):
    """Modified training loop for fusion models."""
    model.train()
    total_loss = 0
    aux_losses = {}
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        names, features, labels = batch
        labels = labels.to(device)
        
        # Prepare features based on modalities
        if len(cfg['dataset']['features']) == 1:
            # Single modality
            feat_name = cfg['dataset']['features'][0]
            features_input = features[feat_name].to(device)
            predictions = model(features_input)
            aux_outputs = {}
        else:
            # Multi-modal
            esm_features = features.get('esm', None)
            text_features = features.get('text', None)
            
            if esm_features is not None and text_features is not None:
                esm_features = esm_features.to(device)
                text_features = text_features.to(device)
                
                # Get predictions and auxiliary outputs
                if hasattr(model, 'forward') and model.forward.__code__.co_argcount > 3:
                    # Model supports auxiliary outputs
                    predictions, aux_outputs = model(esm_features, text_features)
                else:
                    predictions = model({'esm': esm_features, 'text': text_features})
                    aux_outputs = {}
            else:
                # Fallback for other configurations
                for feat_name in features:
                    if features[feat_name] is not None:
                        features[feat_name] = features[feat_name].to(device)
                predictions = model(features)
                aux_outputs = {}
        
        # Compute main loss
        main_loss = criterion(predictions, labels)
        total_loss_batch = main_loss
        
        # Add auxiliary losses if available
        if 'contrastive_loss' in aux_outputs and aux_outputs['contrastive_loss'] is not None:
            contrastive_weight = cfg['model'].get('contrastive_weight', 0.5)
            total_loss_batch += contrastive_weight * aux_outputs['contrastive_loss']
            
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # Gradient clipping
        if cfg['optim'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=cfg['optim']['gradient_clip']
            )
        
        optimizer.step()
        
        # Warmup learning rate
        if cfg['optim'].get('warmup_steps', 0) > 0:
            current_step = epoch * len(train_loader) + batch_idx
            if current_step < cfg['optim']['warmup_steps']:
                lr_scale = current_step / cfg['optim']['warmup_steps']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg['optim']['lr'] * lr_scale
        
        total_loss += total_loss_batch.item()
        
        # Track auxiliary losses
        for key, value in aux_outputs.items():
            if 'loss' in key and isinstance(value, torch.Tensor):
                if key not in aux_losses:
                    aux_losses[key] = []
                aux_losses[key].append(value.item())
    
    # Log auxiliary information
    if aux_losses:
        logger.info("Auxiliary losses:")
        for key, values in aux_losses.items():
            logger.info(f"  {key}: {np.mean(values):.4f}")
    
    return total_loss / len(train_loader)

def train_cafa3_model(config_path: str):
    """Train model on CAFA3 dataset."""
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger(__name__)
    
    # Create output directory
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
    
    logger.info(f"Starting CAFA3 training: {cfg['experiment_name']}")
    logger.info(f"Features: {cfg['dataset']['features']}")
    logger.info(f"Device: {device}")
    
    # Extract aspect from config
    aspect = cfg['experiment_name'].split('_')[-1]
    data_dir = Path(cfg['dataset']['train_names']).parent
    
    # Embedding directories
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
        'structure': '/SAN/bioinf/PFP/embeddings/cafa3/structures'
    }
    
    # Determine if we need structure-specific dataset
    features = cfg['dataset']['features']
    use_structure = 'structure' in features
    
    if use_structure and len(features) == 1:
        # Structure-only: use StructureGraphDataset directly
        graph_config = cfg.get('graph', {})
        
        train_dataset = StructureGraphDataset(
            pdb_dir=embeddings_dir['structure'],
            esm_embedding_dir=embeddings_dir['esm'],
            names_npy=cfg['dataset']['train_names'],
            labels_npy=cfg['dataset']['train_labels'],
            **graph_config
        )
        
        valid_dataset = StructureGraphDataset(
            pdb_dir=embeddings_dir['structure'],
            esm_embedding_dir=embeddings_dir['esm'],
            names_npy=cfg['dataset']['valid_names'],
            labels_npy=cfg['dataset']['valid_labels'],
            **graph_config
        )
        
        collate_fn = collate_graph_batch
        
    else:
        # Single modality (ESM/text) or multi-modal: use unified dataset
        train_dataset = CAFA3MultiModalDataset(
            names_file=cfg['dataset']['train_names'],
            labels_file=cfg['dataset']['train_labels'],
            sequences_file=str(data_dir / f"{aspect}_train_sequences.json"),
            features=features,
            embeddings_dir=embeddings_dir,
            graph_config=cfg.get('graph', None)
        )
        
        valid_dataset = CAFA3MultiModalDataset(
            names_file=cfg['dataset']['valid_names'],
            labels_file=cfg['dataset']['valid_labels'],
            sequences_file=str(data_dir / f"{aspect}_valid_sequences.json"),
            features=features,
            embeddings_dir=embeddings_dir,
            graph_config=cfg.get('graph', None)
        )
        
        collate_fn = collate_multimodal if len(features) > 1 else None
    
    # Create model
    model = create_cafa3_model(cfg, device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=True,
        # num_workers=cfg['dataset'].get('num_workers', 0),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=False,
        # num_workers=cfg['dataset'].get('num_workers', 0),
        collate_fn=collate_fn
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['optim']['lr'],
        weight_decay=cfg['optim']['weight_decay']
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    metric_tracker = EnhancedMetricTracker(device)
    
    # Early stopping
    early_stop = EarlyStop(
        patience=cfg['optim']['patience'],
        min_epochs=cfg['optim'].get('min_epochs', 10)
    )
    
    # Determine experiment type for train/validate functions
    if len(features) == 1 and features[0] != 'structure':
        experiment_type = 'baseline'
    elif len(features) == 1 and features[0] == 'structure':
        experiment_type = 'structure'
    else:
        experiment_type = 'multimodal'
    
    # Training loop
    best_fmax = 0.0
    history = []
    
    for epoch in range(1, cfg['optim']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['optim']['epochs']}")
        
        # Train
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            metric_tracker, device, experiment_type
        )
        
        # Validate
        val_metrics = validate(
            model, valid_loader, criterion,
            metric_tracker, device, experiment_type
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Valid F-max: {val_metrics.get('Fmax_protein', 0):.4f}")
        logger.info(f"Valid mAP: {val_metrics.get('macro_AP', 0):.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Early stopping
        current_fmax = val_metrics.get('Fmax_protein', 0)
        early_stop(-current_fmax, current_fmax, model)
        
        if current_fmax > best_fmax:
            best_fmax = current_fmax
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_fmax': best_fmax,
                'config': cfg
            }, output_dir / 'best_model.pt')
        
        if early_stop.stop():
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save final results
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({
            'best_Fmax_protein': best_fmax,
            'final_epoch': epoch,
            'experiment': cfg['experiment_name'],
            'features': features
        }, f, indent=2)
    
    # Save training history
    import pandas as pd
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    
    logger.info(f"Training complete! Best F-max: {best_fmax:.4f}")
    
    # Generate predictions for test set
    generate_test_predictions(model, cfg, device, output_dir, experiment_type)


def generate_test_predictions(model, cfg, device, output_dir, experiment_type):
    """Generate predictions on test set for CAFA evaluation."""
    
    logger = logging.getLogger(__name__)
    logger.info("Generating test predictions...")
    
    # Load test dataset
    aspect = cfg['experiment_name'].split('_')[-1]
    data_dir = Path(cfg['dataset']['train_names']).parent
    features = cfg['dataset']['features']
    
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
        'structure': '/SAN/bioinf/PFP/embeddings/cafa3/structures'
    }
    
    # Create test dataset
    if experiment_type == 'structure' and len(features) == 1:
        test_dataset = StructureGraphDataset(
            pdb_dir=embeddings_dir['structure'],
            esm_embedding_dir=embeddings_dir['esm'],
            names_npy=str(data_dir / f"{aspect}_test_names.npy"),
            labels_npy=str(data_dir / f"{aspect}_test_labels.npz"),
            **cfg.get('graph', {})
        )
        collate_fn = collate_graph_batch
    else:
        test_dataset = CAFA3MultiModalDataset(
            names_file=str(data_dir / f"{aspect}_test_names.npy"),
            labels_file=str(data_dir / f"{aspect}_test_labels.npz"),
            sequences_file=str(data_dir / f"{aspect}_test_sequences.json"),
            features=features,
            embeddings_dir=embeddings_dir,
            graph_config=cfg.get('graph', None)
        )
        collate_fn = collate_multimodal if len(features) > 1 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=False,
        # num_workers=0,
        collate_fn=collate_fn
    )
    
    # Generate predictions
    model.eval()
    all_predictions = []
    all_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            if len(batch) == 3:
                names, features, _ = batch
            else:
                names, features = batch
                
            # Handle different input types
            if experiment_type == 'baseline':
                # Single modality (ESM/text)
                feat_name = cfg['dataset']['features'][0]
                features = features[feat_name].to(device)


                logits = model(features)
            elif experiment_type == 'structure':
                # Structure only
                for key in features:
                    if isinstance(features[key], torch.Tensor):
                        features[key] = features[key].to(device)
                logits = model(features)
            else:
                # Multi-modal
                for feat_name in features:
                    if feat_name == 'structure' and features[feat_name] is not None:
                        for key in features[feat_name]:
                            if isinstance(features[feat_name][key], torch.Tensor):
                                features[feat_name][key] = features[feat_name][key].to(device)
                    elif features[feat_name] is not None:
                        features[feat_name] = features[feat_name].to(device)
                logits = model(features)
            
            predictions = torch.sigmoid(logits).cpu().numpy()
            all_predictions.append(predictions)
            all_names.extend(names)
    
    # Concatenate predictions
    all_predictions = np.vstack(all_predictions)
    
    # Load GO terms
    with open(data_dir / f"{aspect}_go_terms.json", 'r') as f:
        go_terms = json.load(f)
    
    # Save predictions in CAFA format
    pred_dir = output_dir.parent.parent / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    model_name = cfg['experiment_name'].replace('CAFA3_', '')
    pred_file = pred_dir / f"{model_name}.tsv"
    
    with open(pred_file, 'w') as f:
        for i, protein_id in enumerate(all_names):
            for j, go_term in enumerate(go_terms):
                score = all_predictions[i, j]
                if score > 0.01:  # Threshold for saving
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")
    
    logger.info(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    
    args = parser.parse_args()
    
    train_cafa3_model(args.config)