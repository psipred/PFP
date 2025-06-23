#!/usr/bin/env python3
"""
Training script for CAFA3 experiments
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/train_cafa3.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as ssp
from tqdm.auto import tqdm

# Add project root
sys.path.append('/SAN/bioinf/PFP/PFP')

# Import from existing codebase
from experiments.multimodal_comparison.train_unified import (
    MultiModalDataset, MultiModalFusionModel, EnhancedMetricTracker,
    collate_multimodal, train_epoch, validate
)
from Network.base_go_classifier import BaseGOClassifier
from Network.model_utils import EarlyStop


class CAFA3Dataset(Dataset):
    """Dataset for CAFA3 data with memory caching."""
    
    def __init__(self,
                 names_file: str,
                 labels_file: str,
                 sequences_file: str,
                 embeddings_dir: Dict[str, str],
                 features: list[str] = ['esm'],
                 use_cache: bool = True,
                 preload_cache: bool = False):
        
        self.names = np.load(names_file, allow_pickle=True)
        self.labels = torch.from_numpy(ssp.load_npz(labels_file).toarray()).float()
        self.features = features
        self.use_cache = use_cache
        
        # Initialize cache
        self._cache = {} if use_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load sequences
        import json
        with open(sequences_file, 'r') as f:
            self.sequences = json.load(f)
            
        # Set embedding directories
        self.esm_dir = Path(embeddings_dir.get('esm', ''))
        self.text_dir = Path(embeddings_dir.get('text', ''))
        self.struct_dir = Path(embeddings_dir.get('structure', ''))
        
        # Optionally preload all data into cache
        if preload_cache and use_cache:
            self._preload_cache()
        
    def _preload_cache(self):
        """Preload all data into memory cache."""
        for idx in range(len(self.names)):
            if idx not in self._cache:
                self._load_sample(idx)
                
    def _load_sample(self, idx):
        """Load and cache a single sample."""
        name = self.names[idx]
        features_dict = {}
        
        # Load ESM embeddings
        if 'esm' in self.features:
            esm_file = self.esm_dir / f"{name}.npy"
            if esm_file.exists():
                data = np.load(esm_file, allow_pickle=True).item()
                emb = data['embedding']
                if emb.ndim == 2:  # per-residue → pool
                    emb = emb.mean(axis=0)
                    features_dict['esm'] = torch.from_numpy(emb).float()
                else:
                    features_dict['esm'] = torch.from_numpy(emb).float()
            else:
                exit(f"ESM embedding not found for {name}: {esm_file}")
                
        # Add other modalities as needed
        if 'text' in self.features:
            features_dict['text'] = torch.zeros(768)
            
        if 'structure' in self.features:
            features_dict['structure'] = None
            
        # Cache the features
        if self.use_cache:
            self._cache[idx] = features_dict
            
        return features_dict
        
    def __len__(self):
        return len(self.names)
        
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]
        
        # Check cache first
        if self.use_cache and idx in self._cache:
            features_dict = self._cache[idx]
            self._cache_hits += 1
        else:
            features_dict = self._load_sample(idx)
            self._cache_misses += 1
            
        return name, features_dict, label
    
    def get_cache_stats(self):
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses, 
            'hit_rate': hit_rate,
            'cached_items': len(self._cache) if self._cache else 0
        }

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
    
    # Create datasets
    embeddings_dir = {
        'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
        'structure': '/SAN/bioinf/PFP/embeddings/cafa3/structures'
    }
    
    # Extract aspect from config
    aspect = cfg['experiment_name'].split('_')[-1]  # BPO, CCO, or MFO
    data_dir = Path(cfg['dataset']['train_names']).parent
    
    train_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['train_names'],
        labels_file=cfg['dataset']['train_labels'],
        sequences_file=str(data_dir / f"{aspect}_train_sequences.json"),
        embeddings_dir=embeddings_dir,
        features=cfg['dataset']['features']
    )
    
    valid_dataset = CAFA3Dataset(
        names_file=cfg['dataset']['valid_names'],
        labels_file=cfg['dataset']['valid_labels'],
        sequences_file=str(data_dir / f"{aspect}_valid_sequences.json"),
        embeddings_dir=embeddings_dir,
        features=cfg['dataset']['features']
    )
    
    # Create model
    if len(cfg['dataset']['features']) == 1:
        # Single modality
        feature = cfg['dataset']['features'][0]
        input_dim = 1280 if feature == 'esm' else 768
        
        model = BaseGOClassifier(
            input_dim=input_dim,
            output_dim=cfg['model']['output_dim'],
            projection_dim=1024,
            hidden_dim=512
        ).to(device)
    else:
        # Multi-modal
        model = MultiModalFusionModel(
            features=cfg['dataset']['features'],
            fusion_method=cfg['model'].get('fusion_method', 'concat'),
            output_dim=cfg['model']['output_dim'],
            graph_config=cfg.get('graph', None),
            device=device
        ).to(device)
        
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=True,

        collate_fn=collate_multimodal if len(cfg['dataset']['features']) > 1 else None
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=False,

        collate_fn=collate_multimodal if len(cfg['dataset']['features']) > 1 else None
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
    
    # Training loop
    best_fmax = 0.0

    for epoch in range(1, cfg['optim']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['optim']['epochs']}")
        
        # Train
        if len(cfg['dataset']['features']) == 1:
            # Simplified training for single modality
            train_loss = 0.0
            model.train()
            
            for batch in tqdm(train_loader,
                        desc=f"Epoch {epoch:03d} · train",
                        unit="batch", leave=False):
                names, features, labels = batch
                labels = labels.to(device)
                
                # Extract single feature
                feat_name = cfg['dataset']['features'][0]
                features = features[feat_name].to(device)
                
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            train_loss /= len(train_loader)
        else:
            # Multi-modal training
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                metric_tracker, device, 'multimodal'
            )
        
        # Validate
        if len(cfg['dataset']['features']) == 1:
            # Simplified validation
            model.eval()
            val_loss = 0.0
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(valid_loader,
                            desc=f"Epoch {epoch:03d} · valid",
                            unit="batch", leave=False):
                    names, features, labels = batch
                    labels = labels.to(device)
                    
                    feat_name = cfg['dataset']['features'][0]
                    features = features[feat_name].to(device)
                    
                    logits = model(features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
            
            val_loss /= len(valid_loader)
            
            # Compute metrics
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            val_metrics = metric_tracker.compute_metrics(all_logits, all_labels)
            val_metrics['loss'] = val_loss
        else:
            # Multi-modal validation
            val_metrics = validate(
                model, valid_loader, criterion,
                metric_tracker, device, 'multimodal'
            )
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Valid F-max: {val_metrics.get('Fmax_protein', 0):.4f}")
        logger.info(f"Valid mAP: {val_metrics.get('macro_AP', 0):.4f}")
        
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

    # Save final results (moved outside the loop)
    import json
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({
            'best_Fmax_protein': best_fmax,
            'final_epoch': epoch,
            'experiment': cfg['experiment_name']
        }, f, indent=2)

    logger.info(f"Training complete! Best F-max: {best_fmax:.4f}")

    # Generate predictions for test set
    generate_test_predictions(model, cfg, device, output_dir)


def generate_test_predictions(model, cfg, device, output_dir):
        """Generate predictions on test set for CAFA evaluation."""
        
        logger = logging.getLogger(__name__)
        logger.info("Generating test predictions...")
        
        # Load test dataset
        aspect = cfg['experiment_name'].split('_')[-1]
        data_dir = Path(cfg['dataset']['train_names']).parent
        
        embeddings_dir = {
            'esm': '/SAN/bioinf/PFP/embeddings/cafa3/esm',
            'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
            'structure': '/SAN/bioinf/PFP/embeddings/cafa3/structures'
        }
        
        test_dataset = CAFA3Dataset(
            names_file=str(data_dir / f"{aspect}_test_names.npy"),
            labels_file=str(data_dir / f"{aspect}_test_labels.npz"),
            sequences_file=str(data_dir / f"{aspect}_test_sequences.json"),
            embeddings_dir=embeddings_dir,
            features=cfg['dataset']['features']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg['dataset'].get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_multimodal if len(cfg['dataset']['features']) > 1 else None
        )
        
        # Generate predictions
        model.eval()
        all_predictions = []
        all_names = []
        
        with torch.no_grad():
            for batch in test_loader:
                names, features, _ = batch
                
                if len(cfg['dataset']['features']) == 1:
                    feat_name = cfg['dataset']['features'][0]
                    features = features[feat_name].to(device)
                    logits = model(features)
                else:
                    # Multi-modal
                    for feat_name in features:
                        if feat_name != 'structure' and features[feat_name] is not None:
                            features[feat_name] = features[feat_name].to(device)
                    logits = model(features)
                
                predictions = torch.sigmoid(logits).cpu().numpy()
                all_predictions.append(predictions)
                all_names.extend(names)
        
        # Concatenate all predictions
        all_predictions = np.vstack(all_predictions)
        
        # Load GO terms
        import json
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