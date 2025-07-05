#!/usr/bin/env python3
"""
Unified training script for multi-modal GO prediction experiments
Location: /SAN/bioinf/PFP/PFP/experiments/multimodal_comparison/train_unified.py
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Add project paths
sys.path.append('/SAN/bioinf/PFP/PFP')

# Import required modules
from Network.base_go_classifier import BaseGOClassifier
from Network.model import InterlabelGODataset
from Network.model_utils import EarlyStop
from structure.pdb_graph_utils import StructureGraphDataset
from structure.egnn_model import StructureGOClassifier, collate_graph_batch, EGNN
from metrics import MetricBundle


class MultiModalDataset(Dataset):
    """Dataset that combines multiple modalities."""
    
    def __init__(self, 
                 names_npy: str,
                 labels_npy: str,
                 features: List[str],
                 base_dir: str = "/SAN/bioinf/PFP",
                 graph_config: Optional[Dict] = None):
        
        self.names = np.load(names_npy)
        self.features = features
        self.base_dir = Path(base_dir)
        self.graph_config = graph_config
        
        # Load labels
        if labels_npy.endswith('.npz'):
            import scipy.sparse as ssp
            self.labels = torch.from_numpy(ssp.load_npz(labels_npy).toarray()).float()
        else:
            self.labels = torch.from_numpy(np.load(labels_npy)).float()
        
        # Setup paths
        self.embeddings_base = Path("/SAN/bioinf/PFP/embeddings/cafa5_small")
        self.esm_dir = self.embeddings_base / "esm_af"
        self.text_dir = self.embeddings_base / "prot2text" / "text_embeddings"
        
        # Initialize structure dataset if needed
        if 'structure' in features and graph_config:
            self.struct_dataset = StructureGraphDataset(
                pdb_dir="/SAN/bioinf/PFP/embeddings/structure/pdb_files",
                esm_embedding_dir=str(self.esm_dir),
                names_npy=names_npy,
                labels_npy=labels_npy,
                **graph_config
            )
            # Map indices for structure dataset
            self.struct_idx_map = {name: i for i, name in enumerate(self.struct_dataset.valid_names)}
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]
        
        features_dict = {}
        
        # Load ESM features
        if 'esm' in self.features:
            esm_path = self.esm_dir / f"{name}.npy"
            if esm_path.exists():
                data = np.load(esm_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.dtype == object:
                    data = data.item()
                
                if isinstance(data, dict) and 'embedding' in data:
                    embedding = data['embedding']
                    if embedding.ndim == 2:  # (L, D)
                        embedding = embedding.mean(axis=0)  # Average pooling
                else:
                    embedding = data
                
                features_dict['esm'] = torch.from_numpy(embedding.astype(np.float32))
            else:
                # Return zeros if file not found
                # exit("esm zeros")
                features_dict['esm'] = torch.zeros(1280)
        
        # Load text features
        if 'text' in self.features:
            text_path = self.text_dir / f"{name}.npy"
            if text_path.exists():
                data = np.load(text_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.dtype == object:
                    data = data.item()

                
                if isinstance(data, dict) and 'embedding' in data:
                    embedding = data['embedding']
                    if embedding.ndim == 2:
                        embedding = embedding.mean(axis=0)
                else:
                    embedding = data
                
                features_dict['text'] = torch.from_numpy(embedding.astype(np.float32))
            else:
                exit("text zeros")
                features_dict['text'] = torch.zeros(768)  # Default text embedding size
        
        # Load structure features
        if 'structure' in self.features:
            if name in self.struct_idx_map:
                struct_idx = self.struct_idx_map[name]
                _, graph_data = self.struct_dataset[struct_idx]
                features_dict['structure'] = graph_data
            else:
                # Create dummy graph data
                features_dict['structure'] = None
        
        return name, features_dict, label


class MultiModalFusionModel(torch.nn.Module):
    """Model that fuses multiple modalities."""
    
    def __init__(self, 
                 features: List[str],
                 fusion_method: str,
                 output_dim: int,
                 graph_config: Optional[Dict] = None,
                 device: str = 'cuda'):
        
        super().__init__()
        self.features = features
        self.fusion_method = fusion_method
        self.device = device
        
        # Feature dimensions
        self.feature_dims = {
            'esm': 1280,
            'prostt5': 1024,
            'prott5': 1024,
            'text': 768,
            'structure': 512  # After EGNN encoding
        }
        
        # Initialize encoders
        if 'structure' in features:
            egnn_config = {
                'input_dim': 1280 if graph_config.get('use_esm_features', True) else 20,
                'hidden_dim': 256,
                'output_dim': 512,
                'n_layers': 4,
                'edge_dim': 4,
                'dropout': 0.3,
                'update_pos': False,
                'pool': 'mean'
            }
            self.structure_encoder = EGNN(**egnn_config)
        
        # Calculate fusion dimension
        self.fusion_dim = sum(self.feature_dims[f] for f in features)
        
        # Initialize fusion layer
        if fusion_method == 'concat':
            self.fusion = None  # Simple concatenation
        elif fusion_method == 'attention':
            self.fusion = MultiHeadAttentionFusion(
                feature_dims=[self.feature_dims[f] for f in features],
                output_dim=1024
            )
            self.fusion_dim = 1024
        elif fusion_method == 'mmstie':
            # Use the pretrained MMStie fusion
            from Network.dnn import AP_align_fuse
            self.fusion = AP_align_fuse(tau=0.8, hidden_size=256)
            self.fusion_dim = 2048
            
            # Load pretrained weights
            ckpt_path = "/SAN/bioinf/PFP/pretrained/best_model_fuse_0.8322829131652661.pt"
            if os.path.exists(ckpt_path):
                old_weights = torch.load(ckpt_path, map_location=device)
                old_weights.pop("classifier_token.weight", None)
                old_weights.pop("classifier_token.bias", None)
                self.fusion.load_state_dict(old_weights, strict=False)
        
        # Initialize classifier
        self.classifier = BaseGOClassifier(
            input_dim=self.fusion_dim,
            output_dim=output_dim,
            projection_dim=1024,
            hidden_dim=512
        )
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model."""
        encoded_features = []
        
        # Encode each modality
        for feat_name in self.features:
            if feat_name in features_dict and features_dict[feat_name] is not None:
                if feat_name == 'structure':
                    # Handle graph batch
                    graph_data = features_dict[feat_name]
                    if graph_data is not None:
                        encoded = self.structure_encoder(graph_data)
                    else:
                        # Use zeros if structure not available
                        batch_size = features_dict.get('esm', features_dict.get('text')).shape[0]
                        encoded = torch.zeros(batch_size, self.feature_dims['structure']).to(self.device)
                else:
                    encoded = features_dict[feat_name]
                
                encoded_features.append(encoded)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat(encoded_features, dim=-1)
        elif self.fusion_method == 'attention':
            fused = self.fusion(encoded_features)
        elif self.fusion_method == 'mmstie' and len(encoded_features) == 2:
            # MMStie expects text and sequence embeddings
            text_emb, seq_emb = encoded_features[0], encoded_features[1]
            outputs = self.fusion(text_emb, seq_emb)
            fused = outputs["token_embeddings"]
        else:
            fused = torch.cat(encoded_features, dim=-1)
        
        # Classify
        logits = self.classifier(fused)
        return logits


class MultiHeadAttentionFusion(torch.nn.Module):
    """Multi-head attention based fusion."""
    
    def __init__(self, feature_dims: List[int], output_dim: int, n_heads: int = 8):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Project each feature to common dimension
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Multi-head attention
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(output_dim * len(feature_dims), output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using attention."""
        # Project features
        projected = []
        for feat, proj in zip(features, self.projections):
            projected.append(proj(feat))
        
        # Stack for attention
        stacked = torch.stack(projected, dim=1)  # (B, n_features, D)
        
        # Apply attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Flatten and project
        flattened = attended.reshape(attended.shape[0], -1)
        output = self.output_proj(flattened)
        
        return output


class EnhancedMetricTracker:
    """Track and compute enhanced metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics = MetricBundle(device)
        self.metrics_cpu = MetricBundle('cpu')
        
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        # Move to CPU for some metrics
        logits_cpu = logits.cpu()
        labels_cpu = labels.cpu()
        
        # Basic metrics
        base_metrics = self.metrics_cpu(logits_cpu, labels_cpu)
        
        # Additional metrics
        probs = torch.sigmoid(logits_cpu)
        probs_np = probs.numpy()
        labels_np = labels_cpu.numpy()
        
        # Macro and Micro AP
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        ap_scores = []
        auroc_scores = []
        
        for i in range(labels.shape[1]):
            if labels_np[:, i].sum() > 0:  # Only for terms with positive samples
                ap = average_precision_score(labels_np[:, i], probs_np[:, i])
                ap_scores.append(ap)
                
                if len(np.unique(labels_np[:, i])) == 2:
                    auroc = roc_auc_score(labels_np[:, i], probs_np[:, i])
                    auroc_scores.append(auroc)
        
        metrics = {
            **base_metrics,
            'macro_AP': np.mean(ap_scores) if ap_scores else 0.0,
            'micro_AP': average_precision_score(labels_np.ravel(), probs_np.ravel()),
            'macro_AUROC': np.mean(auroc_scores) if auroc_scores else 0.0,
            'Fmax_protein': self._compute_protein_centric_fmax(probs, labels_cpu)
        }
        
        # Coverage metrics
        pred_binary = (probs > 0.5).float()
        coverages = []
        for i in range(labels.shape[0]):
            true_terms = labels_cpu[i].sum()
            if true_terms > 0:
                predicted_true = (pred_binary[i] * labels_cpu[i]).sum()
                coverages.append((predicted_true / true_terms).item())
        metrics['coverage'] = np.mean(coverages) if coverages else 0.0
        
        return metrics
    
    def _compute_protein_centric_fmax(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute protein-centric F-max."""
        thresholds = torch.linspace(0, 1, 51)
        f1_scores = []
        
        for thr in thresholds:
            pred = (probs > thr).float()
            
            # Compute F1 for each protein
            protein_f1s = []
            for i in range(probs.shape[0]):
                tp = (pred[i] * labels[i]).sum()
                fp = (pred[i] * (1 - labels[i])).sum()
                fn = ((1 - pred[i]) * labels[i]).sum()
                
                prec = tp / (tp + fp + 1e-9)
                rec = tp / (tp + fn + 1e-9)
                f1 = 2 * prec * rec / (prec + rec + 1e-9)
                protein_f1s.append(f1.item())
            
            f1_scores.append(np.mean(protein_f1s))
        
        return max(f1_scores)


def collate_multimodal(batch):
    """Custom collate function for multi-modal data."""
    names = [item[0] for item in batch]
    features_list = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    
    # Collate each feature type
    collated_features = {}
    feature_names = features_list[0].keys()
    
    for feat_name in feature_names:
        if feat_name == 'structure':
            # Handle graph data separately
            graph_data_list = [f[feat_name] for f in features_list if f[feat_name] is not None]
            if graph_data_list:
                # Create list of (name, graph_data) tuples for collate_graph_batch
                graph_batch = [(n, g) for n, g, f in zip(names, graph_data_list, features_list) 
                               if f[feat_name] is not None]
                
                # collate_graph_batch returns 3 values when labels are present
                collate_result = collate_graph_batch(graph_batch)
                
                if len(collate_result) == 3:
                    # With labels: (names, batch_data, labels)
                    _, collated_graph, _ = collate_result
                else:
                    # Without labels: (names, batch_data)
                    _, collated_graph = collate_result
                
                collated_features[feat_name] = collated_graph
            else:
                collated_features[feat_name] = None
        else:
            # Stack regular features
            feat_tensors = [f[feat_name] for f in features_list]
            collated_features[feat_name] = torch.stack(feat_tensors)
    
    return names, collated_features, labels


def train_epoch(model, train_loader, optimizer, criterion, metric_tracker, device, experiment_type):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        names, features, labels = batch
        labels = labels.to(device)
        # features = features.to(device)

        # Move features to device
        for feat_name in features:
            if feat_name == 'structure' and features[feat_name] is not None:
                # Move graph data
                for key in features[feat_name]:
                    if isinstance(features[feat_name][key], torch.Tensor):
                        features[feat_name][key] = features[feat_name][key].to(device)
            elif feat_name == 'baseline':
                # Handle baseline features

                features = features.to(device)

            # elif feat_name == 'text':
            #     features = features[feat_name].to(device)


            elif features is not None:

                # print(features)
                # features = features[feat_name].to(device)
                # exit(features)
                features[feat_name] = features[feat_name].to(device)



        # Forward pass
        optimizer.zero_grad()
        
        if experiment_type == 'baseline':
            # Single modality
            # feat_name = list(features.keys())[0]
            # logits = model(features[feat_name])
            # exit(features)

            features = features[feat_name].to(device)

            logits = model(features)
        else:
            # Multi-modal or structure

            logits = model(features)
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / num_batches


def validate(model, valid_loader, criterion, metric_tracker, device, experiment_type):
    """Validate the model."""
    model.eval()
    epoch_loss = 0.0
    all_logits = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            names, features, labels = batch
            labels = labels.to(device)
            # features = features.to(device)
            
            # Move features to device
            for feat_name in features:
                if feat_name == 'structure' and features[feat_name] is not None:
                    for key in features[feat_name]:
                        if isinstance(features[feat_name][key], torch.Tensor):
                            features[feat_name][key] = features[feat_name][key].to(device)
                elif feat_name == 'baseline':
                    # Handle baseline features
                    features = features.to(device)
                elif features is not None:
                    features[feat_name] = features[feat_name].to(device)
            
            # Forward pass
            if experiment_type == 'baseline':
                # Single‑modality baseline uses one tensor
                feat_name = list(features.keys())[0]
                logits = model(features[feat_name].to(device))
            else:
                # ----- Dual‑stream aware forward -----
                if (
                    len(features) == 2
                    and hasattr(model, "forward")
                    and model.forward.__code__.co_argcount > 3
                ):
                    # Keep key order deterministic (Python ≥3.7 preserves insertion order)
                    feat1, feat2 = list(features)
                    input1 = features[feat1].to(device)
                    input2 = features[feat2].to(device)

                    # Some fusion models return (logits, aux_outputs)
                    out = model(input1, input2)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    # Standard single‑dict models
                    logits = model(features)
                # -------------------------------------
            
            loss = criterion(logits, labels)
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = metric_tracker.compute_metrics(all_logits, all_labels)
    metrics['loss'] = epoch_loss / num_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train multi-modal GO prediction model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--experiment-type', type=str, required=True,
                       choices=['baseline', 'structure', 'multimodal'],
                       help="Type of experiment")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))
    
    # Setup logging
    log_dir = Path(cfg['log']['out_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {cfg['experiment_name']}")
    logger.info(f"Experiment type: {args.experiment_type}")
    logger.info(f"Device: {device}")
    
    # Create datasets
    if args.experiment_type == 'baseline':
        # Use the original dataset class
        embedding_type = cfg['dataset']['embedding_type']
        
        train_dataset = InterlabelGODataset(
            features_dir=cfg.get('data_dir', '/SAN/bioinf/PFP/PFP/Data'),
            embedding_type=embedding_type,
            names_npy=cfg['dataset']['train_names'],
            labels_npy=cfg['dataset']['train_labels']
        )
        
        valid_dataset = InterlabelGODataset(
            features_dir=cfg.get('data_dir', '/SAN/bioinf/PFP/PFP/Data'),
            embedding_type=embedding_type,
            names_npy=cfg['dataset']['valid_names'],
            labels_npy=cfg['dataset']['valid_labels']
        )
        
        # Create model
        model = BaseGOClassifier(
            input_dim=1280 if 'esm' in embedding_type else 768,
            output_dim=cfg['model']['output_dim'],
            projection_dim=1024,
            hidden_dim=512
        ).to(device)
        
        collate_fn = None  # Use default
        
    elif args.experiment_type == 'structure':
        # Use structure dataset
        graph_config = cfg['graph']
        
        train_dataset = StructureGraphDataset(
            pdb_dir="/SAN/bioinf/PFP/embeddings/structure/pdb_files",
            esm_embedding_dir="/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af",
            names_npy=cfg['dataset']['train_names'],
            labels_npy=cfg['dataset']['train_labels'],
            **graph_config
        )
        
        
        valid_dataset = StructureGraphDataset(
            pdb_dir="/SAN/bioinf/PFP/embeddings/structure/pdb_files",
            esm_embedding_dir="/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af",
            names_npy=cfg['dataset']['valid_names'],
            labels_npy=cfg['dataset']['valid_labels'],
            **graph_config
        )
        
        # Create structure model
        egnn_config = {
            'input_dim': 1280 if graph_config.get('use_esm_features', True) else 20,
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
            'output_dim': cfg['model']['output_dim'],
            'hidden_dim': 512,
            'projection_dim': 512
        }
        
        model = StructureGOClassifier(
            egnn_config=egnn_config,
            classifier_config=classifier_config,
            use_mmstie_fusion=False
        ).to(device)
        
        collate_fn = collate_graph_batch
        
    else:  # multimodal
        # Use multi-modal dataset
        features = cfg['dataset']['features']
        graph_config = cfg.get('graph', None)
        
        train_dataset = MultiModalDataset(
            names_npy=cfg['dataset']['train_names'],
            labels_npy=cfg['dataset']['train_labels'],
            features=features,
            graph_config=graph_config
        )
        
        valid_dataset = MultiModalDataset(
            names_npy=cfg['dataset']['valid_names'],
            labels_npy=cfg['dataset']['valid_labels'],
            features=features,
            graph_config=graph_config
        )
        
        # Create multi-modal model
        model = MultiModalFusionModel(
            features=features,
            fusion_method=cfg['model']['fusion_method'],
            output_dim=cfg['model']['output_dim'],
            graph_config=graph_config,
            device=device
        ).to(device)
        
        collate_fn = collate_multimodal
    
    # Log dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=True,
        num_workers=cfg['dataset'].get('num_workers', 4),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg['dataset'].get('batch_size', 32),
        shuffle=False,
        num_workers=cfg['dataset'].get('num_workers', 4),
        collate_fn=collate_fn
    )
    
    # Setup training
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
        min_epochs=cfg['optim'].get('min_epochs', 10),
        monitor=cfg['optim'].get('monitor', 'loss')
    )
    
    # Training history
    history = []
    best_metrics = None
    start_time = time.time()
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(1, cfg['optim']['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['optim']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            metric_tracker, device, args.experiment_type
        )
        
        # Validate
        val_metrics = validate(
            model, valid_loader, criterion,
            metric_tracker, device, args.experiment_type
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Valid F-max: {val_metrics.get('Fmax_protein', 0):.4f}")
        logger.info(f"Valid mAP: {val_metrics.get('macro_AP', 0):.4f}")
        
        # Update history
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'valid_{k}': v for k, v in val_metrics.items()}
        }
        history.append(epoch_metrics)
        
        # Early stopping
        monitor_metric = val_metrics.get(cfg['optim']['monitor'].replace('valid_', ''), val_metrics['loss'])
        if 'fmax' in cfg['optim']['monitor'].lower():
            monitor_metric = -monitor_metric  # Maximize
        
        early_stop(monitor_metric, val_metrics.get('Fmax_protein', 0), model)
        
        # Update best metrics
        if best_metrics is None or monitor_metric < best_metrics['monitor_value']:
            best_metrics = {
                'monitor_value': monitor_metric,
                'epoch': epoch,
                **{f'best_{k}': v for k, v in val_metrics.items()}
            }
        
        if early_stop.stop():
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint
        if epoch % cfg['log'].get('save_every', 10) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': cfg
            }
            torch.save(checkpoint, log_dir / f'checkpoint_epoch{epoch}.pt')
    
    # Training complete
    training_time = time.time() - start_time
    
    # Restore best model
    if early_stop.has_backup_model():
        model = early_stop.restore(model)
    
    # Save final model and results
    final_model = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'best_metrics': best_metrics,
        'training_time': training_time
    }
    torch.save(final_model, log_dir / 'final_model.pt')
    
    # Save metrics
    with open(log_dir / 'final_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / 'training_history.csv', index=False)
    
    # Plot training curves if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(history_df['epoch'], history_df['train_loss'], label='Train')
        ax1.plot(history_df['epoch'], history_df['valid_loss'], label='Valid')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics plot
        ax2.plot(history_df['epoch'], history_df['valid_Fmax_protein'], label='F-max')
        ax2.plot(history_df['epoch'], history_df['valid_macro_AP'], label='mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(log_dir / 'training_curves.png', dpi=300)
        plt.close()
    except:
        pass
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Total time: {training_time/3600:.2f} hours")
    logger.info(f"Best F-max: {best_metrics.get('best_Fmax_protein', 0):.4f} at epoch {best_metrics['epoch']}")
    logger.info(f"Results saved to: {log_dir}")


if __name__ == "__main__":
    main()