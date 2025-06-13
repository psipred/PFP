#!/usr/bin/env python3
"""
Consolidated enhanced training script with comprehensive metrics for structure-based GO prediction.
Location: /SAN/bioinf/PFP/PFP/structure/experiment_pipeline/train_with_metrics.py
"""

import hydra
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
import sys

sys.path.append('/SAN/bioinf/PFP/PFP')

from structure.pdb_graph_utils import StructureGraphDataset
from structure.egnn_model import StructureGOClassifier, collate_graph_batch
from metrics import MetricBundle as BaseMetricBundle
from Network.model_utils import EarlyStop


class EnhancedMetricBundle(BaseMetricBundle):
    """Extended metrics for comprehensive evaluation."""
    
    def __init__(self, device):
        super().__init__(device)
        
    def __call__(self, logits, labels, return_detailed=False):
        # Get base metrics
        base_metrics = super().__call__(logits, labels)
        
        # Convert to numpy for sklearn metrics
        probs = torch.sigmoid(logits)
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Additional metrics
        enhanced_metrics = base_metrics.copy()
        
        # 1. Macro and Micro AP
        ap_scores = [] # collector for per-term AP values
        for i in range(labels.shape[1]): # iterate over every GO term/column
            if labels_np[:, i].sum() > 0:  # Only for terms with positive samples
                ap = average_precision_score(labels_np[:, i], probs_np[:, i])
                ap_scores.append(ap)
        
        enhanced_metrics['macro_AP'] = np.mean(ap_scores) if ap_scores else 0.0
        enhanced_metrics['micro_AP'] = average_precision_score(labels_np.ravel(), probs_np.ravel())
        
        # 2. Protein-centric F-max
        enhanced_metrics['Fmax_protein'] = self._compute_fmax_protein_centric(probs, labels)
        
        # 3. AUROC
        try:
            auroc_scores = []
            for i in range(labels.shape[1]):
                if len(np.unique(labels_np[:, i])) == 2:
                    auroc = roc_auc_score(labels_np[:, i], probs_np[:, i])
                    auroc_scores.append(auroc)
            enhanced_metrics['macro_AUROC'] = np.mean(auroc_scores) if auroc_scores else 0.0
        except:
            enhanced_metrics['macro_AUROC'] = 0.0
        
        # 4. Coverage
        pred_binary = (probs > 0.5).float()
        coverages = []
        for i in range(labels.shape[0]):
            true_terms = labels[i].sum()
            if true_terms > 0:
                predicted_true = (pred_binary[i] * labels[i]).sum()
                coverages.append((predicted_true / true_terms).item())
        enhanced_metrics['coverage'] = np.mean(coverages) if coverages else 0.0
        
        # 5. Top-k metrics
        for k in [5, 10]:
            enhanced_metrics[f'precision@{k}'] = self._precision_at_k(probs, labels, k)
            enhanced_metrics[f'recall@{k}'] = self._recall_at_k(probs, labels, k)
        
        return enhanced_metrics
    
    def _compute_fmax_protein_centric(self, probs, labels):
        """Compute F-max with protein-centric evaluation."""
        thresholds = torch.linspace(0, 1, 51, device=self.device)
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
    
    def _precision_at_k(self, probs, labels, k):
        """Compute precision@k."""
        precisions = []
        for i in range(probs.shape[0]):
            top_k_idx = torch.topk(probs[i], min(k, probs.shape[1]))[1]
            correct = labels[i, top_k_idx].sum().item()
            precisions.append(correct / min(k, probs.shape[1]))
        return np.mean(precisions)
    
    def _recall_at_k(self, probs, labels, k):
        """Compute recall@k."""
        recalls = []
        for i in range(probs.shape[0]):
            n_true = labels[i].sum().item()
            if n_true > 0:
                top_k_idx = torch.topk(probs[i], min(k, probs.shape[1]))[1]
                covered = labels[i, top_k_idx].sum().item()
                recalls.append(covered / n_true)
        return np.mean(recalls) if recalls else 0.0


def setup_experiment(cfg: DictConfig):
    """Setup experiment directories and logging."""
    exp_dir = Path(cfg.log.out_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "results" / "predictions").mkdir(exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger("structure_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(exp_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, exp_dir


def train_epoch(model, train_loader, optimizer, criterion, metrics, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_metrics = []
    
    for batch in tqdm(train_loader, desc="Training"):
        if len(batch) == 3:
            names, graph_data, labels = batch
            labels = labels.to(device).float()
        else:
            continue
        
        # Move graph data to device
        for key in graph_data:
            if isinstance(graph_data[key], torch.Tensor):
                graph_data[key] = graph_data[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(graph_data)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = metrics(logits, labels)
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    avg_metrics['loss'] = epoch_loss / len(train_loader)
    
    return avg_metrics


def validate(model, valid_loader, criterion, metrics, device, save_predictions=False, save_path=None):
    """Validate model."""
    model.eval()
    epoch_loss = 0.0
    all_logits = []
    all_labels = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation"):
            if len(batch) == 3:
                names, graph_data, labels = batch
                labels = labels.to(device).float()
            else:
                continue
            
            # Move graph data to device
            for key in graph_data:
                if isinstance(graph_data[key], torch.Tensor):
                    graph_data[key] = graph_data[key].to(device)
            
            # Forward pass
            logits = model(graph_data)
            loss = criterion(logits, labels)
            
            epoch_loss += loss.item()
            
            # Collect predictions
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)
    
    # Compute metrics
    if all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        epoch_metrics = metrics(all_logits, all_labels)
        epoch_metrics['loss'] = epoch_loss / len(valid_loader)
        
        # Save predictions if requested
        if save_predictions and save_path:
            np.savez_compressed(
                save_path,
                names=all_names,
                logits=all_logits.numpy(),
                labels=all_labels.numpy(),
                probs=torch.sigmoid(all_logits).numpy()
            )
    else:
        epoch_metrics = {'loss': float('inf')}
    
    return epoch_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="structure_config")
def main(cfg: DictConfig):
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup experiment
    logger, exp_dir = setup_experiment(cfg)
    
    logger.info("="*60)
    logger.info("Structure-Based GO Prediction with Enhanced Metrics")
    logger.info("="*60)
    logger.info(f"Experiment: {cfg.aspect}_{cfg.graph.type}_k{cfg.graph.k}")
    logger.info(f"Device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = StructureGraphDataset(
        pdb_dir=cfg.data.pdb_dir,
        esm_embedding_dir=cfg.data.esm_embedding_dir,
        names_npy=cfg.dataset.train_names,
        labels_npy=cfg.dataset.train_labels,
        graph_type=cfg.graph.type,
        k=cfg.graph.k,
        radius=cfg.graph.radius,
        use_esm_node_features=cfg.graph.use_esm_features,
        cache_graphs=cfg.graph.cache_graphs
    )
    
    valid_dataset = StructureGraphDataset(
        pdb_dir=cfg.data.pdb_dir,
        esm_embedding_dir=cfg.data.esm_embedding_dir,
        names_npy=cfg.dataset.valid_names,
        labels_npy=cfg.dataset.valid_labels,
        graph_type=cfg.graph.type,
        k=cfg.graph.k,
        radius=cfg.graph.radius,
        use_esm_node_features=cfg.graph.use_esm_features,
        cache_graphs=cfg.graph.cache_graphs
    )
    
    logger.info(f"Training proteins with PDB: {len(train_dataset)}")
    logger.info(f"Validation proteins with PDB: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_graph_batch,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_graph_batch
    )
    
    # Create model
    logger.info("Initializing model...")
    egnn_config = {
        'input_dim': 1280 if cfg.graph.use_esm_features else 20,
        'hidden_dim': cfg.model.hidden_dim,
        'output_dim': cfg.model.embedding_dim,
        'n_layers': cfg.model.n_layers,
        'edge_dim': 4,
        'dropout': cfg.model.dropout,
        'update_pos': cfg.model.get('update_pos', False),
        'pool': cfg.model.get('pool', 'mean')
    }
    
    classifier_config = {
        'input_dim': cfg.model.embedding_dim,
        'output_dim': cfg.model.output_dim,
        'hidden_dim': cfg.model.classifier_hidden_dim,
        'projection_dim': cfg.model.get('projection_dim', cfg.model.embedding_dim),
    }
    
    model = StructureGOClassifier(
        egnn_config=egnn_config,
        classifier_config=classifier_config,
        use_mmstie_fusion=cfg.model.get('use_mmstie_fusion', False)
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Metrics
    metrics = EnhancedMetricBundle(device)
    metrics_cpu = EnhancedMetricBundle("cpu")
    
    # Early stopping
    early_stop = EarlyStop(
        patience=cfg.optim.patience,
        min_epochs=cfg.optim.get('min_epochs', 10),
        monitor=cfg.optim.get('monitor', 'loss')
    )
    
    # Training history
    history = []
    best_metrics = None
    start_time = time.time()
    
    # Training loop
    logger.info(f"\nStarting training for {cfg.optim.epochs} epochs...")
    
    for epoch in range(1, cfg.optim.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.optim.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, metrics, device)
        
        # Validate
        save_preds = (epoch % cfg.log.save_every == 0)
        pred_path = exp_dir / "results" / "predictions" / f"predictions_epoch{epoch:03d}.npz" if save_preds else None
        
        valid_metrics = validate(model, valid_loader, criterion, metrics_cpu, device, 
                               save_predictions=save_preds, save_path=pred_path)
        
        # Log results
        epoch_results = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'valid_{k}': v for k, v in valid_metrics.items()}
        }
        history.append(epoch_results)
        
        # Log key metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train F-max: {train_metrics.get('Fmax_protein', 0):.4f}")
        logger.info(f"Valid Loss: {valid_metrics['loss']:.4f}, "
                   f"Valid F-max: {valid_metrics.get('Fmax_protein', 0):.4f}, "
                   f"Valid mAP: {valid_metrics.get('macro_AP', 0):.4f}")
        
        # Early stopping
        monitor_metric = valid_metrics.get(cfg.optim.monitor.replace('valid_', ''), valid_metrics['loss'])
        if 'fmax' in cfg.optim.monitor.lower():
            monitor_metric = -monitor_metric  # Maximize F-max
        
        early_stop(monitor_metric, valid_metrics.get('Fmax_protein', 0), model)
        
        # Update best metrics
        if best_metrics is None or monitor_metric < best_metrics['monitor_value']:
            best_metrics = {
                'monitor_value': monitor_metric,
                'epoch': epoch,
                **valid_metrics
            }
        
        if early_stop.stop():
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint
        if epoch % cfg.log.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }
            torch.save(checkpoint, exp_dir / "checkpoints" / f"checkpoint_epoch{epoch:03d}.pt")
    
    # Training complete
    if early_stop.has_backup_model():
        model = early_stop.restore(model)
    
    # Save final model and results
    training_time = time.time() - start_time
    
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'best_metrics': best_metrics,
        'training_time': training_time
    }
    torch.save(final_model_data, exp_dir / "final_model.pt")
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "results" / "training_history.csv", index=False)
    
    # Save summary
    summary = {
        'experiment_name': f"{cfg.aspect}_{cfg.graph.type}_k{cfg.graph.k}",
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_time': training_time,
        'best_metrics': best_metrics,
        'final_epoch': len(history)
    }
    
    with open(exp_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Total time: {training_time/3600:.2f} hours")
    logger.info(f"Best F-max: {best_metrics.get('Fmax_protein', 0):.4f} at epoch {best_metrics['epoch']}")
    logger.info(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()