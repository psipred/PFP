import hydra
import torch
import numpy as np
import random
import os
import logging
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from pdb_graph_utils import StructureGraphDataset
from egnn_model import StructureGOClassifier, collate_graph_batch
import sys
sys.path.append('/SAN/bioinf/PFP/PFP')        

from metrics import MetricBundle
from Network.model_utils import EarlyStop

# Optional imports for visualization
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAVE_VIZ = True
except ImportError:
    HAVE_VIZ = False


def setup_logger(log_dir: str):
    """Setup logging configuration."""
    logger = logging.getLogger("structure_trainer")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "train_structure.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


@hydra.main(version_base=None, config_path="configs", config_name="structure_config")
def main(cfg: DictConfig):
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logger
    os.makedirs(cfg.log.out_dir, exist_ok=True)
    logger = setup_logger(cfg.log.out_dir)
    
    logger.info("="*60)
    logger.info("Structure-Based GO Prediction Training")
    logger.info("="*60)
    
    # Create datasets
    logger.info("Loading structure graph datasets...")
    
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
    
    # Log dataset summary
    train_summary = train_dataset.get_summary()
    valid_summary = valid_dataset.get_summary()
    
    logger.info("\nDataset Summary:")
    logger.info(f"Training set:")
    for key, value in train_summary.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Validation set:")
    logger.info(f"  proteins_with_pdb: {valid_summary['proteins_with_pdb']}")

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

    # Initialize model
    logger.info("\nInitializing EGNN model...")
    
    # EGNN configuration
    egnn_config = {
        'input_dim': 1280 if cfg.graph.use_esm_features else 20,  # ESM or one-hot
        'hidden_dim': cfg.model.hidden_dim,
        'output_dim': cfg.model.embedding_dim,
        'n_layers': cfg.model.n_layers,
        'edge_dim': 4,  # distance + 3D direction
        'dropout': cfg.model.dropout,
        'update_pos': cfg.model.get('update_pos', False),
        'pool': cfg.model.get('pool', 'mean')
    }
    
    # Classifier configuration
    classifier_config = {
        'input_dim': cfg.model.embedding_dim,
        'output_dim': cfg.model.output_dim,
        'hidden_dim': cfg.model.classifier_hidden_dim,
        'projection_dim': cfg.model.get('projection_dim', cfg.model.embedding_dim),
        # 'n_layers': cfg.model.get('classifier_layers', 2), 
        # TypeError: BaseGOClassifier.__init__() got an unexpected keyword argument 'n_layers'
        # 'dropout': cfg.model.dropout,
        # 'activation': cfg.model.get('activation', 'relu')
    }
    
    model = StructureGOClassifier(
        egnn_config=egnn_config,
        classifier_config=classifier_config,
        use_mmstie_fusion=cfg.model.get('use_mmstie_fusion', False)
    ).to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("\nModel Configuration:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  EGNN layers: {cfg.model.n_layers}")
    logger.info(f"  Hidden dim: {cfg.model.hidden_dim}")
    logger.info(f"  Graph type: {cfg.graph.type}")
    if cfg.graph.type == 'knn':
        logger.info(f"  k neighbors: {cfg.graph.k}")
    else:
        logger.info(f"  Radius: {cfg.graph.radius} Å")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    if cfg.optim.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=cfg.optim.get('scheduler_patience', 5),
            verbose=True
        )
    else:
        scheduler = None
    
    # Metrics
    metrics = MetricBundle(device)
    metrics_cpu = MetricBundle("cpu")
    
    # Early stopping
    early_stop = EarlyStop(
        patience=cfg.optim.patience,
        min_epochs=cfg.optim.get('min_epochs', 10),
        monitor=cfg.optim.get('monitor', 'loss')
    )
    
    # Training history
    history = []
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Epochs: {cfg.optim.epochs}")
    logger.info(f"Batch size: {cfg.dataset.batch_size}")
    logger.info(f"Learning rate: {cfg.optim.lr}")
    
    for epoch in range(1, cfg.optim.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]")
        for batch in progress_bar:
            if len(batch) == 3:
                names, graph_data, labels = batch
                labels = labels.to(device).float()
            else:
                names, graph_data = batch
                labels = None
                
            # Move graph data to device
            for key in graph_data:
                if isinstance(graph_data[key], torch.Tensor):
                    graph_data[key] = graph_data[key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(graph_data)
            
            if labels is not None:
                loss = criterion(logits, labels)
                loss.backward()
                
                # Gradient clipping
                if cfg.optim.get('clip_grad', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation phase
        if epoch % cfg.log.eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch:03d} [Valid]"):
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
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Collect predictions
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            # Compute metrics
            if all_logits:
                all_logits = torch.cat(all_logits, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                metrics_report = metrics_cpu(all_logits, all_labels)
            else:
                metrics_report = {'AP': 0, 'Fmax': 0}
            
            # Log results
            logger.info(f"\nEpoch {epoch:03d} Summary:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Valid Loss: {avg_val_loss:.4f}")
            logger.info(f"  mAP: {metrics_report['AP']:.4f}")
            logger.info(f"  Fmax: {metrics_report['Fmax']:.4f}")
            
            # Save history
            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'mAP': metrics_report['AP'],
                'Fmax': metrics_report['Fmax']
            })
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step(avg_val_loss)
            
            # Early stopping
            monitor_value = avg_val_loss if cfg.optim.monitor == 'loss' else -metrics_report['Fmax']
            early_stop(monitor_value, metrics_report['Fmax'], model)
            
            if early_stop.stop():
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    # Save final model
    if early_stop.has_backup_model():
        model = early_stop.restore(model)
    
    final_model_path = Path(cfg.log.out_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'egnn_config': egnn_config,
        'classifier_config': classifier_config,
        'graph_config': {
            'type': cfg.graph.type,
            'k': cfg.graph.k,
            'radius': cfg.graph.radius,
            'use_esm_features': cfg.graph.use_esm_features
        }
    }, final_model_path)
    
    logger.info(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    if HAVE_VIZ and history:
        # Save CSV
        df = pd.DataFrame(history)
        csv_path = Path(cfg.log.out_dir) / "training_history.csv"
        df.to_csv(csv_path, index=False)
        
        # Plot curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Valid Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Metrics plot
        ax2.plot(df['epoch'], df['mAP'], label='mAP')
        ax2.plot(df['epoch'], df['Fmax'], label='Fmax')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Metrics')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = Path(cfg.log.out_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"Training curves saved to: {plot_path}")
    
    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()