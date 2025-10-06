#!/usr/bin/env python3
"""
ESM2 MLP Fine-tuning with Optional Graph Propagation
Add --use_graph flag to enable GO DAG propagation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# GO DAG Parser - NEW
# ============================================================================

class GODagParser:
    """Parse GO OBO file and build adjacency matrix"""
    
    def __init__(self, obo_path: str):
        self.obo_path = Path(obo_path)
        self.go_terms = {}  # id -> {name, namespace, parents}
        
    def parse(self):
        """Parse OBO file"""
        print(f"📖 Parsing GO OBO file: {self.obo_path}")
        
        with open(self.obo_path, 'r') as f:
            lines = f.readlines()
        
        current_term = None
        for line in lines:
            line = line.strip()
            
            if line == "[Term]":
                if current_term and 'id' in current_term:
                    self.go_terms[current_term['id']] = current_term
                current_term = {'parents': [], 'namespace': None}
                
            elif current_term is not None:
                if line.startswith('id: GO:'):
                    current_term['id'] = line.split('id: ')[1]
                elif line.startswith('name: '):
                    current_term['name'] = line.split('name: ')[1]
                elif line.startswith('namespace: '):
                    current_term['namespace'] = line.split('namespace: ')[1]
                elif line.startswith('is_a: GO:'):
                    parent = line.split('is_a: ')[1].split(' !')[0]
                    current_term['parents'].append(parent)
        
        if current_term and 'id' in current_term:
            self.go_terms[current_term['id']] = current_term
        
        print(f"  Found {len(self.go_terms)} GO terms")
        return self.go_terms
    
    def build_adjacency_matrix(self, go_list: List[str], normalize: bool = True):
        """
        Build adjacency matrix for specific GO terms
        A[i,j] = 1 if GO_i is parent of GO_j (child -> parent)
        """
        n = len(go_list)
        go_to_idx = {go: i for i, go in enumerate(go_list)}
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float32)
        
        for i, go_id in enumerate(go_list):
            if go_id in self.go_terms:
                parents = self.go_terms[go_id]['parents']
                for parent in parents:
                    if parent in go_to_idx:
                        j = go_to_idx[parent]
                        A[j, i] = 1.0  # Parent j <- Child i
        
        # Add self-loops
        A = A + np.eye(n, dtype=np.float32)
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        if normalize:
            rowsum = A.sum(axis=1)
            d_inv_sqrt = np.power(rowsum, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            D_inv_sqrt = np.diag(d_inv_sqrt)
            A = D_inv_sqrt @ A @ D_inv_sqrt
        
        print(f"  Adjacency matrix: {A.shape}, Edges: {(A > 0).sum()}, Density: {(A > 0).sum() / (n*n):.4f}")
        
        return torch.FloatTensor(A)

# ============================================================================
# Configuration - UPDATED
# ============================================================================

class Config:
    """Central configuration for CAFA3 experiments"""
    
    # Data paths
    CAFA3_BASE = Path("/home/zijianzhou/Datasets/cafa3")
    ESM_EMBEDDINGS_DIR = Path("/home/zijianzhou/Datasets/esm")
    GO_OBO_PATH = CAFA3_BASE / "go.obo"
    
    # Model settings
    ESM_DIM = 1280  # ESM2 650M embedding dimension
    
    # Training settings
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # MLP settings
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    
    # Graph settings - NEW
    USE_GRAPH = False
    GRAPH_HIDDEN_DIM = 256
    
    # Data settings
    MIN_GO_FREQUENCY = 0
    MAX_GO_FREQUENCY = 1
    
    # Debug settings
    DEBUG_MODE = False
    DEBUG_SAMPLES = 100
    DEBUG_GO_TERMS = 50
    
    # Output paths
    OUTPUT_DIR = Path("./cafa3_mlp_experiments")
    
    def __init__(self, go_aspect: str = "mf", debug_mode: bool = False, use_graph: bool = False):
        self.GO_ASPECT = go_aspect
        self.DEBUG_MODE = debug_mode
        self.USE_GRAPH = use_graph
        
        # Create output directories
        suffix = "_graph" if use_graph else "_baseline"
        aspect_dir = self.OUTPUT_DIR / f"{go_aspect}{suffix}"
        self.CHECKPOINT_DIR = aspect_dir / "checkpoints"
        self.RESULTS_DIR = aspect_dir / "results"
        
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        
        if debug_mode:
            self.NUM_EPOCHS = 2
            print(f"🔧 Debug mode enabled")
        
        if use_graph:
            print(f"🔗 Graph propagation ENABLED")

# ============================================================================
# Data Loader - SAME
# ============================================================================

class CAFA3DataLoader:
    """Load CAFA3 dataset with pre-computed ESM embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
        self.go_columns = []
        
    def load_aspect_data(self, aspect: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"📂 Loading {aspect.upper()} aspect data...")
        
        train_path = self.config.CAFA3_BASE / f"{aspect}-training.csv"
        val_path = self.config.CAFA3_BASE / f"{aspect}-validation.csv"
        test_path = self.config.CAFA3_BASE / f"{aspect}-test.csv"
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_data(self) -> Tuple:
        train_data, val_data, test_data = self.load_aspect_data(self.config.GO_ASPECT)
        
        # Get GO columns
        self.go_columns = [col for col in train_data.columns if col.startswith('GO:')]
        print(f"📊 Found {len(self.go_columns)} GO terms")
        
        # Filter GO terms by frequency
        go_frequencies = train_data[self.go_columns].mean()
        valid_go_terms = go_frequencies[
            (go_frequencies >= self.config.MIN_GO_FREQUENCY) & 
            (go_frequencies <= self.config.MAX_GO_FREQUENCY)
        ].index.tolist()
        
        print(f"📊 Filtered GO terms: {len(self.go_columns)} -> {len(valid_go_terms)}")
        self.go_columns = valid_go_terms
        
        # Debug mode
        if self.config.DEBUG_MODE:
            print(f"🔧 Debug mode: using {self.config.DEBUG_SAMPLES} samples")
            train_data = train_data.head(self.config.DEBUG_SAMPLES)
            val_data = val_data.head(min(self.config.DEBUG_SAMPLES // 5, len(val_data)))
            test_data = test_data.head(min(self.config.DEBUG_SAMPLES // 5, len(test_data)))
            
            if len(self.go_columns) > self.config.DEBUG_GO_TERMS:
                self.go_columns = self.go_columns[:self.config.DEBUG_GO_TERMS]
        
        # Extract data
        train_proteins = train_data['proteins'].tolist()
        val_proteins = val_data['proteins'].tolist()
        test_proteins = test_data['proteins'].tolist()
        
        train_labels = train_data[self.go_columns].values.astype(np.float32)
        val_labels = val_data[self.go_columns].values.astype(np.float32)
        test_labels = test_data[self.go_columns].values.astype(np.float32)
        
        print(f"\n📈 Dataset Statistics:")
        print(f"  GO Aspect: {self.config.GO_ASPECT.upper()}")
        print(f"  GO terms: {len(self.go_columns)}")
        print(f"  Train samples: {len(train_labels)}")
        print(f"  Val samples: {len(val_labels)}")
        print(f"  Test samples: {len(test_labels)}")
        print(f"  Avg GO terms per protein: {train_labels.sum(axis=1).mean():.2f}")
        print(f"  Label density: {train_labels.mean():.4f}")
        
        return (train_proteins, train_labels, 
                val_proteins, val_labels,
                test_proteins, test_labels, 
                self.go_columns)

# ============================================================================
# PyTorch Dataset - SAME
# ============================================================================

class CAFA3MLPDataset(Dataset):
    """Simple dataset - directly load embeddings"""
    
    def __init__(self, proteins: List[str], labels: np.ndarray, embeddings_dir: Path):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.embeddings_dir = embeddings_dir
        
        # Pre-load all embeddings into memory for speed
        print(f"📥 Pre-loading {len(proteins)} embeddings...")
        self.embeddings = []
        missing_count = 0
        
        for protein_id in tqdm(proteins, desc="Loading embeddings"):
            emb_path = self.embeddings_dir / f"{protein_id}.npy"
            
            if emb_path.exists():
                try:
                    embedding = np.load(emb_path, allow_pickle=True)
                    
                    # Convert to float32 array
                    if isinstance(embedding, np.ndarray) and embedding.dtype == object:
                        embedding = embedding.item()
                    
                    # Handle dict format
                    if isinstance(embedding, dict):
                        if 'mean' in embedding:
                            embedding = embedding['mean']
                        elif 'pooled' in embedding:
                            embedding = embedding['pooled']
                        elif 'embedding' in embedding:
                            embedding = embedding['embedding']
                            if len(embedding.shape) == 2:
                                embedding = embedding.mean(axis=0)
                    
                    # Convert to numpy array
                    embedding = np.asarray(embedding, dtype=np.float32)
                    
                    # Handle 2D arrays (take mean)
                    if len(embedding.shape) == 2:
                        embedding = embedding.mean(axis=0)
                    
                    # Ensure 1D
                    embedding = embedding.flatten()
                    
                    # Ensure correct dimension (1280)
                    if embedding.shape[0] != 1280:
                        if embedding.shape[0] > 1280:
                            embedding = embedding[:1280]
                        else:
                            embedding = np.pad(embedding, (0, 1280 - embedding.shape[0]))
                    
                    self.embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"Error loading {protein_id}: {e}")
                    self.embeddings.append(np.zeros(1280, dtype=np.float32))
                    missing_count += 1
            else:
                self.embeddings.append(np.zeros(1280, dtype=np.float32))
                missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count}/{len(proteins)} embeddings missing or failed to load")
        
        # Convert to tensor
        self.embeddings = torch.FloatTensor(np.array(self.embeddings))
        print(f"✅ Loaded embeddings shape: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ============================================================================
# Model - UPDATED WITH GRAPH PROPAGATION
# ============================================================================

class GraphPropagation(nn.Module):
    """Graph propagation layer: H' = Â·H·W_A"""
    
    def __init__(self, num_features: int, adjacency_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('A', adjacency_matrix)  # Normalized adjacency matrix
        self.W_A = nn.Linear(num_features, num_features, bias=False)  # Learnable transformation
        
    def forward(self, H):
        """
        H: [batch_size, num_go_terms] - latent features
        Returns: [batch_size, num_go_terms] - propagated features
        """
        # H' = Â·H·W_A
        H_transformed = self.W_A(H)  # [B, N] -> [B, N]
        H_propagated = H_transformed @ self.A.T  # [B, N] @ [N, N] = [B, N]
        return H_propagated


class ESM2MLPClassifier(nn.Module):
    """MLP classifier with optional graph propagation"""
    
    def __init__(self, config: Config, num_go_terms: int, adjacency_matrix=None):
        super().__init__()
        self.use_graph = config.USE_GRAPH
        
        # MLP backbone (produces latent features H)
        self.mlp = nn.Sequential(
            nn.Linear(config.ESM_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.HIDDEN_DIM, num_go_terms)
        
        # Graph propagation (optional)
        if self.use_graph:
            assert adjacency_matrix is not None, "Adjacency matrix required for graph propagation"
            self.graph_prop = GraphPropagation(num_go_terms, adjacency_matrix)
            print(f"  🔗 Graph propagation layer added")
        
    def forward(self, x):
        # MLP produces latent features
        features = self.mlp(x)  # [B, hidden_dim]
        H = self.output_proj(features)  # [B, num_go_terms]
        
        # Optional graph propagation
        if self.use_graph:
            H = self.graph_prop(H)  # H' = Â·H·W_A
        
        return H  # Return logits

# ============================================================================
# Trainer - SAME
# ============================================================================

class Trainer:
    """Simple full-precision trainer"""
    
    def __init__(self, model, config: Config, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Simple optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []
        
        mode = "Graph" if config.USE_GRAPH else "Baseline"
        print(f"🔧 Trainer: Device={self.device}, Mode={mode}")
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for embeddings, labels in tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(dataloader, desc="Evaluating"):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(embeddings)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.calculate_metrics(all_labels, all_preds)
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        from sklearn.metrics import average_precision_score
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum(axis=1)
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum(axis=1)
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum(axis=1)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        auprcs = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                try:
                    auprc = average_precision_score(y_true[:, i], y_pred[:, i])
                    auprcs.append(auprc)
                except:
                    pass
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'auprc': np.mean(auprcs) if auprcs else 0.0
        }
    
    def train(self, train_loader, val_loader, num_epochs=None):
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_f1 = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: P={val_metrics['precision']:.3f}, "
                  f"R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}, "
                  f"AUPRC={val_metrics['auprc']:.3f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                **val_metrics
            })
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                print(f"  ✅ New best model! F1: {best_val_f1:.3f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch + 1}")
                break
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics):
        suffix = "_graph" if self.config.USE_GRAPH else "_baseline"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        path = self.config.CHECKPOINT_DIR / f"best_model_{self.config.GO_ASPECT}{suffix}.pt"
        torch.save(checkpoint, path)

# ============================================================================
# Visualization - UPDATED
# ============================================================================

def visualize_results(history: List[Dict], config: Config):
    """Create training visualization"""
    df = pd.DataFrame(history)
    
    mode = "Graph" if config.USE_GRAPH else "Baseline"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'CAFA3 MLP Training - {config.GO_ASPECT.upper()} ({mode})', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 score
    axes[0, 1].plot(df['epoch'], df['f1'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1')
    axes[0, 1].grid(True, alpha=0.3)
    
    # All metrics
    axes[1, 0].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['f1'], label='F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('All Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(df['epoch'], df['lr'], linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = "_graph" if config.USE_GRAPH else "_baseline"
    save_path = config.RESULTS_DIR / f'training_curves_{config.GO_ASPECT}{suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Plots saved to {save_path}")

# ============================================================================
# Main - UPDATED
# ============================================================================

def main(go_aspect: str = "mf", debug_mode: bool = False, use_graph: bool = False):
    """Main training pipeline"""
    
    mode = "Graph Propagation" if use_graph else "Baseline MLP"
    print(f"🚀 CAFA3 ESM2 Training - {mode}")
    print("=" * 60)
    
    config = Config(go_aspect=go_aspect, debug_mode=debug_mode, use_graph=use_graph)
    
    # Load data
    data_loader = CAFA3DataLoader(config)
    (train_proteins, train_labels, 
     val_proteins, val_labels,
     test_proteins, test_labels, 
     go_terms) = data_loader.prepare_data()
    
    # Build adjacency matrix if using graph
    adjacency_matrix = None
    if use_graph:
        go_parser = GODagParser(config.GO_OBO_PATH)
        go_parser.parse()
        adjacency_matrix = go_parser.build_adjacency_matrix(go_terms, normalize=True)
    
    # Create datasets (pre-loads embeddings)
    train_dataset = CAFA3MLPDataset(train_proteins, train_labels, config.ESM_EMBEDDINGS_DIR)
    val_dataset = CAFA3MLPDataset(val_proteins, val_labels, config.ESM_EMBEDDINGS_DIR)
    test_dataset = CAFA3MLPDataset(test_proteins, test_labels, config.ESM_EMBEDDINGS_DIR)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = ESM2MLPClassifier(config, num_go_terms=len(go_terms), adjacency_matrix=adjacency_matrix)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {total_params:,} parameters")
    
    # Train
    trainer = Trainer(model, config)
    print("\n🏋️ Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save results
    suffix = "_graph" if use_graph else "_baseline"
    pd.DataFrame(history).to_csv(
        config.RESULTS_DIR / f'training_history_{go_aspect}{suffix}.csv', index=False
    )
    
    visualize_results(history, config)
    
    # Test
    print("\n📊 Final test evaluation...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Metrics: P={test_metrics['precision']:.3f}, "
          f"R={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}, "
          f"AUPRC={test_metrics['auprc']:.3f}")
    
    # Save final results
    results = {
        'go_aspect': go_aspect,
        'num_go_terms': len(go_terms),
        'use_graph': use_graph,
        'test_metrics': test_metrics,
        'best_val_f1': max([h['f1'] for h in history]),
        'num_epochs': len(history)
    }
    
    with open(config.RESULTS_DIR / f'final_results_{go_aspect}{suffix}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Complete! Results: {config.RESULTS_DIR}")
    
    return model, history, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", type=str, default="mf", choices=["mf", "bp", "cc"])
    parser.add_argument("--debug", action="store_true", help="Debug mode with small data")
    parser.add_argument("--use_graph", action="store_true", help="Enable graph propagation")
    args = parser.parse_args()
    
    main(go_aspect=args.aspect, debug_mode=args.debug, use_graph=args.use_graph)