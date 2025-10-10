#!/usr/bin/env python3
"""
ESM2 MLP Fine-tuning with CAFA Evaluation
Add --use_graph flag to enable GO DAG graph convolution
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

# Import CAFA evaluator
try:
    from cafaeval.evaluation import cafa_eval, write_results
    CAFA_AVAILABLE = True
except ImportError:
    print("Warning: cafaeval not installed. CAFA metrics will not be available.")
    print("Install with: pip install cafaeval")
    CAFA_AVAILABLE = False

# ============================================================================
# GO DAG Parser
# ============================================================================

class GODagParser:
    """Parse GO OBO file and build adjacency matrix"""
    
    def __init__(self, obo_path: str):
        self.obo_path = Path(obo_path)
        self.go_terms = {}
        
    def parse(self):
        """Parse OBO file"""
        print(f"Parsing GO OBO file: {self.obo_path}")
        
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
    
    def build_adjacency_matrix(self, go_list: List[str]):
        """Build adjacency matrix for graph convolution"""
        n = len(go_list)
        go_to_idx = {go: i for i, go in enumerate(go_list)}
        
        A = np.zeros((n, n), dtype=np.float32)
        
        for i, go_id in enumerate(go_list):
            A[i, i] = 1.0
            
            if go_id in self.go_terms:
                parents = self.go_terms[go_id]['parents']
                for parent in parents:
                    if parent in go_to_idx:
                        j = go_to_idx[parent]
                        A[i, j] = 1.0
        
        print(f"  Adjacency matrix: {A.shape}, Edges: {(A > 0).sum()}, Density: {(A > 0).sum() / (n*n):.4f}")
        
        A_sparse = torch.FloatTensor(A).to_sparse()
        return A_sparse

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration for CAFA3 experiments"""
    
    CAFA3_BASE = Path("/home/zijianzhou/Datasets/cafa3")
    ESM_EMBEDDINGS_DIR = Path("/home/zijianzhou/Datasets/esm")
    GO_OBO_PATH = CAFA3_BASE / "go.obo"
    
    ESM_DIM = 1280
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    DROPOUT = 0.3
    
    MIN_GO_FREQUENCY = 0
    MAX_GO_FREQUENCY = 1
    
    DEBUG_MODE = False
    DEBUG_SAMPLES = 100
    DEBUG_GO_TERMS = 50
    
    OUTPUT_DIR = Path("./cafa3_comparison_experiments")
    
    def __init__(self, go_aspect: str = "mf", debug_mode: bool = False, use_graph: bool = False):
        self.GO_ASPECT = go_aspect
        self.DEBUG_MODE = debug_mode
        self.USE_GRAPH = use_graph
        
        suffix = "_protgo" if use_graph else "_baseline"
        aspect_dir = self.OUTPUT_DIR / f"{go_aspect}{suffix}"
        self.CHECKPOINT_DIR = aspect_dir / "checkpoints"
        self.RESULTS_DIR = aspect_dir / "results"
        self.CAFA_DIR = aspect_dir / "cafa_eval"
        
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        self.CAFA_DIR.mkdir(exist_ok=True, parents=True)
        
        if debug_mode:
            self.NUM_EPOCHS = 2
            print(f"Debug mode enabled")
        
        if use_graph:
            print(f"Graph convolution ENABLED (ProtGO method)")
        else:
            print(f"Baseline MLP (no graph)")

# ============================================================================
# Data Loader
# ============================================================================

class CAFA3DataLoader:
    """Load CAFA3 dataset with pre-computed ESM embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
        self.go_columns = []
        
    def load_aspect_data(self, aspect: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"Loading {aspect.upper()} aspect data...")
        
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
        
        self.go_columns = [col for col in train_data.columns if col.startswith('GO:')]
        print(f"Found {len(self.go_columns)} GO terms")
        
        go_frequencies = train_data[self.go_columns].mean()
        valid_go_terms = go_frequencies[
            (go_frequencies >= self.config.MIN_GO_FREQUENCY) & 
            (go_frequencies <= self.config.MAX_GO_FREQUENCY)
        ].index.tolist()
        
        print(f"Filtered GO terms: {len(self.go_columns)} -> {len(valid_go_terms)}")
        self.go_columns = valid_go_terms
        
        if self.config.DEBUG_MODE:
            print(f"Debug mode: using {self.config.DEBUG_SAMPLES} samples")
            train_data = train_data.head(self.config.DEBUG_SAMPLES)
            val_data = val_data.head(min(self.config.DEBUG_SAMPLES // 5, len(val_data)))
            test_data = test_data.head(min(self.config.DEBUG_SAMPLES // 5, len(test_data)))
            
            if len(self.go_columns) > self.config.DEBUG_GO_TERMS:
                self.go_columns = self.go_columns[:self.config.DEBUG_GO_TERMS]
        
        train_proteins = train_data['proteins'].tolist()
        val_proteins = val_data['proteins'].tolist()
        test_proteins = test_data['proteins'].tolist()
        
        train_labels = train_data[self.go_columns].values.astype(np.float32)
        val_labels = val_data[self.go_columns].values.astype(np.float32)
        test_labels = test_data[self.go_columns].values.astype(np.float32)
        
        print(f"\nDataset Statistics:")
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
# PyTorch Dataset
# ============================================================================

class CAFA3MLPDataset(Dataset):
    """Simple dataset - directly load embeddings"""
    
    def __init__(self, proteins: List[str], labels: np.ndarray, embeddings_dir: Path):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.embeddings_dir = embeddings_dir
        
        print(f"Pre-loading {len(proteins)} embeddings...")
        self.embeddings = []
        missing_count = 0
        
        for protein_id in tqdm(proteins, desc="Loading embeddings"):
            emb_path = self.embeddings_dir / f"{protein_id}.npy"
            
            if emb_path.exists():
                try:
                    embedding = np.load(emb_path, allow_pickle=True)
                    
                    if isinstance(embedding, np.ndarray) and embedding.dtype == object:
                        embedding = embedding.item()
                    
                    if isinstance(embedding, dict):
                        if 'mean' in embedding:
                            embedding = embedding['mean']
                        elif 'pooled' in embedding:
                            embedding = embedding['pooled']
                        elif 'embedding' in embedding:
                            embedding = embedding['embedding']
                            if len(embedding.shape) == 2:
                                embedding = embedding.mean(axis=0)
                    
                    embedding = np.asarray(embedding, dtype=np.float32)
                    
                    if len(embedding.shape) == 2:
                        embedding = embedding.mean(axis=0)
                    
                    embedding = embedding.flatten()
                    
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
            print(f"Warning: {missing_count}/{len(proteins)} embeddings missing or failed to load")
        
        self.embeddings = torch.FloatTensor(np.array(self.embeddings))
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ============================================================================
# Model Components
# ============================================================================

class GraphConvolution(nn.Module):
    """Graph convolution layer from ProtGO"""
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.kaiming_normal_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support.transpose(0, 1)).t()
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ESM2Classifier(nn.Module):
    """Unified classifier with optional graph convolution"""
    
    def __init__(self, config: Config, num_go_terms: int, adjacency_matrix=None):
        super().__init__()
        self.use_graph = config.USE_GRAPH
        
        self.fc1 = nn.Linear(config.ESM_DIM, 512)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, num_go_terms)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        
        if self.use_graph:
            assert adjacency_matrix is not None, "Adjacency matrix required for graph mode"
            self.gc1 = GraphConvolution(num_go_terms, num_go_terms)
            self.register_buffer('adj', adjacency_matrix)
            print(f"  Model: FC1({config.ESM_DIM} -> {2*config.ESM_DIM}) -> ReLU -> FC2({2*config.ESM_DIM} -> {num_go_terms}) -> GraphConv({num_go_terms} -> {num_go_terms})")
        else:
            print(f"  Model: FC1({config.ESM_DIM} -> {2*config.ESM_DIM}) -> ReLU -> FC2({2*config.ESM_DIM} -> {num_go_terms})")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        if self.use_graph:
            x = self.gc1(x, self.adj)
        
        return x

# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Simple full-precision trainer"""
    
    def __init__(self, model, config: Config, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.6
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []
        
        mode = "ProtGO" if config.USE_GRAPH else "Baseline"
        print(f"Trainer initialized: Device={self.device}, Mode={mode}, LR={config.LEARNING_RATE}")
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for embeddings, labels in tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            
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
        
        return avg_loss, metrics, all_preds
    
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
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _ = self.evaluate(val_loader)
            
            self.scheduler.step()
            
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
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                print(f"  New best model! F1: {best_val_f1:.3f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics):
        suffix = "_protgo" if self.config.USE_GRAPH else "_baseline"
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
# CAFA Evaluation
# ============================================================================

def run_cafa_evaluation(proteins, y_true, y_pred, go_terms, config, split_name="test"):
    """Run CAFA evaluation and return metrics"""
    
    if not CAFA_AVAILABLE:
        print("CAFA evaluator not available, skipping CAFA metrics")
        return {}
    
    print(f"\nRunning CAFA evaluation on {split_name} set...")
    
    # Create temporary files for CAFA evaluation
    pred_dir = config.CAFA_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)
    
    # Write prediction file
    pred_file = pred_dir / f"{split_name}_predictions.tsv"
    with open(pred_file, 'w') as f:
        for i, protein in enumerate(proteins):
            for j, go_term in enumerate(go_terms):
                score = y_pred[i, j]
                if score > 0:  # Only write non-zero predictions
                    f.write(f"{protein}\t{go_term}\t{score:.6f}\n")
    
    # Write ground truth file
    gt_file = config.CAFA_DIR / f"{split_name}_ground_truth.tsv"
    with open(gt_file, 'w') as f:
        for i, protein in enumerate(proteins):
            for j, go_term in enumerate(go_terms):
                if y_true[i, j] == 1:
                    f.write(f"{protein}\t{go_term}\n")
    
    # Run CAFA evaluation
    try:
        results_df, best_scores = cafa_eval(
            str(config.GO_OBO_PATH),
            str(pred_dir),
            str(gt_file),
            norm='cafa',
            prop='max'
        )
        
        # Save detailed results
        results_df.to_csv(config.CAFA_DIR / f"{split_name}_evaluation_all.tsv", sep='\t')
        
        # Extract best metrics (Fmax, Smin, wFmax)
        cafa_metrics = {}
        
        # Fmax - Maximum F-measure
        if 'f' in best_scores and not best_scores['f'].empty:
            row = best_scores['f'].iloc[0]
            cafa_metrics['fmax'] = row.get('f', 0.0)
            cafa_metrics['fmax_precision'] = row.get('pr', 0.0)
            cafa_metrics['fmax_recall'] = row.get('rc', 0.0)
            cafa_metrics['fmax_threshold'] = row.get('tau', 0.0)
        
        # Smin - Minimum Semantic Distance
        if 's' in best_scores and not best_scores['s'].empty:
            row = best_scores['s'].iloc[0]
            cafa_metrics['smin'] = row.get('s', 0.0)
            cafa_metrics['smin_ru'] = row.get('ru', 0.0)  # Remaining uncertainty
            cafa_metrics['smin_mi'] = row.get('mi', 0.0)  # Misinformation
            cafa_metrics['smin_threshold'] = row.get('tau', 0.0)
        
        # wFmax - Weighted Maximum F-measure (if IA file was provided)
        if 'wf' in best_scores and not best_scores['wf'].empty:
            row = best_scores['wf'].iloc[0]
            cafa_metrics['wfmax'] = row.get('wf', 0.0)
            cafa_metrics['wfmax_precision'] = row.get('wpr', 0.0)
            cafa_metrics['wfmax_recall'] = row.get('wrc', 0.0)
            cafa_metrics['wfmax_threshold'] = row.get('tau', 0.0)
        
        # Micro F-measure
        if 'f_micro' in best_scores and not best_scores['f_micro'].empty:
            row = best_scores['f_micro'].iloc[0]
            cafa_metrics['f_micro'] = row.get('f_micro', 0.0)
            cafa_metrics['f_micro_precision'] = row.get('pr_micro', 0.0)
            cafa_metrics['f_micro_recall'] = row.get('rc_micro', 0.0)
        
        print(f"\nCAFA Best Metrics:")
        print(f"  Fmax: {cafa_metrics.get('fmax', 0):.4f} (P={cafa_metrics.get('fmax_precision', 0):.3f}, R={cafa_metrics.get('fmax_recall', 0):.3f}, τ={cafa_metrics.get('fmax_threshold', 0):.2f})")
        if 'smin' in cafa_metrics:
            print(f"  Smin: {cafa_metrics.get('smin', 0):.4f} (RU={cafa_metrics.get('smin_ru', 0):.3f}, MI={cafa_metrics.get('smin_mi', 0):.3f}, τ={cafa_metrics.get('smin_threshold', 0):.2f})")
        if 'wfmax' in cafa_metrics:
            print(f"  wFmax: {cafa_metrics.get('wfmax', 0):.4f} (wP={cafa_metrics.get('wfmax_precision', 0):.3f}, wR={cafa_metrics.get('wfmax_recall', 0):.3f})")
        if 'f_micro' in cafa_metrics:
            print(f"  F-micro: {cafa_metrics.get('f_micro', 0):.4f}")
        
        return cafa_metrics
        
    except Exception as e:
        print(f"Error running CAFA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ============================================================================
# Visualization
# ============================================================================

def visualize_results(history: List[Dict], config: Config):
    """Create training visualization"""
    df = pd.DataFrame(history)
    
    mode = "ProtGO" if config.USE_GRAPH else "Baseline"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'CAFA3 Training - {config.GO_ASPECT.upper()} ({mode})', fontsize=16)
    
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['epoch'], df['f1'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['f1'], label='F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('All Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['epoch'], df['lr'], linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = "_protgo" if config.USE_GRAPH else "_baseline"
    save_path = config.RESULTS_DIR / f'training_curves_{config.GO_ASPECT}{suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {save_path}")

# ============================================================================
# Main
# ============================================================================

def main(go_aspect: str = "mf", debug_mode: bool = False, use_graph: bool = False):
    """Main training pipeline"""
    
    mode = "ProtGO (with Graph Convolution)" if use_graph else "Baseline MLP"
    print(f"CAFA3 ESM2 Training - {mode}")
    print("=" * 60)
    
    config = Config(go_aspect=go_aspect, debug_mode=debug_mode, use_graph=use_graph)
    
    # Load data
    data_loader = CAFA3DataLoader(config)
    (train_proteins, train_labels, 
     val_proteins, val_labels,
     test_proteins, test_labels, 
     go_terms) = data_loader.prepare_data()
    
    # Build adjacency matrix (only if using graph)
    adjacency_matrix = None
    if use_graph:
        go_parser = GODagParser(config.GO_OBO_PATH)
        go_parser.parse()
        adjacency_matrix = go_parser.build_adjacency_matrix(go_terms)
    
    # Create datasets
    train_dataset = CAFA3MLPDataset(train_proteins, train_labels, config.ESM_EMBEDDINGS_DIR)
    val_dataset = CAFA3MLPDataset(val_proteins, val_labels, config.ESM_EMBEDDINGS_DIR)
    test_dataset = CAFA3MLPDataset(test_proteins, test_labels, config.ESM_EMBEDDINGS_DIR)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = ESM2Classifier(config, num_go_terms=len(go_terms), adjacency_matrix=adjacency_matrix)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train
    trainer = Trainer(model, config)
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    suffix = "_protgo" if use_graph else "_baseline"
    pd.DataFrame(history).to_csv(
        config.RESULTS_DIR / f'training_history_{go_aspect}{suffix}.csv', index=False
    )
    
    visualize_results(history, config)
    
    # Test evaluation with both simple and CAFA metrics
    print("\nFinal test evaluation...")
    test_loss, test_metrics, test_preds = trainer.evaluate(test_loader)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Metrics: P={test_metrics['precision']:.3f}, "
          f"R={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}, "
          f"AUPRC={test_metrics['auprc']:.3f}")
    
    # Run CAFA evaluation
    cafa_metrics = run_cafa_evaluation(
        test_proteins, 
        test_labels, 
        test_preds, 
        go_terms, 
        config, 
        split_name="test"
    )
    
    # Combine all metrics
    all_test_metrics = {**test_metrics, **cafa_metrics}
    
    # Save final results
    results = {
        'go_aspect': go_aspect,
        'num_go_terms': len(go_terms),
        'method': 'protgo' if use_graph else 'baseline',
        'use_graph': use_graph,
        'test_metrics': all_test_metrics,
        'best_val_f1': max([h['f1'] for h in history]),
        'num_epochs': len(history)
    }
    
    with open(config.RESULTS_DIR / f'final_results_{go_aspect}{suffix}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete! Results saved to: {config.RESULTS_DIR}")
    print(f"CAFA evaluation files saved to: {config.CAFA_DIR}")
    
    return model, history, all_test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAFA3 ESM2 Training with Optional Graph Convolution')
    parser.add_argument("--aspect", type=str, default="mf", choices=["mf", "bp", "cc"],
                       help="GO aspect to train on")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode with small data")
    parser.add_argument("--use_graph", action="store_true", 
                       help="Enable ProtGO graph convolution layer")
    args = parser.parse_args()
    
    main(go_aspect=args.aspect, debug_mode=args.debug, use_graph=args.use_graph)