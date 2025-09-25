#!/usr/bin/env python3
"""
ESM2 Fine-tuning for CAFA3 GO Prediction
Simplified implementation for CAFA3 dataset with binary GO annotations
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration for CAFA3 experiments"""
    
    # Data paths
    CAFA3_BASE = Path("/home/zijianzhou/Datasets/cafa3")
    
    # GO Aspects to use
    GO_ASPECTS = ["mf", "bp", "cc"]  # molecular function, biological process, cellular component
    
    # Model settings
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    MAX_SEQ_LENGTH = 1022  # ESM2 max minus special tokens
    
    # Training settings
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Memory/performance settings
    USE_AMP = True  # Enable mixed precision
    AMP_DTYPE = 'auto'  # 'auto' | 'fp16' | 'bf16' | 'none'
    ENABLE_GRADIENT_CHECKPOINTING = True  # Save activation memory
    
    # LoRA settings
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_TARGET_MODULES = ["query", "value"]
    
    # Data settings
    MIN_GO_FREQUENCY = 0.01  # Minimum frequency of positive samples for a GO term
    MAX_GO_FREQUENCY = 0.99  # Maximum frequency (to filter out too common terms)
    
    # Debug settings
    DEBUG_MODE = False
    DEBUG_SAMPLES = 1000
    DEBUG_GO_TERMS = 100  # Number of GO terms to use in debug mode
    
    # Output paths
    OUTPUT_DIR = Path("./cafa3_experiments")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    def __init__(self, go_aspect: str = "mf", debug_mode: bool = False):
        """
        Initialize configuration
        
        Args:
            go_aspect: Which GO aspect to train on ("mf", "bp", "cc", or "all")
            debug_mode: Whether to run in debug mode
        """
        self.GO_ASPECT = go_aspect
        self.DEBUG_MODE = debug_mode
        
        # Create output directories
        aspect_dir = self.OUTPUT_DIR / go_aspect
        self.CHECKPOINT_DIR = aspect_dir / "checkpoints"
        self.RESULTS_DIR = aspect_dir / "results"
        
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        
        # Adjust settings for debug mode
        if debug_mode:
            self.NUM_EPOCHS = 2
            print(f"🔧 Debug mode enabled - using reduced settings")
            print(f"   - Samples: {self.DEBUG_SAMPLES}")
            print(f"   - GO terms: {self.DEBUG_GO_TERMS}")
            print(f"   - Epochs: {self.NUM_EPOCHS}")

# ============================================================================
# CAFA3 Data Loader
# ============================================================================

class CAFA3DataLoader:
    """Load and process CAFA3 dataset with binary GO annotations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.go_columns = []
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_aspect_data(self, aspect: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/val/test data for a specific GO aspect
        
        Args:
            aspect: One of "mf", "bp", "cc"
        """
        print(f"📂 Loading {aspect.upper()} aspect data...")
        
        # File paths
        train_path = self.config.CAFA3_BASE / f"{aspect}-training.csv"
        val_path = self.config.CAFA3_BASE / f"{aspect}-validation.csv"
        test_path = self.config.CAFA3_BASE / f"{aspect}-test.csv"
        
        # Load CSVs
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_data(self) -> Tuple:
        """
        Prepare data for training
        
        Returns:
            Tuple of (sequences, labels) for train/val/test
        """
        
        if self.config.GO_ASPECT == "all":
            # Combine all three aspects
            print("📂 Loading all GO aspects (MF + BP + CC)...")
            all_train, all_val, all_test = [], [], []
            
            for aspect in ["mf", "bp", "cc"]:
                train_df, val_df, test_df = self.load_aspect_data(aspect)
                all_train.append(train_df)
                all_val.append(val_df)
                all_test.append(test_df)
            
            # Merge on proteins and sequences columns
            self.train_data = all_train[0][['proteins', 'sequences']].copy()
            self.val_data = all_val[0][['proteins', 'sequences']].copy()
            self.test_data = all_test[0][['proteins', 'sequences']].copy()
            
            # Combine GO columns from all aspects
            for i, aspect_dfs in enumerate(zip(all_train, all_val, all_test)):
                for df in aspect_dfs:
                    go_cols = [col for col in df.columns if col.startswith('GO:')]
                    if i == 0:
                        self.train_data = pd.concat([self.train_data, df[go_cols]], axis=1)
                        self.val_data = pd.concat([self.val_data, all_val[i][go_cols]], axis=1)
                        self.test_data = pd.concat([self.test_data, all_test[i][go_cols]], axis=1)
                    else:
                        # Add non-overlapping GO terms
                        new_go_cols = [col for col in go_cols if col not in self.train_data.columns]
                        if new_go_cols:
                            self.train_data = pd.concat([self.train_data, df[new_go_cols]], axis=1)
                            self.val_data = pd.concat([self.val_data, all_val[i][new_go_cols]], axis=1)
                            self.test_data = pd.concat([self.test_data, all_test[i][new_go_cols]], axis=1)
        else:
            # Load single aspect
            self.train_data, self.val_data, self.test_data = self.load_aspect_data(self.config.GO_ASPECT)
        
        # Get GO columns
        self.go_columns = [col for col in self.train_data.columns if col.startswith('GO:')]
        print(f"📊 Found {len(self.go_columns)} GO terms")
        
        # Filter sequences by length
        print(f"✂️  Filtering sequences longer than {self.config.MAX_SEQ_LENGTH}...")
        self.train_data = self.train_data[self.train_data['sequences'].str.len() <= self.config.MAX_SEQ_LENGTH]
        self.val_data = self.val_data[self.val_data['sequences'].str.len() <= self.config.MAX_SEQ_LENGTH]
        self.test_data = self.test_data[self.test_data['sequences'].str.len() <= self.config.MAX_SEQ_LENGTH]
        
        # Filter GO terms by frequency
        self.filter_go_terms_by_frequency()
        
        # Debug mode: use subset
        if self.config.DEBUG_MODE:
            print(f"🔧 Debug mode: using {self.config.DEBUG_SAMPLES} samples and {self.config.DEBUG_GO_TERMS} GO terms")
            self.train_data = self.train_data.head(self.config.DEBUG_SAMPLES)
            self.val_data = self.val_data.head(min(self.config.DEBUG_SAMPLES // 5, len(self.val_data)))
            self.test_data = self.test_data.head(min(self.config.DEBUG_SAMPLES // 5, len(self.test_data)))
            
            # Use only first N GO terms for debugging
            if len(self.go_columns) > self.config.DEBUG_GO_TERMS:
                self.go_columns = self.go_columns[:self.config.DEBUG_GO_TERMS]
        
        # Extract sequences and labels
        train_sequences = self.train_data['sequences'].tolist()
        val_sequences = self.val_data['sequences'].tolist()
        test_sequences = self.test_data['sequences'].tolist()
        
        train_labels = self.train_data[self.go_columns].values.astype(np.float32)
        val_labels = self.val_data[self.go_columns].values.astype(np.float32)
        test_labels = self.test_data[self.go_columns].values.astype(np.float32)
        
        # Print statistics
        self.print_data_statistics(train_labels, val_labels, test_labels)
        
        return (train_sequences, train_labels, 
                val_sequences, val_labels,
                test_sequences, test_labels, 
                self.go_columns)
    
    def filter_go_terms_by_frequency(self):
        """Filter GO terms that are too rare or too common"""
        go_frequencies = self.train_data[self.go_columns].mean()
        
        # Find GO terms within frequency range
        valid_go_terms = go_frequencies[
            (go_frequencies >= self.config.MIN_GO_FREQUENCY) & 
            (go_frequencies <= self.config.MAX_GO_FREQUENCY)
        ].index.tolist()
        
        print(f"📊 Filtered GO terms: {len(self.go_columns)} -> {len(valid_go_terms)}")
        print(f"   (keeping terms with {self.config.MIN_GO_FREQUENCY:.1%} - {self.config.MAX_GO_FREQUENCY:.1%} frequency)")
        
        self.go_columns = valid_go_terms
    
    def print_data_statistics(self, train_labels, val_labels, test_labels):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("📈 Dataset Statistics:")
        print("="*60)
        print(f"GO Aspect: {self.config.GO_ASPECT.upper()}")
        print(f"Number of GO terms: {len(self.go_columns)}")
        print(f"\nSamples:")
        print(f"  Train: {len(train_labels)}")
        print(f"  Val:   {len(val_labels)}")
        print(f"  Test:  {len(test_labels)}")
        print(f"\nLabel Statistics (Train):")
        print(f"  Avg GO terms per protein: {train_labels.sum(axis=1).mean():.2f}")
        print(f"  Label density: {train_labels.mean():.4f}")
        print(f"  Min GO terms per protein: {train_labels.sum(axis=1).min():.0f}")
        print(f"  Max GO terms per protein: {train_labels.sum(axis=1).max():.0f}")
        print("="*60)

# ============================================================================
# PyTorch Dataset
# ============================================================================

class CAFA3Dataset(Dataset):
    """PyTorch dataset for CAFA3"""
    
    def __init__(self, sequences: List[str], labels: np.ndarray, tokenizer, max_length: int = 1022):
        self.sequences = sequences
        self.labels = torch.FloatTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return raw sequence and label; tokenization happens in collate_fn
        return {
            'sequence': self.sequences[idx],
            'labels': self.labels[idx]
        }


def create_collate_fn(tokenizer, max_length: int):
    """Dynamic padding collate function to reduce memory usage."""
    def collate(batch):
        sequences = [b['sequence'] for b in batch]
        labels = torch.stack([b['labels'] for b in batch], dim=0)
        encoding = tokenizer(
            sequences,
            padding=True,            # pad to longest in batch
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels
        }
    return collate

# ============================================================================
# Model Implementations
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
    def forward(self, x):
        original_out = F.linear(x, self.weight)
        lora_out = x @ self.lora_A @ self.lora_B * self.scaling
        return original_out + lora_out


class ESM2ForCAFA3(nn.Module):
    """ESM2 model for CAFA3 GO prediction"""
    
    def __init__(self, config: Config, num_go_terms: int, use_lora: bool = True):
        super().__init__()
        self.config = config
        self.use_lora = use_lora
        
        # Load ESM2
        print(f"🔄 Loading {config.MODEL_NAME}...")
        self.esm = EsmModel.from_pretrained(config.MODEL_NAME)
        self.hidden_size = self.esm.config.hidden_size
        
        # Enable gradient checkpointing to save memory (if available)
        if getattr(config, 'ENABLE_GRADIENT_CHECKPOINTING', False):
            try:
                if hasattr(self.esm, 'gradient_checkpointing_enable'):
                    self.esm.gradient_checkpointing_enable()
                if hasattr(self.esm.config, 'gradient_checkpointing'):
                    self.esm.config.gradient_checkpointing = True
                if hasattr(self.esm.config, 'use_cache'):
                    self.esm.config.use_cache = False
                print("🧠 Gradient checkpointing enabled")
            except Exception as e:
                print(f"⚠️ Could not enable gradient checkpointing: {e}")
        
        # Freeze base model
        for param in self.esm.parameters():
            param.requires_grad = False
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora()
        
        # GO prediction head (kept simple for maintainability)
        self.dropout = nn.Dropout(0.3)
        self.go_classifier = nn.Linear(self.hidden_size, num_go_terms)
        
    def _apply_lora(self):
        """Apply LoRA to attention layers"""
        print(f"🔧 Applying LoRA (rank={self.config.LORA_RANK}, alpha={self.config.LORA_ALPHA})")
        lora_count = 0
        
        for name, module in self.esm.named_modules():
            if any(target in name for target in self.config.LORA_TARGET_MODULES):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.config.LORA_RANK,
                        alpha=self.config.LORA_ALPHA
                    )
                    lora_layer.weight.data = module.weight.data.clone()
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.esm
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, child_name, lora_layer)
                    lora_count += 1
        
        print(f"   Applied LoRA to {lora_count} modules")
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.go_classifier(pooled_output)
        return logits

# ============================================================================
# Training Functions
# ============================================================================

class Trainer:
    """Training manager for CAFA3 experiments"""
    
    def __init__(self, model, config: Config, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # AMP setup
        self.use_amp = bool(getattr(config, 'USE_AMP', False)) and torch.cuda.is_available()
        amp_dtype = getattr(config, 'AMP_DTYPE', 'auto')
        if amp_dtype == 'auto':
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        elif amp_dtype == 'bf16':
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == 'fp16':
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.history = []
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for i, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = self.model(input_ids, attention_mask)
                    # Early shape check to surface issues clearly
                    if logits.size(-1) != labels.size(-1):
                        raise ValueError(
                            f"Logits/labels dim mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}"
                        )
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (i + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
                pbar.set_postfix({'loss': total_loss / (i + 1)})
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = self.calculate_metrics(all_labels, all_preds)
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """Calculate evaluation metrics"""
        from sklearn.metrics import average_precision_score
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate per-sample metrics
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum(axis=1)
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum(axis=1)
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum(axis=1)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Calculate AUPRC for each GO term
        auprcs = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:  # Only if there are positive samples
                auprc = average_precision_score(y_true[:, i], y_pred[:, i])
                auprcs.append(auprc)
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'auprc': np.mean(auprcs) if auprcs else 0.0
        }
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Full training loop"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Log results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: P={val_metrics['precision']:.3f}, "
                  f"R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}, "
                  f"AUPRC={val_metrics['auprc']:.3f}")
            
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            })
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, val_metrics)
                print(f"  ✅ New best model saved! (F1: {best_val_f1:.3f})")
        
        return self.history
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        path = self.config.CHECKPOINT_DIR / f"best_model_{self.config.GO_ASPECT}.pt"
        torch.save(checkpoint, path)

# ============================================================================
# Main Pipeline
# ============================================================================

def visualize_results(history: List[Dict], config: Config):
    """Create training visualization"""
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'CAFA3 Training Results - {config.GO_ASPECT.upper()} Aspect', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 plot
    axes[0, 1].plot(df['epoch'], df['f1'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUPRC plot
    axes[0, 2].plot(df['epoch'], df['auprc'], linewidth=2, color='purple')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUPRC')
    axes[0, 2].set_title('Validation AUPRC')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision plot
    axes[1, 0].plot(df['epoch'], df['precision'], linewidth=2, color='blue')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Validation Precision')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall plot
    axes[1, 1].plot(df['epoch'], df['recall'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Validation Recall')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined metrics
    axes[1, 2].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    axes[1, 2].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[1, 2].plot(df['epoch'], df['f1'], label='F1', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('All Metrics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = config.RESULTS_DIR / f'training_curves_{config.GO_ASPECT}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 Plots saved to {save_path}")


def main(go_aspect: str = "mf", debug_mode: bool = False, use_lora: bool = True):
    """
    Main training pipeline for CAFA3
    
    Args:
        go_aspect: Which GO aspect to train on ("mf", "bp", "cc", or "all")
        debug_mode: If True, use small subset for quick testing
        use_lora: If True, use LoRA fine-tuning, else use last-layer fine-tuning
    """
    
    print("🚀 Starting CAFA3 ESM2 Fine-tuning Pipeline")
    print("=" * 60)
    
    # Initialize configuration
    config = Config(go_aspect=go_aspect, debug_mode=debug_mode)
    
    # Load CAFA3 data
    print(f"\n📚 Loading CAFA3 Dataset ({go_aspect.upper()} aspect)...")
    data_loader = CAFA3DataLoader(config)
    
    # Prepare datasets
    (train_sequences, train_labels, 
     val_sequences, val_labels,
     test_sequences, test_labels, 
     go_terms) = data_loader.prepare_data()
    
    # Save metadata
    print("\n💾 Saving metadata...")
    metadata = {
        'go_aspect': go_aspect,
        'go_terms': go_terms,
        'num_go_terms': len(go_terms),
        'num_train': len(train_sequences),
        'num_val': len(val_sequences),
        'num_test': len(test_sequences),
        'debug_mode': debug_mode,
        'use_lora': use_lora
    }
    with open(config.RESULTS_DIR / f'metadata_{go_aspect}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Initialize tokenizer
    print("\n🔧 Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = CAFA3Dataset(train_sequences, train_labels, tokenizer, config.MAX_SEQ_LENGTH)
    val_dataset = CAFA3Dataset(val_sequences, val_labels, tokenizer, config.MAX_SEQ_LENGTH)
    test_dataset = CAFA3Dataset(test_sequences, test_labels, tokenizer, config.MAX_SEQ_LENGTH)
    
    # Create dataloaders with dynamic padding
    collate_fn = create_collate_fn(tokenizer, config.MAX_SEQ_LENGTH)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model using actual dataset label width to avoid mismatches
    num_go_terms = int(train_dataset.labels.shape[1])
    # Sanity check all splits match
    assert val_dataset.labels.shape[1] == num_go_terms, (
        f"Val labels width {val_dataset.labels.shape[1]} != train {num_go_terms}")
    assert test_dataset.labels.shape[1] == num_go_terms, (
        f"Test labels width {test_dataset.labels.shape[1]} != train {num_go_terms}")
    model = ESM2ForCAFA3(config, num_go_terms=num_go_terms, use_lora=use_lora)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"  Using {'LoRA' if use_lora else 'Last-layer'} fine-tuning")
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train model
    print("\n🏋️ Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Save training history
    pd.DataFrame(history).to_csv(config.RESULTS_DIR / f'training_history_{go_aspect}.csv', index=False)
    
    # Visualize results
    visualize_results(history, config)
    
    # Final evaluation on test set
    print("\n📊 Final evaluation on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Metrics: P={test_metrics['precision']:.3f}, "
          f"R={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}, "
          f"AUPRC={test_metrics['auprc']:.3f}")
    
    # Save final results
    final_results = {
        'go_aspect': go_aspect,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'use_lora': use_lora,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'training_history': history
    }
    
    with open(config.RESULTS_DIR / f'final_results_{go_aspect}.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ Training complete! Results saved to: {config.RESULTS_DIR}")
    
    return model, history, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ESM2 on CAFA3 dataset")
    parser.add_argument("--aspect", type=str, default="mf", 
                        choices=["mf", "bp", "cc", "all"],
                        help="GO aspect to train on (mf=molecular function, bp=biological process, cc=cellular component)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode with small dataset")
    parser.add_argument("--full", action="store_true", 
                        help="Run with full dataset")
    parser.add_argument("--use-lora", action="store_true", default=True, 
                        help="Use LoRA fine-tuning")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", 
                        help="Use last-layer fine-tuning")
    
    args = parser.parse_args()
    
    if args.full:
        print(f"🔥 Running with FULL CAFA3 dataset ({args.aspect.upper()} aspect)")
        main(go_aspect=args.aspect, debug_mode=False, use_lora=args.use_lora)
    else:
        print(f"🐛 Running in DEBUG mode ({args.aspect.upper()} aspect)")
        main(go_aspect=args.aspect, debug_mode=True, use_lora=args.use_lora)
