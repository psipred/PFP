#!/usr/bin/env python3
"""
ESM2 Fine-tuning for CAFA3 GO Prediction with F-max evaluation
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration for CAFA3 experiments"""
    
    # Data paths
    CAFA3_BASE = Path("/home/zijianzhou/Datasets/cafa3")
    
    # GO Aspects to use
    GO_ASPECTS = ["mf", "bp", "cc"]
    
    # Model settings
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    MAX_SEQ_LENGTH = 1022
    
    # Training settings
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Early stopping



    MIN_EPOCHS = 10
    
    # Memory/performance settings
    USE_AMP = True
    AMP_DTYPE = 'fp16'
    ENABLE_GRADIENT_CHECKPOINTING = True
    
    # LoRA settings
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_TARGET_MODULES = ["query", "value"]
    
    # Data settings
    MIN_GO_FREQUENCY = 0
    MAX_GO_FREQUENCY = 1
    
    # Debug settings
    DEBUG_MODE = False
    DEBUG_SAMPLES = 100
    DEBUG_GO_TERMS = 50
    
    # Output paths
    OUTPUT_DIR = Path("./cafa3_experiments")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    def __init__(self, go_aspect: str = "mf", debug_mode: bool = False):
        """Initialize configuration"""
        self.GO_ASPECT = go_aspect
        self.DEBUG_MODE = debug_mode
        
        # Create output directories
        aspect_dir = self.OUTPUT_DIR / go_aspect
        self.CHECKPOINT_DIR = aspect_dir / "checkpoints"
        self.RESULTS_DIR = aspect_dir / "results"
        
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        
        if debug_mode:
            self.NUM_EPOCHS = 2
            print(f"🔧 Debug mode enabled - using reduced settings")

# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStop:
    """Early stopping utility"""
    
    def __init__(self, patience=5, min_epochs=10):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.stop_flag = False
        self.epoch = 0
        
    def __call__(self, loss, score, model):
        """
        Args:
            loss: Current validation loss (not used but kept for compatibility)
            score: Current validation score (e.g., F-max)
            model: Model to save if best
        """
        self.epoch += 1
        
        # Don't stop before minimum epochs
        if self.epoch < self.min_epochs:
            self.best_score = score if self.best_score is None else max(self.best_score, score)
            return
        
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.best_score = score
            self.counter = 0
    
    def stop(self):
        return self.stop_flag

# ============================================================================
# Evaluation Metrics - F-max only
# ============================================================================

def calculate_fmax(predictions, labels, thresholds=None):
    """
    Calculate F-max score (CAFA metric)
    
    Args:
        predictions: Tensor of predictions (after sigmoid)
        labels: Tensor of true labels
        thresholds: Optional specific thresholds to evaluate
    
    Returns:
        tuple: (fmax, best_threshold, precision, recall)
    """
    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.99, 99)
    
    predictions = predictions.cpu()
    labels = labels.cpu()
    
    fmax = 0.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in thresholds:
        pred_binary = (predictions >= threshold).float()
        
        # Calculate true positives, false positives, false negatives
        tp = (pred_binary * labels).sum().item()
        fp = (pred_binary * (1 - labels)).sum().item()
        fn = ((1 - pred_binary) * labels).sum().item()
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > fmax:
                fmax = f1
                best_threshold = threshold.item()
                best_precision = precision
                best_recall = recall
    
    return fmax, best_threshold, best_precision, best_recall

def calculate_auprc(y_true, y_pred):
    """
    Calculate average AUPRC across all GO terms
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predictions (numpy array, after sigmoid)
    
    Returns:
        Average AUPRC score
    """
    auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # Only if there are positive samples
            auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            auprcs.append(auprc)
    
    return np.mean(auprcs) if auprcs else 0.0

# ============================================================================
# Data Loading
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
        """Load train/val/test data for a specific GO aspect"""
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
        """Prepare data for training"""
        
        if self.config.GO_ASPECT == "all":
            # Combine all three aspects
            print("📂 Loading all GO aspects (MF + BP + CC)...")
            all_train, all_val, all_test = [], [], []
            
            for aspect in ["mf", "bp", "cc"]:
                train_df, val_df, test_df = self.load_aspect_data(aspect)
                all_train.append(train_df)
                all_val.append(val_df)
                all_test.append(test_df)
            
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
        
        # Filter GO terms by frequency
        self.filter_go_terms_by_frequency()
        
        # Debug mode: use subset
        if self.config.DEBUG_MODE:
            print(f"🔧 Debug mode: using {self.config.DEBUG_SAMPLES} samples and {self.config.DEBUG_GO_TERMS} GO terms")
            self.train_data = self.train_data.head(self.config.DEBUG_SAMPLES)
            self.val_data = self.val_data.head(min(self.config.DEBUG_SAMPLES // 5, len(self.val_data)))
            self.test_data = self.test_data.head(min(self.config.DEBUG_SAMPLES // 5, len(self.test_data)))
            
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
        
        valid_go_terms = go_frequencies[
            (go_frequencies >= self.config.MIN_GO_FREQUENCY) & 
            (go_frequencies <= self.config.MAX_GO_FREQUENCY)
        ].index.tolist()
        
        print(f"📊 Filtered GO terms: {len(self.go_columns)} -> {len(valid_go_terms)}")
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
            padding=True,
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
# Model Implementation
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
        
        # Enable gradient checkpointing to save memory
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
            # Simple linear classifier for LoRA
            self.dropout = nn.Dropout(0.3)
            self.go_classifier = nn.Linear(self.hidden_size, num_go_terms)
            print("📊 Using direct linear classifier (LoRA mode)")
        else:
            # 3-layer MLP for non-LoRA
            self.dropout = nn.Dropout(0.3)
            self.go_classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size // 4, num_go_terms)
            )
            print("📊 Using 3-layer MLP classifier (non-LoRA mode)")
        
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
        
        # Early stopping
        self.early_stop = EarlyStop(
            patience=getattr(config, 'PATIENCE', 5),
            min_epochs=getattr(config, 'MIN_EPOCHS', 10)
        )
        
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
                    if logits.size(-1) != labels.size(-1):
                        raise ValueError(f"Logits/labels dim mismatch")
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
        """Evaluate model using F-max metric only"""
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
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())
        
        # Combine all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate F-max (primary CAFA metric)
        fmax, best_threshold, precision, recall = calculate_fmax(all_preds, all_labels)
        
        # Calculate AUPRC as secondary metric
        auprc = calculate_auprc(all_labels.numpy(), all_preds.numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'fmax': fmax,
            'best_threshold': best_threshold,
            'precision': precision,
            'recall': recall,
            'auprc': auprc
        }
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Full training loop with early stopping based on F-max"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        best_val_fmax = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Log results
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F-max: {val_metrics['fmax']:.4f} (threshold={val_metrics['best_threshold']:.3f})")
            print(f"  Val P/R: {val_metrics['precision']:.3f} / {val_metrics['recall']:.3f}")
            print(f"  Val AUPRC: {val_metrics['auprc']:.3f}")
            
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **val_metrics
            })
            
            # Early stopping based on F-max
            self.early_stop(-val_metrics['loss'], val_metrics['fmax'], self.model)
            
            # Save best model based on F-max
            if val_metrics['fmax'] > best_val_fmax:
                best_val_fmax = val_metrics['fmax']
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_metrics)
                print(f"  ✅ New best model saved! (F-max: {best_val_fmax:.4f})")
            
            # Check early stopping
            if self.early_stop.stop():
                print(f"  ⚠️ Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Restored best model with F-max: {best_val_fmax:.4f}")
        
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
        path = self.config.CHECKPOINT_DIR / f"best_model_{self.config.GO_ASPECT}_fmax{metrics['fmax']:.4f}.pt"
        torch.save(checkpoint, path)

# ============================================================================
# Visualization
# ============================================================================

def visualize_results(history: List[Dict], config: Config):
    """Create training visualization"""
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'CAFA3 Training Results - {config.GO_ASPECT.upper()} Aspect', fontsize=16)
    
    # Loss plot
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F-max plot (primary metric)
    axes[0, 1].plot(df['epoch'], df['fmax'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F-max Score')
    axes[0, 1].set_title('Validation F-max Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=df['fmax'].max(), color='r', linestyle='--', alpha=0.5, 
                       label=f'Best: {df["fmax"].max():.4f}')
    axes[0, 1].legend()
    
    # AUPRC plot
    axes[1, 0].plot(df['epoch'], df['auprc'], linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUPRC')
    axes[1, 0].set_title('Validation AUPRC')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision and Recall
    axes[1, 1].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    axes[1, 1].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall at Best Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = config.RESULTS_DIR / f'training_curves_{config.GO_ASPECT}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plots saved to {save_path}")

# ============================================================================
# Main Pipeline
# ============================================================================

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
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize model
    num_go_terms = int(train_dataset.labels.shape[1])
    assert val_dataset.labels.shape[1] == num_go_terms, "Val labels width mismatch"
    assert test_dataset.labels.shape[1] == num_go_terms, "Test labels width mismatch"
    
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
    test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test F-max: {test_metrics['fmax']:.4f} (threshold={test_metrics['best_threshold']:.3f})")
    print(f"  Test P/R: {test_metrics['precision']:.3f} / {test_metrics['recall']:.3f}")
    print(f"  Test AUPRC: {test_metrics['auprc']:.3f}")
    
    # Save final results
    best_val_fmax = max([h['fmax'] for h in history])
    final_results = {
        'go_aspect': go_aspect,
        'use_lora': use_lora,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'best_val_fmax': best_val_fmax,
        'test_metrics': test_metrics,
        'training_history': history
    }
    
    with open(config.RESULTS_DIR / f'final_results_{go_aspect}.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ Training complete! Results saved to: {config.RESULTS_DIR}")
    print(f"   Best Validation F-max: {best_val_fmax:.4f}")
    print(f"   Test F-max: {test_metrics['fmax']:.4f}")
    
    return model, history, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ESM2 on CAFA3 dataset")
    parser.add_argument("--aspect", type=str, default="mf", 
                        choices=["mf", "bp", "cc", "all"],
                        help="GO aspect to train on")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode with small dataset")
    parser.add_argument("--use-lora", action="store_true", default=True, 
                        help="Use LoRA fine-tuning")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", 
                        help="Use last-layer fine-tuning")
    
    args = parser.parse_args()
    
    if args.debug:
        print(f"🐛 Running in DEBUG mode ({args.aspect.upper()} aspect)")
        main(go_aspect=args.aspect, debug_mode=True, use_lora=args.use_lora)
    else:
        print(f"🔥 Running with FULL CAFA3 dataset ({args.aspect.upper()} aspect)")
        main(go_aspect=args.aspect, debug_mode=False, use_lora=args.use_lora)