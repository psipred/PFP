"""Training script for CAFA3 PLM experiments."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from cafa3_config import CAFA3Config
from cafa3_dataset import CAFA3PLMDataset, collate_fn
from plm_classifier import PLMClassifier


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def compute_fmax(y_true, y_pred, thresholds=None):
    """Compute Fmax metric."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    best_fmax = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum(axis=1)
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum(axis=1)
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum(axis=1)
        
        precision = (tp / (tp + fp + 1e-10)).mean()
        recall = (tp / (tp + fn + 1e-10)).mean()
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return best_fmax, best_threshold, best_precision, best_recall


def compute_auprc(y_true, y_pred):
    """Compute micro and macro AUPRC."""
    from sklearn.metrics import average_precision_score
    
    # Micro-averaged
    micro_auprc = average_precision_score(y_true.ravel(), y_pred.ravel())
    
    # Macro-averaged
    term_auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            term_auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            term_auprcs.append(term_auprc)
    
    macro_auprc = np.mean(term_auprcs) if term_auprcs else 0.0
    
    return micro_auprc, macro_auprc


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    # Compute metrics
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall, micro_auprc, macro_auprc


def train_model(config):
    """Main training function."""
    device = config.device
    print(f"\nTraining {config.plm_type.upper()} model for {config.aspect} on {device}")
    
    # Load datasets
    train_dataset = CAFA3PLMDataset(
        config.data_dir, config.embedding_dir, 
        config.aspect, 'train', config.plm_type
    )
    val_dataset = CAFA3PLMDataset(
        config.data_dir, config.embedding_dir,
        config.aspect, 'valid', config.plm_type
    )
    test_dataset = CAFA3PLMDataset(
        config.data_dir, config.embedding_dir,
        config.aspect, 'test', config.plm_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, collate_fn=collate_fn, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )
    
    # Get number of GO terms
    num_go_terms = train_dataset.labels.shape[1]
    print(f"Number of GO terms: {num_go_terms}")
    
    # Create model
    model = PLMClassifier(
        num_go_terms=num_go_terms,
        embedding_dim=config.embedding_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.min_delta
    )
    
    # Training loop
    best_val_fmax = 0
    best_threshold = 0.5
    best_epoch = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Fmax: {val_fmax:.4f} (t={val_threshold:.3f})")
        print(f"  Precision: {val_prec:.4f}, Recall: {val_recall:.4f}")
        print(f"  Micro-AUPRC: {val_micro_auprc:.4f}, Macro-AUPRC: {val_macro_auprc:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fmax': val_fmax,
            'threshold': val_threshold,
            'precision': val_prec,
            'recall': val_recall,
            'micro_auprc': val_micro_auprc,
            'macro_auprc': val_macro_auprc
        })
        
        # Save best model
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_threshold = val_threshold
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.checkpoint_dir / "best_model.pt")
            print(f"  ✓ New best: {best_val_fmax:.4f}")
        
        # Early stopping
        if early_stopping(val_fmax):
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation Fmax: {best_val_fmax:.4f} at epoch {best_epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(config.checkpoint_dir / "best_model.pt"))
    test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{config.plm_type.upper()} Test Results:")
    print(f"  Fmax: {test_fmax:.4f} (threshold={test_threshold:.3f})")
    print(f"  Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
    print(f"  Micro-AUPRC: {test_micro_auprc:.4f}, Macro-AUPRC: {test_macro_auprc:.4f}")
    
    # Save results
    results = {
        'plm_type': config.plm_type,
        'aspect': config.aspect,
        'num_go_terms': num_go_terms,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'micro_auprc': float(test_micro_auprc),
            'macro_auprc': float(test_macro_auprc)
        },
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    with open(config.results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(config.results_dir / "training_history.csv", index=False)
    
    print(f"\n✓ Results saved to: {config.results_dir}")
    
    return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PLM classifier on CAFA3 data")
    parser.add_argument('--plm', type=str, required=True, 
                       choices=['esm', 'prott5', 'prostt5'],
                       help='PLM type to use')
    parser.add_argument('--aspect', type=str, required=True,
                       choices=['BPO', 'CCO', 'MFO'],
                       help='GO aspect')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    # Setup config
    config = CAFA3Config(
        plm_type=args.plm,
        aspect=args.aspect,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    print("="*70)
    print(f"CAFA3 PLM Experiment")
    print(f"PLM: {args.plm.upper()}, Aspect: {args.aspect}")
    print("="*70)
    
    # Train model
    results = train_model(config)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()