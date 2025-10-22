"""Training script for triple modality adaptive fusion."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from triple_adaptive_model import TripleModalityAdaptiveFusion
from triple_dataset import TripleModalityDataset, collate_fn
sys.path.append('/home/zijianzhou/project/PFP/experiments/PLMs')
from cafa3_config import CAFA3Config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        text = batch['text'].to(device)
        prott5 = batch['prott5'].to(device)
        esm = batch['esm'].to(device)
        labels = batch['labels'].to(device)
        
        logits, diversity_loss = model(text, prott5, esm)
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
            text = batch['text'].to(device)
            prott5 = batch['prott5'].to(device)
            esm = batch['esm'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(text, prott5, esm)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall, micro_auprc, macro_auprc


def evaluate_with_cafa_triple(model, loader, device, protein_ids, go_terms, obo_file, output_dir):
    """CAFA evaluation wrapper for triple modality model."""
    try:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from text.utils.cafa_evaluation import evaluate_with_cafa
        
        # Create a wrapper loader that returns compatible batches
        class TripleToSingleBatch:
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for batch in self.loader:
                    # Concatenate all modalities as "embeddings"
                    yield {
                        'embeddings': torch.cat([
                            batch['text'],
                            batch['prott5'],
                            batch['esm']
                        ], dim=-1),
                        'labels': batch['labels']
                    }
            
            def __len__(self):
                return len(self.loader)
        
        # Create wrapper model that expects concatenated input
        class ModelWrapper(nn.Module):
            def __init__(self, triple_model):
                super().__init__()
                self.model = triple_model
                self.text_dim = 768
                self.prott5_dim = 1024
                self.esm_dim = 1280
            
            def forward(self, embeddings):
                text = embeddings[:, :self.text_dim]
                prott5 = embeddings[:, self.text_dim:self.text_dim+self.prott5_dim]
                esm = embeddings[:, self.text_dim+self.prott5_dim:]
                logits, _ = self.model(text, prott5, esm)
                return logits
        
        wrapped_model = ModelWrapper(model).to(device)
        wrapped_loader = TripleToSingleBatch(loader)
        
        return evaluate_with_cafa(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type='triple_fusion',
            model_name='triple_fusion'
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        return {}


def train_model(aspect):
    """Main training function."""
    # Configuration
    seed = 42
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining Triple Adaptive Fusion for {aspect}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    
    # Setup directories
    data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
    embedding_dirs = {
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/text',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/prott5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/esm'
    }
    output_dir = Path(f"/home/zijianzhou/project/PFP/experiments/PLMs/plm_results/triple_fusion/{aspect}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = TripleModalityDataset(data_dir, embedding_dirs, aspect, 'train')
    val_dataset = TripleModalityDataset(data_dir, embedding_dirs, aspect, 'valid')
    test_dataset = TripleModalityDataset(data_dir, embedding_dirs, aspect, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                           collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # Create model
    num_go_terms = train_dataset.labels.shape[1]
    model = TripleModalityAdaptiveFusion(
        text_dim=768,
        prott5_dim=1024,
        esm_dim=1280,
        hidden_dim=512,
        num_go_terms=num_go_terms
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_fmax = 0
    patience_counter = 0
    patience = 10
    
    history = []
    
    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\nEpoch {epoch}/50")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Fmax: {val_fmax:.4f} (t={val_threshold:.3f})")
        print(f"  Precision: {val_prec:.4f}, Recall: {val_recall:.4f}")
        print(f"  Micro-AUPRC: {val_micro_auprc:.4f}, Macro-AUPRC: {val_macro_auprc:.4f}")
        
        history.append({
            'epoch': epoch,
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
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ New best: {best_val_fmax:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_fmax:.4f} (threshold={test_threshold:.3f})")
    print(f"  Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
    print(f"  Micro-AUPRC: {test_micro_auprc:.4f}, Macro-AUPRC: {test_macro_auprc:.4f}")
    
    # CAFA EVALUATION
    obo_file = Path("/home/zijianzhou/project/PFP/go.obo")
    cafa_metrics = {}
    
    if obo_file.exists():
        print("\n" + "="*70)
        print("CAFA-STYLE EVALUATION")
        print("="*70)
        
        # Load GO terms for this aspect
        go_terms_file = data_dir / f"{aspect}_go_terms.json"
        with open(go_terms_file, 'r') as f:
            go_terms = json.load(f)
        
        # Get test protein IDs
        test_protein_ids = test_dataset.protein_ids.tolist()
        
        cafa_metrics = evaluate_with_cafa_triple(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir / 'cafa_eval'
        )
    else:
        print(f"\nWarning: OBO file not found at {obo_file}. Skipping CAFA evaluation.")
    
    # Save results
    results = {
        'model_type': 'triple_adaptive_fusion',
        'aspect': aspect,
        'num_go_terms': num_go_terms,
        'seed': seed,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'micro_auprc': float(test_micro_auprc),
            'macro_auprc': float(test_macro_auprc),
            **cafa_metrics
        },
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': int(best_epoch),
        'total_epochs': epoch,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--aspect', type=str, required=True, choices=['BPO', 'CCO', 'MFO'])
    args = parser.parse_args()
    
    train_model(args.aspect)