"""Training script for pairwise modality adaptive fusion."""

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

from pairwise_adaptive_model import PairwiseModalityAdaptiveFusion
from pairwise_dataset import PairwiseModalityDataset, collate_fn
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
    """Compute micro-averaged Fmax."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    best_fmax = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
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
    
    micro_auprc = average_precision_score(y_true.ravel(), y_pred.ravel())
    
    term_auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            term_auprc = average_precision_score(y_true[:, i], y_pred[:, i])
            term_auprcs.append(term_auprc)
    
    macro_auprc = np.mean(term_auprcs) if term_auprcs else 0.0
    
    return micro_auprc, macro_auprc


def train_epoch(model, loader, optimizer, criterion, device, 
                diversity_weight=0.0, gate_entropy_weight=0.001):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        mod1 = batch['mod1'].to(device)
        mod2 = batch['mod2'].to(device)
        labels = batch['labels'].to(device)
        
        logits, diversity_loss, gate_entropy_loss = model(mod1, mod2)
        loss = criterion(logits, labels)
        
        if diversity_loss is not None:
            loss = loss + diversity_weight * diversity_loss
        
        if gate_entropy_loss is not None:
            loss = loss + gate_entropy_weight * gate_entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def log_gate_statistics(model, loader, device):
    """Log gate behavior on validation set."""
    model.eval()
    all_gate1 = []
    all_gate2 = []
    
    with torch.no_grad():
        for batch in loader:
            mod1 = batch['mod1'].to(device)
            mod2 = batch['mod2'].to(device)
            
            h1 = model.mod1_transform(mod1)
            h2 = model.mod2_transform(mod2)
            
            g1, g2 = model.compute_adaptive_gates(h1, h2)
            
            all_gate1.append(g1.cpu())
            all_gate2.append(g2.cpu())
    
    gate1_mean = torch.cat(all_gate1).mean().item()
    gate2_mean = torch.cat(all_gate2).mean().item()
    
    print(f"  Gate means: {model.mod1_name}={gate1_mean:.3f}, {model.mod2_name}={gate2_mean:.3f}")
    
    return gate1_mean, gate2_mean


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            mod1 = batch['mod1'].to(device)
            mod2 = batch['mod2'].to(device)
            labels = batch['labels'].to(device)

            logits, _, _ = model(mod1, mod2)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    return total_loss / len(loader), fmax, threshold, precision, recall, micro_auprc, macro_auprc


def evaluate_with_cafa_pairwise(model, loader, device, protein_ids, go_terms, obo_file, output_dir, dim_config):
    """CAFA evaluation wrapper for pairwise modality model."""
    try:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from text.utils.cafa_evaluation import evaluate_with_cafa
        
        mod1, mod2 = model.modality_pair.split('_')
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        
        class PairwiseToSingleBatch:
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for batch in self.loader:
                    yield {
                        'embeddings': torch.cat([
                            batch['mod1'],
                            batch['mod2']
                        ], dim=-1),
                        'labels': batch['labels']
                    }
            
            def __len__(self):
                return len(self.loader)
        
        class ModelWrapper(nn.Module):
            def __init__(self, pairwise_model, dim1, dim2):
                super().__init__()
                self.model = pairwise_model
                self.dim1 = dim1
                self.dim2 = dim2
            
            def forward(self, embeddings):
                mod1 = embeddings[:, :self.dim1]
                mod2 = embeddings[:, self.dim1:]
                logits, _, _ = self.model(mod1, mod2)
                return logits
        
        wrapped_model = ModelWrapper(model, dim1, dim2).to(device)
        wrapped_loader = PairwiseToSingleBatch(loader)
        
        return evaluate_with_cafa(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type='pairwise_fusion',
            model_name=model.modality_pair
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        return {}


def train_model(modality_pair, aspect, use_diversity_loss=False, diversity_weight=0.01):
    """Main training function."""
    seed = 42
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Experiment name
    exp_name = f"{modality_pair}_fusion"
    if use_diversity_loss:
        exp_name = f"{modality_pair}_fusion_div"
    
    print(f"\nTraining Pairwise Adaptive Fusion: {modality_pair.upper()}")
    print(f"Aspect: {aspect}")
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Diversity Loss: {'ENABLED' if use_diversity_loss else 'DISABLED'}")
    if use_diversity_loss:
        print(f"Diversity Weight: {diversity_weight}")
    
    # Dimension configuration
    dim_config = {
        'text': 768,
        'prott5': 1024,
        'prostt5': 1024,
        'esm': 1280
    }
    
    # Setup directories
    data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
    embedding_dirs = {
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/text',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/prott5',
        'prostt5': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/prostt5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/esm'
    }
    
    output_dir = Path(f"/home/zijianzhou/project/PFP/experiments/PLMs/plm_results/{exp_name}/{aspect}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'train')
    val_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'valid')
    test_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                           collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # Create model
    num_go_terms = train_dataset.labels.shape[1]
    model = PairwiseModalityAdaptiveFusion(
        modality_pair=modality_pair,
        dim_config=dim_config,
        hidden_dim=512,
        num_go_terms=num_go_terms,
        use_diversity_loss=use_diversity_loss,
        diversity_weight=diversity_weight
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_fmax = 0
    patience_counter = 0
    patience = 5
    history = []
    
    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                diversity_weight if use_diversity_loss else 0.0)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc = evaluate(
            model, val_loader, criterion, device
        )
        
        if epoch % 5 == 0:
            log_gate_statistics(model, val_loader, device)

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
        
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ New best: {best_val_fmax:.4f}")
        else:
            patience_counter += 1
        
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
    
    # CAFA evaluation
    obo_file = Path("/home/zijianzhou/project/PFP/go.obo")
    cafa_metrics = {}
    
    if obo_file.exists():
        print("\n" + "="*70)
        print("CAFA-STYLE EVALUATION")
        print("="*70)
        
        go_terms_file = data_dir / f"{aspect}_go_terms.json"
        with open(go_terms_file, 'r') as f:
            go_terms = json.load(f)
        
        test_protein_ids = test_dataset.protein_ids.tolist()
        
        cafa_metrics = evaluate_with_cafa_pairwise(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir / 'cafa_eval',
            dim_config=dim_config
        )
    else:
        print(f"\nWarning: OBO file not found at {obo_file}. Skipping CAFA evaluation.")
    
    # Save results
    results = {
        'model_type': 'pairwise_adaptive_fusion',
        'modality_pair': modality_pair,
        'experiment_name': exp_name,
        'aspect': aspect,
        'num_go_terms': num_go_terms,
        'seed': seed,
        'use_diversity_loss': use_diversity_loss,
        'diversity_weight': diversity_weight if use_diversity_loss else None,
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
    print(f"✓ Experiment name: {exp_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality-pair', type=str, required=True, 
                   choices=['text_prott5', 'text_esm', 'prott5_esm', 'prott5_prostt5'])
    parser.add_argument('--aspect', type=str, required=True, choices=['BPO', 'CCO', 'MFO'])
    parser.add_argument('--use-diversity-loss', action='store_true')
    parser.add_argument('--diversity-weight', type=float, default=0.01)
    args = parser.parse_args()
    
    train_model(args.modality_pair, args.aspect, args.use_diversity_loss, args.diversity_weight)