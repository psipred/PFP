"""Training script for simplified fusion model."""

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

from pairwise_adaptive_model import SimplifiedPairwiseFusion, FocalBCELoss
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


def worker_init_fn(worker_id):
    """Initialize each worker with a different seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def train_epoch(model, loader, optimizer, criterion, scheduler, device, entropy_weight=0.02):
    """Training with entropy regularization."""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_entropy = 0
    gate_stats = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        mod1 = batch['mod1'].to(device)
        mod2 = batch['mod2'].to(device)
        ppi = batch['ppi'].to(device) if 'ppi' in batch else None
        ppi_flag = batch['ppi_flag'].to(device) if 'ppi_flag' in batch else None
        labels = batch['labels'].to(device)
        
        logits, gate1_mean, gate2_mean = model(mod1, mod2, ppi, ppi_flag)
        main_loss = criterion(logits, labels)
        
        # Strong entropy regularization to prevent collapse
        gates = torch.stack([gate1_mean, gate2_mean])
        entropy = -(gates * torch.log(gates + 1e-8) + (1-gates) * torch.log(1-gates + 1e-8)).mean()
        
        loss = main_loss - entropy_weight * entropy  # Maximize entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_entropy += entropy.item()
        gate_stats.append([gate1_mean.item(), gate2_mean.item()])
    
    avg_gates = torch.tensor(gate_stats).mean(0)
    
    return {
        'total_loss': total_loss / len(loader),
        'main_loss': total_main_loss / len(loader),
        'entropy': total_entropy / len(loader),
        'gate1_mean': avg_gates[0].item(),
        'gate2_mean': avg_gates[1].item()
    }


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    gate_stats = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            mod1 = batch['mod1'].to(device)
            mod2 = batch['mod2'].to(device)
            ppi = batch['ppi'].to(device) if 'ppi' in batch else None
            ppi_flag = batch['ppi_flag'].to(device) if 'ppi_flag' in batch else None
            labels = batch['labels'].to(device)

            logits, gate1_mean, gate2_mean = model(mod1, mod2, ppi, ppi_flag)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            gate_stats.append([gate1_mean.item(), gate2_mean.item()])
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
    
    avg_gates = torch.tensor(gate_stats).mean(0)
    
    return (total_loss / len(loader), fmax, threshold, precision, recall, 
            micro_auprc, macro_auprc, avg_gates[0].item(), avg_gates[1].item())


def evaluate_with_cafa_simplified(model, loader, device, protein_ids, go_terms, obo_file, output_dir, dim_config, use_ppi):
    """CAFA evaluation wrapper for simplified model."""
    try:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from text.utils.cafa_evaluation import evaluate_with_cafa
        
        mod1, mod2 = model.modality_pair.split('_')
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        ppi_dim = 512
        
        class SimplifiedToSingleBatch:
            def __init__(self, loader, use_ppi):
                self.loader = loader
                self.use_ppi = use_ppi
            
            def __iter__(self):
                for batch in self.loader:
                    embeddings = torch.cat([
                        batch['mod1'],
                        batch['mod2']
                    ], dim=-1)
                    
                    if self.use_ppi:
                        embeddings = torch.cat([
                            embeddings,
                            batch['ppi'],
                            batch['ppi_flag']
                        ], dim=-1)
                    
                    yield {
                        'embeddings': embeddings,
                        'labels': batch['labels']
                    }
            
            def __len__(self):
                return len(self.loader)
        
        class ModelWrapper(nn.Module):
            def __init__(self, simplified_model, dim1, dim2, use_ppi, ppi_dim=512):
                super().__init__()
                self.model = simplified_model
                self.dim1 = dim1
                self.dim2 = dim2
                self.use_ppi = use_ppi
                self.ppi_dim = ppi_dim
            
            def forward(self, embeddings):
                mod1 = embeddings[:, :self.dim1]
                mod2 = embeddings[:, self.dim1:self.dim1+self.dim2]
                
                if self.use_ppi:
                    ppi = embeddings[:, self.dim1+self.dim2:self.dim1+self.dim2+self.ppi_dim]
                    ppi_flag = embeddings[:, -1:]
                    logits, _, _ = self.model(mod1, mod2, ppi, ppi_flag)
                else:
                    logits, _, _ = self.model(mod1, mod2, None, None)
                
                return logits
        
        wrapped_model = ModelWrapper(model, dim1, dim2, use_ppi, ppi_dim).to(device)
        wrapped_loader = SimplifiedToSingleBatch(loader, use_ppi)
        
        return evaluate_with_cafa(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type='simplified_fusion',
            model_name=f"{model.modality_pair}_{'ppi' if use_ppi else 'base'}"
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_simplified_model(modality_pair, aspect, use_ppi=True):
    """Main training function for simplified model."""
    seed = 42
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    config = {
        'hidden_dim': 512,
        'dropout': 0.3,
        'entropy_weight': 0.0005,  # Stronger entropy
        'lr': 1e-3,
        'weight_decay': 0.01,
        'batch_size': 32,
        'max_epochs': 50,
        'patience': 5,  
        'gradient_clip': 1.0,
        'warmup_ratio': 0.1,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25
    }
    
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
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/text',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/prott5',
        'prostt5': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/prostt5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/esm',
        'ppi': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/ppi'
    }
    
    exp_name = f"{modality_pair}_simplified{'_ppi' if use_ppi else ''}"
    output_dir = Path(f"/home/zijianzhou/project/PFP/experiments/PLMs/gatefuse/results_simplified/{exp_name}/{aspect}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training SIMPLIFIED Fusion Model")
    print(f"Modality Pair: {modality_pair.upper()}")
    print(f"Aspect: {aspect}")
    print(f"PPI: {'ENABLED' if use_ppi else 'DISABLED'}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'train', use_ppi=use_ppi)
    val_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'valid', use_ppi=use_ppi)
    test_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'test', use_ppi=use_ppi)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                             collate_fn=collate_fn, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                           collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # Create model
    num_go_terms = train_dataset.labels.shape[1]
    model = SimplifiedPairwiseFusion(
        modality_pair=modality_pair,
        dim_config=dim_config,
        hidden_dim=config['hidden_dim'],
        num_go_terms=num_go_terms,
        dropout=config['dropout'],
        use_ppi=use_ppi,
        ppi_dim=512
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    criterion = FocalBCELoss(gamma=config['focal_gamma'], alpha=config['focal_alpha']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['max_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=num_training_steps,
        pct_start=config['warmup_ratio'],
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_fmax = 0
    patience_counter = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(1, config['max_epochs'] + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scheduler, 
                                   device, config['entropy_weight'])
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc, val_gate1, val_gate2 = val_results
        
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        print(f"  Train: Loss={train_metrics['total_loss']:.4f}, Main={train_metrics['main_loss']:.4f}, "
              f"Entropy={train_metrics['entropy']:.4f}")
        print(f"  Gates: {model.mod1_name}={train_metrics['gate1_mean']:.3f}, "
              f"{model.mod2_name}={train_metrics['gate2_mean']:.3f}")
        print(f"  Val: Loss={val_loss:.4f}, Fmax={val_fmax:.4f} (t={val_threshold:.3f})")
        print(f"  Val Gates: {model.mod1_name}={val_gate1:.3f}, {model.mod2_name}={val_gate2:.3f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        history.append({
            'epoch': epoch,
            **train_metrics,
            'val_loss': val_loss,
            'val_fmax': val_fmax,
            'val_threshold': val_threshold,
            'val_precision': val_prec,
            'val_recall': val_recall,
            'val_micro_auprc': val_micro_auprc,
            'val_macro_auprc': val_macro_auprc,
            'val_gate1': val_gate1,
            'val_gate2': val_gate2,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ New best model saved (Fmax: {best_val_fmax:.4f})")
        else:
            patience_counter += 1
        
        # Check for gate collapse
        if abs(val_gate1 - 0.5) > 0.45 or abs(val_gate2 - 0.5) > 0.45:
            print(f"  ⚠️  Warning: Gate collapse detected!")
        
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_results = evaluate(model, test_loader, criterion, device)
    test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc, test_gate1, test_gate2 = test_results
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_fmax:.4f} (threshold={test_threshold:.3f})")
    print(f"  Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")
    print(f"  Micro-AUPRC: {test_micro_auprc:.4f}, Macro-AUPRC: {test_macro_auprc:.4f}")
    print(f"  Final Gates: {model.mod1_name}={test_gate1:.3f}, {model.mod2_name}={test_gate2:.3f}")
    
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
        
        cafa_metrics = evaluate_with_cafa_simplified(
            model=model,
            loader=test_loader,
            device=device,
            protein_ids=test_protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir / 'cafa_eval',
            dim_config=dim_config,
            use_ppi=use_ppi
        )
    
    # Save results
    results = {
        'model_type': 'simplified_pairwise_fusion',
        'modality_pair': modality_pair,
        'experiment_name': exp_name,
        'aspect': aspect,
        'num_go_terms': num_go_terms,
        'num_parameters': n_params,
        'seed': seed,
        'use_ppi': use_ppi,
        'config': config,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'micro_auprc': float(test_micro_auprc),
            'macro_auprc': float(test_macro_auprc),
            'gate1_mean': float(test_gate1),
            'gate2_mean': float(test_gate2),
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
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality-pair', type=str, required=True,
                       choices=['text_prott5', 'text_esm', 'prott5_esm', 'prott5_prostt5'])
    parser.add_argument('--aspect', type=str, required=True, choices=['BPO', 'CCO', 'MFO'])
    parser.add_argument('--no-ppi', action='store_true', help='Disable PPI')
    args = parser.parse_args()
    
    train_simplified_model(args.modality_pair, args.aspect, use_ppi=not args.no_ppi)