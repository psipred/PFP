"""Ablation study training script for prott5+text fusion."""

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

from ablation_model import PairwiseAblationModel
sys.path.append('/home/zijianzhou/project/PFP/experiments/PLMs')
from gatefuse.pairwise_dataset import PairwiseModalityDataset, collate_fn
from cafa3_config import CAFA3Config


def check_embeddings_exist(data_dir, embedding_dirs, modality_pair, aspect, split):
    """Check which embeddings are missing."""
    names_file = data_dir / f"{aspect}_{split}_names.npy"
    protein_ids = np.load(names_file, allow_pickle=True)
    
    mod1, mod2 = modality_pair.split('_')
    mod1_dir = Path(embedding_dirs[mod1])
    mod2_dir = Path(embedding_dirs[mod2])
    
    missing = {mod1: [], mod2: []}
    
    for protein_id in protein_ids:
        mod1_file = mod1_dir / f"{protein_id}.npy"
        mod2_file = mod2_dir / f"{protein_id}.npy"
        
        if not mod1_file.exists():
            missing[mod1].append(protein_id)
        if not mod2_file.exists():
            missing[mod2].append(protein_id)
    
    total_missing = len(set(missing[mod1] + missing[mod2]))
    
    if total_missing > 0:
        print(f"\nWARNING - Missing embeddings in {split} set:")
        print(f"  {mod1}: {len(missing[mod1])} missing")
        print(f"  {mod2}: {len(missing[mod2])} missing")
        print(f"  Total proteins affected: {total_missing}/{len(protein_ids)}")
        print(f"  Will be filtered out during training")
    
    return missing


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


def log_gate_statistics(model, loader, device):
    """Log gate behavior on validation set."""
    if not model.use_adaptive_gates:
        return None, None
    
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
    gate1_std = torch.cat(all_gate1).std().item()
    gate2_std = torch.cat(all_gate2).std().item()
    
    print(f"  Gate stats: {model.mod1_name}={gate1_mean:.3f}±{gate1_std:.3f}, "
          f"{model.mod2_name}={gate2_mean:.3f}±{gate2_std:.3f}")
    
    return {
        f'{model.mod1_name}_mean': gate1_mean,
        f'{model.mod1_name}_std': gate1_std,
        f'{model.mod2_name}_mean': gate2_mean,
        f'{model.mod2_name}_std': gate2_std
    }


def evaluate_with_cafa_pairwise(model, loader, device, protein_ids, go_terms, obo_file, output_dir, dim_config):
    """CAFA evaluation wrapper."""
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
            model_type='ablation',
            model_name=model.modality_pair
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        return {}


def run_ablation(ablation_config, aspect, data_dir, embedding_dirs, output_base_dir, 
                 dim_config, seed=42):
    """Run single ablation experiment."""
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    exp_name = ablation_config['name']
    print(f"\n{'='*70}")
    print(f"Ablation: {exp_name}")
    print(f"Aspect: {aspect}")
    print(f"{'='*70}")
    
    # Setup output directory
    output_dir = output_base_dir / exp_name / aspect
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for missing embeddings
    modality_pair = 'text_prott5'
    print("\nChecking embedding availability...")
    check_embeddings_exist(data_dir, embedding_dirs, modality_pair, aspect, 'train')
    check_embeddings_exist(data_dir, embedding_dirs, modality_pair, aspect, 'valid')
    check_embeddings_exist(data_dir, embedding_dirs, modality_pair, aspect, 'test')
    
    # Load datasets
    modality_pair = 'text_prott5'
    train_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'train', cache_embeddings=True)
    val_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'valid', cache_embeddings=True)
    test_dataset = PairwiseModalityDataset(data_dir, embedding_dirs, modality_pair, aspect, 'test', cache_embeddings=True)
    
    # # Filter out samples with missing embeddings
    # print(f"Train samples before filtering: {len(train_dataset)}")
    # valid_train_indices = [idx for idx in range(len(train_dataset)) if idx in train_dataset.embedding_cache]
    # valid_val_indices = [idx for idx in range(len(val_dataset)) if idx in val_dataset.embedding_cache]
    # valid_test_indices = [idx for idx in range(len(test_dataset)) if idx in test_dataset.embedding_cache]
    
    # from torch.utils.data import Subset
    # train_dataset = Subset(train_dataset, valid_train_indices)
    # val_dataset = Subset(val_dataset, valid_val_indices)
    # test_dataset = Subset(test_dataset, valid_test_indices)
    
    # print(f"Train samples after filtering: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    # exit()
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                           collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    # Create model with ablation settings
    num_go_terms = train_dataset.labels.shape[1]
    model = PairwiseAblationModel(
        modality_pair=modality_pair,
        dim_config=dim_config,
        hidden_dim=512,
        num_go_terms=num_go_terms,
        **ablation_config['model_kwargs']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_fmax = 0
    patience_counter = 0
    patience = 5
    history = []
    
    diversity_weight = ablation_config['model_kwargs'].get('diversity_weight', 0.0)
    gate_entropy_weight = ablation_config['model_kwargs'].get('gate_entropy_weight', 0.001)
    
    for epoch in range(1, 2):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                diversity_weight, gate_entropy_weight)
        val_loss, val_fmax, val_threshold, val_prec, val_recall, val_micro_auprc, val_macro_auprc = evaluate(
            model, val_loader, criterion, device
        )
        
        gate_stats = None
        if epoch % 10 == 0:
            gate_stats = log_gate_statistics(model, val_loader, device)

        print(f"\nEpoch {epoch}/50")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Fmax: {val_fmax:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fmax': val_fmax,
            'threshold': val_threshold,
            'precision': val_prec,
            'recall': val_recall,
            'micro_auprc': val_micro_auprc,
            'macro_auprc': val_macro_auprc,
            **(gate_stats or {})
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
    
    # Final gate statistics
    final_gate_stats = log_gate_statistics(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Fmax: {test_fmax:.4f}")
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
    
    # Save results
    results = {
        'ablation_name': exp_name,
        'ablation_description': ablation_config['description'],
        'modality_pair': modality_pair,
        'aspect': aspect,
        'num_go_terms': num_go_terms,
        'num_parameters': n_params,
        'seed': seed,
        'model_config': ablation_config['model_kwargs'],
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_prec),
            'recall': float(test_recall),
            'micro_auprc': float(test_micro_auprc),
            'macro_auprc': float(test_macro_auprc),
            **(final_gate_stats or {}),
            **cafa_metrics
        },
        'best_val_fmax': float(best_val_fmax),
        'best_epoch': int(best_epoch),
        'total_epochs': epoch,
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return results


def main():
    """Run all ablation studies."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--aspect', type=str, required=True, choices=['BPO', 'CCO', 'MFO'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Configuration
    dim_config = {
        'text': 768,
        'prott5': 1024,
        'prostt5': 1024,
        'esm': 1280
    }
    
    data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
    embedding_dirs = {
        'text': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/text',
        'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/prott5',
        'prostt5': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/prostt5',
        'esm': '/home/zijianzhou/project/PFP/experiments/PLMs/embedding_cache/esm'
    }
    
    output_base_dir = Path(f"/home/zijianzhou/project/PFP/experiments/PLMs/ablation_study")
    
    # Define ablation experiments
    ablations = [
        # {
        #     'name': '01_full_model',
        #     'description': 'Full model with all components',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': True,
        #         'use_interaction': True,
        #         'use_diversity_loss': False,
        #         'use_gate_entropy': True,
        #         'learnable_temperature': True,
        #         'temperature': 1.5,
        #         'gate_entropy_weight': 0.001
        #     }
        # },
        # {
        #     'name': '02_no_gate_adjuster',
        #     'description': 'Remove gate adjustment mechanism',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': False,
        #         'use_interaction': True,
        #         'use_diversity_loss': False,
        #         'use_gate_entropy': True,
        #         'learnable_temperature': True,
        #         'temperature': 1.5,
        #         'gate_entropy_weight': 0.001
        #     }
        # },
        # {
        #     'name': '03_fixed_temperature',
        #     'description': 'Fixed temperature (not learnable)',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': True,
        #         'use_interaction': True,
        #         'use_diversity_loss': False,
        #         'use_gate_entropy': True,
        #         'learnable_temperature': False,
        #         'temperature': 1.5,
        #         'gate_entropy_weight': 0.001
        #     }
        # },
        # {
        #     'name': '04_no_interaction',
        #     'description': 'Remove cross-modal interaction layer',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': True,
        #         'use_interaction': False,
        #         'use_diversity_loss': False,
        #         'use_gate_entropy': True,
        #         'learnable_temperature': True,
        #         'temperature': 1.5,
        #         'gate_entropy_weight': 0.001
        #     }
        # },
        # {
        #     'name': '05_no_gate_entropy',
        #     'description': 'Remove gate entropy regularization',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': True,
        #         'use_interaction': True,
        #         'use_diversity_loss': False,
        #         'use_gate_entropy': False,
        #         'learnable_temperature': True,
        #         'temperature': 1.5
        #     }
        # },
        # {
        #     'name': '06_fixed_gates_equal',
        #     'description': 'Fixed equal gates (0.5, 0.5)',
        #     'model_kwargs': {
        #         'use_adaptive_gates': False,
        #         'use_interaction': True,
        #         'fixed_gate_weights': [0.5, 0.5]
        #     }
        # },
        # {
        #     'name': '07_fixed_gates_text_bias',
        #     'description': 'Fixed gates biased toward text (0.7, 0.3)',
        #     'model_kwargs': {
        #         'use_adaptive_gates': False,
        #         'use_interaction': True,
        #         'fixed_gate_weights': [0.7, 0.3]
        #     }
        # },
        # {
        #     'name': '08_fixed_gates_prott5_bias',
        #     'description': 'Fixed gates biased toward prott5 (0.3, 0.7)',
        #     'model_kwargs': {
        #         'use_adaptive_gates': False,
        #         'use_interaction': True,
        #         'fixed_gate_weights': [0.3, 0.7]
        #     }
        # },
        # {
        #     'name': '09_with_diversity_loss',
        #     'description': 'Full model + diversity loss',
        #     'model_kwargs': {
        #         'use_adaptive_gates': True,
        #         'use_gate_adjuster': True,
        #         'use_interaction': True,å
        #         'use_diversity_loss': True,
        #         'diversity_weight': 0.01,
        #         'use_gate_entropy': True,
        #         'learnable_temperature': True,
        #         'temperature': 1.5,
        #         'gate_entropy_weight': 0.001
        #     }
        # },
        {
            'name': '10_minimal_model',
            'description': 'Minimal: fixed equal gates, no interaction, no regularization',
            'model_kwargs': {
                'use_adaptive_gates': False,
                'use_interaction': False,
                'fixed_gate_weights': [0.5, 0.5]
            }
        }
    ]
    
    # Run all ablations
    print("="*70)
    print(f"ABLATION STUDY: prott5+text fusion")
    print(f"Aspect: {args.aspect}")
    print(f"Total experiments: {len(ablations)}")
    print(f"Seed: {args.seed}")
    print("="*70)
    
    all_results = {}
    for ablation_config in ablations:
        try:
            results = run_ablation(
                ablation_config=ablation_config,
                aspect=args.aspect,
                data_dir=data_dir,
                embedding_dirs=embedding_dirs,
                output_base_dir=output_base_dir,
                dim_config=dim_config,
                seed=args.seed
            )
            all_results[ablation_config['name']] = results
        except Exception as e:
            print(f"\nERROR in {ablation_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    summary_data = []
    for name, results in all_results.items():
        summary_data.append({
            'Ablation': name,
            'Description': results['ablation_description'],
            'Fmax': results['test_metrics']['fmax'],
            'Micro-AUPRC': results['test_metrics']['micro_auprc'],
            'Macro-AUPRC': results['test_metrics']['macro_auprc'],
            'Precision': results['test_metrics']['precision'],
            'Recall': results['test_metrics']['recall'],
            'Params': results['num_parameters']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Fmax', ascending=False)
    
    print(f"\n{summary_df.to_string(index=False)}")
    
    # Save summary
    summary_output_dir = output_base_dir / 'summary' / args.aspect
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_dir / "ablation_summary.csv", index=False)
    
    # Save detailed results
    with open(summary_output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_output_dir}")
    print("\n✓ Ablation study complete!")


if __name__ == "__main__":
    main()