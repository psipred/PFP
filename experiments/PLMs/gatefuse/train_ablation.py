"""Ablation study for text_prott5 fusion model with CAFA evaluation."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np

sys.path.append('/home/zijianzhou/project/PFP/experiments/PLMs/gatefuse')
from train_pairwise_fusion import train_simplified_model, set_seed
from pairwise_adaptive_model import SimplifiedPairwiseFusion, FocalBCELoss


class SimpleConcat(nn.Module):
    """Baseline: Simple concatenation without gating."""
    def __init__(self, modality_pair, dim_config, hidden_dim=512, num_go_terms=677, 
                 dropout=0.3, use_ppi=False, ppi_dim=512):
        super().__init__()
        self.modality_pair = modality_pair
        self.use_ppi = use_ppi
        
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        dim1 = dim_config[mod1]
        dim2 = dim_config[mod2]
        
        # Simple transforms
        self.mod1_transform = nn.Sequential(
            nn.Linear(dim1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mod2_transform = nn.Sequential(
            nn.Linear(dim2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if use_ppi:
            self.ppi_transform = nn.Sequential(
                nn.Linear(ppi_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.ppi_missing_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # Direct fusion without gating
        fusion_dim = hidden_dim * 2
        if use_ppi:
            fusion_dim += hidden_dim
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_go_terms)
        )
    
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        batch_size = mod1_features.size(0)
        device = mod1_features.device
        
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        if self.use_ppi:
            if ppi_features is not None and ppi_flag is not None:
                ppi_h = self.ppi_transform(ppi_features)
                ppi_flag = ppi_flag.view(batch_size, 1).float()
                ppi_h = ppi_flag * ppi_h + (1 - ppi_flag) * self.ppi_missing_embed.unsqueeze(0)
            else:
                ppi_h = self.ppi_missing_embed.unsqueeze(0).expand(batch_size, -1)
            
            combined = torch.cat([h1, h2, ppi_h], dim=-1)
        else:
            combined = torch.cat([h1, h2], dim=-1)
        
        output = self.fusion(combined)
        
        # Return dummy gates for compatibility
        return output, torch.tensor(0.5), torch.tensor(0.5)


class NoResidualGating(SimplifiedPairwiseFusion):
    """Model without residual connections in gating."""
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        batch_size = mod1_features.size(0)
        device = mod1_features.device
        
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        if self.use_ppi:
            if ppi_features is not None and ppi_flag is not None:
                ppi_h = self.ppi_transform(ppi_features)
                ppi_flag = ppi_flag.view(batch_size, 1).float()
                ppi_h = ppi_flag * ppi_h + (1 - ppi_flag) * self.ppi_missing_embed.unsqueeze(0)
            else:
                ppi_h = self.ppi_missing_embed.unsqueeze(0).expand(batch_size, -1)
                ppi_flag = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        
        gate1, gate2 = self.compute_gates(h1, h2, ppi_flag if self.use_ppi else None)
        
        # NO residual connection
        gated_h1 = gate1 * h1
        gated_h2 = gate2 * h2
        
        if self.use_ppi:
            combined = torch.cat([gated_h1, gated_h2, ppi_h], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2], dim=-1)
        
        output = self.fusion(combined)
        return output, gate1.mean(), gate2.mean()


class NoLearnablePPI(SimplifiedPairwiseFusion):
    """Model without learnable missing PPI embedding."""
    def forward(self, mod1_features, mod2_features, ppi_features=None, ppi_flag=None):
        batch_size = mod1_features.size(0)
        device = mod1_features.device
        
        h1 = self.mod1_transform(mod1_features)
        h2 = self.mod2_transform(mod2_features)
        
        if self.use_ppi:
            if ppi_features is not None and ppi_flag is not None:
                ppi_h = self.ppi_transform(ppi_features)
                ppi_flag = ppi_flag.view(batch_size, 1).float()
                # Use zeros instead of learnable embedding
                ppi_h = ppi_flag * ppi_h
            else:
                ppi_h = torch.zeros(batch_size, 512, device=device)
                ppi_flag = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        
        gate1, gate2 = self.compute_gates(h1, h2, ppi_flag if self.use_ppi else None)
        
        alpha = 0.1
        gated_h1 = gate1 * h1 + alpha * h1
        gated_h2 = gate2 * h2 + alpha * h2
        
        if self.use_ppi:
            combined = torch.cat([gated_h1, gated_h2, ppi_h], dim=-1)
        else:
            combined = torch.cat([gated_h1, gated_h2], dim=-1)
        
        output = self.fusion(combined)
        return output, gate1.mean(), gate2.mean()


class BCELoss(nn.Module):
    """Standard BCE loss (for ablation vs Focal BCE)."""
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


def evaluate_with_cafa_ablation(model, loader, device, protein_ids, go_terms, obo_file, 
                                 output_dir, dim_config, use_ppi, model_class):
    """CAFA evaluation wrapper for ablation models."""
    try:
        # Fix: Use correct import path
        sys.path.append('/home/zijianzhou/project/PFP/experiments')
        from text.utils.cafa_evaluation import evaluate_with_cafa
        
        dim1 = dim_config['text']
        dim2 = dim_config['prott5']
        ppi_dim = 512
        
        class ModelToSingleBatch:
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
            def __init__(self, model, dim1, dim2, use_ppi, ppi_dim=512):
                super().__init__()
                self.model = model
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
        wrapped_loader = ModelToSingleBatch(loader, use_ppi)
        
        return evaluate_with_cafa(
            model=wrapped_model,
            loader=wrapped_loader,
            device=device,
            protein_ids=protein_ids,
            go_terms=go_terms,
            obo_file=obo_file,
            output_dir=output_dir,
            model_type='ablation_model',
            model_name=model_class
        )
    except Exception as e:
        print(f"\nWarning: CAFA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_ablation_study():
    """Run complete ablation study with CAFA evaluation."""
    
    modality_pair = 'text_prott5'
    aspects = ['BPO', 'CCO', 'MFO']
    
    # IMPORTANT: Full model runs FIRST to verify reproducibility
    ablation_configs = {
        '1_full_model': {
            'model_class': 'SimplifiedPairwiseFusion',
            'use_ppi': True,
            'entropy_weight': 0.0005,
            'use_focal_loss': True,
            'description': 'Full model with all components (BASELINE - should match previous results)'
        },
        '2_no_ppi': {
            'model_class': 'SimplifiedPairwiseFusion',
            'use_ppi': False,
            'entropy_weight': 0.0005,
            'use_focal_loss': True,
            'description': 'Without PPI integration'
        },
        '3_no_learnable_ppi': {
            'model_class': 'NoLearnablePPI',
            'use_ppi': True,
            'entropy_weight': 0.0005,
            'use_focal_loss': True,
            'description': 'With PPI but no learnable missing embedding (zeros)'
        },
        '4_no_entropy': {
            'model_class': 'SimplifiedPairwiseFusion',
            'use_ppi': True,
            'entropy_weight': 0.0,
            'use_focal_loss': True,
            'description': 'No entropy regularization (may cause gate collapse)'
        },
        '5_no_residual': {
            'model_class': 'NoResidualGating',
            'use_ppi': True,
            'entropy_weight': 0.0005,
            'use_focal_loss': True,
            'description': 'No residual connections in gating'
        },
        '6_bce_loss': {
            'model_class': 'SimplifiedPairwiseFusion',
            'use_ppi': True,
            'entropy_weight': 0.0005,
            'use_focal_loss': False,
            'description': 'Standard BCE instead of Focal BCE'
        },
        '7_baseline_concat': {
            'model_class': 'SimpleConcat',
            'use_ppi': True,
            'entropy_weight': 0.0,
            'use_focal_loss': True,
            'description': 'Simple concatenation without adaptive gating'
        }
    }
    
    output_base = Path('/home/zijianzhou/project/PFP/experiments/PLMs/gatefuse/case_study/ablation_results')
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"ABLATION: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}\n")
        
        config_results = {}
        
        for aspect in aspects:
            print(f"\n--- Training {aspect} ---")
            
            try:
                from train_pairwise_fusion import (
                    PairwiseModalityDataset, collate_fn,
                    compute_fmax, compute_auprc
                )
                from torch.utils.data import DataLoader
                from tqdm import tqdm
                
                set_seed(42)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Setup
                data_dir = Path("/home/zijianzhou/project/PFP/experiments/PLMs/data")
                embedding_dirs = {
                    'text': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/text',
                    'prott5': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/prott5',
                    'ppi': '/home/zijianzhou/project/PFP/experiments/PLMs/data/embedding_cache/ppi'
                }
                
                dim_config = {'text': 768, 'prott5': 1024}
                
                # Load datasets
                train_dataset = PairwiseModalityDataset(
                    data_dir, embedding_dirs, modality_pair, aspect, 'train',
                    use_ppi=config['use_ppi']
                )
                val_dataset = PairwiseModalityDataset(
                    data_dir, embedding_dirs, modality_pair, aspect, 'valid',
                    use_ppi=config['use_ppi']
                )
                test_dataset = PairwiseModalityDataset(
                    data_dir, embedding_dirs, modality_pair, aspect, 'test',
                    use_ppi=config['use_ppi']
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                        collate_fn=collate_fn, num_workers=8, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                       collate_fn=collate_fn, num_workers=8, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                        collate_fn=collate_fn, num_workers=8, pin_memory=True)
                
                num_go_terms = train_dataset.labels.shape[1]
                
                # Create model based on config
                if config['model_class'] == 'SimpleConcat':
                    model = SimpleConcat(
                        modality_pair=modality_pair,
                        dim_config=dim_config,
                        hidden_dim=512,
                        num_go_terms=num_go_terms,
                        dropout=0.3,
                        use_ppi=config['use_ppi'],
                        ppi_dim=512
                    ).to(device)
                elif config['model_class'] == 'NoResidualGating':
                    model = NoResidualGating(
                        modality_pair=modality_pair,
                        dim_config=dim_config,
                        hidden_dim=512,
                        num_go_terms=num_go_terms,
                        dropout=0.3,
                        use_ppi=config['use_ppi'],
                        ppi_dim=512
                    ).to(device)
                elif config['model_class'] == 'NoLearnablePPI':
                    model = NoLearnablePPI(
                        modality_pair=modality_pair,
                        dim_config=dim_config,
                        hidden_dim=512,
                        num_go_terms=num_go_terms,
                        dropout=0.3,
                        use_ppi=config['use_ppi'],
                        ppi_dim=512
                    ).to(device)
                else:  # SimplifiedPairwiseFusion
                    model = SimplifiedPairwiseFusion(
                        modality_pair=modality_pair,
                        dim_config=dim_config,
                        hidden_dim=512,
                        num_go_terms=num_go_terms,
                        dropout=0.3,
                        use_ppi=config['use_ppi'],
                        ppi_dim=512
                    ).to(device)
                
                # Training setup - loss function based on config
                if config['use_focal_loss']:
                    criterion = FocalBCELoss(gamma=2.0, alpha=0.25).to(device)
                else:
                    criterion = BCELoss().to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                
                # Use same number of epochs as main training
                max_epochs = 1
                num_training_steps = len(train_loader) * max_epochs
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=1e-3, total_steps=num_training_steps,
                    pct_start=0.1, anneal_strategy='cos'
                )
                
                # Training function
                def train_epoch_ablation(model, loader, optimizer, criterion, scheduler, device, entropy_weight):
                    model.train()
                    total_loss = 0
                    
                    for batch in tqdm(loader, desc="Training", leave=False):
                        mod1 = batch['mod1'].to(device)
                        mod2 = batch['mod2'].to(device)
                        ppi = batch['ppi'].to(device) if 'ppi' in batch else None
                        ppi_flag = batch['ppi_flag'].to(device) if 'ppi_flag' in batch else None
                        labels = batch['labels'].to(device)
                        
                        logits, gate1_mean, gate2_mean = model(mod1, mod2, ppi, ppi_flag)
                        main_loss = criterion(logits, labels)
                        
                        if entropy_weight > 0:
                            gates = torch.stack([gate1_mean, gate2_mean])
                            entropy = -(gates * torch.log(gates + 1e-8) + (1-gates) * torch.log(1-gates + 1e-8)).mean()
                            loss = main_loss - entropy_weight * entropy
                        else:
                            loss = main_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        
                        total_loss += loss.item()
                    
                    return total_loss / len(loader)
                
                # Evaluation function
                def evaluate_ablation(model, loader, criterion, device):
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
                            
                            logits, gate1, gate2 = model(mod1, mod2, ppi, ppi_flag)
                            loss = criterion(logits, labels)
                            
                            total_loss += loss.item()
                            all_preds.append(torch.sigmoid(logits).cpu().numpy())
                            all_labels.append(labels.cpu().numpy())
                            gate_stats.append([gate1.item(), gate2.item()])
                    
                    y_pred = np.vstack(all_preds)
                    y_true = np.vstack(all_labels)
                    
                    fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
                    micro_auprc, macro_auprc = compute_auprc(y_true, y_pred)
                    
                    avg_gates = np.array(gate_stats).mean(0)
                    
                    return (total_loss / len(loader), fmax, threshold, precision, recall,
                           micro_auprc, macro_auprc, avg_gates[0], avg_gates[1])
                
                # Training loop
                best_val_fmax = 0
                patience_counter = 0
                patience = 5
                best_epoch = 0
                
                print(f"Starting training (max {max_epochs} epochs, patience {patience})...")
                for epoch in range(1, max_epochs + 1):
                    train_loss = train_epoch_ablation(
                        model, train_loader, optimizer, criterion, scheduler,
                        device, config['entropy_weight']
                    )
                    
                    val_results = evaluate_ablation(model, val_loader, criterion, device)
                    val_loss, val_fmax = val_results[0], val_results[1]
                    
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}: Val Fmax = {val_fmax:.4f}")
                    
                    if val_fmax > best_val_fmax:
                        best_val_fmax = val_fmax
                        best_epoch = epoch
                        patience_counter = 0
                        # Save best model
                        save_dir = output_base / config_name / aspect
                        save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), save_dir / "best_model.pt")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                # Test evaluation
                model.load_state_dict(torch.load(save_dir / "best_model.pt"))
                test_results = evaluate_ablation(model, test_loader, criterion, device)
                test_loss, test_fmax, test_threshold, test_prec, test_recall, test_micro_auprc, test_macro_auprc, test_gate1, test_gate2 = test_results
                
                # CAFA evaluation
                obo_file = Path("/home/zijianzhou/project/PFP/go.obo")
                cafa_metrics = {}
                
                if obo_file.exists():
                    print("\nRunning CAFA evaluation...")
                    go_terms_file = data_dir / f"{aspect}_go_terms.json"
                    with open(go_terms_file, 'r') as f:
                        go_terms = json.load(f)
                    
                    test_protein_ids = test_dataset.protein_ids.tolist()
                    
                    cafa_metrics = evaluate_with_cafa_ablation(
                        model=model,
                        loader=test_loader,
                        device=device,
                        protein_ids=test_protein_ids,
                        go_terms=go_terms,
                        obo_file=obo_file,
                        output_dir=save_dir / 'cafa_eval',
                        dim_config=dim_config,
                        use_ppi=config['use_ppi'],
                        model_class=config_name
                    )
                
                config_results[aspect] = {
                    'test_fmax': float(test_fmax),
                    'test_threshold': float(test_threshold),
                    'test_precision': float(test_prec),
                    'test_recall': float(test_recall),
                    'test_micro_auprc': float(test_micro_auprc),
                    'test_macro_auprc': float(test_macro_auprc),
                    'test_gate1': float(test_gate1),
                    'test_gate2': float(test_gate2),
                    'best_val_fmax': float(best_val_fmax),
                    'best_epoch': int(best_epoch),
                    'total_epochs': epoch,
                    **cafa_metrics
                }
                
                print(f"\n{aspect} Results:")
                print(f"  Test Fmax: {test_fmax:.4f}")
                print(f"  Test Micro-AUPRC: {test_micro_auprc:.4f}")
                if 'cafa_fmax' in cafa_metrics:
                    print(f"  CAFA Fmax: {cafa_metrics['cafa_fmax']:.3f}")
                print(f"  Gates: text={test_gate1:.3f}, prott5={test_gate2:.3f}")
                
                # For full model, compare with expected results
                if config_name == '1_full_model':
                    print(f"\n  ** REPRODUCIBILITY CHECK **")
                    print(f"  Expected from previous runs:")
                    expected = {
                        'BPO': {'fmax': 0.530, 'cafa_fmax': 0.598},
                        'CCO': {'fmax': 0.629, 'cafa_fmax': 0.695},
                        'MFO': {'fmax': 0.613, 'cafa_fmax': 0.675}
                    }
                    if aspect in expected:
                        exp = expected[aspect]
                        fmax_diff = abs(test_fmax - exp['fmax'])
                        print(f"  Expected Fmax: {exp['fmax']:.3f}, Got: {test_fmax:.4f}, Diff: {fmax_diff:.4f}")
                        if 'cafa_fmax' in cafa_metrics:
                            cafa_diff = abs(cafa_metrics['cafa_fmax'] - exp['cafa_fmax'])
                            print(f"  Expected CAFA Fmax: {exp['cafa_fmax']:.3f}, Got: {cafa_metrics['cafa_fmax']:.3f}, Diff: {cafa_diff:.3f}")
                        if fmax_diff < 0.01:
                            print(f"  ✓ REPRODUCIBLE!")
                        else:
                            print(f"  ⚠ Difference > 0.01, check for issues")
                
            except Exception as e:
                print(f"ERROR in {config_name}/{aspect}: {e}")
                import traceback
                traceback.print_exc()
                config_results[aspect] = {'error': str(e)}
        
        all_results[config_name] = {
            'description': config['description'],
            'config': config,
            'results': config_results
        }
        
        # Save intermediate results
        with open(output_base / f'{config_name}_results.json', 'w') as f:
            json.dump(all_results[config_name], f, indent=2)
    
    # Save comprehensive results
    with open(output_base / 'ablation_study_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {output_base}")
    print(f"{'='*80}\n")
    
    # Print summary comparison
    print("\nSUMMARY - Test Fmax Comparison:")
    print("="*80)
    for config_name, config_data in all_results.items():
        print(f"\n{config_name}: {config_data['description']}")
        if 'results' in config_data:
            for aspect in aspects:
                if aspect in config_data['results'] and 'error' not in config_data['results'][aspect]:
                    result = config_data['results'][aspect]
                    print(f"  {aspect}: Fmax={result['test_fmax']:.4f}, CAFA={result.get('cafa_fmax', 'N/A')}, Gates=({result['test_gate1']:.3f}, {result['test_gate2']:.3f})")
    
    return all_results


if __name__ == "__main__":
    results = run_ablation_study()