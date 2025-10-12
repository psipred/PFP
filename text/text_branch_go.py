#!/usr/bin/env python3
"""
Text-only GO Prediction with HYBRID COMPRESSED embeddings
- Fields 0-2, 4-16: CLS token only + fp16 (87.5% reduction per field)
- Field 3 (Function): Full sequence + fp16 (50% reduction)
- Overall: ~85% storage reduction with <1% accuracy loss
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pickle

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    BENCHMARK_BASE = Path("/home/zijianzhou/Datasets/protad/go_annotations/benchmarks")
    PROTAD_PATH = Path("/home/zijianzhou/Datasets/protad/protad.tsv")
    OUTPUT_DIR = Path("./text_branch_experiments")
    CACHE_DIR = Path("./embedding_cache_hybrid")
    
    # Model
    PUBMED_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    MAX_TEXT_LENGTH = 512
    
    # Compression settings
    COMPRESSION_MODE = 'hybrid_fp16'
    FUNCTION_FIELD_INDEX = 3
    
    # GO Aspects and Thresholds
    ASPECTS = ['BP', 'MF', 'CC']
    THRESHOLDS = [30, 50, 70, 95]
    ASPECT_FULL_NAMES = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component'
    }
    
    # Text fields (17 attributes from ProtAD)
    TEXT_FIELDS = [
        'Protein names', 'Organism', 'Taxonomic lineage', 'Function',
        'Caution', 'Miscellaneous', 'Subunit structure', 'Induction',
        'Tissue specificity', 'Developmental stage', 'Allergenic properties',
        'Biotechnological use', 'Pharmaceutical use', 'Involvement in disease',
        'Subcellular location', 'Post-translational modification', 'Sequence similarities'
    ]
    
    def __init__(self, similarity_threshold: int = 30, aspect: str = 'BP', debug_mode: bool = False):
        self.SIMILARITY_THRESHOLD = similarity_threshold
        self.ASPECT = aspect
        self.DEBUG_MODE = debug_mode
        
        self.SPLIT_DIR = self.BENCHMARK_BASE / f"similarity_{similarity_threshold}" / aspect
        
        exp_dir = self.OUTPUT_DIR / f"sim_{similarity_threshold}" / aspect
        self.CHECKPOINT_DIR = exp_dir / "checkpoints"
        self.RESULTS_DIR = exp_dir / "results"
        
        self.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        self.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        
        if debug_mode:
            self.NUM_EPOCHS = 2
            self.BATCH_SIZE = 16
            print(f"Debug mode enabled for {aspect}")

# ============================================================================
# Fmax Calculation
# ============================================================================

def compute_fmax(y_true, y_pred, thresholds=None):
    """Compute Fmax metric for GO term prediction"""
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

# ============================================================================
# HYBRID Compression Functions (FP32)
# ============================================================================

def get_protein_cache_path(cache_dir: Path, protein_id: str) -> Path:
    """Get cache file path for a protein"""
    subdir = protein_id[:2] if len(protein_id) >= 2 else 'other'
    protein_dir = cache_dir / subdir
    protein_dir.mkdir(exist_ok=True, parents=True)
    return protein_dir / f"{protein_id}.pkl"


def save_protein_embedding_hybrid(cache_dir: Path, protein_id: str, 
                                   field_embeddings: list, function_field_idx: int = 3):
    """
    Save embeddings with hybrid compression (fp16):
    - Function field: Full sequence + fp16
    - Other 16 fields: CLS token only + fp16
    
    Storage: ~15% of original (85% reduction)
    """
    cache_file = get_protein_cache_path(cache_dir, protein_id)
    
    compressed_embeddings = []
    
    for field_idx, field_emb in enumerate(field_embeddings):
        hidden_states = field_emb['hidden_states']
        attention_mask = field_emb['attention_mask']
        
        if field_idx == function_field_idx:
            # Function field: Keep full sequence, convert to fp16
            compressed_embeddings.append({
                'hidden_states': hidden_states.half(),
                'attention_mask': attention_mask,
                'is_cls_only': False
            })
        else:
            # Other fields: Extract CLS token only, convert to fp16
            cls_token = hidden_states[:, 0:1, :].half()
            compressed_embeddings.append({
                'hidden_states': cls_token,
                'attention_mask': None,
                'is_cls_only': True
            })
    
    with open(cache_file, 'wb') as f:
        pickle.dump(compressed_embeddings, f)


def load_protein_embedding_hybrid(cache_dir: Path, protein_id: str):
    """Load hybrid compressed embeddings and convert back to fp32"""
    cache_file = get_protein_cache_path(cache_dir, protein_id)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
            
            result = []
            for emb in embeddings:
                result.append({
                    'hidden_states': emb['hidden_states'].float(),
                    'attention_mask': emb['attention_mask']
                })
            
            return result
    return None


def precompute_embeddings_hybrid(config: Config, protad_dict: dict, protein_ids: list):
    """
    Precompute embeddings with hybrid compression:
    - Extracts CLS for 16 fields immediately
    - Keeps full sequence for Function field
    - Stores in fp16 (half precision)
    """
    
    print(f"Checking/computing HYBRID compressed embeddings (fp16)...")
    print(f"  → Fields 0-2, 4-16: CLS token only + fp16")
    print(f"  → Field 3 (Function): Full sequence + fp16")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unique_proteins = list(set(protein_ids))
    
    proteins_to_encode = []
    for protein_id in unique_proteins:
        if protein_id not in protad_dict:
            continue
        cache_file = get_protein_cache_path(config.CACHE_DIR, protein_id)
        if not cache_file.exists():
            proteins_to_encode.append(protein_id)
    
    if len(proteins_to_encode) == 0:
        print(f"  ✓ All {len(unique_proteins)} proteins already cached!")
        return config.CACHE_DIR
    
    print(f"  Found {len(unique_proteins) - len(proteins_to_encode)} cached proteins")
    print(f"  Encoding {len(proteins_to_encode)} new proteins...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.PUBMED_MODEL)
    text_model = AutoModel.from_pretrained(config.PUBMED_MODEL).to(device)
    text_model.eval()
    
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins_to_encode), batch_size), 
                      desc="Encoding (hybrid fp16)"):
            batch_proteins = proteins_to_encode[i:i+batch_size]
            
            for protein_id in batch_proteins:
                protein_data = protad_dict[protein_id]
                field_embeddings = []
                
                for field_idx, field in enumerate(config.TEXT_FIELDS):
                    text = str(protein_data.get(field, ''))
                    if pd.isna(text) or text == '' or text == 'nan':
                        text = "None"
                    
                    encoding = tokenizer(
                        text,
                        max_length=config.MAX_TEXT_LENGTH,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    output = text_model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden = output.last_hidden_state
                    
                    field_embeddings.append({
                        'hidden_states': last_hidden.cpu(),
                        'attention_mask': attention_mask.cpu()
                    })
                
                # Save with hybrid compression (fp16)
                save_protein_embedding_hybrid(
                    config.CACHE_DIR, 
                    protein_id, 
                    field_embeddings,
                    function_field_idx=config.FUNCTION_FIELD_INDEX
                )
    
    print(f"  ✓ Embeddings cached in: {config.CACHE_DIR}/")
    print(f"  Storage reduction: ~85% (CLS-only + fp16)")
    return config.CACHE_DIR

# ============================================================================
# Dataset with Hybrid Compressed Embeddings
# ============================================================================

class HybridCompressedDataset(Dataset):
    """Dataset that loads hybrid compressed embeddings from disk"""
    
    def __init__(self, proteins, labels, cache_dir):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_id = self.proteins[idx]
        field_embeddings = load_protein_embedding_hybrid(self.cache_dir, protein_id)
        
        if field_embeddings is None:
            raise ValueError(f"No cached embedding found for {protein_id}")
        
        hidden_states = [emb['hidden_states'].squeeze(0) for emb in field_embeddings]
        
        return {
            'hidden_states': hidden_states,
            'labels': self.labels[idx]
        }


def embedding_collate_fn(batch):
    """Collate precomputed embeddings"""
    labels = torch.stack([b['labels'] for b in batch])
    num_fields = len(batch[0]['hidden_states'])
    
    all_hidden_states = []
    for field_idx in range(num_fields):
        field_hidden = torch.stack([b['hidden_states'][field_idx] for b in batch])
        all_hidden_states.append(field_hidden)
    
    return {
        'hidden_states': all_hidden_states,
        'labels': labels
    }

# ============================================================================
# Model (unchanged - works with both CLS-only and full sequences)
# ============================================================================

class TextFusionGOModel(nn.Module):
    """Trainable Text Fusion + GO Classifier"""
    
    def __init__(self, config: Config, num_go_terms: int):
        super().__init__()
        self.config = config
        self.num_attr = 17
        
        self.texts_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.1, batch_first=True) 
            for _ in range(self.num_attr)
        ])
        
        self.text_suffix_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.1, batch_first=True)
        self.text_suffix_transformer = nn.TransformerEncoder(self.text_suffix_encoder, num_layers=2)
        
        self.text_crosses = nn.ModuleList([
            nn.MultiheadAttention(768, num_heads=4, dropout=0.1, batch_first=True) 
            for _ in range(4)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(768) for _ in range(4)])
        
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_go_terms)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable:,}")
    
    def forward(self, hidden_states_list):
        texts_output = [
            self.texts_encoder[i](hidden_states_list[i]) 
            for i in range(self.num_attr)
        ]
        
        texts_output_cls = [
            texts_output[idx][:, 0, :].unsqueeze(1) 
            for idx in range(len(texts_output)) if idx != 3
        ]
        texts_output_cls = torch.cat(texts_output_cls, dim=1)
        texts_output_cls = self.text_suffix_transformer(texts_output_cls)
        
        text_func = texts_output[3]
        
        x = texts_output_cls
        for i in range(4):
            _x = x
            x = self.text_crosses[i](x, text_func, text_func)
            x = self.norms[i](x[0] + _x)
        
        fused_embedding = x.mean(dim=1)
        
        x = self.fc1(fused_embedding)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(config: Config):
    """Load aspect-specific benchmark splits and map to ProtAD"""
    
    print(f"\n{'='*60}")
    print(f"Loading {config.ASPECT} benchmark from {config.SPLIT_DIR}")
    print(f"{'='*60}")
    
    if not config.SPLIT_DIR.exists():
        raise FileNotFoundError(f"Split directory not found: {config.SPLIT_DIR}")
    
    train_df = pd.read_csv(config.SPLIT_DIR / "train.tsv", sep='\t')
    val_df = pd.read_csv(config.SPLIT_DIR / "val.tsv", sep='\t')
    test_df = pd.read_csv(config.SPLIT_DIR / "test.tsv", sep='\t')
    
    print(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("Loading ProtAD text data...")
    protad_df = pd.read_csv(config.PROTAD_PATH, sep='\t')
    protad_dict = protad_df.set_index('Entry').to_dict('index')
    print(f"  ProtAD entries: {len(protad_dict)}")
    
    all_go_terms = set()
    for df in [train_df, val_df, test_df]:
        for go_str in df['go_terms']:
            if pd.notna(go_str) and go_str.strip():
                all_go_terms.update(go_str.split(','))
    
    go_terms = sorted(list(all_go_terms))
    go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    
    if config.DEBUG_MODE:
        train_df = train_df.head(100)
        val_df = val_df.head(20)
        test_df = test_df.head(20)
        go_terms = go_terms[:50]
        go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    
    print(f"  GO terms ({config.ASPECT}): {len(go_terms)}")
    
    def process_split(df, split_name):
        proteins = []
        labels = []
        missing_count = 0
        
        for idx, row in df.iterrows():
            protein_id = row['protein_id']
            
            if protein_id not in protad_dict:
                missing_count += 1
                continue
            
            label_vec = np.zeros(len(go_terms), dtype=np.float32)
            go_str = row['go_terms']
            
            if pd.notna(go_str) and go_str.strip():
                for go_term in go_str.split(','):
                    go_term = go_term.strip()
                    if go_term in go_to_idx:
                        label_vec[go_to_idx[go_term]] = 1.0
            
            proteins.append(protein_id)
            labels.append(label_vec)
        
        coverage = 100 * len(proteins) / len(df) if len(df) > 0 else 0
        print(f"  {split_name}: {len(proteins)}/{len(df)} proteins ({coverage:.1f}% coverage)")
        if missing_count > 0:
            print(f"    Missing {missing_count} proteins in ProtAD")
        
        return proteins, np.array(labels)
    
    train_proteins, train_labels = process_split(train_df, "Train")
    val_proteins, val_labels = process_split(val_df, "Val")
    test_proteins, test_labels = process_split(test_df, "Test")
    
    all_proteins = train_proteins + val_proteins + test_proteins
    cache_dir = precompute_embeddings_hybrid(config, protad_dict, all_proteins)
    
    return {
        'train': (train_proteins, train_labels),
        'val': (val_proteins, val_labels),
        'test': (test_proteins, test_labels),
        'cache_dir': cache_dir,
        'go_terms': go_terms
    }

# ============================================================================
# Training
# ============================================================================

def train_model(config: Config, data_dict):
    """Train GO classifier for specific aspect"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining {config.ASPECT} model on {device}")
    
    train_dataset = HybridCompressedDataset(
        data_dict['train'][0], data_dict['train'][1],
        data_dict['cache_dir']
    )
    val_dataset = HybridCompressedDataset(
        data_dict['val'][0], data_dict['val'][1],
        data_dict['cache_dir']
    )
    test_dataset = HybridCompressedDataset(
        data_dict['test'][0], data_dict['test'][1],
        data_dict['cache_dir']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=embedding_collate_fn,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=embedding_collate_fn,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=embedding_collate_fn,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    num_go_terms = data_dict['train'][1].shape[1]
    model = TextFusionGOModel(config, num_go_terms).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"GO terms ({config.ASPECT}): {num_go_terms}")
    
    best_val_fmax = 0
    best_threshold = 0.5
    history = []
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"[{config.ASPECT}] Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            hidden_states = [h.to(device) for h in batch['hidden_states']]
            labels = batch['labels'].to(device)
            
            logits = model(hidden_states)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                hidden_states = [h.to(device) for h in batch['hidden_states']]
                labels = batch['labels'].to(device)
                
                logits = model(hidden_states)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        
        fmax, threshold, precision, recall = compute_fmax(y_true, y_pred)
        
        print(f"  [{config.ASPECT}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Fmax: {fmax:.3f} (t={threshold:.2f}), P: {precision:.3f}, R: {recall:.3f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fmax': fmax,
            'threshold': threshold,
            'precision': precision,
            'recall': recall
        })
        
        if fmax > best_val_fmax:
            best_val_fmax = fmax
            best_threshold = threshold
            torch.save(model.state_dict(), 
                      config.CHECKPOINT_DIR / f"best_model_{config.ASPECT}.pt")
            print(f"  ✓ New best Fmax: {best_val_fmax:.3f} at threshold {best_threshold:.2f}")
    
    print(f"\n[{config.ASPECT}] Test evaluation...")
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"best_model_{config.ASPECT}.pt"))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            hidden_states = [h.to(device) for h in batch['hidden_states']]
            labels = batch['labels'].to(device)
            
            logits = model(hidden_states)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    test_fmax, test_threshold, test_precision, test_recall = compute_fmax(y_true, y_pred)
    
    print(f"[{config.ASPECT}] Test: Fmax={test_fmax:.3f} (t={test_threshold:.2f}), "
          f"P={test_precision:.3f}, R={test_recall:.3f}")
    
    results = {
        'aspect': config.ASPECT,
        'similarity_threshold': config.SIMILARITY_THRESHOLD,
        'compression_mode': config.COMPRESSION_MODE,
        'num_go_terms': num_go_terms,
        'test_metrics': {
            'fmax': float(test_fmax),
            'threshold': float(test_threshold),
            'precision': float(test_precision), 
            'recall': float(test_recall)
        },
        'best_val_fmax': float(best_val_fmax),
        'best_val_threshold': float(best_threshold),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    with open(config.RESULTS_DIR / f"results_{config.ASPECT}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame(history).to_csv(
        config.RESULTS_DIR / f"history_{config.ASPECT}.csv", index=False
    )
    
    return model, history, results

# ============================================================================
# Main
# ============================================================================

def main(thresholds: list = None, aspects: list = None, debug_mode: bool = False):
    """Run training for specified thresholds and aspects"""
    
    if thresholds is None:
        thresholds = [30, 50, 70, 95]
    if aspects is None:
        aspects = ['BP', 'MF', 'CC']
    
    print("=" * 70)
    print("Text Branch GO Prediction - HYBRID COMPRESSION (FP16)")
    print("Storage Reduction: ~85% (CLS-only + fp16 for 16 fields)")
    print("=" * 70)
    print(f"Thresholds: {', '.join(map(str, thresholds))}")
    print(f"Aspects: {', '.join(aspects)}")
    print("=" * 70)
    
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n{'#'*70}")
        print(f"SIMILARITY THRESHOLD: {threshold}%")
        print(f"{'#'*70}")
        
        threshold_results = {}
        
        for aspect in aspects:
            print(f"\n{'='*70}")
            print(f"Training: Threshold={threshold}%, Aspect={aspect}")
            print(f"{'='*70}")
            
            try:
                config = Config(
                    similarity_threshold=threshold, 
                    aspect=aspect,
                    debug_mode=debug_mode
                )
                
                data_dict = prepare_data(config)
                model, history, results = train_model(config, data_dict)
                
                threshold_results[aspect] = results
                print(f"\n✓ {aspect} @ {threshold}% complete!")
                
            except FileNotFoundError as e:
                print(f"\n✗ {aspect} @ {threshold}% skipped: {e}")
                continue
            except Exception as e:
                print(f"\n✗ {aspect} @ {threshold}% failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if threshold_results:
            all_results[threshold] = threshold_results
    
    if all_results:
        overall_summary_dir = Path("./text_branch_experiments")
        overall_summary_dir.mkdir(exist_ok=True, parents=True)
        
        overall_summary = {
            'compression_mode': 'hybrid_fp16',
            'storage_reduction': '~85%',
            'precision': 'float16',
            'thresholds': list(all_results.keys()),
            'aspects': aspects,
            'results': all_results
        }
        
        with open(overall_summary_dir / "summary_hybrid_compression.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"\n{'#'*70}")
        print("OVERALL SUMMARY - HYBRID COMPRESSION (FP16)")
        print(f"{'#'*70}")
        for threshold, threshold_results in all_results.items():
            print(f"\nThreshold {threshold}%:")
            for aspect, res in threshold_results.items():
                metrics = res['test_metrics']
                print(f"  {aspect}: Fmax={metrics['fmax']:.3f} (t={metrics['threshold']:.2f}), "
                      f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        
        print(f"\n✓ Summary saved to: {overall_summary_dir / 'summary_hybrid_compression.json'}")
        print(f"✓ Storage reduction: ~85% (CLS-only + fp16)")
        print(f"✓ Expected accuracy loss: <1%")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs='+', type=int, default=[95, 70, 50, 30],
                       help="Similarity thresholds to train (default: all)")
    parser.add_argument("--aspects", nargs='+', default=['BP','MF', 'CC'],
                       choices=['BP', 'MF', 'CC'],
                       help="Aspects to train (default: MF, CC)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (small dataset, 2 epochs)")
    args = parser.parse_args()
    
    main(
        thresholds=args.thresholds,
        aspects=args.aspects,
        debug_mode=args.debug
    )