"""Embedding computation and caching utilities."""

import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer


def get_cache_path(cache_dir: Path, protein_id: str, model_type: str) -> Path:
    """Get cache file path for a protein."""
    subdir = protein_id[:2] if len(protein_id) >= 2 else 'other'
    protein_dir = cache_dir / model_type / subdir
    protein_dir.mkdir(exist_ok=True, parents=True)
    return protein_dir / f"{protein_id}.pkl"


def load_embedding(cache_dir: Path, protein_id: str, model_type: str):
    """Load cached embedding."""
    cache_file = get_cache_path(cache_dir, protein_id, model_type)
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def save_embedding(cache_dir: Path, protein_id: str, model_type: str, embedding):
    """Save embedding to cache."""
    cache_file = get_cache_path(cache_dir, protein_id, model_type)
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding, f)


def precompute_text_embeddings(config, protad_dict, protein_ids):
    """
    Precompute text embeddings with hybrid compression.
    - Function field: Full sequence + fp16
    - Other fields: CLS token only + fp16
    """
    print("\nPrecomputing text embeddings (hybrid compression)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unique_proteins = list(set(protein_ids))
    proteins_to_encode = [
        p for p in unique_proteins 
        if p in protad_dict and not get_cache_path(config.cache_dir, p, 'text').exists()
    ]
    
    if not proteins_to_encode:
        print(f"  ✓ All {len(unique_proteins)} proteins cached")
        return
    
    print(f"  Encoding {len(proteins_to_encode)} new proteins...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.pubmed_model)
    model = AutoModel.from_pretrained(config.pubmed_model).to(device)
    model.eval()
    
    with torch.no_grad():
        for protein_id in tqdm(proteins_to_encode, desc="Text encoding"):
            protein_data = protad_dict[protein_id]
            field_embeddings = []
            
            for field_idx, field in enumerate(config.text_fields):
                text = str(protein_data.get(field, 'None'))
                if text == '' or text == 'nan':
                    text = 'None'
                
                encoding = tokenizer(
                    text,
                    max_length=config.max_text_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                output = model(
                    input_ids=encoding['input_ids'].to(device),
                    attention_mask=encoding['attention_mask'].to(device)
                )
                
                hidden = output.last_hidden_state.cpu()
                
                # Keep full sequence for Function field (idx=3), CLS only for others
                if field_idx == 3:
                    field_embeddings.append(hidden.half())
                else:
                    field_embeddings.append(hidden[:, 0:1, :].half())
            
            save_embedding(config.cache_dir, protein_id, 'text', field_embeddings)
    
    print(f"  ✓ Text embeddings cached")


def precompute_esm_embeddings(config, sequences_dict, protein_ids):
    """
    Precompute ESM embeddings with proper mean-pooling.
    
    Excludes [CLS] and [EOS] tokens from mean-pooling to get accurate
    protein-level representations. Uses full precision (fp32) for 
    protein function prediction tasks.
    
    Args:
        sequences_dict: Dict mapping protein_id -> amino acid sequence
    """
    print("\nPrecomputing ESM embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unique_proteins = list(set(protein_ids))
    proteins_to_encode = [
        p for p in unique_proteins 
        if p in sequences_dict and not get_cache_path(config.cache_dir, p, 'esm').exists()
    ]
    
    if not proteins_to_encode:
        print(f"  ✓ All {len(unique_proteins)} proteins cached")
        return
    
    print(f"  Encoding {len(proteins_to_encode)} new proteins...")
    
    tokenizer = EsmTokenizer.from_pretrained(config.esm_model)
    model = EsmModel.from_pretrained(config.esm_model).to(device)
    model.eval()
    
    batch_size = 4  # ESM is memory-intensive
    
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins_to_encode), batch_size), desc="ESM encoding"):
            batch_proteins = proteins_to_encode[i:i+batch_size]
            batch_sequences = [sequences_dict[p] for p in batch_proteins]
            
            # Tokenize
            encoded = tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            
            # Get embeddings
            outputs = model(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device)
            )
            
            embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            for j, protein_id in enumerate(batch_proteins):
                # Get sequence length (excluding special tokens)
                seq_len = encoded['attention_mask'][j].sum().item()
                
                # Extract only the amino acid embeddings (exclude [CLS] at position 0 and [EOS] at end)
                # ESM format: [CLS] + sequence + [EOS] + [PAD]...
                aa_embeddings = embeddings[j, 1:seq_len-1, :]  # Exclude first ([CLS]) and last valid token ([EOS])
                
                # Mean pool over amino acids only
                pooled = aa_embeddings.mean(dim=0)  # [hidden_dim]
                
                # Save in full precision (fp32)
                save_embedding(config.cache_dir, protein_id, 'esm', pooled.cpu())
    
    print(f"  ✓ ESM embeddings cached")