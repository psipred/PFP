"""
Compute PubMedBERT text embeddings for extracted UniProt descriptions.

Input: protein_descriptions.tsv (from extract_uniprot_descriptions.py)
Output: Per-protein .npy files containing FP32 (Full Precision) embeddings.

Adapts the "Hybrid Compression" strategy:
- Since the input is a single rich text description, we treat it as the 
  high-value 'Function' field and save the Full Sequence embedding (hidden state).
- Saves in float32 (full precision) as .npy files.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys

# Configuration
DEFAULT_MODEL = "Microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512  # PubMedBERT limit

def get_cache_path(cache_dir, protein_id):
    """Get path for the embedding file."""
    # Save as .npy instead of .pt
    return cache_dir / f"{protein_id}.npy"

def save_embedding(cache_dir, protein_id, embedding):
    """Save embedding to disk safely as .npy."""
    path = get_cache_path(cache_dir, protein_id)
    try:
        # atomic writeish
        temp_path = path.with_suffix('.tmp.npy')
        np.save(temp_path, embedding)
        temp_path.rename(path)
    except Exception as e:
        print(f"Error saving {protein_id}: {e}")
        if temp_path.exists():
            temp_path.unlink()

def load_descriptions(tsv_path):
    """Load descriptions from TSV file."""
    prot_dict = {}
    if not tsv_path.exists():
        print(f"Error: Input file not found at {tsv_path}")
        sys.exit(1)
        
    with open(tsv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t', 1)
            if len(parts) >= 2:
                prot_dict[parts[0]] = parts[1]
    return prot_dict

def precompute_embeddings(data_dir, model_name=DEFAULT_MODEL):
    """
    Precompute text embeddings.
    Adapts reference logic: Full sequence + fp32 for the description.
    """
    # Setup paths
    data_dir = Path(data_dir)
    input_file = data_dir / "embedding_cache" / "uniprot_text" / "protein_descriptions.tsv"
    output_dir = data_dir / "embedding_cache" / "exp_text_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading descriptions from: {input_file}")
    prot_dict = load_descriptions(input_file)
    all_ids = list(prot_dict.keys())
    print(f"Found {len(all_ids)} proteins.")

    # Filter cached
    proteins_to_encode = [
        p for p in all_ids 
        if not get_cache_path(output_dir, p).exists()
    ]
    
    if not proteins_to_encode:
        print(f"✓ All {len(all_ids)} proteins already cached in {output_dir}")
        return

    print(f"Encoding {len(proteins_to_encode)} new proteins...")
    
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Processing Loop
    with torch.no_grad():
        for protein_id in tqdm(proteins_to_encode, desc="Text encoding"):
            text = prot_dict[protein_id]
            
            if text == '' or text == 'nan':
                text = 'None'
            
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length', # consistent tensor shapes
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Model Forward
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract Last Hidden State
            # Shape: [1, Sequence_Length, 768]
            hidden = output.last_hidden_state.cpu()
            
            # Convert to NumPy array (Full Precision Float32)
            embedding_np = hidden.numpy()
            
            save_embedding(output_dir, protein_id, embedding_np)
            
    print(f"✓ Text embeddings cached in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compute PubMedBERT embeddings for protein descriptions.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the 'data' directory (containing embedding_cache)")
    
    args = parser.parse_args()
    
    precompute_embeddings(args.data_dir)

if __name__ == "__main__":
    main()