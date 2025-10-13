"""Data loading and preprocessing."""

import numpy as np
import pandas as pd
from utils.embeddings import precompute_text_embeddings, precompute_esm_embeddings


def load_data(config):
    """
    Load benchmark splits and protein data.
    
    Returns:
        Dict containing splits, GO terms, and protein data
    """
    print(f"\nLoading {config.aspect} benchmark (similarity={config.similarity_threshold}%)")
    
    if not config.split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {config.split_dir}")
    
    # Load splits
    train_df = pd.read_csv(config.split_dir / "train.tsv", sep='\t')
    val_df = pd.read_csv(config.split_dir / "val.tsv", sep='\t')
    test_df = pd.read_csv(config.split_dir / "test.tsv", sep='\t')
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load ProtAD
    protad_df = pd.read_csv(config.protad_path, sep='\t')
    protad_dict = protad_df.set_index('Entry').to_dict('index')
    
    # Extract sequences for ESM
    sequences_dict = protad_df.set_index('Entry')['Sequence'].to_dict()
    
    print(f"  ProtAD entries: {len(protad_dict)}")
    
    # Collect all GO terms
    all_go_terms = set()
    for df in [train_df, val_df, test_df]:
        for go_str in df['go_terms']:
            if pd.notna(go_str) and go_str.strip():
                all_go_terms.update(go_str.split(','))
    
    go_terms = sorted(list(all_go_terms))
    go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    
    # Debug mode: reduce dataset size
    if config.debug_mode:
        train_df = train_df.head(100)
        val_df = val_df.head(20)
        test_df = test_df.head(20)
        go_terms = go_terms[:300]
        go_to_idx = {go: idx for idx, go in enumerate(go_terms)}
    
    print(f"  GO terms: {len(go_terms)}")
    
    # Process each split
    def process_split(df, split_name):
        proteins = []
        labels = []
        
        for _, row in df.iterrows():
            protein_id = row['protein_id']
            
            if protein_id not in protad_dict:
                continue
            
            # Create label vector
            label_vec = np.zeros(len(go_terms), dtype=np.float32)
            go_str = row['go_terms']
            
            if pd.notna(go_str) and go_str.strip():
                for go_term in go_str.split(','):
                    go_term = go_term.strip()
                    if go_term in go_to_idx:
                        label_vec[go_to_idx[go_term]] = 1.0
            
            proteins.append(protein_id)
            labels.append(label_vec)
        
        print(f"  {split_name}: {len(proteins)} proteins")
        return proteins, np.array(labels)
    
    train_proteins, train_labels = process_split(train_df, "Train")
    val_proteins, val_labels = process_split(val_df, "Val")
    test_proteins, test_labels = process_split(test_df, "Test")
    
    # Precompute embeddings
    all_proteins = train_proteins + val_proteins + test_proteins
    precompute_text_embeddings(config, protad_dict, all_proteins)
    precompute_esm_embeddings(config, sequences_dict, all_proteins)
    
    return {
        'train': (train_proteins, train_labels),
        'val': (val_proteins, val_labels),
        'test': (test_proteins, test_labels),
        'go_terms': go_terms,
        'num_go_terms': len(go_terms)
    }