"""Dataset class for CAFA3 PLM embeddings with caching."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class CAFA3PLMDataset(Dataset):
    """Dataset for CAFA3 with precomputed PLM embeddings and in-memory caching."""
    
    def __init__(self, data_dir, embedding_dir, aspect, split, plm_type='esm', cache_embeddings=True):
        """
        Args:
            data_dir: Directory with CAFA3 data files
            embedding_dir: Directory with PLM embeddings
            aspect: 'BPO', 'CCO', or 'MFO'
            split: 'train', 'valid', or 'test'
            plm_type: 'esm', 'prott5', 'prostt5', 'ankh'
            cache_embeddings: If True, load all embeddings into memory at initialization
        """
        self.data_dir = Path(data_dir)
        self.embedding_dir = Path(embedding_dir) / plm_type
        self.aspect = aspect
        self.split = split
        self.plm_type = plm_type
        self.cache_embeddings = cache_embeddings
        
        # Load protein names
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        # Load labels (sparse matrix)
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split}")
        
        # Expected embedding dimension
        self.expected_dim = {
            'esm': 1280,
            'prott5': 1024,
            'prostt5': 1024,
            'ankh': 768
        }[self.plm_type]
        
        # Cache embeddings in memory if requested
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching {len(self.protein_ids)} embeddings in memory...")
            self._cache_all_embeddings()
    
    def _cache_all_embeddings(self):
        """Load all embeddings into memory."""
        missing_embeddings = []
        
        for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="Loading embeddings")):
            emb_file = self.embedding_dir / f"{protein_id}.npy"
            
            if not emb_file.exists():
                missing_embeddings.append(protein_id)
                continue
            
            try:
                # Load embedding - now it's just a simple numpy array
                embedding = np.load(emb_file)
                
                # Convert to tensor
                embedding = torch.FloatTensor(embedding)
                
                # Verify shape
                if embedding.shape[0] != self.expected_dim:
                    raise ValueError(f"Expected dim {self.expected_dim}, got {embedding.shape[0]} for {protein_id}")
                
                self.embedding_cache[idx] = embedding
                
            except Exception as e:
                print(f"Error loading {protein_id}: {e}")
                missing_embeddings.append(protein_id)
        
        if missing_embeddings:
            print(f"WARNING: {len(missing_embeddings)} embeddings not found")
            print(f"First few missing: {missing_embeddings[:5]}")
        
        print(f"Successfully cached {len(self.embedding_cache)}/{len(self.protein_ids)} embeddings")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        
        # Get embedding from cache or load from disk
        if self.cache_embeddings:
            if idx not in self.embedding_cache:
                raise RuntimeError(f"Embedding for {protein_id} not in cache")
            embedding = self.embedding_cache[idx]
        else:
            # Load from disk (slower)
            emb_file = self.embedding_dir / f"{protein_id}.npy"
            if not emb_file.exists():
                raise FileNotFoundError(f"Embedding not found: {emb_file}")
            
            embedding = torch.FloatTensor(np.load(emb_file))
            
            # Verify shape
            if embedding.shape[0] != self.expected_dim:
                raise ValueError(f"Expected dim {self.expected_dim}, got {embedding.shape[0]} for {protein_id}")
        
        return {
            'embedding': embedding,
            'labels': self.labels[idx],
            'protein_id': protein_id
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self.embedding_cache.clear()
        print(f"Cleared embedding cache for {self.aspect} {self.split}")


def collate_fn(batch):
    """Collate function for CAFA3 dataset."""
    embeddings = torch.stack([b['embedding'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    protein_ids = [b['protein_id'] for b in batch]
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'protein_ids': protein_ids
    }