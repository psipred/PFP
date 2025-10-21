"""Dataset class for CAFA3 PLM embeddings - Handles dict format."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class CAFA3PLMDataset(Dataset):
    """Dataset for CAFA3 with precomputed PLM embeddings and in-memory caching."""
    
    def __init__(self, data_dir, embedding_dir, aspect, split, plm_type='esm', cache_embeddings=True):
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
        self.expected_dim_dict = {
            'esm': 1280,
            'prott5': 1024,
            'prostt5': 1024,
            'ankh': 768,
            'text': None  # Will be determined from first embedding
        }
        self.expected_dim = self.expected_dim_dict[self.plm_type]
        
        # Cache embeddings in memory if requested
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching {len(self.protein_ids)} embeddings in memory...")
            self._cache_all_embeddings()
    
    def _load_embedding(self, emb_file):
        """Load embedding with proper handling of various formats."""
        try:
            # Try loading without pickle first
            embedding = np.load(emb_file, allow_pickle=False)
        except (ValueError, OSError):
            # Load with pickle (handles object arrays, dicts, etc.)
            data = np.load(emb_file, allow_pickle=True)
            
            # Handle different formats
            if isinstance(data, dict):
                # Dictionary format - extract embedding
                if 'embedding' in data:
                    embedding = data['embedding']
                elif 'embeddings' in data:
                    embedding = data['embeddings']
                elif 'mean' in data:
                    embedding = data['mean']
                elif 'vector' in data:
                    embedding = data['vector']
                else:
                    # Take first array value
                    for v in data.values():
                        if isinstance(v, np.ndarray):
                            embedding = v
                            break
                    else:
                        raise ValueError(f"Could not find embedding in dict keys: {list(data.keys())}")
            elif isinstance(data, np.ndarray):
                embedding = data
                # Handle object dtype
                if embedding.dtype == object:
                    if embedding.shape == () or embedding.shape == (1,):
                        embedding = embedding.item()
                        if isinstance(embedding, np.ndarray):
                            pass
                        elif isinstance(embedding, dict):
                            # Nested dict
                            for v in embedding.values():
                                if isinstance(v, np.ndarray):
                                    embedding = v
                                    break
                        else:
                            embedding = np.array(embedding, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected type: {type(data)}")
        
        # Ensure numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Ensure float dtype
        if embedding.dtype not in [np.float32, np.float64]:
            embedding = embedding.astype(np.float32)
        
        # Ensure 1D
        if embedding.ndim > 1:
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze()
            elif embedding.shape[1] == 1:
                embedding = embedding.squeeze()
            else:
                # Take mean over sequence dimension if it looks like [seq_len, dim]
                if embedding.shape[0] > embedding.shape[1]:
                    embedding = embedding.mean(axis=0)
                else:
                    raise ValueError(f"Unexpected shape: {embedding.shape}")
        
        return embedding
    
    def _cache_all_embeddings(self):
        """Load all embeddings into memory."""
        missing_embeddings = []
        error_embeddings = []
        
        for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="Loading embeddings")):
            emb_file = self.embedding_dir / f"{protein_id}.npy"
            
            if not emb_file.exists():
                missing_embeddings.append(protein_id)
                continue
            
            try:
                embedding = self._load_embedding(emb_file)
                embedding = torch.FloatTensor(embedding)
                
                # Auto-detect dimension for text embeddings
                if self.plm_type == 'text' and self.expected_dim is None:
                    self.expected_dim = embedding.shape[0]
                    print(f"Auto-detected text embedding dimension: {self.expected_dim}")
                
                # Verify shape
                if embedding.shape[0] != self.expected_dim:
                    raise ValueError(f"Expected dim {self.expected_dim}, got {embedding.shape[0]}")
                
                self.embedding_cache[idx] = embedding
                
            except Exception as e:
                print(f"Error loading {protein_id}: {e}")
                error_embeddings.append(protein_id)
        
        if missing_embeddings:
            print(f"WARNING: {len(missing_embeddings)} embeddings not found")
        if error_embeddings:
            print(f"WARNING: {len(error_embeddings)} embeddings failed to load")
        
        print(f"Successfully cached {len(self.embedding_cache)}/{len(self.protein_ids)} embeddings")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        
        if self.cache_embeddings:
            if idx not in self.embedding_cache:
                raise RuntimeError(f"Embedding for {protein_id} not in cache")
            embedding = self.embedding_cache[idx]
        else:
            emb_file = self.embedding_dir / f"{protein_id}.npy"
            if not emb_file.exists():
                raise FileNotFoundError(f"Embedding not found: {emb_file}")
            
            embedding = self._load_embedding(emb_file)
            embedding = torch.FloatTensor(embedding)
            
            if self.plm_type == 'text' and self.expected_dim is None:
                self.expected_dim = embedding.shape[0]
            
            if embedding.shape[0] != self.expected_dim:
                raise ValueError(f"Expected dim {self.expected_dim}, got {embedding.shape[0]}")
        
        return {
            'embedding': embedding,
            'labels': self.labels[idx],
            'protein_id': protein_id
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self.embedding_cache.clear()
    
    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return self.expected_dim


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