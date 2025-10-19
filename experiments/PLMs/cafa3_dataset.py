"""Dataset class for CAFA3 PLM embeddings."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path


"""Dataset class for CAFA3 PLM embeddings."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path


class CAFA3PLMDataset(Dataset):
    """Dataset for CAFA3 with precomputed PLM embeddings."""
    
    def __init__(self, data_dir, embedding_dir, aspect, split, plm_type='esm'):
        """
        Args:
            data_dir: Directory with CAFA3 data files
            embedding_dir: Directory with PLM embeddings
            aspect: 'BPO', 'CCO', or 'MFO'
            split: 'train', 'valid', or 'test'
            plm_type: 'esm', 'prott5', or 'prostt5'
        """
        self.data_dir = Path(data_dir)
        self.embedding_dir = Path(embedding_dir) / plm_type
        self.aspect = aspect
        self.split = split
        self.plm_type = plm_type
        
        # Load protein names
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        # Load labels (sparse matrix)
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split}")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        
        # Load embedding
        emb_file = self.embedding_dir / f"{protein_id}.npy"
        
        if not emb_file.exists():
            raise FileNotFoundError(f"Embedding not found: {emb_file}")
        
        # Load embedding data
        emb_data = np.load(emb_file, allow_pickle=True)
        
        # Handle different formats
        if isinstance(emb_data, np.ndarray) and emb_data.dtype == object:
            # It's a 0-d array containing a dict
            emb_data = emb_data.item()
        
        # Extract embedding
        if isinstance(emb_data, dict):
            # Check if 'embedding' key exists
            if 'embedding' in emb_data:
                embedding = emb_data['embedding']
            else:
                # Fallback: look for other keys
                raise ValueError(f"No 'embedding' key found in {emb_file}")
        else:
            # It's directly a numpy array
            embedding = emb_data
        
        # Convert to tensor
        embedding = torch.FloatTensor(embedding)
        
        # Handle different shapes
        if len(embedding.shape) == 2:
            # It's a 2D per-residue embedding [seq_len, hidden_dim]
            # Mean pool over sequence dimension
            embedding = embedding.mean(dim=0)  # [hidden_dim]
        elif len(embedding.shape) == 1:
            # Already pooled - good!
            pass
        else:
            raise ValueError(f"Unexpected embedding shape for {protein_id}: {embedding.shape}")
        
        # Verify final shape based on PLM type
        expected_dim = {
            'esm': 1280,
            'prott5': 1024,
            'prostt5': 1024
        }[self.plm_type]
        
        if embedding.shape[0] != expected_dim:
            raise ValueError(f"Expected embedding dim {expected_dim} for {self.plm_type}, "
                           f"got {embedding.shape[0]} for {protein_id}")
        
        return {
            'embedding': embedding,
            'labels': self.labels[idx],
            'protein_id': protein_id
        }


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