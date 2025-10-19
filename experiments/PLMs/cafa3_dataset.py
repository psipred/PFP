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
        emb_data = np.load(emb_file, allow_pickle=True).item()
        
        # Extract mean-pooled embedding
        if isinstance(emb_data, dict):
            embedding = torch.FloatTensor(emb_data['embedding'])
        else:
            embedding = torch.FloatTensor(emb_data)
        
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