"""Dataset for pairwise modality fusion."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class PairwiseModalityDataset(Dataset):
    """Dataset loading two modalities."""
    
    def __init__(self, data_dir, embedding_dirs, modality_pair, aspect, split, cache_embeddings=True):
        self.data_dir = Path(data_dir)
        self.aspect = aspect
        self.split = split
        self.cache_embeddings = cache_embeddings
        self.modality_pair = modality_pair
        
        # Parse modality names
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        self.mod1_dir = Path(embedding_dirs[mod1])
        self.mod2_dir = Path(embedding_dirs[mod2])
        
        # Load protein names and labels
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split} ({modality_pair})")
        
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching embeddings in memory...")
            self._cache_all_embeddings()
    
    def _load_embedding(self, emb_file):
        """Load embedding from file."""
        try:
            embedding = np.load(emb_file, allow_pickle=False)
        except (ValueError, OSError):
            data = np.load(emb_file, allow_pickle=True)
            
            if isinstance(data, dict):
                if 'embedding' in data:
                    embedding = data['embedding']
                elif 'embeddings' in data:
                    embedding = data['embeddings']
                elif 'mean' in data:
                    embedding = data['mean']
                else:
                    embedding = next(v for v in data.values() if isinstance(v, np.ndarray))
            elif isinstance(data, np.ndarray):
                embedding = data
                if embedding.dtype == object:
                    if embedding.shape == () or embedding.shape == (1,):
                        embedding = embedding.item()
                        if isinstance(embedding, dict):
                            embedding = next(v for v in embedding.values() if isinstance(v, np.ndarray))
            else:
                raise ValueError(f"Unexpected type: {type(data)}")
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.dtype not in [np.float32, np.float64]:
            embedding = embedding.astype(np.float32)
        
        if embedding.ndim > 1:
            if embedding.shape[0] == 1 or embedding.shape[1] == 1:
                embedding = embedding.squeeze()
            elif embedding.shape[0] > embedding.shape[1]:
                embedding = embedding.mean(axis=0)
        
        return embedding
    
    def _cache_all_embeddings(self):
        """Load all embeddings into memory."""
        missing = {self.mod1_name: [], self.mod2_name: []}
        
        for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="Loading embeddings")):
            mod1_file = self.mod1_dir / f"{protein_id}.npy"
            mod2_file = self.mod2_dir / f"{protein_id}.npy"
            
            try:
                mod1_emb = torch.FloatTensor(self._load_embedding(mod1_file))
                mod2_emb = torch.FloatTensor(self._load_embedding(mod2_file))
                
                self.embedding_cache[idx] = {
                    self.mod1_name: mod1_emb,
                    self.mod2_name: mod2_emb
                }
            except Exception as e:
                if not mod1_file.exists():
                    missing[self.mod1_name].append(protein_id)
                if not mod2_file.exists():
                    missing[self.mod2_name].append(protein_id)
        
        for modality, ids in missing.items():
            if ids:
                print(f"WARNING: {len(ids)} {modality} embeddings missing")
        
        print(f"Successfully cached {len(self.embedding_cache)}/{len(self.protein_ids)} proteins")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        if idx not in self.embedding_cache:
            raise RuntimeError(f"Embedding for index {idx} not in cache")
        
        embeddings = self.embedding_cache[idx]
        
        return {
            'mod1': embeddings[self.mod1_name],
            'mod2': embeddings[self.mod2_name],
            'labels': self.labels[idx],
            'protein_id': self.protein_ids[idx]
        }


def collate_fn(batch):
    """Collate function for pairwise modality dataset."""
    mod1 = torch.stack([b['mod1'] for b in batch])
    mod2 = torch.stack([b['mod2'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    protein_ids = [b['protein_id'] for b in batch]
    
    return {
        'mod1': mod1,
        'mod2': mod2,
        'labels': labels,
        'protein_ids': protein_ids
    }