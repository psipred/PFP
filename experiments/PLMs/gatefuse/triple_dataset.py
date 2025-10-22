"""Dataset for triple modality fusion."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class TripleModalityDataset(Dataset):
    """Dataset loading text, ProtT5, and ESM embeddings."""
    
    def __init__(self, data_dir, embedding_dirs, aspect, split, cache_embeddings=True):
        self.data_dir = Path(data_dir)
        self.aspect = aspect
        self.split = split
        self.cache_embeddings = cache_embeddings
        
        # Embedding directories
        self.text_dir = Path(embedding_dirs['text'])
        self.prott5_dir = Path(embedding_dirs['prott5'])
        self.esm_dir = Path(embedding_dirs['esm'])
        
        # Load protein names and labels
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split}")
        
        # Cache embeddings
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching embeddings in memory...")
            self._cache_all_embeddings()
    def _load_embedding(self, emb_file):
        """Load embedding from file."""
        try:
            # Try without pickle first
            embedding = np.load(emb_file, allow_pickle=False)
        except (ValueError, OSError):
            # Load with pickle for dict/object arrays
            data = np.load(emb_file, allow_pickle=True)
            
            # Handle dict format
            if isinstance(data, dict):
                if 'embedding' in data:
                    embedding = data['embedding']
                elif 'embeddings' in data:
                    embedding = data['embeddings']
                elif 'mean' in data:
                    embedding = data['mean']
                else:
                    # Take first array value
                    embedding = next(v for v in data.values() if isinstance(v, np.ndarray))
            elif isinstance(data, np.ndarray):
                embedding = data
                # Handle object dtype
                if embedding.dtype == object:
                    if embedding.shape == () or embedding.shape == (1,):
                        embedding = embedding.item()
                        if isinstance(embedding, dict):
                            embedding = next(v for v in embedding.values() if isinstance(v, np.ndarray))
            else:
                raise ValueError(f"Unexpected type: {type(data)}")
        
        # Ensure proper format
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.dtype not in [np.float32, np.float64]:
            embedding = embedding.astype(np.float32)
        
        # Ensure 1D
        if embedding.ndim > 1:
            if embedding.shape[0] == 1:
                embedding = embedding.squeeze()
            elif embedding.shape[1] == 1:
                embedding = embedding.squeeze()
            else:
                # Mean over sequence dimension if needed
                if embedding.shape[0] > embedding.shape[1]:
                    embedding = embedding.mean(axis=0)
        
        return embedding
    
    def _cache_all_embeddings(self):
        """Load all embeddings into memory."""
        missing = {'text': [], 'prott5': [], 'esm': []}

        for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="Loading embeddings")):
            text_file = self.text_dir / f"{protein_id}.npy"
            prott5_file = self.prott5_dir / f"{protein_id}.npy"
            esm_file = self.esm_dir / f"{protein_id}.npy"
            # exit(text_file)
            try:
                text_emb = torch.FloatTensor(self._load_embedding(text_file))
                prott5_emb = torch.FloatTensor(self._load_embedding(prott5_file))
                esm_emb = torch.FloatTensor(self._load_embedding(esm_file))
                
                self.embedding_cache[idx] = {
                    'text': text_emb,
                    'prott5': prott5_emb,
                    'esm': esm_emb
                }
            except Exception as e:
                if not text_file.exists():
                    missing['text'].append(protein_id)
                if not prott5_file.exists():
                    missing['prott5'].append(protein_id)
                if not esm_file.exists():
                    missing['esm'].append(protein_id)
        
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
            'text': embeddings['text'],
            'prott5': embeddings['prott5'],
            'esm': embeddings['esm'],
            'labels': self.labels[idx],
            'protein_id': self.protein_ids[idx]
        }


def collate_fn(batch):
    """Collate function for triple modality dataset."""
    text = torch.stack([b['text'] for b in batch])
    prott5 = torch.stack([b['prott5'] for b in batch])
    esm = torch.stack([b['esm'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    protein_ids = [b['protein_id'] for b in batch]
    
    return {
        'text': text,
        'prott5': prott5,
        'esm': esm,
        'labels': labels,
        'protein_ids': protein_ids
    }