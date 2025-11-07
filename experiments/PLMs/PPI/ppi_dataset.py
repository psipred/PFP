# experiments/PLMs/gatefuse/ppi_dataset.py
"""Dataset with optional PPI embeddings - using pre-extracted files."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class PairwiseWithPPIDataset(Dataset):
    """Dataset loading two modalities + optional PPI (pre-extracted)."""
    
    def __init__(self, data_dir, embedding_dirs, modality_pair, aspect, split, 
                 use_ppi=False, cache_embeddings=True):
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
        
        # PPI directory (pre-extracted)
        self.use_ppi = use_ppi
        self.ppi_dim = 512
        if use_ppi:
            self.ppi_dir = Path(embedding_dirs.get('ppi', 
                Path(__file__).parent.parent / 'embedding_cache' / 'ppi'))
        
        # Load protein names and labels
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split} ({modality_pair})")
        
        # Cache embeddings
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching embeddings in memory...")
            self._cache_all_embeddings()
    
    def _load_embedding(self, emb_file):
        """Load embedding from .npy file."""
        data = np.load(emb_file, allow_pickle=True)
        
        if hasattr(data, 'files'):  # .npz
            for k in ('embedding', 'embeddings', 'mean', 'avg', 'pooled'):
                if k in data.files:
                    embedding = data[k]
                    break
            else:
                embedding = data[data.files[0]]
        else:
            embedding = data
        
        # Handle object arrays
        if isinstance(embedding, np.ndarray) and embedding.dtype == object:
            if embedding.shape == () or embedding.shape == (1,):
                embedding = embedding.item()
            if isinstance(embedding, dict):
                for k in ('embedding', 'embeddings', 'mean', 'avg', 'pooled'):
                    if k in embedding:
                        embedding = embedding[k]
                        break
                else:
                    embedding = next(v for v in embedding.values() if isinstance(v, np.ndarray))
        
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        embedding = embedding.astype(np.float32, copy=False)
        
        # Ensure 1D
        if embedding.ndim == 2:
            if 1 in embedding.shape:
                embedding = embedding.squeeze()
            else:
                if embedding.shape[0] >= embedding.shape[1]:
                    embedding = embedding.mean(axis=0)
                else:
                    embedding = embedding.mean(axis=1)
        elif embedding.ndim > 2:
            embedding = embedding.mean(axis=tuple(range(embedding.ndim - 1)))
        
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)
        
        return embedding
    
    def _cache_all_embeddings(self):
        """Load all embeddings into memory."""
        missing = {self.mod1_name: [], self.mod2_name: [], 'ppi': []}
        failed = {self.mod1_name: [], self.mod2_name: []}
        loaded_count = 0
        ppi_found = 0
        
        for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="Loading embeddings")):
            mod1_file = self.mod1_dir / f"{protein_id}.npy"
            mod2_file = self.mod2_dir / f"{protein_id}.npy"
            
            try:
                if not mod1_file.exists():
                    missing[self.mod1_name].append(protein_id)
                    raise FileNotFoundError(str(mod1_file))
                if not mod2_file.exists():
                    missing[self.mod2_name].append(protein_id)
                    raise FileNotFoundError(str(mod2_file))
                
                mod1_emb = torch.from_numpy(self._load_embedding(mod1_file)).float()
                mod2_emb = torch.from_numpy(self._load_embedding(mod2_file)).float()
                
                # Load PPI if available (pre-extracted by CAFA ID)
                if self.use_ppi:
                    ppi_file = self.ppi_dir / f"{protein_id}.npy"
                    if ppi_file.exists():
                        try:
                            ppi_emb = torch.from_numpy(np.load(ppi_file)).float()
                            ppi_flag = torch.tensor([1.0])
                            ppi_found += 1
                        except Exception:
                            ppi_emb = torch.zeros(self.ppi_dim)
                            ppi_flag = torch.tensor([0.0])
                            missing['ppi'].append(protein_id)
                    else:
                        ppi_emb = torch.zeros(self.ppi_dim)
                        ppi_flag = torch.tensor([0.0])
                        missing['ppi'].append(protein_id)
                else:
                    ppi_emb = torch.zeros(self.ppi_dim)
                    ppi_flag = torch.tensor([0.0])
                
                self.embedding_cache[idx] = {
                    self.mod1_name: mod1_emb,
                    self.mod2_name: mod2_emb,
                    'ppi': ppi_emb,
                    'ppi_flag': ppi_flag
                }
                loaded_count += 1
                
            except FileNotFoundError:
                continue
            except Exception as e:
                try:
                    if mod1_file.exists():
                        try:
                            _ = self._load_embedding(mod1_file)
                        except Exception:
                            failed[self.mod1_name].append(protein_id)
                    if mod2_file.exists():
                        try:
                            _ = self._load_embedding(mod2_file)
                        except Exception:
                            failed[self.mod2_name].append(protein_id)
                except Exception:
                    if mod1_file.exists():
                        failed[self.mod1_name].append(protein_id)
                    if mod2_file.exists():
                        failed[self.mod2_name].append(protein_id)
        
        # Report statistics
        for modality, ids in missing.items():
            if ids and modality != 'ppi':
                print(f"WARNING: {len(ids)} {modality} embeddings MISSING")
        
        for modality, ids in failed.items():
            if ids:
                print(f"WARNING: {len(ids)} {modality} embeddings FAILED to load")
        
        if self.use_ppi:
            ppi_missing = len(missing['ppi'])
            print(f"PPI: {ppi_found} found, {ppi_missing} missing "
                  f"({ppi_found/(ppi_found+ppi_missing)*100:.1f}% coverage)")
        
        print(f"Successfully cached {loaded_count}/{len(self.protein_ids)} proteins")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        if idx not in self.embedding_cache:
            raise RuntimeError(f"Embedding for index {idx} not in cache")
        
        embeddings = self.embedding_cache[idx]
        
        return {
            'mod1': embeddings[self.mod1_name],
            'mod2': embeddings[self.mod2_name],
            'ppi': embeddings['ppi'],
            'ppi_flag': embeddings['ppi_flag'],
            'labels': self.labels[idx],
            'protein_id': self.protein_ids[idx]
        }


def collate_fn(batch):
    """Collate function."""
    mod1 = torch.stack([b['mod1'] for b in batch])
    mod2 = torch.stack([b['mod2'] for b in batch])
    ppi = torch.stack([b['ppi'] for b in batch])
    ppi_flag = torch.stack([b['ppi_flag'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    protein_ids = [b['protein_id'] for b in batch]
    
    return {
        'mod1': mod1,
        'mod2': mod2,
        'ppi': ppi,
        'ppi_flag': ppi_flag,
        'labels': labels,
        'protein_ids': protein_ids
    }