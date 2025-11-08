"""Dataset for pairwise modality fusion with optional PPI."""

import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class PairwiseModalityDataset(Dataset):
    """Dataset loading two modalities with optional PPI."""
    
    def __init__(self, data_dir, embedding_dirs, modality_pair, aspect, split, 
                 use_ppi=False, cache_embeddings=True):
        self.data_dir = Path(data_dir)
        self.aspect = aspect
        self.split = split
        self.cache_embeddings = cache_embeddings
        self.modality_pair = modality_pair
        self.use_ppi = use_ppi
        self.ppi_dim = 512
        
        # Parse modality names
        mod1, mod2 = modality_pair.split('_')
        self.mod1_name = mod1
        self.mod2_name = mod2
        self.mod1_dir = Path(embedding_dirs[mod1])
        self.mod2_dir = Path(embedding_dirs[mod2])
        
        # PPI directory
        if use_ppi:
            self.ppi_dir = Path(embedding_dirs.get('ppi', 
                Path(data_dir).parent / 'embedding_cache' / 'ppi'))
        
        # Load protein names and labels
        names_file = self.data_dir / f"{aspect}_{split}_names.npy"
        self.protein_ids = np.load(names_file, allow_pickle=True)
        
        labels_file = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_sparse = ssp.load_npz(labels_file)
        self.labels = torch.FloatTensor(labels_sparse.toarray())
        
        print(f"Loaded {len(self.protein_ids)} proteins for {aspect} {split} ({modality_pair})")
        if use_ppi:
            print(f"PPI embeddings will be loaded from: {self.ppi_dir}")
        
        self.embedding_cache = {}
        if cache_embeddings:
            print(f"Caching embeddings in memory...")
            self._cache_all_embeddings()
        
    def _load_embedding(self, emb_file):
        """Load embedding from .npy or .npz and return a 1D float32 vector."""
        import numpy as np

        emb_file = str(emb_file)
        data = np.load(emb_file, allow_pickle=True, mmap_mode=None)

        # Handle .npz (NpzFile) vs .npy (ndarray / object)
        if hasattr(data, 'files'):  # np.lib.npyio.NpzFile
            # Try common keys first, then fall back to the first array-like value
            for k in ('embedding', 'embeddings', 'mean', 'avg', 'pooled'):
                if k in data.files:
                    embedding = data[k]
                    break
            else:
                # take first entry
                first_key = data.files[0]
                embedding = data[first_key]
        else:
            embedding = data  # ndarray or object array

        # If it's a 0-d object that actually contains a dict/array, unwrap it
        if isinstance(embedding, np.ndarray) and embedding.dtype == object:
            # Possible cases: array(dict), array(list), scalar object
            if embedding.shape == () or embedding.shape == (1,):
                embedding = embedding.item()
            # If still object after item(), try to find an ndarray inside
            if not isinstance(embedding, np.ndarray):
                if isinstance(embedding, dict):
                    # prefer usual keys; else first ndarray value
                    for k in ('embedding', 'embeddings', 'mean', 'avg', 'pooled'):
                        if k in embedding and isinstance(embedding[k], np.ndarray):
                            embedding = embedding[k]
                            break
                    else:
                        embedding = next(
                            (v for v in embedding.values() if isinstance(v, np.ndarray)),
                            None
                        )
                elif isinstance(embedding, (list, tuple)):
                    # e.g., list of vectors -> stack then pool
                    embedding = np.array(embedding)
        
        if embedding is None:
            raise ValueError(f"Could not find a numeric embedding in {emb_file}")

        # At this point embedding should be an ndarray; coerce dtype
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        if embedding.dtype not in (np.float32, np.float64):
            # This will fail loudly if it cannot be cast -> good (we'll catch upstream)
            embedding = embedding.astype(np.float32, copy=False)
        else:
            embedding = embedding.astype(np.float32, copy=False)

        # Shape handling:
        # - 1D: ok
        # - 2D sequence x dim: mean-pool across sequence axis if it looks like (L, D)
        # - (1, D) or (D, 1): squeeze
        if embedding.ndim == 2:
            # If it's (1, D) / (D, 1), squeeze; else assume (L, D) and mean-pool L
            if 1 in embedding.shape:
                embedding = embedding.squeeze()
            else:
                # Heuristic: pool the longer axis as sequence length
                seq_axis = 0 if embedding.shape[0] >= embedding.shape[1] else 1
                if seq_axis == 0:
                    embedding = embedding.mean(axis=0)
                else:
                    embedding = embedding.mean(axis=1)
        elif embedding.ndim > 2:
            # Too many dims -> fallback to global mean
            embedding = embedding.mean(axis=tuple(range(embedding.ndim - 1)))

        if embedding.ndim != 1:
            # Final sanity check
            embedding = embedding.reshape(-1)

        return embedding


    def _cache_all_embeddings(self):
        """Load all embeddings into memory with proper error reporting."""
        import torch

        missing = {self.mod1_name: [], self.mod2_name: [], 'ppi': []}
        failed  = {self.mod1_name: [], self.mod2_name: [], 'ppi': []}
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

                # Load PPI if enabled
                if self.use_ppi:
                    ppi_file = self.ppi_dir / f"{protein_id}.npy"
                    if ppi_file.exists():
                        try:
                            ppi_emb = torch.from_numpy(np.load(ppi_file)).float()
                            # Ensure correct dimension
                            if ppi_emb.shape[0] != self.ppi_dim:
                                print(f"Warning: PPI embedding for {protein_id} has dimension {ppi_emb.shape[0]}, expected {self.ppi_dim}")
                                ppi_emb = torch.zeros(self.ppi_dim)
                                ppi_flag = torch.tensor([0.0])
                            else:
                                ppi_flag = torch.tensor([1.0])
                                ppi_found += 1
                        except Exception as e:
                            ppi_emb = torch.zeros(self.ppi_dim)
                            ppi_flag = torch.tensor([0.0])
                            failed['ppi'].append(protein_id)
                    else:
                        ppi_emb = torch.zeros(self.ppi_dim)
                        ppi_flag = torch.tensor([0.0])
                        missing['ppi'].append(protein_id)
                else:
                    # Default zero embeddings when PPI is not used
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
                # already recorded in `missing`
                continue
            except Exception as e:
                # File exists but parsing/shape/dtype failed
                # Record which modality failed
                try:
                    # probe individually to attribute failure
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
                    # if even probing fails, mark both to be safe
                    if mod1_file.exists(): failed[self.mod1_name].append(protein_id)
                    if mod2_file.exists(): failed[self.mod2_name].append(protein_id)
                # Optional: log the actual error for the first few cases
                if loaded_count < 3:  # avoid spamming
                    print(f"[ERROR] {protein_id}: {e}")

        # Summary
        for modality, ids in missing.items():
            if ids and modality != 'ppi':
                print(f"WARNING: {len(ids)} {modality} embeddings MISSING (files not found)")
        for modality, ids in failed.items():
            if ids and modality != 'ppi':
                print(f"WARNING: {len(ids)} {modality} embeddings FAILED to load (parse/shape/dtype)")

        if self.use_ppi:
            ppi_missing = len(missing['ppi'])
            ppi_failed = len(failed['ppi'])
            total_ppi_unavailable = ppi_missing + ppi_failed
            if total_ppi_unavailable > 0:
                print(f"PPI: {ppi_found} found, {ppi_missing} missing, {ppi_failed} failed "
                      f"({ppi_found/len(self.protein_ids)*100:.1f}% coverage)")
            else:
                print(f"PPI: {ppi_found}/{len(self.protein_ids)} available ({ppi_found/len(self.protein_ids)*100:.1f}% coverage)")

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
    """Collate function for pairwise modality dataset with PPI."""
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