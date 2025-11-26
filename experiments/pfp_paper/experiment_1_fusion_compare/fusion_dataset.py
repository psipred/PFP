"""Dataset for Multimodal Protein Function Prediction.

Supports embedding normalization for scale consistency across modalities.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import scipy.sparse as ssp


class MultiModalDataset(Dataset):
    """Dataset with per-modality normalization."""
    
    def __init__(self, data_dir, embedding_dirs, seq_model, aspect, split, 
                 normalize='standard', norm_stats=None):
        """
        Args:
            data_dir: Path to data directory
            embedding_dirs: Dict mapping modality name to embedding directory
            seq_model: 'prott5' or 'esm'
            aspect: 'BPO', 'CCO', or 'MFO'
            split: 'train', 'valid', or 'test'
            normalize: 'none', 'l2', 'standard', or 'minmax'
            norm_stats: Pre-computed normalization statistics
        """
        self.data_dir = Path(data_dir)
        self.embedding_dirs = embedding_dirs
        self.seq_model = seq_model
        self.aspect = aspect
        self.split = split
        self.normalize = normalize
        self.norm_stats = norm_stats or {}
        
        # Load protein IDs
        protein_ids_raw = np.load(
            self.data_dir / f"{aspect}_{split}_names.npy",
            allow_pickle=True
        )
        self.protein_ids = np.array([str(pid) for pid in protein_ids_raw])
        
        # Load labels
        labels_npz = self.data_dir / f"{aspect}_{split}_labels.npz"
        labels_npy = self.data_dir / f"{aspect}_{split}_labels.npy"
        
        if labels_npz.exists():
            labels_raw = ssp.load_npz(labels_npz)
            self.labels = np.array(labels_raw.toarray(), dtype=np.float32)
        elif labels_npy.exists():
            labels_raw = np.load(labels_npy, allow_pickle=True)
            self.labels = np.array(labels_raw, dtype=np.float32)
        else:
            raise FileNotFoundError(f"Labels not found: {labels_npz} or {labels_npy}")
        
        print(f"\n{aspect} {split}:")
        print(f"  Proteins: {len(self.protein_ids)}")
        print(f"  GO terms: {self.labels.shape[1]}")
        print(f"  Normalization: {normalize}")
        
        self._check_coverage()
        
        # Compute normalization stats on training set
        if normalize == 'standard' and split == 'train' and not norm_stats:
            print("\n  Computing normalization statistics...")
            self._compute_norm_stats()
    
    def _check_coverage(self):
        """Check modality coverage."""
        text_dir = Path(self.embedding_dirs['text'])
        seq_dir = Path(self.embedding_dirs[self.seq_model])
        struct_dir = Path(self.embedding_dirs['struct'])
        ppi_dir = Path(self.embedding_dirs['ppi'])
        
        text_count = sum(1 for pid in self.protein_ids if (text_dir / f"{pid}.npy").exists())
        seq_count = sum(1 for pid in self.protein_ids if (seq_dir / f"{pid}.npy").exists())
        struct_count = sum(1 for pid in self.protein_ids if (struct_dir / f"{pid}.npy").exists())
        ppi_count = sum(1 for pid in self.protein_ids if (ppi_dir / f"{pid}.npy").exists())
        
        total = len(self.protein_ids)
        print(f"  Coverage:")
        print(f"    Text: {text_count}/{total} ({100*text_count/total:.1f}%)")
        print(f"    {self.seq_model.upper()}: {seq_count}/{total} ({100*seq_count/total:.1f}%)")
        print(f"    Struct: {struct_count}/{total} ({100*struct_count/total:.1f}%)")
        print(f"    PPI: {ppi_count}/{total} ({100*ppi_count/total:.1f}%)")
    
    def _compute_norm_stats(self):
        """Compute normalization statistics from training data."""
        from tqdm import tqdm
        
        modalities = {
            'text': (768, 'text'),
            'seq': (1024 if self.seq_model == 'prott5' else 1280, self.seq_model),
            'struct': (512, 'struct'),
            'ppi': (512, 'ppi')
        }
        
        for mod_key, (dim, mod_name) in modalities.items():
            embeddings = []
            
            # Sample up to 1000 proteins for statistics
            for pid in tqdm(self.protein_ids[:1000], desc=f"  {mod_key} stats", leave=False):
                emb, mask = self._load_embedding_raw(mod_name, pid, dim)
                if mask > 0:
                    embeddings.append(emb)
            
            if embeddings:
                embeddings = np.stack(embeddings)
                self.norm_stats[mod_key] = {
                    'mean': embeddings.mean(axis=0).astype(np.float32),
                    'std': embeddings.std(axis=0).astype(np.float32) + 1e-6
                }
                print(f"    {mod_key}: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        label = self.labels[idx]
        
        seq_dim = 1024 if self.seq_model == 'prott5' else 1280
        
        text_emb, text_mask = self._load_embedding('text', protein_id, 768, 'text')
        seq_emb, seq_mask = self._load_embedding(self.seq_model, protein_id, seq_dim, 'seq')
        struct_emb, struct_mask = self._load_embedding('struct', protein_id, 512, 'struct')
        ppi_emb, ppi_mask = self._load_embedding('ppi', protein_id, 512, 'ppi')
        
        return {
            'seq': torch.from_numpy(seq_emb).float(),
            'seq_mask': torch.tensor([seq_mask], dtype=torch.float32),
            'text': torch.from_numpy(text_emb).float(),
            'text_mask': torch.tensor([text_mask], dtype=torch.float32),
            'struct': torch.from_numpy(struct_emb).float(),
            'struct_mask': torch.tensor([struct_mask], dtype=torch.float32),
            'ppi': torch.from_numpy(ppi_emb).float(),
            'ppi_mask': torch.tensor([ppi_mask], dtype=torch.float32),
            'labels': torch.from_numpy(label).float(),
            'protein_id': protein_id
        }
    
    def _load_embedding_raw(self, modality, protein_id, dim):
        """Load embedding without normalization."""
        emb_file = Path(self.embedding_dirs[modality]) / f"{protein_id}.npy"
        
        if emb_file.exists():
            try:
                emb = np.load(emb_file, allow_pickle=True)
                
                if isinstance(emb, np.ndarray):
                    if emb.dtype == object:
                        if emb.shape == ():
                            emb_obj = emb.item()
                            if isinstance(emb_obj, dict):
                                for key in ['embedding', 'embeddings', 'mean_representations', 'representations']:
                                    if key in emb_obj:
                                        emb = emb_obj[key]
                                        break
                                else:
                                    for v in emb_obj.values():
                                        if isinstance(v, np.ndarray):
                                            emb = v
                                            break
                    
                    emb = np.array(emb, dtype=np.float32)
                    
                    if len(emb.shape) > 1:
                        emb = emb.flatten()
                    
                    if emb.shape[0] != dim:
                        if emb.shape[0] > dim:
                            emb = emb[:dim]
                        else:
                            emb = np.pad(emb, (0, dim - emb.shape[0]), mode='constant')
                    
                    return emb, 1.0
                    
            except Exception:
                pass
        
        return np.zeros(dim, dtype=np.float32), 0.0
    
    def _load_embedding(self, modality, protein_id, dim, mod_key):
        """Load embedding with normalization."""
        emb, mask = self._load_embedding_raw(modality, protein_id, dim)
        
        if mask > 0 and self.normalize != 'none':
            emb = self._normalize(emb, mod_key)
        
        return emb, mask
    
    def _normalize(self, emb, mod_key):
        """Apply normalization."""
        if self.normalize == 'l2':
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
        
        elif self.normalize == 'standard':
            if mod_key in self.norm_stats:
                mean = self.norm_stats[mod_key]['mean']
                std = self.norm_stats[mod_key]['std']
                emb = (emb - mean) / std
        
        elif self.normalize == 'minmax':
            emb_min = emb.min()
            emb_max = emb.max()
            if emb_max - emb_min > 1e-6:
                emb = (emb - emb_min) / (emb_max - emb_min)
        
        return emb


def collate_fn(batch):
    """Collate function for dataloader."""
    return {
        'seq': torch.stack([x['seq'] for x in batch]),
        'seq_mask': torch.stack([x['seq_mask'] for x in batch]),
        'text': torch.stack([x['text'] for x in batch]),
        'text_mask': torch.stack([x['text_mask'] for x in batch]),
        'struct': torch.stack([x['struct'] for x in batch]),
        'struct_mask': torch.stack([x['struct_mask'] for x in batch]),
        'ppi': torch.stack([x['ppi'] for x in batch]),
        'ppi_mask': torch.stack([x['ppi_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
        'protein_ids': [x['protein_id'] for x in batch]
    }