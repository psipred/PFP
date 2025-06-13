import os
import numpy as np
import torch
import scipy.sparse as ssp
from torch.utils.data import Dataset
from tqdm import tqdm

class InterlabelGODataset(Dataset):
    def __init__(self,
        features_dir: str,
        embedding_type: str,
        names_npy: str,
        labels_npy: str = None,
        repr_layers: list = None,
        low_memory: bool = False,
    ):
        # Initialize basic attributes
        self.features_dir = features_dir
        self.names_npy = names_npy
        self.repr_layers = repr_layers if repr_layers is not None else [34, 35, 36]
        
        embedding_type = embedding_type.lower()
        assert embedding_type in {"esm", "esm_mean", "structure", "mmsite", "text"}, \
            f"Unsupported embedding_type '{embedding_type}'"
        self.low_memory = low_memory
        self.feature_cache = dict()
        self.temp = dict()
        self.embedding_type = embedding_type
        
        # Missing data tracking
        self.missing_data_count = 0
        self.missing_data_names = []
        self.failed_indices = set()
        
        if labels_npy is None:
            self.prediction = True
        else:
            self.prediction = False
            self.labels_npy = labels_npy

        # load names, labels
        self.names = np.load(self.names_npy)
        
        if not self.prediction:
            self.labels = self.load_labels(self.labels_npy)

        # Pre-load embeddings if not using low_memory mode
        if not self.low_memory:
            self._preload_embeddings()
    
    def _preload_embeddings(self):
        """Pre-load embeddings and track missing data"""
        print(f"Pre-loading {self.embedding_type} embeddings...")
        
        for i, name in enumerate(tqdm(self.names, desc=f"Loading {self.embedding_type} embeddings")):
            try:
                if self.embedding_type == "mmsite":
                    self.feature_cache[name], self.temp[name] = self.load_featureMM(name)
                elif self.embedding_type == "structure":
                    self.feature_cache[name] = self.load_featureStructure(name)
                elif self.embedding_type in {"esm", "esm_mean", "text"}:
                    subdir = self._get_embedding_subdir(self.embedding_type)
                    self.feature_cache[name] = self._load_feature_file(
                        os.path.join(self.features_dir, subdir), name
                    )
            except Exception as e:
                self.missing_data_count += 1
                self.missing_data_names.append(name)
                self.failed_indices.add(i)
                print(f"Warning: Failed to load {self.embedding_type} embedding for {name}: {e}")
        
        if self.missing_data_count > 0:
            print(f"\n=== MISSING DATA SUMMARY ===")
            print(f"Total missing {self.embedding_type} embeddings: {self.missing_data_count}")
            print(f"Total samples: {len(self.names)}")
            print(f"Success rate: {((len(self.names) - self.missing_data_count) / len(self.names)) * 100:.2f}%")
            print(f"First 10 missing files: {self.missing_data_names[:10]}")
            if len(self.missing_data_names) > 10:
                print(f"... and {len(self.missing_data_names) - 10} more")
            print("="*30)
    
    def _get_embedding_subdir(self, embedding_type):
        """Get the subdirectory for embedding type"""
        EMB_SUBDIR = {
            "esm": "/SAN/bioinf/PFP/embeddings/cafa5_small/esm",
            "esm_mean": "/SAN/bioinf/PFP/embeddings/cafa5_small/esm_mean/",
            "text": "/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/text_embeddings",
            "structure": "/SAN/bioinf/PFP/embeddings/cafa5_small/graph3",
            "mmsite": "embeddings/mmstie", 
        }
        return EMB_SUBDIR[embedding_type]
    
    def __len__(self):
        # Return length excluding failed indices
        return len(self.names) - len(self.failed_indices)
    
    def __getitem__(self, idx):
        # Map the requested index to actual valid indices
        valid_indices = [i for i in range(len(self.names)) if i not in self.failed_indices]
        
        if idx >= len(valid_indices):
            raise IndexError(f"Index {idx} out of range for {len(valid_indices)} valid samples")
        
        actual_idx = valid_indices[idx]
        name = self.names[actual_idx]
        
        # Try to load if not in cache (for low_memory mode)
        if name not in self.feature_cache:
            try:
                if self.embedding_type == "mmsite":
                    self.feature_cache[name], self.temp[name] = self.load_featureMM(name)
                elif self.embedding_type == "structure":
                    self.feature_cache[name] = self.load_featureStructure(name)
                else:
                    subdir = self._get_embedding_subdir(self.embedding_type)
                    self.feature_cache[name] = self._load_feature_file(
                        os.path.join(self.features_dir, subdir), name
                    )
            except Exception as e:
                # This shouldn't happen if pre-loading worked correctly
                print(f"Unexpected error loading {name} at runtime: {e}")
                # Skip to next valid item
                return self.__getitem__((idx + 1) % len(valid_indices))
        
        # Return the data
        if self.prediction:
            if self.embedding_type == "mmsite":
                return name, self.feature_cache[name], self.temp[name]
            else:
                return name, self.feature_cache[name]
        else:   
            label = self.labels[actual_idx]
            if self.embedding_type == "mmsite":
                return name, self.feature_cache[name], self.temp[name], label
            else:
                return name, self.feature_cache[name], label
    
    def get_missing_data_summary(self):
        """Get a summary of missing data"""
        return {
            'total_missing': self.missing_data_count,
            'total_samples': len(self.names),
            'success_rate': ((len(self.names) - self.missing_data_count) / len(self.names)) * 100,
            'missing_names': self.missing_data_names,
            'failed_indices': self.failed_indices
        }
    
    def load_labels(self, labels_npy: str) -> np.ndarray:
        """Load labels from npy or npz file."""
        if labels_npy.endswith(".npy"):
            labels = np.load(labels_npy)
        elif labels_npy.endswith(".npz"):
            labels = ssp.load_npz(labels_npy).toarray()
        else:
            raise Exception("Unknown label file format")
        labels = torch.from_numpy(labels).float()
        return labels
    
    def load_featureStructure(self, name: str) -> torch.Tensor:
        """Load structure embedding from npy file with proper error handling."""
        # Random baseline shortcut
        if str(os.getenv("PFP_RANDOM_STRUCTURE", "0")).lower() in {"1", "true", "yes"}:
            rng = np.random.RandomState(abs(hash(name)) % (2 ** 32))
            arr = rng.randn(1024).astype(np.float32)
            return torch.from_numpy(arr).float()

        subdir = self._get_embedding_subdir("structure")
        file_path = os.path.join(self.features_dir, subdir, f"{name}.npy")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Structure embedding file not found: {file_path}")
        
        try:
            arr = np.load(file_path, allow_pickle=True)
        except Exception as e:
            raise IOError(f"Failed to load {file_path}: {e}")

        # Handle object-array dict wrappers
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                arr = arr.item()
                if isinstance(arr, dict) and "embedding" in arr:
                    arr = arr["embedding"]
                else:
                    raise ValueError(f"Expected dict with key 'embedding', got {type(arr)}")
            except Exception as e:
                raise ValueError(f"Failed to extract embedding from object array: {e}")

        # Pool (L, D) matrices to (D,) vectors
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            arr = arr.mean(axis=0)

        # Ensure float32 dtype before tensor
        if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        return torch.from_numpy(arr).float()
    
    def load_featureMM(self, name: str):
        """Load mmsite embeddings with error handling."""
        try:
            seq_mat = np.load(os.path.join(
                self._get_embedding_subdir('esm'), f'{name}.npy'),
                allow_pickle=True).item()["embedding"]
            text_mat = np.load(os.path.join(
                self._get_embedding_subdir('text'), f'{name}.npy'),
                allow_pickle=True).item()["embedding"]
            
            seq_vec = seq_mat.astype(np.float32)
            text_vec = text_mat.astype(np.float32)
            
            return torch.from_numpy(seq_vec).float(), torch.from_numpy(text_vec).float()
        except Exception as e:
            raise IOError(f"Failed to load mmsite embedding for {name}: {e}")
    
    def _load_feature_file(self, directory: str, name: str) -> torch.Tensor:
        """Load a single .npy file with error handling."""
        path = os.path.join(directory, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            raise IOError(f"Failed to load {path}: {e}")

        # Handle different embedding types
        if self.embedding_type == "esm_mean":
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    arr = arr.item()["embedding"]
                elif arr.ndim == 0 and isinstance(arr.item(), dict):
                    arr = arr.item()["embedding"]
                return torch.from_numpy(arr.astype(np.float32)).float()

        if self.embedding_type == "text":
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    arr = arr.item()["embedding"]
                elif arr.ndim == 0 and isinstance(arr.item(), dict):
                    arr = arr.item()["embedding"]
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                return torch.from_numpy(arr.astype(np.float32)).float()

        # ESM handling
        if self.embedding_type == "esm":
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr.item()

            if isinstance(arr, dict):
                if "mean" in arr:
                    arr = self._avg_layer_means(arr)
                elif "embedding" in arr:
                    arr = self._pool_token_matrix(arr["embedding"])
                else:
                    raise ValueError(f"Unknown keys in ESM file {path}: {arr.keys()}")
            elif isinstance(arr, np.ndarray):
                if arr.ndim == 2:
                    arr = self._pool_token_matrix(arr)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
            else:
                raise TypeError(f"Unsupported ESM data type in {path}: {type(arr)}")

        return torch.from_numpy(arr).float()
    
    def _pool_token_matrix(self, mat: np.ndarray) -> np.ndarray:
        """Default pooling = mean over sequence length dimension."""
        return mat.mean(axis=0).astype(np.float32)
    
    def _avg_layer_means(self, esm_dict: dict) -> np.ndarray:
        """Average the 'mean' vectors for self.repr_layers."""
        return np.stack([esm_dict['mean'][l] for l in self.repr_layers]).mean(axis=0).astype(np.float32)