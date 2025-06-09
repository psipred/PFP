import os, random, pickle
import numpy as np
import torch
import scipy.sparse as ssp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List, Union
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch.nn.functional as F

class InterlabelGODataset(Dataset):
    def __init__(self,
        features_dir: str,
        embedding_type: str,                # "esm" | "t5" | "mmstie"
        names_npy: str,
        labels_npy: str | None = None,
        repr_layers: list | None = None,
        low_memory: bool = False,
        pca: bool = True,
        do_normalize: bool = True,
    ):
        self.features_dir = features_dir
        self.names_npy = names_npy
        self.repr_layers = repr_layers if repr_layers is not None else [34, 35, 36]

        embedding_type = embedding_type.lower()
        assert embedding_type in {"esm", "esm_mean", "t5", "mmsite", "text"}, \
            f"Unsupported embedding_type '{embedding_type}'"
        self.low_memory = low_memory
        self.feature_cache = dict()
        self.temp = dict()
        self.embedding_type = embedding_type
        self.pca = pca
        self.do_normalize = do_normalize
        self.mean_ = None
        self.std_ = None
        
        # Counter for missing data
        self.missing_data_count = 0
        self.missing_data_names = []

        if labels_npy is None:
            self.prediction = True
        else:
            self.prediction = False
            self.labels_npy = labels_npy

        # load names, labels
        self.names = np.load(self.names_npy)

        if not self.prediction:
            self.labels = self.load_labels(self.labels_npy)

        # EMB_SUBDIR mapping
        self.EMB_SUBDIR = {
            "esm": "/scratch0/cafa5_small/esm/",
            "esm_mean": "/scratch0/cafa5_small/esm_mean/",
            "text": "/scratch0/cafa5_small/prot2text/text_embeddings",
            "t5": "embeddings/t5",
            "mmsite": "embeddings/mmstie",
        }

        # Pre-load embeddings if not low_memory
        if not self.low_memory:
            self._preload_embeddings()

    def _preload_embeddings(self):
        """Pre-load all embeddings and handle missing data"""
        if self.embedding_type == "mmsite":
            for name in tqdm(self.names, desc="Loading mmsite embeddings"):
                try:
                    feature, temp = self.load_featureMM(name)
                    self.feature_cache[name] = feature
                    self.temp[name] = temp
                except (FileNotFoundError, Exception) as e:
                    self.missing_data_count += 1
                    self.missing_data_names.append(name)
                    print(f"Warning: Missing data for {name}: {e}")
                    
        elif self.embedding_type == "t5":
            for name in tqdm(self.names, desc="Loading T5 embeddings"):
                try:
                    self.feature_cache[name] = self.load_featureT5(name)
                except (FileNotFoundError, Exception) as e:
                    self.missing_data_count += 1
                    self.missing_data_names.append(name)
                    print(f"Warning: Missing data for {name}: {e}")
                    
        elif self.embedding_type in {"esm", "esm_mean"}:
            subdir = self.EMB_SUBDIR[self.embedding_type]
            for name in tqdm(self.names, desc=f"Loading {self.embedding_type} embeddings"):
                try:
                    self.feature_cache[name] = self._load_feature_file(
                        os.path.join(self.features_dir, subdir), name
                    )
                except (FileNotFoundError, Exception) as e:
                    self.missing_data_count += 1
                    self.missing_data_names.append(name)
                    print(f"Warning: Missing data for {name}: {e}")

        # Print summary of missing data
        if self.missing_data_count > 0:
            print(f"\n=== Missing Data Summary ===")
            print(f"Total missing data entries: {self.missing_data_count}")
            print(f"Total valid data entries: {len(self.names) - self.missing_data_count}")
            print(f"Missing data percentage: {(self.missing_data_count / len(self.names)) * 100:.2f}%")
            print(f"First 10 missing entries: {self.missing_data_names[:10]}")

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

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        # Check if data is missing (either not cached or known to be missing)
        if name in self.missing_data_names:
            return None  # Signal that this data should be skipped

        # Try to load data if not cached
        if name not in self.feature_cache:
            try:
                if self.embedding_type == "mmsite":
                    feature, temp = self.load_featureMM(name)
                    self.feature_cache[name] = feature
                    self.temp[name] = temp
                elif self.embedding_type == "t5":
                    self.feature_cache[name] = self.load_featureT5(name)
                else:
                    subdir = self.EMB_SUBDIR[self.embedding_type]
                    self.feature_cache[name] = self._load_feature_file(
                        os.path.join(self.features_dir, subdir), name
                    )
            except (FileNotFoundError, Exception) as e:
                self.missing_data_count += 1
                self.missing_data_names.append(name)
                print(f"Warning: Missing data for {name} at runtime: {e}")
                return None  # Signal that this data should be skipped

        # Return data based on prediction mode
        if self.prediction:
            if self.embedding_type == "mmsite":
                return name, self.feature_cache[name], self.temp[name]
            else:
                return name, self.feature_cache[name]
        else:
            label = self.labels[idx]
            if self.embedding_type == "mmsite":
                return name, self.feature_cache[name], self.temp[name], label
            else:
                return name, self.feature_cache[name], label

    def load_featureMM(self, name: str):
        """Load mmsite features with error handling"""
        try:
            seq_mat = np.load(os.path.join(
                self.EMB_SUBDIR['esm'], f'{name}.npy'),
                allow_pickle=True).item()["embedding"]
            text_mat = np.load(os.path.join(
                self.EMB_SUBDIR['text'], f'{name}.npy'),
                allow_pickle=True).item()["embedding"]

            seq_vec = seq_mat.astype(np.float32)
            text_vec = text_mat.astype(np.float32)

            return torch.from_numpy(seq_vec).float(), torch.from_numpy(text_vec).float()
        except Exception as e:
            raise FileNotFoundError(f"Could not load mmsite features for {name}: {e}")

    def load_featureT5(self, name: str):
        """Load T5 features with error handling"""
        try:
            subdir = self.EMB_SUBDIR["t5"]  # Fixed key case
            features = np.load(os.path.join(self.features_dir, subdir, f"{name}.npy"), allow_pickle=True)
            if isinstance(features, np.ndarray) and features.ndim == 2:
                features = features.mean(axis=0).astype(np.float32)
            return torch.from_numpy(features).float()
        except Exception as e:
            raise FileNotFoundError(f"Could not load T5 features for {name}: {e}")

    def _load_feature_file(self, directory: str, name: str) -> torch.Tensor:
        """Load a single .npy and return a float32 tensor with error handling"""
        path = os.path.join(directory, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            arr = np.load(path, allow_pickle=True)

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

            return torch.from_numpy(arr).float()
            
        except Exception as e:
            raise FileNotFoundError(f"Could not process file {path}: {e}")

    def _pool_token_matrix(self, mat: np.ndarray) -> np.ndarray:
        """Default pooling = mean over sequence length dimension."""
        return mat.mean(axis=0).astype(np.float32)

    def _avg_layer_means(self, esm_dict: dict) -> np.ndarray:
        """Average the 'mean' vectors for self.repr_layers"""
        return np.stack([esm_dict['mean'][l] for l in self.repr_layers]).mean(axis=0).astype(np.float32)

    def get_missing_data_summary(self):
        """Get summary of missing data"""
        return {
            'total_missing': self.missing_data_count,
            'missing_names': self.missing_data_names,
            'missing_percentage': (self.missing_data_count / len(self.names)) * 100 if len(self.names) > 0 else 0
        }


# Custom collate function to handle None values (missing data)
def collate_fn_skip_none(batch):
    """Custom collate function that filters out None values from batch"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None  # Return None if entire batch is missing
    
    # Use default collate for remaining items
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


# Example usage:
def create_dataloader_with_missing_data_handling(dataset, batch_size=32, **kwargs):
    """Create DataLoader that handles missing data"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn_skip_none,
        **kwargs
    )