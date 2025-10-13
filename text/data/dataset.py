"""Dataset classes for text and ESM embeddings."""

import torch
from torch.utils.data import Dataset
from utils.embeddings import load_embedding


class TextDataset(Dataset):
    """Dataset for text embeddings."""
    
    def __init__(self, proteins, labels, cache_dir):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_id = self.proteins[idx]
        field_embeddings = load_embedding(self.cache_dir, protein_id, 'text')
        
        if field_embeddings is None:
            raise ValueError(f"No text embedding for {protein_id}")
        
        # Convert fp16 back to fp32 and extract tensors
        hidden_states = [emb.float() for emb in field_embeddings]
        
        return {
            'hidden_states': hidden_states,
            'labels': self.labels[idx]
        }


class ESMDataset(Dataset):
    """Dataset for ESM embeddings."""
    
    def __init__(self, proteins, labels, cache_dir):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_id = self.proteins[idx]
        embedding = load_embedding(self.cache_dir, protein_id, 'esm')
        
        if embedding is None:
            raise ValueError(f"No ESM embedding for {protein_id}")
        
        return {
            'embedding': embedding.float(),
            'labels': self.labels[idx]
        }


def text_collate_fn(batch):
    """Collate function for text embeddings."""
    labels = torch.stack([b['labels'] for b in batch])
    num_fields = len(batch[0]['hidden_states'])
    
    all_hidden_states = []
    for field_idx in range(num_fields):
        field_hidden = torch.stack([b['hidden_states'][field_idx].squeeze(0) for b in batch])
        all_hidden_states.append(field_hidden)
    
    return {
        'hidden_states': all_hidden_states,
        'labels': labels
    }


def esm_collate_fn(batch):
    """Collate function for ESM embeddings."""
    embeddings = torch.stack([b['embedding'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    
    return {
        'embeddings': embeddings,
        'labels': labels
    }




class FunctionDataset(Dataset):
    """Dataset for function field embeddings only."""
    
    def __init__(self, proteins, labels, cache_dir):
        self.proteins = proteins
        self.labels = torch.FloatTensor(labels)
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.proteins)
    
    def __getitem__(self, idx):
        protein_id = self.proteins[idx]
        
        # Load all field embeddings
        field_embeddings = load_embedding(self.cache_dir, protein_id, 'text')
        
        if field_embeddings is None:
            raise ValueError(f"No text embedding for {protein_id}")
        
        # Extract only Function field (index 3 in config.text_fields)
        # Function field has full sequence, stored as fp16
        function_embedding = field_embeddings[3].float()  # [1, seq_len, hidden_dim]
        
        return {
            'embedding': function_embedding.squeeze(0),  # [seq_len, hidden_dim]
            'labels': self.labels[idx]
        }


def function_collate_fn(batch):
    """Collate function for function-only embeddings."""
    # Stack embeddings (they're already same length due to padding)
    embeddings = torch.stack([b['embedding'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    
    return {
        'embeddings': embeddings,  # [batch_size, seq_len, hidden_dim]
        'labels': labels
    }