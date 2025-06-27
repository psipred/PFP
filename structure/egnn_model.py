import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Dict
import math


class EGNNLayer(MessagePassing):
    """E(3)-Equivariant Graph Neural Network Layer."""
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int = 4,
                 hidden_dim: int = 128,
                 update_pos: bool = True,
                 normalize: bool = True,
                 aggr: str = "mean"):
        super().__init__(aggr=aggr, node_dim=-2)  # Set node dimension index
        
        self.input_dim = node_dim  # Feature dimension
        self.hidden_dim = hidden_dim
        self.update_pos = update_pos
        self.normalize = normalize
        
        # Edge network: computes messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.input_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.input_dim)
        )
        
        # Position update network (if enabled)
        if update_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.input_dim)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N, node_dim)
            pos: Node positions (N, 3)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
            
        Returns:
            Updated node features and positions
        """
        # Save original for residual
        x_residual = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        
        # Update node features
        x = self.node_mlp(torch.cat([x, out], dim=-1))
        x = self.layer_norm(x + x_residual)
        
        # Update positions if enabled
        if self.update_pos:
            pos = self._update_positions(pos, edge_index, out)
            
        return x, pos
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, pos_i: torch.Tensor, 
                pos_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute messages between nodes."""
        # Concatenate node and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.edge_mlp(msg_input)
        
        if self.normalize:
            # Normalize by distance
            dist = edge_attr[:, 0:1]  # First element is distance
            msg = msg / (dist + 1e-8)
            
        return msg
        
    def _update_positions(self, pos: torch.Tensor, edge_index: torch.Tensor, 
                         messages: torch.Tensor) -> torch.Tensor:
        """Update node positions equivariantly."""
        row, col = edge_index
        
        # Compute position updates
        pos_diff = pos[col] - pos[row]
        dist = torch.norm(pos_diff, dim=-1, keepdim=True)
        
        # Normalize direction
        direction = pos_diff / (dist + 1e-8)
        
        # Compute update magnitude
        update_size = self.pos_mlp(messages[row])
        
        # Apply updates
        pos_updates = torch.zeros_like(pos)
        pos_updates.index_add_(0, row, update_size * direction)
        
        return pos + 0.1 * pos_updates  # Small step size


class EGNN(nn.Module):
    """E(3)-Equivariant Graph Neural Network for protein structures."""
    
    def __init__(self,
                 input_dim: int = 1280,  # ESM embedding dimension
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 n_layers: int = 4,
                 edge_dim: int = 4,
                 dropout: float = 0.1,
                 update_pos: bool = False,  # Usually False for proteins
                 pool: str = "mean"):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.pool = pool
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                update_pos=update_pos
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Pooling
        if pool == "mean":
            self.pool_fn = global_mean_pool
        elif pool == "max":
            self.pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pool}")
            
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through EGNN.
        
        Args:
            batch_data: Dictionary containing:
                - node_features: (N, input_dim)
                - edge_index: (2, E)
                - edge_attr: (E, edge_dim)
                - node_pos: (N, 3)
                - batch: (N,) batch assignment for nodes
                
        Returns:
            Graph-level embeddings (B, output_dim)
        """
        x = batch_data['node_features']
        edge_index = batch_data['edge_index']
        edge_attr = batch_data['edge_attr']
        pos = batch_data['node_pos']
        batch = batch_data.get('batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Project input features
        x = self.input_proj(x)
        
        # Apply EGNN layers
        for layer in self.egnn_layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
            
        # Project to output dimension
        x = self.output_proj(x)
        
        # Pool to graph-level representation
        graph_emb = self.pool_fn(x, batch)
        
        return graph_emb


class StructureGOClassifier(nn.Module):
    """Complete model: EGNN + GO classifier head."""
    
    def __init__(self,
                 egnn_config: dict,
                 classifier_config: dict,
                 use_mmstie_fusion: bool = False):
        super().__init__()
        
        # EGNN for structure encoding
        self.egnn = EGNN(**egnn_config)
        
        # Import and use existing classifier
        from Network.base_go_classifier import BaseGOClassifier
        
        # Adjust input dim for classifier
        classifier_config['input_dim'] = egnn_config['output_dim']
        self.classifier = BaseGOClassifier(**classifier_config)
        
        self.use_mmstie_fusion = use_mmstie_fusion
        
        if use_mmstie_fusion:
            # Optional: fusion with text embeddings
            from Network.dnn import AP_align_fuse
            self.fusion_model = AP_align_fuse(tau=0.8, hidden_size=256)
            # Adjust classifier input
            classifier_config['input_dim'] = 2048  # After fusion
            self.classifier = BaseGOClassifier(**classifier_config)
            
    def forward(self, batch_data: Dict[str, torch.Tensor], 
                text_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch_data: Graph batch data
            text_embeddings: Optional text embeddings for fusion
            
        Returns:
            GO term predictions
        """
        # Encode structure
        struct_emb = self.egnn(batch_data)
        
        if self.use_mmstie_fusion and text_embeddings is not None:
            # Fuse with text
            outputs = self.fusion_model(text_embeddings, struct_emb)
            features = outputs["token_embeddings"]
        else:
            features = struct_emb
            
        # Classify
        logits = self.classifier(features)
        
        return logits


def collate_graph_batch(batch_list):
    """Custom collate function for graph batches."""
    names = [item[0] for item in batch_list]
    graph_dicts = [item[1] for item in batch_list]
    
    # Combine into single batch
    batch_data = {
        'node_features': [],
        'edge_index': [],
        'edge_attr': [],
        'node_pos': [],
        'num_nodes': []
    }
    
    if 'label' in graph_dicts[0]:
        labels = torch.stack([g['label'] for g in graph_dicts])
    else:
        labels = None
        
    # Track node offsets for batching
    node_offset = 0
    batch_assignment = []
    
    for i, graph in enumerate(graph_dicts):
        num_nodes = graph['num_nodes']
        
        # Add node data
        batch_data['node_features'].append(graph['node_features'])
        batch_data['node_pos'].append(graph['node_pos'])
        batch_data['num_nodes'].append(num_nodes)
        
        # Offset edge indices
        edge_index = graph['edge_index'] + node_offset
        batch_data['edge_index'].append(edge_index)
        batch_data['edge_attr'].append(graph['edge_attr'])
        
        # Batch assignment
        batch_assignment.extend([i] * num_nodes)
        
        node_offset += num_nodes
        
    # Concatenate all
    batch_tensor_data = {
        'node_features': torch.cat(batch_data['node_features'], dim=0),
        'edge_index': torch.cat(batch_data['edge_index'], dim=1),
        'edge_attr': torch.cat(batch_data['edge_attr'], dim=0),
        'node_pos': torch.cat(batch_data['node_pos'], dim=0),
        'batch': torch.tensor(batch_assignment, dtype=torch.long)
    }

    if labels is not None:
        return names, batch_tensor_data, labels
    else:
        return names, batch_tensor_data