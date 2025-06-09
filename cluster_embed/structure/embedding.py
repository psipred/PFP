import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import networkx as nx
from node2vec import Node2Vec
from Bio.PDB import PDBParser
from tqdm import tqdm
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ProteinStructureProcessor:
    """
    Processes PDB files to create protein contact map graphs and extract structural embeddings
    following the hierarchical GCN-Pool architecture.
    """
    
    def __init__(self, contact_threshold: float = 10.0, node2vec_dim: int = 128):
        """
        Initialize the protein structure processor.
        
        Args:
            contact_threshold: Distance threshold for creating edges (Angstroms)
            node2vec_dim: Dimension of node2vec structural embeddings
        """
        self.contact_threshold = contact_threshold
        self.node2vec_dim = node2vec_dim
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
    def parse_pdb_structure(self, pdb_file: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Parse PDB file to extract amino acid sequence and Ca coordinates.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Tuple of (amino_acid_sequence, ca_coordinates) or None if failed
        """
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            amino_acids = []
            ca_coords = []
            
            # Extract first model and first chain
            model = list(structure.get_models())[0]
            chain = list(model.get_chains())[0]
            
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid
                    try:
                        # Get Ca atom coordinates
                        ca_atom = residue['CA']
                        ca_coords.append(ca_atom.get_coord())
                        
                        # Get amino acid type
                        aa_name = residue.get_resname()
                        # Convert 3-letter to 1-letter code
                        aa_1letter = self._three_to_one_letter(aa_name)
                        amino_acids.append(aa_1letter)
                        
                    except KeyError:
                        continue  # Skip if no CA atom
            
            if len(amino_acids) == 0:
                print(f"Warning: No valid amino acids found in {pdb_file}")
                return None
                
            print(f"✓ Parsed {len(amino_acids)} amino acids from {os.path.basename(pdb_file)}")
            return amino_acids, np.array(ca_coords)
            
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            return None
    
    def _three_to_one_letter(self, three_letter: str) -> str:
        """Convert 3-letter amino acid code to 1-letter."""
        conversion = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return conversion.get(three_letter, 'X')  # X for unknown
    
    def create_contact_map_graph(self, ca_coords: np.ndarray) -> np.ndarray:
        """
        Create contact map adjacency matrix from Ca coordinates.
        
        Args:
            ca_coords: Array of Ca coordinates (n_residues, 3)
            
        Returns:
            Adjacency matrix (n_residues, n_residues)
        """
        n_residues = len(ca_coords)
        print(f"  → Creating contact map for {n_residues} residues")
        
        # Calculate pairwise distances
        distances = np.zeros((n_residues, n_residues))
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Create adjacency matrix based on distance threshold
        adjacency = (distances <= self.contact_threshold).astype(float)
        
        # Remove self-loops initially (will be added later with normalization)
        np.fill_diagonal(adjacency, 0)
        
        n_edges = np.sum(adjacency) // 2  # Divide by 2 since symmetric
        print(f"  → Contact map created: {n_edges} edges with threshold {self.contact_threshold}Å")
        
        return adjacency
    
    def learn_structural_embeddings(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Learn structural embeddings using node2vec.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            Structural embeddings (n_nodes, embedding_dim)
        """
        print(f"  → Learning structural embeddings with node2vec (dim={self.node2vec_dim})")
        
        # Convert adjacency matrix to networkx graph
        G = nx.from_numpy_array(adjacency)
        
        if len(G.nodes()) == 0:
            print("    Warning: Empty graph, returning zero embeddings")
            return np.zeros((adjacency.shape[0], self.node2vec_dim))
        
        # Initialize node2vec
        node2vec = Node2Vec(
            G, 
            dimensions=self.node2vec_dim,
            walk_length=30,
            num_walks=200,
            workers=1,  # Use single worker to avoid multiprocessing issues
            p=1,        # Return parameter
            q=1         # In-out parameter
        )
        
        # Learn embeddings
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Extract embeddings for all nodes
        embeddings = np.zeros((len(G.nodes()), self.node2vec_dim))
        for i, node in enumerate(sorted(G.nodes())):
            embeddings[i] = model.wv[str(node)]
        
        print(f"  → Structural embeddings learned: shape {embeddings.shape}")
        return embeddings
    
    def create_initial_features(self, amino_acids: List[str], structural_embeddings: np.ndarray) -> np.ndarray:
        """
        Create initial feature vectors by combining one-hot encoded amino acids with structural embeddings.
        
        Args:
            amino_acids: List of amino acid single-letter codes
            structural_embeddings: Structural embeddings from node2vec
            
        Returns:
            Initial feature matrix H^(0) (n_residues, 20 + embedding_dim)
        """
        n_residues = len(amino_acids)
        print(f"  → Creating initial features for {n_residues} amino acids")
        
        # Create one-hot encoding for amino acids
        one_hot = np.zeros((n_residues, 20))
        for i, aa in enumerate(amino_acids):
            if aa in self.aa_to_idx:
                one_hot[i, self.aa_to_idx[aa]] = 1
            # Unknown amino acids remain as zero vectors
        
        # Concatenate one-hot with structural embeddings
        initial_features = np.concatenate([one_hot, structural_embeddings], axis=1)
        
        print(f"  → Initial features created: shape {initial_features.shape}")
        print(f"    - One-hot encoding: {one_hot.shape}")
        print(f"    - Structural embeddings: {structural_embeddings.shape}")
        
        return initial_features


class GraphConvPoolModule(nn.Module):
    """
    Single Graph Convolution + Pooling module for hierarchical feature extraction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, pooling_ratio: float = 0.5):
        """
        Initialize GCN-Pool module.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            pooling_ratio: Fraction of nodes to keep after pooling
        """
        super().__init__()
        self.pooling_ratio = pooling_ratio
        
        # Graph convolution layer
        self.conv = GCNConv(input_dim, hidden_dim)
        
        # Attention layer for pooling
        self.attention = GCNConv(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GCN-Pool module.
        
        Args:
            x: Node features (n_nodes, input_dim)
            edge_index: Edge indices (2, n_edges)
            batch: Batch assignment for each node
            
        Returns:
            Tuple of (pooled_features, pooled_edge_index)
        """
        # Graph convolution to enrich features
        x_conv = F.relu(self.conv(x, edge_index))
        
        # Calculate attention scores for pooling
        attention_scores = torch.sigmoid(self.attention(x_conv, edge_index))
        
        # Select top-k nodes based on attention scores
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Simple top-k selection (can be improved with more sophisticated pooling)
        n_nodes = x.size(0)
        k = max(1, int(n_nodes * self.pooling_ratio))
        
        # Get top-k nodes
        _, top_indices = torch.topk(attention_scores.squeeze(), k)
        
        # Create pooled features
        pooled_x = x_conv[top_indices] * attention_scores[top_indices]
        
        # Create new edge index for pooled graph
        # Map old indices to new indices
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(top_indices)}
        
        # Filter edges that connect selected nodes
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        new_edge_list = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in old_to_new and dst in old_to_new:
                new_edge_list.append([old_to_new[src], old_to_new[dst]])
        
        if new_edge_list:
            pooled_edge_index = torch.tensor(new_edge_list, dtype=torch.long, device=edge_index.device).t()
        else:
            # If no edges remain, create empty edge index
            pooled_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        
        return pooled_x, pooled_edge_index


class HierarchicalStructureGNN(nn.Module):
    """
    Hierarchical Graph Neural Network for protein structure embedding.
    """
    
    def __init__(self, input_dim: int = 148, hidden_dims: List[int] = [256, 256, 256], 
                 pooling_ratios: List[float] = [0.7, 0.5, 0.3], final_dim: int = 512):
        """
        Initialize hierarchical structure GNN.
        
        Args:
            input_dim: Input feature dimension (20 + node2vec_dim)
            hidden_dims: Hidden dimensions for each GCN-Pool module
            pooling_ratios: Pooling ratios for each module
            final_dim: Final embedding dimension
        """
        super().__init__()
        
        self.modules_list = nn.ModuleList()
        
        # Create stacked GCN-Pool modules
        prev_dim = input_dim
        for hidden_dim, pool_ratio in zip(hidden_dims, pooling_ratios):
            self.modules_list.append(GraphConvPoolModule(prev_dim, hidden_dim, pool_ratio))
            prev_dim = hidden_dim
        
        # Final projection layer
        self.final_projection = nn.Linear(prev_dim * 2, final_dim)  # *2 for mean+max pooling
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical GNN.
        
        Args:
            x: Initial node features
            edge_index: Edge indices
            
        Returns:
            Final protein embedding
        """
        current_x = x
        current_edge_index = edge_index
        
        print(f"    → Input to hierarchical GNN: nodes={current_x.shape[0]}, features={current_x.shape[1]}")
        
        # Pass through stacked GCN-Pool modules
        for i, module in enumerate(self.modules_list):
            current_x, current_edge_index = module(current_x, current_edge_index)
            print(f"    → After module {i+1}: nodes={current_x.shape[0]}, features={current_x.shape[1]}")
        
        # Final readout: mean + max pooling
        if current_x.size(0) > 0:
            mean_pool = torch.mean(current_x, dim=0)
            max_pool = torch.max(current_x, dim=0)[0]
            combined = torch.cat([mean_pool, max_pool], dim=0)
        else:
            # Handle edge case of empty graph
            combined = torch.zeros(current_x.size(1) * 2, device=current_x.device)
        
        # Final projection
        final_embedding = self.final_projection(combined)
        
        print(f"    → Final embedding shape: {final_embedding.shape}")
        
        return final_embedding


class ProteinStructureEmbedder:
    """
    Main class for processing proteins and generating structure embeddings.
    """
    
    def __init__(self, contact_threshold: float = 10.0, node2vec_dim: int = 128, 
                 final_embedding_dim: int = 512):
        """
        Initialize the protein structure embedder.
        
        Args:
            contact_threshold: Distance threshold for contact map (Angstroms)
            node2vec_dim: Dimension of node2vec structural embeddings
            final_embedding_dim: Final embedding dimension
        """
        self.processor = ProteinStructureProcessor(contact_threshold, node2vec_dim)
        self.final_embedding_dim = final_embedding_dim
        
        # Initialize hierarchical GNN
        input_dim = 20 + node2vec_dim  # one-hot + structural embeddings
        self.gnn = HierarchicalStructureGNN(
            input_dim=input_dim,
            hidden_dims=[256, 256, 256],
            pooling_ratios=[0.7, 0.5, 0.3],
            final_dim=final_embedding_dim
        )
        
        # Set to evaluation mode (no training)
        self.gnn.eval()
    
    def process_single_protein(self, pdb_file: str) -> Optional[np.ndarray]:
        """
        Process a single protein PDB file to generate structure embedding.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Structure embedding vector or None if failed
        """
        print(f"\nProcessing: {os.path.basename(pdb_file)}")
        
        # Step 1: Parse PDB structure
        result = self.processor.parse_pdb_structure(pdb_file)
        if result is None:
            return None
        
        amino_acids, ca_coords = result
        
        # Step 2: Create contact map graph
        adjacency = self.processor.create_contact_map_graph(ca_coords)
        
        # Step 3: Learn structural embeddings with node2vec
        structural_embeddings = self.processor.learn_structural_embeddings(adjacency)
        
        # Step 4: Create initial feature vectors
        initial_features = self.processor.create_initial_features(amino_acids, structural_embeddings)
        
        # Step 5: Convert to PyTorch tensors and create graph data
        x = torch.FloatTensor(initial_features)
        
        # Add self-loops to adjacency matrix for GCN
        adjacency_with_loops = adjacency + np.eye(adjacency.shape[0])
        edge_index, _ = dense_to_sparse(torch.FloatTensor(adjacency_with_loops))
        
        print(f"  → Graph data: nodes={x.shape[0]}, edges={edge_index.shape[1]}")
        
        # Step 6: Pass through hierarchical GNN
        with torch.no_grad():
            try:
                final_embedding = self.gnn(x, edge_index)
                embedding_np = final_embedding.cpu().numpy()
                
                print(f"  ✓ Generated embedding: shape {embedding_np.shape}")
                return embedding_np
                
            except Exception as e:
                print(f"  ✗ Error in GNN processing: {e}")
                return None
    
    def process_protein_directory(self, pdb_dir: str, output_dir: str, 
                                max_proteins: Optional[int] = None):
        """
        Process all PDB files in a directory and save embeddings.
        
        Args:
            pdb_dir: Directory containing PDB files
            output_dir: Directory to save embeddings
            max_proteins: Maximum number of proteins to process (for testing)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get all PDB files
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        
        if max_proteins:
            pdb_files = pdb_files[:max_proteins]
        
        print(f"Found {len(pdb_files)} PDB files to process")
        print(f"Output directory: {output_dir}")
        print(f"Target embedding dimension: {self.final_embedding_dim}")
        
        successful = 0
        failed = 0
        
        for pdb_file in tqdm(pdb_files, desc="Processing proteins"):
            # Extract protein ID from filename (remove AF- prefix and -F1-model_v4.pdb suffix)
            protein_id = pdb_file.replace('AF-', '').replace('-F1-model_v4.pdb', '')
            
            # Check if embedding already exists
            output_file = os.path.join(output_dir, f"{protein_id}.npy")
            if os.path.exists(output_file):
                print(f"Skipping {protein_id} (already exists)")
                continue
            
            # Process protein
            pdb_path = os.path.join(pdb_dir, pdb_file)
            embedding = self.process_single_protein(pdb_path)

            if embedding is not None:
                # Save embedding
                np.save(output_file, embedding)
                successful += 1
                print(f"  ✓ Saved: {output_file}")
            else:
                failed += 1
                print(f"  ✗ Failed: {protein_id}")
        
        print(f"\n=== Processing Summary ===")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Embeddings saved to: {output_dir}")


def main():
    """
    Main function to run the structure embedding pipeline.
    """
    # Configuration
    pdb_input_dir = "/SAN/bioinf/PFP/embeddings/structure/pdb_files"
    embedding_output_dir = "/SAN/bioinf/PFP/embeddings/structure/structure_embed"
    
    # Initialize embedder
    print("=== Hierarchical Protein Structure Embedding Pipeline ===")
    print("Architecture: Phase 1 (Contact Map + Node2Vec) → Phase 2 (Hierarchical GCN-Pool)")
    print()
    
    embedder = ProteinStructureEmbedder(
        contact_threshold=10.0,    # 10 Angstroms for contact map
        node2vec_dim=128,          # 128-dim structural embeddings
        final_embedding_dim=512    # 512-dim final embeddings
    )
    
    # Process proteins (start with a small subset for testing)
    embedder.process_protein_directory(
        pdb_dir=pdb_input_dir,
        output_dir=embedding_output_dir
        # max_proteins=4  # Remove this parameter to process all proteins
    )
    
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()