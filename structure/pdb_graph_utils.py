import os
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=PDB.PDBExceptions.PDBConstructionWarning)

# --- Compatibility for Biopython >= 1.82 where three_to_one was removed ---
try:
    from Bio.PDB.Polypeptide import three_to_one  # Biopython < 1.82
except ImportError:  # three_to_one no longer available
    from Bio.SeqUtils import seq1 as three_to_one  # Drop‑in replacement
# -------------------------------------------------------------------------

# Setup logger
logger = logging.getLogger(__name__)

class PDBProcessor:
    """Extract sequences and Cα coordinates from PDB files."""
    
    def __init__(self, pdb_dir: str = "/SAN/bioinf/PFP/embeddings/structure/pdb_files"):
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDBParser(QUIET=True)
        self.ppb = PPBuilder()
        
    def extract_sequence_and_coords(self, pdb_path: Union[str, Path]) -> Tuple[str, np.ndarray, str]:
        """
        Extract sequence and Cα coordinates from PDB file.
        
        Returns:
            sequence: Amino acid sequence
            ca_coords: (N, 3) array of Cα coordinates
            protein_id: Extracted protein ID from filename
        """
        pdb_path = Path(pdb_path)
        protein_id = self._extract_protein_id(pdb_path.name)
        
        try:
            structure = self.parser.get_structure(protein_id, str(pdb_path))
            
            # Get first model (AlphaFold structures have single model)
            model = structure[0]
            
            # Extract sequence and coordinates
            sequence = ""
            ca_coords = []
            
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard amino acid
                        try:
                            ca = residue['CA']
                            ca_coords.append(ca.coord)
                            # Get amino acid type
                            aa = three_to_one(residue.resname)
                            sequence += aa
                        except KeyError:
                            # Skip residues without CA
                            continue
                            
            # Truncate to max length 1024
            if len(sequence) > 1024:
                logger.info(f"Truncating {protein_id} from {len(sequence)} to 1024 residues")
                sequence = sequence[:1024]
                ca_coords = ca_coords[:1024]
                
            return sequence, np.array(ca_coords, dtype=np.float32), protein_id
            
        except Exception as e:
            logger.error(f"Failed to process {pdb_path}: {e}")
            raise
            
    def _extract_protein_id(self, pdb_filename: str) -> str:
        """Extract UniProt ID from AlphaFold PDB filename."""
        # Expected format: AF-<UniProtID>-F1-model_v4.pdb
        base = pdb_filename.replace('.pdb', '')
        parts = base.split('-')
        if len(parts) >= 2 and parts[0] == 'AF':
            return parts[1]
        return base
        
    def process_all_pdbs(self) -> Dict[str, Tuple[str, np.ndarray]]:
        """Process all PDB files in directory."""
        pdb_files = list(self.pdb_dir.glob("*.pdb"))
        results = {}
        failed = []
        
        logger.info(f"Processing {len(pdb_files)} PDB files...")
        
        for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
            try:
                seq, coords, pid = self.extract_sequence_and_coords(pdb_path)
                results[pid] = (seq, coords)
            except Exception as e:
                failed.append((pdb_path.name, str(e)))
                
        logger.info(f"Successfully processed: {len(results)}/{len(pdb_files)} PDB files")
        if failed:
            logger.warning(f"Failed to process {len(failed)} files")
            for fname, error in failed[:5]:  # Show first 5 failures
                logger.warning(f"  {fname}: {error}")
                
        return results


class GraphConstructor:
    """Construct k-NN or radius graphs from Cα coordinates."""
    
    def __init__(self, graph_type: str = "knn", k: int = 10, radius: float = 10.0):
        """
        Args:
            graph_type: "knn" or "radius"
            k: Number of neighbors for k-NN graph
            radius: Cutoff distance for radius graph (in Angstroms)
        """
        self.graph_type = graph_type
        self.k = k
        self.radius = radius
        
    def construct_graph(self, ca_coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct graph from Cα coordinates.
        
        Returns:
            edge_index: (2, E) tensor of edges
            edge_attr: (E, D) tensor of edge features
            node_pos: (N, 3) tensor of node positions
        """
        n_nodes = len(ca_coords)
        node_pos = torch.from_numpy(ca_coords).float()
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(node_pos, node_pos)
        
        if self.graph_type == "knn":
            edge_index, edge_attr = self._construct_knn_graph(dist_matrix, node_pos)
        elif self.graph_type == "radius":
            edge_index, edge_attr = self._construct_radius_graph(dist_matrix, node_pos)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
            
        return edge_index, edge_attr, node_pos
        
    def _construct_knn_graph(self, dist_matrix: torch.Tensor, node_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct k-NN graph."""
        n_nodes = dist_matrix.size(0)
        
        # Get k nearest neighbors (excluding self)
        _, indices = torch.topk(dist_matrix, k=min(self.k + 1, n_nodes), largest=False)
        indices = indices[:, 1:]  # Remove self-connections
        
        # Build edge index
        row_indices = torch.arange(n_nodes).view(-1, 1).expand(-1, indices.size(1))
        edge_index = torch.stack([row_indices.flatten(), indices.flatten()], dim=0)
        
        # Compute edge features
        edge_attr = self._compute_edge_features(edge_index, node_pos)
        
        return edge_index, edge_attr
        
    def _construct_radius_graph(self, dist_matrix: torch.Tensor, node_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct radius graph."""
        # Find all pairs within radius
        mask = (dist_matrix < self.radius) & (dist_matrix > 0)  # Exclude self
        edge_index = torch.nonzero(mask).t()
        
        # Compute edge features
        edge_attr = self._compute_edge_features(edge_index, node_pos)
        
        return edge_index, edge_attr
        
    def _compute_edge_features(self, edge_index: torch.Tensor, node_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute edge features: distance and unit direction vector.
        
        Returns:
            edge_attr: (E, 4) tensor [distance, dx, dy, dz]
        """
        src, dst = edge_index
        
        # Compute vectors between connected nodes
        edge_vec = node_pos[dst] - node_pos[src]  # (E, 3)
        
        # Compute distances
        edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)  # (E, 1)
        
        # Normalize direction vectors
        edge_dir = edge_vec / (edge_dist + 1e-8)  # (E, 3)
        
        # Concatenate features
        edge_attr = torch.cat([edge_dist, edge_dir], dim=1)  # (E, 4)
        
        return edge_attr


class StructureGraphDataset(torch.utils.data.Dataset):
    """Dataset that constructs graphs on-the-fly from PDB structures."""
    
    def __init__(self,
        pdb_dir: str,
        esm_embedding_dir: str,
        names_npy: str,
        labels_npy: Optional[str] = None,
        graph_type: str = "knn",
        k: int = 10,
        radius: float = 10.0,
        use_esm_node_features: bool = True,
        cache_graphs: bool = False
    ):
        self.pdb_processor = PDBProcessor(pdb_dir)
        self.graph_constructor = GraphConstructor(graph_type, k, radius)
        self.esm_embedding_dir = Path(esm_embedding_dir)
        self.use_esm_node_features = use_esm_node_features
        self.cache_graphs = cache_graphs
        self.graph_cache = {}
        
        # Load names
        self.names = np.load(names_npy)
        
        # Load labels if provided
        self.labels = None
        if labels_npy is not None:
            if labels_npy.endswith('.npy'):
                self.labels = torch.from_numpy(np.load(labels_npy)).float()
            elif labels_npy.endswith('.npz'):
                import scipy.sparse as ssp
                self.labels = torch.from_numpy(ssp.load_npz(labels_npy).toarray()).float()
                
        # Validate PDB files exist
        self._validate_pdb_files()
        
    def _validate_pdb_files(self):
        """Check which proteins have PDB files available."""
        pdb_dir = Path(self.pdb_processor.pdb_dir)
        available_pdbs = set()
        missing_pdbs = []
        
        for name in self.names:
            pdb_path = pdb_dir / f"AF-{name}-F1-model_v4.pdb"
            if pdb_path.exists():
                available_pdbs.add(name)
            else:
                missing_pdbs.append(name)
                
        logger.info(f"Found PDB files for {len(available_pdbs)}/{len(self.names)} proteins")
        if missing_pdbs:
            logger.warning(f"Missing PDB files for {len(missing_pdbs)} proteins")
            
        # Filter to only available proteins
        self.valid_indices = [i for i, name in enumerate(self.names) if name in available_pdbs]
        self.valid_names = [self.names[i] for i in self.valid_indices]
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        name = self.names[actual_idx]

        # Check cache
        if self.cache_graphs and name in self.graph_cache:
            graph_data = self.graph_cache[name]
        else:

            # Process PDB
            pdb_path = self.pdb_processor.pdb_dir / f"AF-{name}-F1-model_v4.pdb"
            seq, ca_coords, _ = self.pdb_processor.extract_sequence_and_coords(pdb_path)
            
            # Construct graph
            edge_index, edge_attr, node_pos = self.graph_constructor.construct_graph(ca_coords)
            
            # Load node features
            if self.use_esm_node_features:
                node_features = self._load_esm_features(name, len(seq))
            else:
                # One-hot encode amino acids
                node_features = self._encode_amino_acids(seq)
                
            graph_data = {
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'node_pos': node_pos,
                'num_nodes': len(seq)
            }
            
            if self.cache_graphs:
                self.graph_cache[name] = graph_data

                
        # Add label if available
        if self.labels is not None:
            graph_data['label'] = self.labels[actual_idx]
            
        return name, graph_data
        
    def _load_esm_features(self, name: str, seq_len: int) -> torch.Tensor:
        """Load ESM embeddings for node features."""
        esm_path = self.esm_embedding_dir / f"{name}.npy"
        
        if not esm_path.exists():
            logger.warning(f"ESM embedding not found for {name}, using one-hot encoding")
            # Fallback to one-hot encoding
            pdb_path = self.pdb_processor.pdb_dir / f"AF-{name}-F1-model_v4.pdb"
            seq, _, _ = self.pdb_processor.extract_sequence_and_coords(pdb_path)
            return self._encode_amino_acids(seq)
            
        # Load ESM embeddings
        data = np.load(esm_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
            
        embeddings = data['embedding']  # Should be (seq_len, 1280)
        
        # Ensure correct length
        if embeddings.shape[0] != seq_len:
            logger.warning(f"ESM embedding length mismatch for {name}: {embeddings.shape[0]} vs {seq_len}")
            # Truncate or pad as needed
            if embeddings.shape[0] > seq_len:
                embeddings = embeddings[:seq_len]
            else:
                pad_len = seq_len - embeddings.shape[0]
                embeddings = np.pad(embeddings, ((0, pad_len), (0, 0)), mode='constant')
                
        return torch.from_numpy(embeddings).float()
        
    def _encode_amino_acids(self, sequence: str) -> torch.Tensor:
        """One-hot encode amino acid sequence."""
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        encoded = torch.zeros(len(sequence), 20)
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoded[i, aa_to_idx[aa]] = 1.0
            # Unknown amino acids remain as zero vector
                
        return encoded
        
    def get_summary(self):
        """Get summary statistics of the dataset."""
        summary = {
            'total_proteins': len(self.names),
            'proteins_with_pdb': len(self.valid_names),
            'missing_pdbs': len(self.names) - len(self.valid_names),
            'graph_type': self.graph_constructor.graph_type,
            'k': self.graph_constructor.k if self.graph_constructor.graph_type == 'knn' else None,
            'radius': self.graph_constructor.radius if self.graph_constructor.graph_type == 'radius' else None,
            'using_esm_features': self.use_esm_node_features
        }
        return summary