import os
import numpy as np
import torch
import pandas as pd
from Bio.PDB import PDBParser
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import networkx as nx
import random
import warnings
from tqdm import tqdm
from typing import Optional, List
import tempfile

warnings.filterwarnings('ignore')


class Graph:
    """Node2vec graph walker implementation"""
    
    def __init__(self, nx_G, is_directed=False, p=0.8, q=1.2):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """Simulate a random walk starting from start node."""
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_node = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """Repeatedly simulate random walks from each node."""
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        """Get the alias edge setup lists for a given edge."""
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1.0/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(1.0/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)
    
    def preprocess_transition_probs(self):
        """Preprocessing of transition probabilities for guiding the random walks."""
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [1.0 for nbr in range(len(list(G.neighbors(node))))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    @staticmethod
    def alias_setup(probs):
        """Compute utility lists for non-uniform sampling from discrete distributions."""
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    @staticmethod
    def alias_draw(J, q):
        """Draw sample from a non-uniform discrete distribution using alias sampling."""
        K = len(J)
        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]


def seq2onehot(seq):
    """Create 26-dim one-hot encoding for amino acid sequence"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed.get(v, vocab_embed['X']) for v in seq]  # Use 'X' for unknown
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def learn_node2vec_embeddings(walks, nums_node, embedding_dim=30):
    """Learn embeddings by optimizing the Skipgram objective using SGD."""
    # Create temporary file for walks
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        walk_file = f.name
        for walk in walks:
            line = ' '.join(map(str, walk)) + '\n'
            f.write(line)
    
    try:
        walks_iter = LineSentence(walk_file)
        model = Word2Vec(walks_iter, vector_size=embedding_dim, window=5, min_count=0, 
                        hs=1, sg=1, workers=1, epochs=3)  # workers=1 for stability
        
        vectors = []
        for i in range(nums_node):
            if str(i) in model.wv.key_to_index:
                vectors.append(list(model.wv[str(i)]))
            else:
                vectors.append(list(np.zeros(embedding_dim)))
        
        return np.array(vectors)
    
    finally:
        # Clean up temporary file
        if os.path.exists(walk_file):
            os.remove(walk_file)


def add_sequence_features(node2vec_vectors, sequence_onehot):
    """Combine node2vec vectors with sequence one-hot encoding"""
    if node2vec_vectors.shape[0] != sequence_onehot.shape[0]:
        if node2vec_vectors.shape[0] < sequence_onehot.shape[0]:
            sequence_onehot = sequence_onehot[:node2vec_vectors.shape[0], :]
        else:
            # Pad sequence one-hot with zeros
            padding = np.zeros((node2vec_vectors.shape[0] - sequence_onehot.shape[0], sequence_onehot.shape[1]))
            sequence_onehot = np.vstack((sequence_onehot, padding))

    combined_features = np.hstack((node2vec_vectors, sequence_onehot))
    return combined_features


class ProteinStructureProcessor:
    """Enhanced protein structure processor following the reference architecture"""
    
    def __init__(self, contact_threshold: float = 10.0):
        self.contact_threshold = contact_threshold
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
    def parse_pdb_structure(self, pdb_file: str) -> Optional[tuple]:
        """Parse PDB file to extract amino acid sequence and Ca coordinates."""
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
                        ca_atom = residue['CA']
                        ca_coords.append(ca_atom.get_coord())
                        
                        aa_name = residue.get_resname()
                        aa_1letter = self._three_to_one_letter(aa_name)
                        amino_acids.append(aa_1letter)
                        
                    except KeyError:
                        continue
            
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
        return conversion.get(three_letter, 'X')
    
    def create_contact_map(self, ca_coords: np.ndarray) -> np.ndarray:
        """Create contact map distance matrix from Ca coordinates."""
        n_residues = len(ca_coords)
        print(f"  → Creating contact map for {n_residues} residues")
        
        distances = np.zeros((n_residues, n_residues))
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def distance_to_contact_graph(self, distances: np.ndarray) -> nx.Graph:
        """Convert distance matrix to contact graph based on threshold."""
        n_residues = distances.shape[0]
        G = nx.Graph()
        
        # Add all nodes
        G.add_nodes_from(range(n_residues))
        
        # Add edges based on contact threshold
        edges = []
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                if distances[i, j] <= self.contact_threshold and i != j:
                    edges.append((i, j))
        
        G.add_edges_from(edges)
        
        print(f"  → Contact graph created: {len(edges)} edges with threshold {self.contact_threshold}Å")
        return G


class ProteinStructureEmbedder:
    """Main class for processing proteins and generating structure embeddings"""
    
    def __init__(self, contact_threshold: float = 10.0, node2vec_dim: int = 30, 
                 final_embedding_dim: int = 512):
        """
        Initialize the protein structure embedder.
        
        Args:
            contact_threshold: Distance threshold for contact map (Angstroms)
            node2vec_dim: Dimension of node2vec structural embeddings
            final_embedding_dim: Final embedding dimension (for compatibility, actual dim will be node2vec_dim + 26)
        """
        self.processor = ProteinStructureProcessor(contact_threshold)
        self.node2vec_dim = node2vec_dim
        self.final_embedding_dim = final_embedding_dim
        
        # Node2vec parameters (following reference)
        self.p = 0.8
        self.q = 1.2
        self.num_walks = 5
        self.walk_length = 30
    
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
        sequence = ''.join(amino_acids)
        
        # Step 2: Create contact map and convert to graph
        distances = self.processor.create_contact_map(ca_coords)
        contact_graph = self.processor.distance_to_contact_graph(distances)
        
        if contact_graph.number_of_nodes() == 0:
            print(f"  ✗ Empty contact graph for {os.path.basename(pdb_file)}")
            return None
        
        # Step 3: Node2vec on contact graph
        print(f"  → Running node2vec on graph with {contact_graph.number_of_nodes()} nodes")
        
        try:
            graph_walker = Graph(contact_graph, is_directed=False, p=self.p, q=self.q)
            graph_walker.preprocess_transition_probs()
            
            walks = graph_walker.simulate_walks(self.num_walks, self.walk_length)
            node2vec_vectors = learn_node2vec_embeddings(walks, 
                                                       contact_graph.number_of_nodes(), 
                                                       self.node2vec_dim)
            
            print(f"  → Node2vec embeddings: {node2vec_vectors.shape}")
            
        except Exception as e:
            print(f"  ✗ Error in node2vec: {e}")
            return None
        
        # Step 4: Create sequence one-hot encoding
        sequence_onehot = seq2onehot(sequence)
        print(f"  → Sequence one-hot: {sequence_onehot.shape}")
        
        # Step 5: Combine node2vec with sequence features
        combined_features = add_sequence_features(node2vec_vectors, sequence_onehot)
        print(f"  → Combined features: {combined_features.shape}")
        
        # Step 6: Global pooling to get final protein representation
        # Use mean and max pooling like in the original, but simpler
        mean_pool = np.mean(combined_features, axis=0)
        max_pool = np.max(combined_features, axis=0)
        final_embedding = np.concatenate([mean_pool, max_pool])
        
        # Optional: project to target dimension if specified
        if self.final_embedding_dim and self.final_embedding_dim != final_embedding.shape[0]:
            # Simple linear projection (you could make this learnable)
            projection_matrix = np.random.normal(0, 0.1, 
                                               (final_embedding.shape[0], self.final_embedding_dim))
            final_embedding = np.dot(final_embedding, projection_matrix)
        
        print(f"  ✓ Generated embedding: shape {final_embedding.shape}")
        return final_embedding
    
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
        print(f"Architecture: Contact Map → Node2vec → Sequence One-hot → Global Pooling")
        print(f"Node2vec dim: {self.node2vec_dim}, Final dim: {self.final_embedding_dim}")
        
        successful = 0
        failed = 0
        
        for pdb_file in tqdm(pdb_files, desc="Processing proteins"):
            # Extract protein ID from filename
            if pdb_file.startswith('AF-'):
                protein_id = pdb_file.replace('AF-', '').replace('-F1-model_v4.pdb', '')
            else:
                protein_id = pdb_file.replace('.pdb', '')
            
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
    Main function to run the refined structure embedding pipeline.
    """
    # Configuration
    pdb_input_dir = "/SAN/bioinf/PFP/embeddings/structure/pdb_files"
    embedding_output_dir = "/SAN/bioinf/PFP/embeddings/graph3"
    
    # Initialize embedder with refined architecture
    print("=== Refined Protein Structure Embedding Pipeline ===")
    print("Architecture: PDB → Contact Map → Node2vec + Sequence One-hot → Global Pooling")
    print("Following proven architecture from reference paper")
    print()
    
    embedder = ProteinStructureEmbedder(
        contact_threshold=10.0,    # 10 Angstroms for contact map
        node2vec_dim=30,           # 30-dim node2vec (matching reference)
        final_embedding_dim=512    # 512-dim final embeddings
    )
    
    # Process proteins
    embedder.process_protein_directory(
        pdb_dir=pdb_input_dir,
        output_dir=embedding_output_dir
        # max_proteins=4  # Remove this parameter to process all proteins
    )
    
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()