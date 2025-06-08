import os
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import PDBParser, DSSP, NACCESS
from Bio.SeqUtils import seq1
import requests
import gzip
import tempfile
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def download_alphafold_structure(uniprot_id: str, output_dir: str) -> Optional[str]:
    """
    Download AlphaFold structure for a given UniProt ID.
    
    Args:
        uniprot_id: UniProt identifier
        output_dir: Directory to save PDB files
        
    Returns:
        Path to downloaded PDB file or None if failed
    """
    
    pdb_file = os.path.join(output_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")
    
    if os.path.exists(pdb_file):
        return pdb_file
    
    # AlphaFold URL pattern
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(pdb_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return pdb_file
        
    except requests.exceptions.RequestException:
        return None

def compute_secondary_structure_features(structure, chain_id: str = 'A') -> np.ndarray:
    """
    Compute secondary structure features using DSSP.
    """
    try:
        # Run DSSP
        dssp = DSSP(structure[0], structure, dssp='dssp')
        
        ss_counts = {'H': 0, 'B': 0, 'E': 0, 'G': 0, 'I': 0, 'T': 0, 'S': 0, '-': 0}
        total_residues = 0
        
        for key in dssp.keys():
            ss = dssp[key][2]  # Secondary structure
            if ss in ss_counts:
                ss_counts[ss] += 1
            total_residues += 1
        
        # Convert to fractions
        if total_residues > 0:
            ss_features = [ss_counts[ss] / total_residues for ss in ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']]
        else:
            ss_features = [0.0] * 8
            
        return np.array(ss_features, dtype=np.float32)
        
    except:
        return np.zeros(8, dtype=np.float32)

def compute_geometric_features(structure, chain_id: str = 'A') -> np.ndarray:
    """
    Compute geometric and structural features.
    """
    try:
        chain = structure[0][chain_id]
        residues = [r for r in chain if r.get_id()[0] == ' ']  # Only standard residues
        
        if len(residues) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Extract CA coordinates
        ca_coords = []
        for residue in residues:
            if 'CA' in residue:
                ca_coords.append(residue['CA'].get_coord())
        
        if len(ca_coords) < 3:
            return np.zeros(10, dtype=np.float32)
        
        ca_coords = np.array(ca_coords)
        
        # Compute geometric features
        features = []
        
        # 1. Radius of gyration
        center = np.mean(ca_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((ca_coords - center)**2, axis=1)))
        features.append(rg)
        
        # 2. Maximum distance
        from scipy.spatial.distance import pdist
        distances = pdist(ca_coords)
        max_dist = np.max(distances)
        features.append(max_dist)
        
        # 3. Mean pairwise distance
        mean_dist = np.mean(distances)
        features.append(mean_dist)
        
        # 4. Compactness (rg / max_dist)
        compactness = rg / max_dist if max_dist > 0 else 0
        features.append(compactness)
        
        # 5. Asphericity (measure of deviation from spherical shape)
        inertia_tensor = np.cov(ca_coords.T)
        eigenvals = np.linalg.eigvals(inertia_tensor)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
        if eigenvals[0] > 0:
            asphericity = (eigenvals[0] - 0.5 * (eigenvals[1] + eigenvals[2])) / eigenvals[0]
        else:
            asphericity = 0
        features.append(asphericity)
        
        # 6-10. Statistical moments of distance distribution
        features.append(np.std(distances))     # Standard deviation
        features.append(np.min(distances))     # Minimum distance
        features.extend(np.percentile(distances, [25, 50, 75]))  # Quartiles
        
        return np.array(features, dtype=np.float32)
        
    except:
        return np.zeros(10, dtype=np.float32)

def compute_bfactor_features(structure, chain_id: str = 'A') -> np.ndarray:
    """
    Compute B-factor (confidence) related features from AlphaFold.
    """
    try:
        chain = structure[0][chain_id]
        bfactors = []
        
        for residue in chain:
            if residue.get_id()[0] == ' ':  # Standard residue
                for atom in residue:
                    bfactors.append(atom.get_bfactor())
                    break  # Just use first atom (usually CA)
        
        if len(bfactors) == 0:
            return np.zeros(6, dtype=np.float32)
        
        bfactors = np.array(bfactors)
        
        # AlphaFold confidence regions
        very_high = np.mean(bfactors > 90)  # Very high confidence
        confident = np.mean((bfactors > 70) & (bfactors <= 90))  # Confident
        low = np.mean((bfactors > 50) & (bfactors <= 70))  # Low confidence
        very_low = np.mean(bfactors <= 50)  # Very low confidence
        
        # Statistical features
        mean_conf = np.mean(bfactors)
        std_conf = np.std(bfactors)
        
        return np.array([very_high, confident, low, very_low, mean_conf, std_conf], dtype=np.float32)
        
    except:
        return np.zeros(6, dtype=np.float32)

def compute_amino_acid_features(structure, chain_id: str = 'A') -> np.ndarray:
    """
    Compute amino acid composition and properties.
    """
    try:
        chain = structure[0][chain_id]
        aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        
        for residue in chain:
            if residue.get_id()[0] == ' ':  # Standard residue
                resname = residue.get_resname()
                if resname in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                              'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 
                              'THR', 'VAL', 'TRP', 'TYR']:
                    aa = seq1(resname)
                    aa_counts[aa] += 1
        
        total = sum(aa_counts.values())
        if total == 0:
            return np.zeros(20, dtype=np.float32)
        
        # Convert to frequencies
        aa_freq = np.array([aa_counts[aa] / total for aa in 'ACDEFGHIKLMNPQRSTVWY'], dtype=np.float32)
        
        return aa_freq
        
    except:
        return np.zeros(20, dtype=np.float32)

def compute_structure_features(pdb_file: str, embedding_dim: int = 512) -> np.ndarray:
    """
    Compute comprehensive structural features from a PDB file.
    """
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_file)
        
        # Different feature types
        ss_features = compute_secondary_structure_features(structure)      # 8 features
        geom_features = compute_geometric_features(structure)              # 10 features  
        conf_features = compute_bfactor_features(structure)                # 6 features
        aa_features = compute_amino_acid_features(structure)               # 20 features
        
        # Combine all features
        all_features = np.concatenate([ss_features, geom_features, conf_features, aa_features])
        
        # Pad or truncate to desired dimension
        if len(all_features) < embedding_dim:
            # Pad with zeros
            padded = np.zeros(embedding_dim, dtype=np.float32)
            padded[:len(all_features)] = all_features
            return padded
        else:
            # Truncate
            return all_features[:embedding_dim].astype(np.float32)
            
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return np.zeros(embedding_dim, dtype=np.float32)

def generate_structure_embeddings(
    fasta_file: str,
    output_dir: str = "./structure_embeddings",
    embedding_dim: int = 512,
    use_alphafold: bool = True,
    pdb_dir: Optional[str] = None
):
    """
    Generate structure-based embeddings for proteins.
    
    Args:
        fasta_file: Path to FASTA file
        output_dir: Directory to save embeddings
        embedding_dim: Dimension of output embeddings
        use_alphafold: Whether to download AlphaFold structures
        pdb_dir: Directory containing existing PDB files
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create directory for PDB files if using AlphaFold
    if use_alphafold:
        pdb_download_dir = os.path.join(output_dir, "pdb_files")
        os.makedirs(pdb_download_dir, exist_ok=True)
    elif pdb_dir:
        pdb_download_dir = pdb_dir
    else:
        raise ValueError("Must specify either use_alphafold=True or provide pdb_dir")
    
    # Read protein IDs
    protein_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_ids.append(record.id)
    # Collect IDs that have no structure so we can record them later
    missing_proteins: List[str] = []
    
    print(f"Processing {len(protein_ids)} proteins for structure embeddings...")
    
    # Process each protein
    for protein_id in tqdm(protein_ids, desc="Generating structure embeddings"):
        output_file = os.path.join(output_dir, f"{protein_id}.npy")
        
        if os.path.exists(output_file):
            continue
        
        # Get or download structure
        if use_alphafold:
            pdb_file = download_alphafold_structure(protein_id, pdb_download_dir)
        else:
            # Look for existing PDB file
            pdb_file = os.path.join(pdb_download_dir, f"{protein_id}.pdb")
            if not os.path.exists(pdb_file):
                # Try AlphaFold naming convention
                pdb_file = os.path.join(pdb_download_dir, f"AF-{protein_id}-F1-model_v4.pdb")
                if not os.path.exists(pdb_file):
                    pdb_file = None
        
        if pdb_file is None:
            # Record the protein ID and create a zero embedding
            missing_proteins.append(protein_id)
            embedding = np.zeros(embedding_dim, dtype=np.float32)
        else:
            # Compute structural features
            embedding = compute_structure_features(pdb_file, embedding_dim)
        
        # Save embedding
        np.save(
            output_file,
            {"name": protein_id, "embedding": embedding},
            allow_pickle=True
        )
    
    # Write IDs with no structure to a text file
    if missing_proteins:
        missing_file = os.path.join(output_dir, "missing_structures.txt")
        with open(missing_file, "w") as f:
            for pid in missing_proteins:
                f.write(pid + "\n")
        print(f"{len(missing_proteins)} proteins had no structural data; IDs written to {missing_file}")
    print(f"Structure embeddings saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Generate structure embeddings using AlphaFold
    generate_structure_embeddings(
        fasta_file="/SAN/bioinf/PFP/dataset/CAFA5_small/filtered_train_seq.fasta",
        output_dir="/SAN/bioinf/PFP/embeddings/structure",
        embedding_dim=512,
        use_alphafold=True
        
    )