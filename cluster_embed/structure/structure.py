import os
import requests
from Bio import SeqIO
from tqdm import tqdm
from typing import Optional, List

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
    
    # Skip download if file already exists
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

def download_structures_from_fasta(fasta_file: str, output_dir: str):
    """
    Download AlphaFold structures for all proteins in a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        output_dir: Directory to save PDB files
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read protein IDs from FASTA file
    protein_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_ids.append(record.id)
    
    # Remove duplicates while preserving order
    unique_protein_ids = list(dict.fromkeys(protein_ids))
    
    print(f"Found {len(protein_ids)} total proteins, {len(unique_protein_ids)} unique proteins")
    
    # Track missing structures
    missing_proteins: List[str] = []
    
    # Download structures
    for protein_id in tqdm(unique_protein_ids, desc="Downloading AlphaFold structures"):
        pdb_file = download_alphafold_structure(protein_id, output_dir)
        
        if pdb_file is None:
            missing_proteins.append(protein_id)
    
    # Write IDs with no structure to a text file
    if missing_proteins:
        missing_file = os.path.join(output_dir, "missing_structures.txt")
        with open(missing_file, "w") as f:
            for pid in missing_proteins:
                f.write(pid + "\n")
        print(f"{len(missing_proteins)} proteins had no structural data; IDs written to {missing_file}")
    
    print(f"Downloaded {len(unique_protein_ids) - len(missing_proteins)} structures to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Download structures from FASTA file
    download_structures_from_fasta(
        fasta_file="/SAN/bioinf/PFP/dataset/CAFA5_small/filtered_train_seq.fasta",
        output_dir="/SAN/bioinf/PFP/embeddings/structure/pdb_files"
    )   