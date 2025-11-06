from pathlib import Path
import numpy as np
import h5py

# Step 0: Check what files exist
data_dir = Path("../data")
print(f"Looking in directory: {data_dir.absolute()}")
print(f"Directory exists: {data_dir.exists()}")

if data_dir.exists():
    print("\nFiles in data directory:")
    for f in data_dir.glob("*.npy"):
        print(f"  {f.name}")
else:
    print("\nDirectory not found! Please provide the correct path.")
    exit()

# Step 1: Build CAFA ID to UniProt mapping
cafa_to_uniprot = {}
cafa_dir = Path("/home/zijianzhou/Datasets/cafa3/Target files")

for fasta_file in cafa_dir.glob("*.fasta"):
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                parts = line[1:].strip().split()
                if len(parts) >= 2:
                    cafa_id = parts[0]
                    uniprot_id = parts[1]
                    cafa_to_uniprot[cafa_id] = uniprot_id

print(f"\nLoaded {len(cafa_to_uniprot)} CAFA to UniProt mappings")

# Step 2: Build alias to STRING ID mapping
alias_to_string = {}

print("Loading protein.aliases.v12.0.txt...")
with open('../data/string/protein.aliases.v12.0.txt', 'r') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            string_id = parts[0]
            alias = parts[1]
            
            if alias not in alias_to_string:
                alias_to_string[alias] = string_id

# Step 3: Load all proteins from embedding file
print("Loading embedding file proteins...")
embedding_proteins = set()

filename = '/home/zijianzhou/project/PFP/experiments/PLMs/data/string/protein.network.embeddings.v12.0.h5'
with h5py.File(filename, 'r') as f:
    for species_id in f['species'].keys():
        proteins = f['species'][species_id]['proteins'][:]
        proteins = [p.decode('utf-8') for p in proteins]
        embedding_proteins.update(proteins)

print(f"Total proteins in embedding file: {len(embedding_proteins)}")

# Step 4: Process each aspect and split
def map_and_count(proteins):
    """Map proteins to STRING and count how many are in embeddings"""
    mapped_to_string = 0
    in_embeddings = 0
    
    for protein in proteins:
        string_id = None
        
        if protein in alias_to_string:
            string_id = alias_to_string[protein]
            mapped_to_string += 1
        elif protein in cafa_to_uniprot:
            uniprot_id = cafa_to_uniprot[protein]
            if uniprot_id in alias_to_string:
                string_id = alias_to_string[uniprot_id]
                mapped_to_string += 1
        
        if string_id and string_id in embedding_proteins:
            in_embeddings += 1
    
    return mapped_to_string, in_embeddings

print("\n=== Coverage by Aspect and Split ===")
total_proteins = 0
total_mapped = 0
total_in_embeddings = 0

for aspect in ['BPO', 'CCO', 'MFO']:
    print(f"\n{aspect}:")
    for split in ['train', 'valid', 'test']:
        names_file = data_dir / f"{aspect}_{split}_names.npy"
        if names_file.exists():
            proteins = np.load(names_file, allow_pickle=True)
            n_proteins = len(proteins)
            mapped, in_emb = map_and_count(proteins)
            
            total_proteins += n_proteins
            total_mapped += mapped
            total_in_embeddings += in_emb
            
            print(f"  {split:6s}: {n_proteins:5d} proteins | "
                  f"mapped: {mapped:5d} ({mapped/n_proteins*100:5.1f}%) | "
                  f"in embeddings: {in_emb:5d} ({in_emb/n_proteins*100:5.1f}%)")

if total_proteins > 0:
    print(f"\n=== Overall Summary ===")
    print(f"Total proteins: {total_proteins}")
    print(f"Mapped to STRING: {total_mapped} ({total_mapped/total_proteins*100:.2f}%)")
    print(f"Found in embeddings: {total_in_embeddings} ({total_in_embeddings/total_proteins*100:.2f}%)")
else:
    print("\nNo protein files found! Check the file naming pattern.")