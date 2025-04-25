
from tqdm import tqdm
import argparse
import os
import pandas as pd
from tqdm import tqdm 
from Prot2Text.prot2text_dataset.pdb2graph import *
from itertools import combinations
import torch
from torch_geometric.data import Data
from ProteinGraphRGCN import ProteinGraphRGCN
import glob
import multiprocessing as mp
import wget
import Bio
from Bio import SeqIO
from Bio.PDB import PDBParser
import Bio.PDB.Polypeptide as poly
import networkx as nx
from Bio.PDB.DSSP import DSSP
from functools import partial
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes import (
    amino_acid_one_hot,
    expasy_protein_scale,
    meiler_embedding,
    rsa,
    secondary_structure,
    phi,
    psi
)
# from graphein.protein.config import ProteinGraphConfig, DSSPConfig
# from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor
# from graphein.protein.features.nodes.dssp import  phi, psi, asa, rsa, secondary_structure
# from graphein.protein.edges.distance import (add_peptide_bonds,
#                                              add_hydrogen_bond_interactions,
#                                              add_disulfide_interactions,
#                                              add_ionic_interactions,
#                                              add_delaunay_triangulation,
#                                              add_distance_threshold,
#                                              add_sequence_distance_edges,
#                                              add_k_nn_edges)


# class Structure:
#     def __init__(self,


#     ):
        


import matplotlib.pyplot as plt


class ProteinGraphPipeline:
    # def __init__(self, 
    #             fasta_file: Optional[str] = None,
    #             cache_dir: Optional[str] = None,
    #             filtered_fasta_file: Optional[str] = None,
    #             processed_folder: Optional[str] = None):
    # self.fasta_file = fasta_file
    # self.cache_dir = cache_dir
    # self.filtered_fasta_file = filtered_fasta_file
    # self.processed_folder = processed_folder
    def __init__(self, graph_storage = 'new_embedding/structure/PyG'):
        """
        Initialize any default parameters you might need.
        """
        # self.out_dir = pdb_storage
        self.graph_storage = graph_storage

    # def plot_sequential_edges(self, G):
    #         """
    #         Plots the graph G and highlights edges whose "type" attribute equals "sequential".
    #         """
    #         # 1) Separate edges by attribute
    #         seq_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "sequential"]
    #         other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") != "sequential"]

    #         # 2) Choose a layout
    #         pos = nx.spring_layout(G, seed=42)  # or any other layout

    #         # 3) Draw nodes
    #         nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")

    #         # 4) Draw "sequential" edges in one color (e.g., red)
    #         nx.draw_networkx_edges(G, pos, edgelist=seq_edges, edge_color="red", width=2)

    #         # 5) Draw the remaining edges in another color (e.g., gray)
    #         nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color="gray", style="dashed")

    #         # 6) Optionally, add labels
    #         nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    #         # 7) Hide axis & show
    #         plt.axis("off")
    #         plt.title("Graph Highlighting")
    #         plt.show()


    # def parse_fasta(self, fasta_file=None)->dict:
    #         '''
    #         parse fasta file

    #         args:
    #             fasta_file: fasta file path
    #         return:
    #             fasta_dict: fasta dictionary {id: sequence}
    #         '''


    #         fasta_dict = {}
    #         for record in SeqIO.parse(fasta_file, 'fasta'):
    #             fasta_dict[record.id] = str(record.seq)
    #         return fasta_dict  

    def extract_prefixes(self, folder_path):
        """
        Extracts filename prefixes (basename without extension) 
        for all files in the specified folder.
        
        Args:
            folder_path (str): Path to the folder containing files.
            
        Returns:
            List[str]: A list of filename prefixes.
        """
        prefixes = []
        
        # List all files in the folder
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            
            # Check if it is a file (and not a directory)
            if os.path.isfile(full_path):
                # Split off the extension
                prefix, _ = os.path.splitext(filename)
                prefixes.append(prefix)
        
        return prefixes

    def filter_id(self, fasta_file=None, cache_dir=None, filltered_fasta_file=None):
                """
                Find the protein id that in seuqnece dataset but not in the graph, maybe cause by the lack of Alphafold dataset
                Args:
                    fasta_file (str, optional): fasta file path. Defaults to None.
                """

                cache_dir = self.graph_storage
                print(fasta_file, cache_dir, filltered_fasta_file)

                # if fasta_file is None:
                #     fasta_file = self.fasta_file
                # if cache_dir is None: # place store esm embedding
                #     cache_dir = self.cache_dir
                # if filltered_fasta_file is None:
                #     filltered_fasta_file = self.filltered_fasta_file


                # Ensure the directory exists before listing files
                if not os.path.exists(cache_dir):
                    print(f"Directory '{cache_dir}' does not exist. Creating it now...")
                    os.makedirs(cache_dir)  # Create the directory and any missing parent directories



                fasta_dict = self.parse_fasta(fasta_file)
                # get processed fasta ids
                processed_ids = set()
                # 'AF-'+str(prot)+'-F1-model_v4.pdb'
                for file in os.listdir(cache_dir):
                    if file.endswith('.pdb'):
                        processed_ids.add(file.split('.')[0])
                # print(processed_ids)
                # exit()
    #             # filter fasta file
    #             # filltered_fasta_dict = {k for k,v in fasta_dict.items() if k not in processed_ids}
    #             # print(filltered_fasta_dict)


    #             wait_fold = set()
    #             wait_fold = {k for k,v in fasta_dict.items() if 'AF-'+k+'-F1-model_v4' not in processed_ids}


    #             return wait_fold
                # exit()
                # # write filtered fasta file
                # with open(filltered_fasta_file, 'w') as f:
                #     for k,v in filltered_fasta_dict.items():
                #         f.write(f'>{k}\n{v}\n')



                # return filltered_fasta_dict, filltered_fasta_file




    def download_alphafold_structure(
        self,
        uniprot_id: str,
        out_dir: str,
        version: int = 4
        ):
        
        BASE_URL = "https://alphafold.ebi.ac.uk/files/"
        uniprot_id = uniprot_id.upper()

        query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.pdb"
        structure_filename = os.path.join(out_dir, f"AF-{uniprot_id}-F1-model_v{version}.pdb")
        if os.path.exists(structure_filename):
            return structure_filename
        try:
            structure_filename = wget.download(query_url, out=out_dir)
        except:
            print('Error.. could not download: ', f"AF-{uniprot_id}-F1-model_v{version}.pdb")
            return None
        return structure_filename

    # def one_hot_aa(aa, amino_acids):

    def build_protein_graph(
        self,
        pdb_file: str,
        chain_id: str = "A",
        distance_threshold: float = 5.0,
        hbond_cutoff: float = 3.5
    ) -> nx.Graph:
        """
        Builds a protein graph from a PDB file with the following edge types:
        1) 'sequential': edges between consecutive residues in the chain
        2) 'spatial': edges between residues whose alpha-carbons are within
                        'distance_threshold' angstroms
        3) 'hbond': edges if the backbone O–N distance is below 'hbond_cutoff'

        :param pdb_file: Path to the PDB file.
        :param chain_id: Which chain to parse (default 'A').
        :param distance_threshold: Distance cutoff for spatial edges (in Å).
        :param hbond_cutoff: Distance cutoff for a simplistic H-bond check (in Å).
        :return: A NetworkX graph with node and edge attributes.
        """
        # 1. Parse the PDB structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("AF2_model", pdb_file)
        model = structure[0]  # Take the first model if multiple
        chain = model[chain_id]  # Select chain by ID
        residues = list(chain.get_residues())

        # 2. Initialize a NetworkX graph
        G = nx.Graph()


        # Iterate over residues to add nodes
        residue_list = []
        for chain in model:
            for res in chain:
                if res.get_id()[0] != " ":  # skip hetero residues if any
                    print("skip hetero residues")
                    continue
                res_id = res.get_id()[1]      # residue sequence number
                res_name = res.get_resname()  # three-letter code, e.g., "ALA"
                # Map three-letter code to one-letter code:
                res_one_letter = Bio.PDB.Polypeptide.three_to_index(res_name)
                residue_list.append((chain.id, res_id, res_one_letter, res))
                # Add node with initial features (we'll fill features next)
                node_index = len(residue_list) - 1
                G.add_node(node_index, amino_acid=res_one_letter)


        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

        # Example physicochemical property: hydrophobicity scale (Kyte-Doolittle)
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }




            # Compute and assign node feature vectors
        for node_idx, (_, _, one_letter, res) in enumerate(residue_list):
            # One-hot encoding for amino acid type
            # aa_onehot = one_hot_aa(one_letter, amino_acids)
            # print(res)
            # print(one_letter, amino_acids)

            aa_onehot = [0] * len(amino_acids)

            # if one_letter in amino_acids:
            aa_onehot[one_letter] = 1


            # Physicochemical features
            hydrophobic = hydrophobicity.get(one_letter, 0.0)
            # (We can add more properties like charge, polarity, etc.)
            features = aa_onehot + [hydrophobic]
            # Initially attach features to node
            G.nodes[node_idx]["feat"] = features

        # Here each node’s "feat" might start as a 21-dimensional vector (20 for one-hot + 1 for hydrophobicity). 
        # We can extend this with more properties (e.g., isPolar, molecular weight, etc.).


        dssp = DSSP(model, pdb_file)  # run DSSP on the structure
        for node_idx, (chain_id, res_id, aa, _) in enumerate(residue_list):
            dssp_key = (chain_id, res_id)
            if dssp_key in dssp:
                aa, ss, acc, phi, psi = dssp[dssp_key][1:6]  # DSSP tuple fields
                G.nodes[node_idx]["secondary_structure"] = ss  # e.g. 'H', 'E', '-'
                G.nodes[node_idx]["asa"] = acc  # solvent accessibility
                G.nodes[node_idx]["phi"] = phi
                G.nodes[node_idx]["psi"] = psi
                # Optionally, include these in the feature vector
                G.nodes[node_idx]["feat"].extend([acc, phi, psi])



        # Pre-compute coordinates for residue alpha-carbons (for distance-based edges)
        ca_coords = [res['CA'].get_coord() for (_,_,_, res) in residue_list]

        # Peptide bond edges (sequential neighbors in sequence)
        for i in range(len(residue_list) - 1):
            G.add_edge(i, i+1, type="peptide")


        # Hydrogen bond edges (using a simple distance cutoff between donor-H and acceptor)
        # For demonstration, we consider any N-O distance < 3.5Å as a hydrogen bond (this is a simplification)
        for i, j in combinations(range(len(residue_list)), 2):
            atom_N = residue_list[i][3]['N'] if 'N' in residue_list[i][3] else None
            atom_O = residue_list[j][3]['O'] if 'O' in residue_list[j][3] else None
            if atom_N and atom_O:
                dist = atom_N - atom_O  # distance between N and O atoms
                if dist < hbond_cutoff and not G.has_edge(i, j):
                    G.add_edge(i, j, type="hydrogenbond")
            # Also check the other way around (j's N with i's O)
            atom_N2 = residue_list[j][3]['N'] if 'N' in residue_list[j][3] else None
            atom_O2 = residue_list[i][3]['O'] if 'O' in residue_list[i][3] else None
            if atom_N2 and atom_O2:
                dist = atom_N2 - atom_O2
                if dist < hbond_cutoff and not G.has_edge(i, j):
                    G.add_edge(i, j, type="hydrogenbond")

        # Spatial proximity edges (distance threshold between Cα atoms)
        threshold = distance_threshold  # 5 Å cutoff for considering spatial neighbors
        for i, j in combinations(range(len(residue_list)), 2):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < threshold and not G.has_edge(i, j):
                G.add_edge(i, j, type="spatial")

        return G

    def  PrepareGraphData(self, G):
        # Collect node feature matrix and edge index list
        num_nodes = G.number_of_nodes()
        # Node features: list of feature lists -> tensor
        x_feats = [G.nodes[i]["feat"] for i in range(num_nodes)]
        x = torch.tensor(x_feats, dtype=torch.float)

        # Edge indices and types
        edge_index_list = []
        edge_type_list = []
        type_to_id = {"peptide": 0, "hydrogenbond": 1, "spatial": 2}
        for u, v, attr in G.edges(data=True):
            edge_index_list.append([u, v])
            edge_type_list.append(type_to_id[attr['type']])
            # For undirected or symmetric edges, you may add both directions:
            edge_index_list.append([v, u])
            edge_type_list.append(type_to_id[attr['type']])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)


        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        data.num_nodes = num_nodes
        return data

    # It takes the path to a PDB file and returns a PyG Data object.
    def build_protein_graph_f(self, pdb_path):
        # Your implementation here; for demonstration, we use a placeholder.
        # For example, you might have:
        # data = construct_graph(config, pdb_path)
        # pyg_data = convert_nx_to_pyg_data(data)
        # return pyg_data
        # -------------------
        # Here we'll simulate a PyG Data object with dummy values:

        G = self.build_protein_graph(pdb_path, chain_id="A")
        data = self.PrepareGraphData(G)
        return data


    def process_single_pdb(self, pdb_path, processed_folder):
        """
        Process a single PDB file:
        - Extract the UniProt id from the filename.
        - Skip if the output file already exists.
        - Build the protein graph and save the PyG Data.
        """
        filename = os.path.basename(pdb_path)
        uniprot_id = self.parse_uniprot_id(filename)  # your function to extract ID
        out_file = os.path.join(processed_folder, f"{uniprot_id}.pt")
        
        if os.path.exists(out_file):
            print(f"Skipping {filename}: {uniprot_id}.pt already exists.")
            return filename  # indicate skipping
        
        try:
            data = self.build_protein_graph_f(pdb_path)  # your function to build the graph
            torch.save(data, out_file)
            print(f"Processed {filename} -> {uniprot_id}.pt")
            return filename  # processed successfully
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_pdb_folder_parallel(self, raw_folder, processed_folder, num_processes=None):
        """
        Process all PDB files in raw_folder in parallel.
        - Creates a pool of worker processes.
        - Uses imap_unordered to process files concurrently.
        - Displays a progress bar with tqdm.
        """
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        
        pdb_files = glob.glob(os.path.join(raw_folder, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files in {raw_folder}")
        
        # Wrap the function to pass the processed_folder to each worker
        
        process_func = partial(self.process_single_pdb, processed_folder=processed_folder)
        
        with mp.Pool(num_processes) as pool:
            # Use imap_unordered for a progress bar friendly parallel loop
            for _ in tqdm(pool.imap_unordered(process_func, pdb_files), total=len(pdb_files)):
                pass
                
    def parse_uniprot_id(self, pdb_filename):
        """
        Extracts the UniProt ID from a filename of the form:
        AF-<UniProtID>-F1-model_v4.pdb
        E.g. "AF-V9HWF5-F1-model_v4.pdb" -> "V9HWF5"
        """
        base, _ = os.path.splitext(pdb_filename)            # e.g. "AF-V9HWF5-F1-model_v4"
        parts = base.split("-")                             # ["AF", "V9HWF5", "F1", "model_v4"]
        uniprot_id = parts[1]                               # "V9HWF5"
        return uniprot_id
    # def nx_to_pyg_data(self, G):
    #     """
    #     Convert a NetworkX graph `G` into a PyG `Data` object.
    #     Assumes that each node i has a `feat` attribute: G.nodes[i]["feat"].
    #     """
    #     # 1) Collect node features
    #     x_feats = []
    #     for node_id in sorted(G.nodes()):
    #         x_feats.append(G.nodes[node_id]["feat"])
    #     x = torch.tensor(x_feats, dtype=torch.float)
        
    #     # 2) Build edge_index
    #     edge_index_list = []
    #     for u, v in G.edges():
    #         edge_index_list.append([u, v])
    #         edge_index_list.append([v, u])  # undirected or bidirectional

    #     edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        
    #     # If your Nx graph has edge types in G[u][v]["type"], do:
    #     edge_types = []
    #     type_map = {"peptide": 0, "hydrogenbond": 1, "spatial": 2}
    #     for u, v in G.edges():
    #         etype = G[u][v].get("type", "peptide")  # default to 'peptide' if missing
    #         edge_types.append(type_map.get(etype, 0))
    #         edge_types.append(type_map.get(etype, 0))

    #     edge_type = torch.tensor(edge_types, dtype=torch.long)

    #     # 3) Create and return the Data object
    #     data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    #     return data
if __name__ == "__main__":


    argParser = argparse.ArgumentParser()
    argParser.add_argument("--data_save_path", help="folder to save the dataset")
    argParser.add_argument("--fasta_path", help="fasta file containing the protein dataset")
    argParser.add_argument("--split", help="train, test or eval csv?")
    argParser.add_argument("--plm_model", help="protein model to use (from hugging face)")
    # argParser.add_argument("--decoder_model", help="language model to use (from hugging face)")
    # usage:
    # python prepare_dataset.py \
    #   --data_save_path ./data/dataset/ \
    #   --split test --fasta_path ./data/test.csv \
    #   --plm_model facebook/esm2_t12_35M_UR50D \

    # python structure.py --data_save_path new_embedding/structure --fasta_path filtered_train_seq.fasta
    args = argParser.parse_args()


    # # step 1: download the PDB files from AlphaFoldDB
    # isExist = os.path.exists(os.path.join(args.data_save_path, args.split))

    print('downloading the data:\n')



    pipeline = ProteinGraphPipeline()

    raw_folder = "new_embedding/structure/"         # Folder containing PDB files
    processed_folder = "new_embedding/structure/PyG"     # Folder where PyG .pt files will be saved
    # # process_pdb_folder(raw_folder, processed_folder)
    pipeline.process_pdb_folder_parallel(raw_folder, processed_folder,4)




    exit()
    # pdb_file = "new_embedding/structure/AF-A8B1U6-F1-model_v4.pdb"  # example path
    # G = pipeline.build_protein_graph(pdb_file, chain_id="A")
    # print("Number of nodes:", G.number_of_nodes())
    # print("Number of edges:", G.number_of_edges())
        
    # data = pipeline.build_protein_graph_f(pdb_file)
    data = torch.load("new_embedding/structure/PyG/A8B1U6.pt", weights_only=False)
    # nx_to_pyg_data# Convert it in memory:

    # Initialize the RGCN model
    in_dim = data.x.shape[1]         # input feature dimension (size of node feature vector)
    hidden_dim = 64
    out_dim = 64                     # dimension of graph embedding we want
    num_relations = 3  # 3 in our case
    model = ProteinGraphRGCN(in_dim, hidden_dim, out_dim, num_relations)
    graph_emb = model(data.x, data.edge_index, data.edge_type, batch=torch.zeros(data.num_nodes, dtype=torch.long))
    graph_emb = graph_emb.detach().numpy()
    print(data.x.shape)
    print(data.edge_index.shape)
    print(graph_emb.shape)

    
    exit()


    print("Check file exist")

    output_dir = args.data_save_path
    # output_dir = 'new_embedding/structure'

    train_seqs_fasta = args.fasta_path

    print(output_dir, train_seqs_fasta)




    # wait_fold_id = filter_fasta(train_seqs_fasta, output_dir)



    # exit()

    # for prot in tqdm(wait_fold_id):
    #     print(os.path.join(output_dir, 'AF-'+str(prot)+'-F1-model_v4.pdb'))

    #     if os.path.exists(os.path.join(output_dir, 'AF-'+str(prot)+'-F1-model_v4.pdb')):
    #         continue
    #     download_alphafold_structure(uniprot_id=str(prot), out_dir = output_dir)
    #     # break


    # step 2: construct graphs from the pdb files


    pdb_path = "new_embedding/structure/AF-A0R757-F1-model_v4.pdb"  # <-- Replace with your own file
    if not os.path.isfile(pdb_path):
        print(f"PDB file not found: {pdb_path}")
    else:



                # Build the protein graph
        G = build_protein_graph(pdb_path, chain_id="A")


        print("Number of nodes:", G.number_of_nodes())
        print("Number of edges:", G.number_of_edges())

        V = list(G.nodes(data=True))
        E = list(G.edges(data=True))


        data = PrepareGraphData(G)

        # Initialize the RGCN model
        in_dim = data.x.shape[1]         # input feature dimension (size of node feature vector)
        hidden_dim = 64
        out_dim = 64                     # dimension of graph embedding we want
        num_relations = 3  # 3 in our case
        model = ProteinGraphRGCN(in_dim, hidden_dim, out_dim, num_relations)
        graph_emb = model(data.x, data.edge_index, data.edge_type, batch=torch.zeros(data.num_nodes, dtype=torch.long))
        graph_emb = graph_emb.detach().numpy()
        print(data.x.shape)
        print(data.edge_index.shape)
        A = nx.to_numpy_array(G)
        print(A.shape)  # This will be (num_nodes, num_nodes)



        







