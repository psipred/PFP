#!/usr/bin/env python3
"""
Updated CAFA3 Dataset Integration for Multi-Modal GO Prediction
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/cafa3_integration.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml
import json
import requests
import scipy.sparse as ssp

# Add project root to path
sys.path.append('/SAN/bioinf/PFP/PFP')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAFA3DatasetPreparer:
    """Prepare CAFA3 dataset for multi-modal GO prediction."""
    
    def __init__(self, 
                 cafa3_dir: str = "/SAN/bioinf/PFP/dataset/zenodo",
                 output_dir: str = "/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/data",
                 small_subset: bool = False):
        
        self.cafa3_dir = Path(cafa3_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.small_subset = small_subset
        
        # Paths for different modalities
        self.esm_dir = Path("/SAN/bioinf/PFP/embeddings/cafa3/esm")
        self.struct_dir = Path("/SAN/bioinf/PFP/embeddings/cafa3/structures")
        self.text_dir = Path("/SAN/bioinf/PFP/embeddings/cafa3/text")
        
        # Create directories
        for dir_path in [self.esm_dir, self.struct_dir, self.text_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def load_cafa3_data(self, aspect: str, split: str) -> pd.DataFrame:
        """Load CAFA3 CSV file for given aspect and split."""
        aspect_map = {'BPO': 'bp', 'CCO': 'cc', 'MFO': 'mf'}
        filename = f"{aspect_map[aspect]}-{split}.csv"
        filepath = self.cafa3_dir / filename
        
        logger.info(f"Loading {filepath}")
        df = pd.read_csv(filepath)

        if self.small_subset:
            # Take small subset for testing
            n_samples = min(100, len(df))
            df = df.sample(n=n_samples, random_state=42)
            logger.info(f"Using subset of {n_samples} samples for testing")
            
        return df
    
    def prepare_dataset(self):
        """Prepare complete CAFA3 dataset."""
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            logger.info(f"\nProcessing {aspect}...")
            
            # Load train/val/test splits
            train_df = self.load_cafa3_data(aspect, 'training')
            val_df = self.load_cafa3_data(aspect, 'validation')
            test_df = self.load_cafa3_data(aspect, 'test')
            
            # Get GO terms (all columns except proteins and sequences)
            go_columns = [col for col in train_df.columns if col.startswith('GO:')]
            logger.info(f"Found {len(go_columns)} GO terms for {aspect}")
            


            # exit(train_df.columns[:5])
            # Check for data leakage
            self._check_data_leakage(train_df, val_df, test_df, aspect)

            # Process each split
            self._process_split(train_df, aspect, 'train', go_columns)
            self._process_split(val_df, aspect, 'valid', go_columns)
            self._process_split(test_df, aspect, 'test', go_columns)
            
            # Create dataset info
            self._create_dataset_info(aspect, go_columns)
            
    def _check_data_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, aspect: str):
        """Check for data leakage between splits."""



        train_proteins = set(train_df['proteins'].values)
        val_proteins = set(val_df['proteins'].values)
        test_proteins = set(test_df['proteins'].values)
        
        # Check overlaps
        train_val_overlap = train_proteins & val_proteins
        train_test_overlap = train_proteins & test_proteins
        val_test_overlap = val_proteins & test_proteins
        
        if train_val_overlap:
            logger.warning(f"{aspect}: {len(train_val_overlap)} proteins overlap between train and val")
        if train_test_overlap:
            logger.warning(f"{aspect}: {len(train_test_overlap)} proteins overlap between train and test")
        if val_test_overlap:
            logger.warning(f"{aspect}: {len(val_test_overlap)} proteins overlap between val and test")
            
        if not (train_val_overlap or train_test_overlap or val_test_overlap):
            logger.info(f"{aspect}: No data leakage detected ✓")
            
    def _process_split(self, df: pd.DataFrame, aspect: str, split: str, go_columns: List[str]):
        """Process a single data split."""
        
        # Extract protein IDs and sequences
        protein_ids = df['proteins'].values
        sequences = df['sequences'].values
        
        # Extract GO labels (binary 0/1)
        labels = df[go_columns].values.astype(np.float32)
        
        # Save protein names
        names_file = self.output_dir / f"{aspect}_{split}_names.npy"
        np.save(names_file, protein_ids)
        
        # Save labels as sparse matrix (more efficient for sparse labels)
        labels_sparse = ssp.csr_matrix(labels)
        labels_file = self.output_dir / f"{aspect}_{split}_labels.npz"
        ssp.save_npz(labels_file, labels_sparse)
        
        # Save sequences for embedding generation
        seq_file = self.output_dir / f"{aspect}_{split}_sequences.json"
        seq_dict = {pid: seq for pid, seq in zip(protein_ids, sequences)}
        with open(seq_file, 'w') as f:
            json.dump(seq_dict, f)
            
        # Save GO term mapping (same for all splits)
        go_terms_file = self.output_dir / f"{aspect}_go_terms.json"
        with open(go_terms_file, 'w') as f:
            json.dump(go_columns, f)
            
        # Print statistics
        n_positives = (labels > 0).sum()
        sparsity = n_positives / (len(protein_ids) * len(go_columns))
        logger.info(f"  {split}: {len(protein_ids)} proteins, {n_positives} positive labels, "
                   f"sparsity: {sparsity:.4f}")
        
    def _create_dataset_info(self, aspect: str, go_columns: List[str]):
        """Create dataset information file."""
        info = {
            'aspect': aspect,
            'n_go_terms': len(go_columns),
            'go_terms': go_columns[:10],  # Save first 10 as example
            'splits': {}
        }
        
        for split in ['train', 'valid', 'test']:
            names = np.load(self.output_dir / f"{aspect}_{split}_names.npy", allow_pickle=True)
            labels_sparse = ssp.load_npz(self.output_dir / f"{aspect}_{split}_labels.npz")
            
            info['splits'][split] = {
                # 'n_proteins': len(names),
                # 'n_positive_labels': (labels_sparse > 0).sum(),

                'n_proteins': int(len(names)),
                'n_positive_labels': int((labels_sparse > 0).sum()),

                'names_file': f"{aspect}_{split}_names.npy",
                'labels_file': f"{aspect}_{split}_labels.npz",
                'sequences_file': f"{aspect}_{split}_sequences.json"
            }
            
        with open(self.output_dir / f"{aspect}_info.json", 'w') as f:
            json.dump(info, f, indent=2)

class ESMEmbeddingGenerator:
    """Generate ESM embeddings for CAFA3 proteins."""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 batch_size: int = 4):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        
    def generate_embeddings(self):
        """Generate ESM embeddings for all proteins."""
        from transformers import AutoTokenizer, AutoModel
        
        # Load model
        logger.info(f"Loading ESM model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Process each aspect and split
        for aspect in ['BPO', 'CCO', 'MFO']:
            for split in ['train', 'valid', 'test']:
                seq_file = self.data_dir / f"{aspect}_{split}_sequences.json"
                if not seq_file.exists():
                    continue
                    
                with open(seq_file, 'r') as f:
                    sequences = json.load(f)
                    
                logger.info(f"Generating embeddings for {aspect} {split}: {len(sequences)} proteins")
                
                # Process in batches
                protein_ids = list(sequences.keys())
                for i in tqdm(range(0, len(protein_ids), self.batch_size)):
                    batch_ids = protein_ids[i:i+self.batch_size]
                    batch_seqs = [sequences[pid] for pid in batch_ids]
                    
                    # Skip if already exists
                    if all((self.output_dir / f"{pid}.npy").exists() for pid in batch_ids):
                        continue
                    
                    try:
                        # Tokenize
                        encoded = tokenizer(
                            batch_seqs,
                            padding=True,
                            truncation=True,
                            max_length=1024,
                            return_tensors="pt"
                        )
                        
                        input_ids = encoded["input_ids"].to(device)
                        attention_mask = encoded["attention_mask"].to(device)
                        
                        # Generate embeddings
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            embeddings = outputs.last_hidden_state
                            
                        # Save each embedding
                        for idx, pid in enumerate(batch_ids):
                            seq_len = len(batch_seqs[idx])
                            # Mean pooling
                            emb = embeddings[idx, 1:seq_len+1].cpu().numpy()
                            
                            np.save(
                                self.output_dir / f"{pid}.npy",
                                {"name": pid, "embedding": emb}
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")

class CAFA3ExperimentGenerator:
    """Generate experiment configurations for CAFA3."""
    
    def __init__(self, 
                 data_dir: str,
                 base_config_path: str,
                 output_dir: str):
        
        self.data_dir = Path(data_dir)
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_configs(self):
        """Generate configuration files for CAFA3 experiments."""
        
        # Load base config
        with open(self.base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
            
        experiments = []
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            # Load dataset info
            with open(self.data_dir / f"{aspect}_info.json", 'r') as f:
                info = json.load(f)
                
            # Generate configs for different model types
            configs = [
                self._create_config(base_config, aspect, info, 'A_ESM_only', ['esm']),
                self._create_config(base_config, aspect, info, 'B_Text_only', ['text']),
                self._create_config(base_config, aspect, info, 'C2_Structure', ['structure'], 
                                  graph_type='radius', radius=10.0),
                self._create_config(base_config, aspect, info, 'D_ESM_Text', ['esm', 'text']),
                self._create_config(base_config, aspect, info, 'F_Full_Model', 
                                  ['esm', 'text', 'structure'], graph_type='knn', k=10)
            ]

            
            experiments.extend(configs)



        # Save configurations
        for exp in experiments:
            config_path = self.output_dir / f"{exp['experiment_name']}.yaml"
            with open(config_path, 'w') as f:


                yaml.dump(exp, f, default_flow_style=False)
                
        return experiments
        
    def _create_config(self, base_config: dict, aspect: str, info: dict, 
                      model_name: str, features: List[str], **kwargs):
        """Create experiment configuration."""
        import copy
        config = copy.deepcopy(base_config)
        
        config['experiment_name'] = f"CAFA3_{model_name}_{aspect}"
        config['dataset']['train_names'] = str(self.data_dir / f"{aspect}_train_names.npy")
        config['dataset']['train_labels'] = str(self.data_dir / f"{aspect}_train_labels.npz")
        config['dataset']['valid_names'] = str(self.data_dir / f"{aspect}_valid_names.npy")
        config['dataset']['valid_labels'] = str(self.data_dir / f"{aspect}_valid_labels.npz")
        
        config['model']['output_dim'] = info['n_go_terms']
        config['dataset']['features'] = features
        
        if 'structure' in features:
            config['graph'] = {
                'type': kwargs.get('graph_type', 'knn'),
                'k': kwargs.get('k', 10),
                'radius': kwargs.get('radius', 10.0),
                'use_esm_features': True
            }




        config['log']['out_dir'] = f"/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/results/{config['experiment_name']}"
        
        return config


class StructureDataFetcher:
    """Fetch AlphaFold structures for CAFA3 proteins."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.af_base_url = "https://alphafold.ebi.ac.uk/files"
        
    def fetch_structures(self, protein_ids: List[str], max_attempts: int = 3):
        """Fetch AlphaFold structures for given proteins."""
        
        successful = 0
        failed = []
        
        for pid in tqdm(protein_ids, desc="Fetching structures"):
            pdb_file = self.output_dir / f"AF-{pid}-F1-model_v4.pdb"
            
            if pdb_file.exists():
                successful += 1
                continue
                
            # Try to fetch from AlphaFold DB
            url = f"{self.af_base_url}/AF-{pid}-F1-model_v4.pdb"
            
            for attempt in range(max_attempts):
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(pdb_file, 'wb') as f:
                            f.write(response.content)
                        successful += 1
                        break
                    elif response.status_code == 404:
                        # Structure not available
                        failed.append(pid)
                        break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Error fetching {pid} after {max_attempts} attempts: {e}")
                        failed.append(pid)
                    else:
                        continue
                        
        logger.info(f"Successfully fetched {successful}/{len(protein_ids)} structures")
        if failed:
            logger.warning(f"Failed to fetch {len(failed)} structures")
            # Save list of failed proteins
            failed_file = self.output_dir / "failed_proteins.txt"
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed))


def create_small_debug_dataset(cafa3_dir: str, output_dir: str, n_samples: int = 50):
    """Create a very small subset for debugging the pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        aspect_map = {'BPO': 'bp', 'CCO': 'cc', 'MFO': 'mf'}
        
        # Load training data
        train_file = Path(cafa3_dir) / f"{aspect_map[aspect]}-training.csv"
        df = pd.read_csv(train_file)
        
        # Sample proteins
        sampled_df = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        # Split into train/val/test
        train_size = int(0.6 * len(sampled_df))
        val_size = int(0.2 * len(sampled_df))
        
        train_df = sampled_df[:train_size]
        val_df = sampled_df[train_size:train_size+val_size]
        test_df = sampled_df[train_size+val_size:]
        
        # Save debug datasets
        train_df.to_csv(output_path / f"{aspect_map[aspect]}-training-debug.csv", index=False)
        val_df.to_csv(output_path / f"{aspect_map[aspect]}-validation-debug.csv", index=False)
        test_df.to_csv(output_path / f"{aspect_map[aspect]}-test-debug.csv", index=False)
        
        logger.info(f"Created debug dataset for {aspect}: "
                   f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        





def create_submission_script(experiment_name: str, config_path: str, output_dir: str):
    """Create cluster submission script for CAFA3 experiment."""
    
    script_content = f"""#!/bin/bash
#$ -N {experiment_name}
#$ -l tmem=60G
#$ -l h_rt=24:0:0
#$ -j y
#$ -o {output_dir}/logs/{experiment_name}.log
#$ -wd /SAN/bioinf/PFP/PFP
#$ -l gpu=true

echo "Starting CAFA3 experiment: {experiment_name}"
echo "Date: $(date)"

# Activate environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train

# Run training
cd /SAN/bioinf/PFP/PFP
python experiments/cafa3_integration/train_cafa3.py \\
    --config {config_path} \\
    --experiment-name {experiment_name}

echo "Experiment completed: $(date)"
"""
    
    script_path = Path(output_dir) / "scripts" / f"{experiment_name}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    script_path.chmod(0o755)
    
    return script_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CAFA3 dataset integration")
    parser.add_argument('--action', type=str, required=True,
                       choices=['prepare', 'embeddings','structures', 'debug', 'train'],
                       help="Action to perform")
    parser.add_argument('--small-subset', action='store_true',
                       help="Use small subset for testing")
    parser.add_argument('--cafa3-dir', type=str, 
                       default="/SAN/bioinf/PFP/dataset/zenodo",
                       help="CAFA3 dataset directory")
    
    args = parser.parse_args()
    
    base_dir = "/SAN/bioinf/PFP/PFP/experiments/cafa3_integration"
    
    if args.action == 'prepare':
        # Prepare CAFA3 dataset
        preparer = CAFA3DatasetPreparer(
            cafa3_dir=args.cafa3_dir,
            small_subset=args.small_subset
        )
        preparer.prepare_dataset()
    elif args.action == 'embeddings':
        # Generate ESM embeddings
        generator = ESMEmbeddingGenerator(
            data_dir=f"{base_dir}/data",
            output_dir="/SAN/bioinf/PFP/embeddings/cafa3/esm"
        )
        generator.generate_embeddings()

        
    elif args.action == 'structures':
        # Fetch structures
        # First collect all unique protein IDs
        all_proteins = set()
        data_dir = Path(f"{base_dir}/data")
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            for split in ['train', 'valid', 'test']:
                names_file = data_dir / f"{aspect}_{split}_names.npy"
                if names_file.exists():
                    proteins = np.load(names_file, allow_pickle=True)
                    all_proteins.update(proteins)
                    
        logger.info(f"Found {len(all_proteins)} unique proteins across all datasets")
        
        fetcher = StructureDataFetcher("/SAN/bioinf/PFP/embeddings/cafa3/structures")
        fetcher.fetch_structures(list(all_proteins))


    elif args.action == 'train':
        # Generate experiment configs and submission scripts
        generator = CAFA3ExperimentGenerator(
            data_dir=f"{base_dir}/data",
            base_config_path="/SAN/bioinf/PFP/PFP/configs/base_multimodal.yaml",
            output_dir=f"{base_dir}/configs"
        )
        
        experiments = generator.generate_configs()
        
        # Create submission scripts
        for exp in experiments:
            create_submission_script(
                exp['experiment_name'],
                f"{base_dir}/configs/{exp['experiment_name']}.yaml",
                base_dir
            )
            
        print(f"Created {len(experiments)} experiment configurations")
        print(f"Submit with: {base_dir}/scripts/submit_all.sh")
              
    elif args.action == 'debug':
        # Create small debug dataset
        create_small_debug_dataset(
            args.cafa3_dir,
            f"{base_dir}/debug_data"
        )
        


if __name__ == "__main__":
    main()