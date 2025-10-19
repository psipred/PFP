#!/usr/bin/env python3
"""
CAFA3 Dataset Integration for PLM-based GO Prediction
Location: /SAN/bioinf/PFP/PFP/experiments/cafa3_integration/cafa3_integration.py
"""
# python cafa3_integration.py --action prepare
# python cafa3_integration.py --action esm_embeddings  # ESM
# python cafa3_integration.py --action prott5_embeddings
# python cafa3_integration.py --action prostt5_embeddings
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List
from tqdm import tqdm
import json
import scipy.sparse as ssp
import re

# Add project root to path
sys.path.append('/SAN/bioinf/PFP/PFP')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAFA3DatasetPreparer:
    """Prepare CAFA3 dataset for PLM-based GO prediction."""
    
    def __init__(self, 
                 cafa3_dir: str = "/SAN/bioinf/PFP/dataset/zenodo",
                 output_dir: str = "/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/data",
                 small_subset: bool = False):
        
        self.cafa3_dir = Path(cafa3_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.small_subset = small_subset
            
    def load_cafa3_data(self, aspect: str, split: str) -> pd.DataFrame:
        """Load CAFA3 CSV file for given aspect and split."""
        aspect_map = {'BPO': 'bp', 'CCO': 'cc', 'MFO': 'mf'}
        filename = f"{aspect_map[aspect]}-{split}.csv"
        filepath = self.cafa3_dir / filename
        
        logger.info(f"Loading {filepath}")
        df = pd.read_csv(filepath)

        if self.small_subset:
            n_samples = min(100, len(df))
            df = df.sample(n=n_samples, random_state=42)
            logger.info(f"Using subset of {n_samples} samples for testing")
            
        return df
    
    def prepare_dataset(self):
        """Prepare complete CAFA3 dataset."""
        dataset_stats = {}
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            logger.info(f"\nProcessing {aspect}...")
            
            # Load train/val/test splits
            train_df = self.load_cafa3_data(aspect, 'training')
            val_df = self.load_cafa3_data(aspect, 'validation')
            test_df = self.load_cafa3_data(aspect, 'test')
            
            # Get GO terms
            go_columns = [col for col in train_df.columns if col.startswith('GO:')]
            logger.info(f"Found {len(go_columns)} GO terms for {aspect}")
            
            # Check for data leakage
            self._check_data_leakage(train_df, val_df, test_df, aspect)

            # Process each split
            self._process_split(train_df, aspect, 'train', go_columns)
            self._process_split(val_df, aspect, 'valid', go_columns)
            self._process_split(test_df, aspect, 'test', go_columns)
            
            # Create dataset info
            self._create_dataset_info(aspect, go_columns)
            
            # Collect statistics
            dataset_stats[aspect] = {
                'train': train_df,
                'valid': val_df,
                'test': test_df,
                'go_terms': go_columns
            }
        
        # Generate evaluation report
        self._generate_evaluation_report(dataset_stats)

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
        
        # Extract GO labels
        labels = df[go_columns].values.astype(np.float32)
        
        # Save protein names
        names_file = self.output_dir / f"{aspect}_{split}_names.npy"
        np.save(names_file, protein_ids)
        
        # Save labels as sparse matrix
        labels_sparse = ssp.csr_matrix(labels)
        labels_file = self.output_dir / f"{aspect}_{split}_labels.npz"
        ssp.save_npz(labels_file, labels_sparse)
        
        # Save sequences
        seq_file = self.output_dir / f"{aspect}_{split}_sequences.json"
        seq_dict = {pid: seq for pid, seq in zip(protein_ids, sequences)}
        with open(seq_file, 'w') as f:
            json.dump(seq_dict, f)
            
        # Save GO term mapping
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
            'go_terms': go_columns[:10],
            'splits': {}
        }
        
        for split in ['train', 'valid', 'test']:
            names = np.load(self.output_dir / f"{aspect}_{split}_names.npy", allow_pickle=True)
            labels_sparse = ssp.load_npz(self.output_dir / f"{aspect}_{split}_labels.npz")
            
            info['splits'][split] = {
                'n_proteins': int(len(names)),
                'n_positive_labels': int((labels_sparse > 0).sum()),
                'names_file': f"{aspect}_{split}_names.npy",
                'labels_file': f"{aspect}_{split}_labels.npz",
                'sequences_file': f"{aspect}_{split}_sequences.json"
            }
            
        with open(self.output_dir / f"{aspect}_info.json", 'w') as f:
            json.dump(info, f, indent=2)

    def _generate_evaluation_report(self, dataset_stats: Dict):
        """Generate comprehensive evaluation report."""
        logger.info("\n" + "="*80)
        logger.info("CAFA3 DATASET EVALUATION REPORT")
        logger.info("="*80)
        
        # Print statistics for each aspect
        for aspect in ['MFO', 'CCO', 'BPO']:
            if aspect not in dataset_stats:
                continue
                
            data = dataset_stats[aspect]
            go_terms = data['go_terms']
            
            logger.info(f"\n{aspect}:")
            logger.info(f"  GO terms: {len(go_terms)}")
            
            for split in ['train', 'valid', 'test']:
                df = data[split]
                labels = df[go_terms].values
                
                n_proteins = len(df)
                n_annotations = (labels > 0).sum()
                avg_terms = (labels > 0).sum(axis=1).mean()
                
                logger.info(f"  {split.capitalize()}: {n_proteins} proteins, "
                          f"{n_annotations} annotations, "
                          f"avg {avg_terms:.1f} terms/protein")


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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_embeddings(self):
        """Generate ESM embeddings for all proteins."""
        from transformers import AutoTokenizer, AutoModel
        
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
                            # Mean pooling (exclude special tokens)
                            emb = embeddings[idx, 1:seq_len+1].mean(dim=0).cpu().numpy()
                            
                            np.save(
                                self.output_dir / f"{pid}.npy",
                                {"name": pid, "embedding": emb}
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        
        logger.info("ESM embedding generation completed!")


class ProtT5EmbeddingGenerator:
    """Generate ProtT5 embeddings for CAFA3 proteins."""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 model_name: str = "Rostlab/prot_t5_xl_uniref50",
                 batch_size: int = 4,
                 max_length: int = 1024):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """Preprocess sequences for ProtT5."""
        processed_sequences = []
        
        for seq in sequences:
            # Replace rare/ambiguous amino acids
            seq = re.sub(r"[UZOB]", "X", seq)
            # Add whitespace between amino acids
            seq = " ".join(list(seq))
            processed_sequences.append(seq)
            
        return processed_sequences
    
    def generate_embeddings(self):
        """Generate ProtT5 embeddings for all proteins."""
        from transformers import T5Tokenizer, T5EncoderModel
        
        logger.info(f"Loading ProtT5 model: {self.model_name}")
        tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(self.model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if device.type == 'cpu':
            model.float()
        else:
            model.half()
            
        model.eval()
        
        # Process each aspect and split
        for aspect in ['BPO', 'CCO', 'MFO']:
            for split in ['train', 'valid', 'test']:
                seq_file = self.data_dir / f"{aspect}_{split}_sequences.json"
                if not seq_file.exists():
                    continue
                    
                with open(seq_file, 'r') as f:
                    sequences = json.load(f)
                    
                logger.info(f"Generating ProtT5 embeddings for {aspect} {split}: {len(sequences)} proteins")
                
                protein_ids = list(sequences.keys())
                
                for i in tqdm(range(0, len(protein_ids), self.batch_size), desc=f"{aspect} {split}"):
                    batch_ids = protein_ids[i:i+self.batch_size]
                    batch_seqs = [sequences[pid] for pid in batch_ids]
                    
                    # Skip if exists
                    if all((self.output_dir / f"{pid}.npy").exists() for pid in batch_ids):
                        continue
                    
                    try:
                        # Preprocess sequences
                        processed_seqs = self._preprocess_sequences(batch_seqs)
                        
                        # Tokenize
                        ids = tokenizer.batch_encode_plus(
                            processed_seqs,
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors='pt'
                        ).to(device)
                        
                        # Generate embeddings
                        with torch.no_grad():
                            outputs = model(
                                ids.input_ids,
                                attention_mask=ids.attention_mask
                            )
                            embeddings = outputs.last_hidden_state
                        
                        # Extract and save embeddings
                        for idx, (pid, seq) in enumerate(zip(batch_ids, batch_seqs)):
                            seq_len = len(seq)
                            attention_mask_seq = ids.attention_mask[idx]
                            actual_token_len = attention_mask_seq.sum().item()
                            
                            # Skip special tokens
                            start_idx = 1
                            end_idx = min(start_idx + seq_len, actual_token_len - 1)
                            
                            # Get embeddings for sequence
                            seq_embeddings = embeddings[idx, start_idx:end_idx]
                            
                            # Mean pooling
                            pooled_embedding = seq_embeddings.mean(dim=0).cpu().numpy()
                            per_residue_embeddings = seq_embeddings.cpu().numpy()
                            
                            # Save embeddings
                            embedding_data = {
                                "name": pid,
                                "embedding": pooled_embedding,
                                "per_residue_embedding": per_residue_embeddings,
                                "sequence_length": seq_len,
                                "truncated_length": end_idx - start_idx
                            }
                            
                            np.save(
                                self.output_dir / f"{pid}.npy",
                                embedding_data,
                                allow_pickle=True
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing batch starting at index {i}: {e}")
                        
        logger.info("ProtT5 embedding generation completed!")


class ProstT5EmbeddingGenerator:
    """Generate ProstT5 embeddings for CAFA3 proteins."""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 model_name: str = "Rostlab/ProstT5",
                 batch_size: int = 4,
                 embedding_type: str = "AA",
                 max_length: int = 1024):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_type = embedding_type
        self.max_length = max_length
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """Preprocess sequences for ProstT5."""
        processed_sequences = []
        
        for seq in sequences:
            # Replace rare/ambiguous amino acids
            seq = re.sub(r"[UZOB]", "X", seq)
            # Add whitespace
            seq = " ".join(list(seq))
            
            # Add prefix
            if self.embedding_type == "AA":
                seq = "<AA2fold> " + seq
            elif self.embedding_type == "3Di":
                seq = seq.lower()
                seq = "<fold2AA> " + seq
            
            processed_sequences.append(seq)
            
        return processed_sequences
    
    def generate_embeddings(self):
        """Generate ProstT5 embeddings for all proteins."""
        from transformers import T5Tokenizer, T5EncoderModel
        
        logger.info(f"Loading ProstT5 model: {self.model_name}")
        tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(self.model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if device.type == 'cpu':
            model.float()
        else:
            model.half()
            
        model.eval()
        
        # Process each aspect and split
        for aspect in ['BPO', 'CCO', 'MFO']:
            for split in ['train', 'valid', 'test']:
                seq_file = self.data_dir / f"{aspect}_{split}_sequences.json"
                if not seq_file.exists():
                    continue
                    
                with open(seq_file, 'r') as f:
                    sequences = json.load(f)
                    
                logger.info(f"Generating ProstT5 embeddings for {aspect} {split}: {len(sequences)} proteins")
                
                protein_ids = list(sequences.keys())
                
                for i in tqdm(range(0, len(protein_ids), self.batch_size), desc=f"{aspect} {split}"):
                    batch_ids = protein_ids[i:i+self.batch_size]
                    batch_seqs = [sequences[pid] for pid in batch_ids]
                    
                    # Skip if exists
                    if all((self.output_dir / f"{pid}.npy").exists() for pid in batch_ids):
                        continue
                    
                    try:
                        # Preprocess sequences
                        processed_seqs = self._preprocess_sequences(batch_seqs)
                        
                        # Tokenize
                        ids = tokenizer.batch_encode_plus(
                            processed_seqs,
                            add_special_tokens=True,
                            padding="longest",
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors='pt'
                        ).to(device)
                        
                        # Generate embeddings
                        with torch.no_grad():
                            outputs = model(
                                ids.input_ids,
                                attention_mask=ids.attention_mask
                            )
                            embeddings = outputs.last_hidden_state
                        
                        # Extract and save embeddings
                        for idx, (pid, seq) in enumerate(zip(batch_ids, batch_seqs)):
                            seq_len = len(seq)
                            attention_mask_seq = ids.attention_mask[idx]
                            actual_token_len = attention_mask_seq.sum().item()
                            
                            # Skip special tokens and prefix
                            start_idx = 2  # +1 for start token, +1 for prefix
                            end_idx = min(start_idx + seq_len, actual_token_len - 1)
                            
                            # Get embeddings
                            seq_embeddings = embeddings[idx, start_idx:end_idx]
                            
                            # Mean pooling
                            pooled_embedding = seq_embeddings.mean(dim=0).cpu().numpy()
                            per_residue_embeddings = seq_embeddings.cpu().numpy()
                            
                            # Save embeddings
                            embedding_data = {
                                "name": pid,
                                "embedding": pooled_embedding,
                                "per_residue_embedding": per_residue_embeddings,
                                "sequence_length": seq_len,
                                "truncated_length": end_idx - start_idx
                            }
                            
                            np.save(
                                self.output_dir / f"{pid}.npy",
                                embedding_data,
                                allow_pickle=True
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing batch starting at index {i}: {e}")
                        
        logger.info("ProstT5 embedding generation completed!")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CAFA3 dataset integration")
    parser.add_argument('--action', type=str, required=True,
                       choices=['prepare', 'esm_embeddings', 'prott5_embeddings', 'prostt5_embeddings'],
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
        
    elif args.action == 'esm_embeddings':
        # Generate ESM embeddings   
        generator = ESMEmbeddingGenerator(
            data_dir=f"{base_dir}/data",
            output_dir="/SAN/bioinf/PFP/embeddings/cafa3/esm"
        )
        generator.generate_embeddings()
        
    elif args.action == 'prott5_embeddings':
        # Generate ProtT5 embeddings
        generator = ProtT5EmbeddingGenerator(
            data_dir=f"{base_dir}/data",
            output_dir="/SAN/bioinf/PFP/embeddings/cafa3/prott5",
            batch_size=4
        )
        generator.generate_embeddings()
        
    elif args.action == 'prostt5_embeddings':
        # Generate ProstT5 embeddings
        generator = ProstT5EmbeddingGenerator(
            data_dir=f"{base_dir}/data",
            output_dir="/SAN/bioinf/PFP/embeddings/cafa3/prostt5",
            embedding_type='AA',
            batch_size=4
        )
        generator.generate_embeddings()


if __name__ == "__main__":
    main()