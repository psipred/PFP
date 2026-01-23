#!/usr/bin/env python3
"""
Prepare CAFA3 dataset for MMFP training.

Converts raw CAFA3 CSV files to the format expected by MMFP:
- {aspect}_{split}_names.npy - protein IDs
- {aspect}_{split}_labels.npz - GO term labels (sparse matrix)
- {aspect}_{split}_sequences.json - protein sequences
- {aspect}_go_terms.json - GO term list

Usage:
    python scripts/prepare_cafa3_data.py

Input:
    Raw CAFA3 CSV files from: /home/zijianzhou/Datasets/cafa3
    - bp-training.csv, bp-validation.csv, bp-test.csv
    - cc-training.csv, cc-validation.csv, cc-test.csv
    - mf-training.csv, mf-validation.csv, mf-test.csv

Output:
    Processed data in: ./data/
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import json
import scipy.sparse as ssp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Update these paths as needed
# ============================================================================
CAFA3_RAW_DIR = "/home/zijianzhou/Datasets/cafa3"
OUTPUT_DIR = "./data"
# ============================================================================


class CAFA3DatasetPreparer:
    """Prepare CAFA3 dataset for MMFP training."""

    def __init__(self,
                 cafa3_dir: str = CAFA3_RAW_DIR,
                 output_dir: str = OUTPUT_DIR,
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
            logger.info(f"{aspect}: No data leakage detected")

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
        logger.info("CAFA3 DATASET PREPARATION COMPLETE")
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

        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("="*80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CAFA3 dataset for MMFP")
    parser.add_argument('--cafa3-dir', type=str,
                       default=CAFA3_RAW_DIR,
                       help="CAFA3 raw dataset directory")
    parser.add_argument('--output-dir', type=str,
                       default=OUTPUT_DIR,
                       help="Output directory for processed data")
    parser.add_argument('--small-subset', action='store_true',
                       help="Use small subset for testing")

    args = parser.parse_args()

    preparer = CAFA3DatasetPreparer(
        cafa3_dir=args.cafa3_dir,
        output_dir=args.output_dir,
        small_subset=args.small_subset
    )
    preparer.prepare_dataset()


if __name__ == "__main__":
    main()
