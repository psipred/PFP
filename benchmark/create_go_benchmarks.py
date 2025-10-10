#!/usr/bin/env python3
"""
Create GO function prediction benchmarks at different similarity thresholds.
Clusters proteins using MMseqs2 and splits into train/val/test sets.
"""

import os
import random
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Configuration
BASE_DIR = "/home/zijianzhou/Datasets/protad/go_annotations"
FASTA_FILE = os.path.join(BASE_DIR, "proteins_filtered.fasta")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "protein_go_annotations.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmarks")

# Similarity thresholds for benchmarks
SIMILARITY_THRESHOLDS = [0.30, 0.50, 0.70, 0.95]

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)


def run_mmseqs_clustering(fasta_file: str, output_dir: str, similarity: float):
    """Run MMseqs2 clustering at specified similarity threshold."""
    print(f"\n{'='*60}")
    print(f"Running MMseqs2 clustering at {int(similarity*100)}% similarity")
    print(f"{'='*60}")
    
    # Create temporary directory for this clustering
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    db_prefix = os.path.join(output_dir, "db")
    cluster_prefix = os.path.join(output_dir, "cluster")
    tsv_file = os.path.join(output_dir, "clusters.tsv")
    
    try:
        # Create database
        print("Creating database...")
        subprocess.run([
            "mmseqs", "createdb", fasta_file, db_prefix
        ], check=True)
        
        # Run clustering
        print(f"Clustering at {similarity} similarity...")
        subprocess.run([
            "mmseqs", "cluster",
            db_prefix, cluster_prefix, tmp_dir,
            "--min-seq-id", str(similarity),
            "--cov-mode", "1"
        ], check=True)
        
        # Create TSV output
        print("Creating TSV output...")
        subprocess.run([
            "mmseqs", "createtsv",
            db_prefix, db_prefix, cluster_prefix, tsv_file
        ], check=True)
        
        print(f"Clustering complete. Results saved to {tsv_file}")
        
    finally:
        # Cleanup (but keep clusters.tsv for now)
        print("Cleaning up temporary files...")
        for file in Path(output_dir).glob("cluster*"):
            if file.name != "clusters.tsv":
                file.unlink()
        for file in Path(output_dir).glob("db*"):
            file.unlink()
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    return tsv_file


def read_clusters(tsv_file: str):
    """Read cluster assignments from MMseqs2 TSV output."""
    print("Reading cluster assignments...")
    cluster_to_proteins = {}
    
    with open(tsv_file) as f:
        for line in tqdm(f, desc="Reading clusters"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cluster_id, protein_id = parts[0], parts[1]
                if cluster_id not in cluster_to_proteins:
                    cluster_to_proteins[cluster_id] = []
                cluster_to_proteins[cluster_id].append(protein_id)
    
    print(f"Found {len(cluster_to_proteins)} clusters")
    return cluster_to_proteins


def split_clusters(cluster_to_proteins: dict):
    """Split clusters into train/val/test sets."""
    cluster_ids = list(cluster_to_proteins.keys())
    random.shuffle(cluster_ids)
    
    n_clusters = len(cluster_ids)
    train_end = int(n_clusters * TRAIN_RATIO)
    val_end = int(n_clusters * (TRAIN_RATIO + VAL_RATIO))
    
    train_clusters = cluster_ids[:train_end]
    val_clusters = cluster_ids[train_end:val_end]
    test_clusters = cluster_ids[val_end:]
    
    print(f"Split: {len(train_clusters)} train, {len(val_clusters)} val, {len(test_clusters)} test clusters")
    
    # Select one protein from each cluster
    def select_from_clusters(cluster_list):
        proteins = set()
        for cluster_id in tqdm(cluster_list, desc="Selecting proteins"):
            cluster_proteins = cluster_to_proteins[cluster_id]
            selected = random.choice(cluster_proteins)
            proteins.add(selected)
        return proteins
    
    train_proteins = select_from_clusters(train_clusters)
    val_proteins = select_from_clusters(val_clusters)
    test_proteins = select_from_clusters(test_clusters)
    
    return train_proteins, val_proteins, test_proteins


def create_splits(annotations_df: pd.DataFrame, train_ids: set, val_ids: set, test_ids: set, output_dir: str):
    """Create and save train/val/test splits with GO annotations."""
    print("\nCreating dataset splits...")
    
    train = annotations_df[annotations_df['protein_id'].isin(train_ids)]
    val = annotations_df[annotations_df['protein_id'].isin(val_ids)]
    test = annotations_df[annotations_df['protein_id'].isin(test_ids)]
    
    print(f"Train: {len(train)} proteins")
    print(f"Val:   {len(val)} proteins")
    print(f"Test:  {len(test)} proteins")
    
    # Save splits
    train.to_csv(os.path.join(output_dir, "train.tsv"), sep='\t', index=False)
    val.to_csv(os.path.join(output_dir, "val.tsv"), sep='\t', index=False)
    test.to_csv(os.path.join(output_dir, "test.tsv"), sep='\t', index=False)
    
    # Save statistics
    stats = {
        'similarity_threshold': os.path.basename(output_dir),
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
        'n_total': len(train) + len(val) + len(test)
    }
    
    with open(os.path.join(output_dir, "split_stats.txt"), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Splits saved to {output_dir}")
    return stats


def main():
    print("GO Function Prediction Benchmark Creation")
    print("="*60)
    
    # Load GO annotations
    print(f"\nLoading GO annotations from {ANNOTATIONS_FILE}")
    annotations = pd.read_csv(ANNOTATIONS_FILE, sep='\t')
    print(f"Loaded annotations for {len(annotations)} proteins")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_stats = []
    
    # Process each similarity threshold
    for similarity in SIMILARITY_THRESHOLDS:
        threshold_name = f"similarity_{int(similarity*100)}"
        benchmark_dir = os.path.join(OUTPUT_DIR, threshold_name)
        os.makedirs(benchmark_dir, exist_ok=True)
        
        print(f"\n{'#'*60}")
        print(f"Processing benchmark: {threshold_name}")
        print(f"{'#'*60}")
        
        # Run clustering
        tsv_file = run_mmseqs_clustering(FASTA_FILE, benchmark_dir, similarity)
        
        # Read clusters
        cluster_to_proteins = read_clusters(tsv_file)
        
        # Split into train/val/test
        train_ids, val_ids, test_ids = split_clusters(cluster_to_proteins)
        
        # Create and save splits
        stats = create_splits(annotations, train_ids, val_ids, test_ids, benchmark_dir)
        stats['threshold'] = similarity
        all_stats.append(stats)
        
        # Remove cluster TSV to save space
        os.remove(tsv_file)
    
    # Save summary statistics
    summary_file = os.path.join(OUTPUT_DIR, "benchmark_summary.txt")
    print(f"\n{'='*60}")
    print("BENCHMARK CREATION COMPLETE")
    print(f"{'='*60}")
    print("\nSummary:")
    
    with open(summary_file, 'w') as f:
        f.write("GO Function Prediction Benchmarks\n")
        f.write("="*60 + "\n\n")
        for stats in all_stats:
            line = f"Similarity {int(stats['threshold']*100)}%: {stats['n_train']} train, {stats['n_val']} val, {stats['n_test']} test"
            print(line)
            f.write(line + "\n")
        f.write(f"\nAll benchmarks saved to: {OUTPUT_DIR}/\n")
    
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()