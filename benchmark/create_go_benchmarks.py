#!/usr/bin/env python3
"""
Create GO function prediction benchmarks at different similarity thresholds.
Splits by GO aspect (BP, MF, CC) and ensures no unseen test labels in training.
Only keeps GO terms that appear more than MIN_TERM_FREQUENCY times.
"""

import os
import random
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
BASE_DIR = "/home/zijianzhou/Datasets/protad/go_annotations"
FASTA_FILE = os.path.join(BASE_DIR, "proteins_filtered.fasta")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "protein_go_annotations.tsv")
GO_INFO_FILE = os.path.join(BASE_DIR, "go_terms_info.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmarks")

# Similarity thresholds for benchmarks
SIMILARITY_THRESHOLDS = [0.30, 0.50, 0.70, 0.95]

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Minimum frequency threshold for GO terms
MIN_TERM_FREQUENCY = 20

# GO aspect mapping
ASPECT_NAMES = {
    'biological_process': 'BP',
    'molecular_function': 'MF',
    'cellular_component': 'CC'
}

random.seed(42)


def load_go_aspect_info(go_info_file: str):
    """Load GO term aspect information."""
    print(f"Loading GO term aspects from {go_info_file}...")
    go_to_aspect = {}
    
    df = pd.read_csv(go_info_file, sep='\t')
    for _, row in df.iterrows():
        go_id = row['go_id']
        aspect = row['aspect']
        go_to_aspect[go_id] = aspect
    
    print(f"Loaded aspect info for {len(go_to_aspect)} GO terms")
    return go_to_aspect


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
        # similarity is 30, 50, 70, 95
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
        # Cleanup
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


def filter_annotations_by_aspect(annotations_df: pd.DataFrame, 
                                   go_to_aspect: dict,
                                   aspect: str):
    """Filter annotations to only include GO terms from specified aspect."""
    print(f"\nFiltering for aspect: {aspect}")
    
    filtered_rows = []
    for _, row in annotations_df.iterrows():
        protein_id = row['protein_id']
        go_terms = row['go_terms'].split(',')
        
        # Filter GO terms by aspect
        aspect_terms = [term for term in go_terms if go_to_aspect.get(term) == aspect]
        
        if aspect_terms:
            filtered_rows.append({
                'protein_id': protein_id,
                'go_terms': ','.join(aspect_terms)
            })
    
    filtered_df = pd.DataFrame(filtered_rows)
    print(f"Proteins with {aspect} annotations: {len(filtered_df)}")
    
    return filtered_df


def filter_by_frequency(df: pd.DataFrame, min_frequency: int):
    """
    Filter GO terms by minimum frequency.
    Only keeps GO terms that appear >= min_frequency times across all proteins.
    """
    print(f"\nFiltering GO terms with frequency >= {min_frequency}...")
    
    # Count GO term frequencies
    go_counter = Counter()
    for go_terms_str in df['go_terms']:
        go_terms = go_terms_str.split(',')
        go_counter.update(go_terms)
    
    print(f"  Total unique GO terms before filtering: {len(go_counter)}")
    
    # Keep only frequent terms
    frequent_terms = {term for term, count in go_counter.items() if count >= min_frequency}
    print(f"  GO terms with frequency >= {min_frequency}: {len(frequent_terms)}")
    
    # Filter dataframe
    filtered_rows = []
    removed_proteins = 0
    
    for _, row in df.iterrows():
        protein_id = row['protein_id']
        go_terms = row['go_terms'].split(',')
        
        # Keep only frequent terms
        filtered_terms = [term for term in go_terms if term in frequent_terms]
        
        if filtered_terms:
            filtered_rows.append({
                'protein_id': protein_id,
                'go_terms': ','.join(filtered_terms)
            })
        else:
            removed_proteins += 1
    
    filtered_df = pd.DataFrame(filtered_rows)
    print(f"  Proteins after filtering: {len(filtered_df)} (removed {removed_proteins})")
    
    return filtered_df


def remove_unseen_test_labels(train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame):
    """
    Remove proteins from val/test that have GO terms not seen in training.
    This ensures all labels in val/test appear at least once in training.
    """
    print("\nRemoving unseen test labels...")
    
    # Collect all GO terms from training set
    train_go_terms = set()
    for go_terms_str in train_df['go_terms']:
        train_go_terms.update(go_terms_str.split(','))
    
    print(f"Training set has {len(train_go_terms)} unique GO terms")
    
    # Filter validation set
    val_filtered = []
    val_removed = 0
    for _, row in val_df.iterrows():
        go_terms = set(row['go_terms'].split(','))
        # Keep only GO terms that appear in training
        valid_terms = go_terms & train_go_terms
        
        if valid_terms:
            val_filtered.append({
                'protein_id': row['protein_id'],
                'go_terms': ','.join(valid_terms)
            })
        else:
            val_removed += 1
    
    # Filter test set
    test_filtered = []
    test_removed = 0
    for _, row in test_df.iterrows():
        go_terms = set(row['go_terms'].split(','))
        # Keep only GO terms that appear in training
        valid_terms = go_terms & train_go_terms
        
        if valid_terms:
            test_filtered.append({
                'protein_id': row['protein_id'],
                'go_terms': ','.join(valid_terms)
            })
        else:
            test_removed += 1
    
    val_df_filtered = pd.DataFrame(val_filtered)
    test_df_filtered = pd.DataFrame(test_filtered)
    
    print(f"Validation: kept {len(val_df_filtered)}, removed {val_removed} proteins")
    print(f"Test: kept {len(test_df_filtered)}, removed {test_removed} proteins")
    
    return train_df, val_df_filtered, test_df_filtered


def create_aspect_splits(annotations_df: pd.DataFrame,
                         go_to_aspect: dict,
                         train_ids: set,
                         val_ids: set,
                         test_ids: set,
                         output_dir: str):
    """Create splits for each GO aspect separately."""
    print("\n" + "="*60)
    print("Creating aspect-specific splits")
    print("="*60)
    
    all_stats = {}
    
    for aspect_full, aspect_short in ASPECT_NAMES.items():
        print(f"\n{'#'*60}")
        print(f"Processing aspect: {aspect_full} ({aspect_short})")
        print(f"{'#'*60}")
        
        # Filter annotations by aspect
        aspect_df = filter_annotations_by_aspect(annotations_df, go_to_aspect, aspect_full)
        
        if len(aspect_df) == 0:
            print(f"No proteins for aspect {aspect_full}, skipping...")
            continue
        
        # Filter by minimum frequency BEFORE splitting
        aspect_df = filter_by_frequency(aspect_df, MIN_TERM_FREQUENCY)
        
        if len(aspect_df) == 0:
            print(f"No proteins remaining after frequency filtering for {aspect_full}, skipping...")
            continue
        
        # Split by protein IDs
        train = aspect_df[aspect_df['protein_id'].isin(train_ids)]
        val = aspect_df[aspect_df['protein_id'].isin(val_ids)]
        test = aspect_df[aspect_df['protein_id'].isin(test_ids)]
        
        print(f"Before unseen label filtering - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Remove unseen test labels
        train, val, test = remove_unseen_test_labels(train, val, test)
        
        print(f"After unseen label filtering - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Create aspect-specific output directory
        aspect_dir = os.path.join(output_dir, aspect_short)
        os.makedirs(aspect_dir, exist_ok=True)
        
        # Save splits
        train.to_csv(os.path.join(aspect_dir, "train.tsv"), sep='\t', index=False)
        val.to_csv(os.path.join(aspect_dir, "val.tsv"), sep='\t', index=False)
        test.to_csv(os.path.join(aspect_dir, "test.tsv"), sep='\t', index=False)
        
        # Calculate statistics
        train_go_terms = set()
        for go_terms_str in train['go_terms']:
            train_go_terms.update(go_terms_str.split(','))
        
        val_go_terms = set()
        for go_terms_str in val['go_terms']:
            val_go_terms.update(go_terms_str.split(','))
        
        test_go_terms = set()
        for go_terms_str in test['go_terms']:
            test_go_terms.update(go_terms_str.split(','))
        
        stats = {
            'aspect': aspect_short,
            'min_frequency': MIN_TERM_FREQUENCY,
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test),
            'n_total': len(train) + len(val) + len(test),
            'n_train_labels': len(train_go_terms),
            'n_val_labels': len(val_go_terms),
            'n_test_labels': len(test_go_terms),
            'unseen_val_labels': len(val_go_terms - train_go_terms),
            'unseen_test_labels': len(test_go_terms - train_go_terms)
        }
        
        # Save statistics
        with open(os.path.join(aspect_dir, "split_stats.txt"), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        all_stats[aspect_short] = stats
        
        print(f"\nStatistics for {aspect_short}:")
        print(f"  Min frequency threshold: {MIN_TERM_FREQUENCY}")
        print(f"  Train proteins: {stats['n_train']}, labels: {stats['n_train_labels']}")
        print(f"  Val proteins: {stats['n_val']}, labels: {stats['n_val_labels']}")
        print(f"  Test proteins: {stats['n_test']}, labels: {stats['n_test_labels']}")
        print(f"  Unseen val labels: {stats['unseen_val_labels']}")
        print(f"  Unseen test labels: {stats['unseen_test_labels']}")
    
    return all_stats


def main():
    print("GO Function Prediction Benchmark Creation (By Aspect)")
    print(f"Minimum GO term frequency: {MIN_TERM_FREQUENCY}")
    print("="*60)
    
    # Load GO annotations
    print(f"\nLoading GO annotations from {ANNOTATIONS_FILE}")
    annotations = pd.read_csv(ANNOTATIONS_FILE, sep='\t')
    print(f"Loaded annotations for {len(annotations)} proteins")
    
    # Load GO aspect information
    go_to_aspect = load_go_aspect_info(GO_INFO_FILE)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
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
        
        # Create aspect-specific splits
        aspect_stats = create_aspect_splits(
            annotations, go_to_aspect, 
            train_ids, val_ids, test_ids, 
            benchmark_dir
        )
        
        # Store results
        for aspect, stats in aspect_stats.items():
            result = {
                'threshold': similarity,
                'threshold_name': threshold_name,
                **stats
            }
            all_results.append(result)
        
        # Remove cluster TSV to save space
        os.remove(tsv_file)
    
    # Save comprehensive summary
    summary_file = os.path.join(OUTPUT_DIR, "benchmark_summary.txt")
    print(f"\n{'='*60}")
    print("BENCHMARK CREATION COMPLETE")
    print(f"{'='*60}")
    print("\nSummary:")
    
    with open(summary_file, 'w') as f:
        f.write("GO Function Prediction Benchmarks (By Aspect)\n")
        f.write(f"Minimum GO term frequency: {MIN_TERM_FREQUENCY}\n")
        f.write("="*60 + "\n\n")
        
        for threshold in SIMILARITY_THRESHOLDS:
            threshold_results = [r for r in all_results if r['threshold'] == threshold]
            f.write(f"\nSimilarity {int(threshold*100)}%:\n")
            f.write("-" * 40 + "\n")
            
            for result in threshold_results:
                aspect = result['aspect']
                line = (f"  {aspect}: {result['n_train']} train, {result['n_val']} val, "
                       f"{result['n_test']} test | "
                       f"Labels - Train: {result['n_train_labels']}, "
                       f"Val: {result['n_val_labels']}, Test: {result['n_test_labels']} | "
                       f"Unseen: Val={result['unseen_val_labels']}, Test={result['unseen_test_labels']}")
                print(line)
                f.write(line + "\n")
        
        f.write(f"\nAll benchmarks saved to: {OUTPUT_DIR}/\n")
        f.write("\nDirectory structure:\n")
        f.write("  benchmarks/\n")
        f.write("    similarity_XX/\n")
        f.write("      BP/  (biological_process)\n")
        f.write("      MF/  (molecular_function)\n")
        f.write("      CC/  (cellular_component)\n")
        f.write(f"\nNote: Only GO terms appearing >= {MIN_TERM_FREQUENCY} times are included\n")
    
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()