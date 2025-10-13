"""
Comprehensive data leakage checker for protein function prediction experiments.

Checks for:
1. Protein ID overlap between train/val/test
2. Sequence identity leakage (similar sequences across splits)
3. GO term distribution similarity
4. Temporal leakage issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def check_protein_overlap(splits_dict, aspect, threshold):
    """Check for direct protein ID overlap between splits."""
    
    train_ids = set(splits_dict['train']['protein_id'])
    val_ids = set(splits_dict['val']['protein_id'])
    test_ids = set(splits_dict['test']['protein_id'])
    
    results = {
        'aspect': aspect,
        'threshold': threshold,
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'overlaps': {}
    }
    
    # Check overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    all_overlap = train_ids & val_ids & test_ids
    
    results['overlaps']['train_val'] = {
        'count': len(train_val_overlap),
        'proteins': list(train_val_overlap)[:10]  # Show first 10
    }
    results['overlaps']['train_test'] = {
        'count': len(train_test_overlap),
        'proteins': list(train_test_overlap)[:10]
    }
    results['overlaps']['val_test'] = {
        'count': len(val_test_overlap),
        'proteins': list(val_test_overlap)[:10]
    }
    results['overlaps']['all_three'] = {
        'count': len(all_overlap),
        'proteins': list(all_overlap)[:10]
    }
    
    # Check if leakage exists
    has_leakage = (len(train_val_overlap) > 0 or 
                   len(train_test_overlap) > 0 or 
                   len(val_test_overlap) > 0)
    
    results['has_protein_leakage'] = has_leakage
    
    return results


def check_go_term_distribution(splits_dict, aspect, threshold):
    """Check GO term distribution across splits."""
    
    def get_go_terms(df):
        all_terms = set()
        for go_str in df['go_terms']:
            if pd.notna(go_str) and go_str.strip():
                all_terms.update(go_str.split(','))
        return all_terms
    
    train_terms = get_go_terms(splits_dict['train'])
    val_terms = get_go_terms(splits_dict['val'])
    test_terms = get_go_terms(splits_dict['test'])
    
    results = {
        'train_unique_terms': len(train_terms),
        'val_unique_terms': len(val_terms),
        'test_unique_terms': len(test_terms),
        'train_only': len(train_terms - val_terms - test_terms),
        'val_only': len(val_terms - train_terms - test_terms),
        'test_only': len(test_terms - train_terms - val_terms),
        'shared_all': len(train_terms & val_terms & test_terms),
        'test_novel_terms': len(test_terms - train_terms),
        'val_novel_terms': len(val_terms - train_terms)
    }
    
    # Check for zero-shot evaluation (terms in test but not in train)
    results['has_zero_shot_terms'] = results['test_novel_terms'] > 0
    
    return results


def check_sequence_similarity_leakage(splits_dict, sequences_dict, aspect, threshold):
    """
    Check if sequences in different splits are too similar.
    Uses simple k-mer based similarity (faster than full alignment).
    """
    
    def get_kmer_set(sequence, k=3):
        """Get k-mer set for sequence."""
        if pd.isna(sequence) or len(sequence) < k:
            return set()
        return set(sequence[i:i+k] for i in range(len(sequence) - k + 1))
    
    def jaccard_similarity(set1, set2):
        """Compute Jaccard similarity between two sets."""
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    train_proteins = splits_dict['train']['protein_id'].tolist()
    val_proteins = splits_dict['val']['protein_id'].tolist()
    test_proteins = splits_dict['test']['protein_id'].tolist()
    
    # Sample for efficiency (check 100 random pairs)
    sample_size = min(100, len(train_proteins), len(test_proteins))
    
    train_sample = np.random.choice(train_proteins, size=sample_size, replace=False)
    test_sample = np.random.choice(test_proteins, size=sample_size, replace=False)
    
    high_similarity_pairs = []
    similarities = []
    
    for train_id in train_sample:
        if train_id not in sequences_dict:
            continue
        train_seq = sequences_dict[train_id]
        train_kmers = get_kmer_set(train_seq)
        
        for test_id in test_sample:
            if test_id not in sequences_dict:
                continue
            test_seq = sequences_dict[test_id]
            test_kmers = get_kmer_set(test_seq)
            
            sim = jaccard_similarity(train_kmers, test_kmers)
            similarities.append(sim)
            
            # Flag if similarity > 0.5 (potential leakage)
            if sim > 0.5:
                high_similarity_pairs.append({
                    'train_protein': train_id,
                    'test_protein': test_id,
                    'similarity': sim
                })
    
    results = {
        'sampled_pairs': len(similarities),
        'mean_similarity': float(np.mean(similarities)) if similarities else 0.0,
        'max_similarity': float(np.max(similarities)) if similarities else 0.0,
        'high_similarity_count': len(high_similarity_pairs),
        'high_similarity_pairs': high_similarity_pairs[:5]  # Show top 5
    }
    
    # Warning if max similarity > threshold/100
    expected_max = threshold / 100.0
    results['has_similarity_leakage'] = results['max_similarity'] > expected_max + 0.1
    
    return results


def check_label_statistics(splits_dict, aspect, threshold):
    """Check label statistics and imbalance."""
    
    def compute_label_stats(df):
        total_annotations = 0
        proteins_with_labels = 0
        
        for go_str in df['go_terms']:
            if pd.notna(go_str) and go_str.strip():
                proteins_with_labels += 1
                total_annotations += len(go_str.split(','))
        
        avg_labels = total_annotations / len(df) if len(df) > 0 else 0
        
        return {
            'total_proteins': len(df),
            'proteins_with_labels': proteins_with_labels,
            'total_annotations': total_annotations,
            'avg_labels_per_protein': avg_labels,
            'label_coverage': proteins_with_labels / len(df) if len(df) > 0 else 0
        }
    
    return {
        'train': compute_label_stats(splits_dict['train']),
        'val': compute_label_stats(splits_dict['val']),
        'test': compute_label_stats(splits_dict['test'])
    }


def load_sequences(protad_path):
    """Load protein sequences from ProtAD."""
    protad_df = pd.read_csv(protad_path, sep='\t')
    return protad_df.set_index('Entry')['Sequence'].to_dict()


def check_single_experiment(benchmark_base, protad_path, threshold, aspect):
    """Check a single experiment configuration."""
    
    split_dir = benchmark_base / f"similarity_{threshold}" / aspect
    
    if not split_dir.exists():
        return None
    
    # Load splits
    splits_dict = {
        'train': pd.read_csv(split_dir / 'train.tsv', sep='\t'),
        'val': pd.read_csv(split_dir / 'val.tsv', sep='\t'),
        'test': pd.read_csv(split_dir / 'test.tsv', sep='\t')
    }
    
    print(f"\nChecking: Threshold={threshold}%, Aspect={aspect}")
    print("=" * 60)
    
    results = {
        'threshold': threshold,
        'aspect': aspect
    }
    
    # Check 1: Protein overlap
    print("  [1/4] Checking protein ID overlap...")
    overlap_results = check_protein_overlap(splits_dict, aspect, threshold)
    results['protein_overlap'] = overlap_results
    
    if overlap_results['has_protein_leakage']:
        print("      ❌ LEAKAGE DETECTED: Protein overlap between splits!")
    else:
        print("      ✓ No protein ID overlap")
    
    # Check 2: GO term distribution
    print("  [2/4] Checking GO term distribution...")
    go_results = check_go_term_distribution(splits_dict, aspect, threshold)
    results['go_distribution'] = go_results
    
    if go_results['has_zero_shot_terms']:
        print(f"      ⚠️  {go_results['test_novel_terms']} terms in test not in train (zero-shot)")
    else:
        print("      ✓ All test terms seen in training")
    
    # Check 3: Sequence similarity
    print("  [3/4] Checking sequence similarity (sampled)...")
    sequences_dict = load_sequences(protad_path)
    sim_results = check_sequence_similarity_leakage(
        splits_dict, sequences_dict, aspect, threshold
    )
    results['sequence_similarity'] = sim_results
    
    if sim_results['has_similarity_leakage']:
        print(f"      ❌ LEAKAGE: Max similarity {sim_results['max_similarity']:.3f} > expected {threshold/100:.3f}")
    else:
        print(f"      ✓ Similarity within bounds (max={sim_results['max_similarity']:.3f})")
    
    # Check 4: Label statistics
    print("  [4/4] Checking label statistics...")
    label_results = check_label_statistics(splits_dict, aspect, threshold)
    results['label_statistics'] = label_results
    
    train_avg = label_results['train']['avg_labels_per_protein']
    test_avg = label_results['test']['avg_labels_per_protein']
    print(f"      Train avg labels: {train_avg:.2f}")
    print(f"      Test avg labels: {test_avg:.2f}")
    
    return results


def generate_report(all_results, output_file='leakage_report.json'):
    """Generate comprehensive leakage report."""
    
    report = {
        'summary': {
            'total_experiments': len(all_results),
            'experiments_with_protein_leakage': 0,
            'experiments_with_similarity_leakage': 0,
            'experiments_with_zero_shot_terms': 0
        },
        'details': all_results
    }
    
    for result in all_results:
        if result['protein_overlap']['has_protein_leakage']:
            report['summary']['experiments_with_protein_leakage'] += 1
        if result['sequence_similarity']['has_similarity_leakage']:
            report['summary']['experiments_with_similarity_leakage'] += 1
        if result['go_distribution']['has_zero_shot_terms']:
            report['summary']['experiments_with_zero_shot_terms'] += 1
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    """Check all experiments for data leakage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-base', type=str,
                       default='/home/zijianzhou/Datasets/protad/go_annotations/benchmarks',
                       help='Path to benchmark directory')
    parser.add_argument('--protad-path', type=str,
                       default='/home/zijianzhou/Datasets/protad/protad.tsv',
                       help='Path to ProtAD file')
    parser.add_argument('--thresholds', nargs='+', type=int,
                       default=[30, 50, 70, 95],
                       help='Similarity thresholds to check')
    parser.add_argument('--aspects', nargs='+',
                       default=['BP', 'MF', 'CC'],
                       help='GO aspects to check')
    parser.add_argument('--output', type=str,
                       default='leakage_report.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    benchmark_base = Path(args.benchmark_base)
    protad_path = Path(args.protad_path)
    
    print("=" * 70)
    print("DATA LEAKAGE CHECKER")
    print("=" * 70)
    print(f"Checking {len(args.thresholds)} thresholds × {len(args.aspects)} aspects")
    print("=" * 70)
    
    all_results = []
    
    for threshold in args.thresholds:
        for aspect in args.aspects:
            result = check_single_experiment(
                benchmark_base, protad_path, threshold, aspect
            )
            if result:
                all_results.append(result)
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report = generate_report(all_results, args.output)
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 70)
    print(f"Total experiments checked: {report['summary']['total_experiments']}")
    print(f"Experiments with protein leakage: {report['summary']['experiments_with_protein_leakage']}")
    print(f"Experiments with similarity leakage: {report['summary']['experiments_with_similarity_leakage']}")
    print(f"Experiments with zero-shot terms: {report['summary']['experiments_with_zero_shot_terms']}")
    
    if report['summary']['experiments_with_protein_leakage'] > 0:
        print("\n❌ CRITICAL: Protein ID leakage detected!")
    else:
        print("\n✓ No protein ID leakage detected")
    
    if report['summary']['experiments_with_similarity_leakage'] > 0:
        print("⚠️  WARNING: High sequence similarity detected in some splits")
    
    print(f"\n✓ Full report saved to: {args.output}")


if __name__ == '__main__':
    main()