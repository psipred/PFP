"""CAFA-style evaluation with Information Accretion support.

This module provides evaluation utilities compatible with the CAFA
(Critical Assessment of Function Annotation) benchmark.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


def compute_fmax(y_true: np.ndarray, y_pred: np.ndarray,
                 thresholds: Optional[np.ndarray] = None) -> tuple:
    """Compute micro-averaged F-max.

    Args:
        y_true: Binary ground truth labels [n_samples, n_terms]
        y_pred: Prediction scores [n_samples, n_terms]
        thresholds: Thresholds to evaluate (default: 100 values from 0.01 to 0.99)

    Returns:
        Tuple of (fmax, best_threshold, precision, recall)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    best_fmax = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)

        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_fmax, best_threshold, best_precision, best_recall


def compute_information_accretion(
    labels: np.ndarray,
    go_terms: List[str],
    obo_file: str
) -> Dict[str, float]:
    """Compute Information Accretion (IA) for GO terms from training annotations.

    Uses proper propagation and multi-parent handling:
    - Propagates annotations to all ancestors
    - IA(v) = -log2(count(v) / count(proteins with ALL parents of v))

    Args:
        labels: Binary annotation matrix [n_proteins, n_terms]
        go_terms: List of GO term IDs (same order as labels columns)
        obo_file: Path to GO obo file

    Returns:
        Dict mapping GO term ID to IA value
    """
    import obonet
    import networkx as nx
    from scipy.sparse import dok_matrix

    # Load ontology
    ontology = obonet.read_obo(str(obo_file))

    # Keep only is_a and part_of edges
    if isinstance(ontology, nx.MultiDiGraph):
        remove_edges = [(i, j, k) for i, j, k in ontology.edges
                        if k not in ("is_a", "part_of")]
        ontology.remove_edges_from(remove_edges)

    # Build term index
    term_to_idx = {term: i for i, term in enumerate(go_terms)}
    n_proteins = labels.shape[0]
    n_terms = len(go_terms)

    # Get ancestors for each term
    ancestors = {}
    for term in go_terms:
        if term in ontology:
            ancestors[term] = nx.descendants(ontology, term)
        else:
            ancestors[term] = set()

    # Create count matrix with propagated annotations
    count_matrix = dok_matrix((n_proteins + 1, n_terms), dtype=np.int8)
    count_matrix[n_proteins, :] = 1  # dummy protein for regularization

    for i in range(n_proteins):
        annotated_terms = set()
        pos = np.where(labels[i] == 1)[0]
        for j in pos:
            term = go_terms[j]
            annotated_terms.add(term)
            for anc in ancestors.get(term, ()):
                if anc in term_to_idx:
                    annotated_terms.add(anc)
        for term in annotated_terms:
            count_matrix[i, term_to_idx[term]] = 1

    count_matrix = count_matrix.tocsc()

    # Compute IA for each term
    ia_values = {}
    for term in go_terms:
        idx = term_to_idx[term]

        # Get direct parents
        if term in ontology:
            parents = set(nx.descendants_at_distance(ontology, term, 1))
            parents = [p for p in parents if p in term_to_idx]
        else:
            parents = []

        prots_with_term = count_matrix[:, idx].sum()

        if not parents:
            prots_with_parents = n_proteins + 1
        else:
            parent_indices = [term_to_idx[p] for p in parents]
            num_parents = len(parent_indices)
            parent_counts = count_matrix[:, parent_indices].toarray().sum(axis=1)
            prots_with_parents = (parent_counts == num_parents).sum()

        if prots_with_term == prots_with_parents:
            ia_values[term] = 0.0
        else:
            ia_values[term] = -np.log2(prots_with_term / prots_with_parents)

    return ia_values


def save_ia_file(ia_values: Dict[str, float], output_file: str) -> None:
    """Save IA values in cafaeval format."""
    with open(output_file, 'w') as f:
        for term, ia in ia_values.items():
            f.write(f"{term}\t{ia:.6f}\n")


def save_predictions_cafa_format(
    predictions: np.ndarray,
    protein_ids: List[str],
    go_terms: List[str],
    output_file: str
) -> None:
    """Save predictions in CAFA format (protein_id, term_id, score)."""
    with open(output_file, 'w') as f:
        for i, protein_id in enumerate(protein_ids):
            for j, go_term in enumerate(go_terms):
                score = predictions[i, j]
                if score > 0:
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")


def save_ground_truth_cafa_format(
    labels: np.ndarray,
    protein_ids: List[str],
    go_terms: List[str],
    output_file: str
) -> None:
    """Save ground truth in CAFA format (protein_id, term_id)."""
    with open(output_file, 'w') as f:
        for i, protein_id in enumerate(protein_ids):
            for j, go_term in enumerate(go_terms):
                if labels[i, j] == 1:
                    f.write(f"{protein_id}\t{go_term}\n")


def run_cafa_evaluator(
    obo_file: str,
    pred_file: str,
    truth_file: str,
    output_dir: str,
    ia_file: Optional[str] = None
) -> Optional[Path]:
    """Run CAFA evaluator using cafaeval package.

    Args:
        obo_file: Path to GO ontology file
        pred_file: Path to predictions directory
        truth_file: Path to ground truth file
        output_dir: Output directory for results
        ia_file: Optional path to Information Accretion file

    Returns:
        Path to results directory, or None if evaluation failed
    """
    try:
        from cafaeval.evaluation import cafa_eval, write_results

        print(f"Running CAFA evaluation...")
        print(f"  OBO file: {obo_file}")
        print(f"  Predictions: {pred_file}")
        print(f"  Ground truth: {truth_file}")
        if ia_file:
            print(f"  IA file: {ia_file}")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            if ia_file:
                results = cafa_eval(
                    str(obo_file),
                    str(pred_file),
                    str(truth_file),
                    ia=str(ia_file),
                    norm='cafa',
                    prop='max'
                )
            else:
                results = cafa_eval(
                    str(obo_file),
                    str(pred_file),
                    str(truth_file),
                    norm='cafa',
                    prop='max'
                )
        except TypeError as e:
            print(f"  Note: Using minimal cafa_eval parameters due to: {e}")
            results = cafa_eval(
                str(obo_file),
                str(pred_file),
                str(truth_file)
            )

        write_results(*results, out_dir=str(output_dir))
        print(f"  CAFA evaluation complete. Results saved to: {output_dir}")
        return output_dir

    except ImportError:
        print("cafaeval package not found. Install with: pip install cafaeval")
        return None
    except Exception as e:
        print(f"Error during CAFA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_with_cafa(
    model,
    loader,
    device,
    protein_ids: List[str],
    go_terms: List[str],
    obo_file: str,
    output_dir: str,
    model_type: str = 'fusion',
    model_name: str = 'model',
    train_labels: Optional[np.ndarray] = None,
    ia_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate model predictions using CAFA evaluator.

    Args:
        model: Trained PyTorch model
        loader: Test data loader
        device: Torch device
        protein_ids: List of protein IDs in test set
        go_terms: List of GO term IDs
        obo_file: Path to GO obo file
        output_dir: Directory to save results
        model_type: Model type identifier
        model_name: Name for the prediction file
        train_labels: Training labels for IA computation [n_train, n_terms]
        ia_file: Optional path to a precomputed IA file

    Returns:
        Dict with CAFA metrics
    """
    import torch
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    temp_dir = output_dir / 'cafa_temp'
    temp_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = temp_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating predictions for CAFA evaluation...")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            # Handle different input formats
            if 'embeddings' in batch:
                inputs = batch['embeddings'].to(device)
                logits = model(inputs)
            else:
                # Multimodal input
                seq = batch['seq'].to(device)
                seq_mask = batch['seq_mask'].to(device)
                text = batch['text'].to(device)
                text_mask = batch['text_mask'].to(device)
                struct = batch['struct'].to(device)
                struct_mask = batch['struct_mask'].to(device)
                ppi = batch['ppi'].to(device)
                ppi_mask = batch['ppi_mask'].to(device)

                logits, _, _ = model(
                    seq, seq_mask, text, text_mask,
                    struct, struct_mask, ppi, ppi_mask
                )

            labels = batch['labels']
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    predictions = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    # Save predictions and ground truth
    pred_file = pred_dir / f"{model_name}.tsv"
    truth_file = temp_dir / "ground_truth.tsv"

    print("Saving predictions in CAFA format...")
    save_predictions_cafa_format(predictions, protein_ids, go_terms, pred_file)
    save_ground_truth_cafa_format(labels, protein_ids, go_terms, truth_file)

    # Compute and save IA if training labels provided
    computed_ia_file = ia_file
    if computed_ia_file is None and train_labels is not None:
        print("Computing Information Accretion from training data...")
        try:
            ia_values = compute_information_accretion(train_labels, go_terms, obo_file)
            computed_ia_file = temp_dir / "ia.txt"
            save_ia_file(ia_values, computed_ia_file)
            print(f"  IA computed for {len(ia_values)} terms")
        except Exception as e:
            print(f"  Warning: IA computation failed: {e}")
            computed_ia_file = None
    elif computed_ia_file is not None:
        print(f"Using precomputed IA file: {computed_ia_file}")

    # Run CAFA evaluator
    cafa_results_dir = output_dir / 'cafa_results'
    result_path = run_cafa_evaluator(
        obo_file=obo_file,
        pred_file=str(pred_dir),
        truth_file=str(truth_file),
        output_dir=str(cafa_results_dir),
        ia_file=str(computed_ia_file) if computed_ia_file else None
    )

    metrics = {}

    if result_path:
        # Parse results
        all_results_file = Path(result_path) / "evaluation_all.tsv"
        if all_results_file.exists():
            df = pd.read_csv(all_results_file, sep='\t')
            best_f = df.loc[df['f'].idxmax()]

            metrics = {
                'fmax': float(best_f['f']),
                'threshold': float(best_f['tau']),
                'precision': float(best_f['pr']),
                'recall': float(best_f['rc']),
            }

            if 'cov' in best_f:
                metrics['coverage'] = float(best_f['cov'])

            # Weighted S-min
            if 's_w' in df.columns:
                best_sw_row = df.loc[df['s_w'].idxmin()]
                metrics.update({
                    'wsmin': float(best_sw_row['s_w']),
                    'wsmin_threshold': float(best_sw_row['tau']),
                    'wru': float(best_sw_row['ru_w']),
                    'wmi': float(best_sw_row['mi_w'])
                })

        # S-min
        best_s_file = Path(result_path) / "evaluation_best_s.tsv"
        if best_s_file.exists():
            df_s = pd.read_csv(best_s_file, sep="\t")
            row = df_s.iloc[0]
            metrics["smin"] = float(row["s"])

        # Weighted F-max
        best_fw_file = Path(result_path) / "evaluation_best_f_w.tsv"
        if best_fw_file.exists():
            df_w = pd.read_csv(best_fw_file, sep="\t")
            row = df_w.iloc[0]
            metrics["wfmax"] = float(row["f_w"])
            metrics["wprecision"] = float(row["pr_w"])
            metrics["wrecall"] = float(row["rc_w"])
            metrics["wthreshold"] = float(row["tau"])

        # Print results
        print("\nCAFA Evaluation Results:")
        print(f"  Fmax: {metrics.get('fmax', 0):.4f} (tau={metrics.get('threshold', 0):.2f})")
        print(f"  Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}")
        if 'wfmax' in metrics:
            print(f"  Weighted Fmax: {metrics['wfmax']:.4f}")
        if 'smin' in metrics:
            print(f"  Smin: {metrics['smin']:.4f}")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    return metrics
