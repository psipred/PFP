"""Evaluation metrics for GO prediction."""

import numpy as np


def compute_fmax(y_true, y_pred, thresholds=None):
    """
    Compute Fmax metric for GO term prediction.
    
    Args:
        y_true: Ground truth labels (N x K)
        y_pred: Predicted probabilities (N x K)
        thresholds: List of thresholds to evaluate
    
    Returns:
        fmax, best_threshold, precision, recall
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    best_fmax = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum(axis=1)
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum(axis=1)
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum(axis=1)
        
        precision = (tp / (tp + fp + 1e-10)).mean()
        recall = (tp / (tp + fn + 1e-10)).mean()
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return best_fmax, best_threshold, best_precision, best_recall