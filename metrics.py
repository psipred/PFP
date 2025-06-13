import torch

class MetricBundle:
    def __init__(self, device):
        self.device = device

    def __call__(self, logits, labels):
        probs = torch.sigmoid(logits)
        eps = 1e-9
        
        # Compute macro average precision (per-class, then average)
        num_classes = probs.shape[1]
        ap_scores = []
        
        for class_idx in range(num_classes):
            class_probs = probs[:, class_idx]
            class_labels = labels[:, class_idx]
            
            # Skip classes with no positive samples
            if class_labels.sum() == 0:
                continue
                
            # Sort by probability (descending)
            sorted_idx = torch.argsort(class_probs, descending=True)
            sorted_labels = class_labels[sorted_idx]
            
            # Compute cumulative TP and FP
            tp = sorted_labels.cumsum(dim=0).float()
            fp = (1 - sorted_labels).cumsum(dim=0).float()
            
            # Compute precision and recall
            precision = tp / (tp + fp + eps)
            recall = tp / (class_labels.sum() + eps)
            
            # Compute average precision using trapezoidal rule
            # AP = sum of (precision[i] * (recall[i] - recall[i-1]))
            recall_diff = torch.cat([torch.tensor([0.0], device=self.device), 
                                   recall[1:] - recall[:-1]])
            ap = (precision * recall_diff).sum()
            ap_scores.append(ap.item())
        
        map_ = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

        # F-max computation (this part was already memory-efficient)
        thresholds = torch.linspace(0, 1, 101, device=self.device)
        f1_scores = []
        for thr in thresholds:
            pred = (probs > thr).float()
            tp = (pred * labels).sum()
            fp = (pred * (1 - labels)).sum()
            fn = ((1 - pred) * labels).sum()
            
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            f1_scores.append(f1.item())
        
        fmax = max(f1_scores)

        return {"AP": map_, "Fmax": fmax}