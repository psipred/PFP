import torch

class MetricBundle:
    def __init__(self, device):
        self.device = device

    def __call__(self, logits, labels):
        probs = torch.sigmoid(logits)
        eps = 1e-9
        # average precision (macro)
        sorted_idx = torch.argsort(probs, dim=0, descending=True)
        tp = labels[sorted_idx].cumsum(dim=0)
        fp = (1 - labels)[sorted_idx].cumsum(dim=0)
        precision = tp / (tp + fp + eps)
        recall    = tp / (labels.sum(dim=0) + eps)
        # Align lengths: use precision[1:] so both tensors have the same size (N‑1)
        ap = (precision[1:] * (recall[1:] - recall[:-1]).clamp(min=0)).sum(dim=0)
        map_ = ap.mean().item()

        # F-max
        thresholds = torch.linspace(0, 1, 101, device=self.device)
        f1_scores = []
        for thr in thresholds:
            pred = (probs > thr).float()
            tp = (pred * labels).sum()
            prec = tp / (pred.sum() + eps)
            rec  = tp / (labels.sum() + eps)
            f1_scores.append((2 * prec * rec / (prec + rec + eps)).item())
        fmax = max(f1_scores)

        return {"AP": map_, "Fmax": fmax}