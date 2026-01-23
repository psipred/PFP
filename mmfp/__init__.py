"""MMFP: MultiModal Fusion for Protein Function Prediction."""

from .models import (
    MultiModalFusionModel,
    ModalityEncoder,
    ConcatFusion,
    BilinearGatedFusion,
    create_model,
    FUSION_REGISTRY,
)
from .dataset import MultiModalDataset, collate_fn
from .evaluation import evaluate_with_cafa, compute_information_accretion

__version__ = "1.0.0"
__all__ = [
    "MultiModalFusionModel",
    "ModalityEncoder",
    "ConcatFusion",
    "BilinearGatedFusion",
    "create_model",
    "FUSION_REGISTRY",
    "MultiModalDataset",
    "collate_fn",
    "evaluate_with_cafa",
    "compute_information_accretion",
]
