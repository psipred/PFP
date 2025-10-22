from pathlib import Path
from dataclasses import dataclass
import torch


@dataclass
class CAFA3Config:
    """Training configuration for CAFA3 experiments."""
    
    # Paths
    data_dir: Path = Path("./data")
    embedding_dir: Path = Path("./embedding_cache")
    output_dir: Path = Path("./plm_results")

    # Model settings
    plm_type: str = "esm"  # 'esm', 'prott5', 'prostt5', 'ankh'
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    
    # Early stopping
    early_stopping_patience: int = 5
    min_delta: float = 0.0001
    
    # Experiment settings
    aspect: str = 'BPO'  # 'BPO', 'CCO', 'MFO'
    
    # Reproducibility
    seed: int = 42  # ADD THIS
    
    def __post_init__(self):
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup paths
        exp_dir = self.output_dir / self.plm_type / self.aspect
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.results_dir = exp_dir / "results"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Set embedding dimension based on PLM
        self.embedding_dims = {
            'esm': 1280,
            'prott5': 1024,
            'prostt5': 1024,
            'ankh': 768,  # Ankh-base
            'text': 768
        }
        self.embedding_dim = self.embedding_dims[self.plm_type]