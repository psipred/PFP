"""Configuration management for protein function prediction."""

from pathlib import Path
from dataclasses import dataclass
import os


@dataclass
class Config:
    """Training and model configuration."""
    
    # Paths
    benchmark_base: Path = Path("../benchmark/go_annotations/benchmarks")
    protad_path: Path = Path("../benchmark/protad.tsv")
    # benchmark_base: Path = Path("/home/zijianzhou/Datasets/protad/go_annotations/benchmarks")
    # protad_path: Path = Path("/home/zijianzhou/Datasets/protad/protad.tsv")
    # benchmark_base: Path = Path("../benchmark/go_annotations/benchmarks")
    # protad_path: Path = Path("../benchmark/protad.tsv")
    benchmark_base: Path = Path("/home/zijianzhou/Datasets/protad/go_annotations/benchmarks")
    protad_path: Path = Path("/home/zijianzhou/Datasets/protad/protad.tsv")
    output_dir: Path = Path("./experiments")
    cache_dir: Path = Path("./embedding_cache")
    
    # Model settings
    pubmed_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_epochs: int = 50
    max_text_length: int = 512
    
    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 0.0001
    
    # Experiment settings
    similarity_threshold: int = 30
    aspect: str = 'BP'  # BP, MF, or CC
    debug_mode: bool = False
    
    # Text fields from ProtAD
    text_fields: list = None
    device_mode: str = os.getenv('DEVICE_MODE', 'auto')  # 'auto', 'single', 'multi'

    def __post_init__(self):
        if self.text_fields is None:
            self.text_fields = [
                'Protein names', 'Organism', 'Taxonomic lineage', 'Function',
                'Caution', 'Miscellaneous', 'Subunit structure', 'Induction',
                'Tissue specificity', 'Developmental stage', 'Allergenic properties',
                'Biotechnological use', 'Pharmaceutical use', 'Involvement in disease',
                'Subcellular location', 'Post-translational modification', 'Sequence similarities'
            ]
        self.setup_devices()
        # Setup paths
        self.split_dir = self.benchmark_base / f"similarity_{self.similarity_threshold}" / self.aspect
        exp_dir = self.output_dir / f"sim_{self.similarity_threshold}" / self.aspect
        self.checkpoint_dir = exp_dir / "checkpoints"
        self.results_dir = exp_dir / "results"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Debug mode adjustments
        if self.debug_mode:
            self.num_epochs = 2
            self.batch_size = 16
            self.early_stopping_patience = 2


    def setup_devices(self):
        """Auto-detect and configure available GPUs."""
        import torch
        
        if not torch.cuda.is_available():
            self.device = 'cpu'
            self.n_gpus = 0
            return
        
        self.n_gpus = torch.cuda.device_count()
        
        if self.device_mode == 'single' or self.n_gpus == 1:
            # Use single GPU (your 4070ti)
            self.device = 'cuda:0'
            self.use_ddp = False
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        
        elif self.device_mode == 'multi' and self.n_gpus > 1:
            # Use DataParallel for multi-GPU (your 6x 1080ti node)
            self.device = 'cuda'
            self.use_ddp = True
            print(f"Using {self.n_gpus} GPUs with DataParallel")
            for i in range(self.n_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        else:
            # Auto mode
            self.device = 'cuda:0' if self.n_gpus == 1 else 'cuda'
            self.use_ddp = self.n_gpus > 1
            print(f"Auto-detected {self.n_gpus} GPU(s)")