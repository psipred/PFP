"""Configuration management for protein function prediction."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Training and model configuration."""
    
    # Paths
    benchmark_base: Path = Path("/home/zijianzhou/Datasets/protad/go_annotations/benchmarks")
    protad_path: Path = Path("/home/zijianzhou/Datasets/protad/protad.tsv")
    output_dir: Path = Path("./experiments")
    cache_dir: Path = Path("./embedding_cache")
    
    # Model settings
    pubmed_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    max_text_length: int = 512
    
    # Experiment settings
    similarity_threshold: int = 30
    aspect: str = 'BP'  # BP, MF, or CC
    debug_mode: bool = False
    
    # Text fields from ProtAD
    text_fields: list = None
    
    def __post_init__(self):
        if self.text_fields is None:
            self.text_fields = [
                'Protein names', 'Organism', 'Taxonomic lineage', 'Function',
                'Caution', 'Miscellaneous', 'Subunit structure', 'Induction',
                'Tissue specificity', 'Developmental stage', 'Allergenic properties',
                'Biotechnological use', 'Pharmaceutical use', 'Involvement in disease',
                'Subcellular location', 'Post-translational modification', 'Sequence similarities'
            ]
        
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