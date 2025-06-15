#!/usr/bin/env python3
"""
Comprehensive Experimental Pipeline for Multi-Modal Gene Ontology Prediction
Location: /SAN/bioinf/PFP/PFP/experiments/multimodal_comparison/experiment_pipeline.py
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import subprocess
from dataclasses import dataclass, asdict
import itertools
from tqdm import tqdm

# Add project root to path
sys.path.append('/SAN/bioinf/PFP/PFP')


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    experiment_type: str  # 'baseline', 'structure', 'multimodal'
    features: List[str]  # ['esm', 'text', 'structure']
    graph_config: Optional[Dict[str, Any]] = None
    fusion_method: str = 'concat'  # 'concat', 'attention', 'mmstie'
    aspect: str = 'CCO'  # 'BPO', 'CCO', 'MFO'
    fold: int = 0
    
    def to_dict(self):
        return asdict(self)


class DatasetAligner:
    """Ensure all experiments use the same aligned dataset."""
    
    def __init__(self, base_dir: str = "/SAN/bioinf/PFP/PFP"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "Data" / "network_training_data"
        self.embeddings_base = Path("/SAN/bioinf/PFP/embeddings/cafa5_small")
        self.pdb_dir = Path("/SAN/bioinf/PFP/embeddings/structure/pdb_files")
        
        # Paths to different embedding types
        self.esm_dir = self.embeddings_base / "esm_af"
        self.text_dir = self.embeddings_base / "prot2text" / "text_embeddings"
        
        self.logger = logging.getLogger(__name__)
        
    def find_common_proteins(self, aspect: str, fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Find proteins that have all required data modalities."""
        
        # Load train and validation names
        train_names = np.load(
            self.data_dir / aspect / f"{aspect}_train_names_fold{fold}.npy"
        )
        valid_names = np.load(
            self.data_dir / aspect / f"{aspect}_valid_names_fold{fold}.npy"
        )
        
        all_names = np.concatenate([train_names, valid_names])
        
        # Check availability of each modality
        proteins_with_all_data = []
        missing_data_summary = {
            'esm': 0,
            'text': 0,
            'structure': 0,
            'multiple': 0
        }
        
        self.logger.info(f"Checking data availability for {len(all_names)} proteins...")
        
        for name in tqdm(all_names, desc="Validating data availability"):
            has_esm = (self.esm_dir / f"{name}.npy").exists()
            has_text = (self.text_dir / f"{name}.npy").exists()
            has_pdb = (self.pdb_dir / f"AF-{name}-F1-model_v4.pdb").exists()
            
            if has_esm and has_text and has_pdb:
                proteins_with_all_data.append(name)
            else:
                # Track what's missing
                missing_count = sum([not has_esm, not has_text, not has_pdb])
                if missing_count > 1:
                    missing_data_summary['multiple'] += 1
                else:
                    if not has_esm:
                        missing_data_summary['esm'] += 1
                    if not has_text:
                        missing_data_summary['text'] += 1
                    if not has_pdb:
                        missing_data_summary['structure'] += 1
        
        # Split back into train/valid
        proteins_with_all_data = set(proteins_with_all_data)
        
        aligned_train_names = [n for n in train_names if n in proteins_with_all_data]
        aligned_valid_names = [n for n in valid_names if n in proteins_with_all_data]
        
        self.logger.info(f"\nData availability summary:")
        self.logger.info(f"  Total proteins: {len(all_names)}")
        self.logger.info(f"  With all modalities: {len(proteins_with_all_data)} "
                        f"({len(proteins_with_all_data)/len(all_names)*100:.1f}%)")
        self.logger.info(f"  Missing ESM only: {missing_data_summary['esm']}")
        self.logger.info(f"  Missing Text only: {missing_data_summary['text']}")
        self.logger.info(f"  Missing Structure only: {missing_data_summary['structure']}")
        self.logger.info(f"  Missing multiple: {missing_data_summary['multiple']}")
        self.logger.info(f"\nAligned dataset sizes:")
        self.logger.info(f"  Train: {len(aligned_train_names)} (was {len(train_names)})")
        self.logger.info(f"  Valid: {len(aligned_valid_names)} (was {len(valid_names)})")
        
        return np.array(aligned_train_names), np.array(aligned_valid_names)
    
    def create_aligned_datasets(self, output_dir: str):
        """Create aligned datasets for all aspects."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        alignment_summary = {}
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {aspect}")
            self.logger.info('='*60)
            
            aligned_train, aligned_valid = self.find_common_proteins(aspect, fold=0)
            
            # Save aligned names
            np.save(output_path / f"{aspect}_aligned_train_names.npy", aligned_train)
            np.save(output_path / f"{aspect}_aligned_valid_names.npy", aligned_valid)
            
            # Create aligned labels by filtering original labels
            original_train_labels = np.load(
                self.data_dir / aspect / f"{aspect}_train_labels_fold0.npz"
            )
            original_valid_labels = np.load(
                self.data_dir / aspect / f"{aspect}_valid_labels_fold0.npz"
            )
            
            # Get indices of aligned proteins in original arrays
            original_train_names = np.load(
                self.data_dir / aspect / f"{aspect}_train_names_fold0.npy"
            )
            original_valid_names = np.load(
                self.data_dir / aspect / f"{aspect}_valid_names_fold0.npy"
            )
            
            train_indices = [i for i, n in enumerate(original_train_names) 
                           if n in aligned_train]
            valid_indices = [i for i, n in enumerate(original_valid_names) 
                           if n in aligned_valid]
            
            # Filter labels
            import scipy.sparse as ssp
            if isinstance(original_train_labels, np.lib.npyio.NpzFile):
                train_labels_sparse = ssp.load_npz(
                    self.data_dir / aspect / f"{aspect}_train_labels_fold0.npz"
                )
                valid_labels_sparse = ssp.load_npz(
                    self.data_dir / aspect / f"{aspect}_valid_labels_fold0.npz"
                )
                
                aligned_train_labels = train_labels_sparse[train_indices]
                aligned_valid_labels = valid_labels_sparse[valid_indices]
                
                ssp.save_npz(
                    output_path / f"{aspect}_aligned_train_labels.npz",
                    aligned_train_labels
                )
                ssp.save_npz(
                    output_path / f"{aspect}_aligned_valid_labels.npz",
                    aligned_valid_labels
                )
            
            alignment_summary[aspect] = {
                'original_train': len(original_train_names),
                'original_valid': len(original_valid_names),
                'aligned_train': len(aligned_train),
                'aligned_valid': len(aligned_valid),
                'coverage': (len(aligned_train) + len(aligned_valid)) / 
                           (len(original_train_names) + len(original_valid_names))
            }
        
        # Save alignment summary
        with open(output_path / "alignment_summary.json", 'w') as f:
            json.dump(alignment_summary, f, indent=2)
        
        return alignment_summary


class ExperimentGenerator:
    """Generate experiment configurations according to the design."""
    
    def __init__(self, aligned_data_dir: str, base_config_path: str):
        self.aligned_data_dir = Path(aligned_data_dir)
        self.base_config_path = Path(base_config_path)
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_all_experiments(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        experiments = []
        
        # Group A: Baseline Models (Single Modality)
        experiments.extend(self._generate_baseline_experiments())
        
        # Group B: Structure-Only Models
        experiments.extend(self._generate_structure_experiments())
        
        # Group C: Multi-Modal Combination Models
        experiments.extend(self._generate_multimodal_experiments())
        
        return experiments
    
    def _generate_baseline_experiments(self) -> List[ExperimentConfig]:
        """Generate baseline single-modality experiments."""
        experiments = []
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            # Model A: ESM-only
            experiments.append(ExperimentConfig(
                name=f"A_ESM_only_{aspect}",
                experiment_type='baseline',
                features=['esm'],
                aspect=aspect
            ))
            
            # Model B: Text-only
            experiments.append(ExperimentConfig(
                name=f"B_Text_only_{aspect}",
                experiment_type='baseline',
                features=['text'],
                aspect=aspect
            ))
        
        return experiments
    
    def _generate_structure_experiments(self) -> List[ExperimentConfig]:
        """Generate structure-only experiments with different graph configs."""
        experiments = []
        
        # Define graph configurations
        graph_configs = [
            # C1: Radius Graph + One-Hot
            {
                'name': 'C1_Radius_OneHot',
                'type': 'radius',
                'radius': 10.0,
                'use_esm_features': False
            },
            # C2: Radius Graph + ESM
            {
                'name': 'C2_Radius_ESM',
                'type': 'radius',
                'radius': 10.0,
                'use_esm_features': True
            },
            # C3: k-NN Graph + One-Hot
            {
                'name': 'C3_kNN_OneHot',
                'type': 'knn',
                'k': 10,
                'use_esm_features': False
            },
            # C4: k-NN Graph + ESM
            {
                'name': 'C4_kNN_ESM',
                'type': 'knn',
                'k': 10,
                'use_esm_features': True
            }
        ]
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            for graph_config in graph_configs:
                experiments.append(ExperimentConfig(
                    name=f"{graph_config['name']}_{aspect}",
                    experiment_type='structure',
                    features=['structure'],
                    graph_config=graph_config,
                    aspect=aspect
                ))
        
        return experiments
    
    def _generate_multimodal_experiments(self) -> List[ExperimentConfig]:
        """Generate multi-modal combination experiments."""
        experiments = []
        
        for aspect in ['BPO', 'CCO', 'MFO']:
            # Model D: ESM + Text
            experiments.append(ExperimentConfig(
                name=f"D_ESM_Text_{aspect}",
                experiment_type='multimodal',
                features=['esm', 'text'],
                fusion_method='concat',
                aspect=aspect
            ))
            
            # Model E: ESM + Best Structure (to be determined)
            experiments.append(ExperimentConfig(
                name=f"E_ESM_Structure_{aspect}",
                experiment_type='multimodal',
                features=['esm', 'structure'],
                fusion_method='concat',
                graph_config={
                    'name': 'best_structure',
                    'type': 'knn',
                    'k': 10,
                    'use_esm_features': True
                },
                aspect=aspect
            ))
            
            # Model F: ESM + Text + Best Structure
            experiments.append(ExperimentConfig(
                name=f"F_ESM_Text_Structure_{aspect}",
                experiment_type='multimodal',
                features=['esm', 'text', 'structure'],
                fusion_method='concat',
                graph_config={
                    'name': 'best_structure',
                    'type': 'knn',
                    'k': 10,
                    'use_esm_features': True
                },
                aspect=aspect
            ))
        
        return experiments
    
    def create_config_files(self, experiments: List[ExperimentConfig], output_dir: str):
        """Create configuration files for each experiment."""
        configs_dir = Path(output_dir) / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        for exp in experiments:
            config = self._create_experiment_config(exp)
            
            config_path = configs_dir / f"{exp.name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def _create_experiment_config(self, exp: ExperimentConfig) -> Dict[str, Any]:
        """Create detailed configuration for an experiment."""
        config = self.base_config.copy()
        
        # Update paths to use aligned data
        config['dataset']['train_names'] = str(
            self.aligned_data_dir / f"{exp.aspect}_aligned_train_names.npy"
        )
        config['dataset']['train_labels'] = str(
            self.aligned_data_dir / f"{exp.aspect}_aligned_train_labels.npz"
        )
        config['dataset']['valid_names'] = str(
            self.aligned_data_dir / f"{exp.aspect}_aligned_valid_names.npy"
        )
        config['dataset']['valid_labels'] = str(
            self.aligned_data_dir / f"{exp.aspect}_aligned_valid_labels.npz"
        )
        
        # Set aspect-specific output dimensions
        output_dims = {'BPO': 1302, 'CCO': 453, 'MFO': 483}
        config['model']['output_dim'] = output_dims[exp.aspect]
        
        # Configure based on experiment type
        if exp.experiment_type == 'baseline':
            if 'esm' in exp.features:
                config['dataset']['embedding_type'] = 'esm_mean'
            elif 'text' in exp.features:
                config['dataset']['embedding_type'] = 'text'
        
        elif exp.experiment_type == 'structure':
            config['dataset']['embedding_type'] = 'structure'
            config['graph'] = exp.graph_config
            config['model']['use_structure'] = True
        
        elif exp.experiment_type == 'multimodal':
            config['dataset']['embedding_type'] = 'multimodal'
            config['dataset']['features'] = exp.features
            config['model']['fusion_method'] = exp.fusion_method
            
            if 'structure' in exp.features:
                config['graph'] = exp.graph_config
        
        # Set experiment-specific paths
        config['log']['out_dir'] = f"/SAN/bioinf/PFP/PFP/experiments/multimodal_comparison/results/{exp.name}"
        config['experiment_name'] = exp.name
        
        return config


class ExperimentRunner:
    """Manage experiment execution on the cluster."""
    
    def __init__(self, experiments_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.scripts_dir = self.experiments_dir / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
    def create_submission_scripts(self, experiments: List[ExperimentConfig]):
        """Create cluster submission scripts."""
        
        # Individual scripts for each experiment
        for exp in experiments:
            script_content = f"""#!/bin/bash
#$ -N {exp.name}
#$ -l h_vmem=48G
#$ -l tmem=60G
#$ -l h_rt=24:0:0
#$ -j y
#$ -o {self.experiments_dir}/logs/{exp.name}.log
#$ -wd /SAN/bioinf/PFP/PFP
#$ -l gpu=true
#$ -l h=!walter* # avoid hoots and walter nodes
#$ -pe gpu 1

echo "Starting experiment: {exp.name}"
echo "Date: $(date)"
echo "Node: $(hostname)"

# Activate environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run appropriate training script based on experiment type
cd /SAN/bioinf/PFP/PFP

if [[ "{exp.experiment_type}" == "structure" ]]; then
    python experiments/multimodal_comparison/train_unified.py \\
        --config {self.experiments_dir}/configs/{exp.name}.yaml \\
        --experiment-type structure
elif [[ "{exp.experiment_type}" == "multimodal" ]]; then
    python experiments/multimodal_comparison/train_unified.py \\
        --config {self.experiments_dir}/configs/{exp.name}.yaml \\
        --experiment-type multimodal
else
    python experiments/multimodal_comparison/train_unified.py \\
        --config {self.experiments_dir}/configs/{exp.name}.yaml \\
        --experiment-type baseline
fi

echo "Experiment completed: $(date)"
"""
            
            script_path = self.scripts_dir / f"{exp.name}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)
        
        # Create master submission script
        self._create_master_script(experiments)
        
        # Create monitoring script
        self._create_monitoring_script()
    
    def _create_master_script(self, experiments: List[ExperimentConfig]):
        """Create master submission script with proper ordering."""
        
        # Group experiments by type for ordered execution
        baseline_exps = [e for e in experiments if e.experiment_type == 'baseline']
        structure_exps = [e for e in experiments if e.experiment_type == 'structure']
        multimodal_exps = [e for e in experiments if e.experiment_type == 'multimodal']
        
        script_content = f"""#!/bin/bash
# Master submission script for multi-modal GO prediction experiments

SCRIPTS_DIR="{self.scripts_dir}"

echo "Submitting Multi-Modal GO Prediction Experiments"
echo "=============================================="

# Phase 1: Baseline experiments (can run in parallel)
echo "Phase 1: Submitting baseline experiments..."
"""
        
        for exp in baseline_exps:
            script_content += f'qsub "$SCRIPTS_DIR/{exp.name}.sh"\n'
        
        script_content += """
# Wait for baselines to complete before starting structure experiments
echo "Waiting 30 seconds before Phase 2..."
sleep 30

# Phase 2: Structure experiments (can run in parallel)
echo "Phase 2: Submitting structure experiments..."
"""
        
        for exp in structure_exps:
            script_content += f'qsub "$SCRIPTS_DIR/{exp.name}.sh"\n'
            script_content += "sleep 2\n"  # Stagger submissions
        
        script_content += """
# Note: Multi-modal experiments should be submitted after determining best structure config
echo "Phase 3: Multi-modal experiments will be submitted after structure experiments complete."
echo "Use: ./submit_multimodal.sh after analyzing structure results"

echo "Initial submission complete. Monitor with: ./monitor.sh"
"""
        
        master_path = self.scripts_dir / "submit_all.sh"
        with open(master_path, 'w') as f:
            f.write(script_content)
        master_path.chmod(0o755)
        
        # Create separate script for multimodal phase
        multimodal_script = """#!/bin/bash
# Submit multi-modal experiments after best structure config is determined

SCRIPTS_DIR=\"""" + str(self.scripts_dir) + """\"

echo "Submitting multi-modal experiments..."
"""
        
        for exp in multimodal_exps:
            multimodal_script += f'qsub "$SCRIPTS_DIR/{exp.name}.sh"\n'
            multimodal_script += "sleep 2\n"
        
        multimodal_path = self.scripts_dir / "submit_multimodal.sh"
        with open(multimodal_path, 'w') as f:
            f.write(multimodal_script)
        multimodal_path.chmod(0o755)
    
    def _create_monitoring_script(self):
        """Create experiment monitoring script."""
        script_content = f"""#!/bin/bash
# Monitor multi-modal GO prediction experiments

echo "Multi-Modal GO Prediction Experiment Status"
echo "=========================================="
echo "Time: $(date)"
echo

# Check running jobs
echo "RUNNING JOBS:"
qstat | grep -E "A_|B_|C[1-4]_|D_|E_|F_" | while read line; do
    job_id=$(echo $line | awk '{{print $1}}')
    job_name=$(echo $line | awk '{{print $3}}')
    status=$(echo $line | awk '{{print $5}}')
    echo "  $job_name (ID: $job_id, Status: $status)"
done

echo

# Check completed experiments
echo "COMPLETED EXPERIMENTS:"
find {self.experiments_dir}/results -name "final_metrics.json" 2>/dev/null | while read f; do
    exp_name=$(basename $(dirname $f))
    fmax=$(python -c "import json; data=json.load(open('$f')); print(f'F-max: {{data.get(\"best_Fmax_protein\", 0):.4f}}')")
    echo "  $exp_name - $fmax"
done

echo

# Check failed experiments
echo "FAILED EXPERIMENTS:"
grep -l "Error\\|Traceback" {self.experiments_dir}/logs/*.log 2>/dev/null | while read f; do
    exp_name=$(basename $f .log)
    echo "  $exp_name - Check log: $f"
done

echo
echo "For detailed progress, check individual logs in: {self.experiments_dir}/logs/"
"""
        
        monitor_path = self.scripts_dir / "monitor.sh"
        with open(monitor_path, 'w') as f:
            f.write(script_content)
        monitor_path.chmod(0o755)


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir.parent / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_all_results(self) -> pd.DataFrame:
        """Collect results from all experiments."""
        results = []
        
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            metrics_file = exp_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Parse experiment name
                exp_name = exp_dir.name
                parts = exp_name.split('_')
                
                result = {
                    'experiment': exp_name,
                    'model': parts[0],  # A, B, C1-C4, D, E, F
                    'aspect': parts[-1],  # BPO, CCO, MFO
                    **metrics
                }
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def create_analysis_report(self, df: pd.DataFrame):
        """Create comprehensive analysis report."""
        report_path = self.analysis_dir / "experiment_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Multi-Modal GO Prediction Experimental Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall best performance
            f.write("## 1. Best Overall Performance\n\n")
            best_idx = df['best_Fmax_protein'].idxmax()
            best_exp = df.loc[best_idx]
            
            f.write(f"**Best Model:** {best_exp['experiment']}\n")
            f.write(f"- F-max: {best_exp['best_Fmax_protein']:.4f}\n")
            f.write(f"- mAP: {best_exp.get('best_macro_AP', 0):.4f}\n")
            f.write(f"- AUROC: {best_exp.get('best_macro_AUROC', 0):.4f}\n\n")
            
            # Results by model group
            f.write("## 2. Results by Model Group\n\n")
            
            # Group A: Baselines
            f.write("### Group A: Baseline Models\n\n")
            baseline_models = ['A', 'B']
            for model in baseline_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    avg_fmax = model_df['best_Fmax_protein'].mean()
                    f.write(f"- Model {model}: Average F-max = {avg_fmax:.4f}\n")
            
            # Group B: Structure models
            f.write("\n### Group B: Structure Models\n\n")
            structure_models = ['C1', 'C2', 'C3', 'C4']
            structure_results = []
            for model in structure_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    avg_fmax = model_df['best_Fmax_protein'].mean()
                    structure_results.append((model, avg_fmax))
                    f.write(f"- Model {model}: Average F-max = {avg_fmax:.4f}\n")
            
            # Determine best structure configuration
            if structure_results:
                best_structure = max(structure_results, key=lambda x: x[1])
                f.write(f"\n**Best Structure Configuration:** {best_structure[0]}\n")
            
            # Group C: Multi-modal models
            f.write("\n### Group C: Multi-Modal Models\n\n")
            multimodal_models = ['D', 'E', 'F']
            for model in multimodal_models:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    avg_fmax = model_df['best_Fmax_protein'].mean()
                    f.write(f"- Model {model}: Average F-max = {avg_fmax:.4f}\n")
            
            # Results by aspect
            f.write("\n## 3. Results by GO Aspect\n\n")
            for aspect in ['BPO', 'CCO', 'MFO']:
                f.write(f"### {aspect}\n\n")
                aspect_df = df[df['aspect'] == aspect].sort_values('best_Fmax_protein', ascending=False)
                if not aspect_df.empty:
                    f.write("| Model | F-max | mAP | AUROC |\n")
                    f.write("|-------|-------|-----|-------|\n")
                    for _, row in aspect_df.head(5).iterrows():
                        f.write(f"| {row['model']} | {row['best_Fmax_protein']:.4f} | "
                               f"{row.get('best_macro_AP', 0):.4f} | "
                               f"{row.get('best_macro_AUROC', 0):.4f} |\n")
                f.write("\n")
        
        # Save detailed results
        df.to_csv(self.analysis_dir / "all_results.csv", index=False)
        
        print(f"Analysis report saved to: {report_path}")
        
    def create_visualizations(self, df: pd.DataFrame):
        """Create publication-ready figures."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # Figure 1: Overall Performance of Feature Combinations
            self._create_figure1_overall_performance(df)
            
            # Figure 2: Comparative Analysis of Structural Representations
            self._create_figure2_structure_comparison(df)
            
            # Figure 3: Precision-Recall Analysis
            self._create_figure3_pr_analysis(df)
            
            # Figure 4: Term-Centric Performance Gain Analysis
            self._create_figure4_term_analysis(df)
            
        except ImportError:
            print("matplotlib/seaborn not available, skipping visualizations")
    
    def _create_figure1_overall_performance(self, df: pd.DataFrame):
        """Create overall performance comparison figure."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for main models
        main_models = ['A', 'B', 'C2', 'C4', 'D', 'E', 'F']  # Best from each category
        model_names = {
            'A': 'ESM-only',
            'B': 'Text-only', 
            'C2': 'Structure (Radius+ESM)',
            'C4': 'Structure (kNN+ESM)',
            'D': 'ESM+Text',
            'E': 'ESM+Structure',
            'F': 'Full Model'
        }
        
        # Group by model and aspect
        plot_data = []
        for aspect in ['BPO', 'CCO', 'MFO']:
            for model in main_models:
                model_data = df[(df['model'] == model) & (df['aspect'] == aspect)]
                if not model_data.empty:
                    plot_data.append({
                        'Model': model_names.get(model, model),
                        'Aspect': aspect,
                        'F-max': model_data['best_Fmax_protein'].values[0]
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        x = np.arange(len(main_models))
        width = 0.25
        
        for i, aspect in enumerate(['BPO', 'CCO', 'MFO']):
            aspect_data = plot_df[plot_df['Aspect'] == aspect]
            values = []
            for model in main_models:
                model_val = aspect_data[aspect_data['Model'] == model_names.get(model, model)]
                if not model_val.empty:
                    values.append(model_val['F-max'].values[0])
                else:
                    values.append(0)
            
            ax.bar(x + i*width, values, width, label=aspect)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('F-max', fontsize=12)
        ax.set_title('Overall Performance of Feature Combinations', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([model_names.get(m, m) for m in main_models], rotation=45, ha='right')
        ax.legend(title='GO Aspect')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'figure1_overall_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_figure2_structure_comparison(self, df: pd.DataFrame):
        """Create structure model comparison figure."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Structure models only
        structure_models = ['C1', 'C2', 'C3', 'C4']
        model_names = {
            'C1': 'Radius + One-Hot',
            'C2': 'Radius + ESM',
            'C3': 'k-NN + One-Hot',
            'C4': 'k-NN + ESM'
        }
        
        # Calculate average F-max across aspects
        structure_data = []
        for model in structure_models:
            model_df = df[df['model'] == model]
            if not model_df.empty:
                avg_fmax = model_df['best_Fmax_protein'].mean()
                std_fmax = model_df['best_Fmax_protein'].std()
                structure_data.append({
                    'Model': model_names[model],
                    'Graph': 'Radius' if 'C1' in model or 'C2' in model else 'k-NN',
                    'Features': 'One-Hot' if 'C1' in model or 'C3' in model else 'ESM',
                    'F-max': avg_fmax,
                    'Std': std_fmax
                })
        
        struct_df = pd.DataFrame(structure_data)
        
        # Create grouped bar chart
        x = np.arange(len(structure_models))
        ax.bar(x, struct_df['F-max'], yerr=struct_df['Std'], capsize=5,
               color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        
        # Add value labels
        for i, (_, row) in enumerate(struct_df.iterrows()):
            ax.text(i, row['F-max'] + row['Std'] + 0.005, f"{row['F-max']:.3f}", 
                   ha='center', va='bottom')
        
        ax.set_xlabel('Structure Model Configuration', fontsize=12)
        ax.set_ylabel('Average F-max', fontsize=12)
        ax.set_title('Comparative Analysis of Structural Representations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(struct_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'figure2_structure_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_figure3_pr_analysis(self, df: pd.DataFrame):
        """Create precision-recall analysis figure."""
        import matplotlib.pyplot as plt
        
        # This would require loading actual prediction files
        # For now, create a placeholder showing AUPR values
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, aspect in enumerate(['BPO', 'CCO', 'MFO']):
            ax = axes[idx]
            
            # Get top models for this aspect
            aspect_df = df[df['aspect'] == aspect].sort_values('best_Fmax_protein', ascending=False).head(3)
            
            # Plot bars for AUPR
            models = aspect_df['model'].tolist()
            auprs = aspect_df['best_macro_AP'].tolist()
            
            bars = ax.bar(range(len(models)), auprs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylim(0, 1)
            ax.set_xlabel('Model')
            ax.set_ylabel('AUPR')
            ax.set_title(f'{aspect} - Top Models by AUPR')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models)
            
            # Add value labels
            for bar, aupr in zip(bars, auprs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{aupr:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Precision-Recall Analysis of Top Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'figure3_pr_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_figure4_term_analysis(self, df: pd.DataFrame):
        """Create term-centric performance gain analysis."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compare ESM-only baseline vs best multi-modal
        baseline_df = df[df['model'] == 'A']
        multimodal_df = df[df['model'] == 'F']
        
        if not baseline_df.empty and not multimodal_df.empty:
            # Calculate average improvements
            improvements = []
            for aspect in ['BPO', 'CCO', 'MFO']:
                base = baseline_df[baseline_df['aspect'] == aspect]['best_Fmax_protein'].values
                multi = multimodal_df[multimodal_df['aspect'] == aspect]['best_Fmax_protein'].values
                
                if len(base) > 0 and len(multi) > 0:
                    improvement = ((multi[0] - base[0]) / base[0]) * 100
                    improvements.append({
                        'Aspect': aspect,
                        'Baseline F-max': base[0],
                        'Multi-modal F-max': multi[0],
                        'Improvement %': improvement
                    })
            
            imp_df = pd.DataFrame(improvements)
            
            # Create bar chart
            x = np.arange(len(imp_df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, imp_df['Baseline F-max'], width, label='ESM-only')
            bars2 = ax.bar(x + width/2, imp_df['Multi-modal F-max'], width, label='Full Model')
            
            # Add improvement percentages
            for i, (_, row) in enumerate(imp_df.iterrows()):
                y_pos = max(row['Baseline F-max'], row['Multi-modal F-max']) + 0.01
                ax.text(i, y_pos, f"+{row['Improvement %']:.1f}%", 
                       ha='center', va='bottom', fontweight='bold', color='green')
            
            ax.set_xlabel('GO Aspect', fontsize=12)
            ax.set_ylabel('F-max', fontsize=12)
            ax.set_title('Performance Gain: ESM-only vs Full Multi-modal Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(imp_df['Aspect'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'figure4_term_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-modal GO prediction experiment pipeline")
    parser.add_argument('--action', type=str, required=True,
                       choices=['align', 'generate', 'analyze'],
                       help="Action to perform")
    parser.add_argument('--experiments-dir', type=str,
                       default='/SAN/bioinf/PFP/PFP/experiments/multimodal_comparison',
                       help="Experiments directory")
    parser.add_argument('--base-config', type=str,
                       default='/SAN/bioinf/PFP/PFP/configs/base_multimodal.yaml',
                       help="Base configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    (experiments_dir / "logs").mkdir(exist_ok=True)
    
    if args.action == 'align':
        # Step 1: Create aligned datasets
        print("Creating aligned datasets...")
        aligner = DatasetAligner()
        aligned_dir = experiments_dir / "aligned_data"
        alignment_summary = aligner.create_aligned_datasets(aligned_dir)
        
        print("\nAlignment Summary:")
        for aspect, stats in alignment_summary.items():
            print(f"  {aspect}: {stats['coverage']*100:.1f}% coverage")
        
    elif args.action == 'generate':
        # Step 2: Generate experiment configurations
        print("Generating experiment configurations...")
        
        # First ensure we have aligned data
        aligned_dir = experiments_dir / "aligned_data"
        if not aligned_dir.exists():
            print("ERROR: Aligned data not found. Run with --action align first.")
            return
        
        generator = ExperimentGenerator(aligned_dir, args.base_config)
        experiments = generator.generate_all_experiments()
        
        print(f"Generated {len(experiments)} experiment configurations")
        
        # Create config files
        generator.create_config_files(experiments, experiments_dir)
        
        # Create submission scripts
        runner = ExperimentRunner(experiments_dir)
        runner.create_submission_scripts(experiments)
        
        print(f"\nExperiment setup complete!")
        print(f"Configs created in: {experiments_dir}/configs/")
        print(f"Scripts created in: {experiments_dir}/scripts/")
        print(f"\nTo submit experiments: {experiments_dir}/scripts/submit_all.sh")
        print(f"To monitor progress: {experiments_dir}/scripts/monitor.sh")
        
    elif args.action == 'analyze':
        # Step 3: Analyze results
        print("Analyzing experimental results...")
        
        results_dir = experiments_dir / "results"
        if not results_dir.exists():
            print("ERROR: No results directory found.")
            return
        
        analyzer = ResultsAnalyzer(results_dir)
        
        # Collect all results
        df = analyzer.collect_all_results()
        
        if df.empty:
            print("No completed experiments found.")
            return
        
        print(f"Found {len(df)} completed experiments")
        
        # Create analysis report
        analyzer.create_analysis_report(df)
        
        # Create visualizations
        analyzer.create_visualizations(df)
        
        print(f"\nAnalysis complete!")
        print(f"Report saved to: {analyzer.analysis_dir}/experiment_report.md")
        print(f"Figures saved to: {analyzer.analysis_dir}/")


if __name__ == "__main__":
    main()