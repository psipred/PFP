#!/usr/bin/env python3
"""
Experiment manager for structure-based GO prediction.
Generates configurations and manages experiment execution.

"""

import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import itertools


class ExperimentManager:
    """Manage experiments for structure-based GO prediction."""
    
    def __init__(self, base_config_path: str, pipeline_dir: str):
        self.base_config_path = Path(base_config_path)
        self.pipeline_dir = Path(pipeline_dir)
        self.configs_dir = self.pipeline_dir / "configs"
        self.scripts_dir = self.pipeline_dir / "scripts"
        
        # Create directories
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_experiment_configs(self):
        """Generate all experiment configurations."""
        experiments = []
        
        # 1. Graph construction experiments
        graph_configs = [
            {'name': 'graph_knn_k5', 'params': {'graph': {'type': 'knn', 'k': 5}}},
            {'name': 'graph_knn_k10', 'params': {'graph': {'type': 'knn', 'k': 10}}},
            {'name': 'graph_knn_k15', 'params': {'graph': {'type': 'knn', 'k': 15}}},
            {'name': 'graph_knn_k20', 'params': {'graph': {'type': 'knn', 'k': 20}}},
            {'name': 'graph_radius_r8', 'params': {'graph': {'type': 'radius', 'radius': 8.0}}},
            {'name': 'graph_radius_r10', 'params': {'graph': {'type': 'radius', 'radius': 10.0}}},
            {'name': 'graph_radius_r12', 'params': {'graph': {'type': 'radius', 'radius': 12.0}}},
        ]
        
        # 2. Model architecture experiments
        model_configs = [
            {'name': 'model_layers_2', 'params': {'model': {'n_layers': 2}}},
            {'name': 'model_layers_4', 'params': {'model': {'n_layers': 4}}},
            {'name': 'model_layers_6', 'params': {'model': {'n_layers': 6}}},
            {'name': 'model_hidden_128', 'params': {'model': {'hidden_dim': 128, 'embedding_dim': 256}}},
            {'name': 'model_hidden_256', 'params': {'model': {'hidden_dim': 256, 'embedding_dim': 512}}},
            {'name': 'model_hidden_512', 'params': {'model': {'hidden_dim': 512, 'embedding_dim': 1024}}},
        ]
        
        # # 3. Training experiments
        # training_configs = [
        #     {'name': 'train_lr_5e4', 'params': {'optim': {'lr': 5e-4}}},
        #     {'name': 'train_lr_1e3', 'params': {'optim': {'lr': 1e-3}}},
        #     {'name': 'train_batch_8', 'params': {'dataset': {'batch_size': 8}}},
        #     {'name': 'train_batch_32', 'params': {'dataset': {'batch_size': 32}}},
        # ]
        
        # 4. Best configurations for each GO aspect
        aspect_configs = []
        for aspect in ['BPO', 'CCO', 'MFO']:
            output_dim = {'BPO': 1302, 'CCO': 453, 'MFO': 483}[aspect]
            
            # Best overall config
            aspect_configs.append({
                'name': f'{aspect.lower()}_best_v1',
                'params': {
                    'aspect': aspect,
                    'graph': {'type': 'knn', 'k': 10},
                    'model': {'output_dim': output_dim, 'n_layers': 4, 'hidden_dim': 256},
                    'dataset': {
                        'train_names': f'${{data_dir}}/{aspect}/{aspect}_train_names_fold${{fold}}.npy',
                        'train_labels': f'${{data_dir}}/{aspect}/{aspect}_train_labels_fold${{fold}}.npz',
                        'valid_names': f'${{data_dir}}/{aspect}/{aspect}_valid_names_fold${{fold}}.npy',
                        'valid_labels': f'${{data_dir}}/{aspect}/{aspect}_valid_labels_fold${{fold}}.npz'
                    }
                }
            })
        
        # Combine all experiments
        all_configs = graph_configs + model_configs
        
        # Generate configuration files
        for config_spec in all_configs:
            config = self._create_config_variant(config_spec['params'])
            config['log']['out_dir'] = f"/SAN/bioinf/PFP/PFP/structure/experiments/{config_spec['name']}"
            
            # Save config
            config_path = self.configs_dir / f"{config_spec['name']}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            experiments.append({
                'name': config_spec['name'],
                'config_path': str(config_path),
                'type': config_spec['name'].split('_')[0]
            })
        
        return experiments
    
    def _create_config_variant(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Create a config variant by updating base config."""
        import copy
        config = copy.deepcopy(self.base_config)
        
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        return update_dict(config, updates)
    
    def create_submission_scripts(self, experiments: List[Dict]):
        """Create qsub submission scripts."""
      
        # Individual experiment scripts
        for exp in experiments:
# $ -l h=!hoots-207-1.local  
            script_content = f"""#!/bin/bash
#$ -N {exp['name']}
#$ -l h_vmem=50G
#$ -l tmem=50G
#$ -l h_rt=8:0:0
#$ -j y
#$ -o /SAN/bioinf/PFP/PFP/structure/experiments/logs/{exp['name']}.log
#$ -wd /SAN/bioinf/PFP/PFP/structure
#$ -l gpu=true
#$ -l h=!walter*

#$ -pe gpu 1

echo "Starting experiment: {exp['name']}"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Activate environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run training
cd /SAN/bioinf/PFP/PFP/structure/experiment_pipeline
python train_with_metrics.py --config-path=./configs --config-name={exp['name']}

echo "Experiment completed: $(date)"
"""
            
            script_path = self.scripts_dir / f"{exp['name']}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)
        
        # Master submission script
        master_script = """#!/bin/bash
# Master script to submit experiments

SCRIPT_DIR="/SAN/bioinf/PFP/PFP/structure/experiment_pipeline/scripts"

# Priority experiments (run first)
PRIORITY_EXPS=(
    "graph_knn_k5"
    "graph_knn_k10"
    "graph_knn_k15"
    "graph_knn_k20"
    "graph_radius_r8"
    "graph_radius_r10"
    "graph_radius_r12"
    "model_layers_2"
    "model_layers_4"
    "model_layers_6"
    "model_hidden_256"
    "model_hidden_512"


)


echo "Submitting priority experiments..."
for exp in "${PRIORITY_EXPS[@]}"; do
    if [ -f "$SCRIPT_DIR/${exp}.sh" ]; then
        echo "Submitting: $exp"
        qsub "$SCRIPT_DIR/${exp}.sh"
        sleep 1
    fi
done

echo "Priority experiments submitted."
echo "To submit remaining experiments, use:"
echo "  for script in $SCRIPT_DIR/*.sh; do qsub \\$script; sleep 60; done"
"""
        
        master_path = self.scripts_dir / "submit_all.sh"
        with open(master_path, 'w') as f:
            f.write(master_script)
        master_path.chmod(0o755)
        
        # Create monitoring script
        monitor_script = """#!/bin/bash
echo "Structure GO Prediction Experiments Status"
echo "=========================================="

# Running jobs
echo -e "\\nRunning:"
qstat | grep -E "graph_|model_|train_|bpo_|cco_|mfo_" | awk '{print $1, $3, $5, $8}'

# Completed experiments
echo -e "\\nCompleted:"
find /SAN/bioinf/PFP/PFP/structure/experiments -name "training_summary.json" | while read f; do
    exp=$(basename $(dirname $f))
    echo "  - $exp"
done

# Failed experiments
echo -e "\\nFailed (check logs):"
grep -l "Error\\|Traceback" /SAN/bioinf/PFP/PFP/structure/experiments/logs/*.log 2>/dev/null | while read f; do
    echo "  - $(basename $f .log)"
done
"""
        
        monitor_path = self.scripts_dir / "monitor.sh"
        with open(monitor_path, 'w') as f:
            f.write(monitor_script)
        monitor_path.chmod(0o755)
    
    def collect_results(self, experiments_dir: str = "/SAN/bioinf/PFP/PFP/structure/experiments") -> pd.DataFrame:
        """Collect results from all completed experiments."""
        results = []
        exp_dir = Path(experiments_dir)
        
        for exp_path in exp_dir.iterdir():
            if not exp_path.is_dir():
                continue
            
            summary_file = exp_path / "training_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                result = {
                    'experiment': exp_path.name,
                    'training_hours': summary.get('training_time', 0) / 3600,
                    'final_epoch': summary.get('final_epoch', 0)
                }
                
                # Add configuration details
                config = summary.get('config', {})
                result['graph_type'] = config.get('graph', {}).get('type', 'unknown')
                result['graph_k'] = config.get('graph', {}).get('k', None)
                result['graph_radius'] = config.get('graph', {}).get('radius', None)
                result['n_layers'] = config.get('model', {}).get('n_layers', None)
                result['hidden_dim'] = config.get('model', {}).get('hidden_dim', None)
                result['learning_rate'] = config.get('optim', {}).get('lr', None)
                result['batch_size'] = config.get('dataset', {}).get('batch_size', None)
                
                # Add metrics
                if 'best_metrics' in summary:
                    for metric, value in summary['best_metrics'].items():
                        if metric not in ['epoch', 'monitor_value']:
                            result[f'best_{metric}'] = value
                
                results.append(result)
        
        return pd.DataFrame(results)


def main():
    """Main function to setup experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage structure GO prediction experiments")
    parser.add_argument('action', choices=['setup', 'collect', 'status'],
                       help="Action to perform")
    parser.add_argument('--base-config', type=str,
                       default="/SAN/bioinf/PFP/PFP/structure/configs/structure_config.yaml",
                       help="Base configuration file")
    parser.add_argument('--pipeline-dir', type=str,
                       default="/SAN/bioinf/PFP/PFP/structure/experiment_pipeline",
                       help="Pipeline directory")
    
    args = parser.parse_args()
    
    manager = ExperimentManager(args.base_config, args.pipeline_dir)
    
    if args.action == 'setup':
        print("Setting up experiments...")
        experiments = manager.generate_experiment_configs()
        manager.create_submission_scripts(experiments)
        
        print(f"\nGenerated {len(experiments)} experiment configurations:")
        for exp in experiments[:10]:  # Show first 10
            print(f"  - {exp['name']} ({exp['type']})")
        
        print(f"\nConfiguration files: {manager.configs_dir}")
        print(f"Submission scripts: {manager.scripts_dir}")
        print(f"\nTo submit experiments: {manager.scripts_dir}/submit_all.sh")
        print(f"To monitor: {manager.scripts_dir}/monitor.sh")
        
    elif args.action == 'collect':
        print("Collecting results...")
        results_df = manager.collect_results()
        
        if not results_df.empty:
            # Save results
            output_file = manager.pipeline_dir / "all_results.csv"
            results_df.to_csv(output_file, index=False)
            
            # Show summary
            print(f"\nCollected {len(results_df)} experiment results")
            print("\nTop 5 by F-max:")
            if 'best_Fmax_protein' in results_df.columns:
                top_5 = results_df.nlargest(5, 'best_Fmax_protein')[
                    ['experiment', 'best_Fmax_protein', 'best_macro_AP', 'graph_type', 'n_layers']
                ]
                print(top_5.to_string(index=False))
            
            print(f"\nFull results saved to: {output_file}")
        else:
            print("No completed experiments found.")
    
    elif args.action == 'status':
        print("Checking experiment status...")
        import subprocess
        
        # Count experiments
        total_configs = len(list(manager.configs_dir.glob("*.yaml")))
        
        # Count completed
        completed = 0
        experiments_dir = Path("/SAN/bioinf/PFP/PFP/structure/experiments")
        for exp_dir in experiments_dir.iterdir():
            if (exp_dir / "training_summary.json").exists():
                completed += 1
        
        print(f"\nTotal experiments: {total_configs}")
        print(f"Completed: {completed}")
        print(f"Remaining: {total_configs - completed}")
        
        # Show running jobs
        try:
            result = subprocess.run(['qstat'], capture_output=True, text=True)
            running_jobs = [line for line in result.stdout.split('\n') 
                          if any(x in line for x in ['graph_', 'model_', 'train_', 'bpo_', 'cco_', 'mfo_'])]
            print(f"\nRunning jobs: {len(running_jobs)}")
            for job in running_jobs[:5]:  # Show first 5
                print(f"  {job}")
        except:
            print("\nCould not check running jobs (qstat not available)")


if __name__ == "__main__":
    main()