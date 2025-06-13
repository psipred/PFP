#!/usr/bin/env python3
"""
Analysis and visualization tools for structure-based GO prediction experiments.
Location: /SAN/bioinf/PFP/PFP/structure/experiment_pipeline/analyze_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import argparse


class ResultsAnalyzer:
    """Analyze and visualize experiment results."""
    
    def __init__(self, experiments_dir: str = "/SAN/bioinf/PFP/PFP/structure/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.results_df = None
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def load_all_results(self) -> pd.DataFrame:
        """Load results from all experiments."""
        results = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Load summary
            summary_file = exp_dir / "training_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # Load training history
                history_file = exp_dir / "results" / "training_history.csv"
                if history_file.exists():
                    history = pd.read_csv(history_file)
                else:
                    history = None
                
                # Extract experiment info
                exp_name = exp_dir.name
                exp_type = exp_name.split('_')[0]
                
                result = {
                    'experiment': exp_name,
                    'type': exp_type,
                    'training_hours': summary.get('training_time', 0) / 3600,
                    'final_epoch': summary.get('final_epoch', 0),
                    'history': history
                }
                
                # Add config details
                config = summary.get('config', {})
                result.update({
                    'graph_type': config.get('graph', {}).get('type'),
                    'graph_k': config.get('graph', {}).get('k'),
                    'graph_radius': config.get('graph', {}).get('radius'),
                    'n_layers': config.get('model', {}).get('n_layers'),
                    'hidden_dim': config.get('model', {}).get('hidden_dim'),
                    'learning_rate': config.get('optim', {}).get('lr'),
                    'batch_size': config.get('dataset', {}).get('batch_size'),
                    'aspect': config.get('aspect', 'CCO')
                })
                
                # Add best metrics
                if 'best_metrics' in summary:
                    for metric, value in summary['best_metrics'].items():
                        if metric not in ['epoch', 'monitor_value']:
                            result[f'best_{metric}'] = value
                
                results.append(result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def create_summary_report(self, output_dir: str):
        """Create comprehensive summary report with visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.results_df is None:
            self.load_all_results()
        
        # 1. Overall performance comparison
        self._plot_performance_comparison(output_path)
        
        # 2. Hyperparameter analysis
        self._plot_hyperparameter_analysis(output_path)
        
        # 3. Learning curves
        self._plot_learning_curves(output_path)
        
        # 4. Generate text report
        self._generate_text_report(output_path)
        
        print(f"Analysis complete. Results saved to: {output_path}")
    
    def _plot_performance_comparison(self, output_path: Path):
        """Create performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top experiments by F-max
        ax = axes[0, 0]
        top_10 = self.results_df.nlargest(10, 'best_Fmax_protein')
        
        colors = {'graph': 'blue', 'model': 'green', 'train': 'orange', 
                 'bpo': 'red', 'cco': 'purple', 'mfo': 'brown'}
        bar_colors = [colors.get(exp.split('_')[0], 'gray') for exp in top_10['experiment']]
        
        bars = ax.barh(range(len(top_10)), top_10['best_Fmax_protein'], color=bar_colors)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['experiment'])
        ax.set_xlabel('F-max (Protein-centric)')
        ax.set_title('Top 10 Experiments by F-max')
        ax.invert_yaxis()
        
        # Add values
        for i, v in enumerate(top_10['best_Fmax_protein']):
            ax.text(v + 0.002, i, f'{v:.3f}', va='center')
        
        # 2. Metrics comparison
        ax = axes[0, 1]
        metrics = ['best_Fmax_protein', 'best_macro_AP', 'best_coverage', 'best_macro_AUROC']
        metric_names = ['F-max', 'mAP', 'Coverage', 'AUROC']
        
        # Get top 5 experiments
        top_5 = self.results_df.nlargest(5, 'best_Fmax_protein')
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            values = [row.get(m, 0) for m in metrics]
            ax.bar(x + i*width, values, width, label=row['experiment'][:15])
        
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel('Score')
        ax.set_title('Multi-metric Comparison (Top 5)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Graph construction comparison
        ax = axes[1, 0]
        graph_df = self.results_df[self.results_df['type'] == 'graph']
        
        # k-NN results
        knn_df = graph_df[graph_df['graph_type'] == 'knn'].sort_values('graph_k')
        if not knn_df.empty:
            ax.plot(knn_df['graph_k'], knn_df['best_Fmax_protein'], 
                   'o-', label='k-NN', markersize=8, linewidth=2)
        
        # Radius results
        radius_df = graph_df[graph_df['graph_type'] == 'radius'].sort_values('graph_radius')
        if not radius_df.empty:
            ax2 = ax.twiny()
            ax2.plot(radius_df['graph_radius'], radius_df['best_Fmax_protein'],
                    's-', color='orange', label='Radius', markersize=8, linewidth=2)
            ax2.set_xlabel('Radius (Å)', color='orange')
            ax2.tick_params(axis='x', labelcolor='orange')
        
        ax.set_xlabel('k (neighbors)')
        ax.set_ylabel('F-max')
        ax.set_title('Graph Construction Methods')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Training efficiency
        ax = axes[1, 1]
        ax.scatter(self.results_df['training_hours'], 
                  self.results_df['best_Fmax_protein'],
                  s=self.results_df['final_epoch']*2,
                  alpha=0.6)
        
        ax.set_xlabel('Training Time (hours)')
        ax.set_ylabel('Best F-max')
        ax.set_title('Training Efficiency')
        
        # Add trend line
        mask = ~self.results_df[['training_hours', 'best_Fmax_protein']].isna().any(axis=1)
        if mask.sum() > 2:
            z = np.polyfit(self.results_df.loc[mask, 'training_hours'], 
                          self.results_df.loc[mask, 'best_Fmax_protein'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.results_df['training_hours'].min(), 
                                self.results_df['training_hours'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hyperparameter_analysis(self, output_path: Path):
        """Analyze hyperparameter effects."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Define parameters to analyze
        params = [
            ('n_layers', 'Number of Layers'),
            ('hidden_dim', 'Hidden Dimension'),
            ('learning_rate', 'Learning Rate'),
            ('batch_size', 'Batch Size'),
            ('graph_k', 'k (for k-NN)'),
            ('graph_radius', 'Radius (Å)')
        ]
        
        for idx, (param, label) in enumerate(params):
            ax = axes[idx]
            
            # Get data for this parameter
            param_data = self.results_df[[param, 'best_Fmax_protein']].dropna()
            
            if len(param_data) > 0:
                # Group by parameter value
                grouped = param_data.groupby(param)['best_Fmax_protein'].agg(['mean', 'std', 'count'])
                
                # Plot
                x = grouped.index
                y = grouped['mean']
                yerr = grouped['std']
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, markersize=8)
                
                # Add sample sizes
                for i, (x_val, row) in enumerate(grouped.iterrows()):
                    ax.text(x_val, row['mean'] + row['std'] + 0.01, 
                           f"n={row['count']}", ha='center', fontsize=8)
                
                ax.set_xlabel(label)
                ax.set_ylabel('F-max')
                ax.set_title(f'Effect of {label}')
                ax.grid(True, alpha=0.3)
                
                # Log scale for learning rate
                if param == 'learning_rate':
                    ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_path / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self, output_path: Path):
        """Plot learning curves for top experiments."""
        # Get top 6 experiments
        top_exps = self.results_df.nlargest(6, 'best_Fmax_protein')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, exp_row) in enumerate(top_exps.iterrows()):
            ax = axes[idx]
            
            # Get history
            history = exp_row.get('history')
            if history is not None and isinstance(history, pd.DataFrame):
                # Plot loss
                ax2 = ax.twinx()
                ax.plot(history['epoch'], history['train_loss'], 
                       'b-', label='Train Loss', alpha=0.7)
                ax.plot(history['epoch'], history['valid_loss'], 
                       'b--', label='Valid Loss', alpha=0.7)
                ax.set_ylabel('Loss', color='b')
                ax.tick_params(axis='y', labelcolor='b')
                
                # Plot F-max
                if 'valid_Fmax_protein' in history.columns:
                    ax2.plot(history['epoch'], history['valid_Fmax_protein'], 
                            'r-', label='Valid F-max', linewidth=2)
                    ax2.set_ylabel('F-max', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                
                ax.set_xlabel('Epoch')
                ax.set_title(exp_row['experiment'])
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, output_path: Path):
        """Generate comprehensive text report."""
        with open(output_path / 'analysis_report.txt', 'w') as f:
            f.write("STRUCTURE-BASED GO PREDICTION EXPERIMENT ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total experiments analyzed: {len(self.results_df)}\n")
            f.write(f"Experiments by type:\n")
            for exp_type, count in self.results_df['type'].value_counts().items():
                f.write(f"  - {exp_type}: {count}\n")
            
            # Best overall performance
            f.write("\n1. BEST OVERALL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            
            best_idx = self.results_df['best_Fmax_protein'].idxmax()
            best_exp = self.results_df.loc[best_idx]
            
            f.write(f"Experiment: {best_exp['experiment']}\n")
            f.write(f"F-max: {best_exp['best_Fmax_protein']:.4f}\n")
            f.write(f"mAP: {best_exp.get('best_macro_AP', 0):.4f}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Graph: {best_exp['graph_type']} ")
            if best_exp['graph_type'] == 'knn':
                f.write(f"(k={best_exp['graph_k']})\n")
            else:
                f.write(f"(r={best_exp['graph_radius']}Å)\n")
            f.write(f"  - Layers: {best_exp['n_layers']}\n")
            f.write(f"  - Hidden dim: {best_exp['hidden_dim']}\n")
            f.write(f"  - Learning rate: {best_exp['learning_rate']}\n")
            
            # Best by category
            f.write("\n2. BEST BY CATEGORY\n")
            f.write("-" * 40 + "\n")
            
            for exp_type in ['graph', 'model', 'train']:
                type_df = self.results_df[self.results_df['type'] == exp_type]
                if not type_df.empty:
                    best = type_df.loc[type_df['best_Fmax_protein'].idxmax()]
                    f.write(f"\n{exp_type.upper()}:\n")
                    f.write(f"  Experiment: {best['experiment']}\n")
                    f.write(f"  F-max: {best['best_Fmax_protein']:.4f}\n")
            
            # Hyperparameter insights
            f.write("\n3. HYPERPARAMETER INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Graph construction
            graph_df = self.results_df[self.results_df['type'] == 'graph']
            if not graph_df.empty:
                f.write("\nGraph Construction:\n")
                knn_mean = graph_df[graph_df['graph_type'] == 'knn']['best_Fmax_protein'].mean()
                radius_mean = graph_df[graph_df['graph_type'] == 'radius']['best_Fmax_protein'].mean()
                f.write(f"  k-NN average F-max: {knn_mean:.4f}\n")
                f.write(f"  Radius average F-max: {radius_mean:.4f}\n")
                
                # Best k value
                knn_df = graph_df[graph_df['graph_type'] == 'knn']
                if not knn_df.empty:
                    best_k_idx = knn_df['best_Fmax_protein'].idxmax()
                    best_k = knn_df.loc[best_k_idx, 'graph_k']
                    f.write(f"  Best k value: {best_k}\n")
            
            # Model architecture
            f.write("\nModel Architecture:\n")
            for n_layers in sorted(self.results_df['n_layers'].dropna().unique()):
                layer_mean = self.results_df[self.results_df['n_layers'] == n_layers]['best_Fmax_protein'].mean()
                f.write(f"  {n_layers} layers: {layer_mean:.4f} average F-max\n")
            
            # Training efficiency
            f.write("\n4. TRAINING EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            
            f.write(f"Average training time: {self.results_df['training_hours'].mean():.2f} hours\n")
            f.write(f"Average epochs: {self.results_df['final_epoch'].mean():.1f}\n")
            
            # Find most efficient (high performance, low time)
            efficiency = self.results_df['best_Fmax_protein'] / (self.results_df['training_hours'] + 0.1)
            most_efficient_idx = efficiency.idxmax()
            most_efficient = self.results_df.loc[most_efficient_idx]
            
            f.write(f"\nMost efficient experiment: {most_efficient['experiment']}\n")
            f.write(f"  F-max: {most_efficient['best_Fmax_protein']:.4f}\n")
            f.write(f"  Training time: {most_efficient['training_hours']:.2f} hours\n")
            
            # Statistical summary
            f.write("\n5. STATISTICAL SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            metrics = ['best_Fmax_protein', 'best_macro_AP', 'best_coverage']
            for metric in metrics:
                if metric in self.results_df.columns:
                    values = self.results_df[metric].dropna()
                    if len(values) > 0:
                        f.write(f"\n{metric}:\n")
                        f.write(f"  Mean: {values.mean():.4f}\n")
                        f.write(f"  Std: {values.std():.4f}\n")
                        f.write(f"  Min: {values.min():.4f}\n")
                        f.write(f"  Max: {values.max():.4f}\n")
        
        # Save detailed results
        self.results_df.drop(columns=['history'], errors='ignore').to_csv(
            output_path / 'all_results_detailed.csv', index=False
        )
        
        print(f"Report saved to: {output_path / 'analysis_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze structure GO prediction results")
    parser.add_argument('--experiments-dir', type=str,
                       default="/SAN/bioinf/PFP/PFP/structure/experiments",
                       help="Directory containing experiments")
    parser.add_argument('--output-dir', type=str,
                       default="/SAN/bioinf/PFP/PFP/structure/experiment_pipeline/analysis",
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.experiments_dir)
    analyzer.create_summary_report(args.output_dir)


if __name__ == "__main__":
    main()