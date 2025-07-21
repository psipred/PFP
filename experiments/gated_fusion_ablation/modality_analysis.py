#!/usr/bin/env python3
"""
Modality Importance Analysis for Gated Fusion Models
Provides detailed insights into which modality drives predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.stats import spearmanr, pearsonr
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class ModalityImportanceAnalyzer:
    """Analyze which modality is decisive for predictions."""
    
    def __init__(self, model_path: str, data_dir: str, aspect: str):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.aspect = aspect
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Expected logits per GO aspect
        self.aspect_output_dims = {
            'BPO': 3992,
            'CCO': 551,
            'MFO': 677
        }
        # Fallback to 677 if the aspect is unknown
        self.expected_output_dim = self.aspect_output_dims.get(self.aspect, 677)

        # Load GO terms
        with open(self.data_dir / f"{aspect}_go_terms.json", 'r') as f:
            self.go_terms = json.load(f)

        self.results = {
            'per_protein': [],
            'per_go_term': defaultdict(list),
            'global_stats': {}
        }
        
    def load_model(self, model_class):
        """Load trained model."""
        # 3) Build a fresh model instance with the aspect‑specific output_dim.
        model = model_class(output_dim=self.expected_output_dim).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def analyze_sample(self, model, text_features, prott5_features, labels, protein_name):
        """Analyze modality importance for a single sample."""
        with torch.no_grad():   
            # Get predictions with both modalities
            full_pred, interp_data = model(text_features, prott5_features)
            full_pred_sig = torch.sigmoid(full_pred)
            
            # Get predictions with zeroed modalities
            # Zero out text
            zero_text = torch.zeros_like(text_features)
            prott5_only_pred, _ = model(zero_text, prott5_features)
            prott5_only_sig = torch.sigmoid(prott5_only_pred)
            
            # Zero out ProtT5
            zero_prott5 = torch.zeros_like(prott5_features)
            text_only_pred, _ = model(text_features, zero_prott5)
            text_only_sig = torch.sigmoid(text_only_pred)
            
            # Calculate importance scores
            text_importance = (full_pred_sig - prott5_only_sig).abs()
            prott5_importance = (full_pred_sig - text_only_sig).abs()
            
            # Synergy score (when combined is better than either alone)
            synergy = full_pred_sig - torch.max(text_only_sig, prott5_only_sig)
            
            # Get gate values if available
            text_gate = interp_data.get('text_gate_mean', None)
            prott5_gate = interp_data.get('prott5_gate_mean', None)
            
            # Per-GO term analysis
            go_term_results = []
            for i, go_term in enumerate(self.go_terms):
                result = {
                    'go_term': go_term,
                    'label': labels[0, i].item(),
                    'prediction': full_pred_sig[0, i].item(),
                    'text_only_pred': text_only_sig[0, i].item(),
                    'prott5_only_pred': prott5_only_sig[0, i].item(),
                    'text_importance': text_importance[0, i].item(),
                    'prott5_importance': prott5_importance[0, i].item(),
                    'synergy': synergy[0, i].item(),
                    'dominant_modality': 'text' if text_importance[0, i] > prott5_importance[0, i] else 'prott5'
                }
                go_term_results.append(result)
                
                # Store for per-GO analysis
                self.results['per_go_term'][go_term].append(result)
            
            # Protein-level summary
            protein_result = {
                'protein': protein_name,
                'avg_text_importance': text_importance.mean().item(),
                'avg_prott5_importance': prott5_importance.mean().item(),
                'avg_synergy': synergy.mean().item(),
                'text_gate': text_gate,
                'prott5_gate': prott5_gate,
                'positive_labels': labels.sum().item(),
                'text_dominant_count': (text_importance > prott5_importance).sum().item(),
                'prott5_dominant_count': (prott5_importance > text_importance).sum().item(),
                'go_term_results': go_term_results
            }
            
            self.results['per_protein'].append(protein_result)
            
            return protein_result
    
    def analyze_dataset(self, model, dataset, num_samples=None):
        """Analyze modality importance across dataset."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False
        )
        
        if num_samples:
            loader = list(loader)[:num_samples]
        
        for idx, (names, features, labels) in enumerate(tqdm(loader, desc="Analyzing samples")):
            text_feat = features['text'].to(self.device)
            prott5_feat = features['prott5'].to(self.device)
            labels = labels.to(self.device)
            
            self.analyze_sample(model, text_feat, prott5_feat, labels, names[0])
    
    def compute_global_statistics(self):
        """Compute global statistics from analyzed samples."""
        
        # Protein-level statistics
        protein_df = pd.DataFrame([
            {k: v for k, v in p.items() if k != 'go_term_results'}
            for p in self.results['per_protein']
        ])
        
        self.results['global_stats']['protein_level'] = {
            'text_dominant_proteins': (protein_df['text_dominant_count'] > 
                                     protein_df['prott5_dominant_count']).sum(),
            'prott5_dominant_proteins': (protein_df['prott5_dominant_count'] > 
                                       protein_df['text_dominant_count']).sum(),
            'balanced_proteins': (protein_df['text_dominant_count'] == 
                                protein_df['prott5_dominant_count']).sum(),
            'avg_text_importance': protein_df['avg_text_importance'].mean(),
            'avg_prott5_importance': protein_df['avg_prott5_importance'].mean(),
            'avg_synergy': protein_df['avg_synergy'].mean(),
            'synergy_positive_ratio': (protein_df['avg_synergy'] > 0).mean()
        }
        
        # GO term-level statistics
        go_stats = []
        for go_term, results in self.results['per_go_term'].items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            # Only analyze positive labels
            pos_df = df[df['label'] == 1]
            
            if len(pos_df) > 0:
                go_stats.append({
                    'go_term': go_term,
                    'n_positive': len(pos_df),
                    'avg_text_importance': pos_df['text_importance'].mean(),
                    'avg_prott5_importance': pos_df['prott5_importance'].mean(),
                    'text_dominant_ratio': (pos_df['dominant_modality'] == 'text').mean(),
                    'avg_synergy': pos_df['synergy'].mean(),
                    'text_only_avg_pred': pos_df['text_only_pred'].mean(),
                    'prott5_only_avg_pred': pos_df['prott5_only_pred'].mean(),
                    'combined_avg_pred': pos_df['prediction'].mean()
                })
        
        go_stats_df = pd.DataFrame(go_stats)
        
        # Identify GO terms with strong modality preferences
        text_preferred_terms = go_stats_df[go_stats_df['text_dominant_ratio'] > 0.7]
        prott5_preferred_terms = go_stats_df[go_stats_df['text_dominant_ratio'] < 0.3]
        balanced_terms = go_stats_df[
            (go_stats_df['text_dominant_ratio'] >= 0.3) & 
            (go_stats_df['text_dominant_ratio'] <= 0.7)
        ]
        
        self.results['global_stats']['go_term_level'] = {
            'text_preferred_terms': text_preferred_terms['go_term'].tolist(),
            'prott5_preferred_terms': prott5_preferred_terms['go_term'].tolist(),
            'balanced_terms': balanced_terms['go_term'].tolist(),
            'n_text_preferred': len(text_preferred_terms),
            'n_prott5_preferred': len(prott5_preferred_terms),
            'n_balanced': len(balanced_terms)
        }
        
        # Gate correlation analysis (if gates are available)
        if 'text_gate' in protein_df.columns and protein_df['text_gate'].notna().any():
            gate_corr = pearsonr(
                protein_df['avg_text_importance'].fillna(0),
                protein_df['text_gate'].fillna(0)
            )
            self.results['global_stats']['gate_correlations'] = {
                'text_importance_gate_corr': gate_corr[0],
                'text_importance_gate_pval': gate_corr[1]
            }
        
        return go_stats_df
    
    def visualize_results(self, output_dir: Path):
        """Create comprehensive visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Modality importance distribution
        self._plot_importance_distribution(output_dir)
        
        # 2. GO term modality preferences
        self._plot_go_term_preferences(output_dir)
        
        # 3. Synergy analysis
        self._plot_synergy_analysis(output_dir)
        
        # 4. Sample-level analysis
        self._plot_sample_analysis(output_dir)
        
    def _plot_importance_distribution(self, output_dir: Path):
        """Plot distribution of modality importance scores."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract importance scores
        text_imp = []
        prott5_imp = []
        synergy = []
        
        for protein in self.results['per_protein']:
            text_imp.append(protein['avg_text_importance'])
            prott5_imp.append(protein['avg_prott5_importance'])
            synergy.append(protein['avg_synergy'])
        
        # Plot 1: Histogram of importance scores
        ax = axes[0, 0]
        ax.hist(text_imp, bins=30, alpha=0.5, label='Text', density=True)
        ax.hist(prott5_imp, bins=30, alpha=0.5, label='ProtT5', density=True)
        ax.set_xlabel('Average Importance Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Modality Importance')
        ax.legend()
        
        # Plot 2: Scatter plot of text vs prott5 importance
        ax = axes[0, 1]
        ax.scatter(text_imp, prott5_imp, alpha=0.5)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('Text Importance')
        ax.set_ylabel('ProtT5 Importance')
        ax.set_title('Text vs ProtT5 Importance')
        
        # Plot 3: Synergy distribution
        ax = axes[1, 0]
        ax.hist(synergy, bins=30, alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Synergy Score')
        ax.set_ylabel('Count')
        ax.set_title(f'Synergy Distribution (Positive: {(np.array(synergy) > 0).mean():.1%})')
        
        # Plot 4: Modality dominance pie chart
        ax = axes[1, 1]
        stats = self.results['global_stats']['protein_level']
        sizes = [
            stats['text_dominant_proteins'],
            stats['prott5_dominant_proteins'],
            stats['balanced_proteins']
        ]
        labels = ['Text Dominant', 'ProtT5 Dominant', 'Balanced']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.set_title('Protein-level Modality Dominance')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'modality_importance_distribution.png', dpi=150)
        plt.close()
    
    def _plot_go_term_preferences(self, output_dir: Path):
        """Plot GO term-specific modality preferences."""
        go_stats = []
        for go_term, results in self.results['per_go_term'].items():
            if not results:
                continue
            
            df = pd.DataFrame(results)
            pos_df = df[df['label'] == 1]
            
            if len(pos_df) >= 5:  # Only include terms with enough positive samples
                go_stats.append({
                    'go_term': go_term,
                    'text_importance': pos_df['text_importance'].mean(),
                    'prott5_importance': pos_df['prott5_importance'].mean(),
                    'n_samples': len(pos_df)
                })
        
        if not go_stats:
            return
            
        go_df = pd.DataFrame(go_stats)
        go_df['dominant'] = go_df.apply(
            lambda x: 'text' if x['text_importance'] > x['prott5_importance'] else 'prott5',
            axis=1
        )
        
        # Sort by importance difference
        go_df['importance_diff'] = go_df['text_importance'] - go_df['prott5_importance']
        go_df = go_df.sort_values('importance_diff')
        
        # Select top/bottom terms
        n_show = min(20, len(go_df))
        top_text = go_df.head(n_show // 2)
        top_prott5 = go_df.tail(n_show // 2)
        plot_df = pd.concat([top_text, top_prott5])
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(plot_df))
        colors = ['#e74c3c' if d == 'text' else '#3498db' for d in plot_df['dominant']]
        
        ax.barh(y_pos, plot_df['importance_diff'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['go_term'], fontsize=8)
        ax.set_xlabel('Text Importance - ProtT5 Importance')
        ax.set_title('GO Terms with Strongest Modality Preferences')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.7, label='Text Preferred'),
            Patch(facecolor='#3498db', alpha=0.7, label='ProtT5 Preferred')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'go_term_modality_preferences.png', dpi=150)
        plt.close()
    
    def _plot_synergy_analysis(self, output_dir: Path):
        """Analyze when modalities work together vs independently."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect data
        synergy_data = []
        for protein in self.results['per_protein']:
            for go_result in protein['go_term_results']:
                if go_result['label'] == 1:  # Only positive labels
                    synergy_data.append({
                        'text_pred': go_result['text_only_pred'],
                        'prott5_pred': go_result['prott5_only_pred'],
                        'combined_pred': go_result['prediction'],
                        'synergy': go_result['synergy'],
                        'max_single': max(go_result['text_only_pred'], 
                                        go_result['prott5_only_pred'])
                    })
        
        synergy_df = pd.DataFrame(synergy_data)
        
        # Plot 1: Synergy vs prediction confidence
        ax = axes[0, 0]
        scatter = ax.scatter(synergy_df['combined_pred'], synergy_df['synergy'], 
                           c=synergy_df['synergy'] > 0, cmap='RdYlGn', alpha=0.5)
        ax.set_xlabel('Combined Prediction Score')
        ax.set_ylabel('Synergy Score')
        ax.set_title('Synergy vs Prediction Confidence')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # Plot 2: When does synergy occur?
        ax = axes[0, 1]
        synergy_positive = synergy_df[synergy_df['synergy'] > 0]
        synergy_negative = synergy_df[synergy_df['synergy'] <= 0]
        
        ax.scatter(synergy_positive['text_pred'], synergy_positive['prott5_pred'], 
                  alpha=0.5, label=f'Positive Synergy (n={len(synergy_positive)})', 
                  color='green')
        ax.scatter(synergy_negative['text_pred'], synergy_negative['prott5_pred'], 
                  alpha=0.5, label=f'No/Negative Synergy (n={len(synergy_negative)})', 
                  color='red')
        ax.set_xlabel('Text-only Prediction')
        ax.set_ylabel('ProtT5-only Prediction')
        ax.set_title('Synergy Patterns')
        ax.legend()
        
        # Plot 3: Improvement over best single modality
        ax = axes[1, 0]
        improvement = synergy_df['combined_pred'] - synergy_df['max_single']
        ax.hist(improvement, bins=30, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Improvement over Best Single Modality')
        ax.set_ylabel('Count')
        ax.set_title(f'Fusion Improvement (Positive: {(improvement > 0).mean():.1%})')
        
        # Plot 4: Synergy by prediction range
        ax = axes[1, 1]
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        synergy_df['pred_bin'] = pd.cut(synergy_df['combined_pred'], bins)
        synergy_by_bin = synergy_df.groupby('pred_bin')['synergy'].agg(['mean', 'std'])
        
        x = range(len(synergy_by_bin))
        ax.bar(x, synergy_by_bin['mean'], yerr=synergy_by_bin['std'], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b.left:.1f}-{b.right:.1f}' for b in synergy_by_bin.index])
        ax.set_xlabel('Prediction Score Range')
        ax.set_ylabel('Average Synergy')
        ax.set_title('Synergy by Prediction Confidence')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'synergy_analysis.png', dpi=150)
        plt.close()
    
    def _plot_sample_analysis(self, output_dir: Path):
        """Detailed analysis of specific samples."""
        # Select interesting samples
        samples = []
        
        # Find samples with high text importance
        text_dominant = sorted(self.results['per_protein'], 
                             key=lambda x: x['avg_text_importance'], 
                             reverse=True)[:2]
        
        # Find samples with high prott5 importance
        prott5_dominant = sorted(self.results['per_protein'], 
                               key=lambda x: x['avg_prott5_importance'], 
                               reverse=True)[:2]
        
        # Find samples with high synergy
        high_synergy = sorted(self.results['per_protein'], 
                            key=lambda x: x['avg_synergy'], 
                            reverse=True)[:2]
        
        samples.extend(text_dominant + prott5_dominant + high_synergy)
        
        # Create detailed plots for selected samples
        n_samples = min(6, len(samples))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, sample in enumerate(samples[:n_samples]):
            ax = axes[idx]
            
            # Extract GO term results
            go_results = pd.DataFrame(sample['go_term_results'])
            positive_gos = go_results[go_results['label'] == 1]
            
            if len(positive_gos) == 0:
                continue
            
            # Sort by prediction score
            positive_gos = positive_gos.sort_values('prediction', ascending=False).head(10)
            
            # Create grouped bar plot
            x = np.arange(len(positive_gos))
            width = 0.25
            
            ax.bar(x - width, positive_gos['text_only_pred'], width, label='Text only', alpha=0.7)
            ax.bar(x, positive_gos['prott5_only_pred'], width, label='ProtT5 only', alpha=0.7)
            ax.bar(x + width, positive_gos['prediction'], width, label='Combined', alpha=0.7)
            
            ax.set_xlabel('GO Terms (top 10)', fontsize=8)
            ax.set_ylabel('Prediction Score', fontsize=8)
            ax.set_title(f"{sample['protein']}\n"
                        f"Text imp: {sample['avg_text_importance']:.2f}, "
                        f"ProtT5 imp: {sample['avg_prott5_importance']:.2f}", 
                        fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(range(1, len(positive_gos) + 1), fontsize=6)
            ax.legend(fontsize=6)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_level_analysis.png', dpi=150)
        plt.close()
    
    def generate_report(self, output_path: Path):
        """Generate comprehensive markdown report."""
        with open(output_path, 'w') as f:
            f.write("# Modality Importance Analysis Report\n\n")
            f.write(f"**Aspect**: {self.aspect}\n")
            f.write(f"**Number of proteins analyzed**: {len(self.results['per_protein'])}\n\n")
            
            # Global statistics
            f.write("## Global Statistics\n\n")
            
            protein_stats = self.results['global_stats']['protein_level']
            f.write("### Protein-level Modality Dominance\n")
            f.write(f"- Text dominant: {protein_stats['text_dominant_proteins']} proteins\n")
            f.write(f"- ProtT5 dominant: {protein_stats['prott5_dominant_proteins']} proteins\n")
            f.write(f"- Balanced: {protein_stats['balanced_proteins']} proteins\n")
            f.write(f"- Average text importance: {protein_stats['avg_text_importance']:.3f}\n")
            f.write(f"- Average ProtT5 importance: {protein_stats['avg_prott5_importance']:.3f}\n")
            f.write(f"- Average synergy: {protein_stats['avg_synergy']:.3f}\n")
            f.write(f"- Positive synergy ratio: {protein_stats['synergy_positive_ratio']:.1%}\n\n")
            
            # GO term statistics
            go_stats = self.results['global_stats']['go_term_level']
            f.write("### GO Term-level Modality Preferences\n")
            f.write(f"- Text-preferred terms: {go_stats['n_text_preferred']}\n")
            f.write(f"- ProtT5-preferred terms: {go_stats['n_prott5_preferred']}\n")
            f.write(f"- Balanced terms: {go_stats['n_balanced']}\n\n")
            
            # Top text-preferred GO terms
            if go_stats['text_preferred_terms']:
                f.write("#### Top Text-Preferred GO Terms\n")
                for term in go_stats['text_preferred_terms'][:10]:
                    f.write(f"- {term}\n")
                f.write("\n")
            
            # Top ProtT5-preferred GO terms
            if go_stats['prott5_preferred_terms']:
                f.write("#### Top ProtT5-Preferred GO Terms\n")
                for term in go_stats['prott5_preferred_terms'][:10]:
                    f.write(f"- {term}\n")
                f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            
            # Insight 1: Overall modality preference
            if protein_stats['avg_text_importance'] > protein_stats['avg_prott5_importance'] * 1.2:
                f.write("1. **Text features are generally more important** for this aspect, "
                       f"with {protein_stats['avg_text_importance']/protein_stats['avg_prott5_importance']:.1f}x "
                       "higher average importance.\n")
            elif protein_stats['avg_prott5_importance'] > protein_stats['avg_text_importance'] * 1.2:
                f.write("1. **ProtT5 features are generally more important** for this aspect, "
                       f"with {protein_stats['avg_prott5_importance']/protein_stats['avg_text_importance']:.1f}x "
                       "higher average importance.\n")
            else:
                f.write("1. **Both modalities contribute equally** on average, suggesting "
                       "complementary information.\n")
            
            # Insight 2: Synergy
            if protein_stats['synergy_positive_ratio'] > 0.7:
                f.write(f"2. **Strong synergy between modalities**: {protein_stats['synergy_positive_ratio']:.0%} "
                       "of proteins benefit from fusion.\n")
            elif protein_stats['synergy_positive_ratio'] < 0.3:
                f.write(f"2. **Limited synergy**: Only {protein_stats['synergy_positive_ratio']:.0%} "
                       "of proteins benefit from fusion.\n")
            else:
                f.write(f"2. **Moderate synergy**: {protein_stats['synergy_positive_ratio']:.0%} "
                       "of proteins show improved predictions with fusion.\n")
            
            # Insight 3: GO term specificity
            f.write(f"3. **GO term modality preferences**: Out of analyzed terms, "
                   f"{go_stats['n_text_preferred']} prefer text, "
                   f"{go_stats['n_prott5_preferred']} prefer ProtT5, and "
                   f"{go_stats['n_balanced']} are balanced.\n")


def run_modality_analysis(model_dir: str, aspect: str, output_dir: str):
    """Run complete modality importance analysis."""
    from ablation_study import FullGatedFusion, CAFA3Dataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ModalityImportanceAnalyzer(
        model_path=f"{model_dir}/5_full_gated_{aspect}_best.pt",
        data_dir="/SAN/bioinf/PFP/PFP/experiments/cafa3_integration/data",
        aspect=aspect
    )
    
    # Load model
    model = analyzer.load_model(FullGatedFusion)
    
    # Load test dataset
    embeddings_dir = {
        'text': '/SAN/bioinf/PFP/embeddings/cafa3/text',
        'prott5': '/SAN/bioinf/PFP/embeddings/cafa3/prott5'
    }
    
    test_dataset = CAFA3Dataset(
        names_file=analyzer.data_dir / f"{aspect}_test_names.npy",
        labels_file=analyzer.data_dir / f"{aspect}_test_labels.npz",
        features=['text', 'prott5'],
        embeddings_dir=embeddings_dir
    )
    
    # Analyze dataset
    logger.info(f"Analyzing {aspect} test set...")
    analyzer.analyze_dataset(model, test_dataset, num_samples=500)  # Analyze 500 samples
    
    # Compute global statistics
    analyzer.compute_global_statistics()
    
    # Generate visualizations
    viz_dir = output_dir / f"{aspect}_visualizations"
    analyzer.visualize_results(viz_dir)
    
    # Generate report
    report_path = output_dir / f"{aspect}_modality_analysis_report.md"
    analyzer.generate_report(report_path)
    
    # Save raw results
    results_path = output_dir / f"{aspect}_modality_analysis_results.json"
    with open(results_path, 'w') as f:
        # Convert results to serializable format
        serializable_results = {
            'global_stats': analyzer.results['global_stats'],
            'n_proteins_analyzed': len(analyzer.results['per_protein'])
        }
        json.dump(serializable_results, f, indent=2, cls=NpEncoder)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return analyzer


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Modality Importance Analysis")
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--aspects', nargs='+', default=['BPO', 'CCO', 'MFO'],
                       help='GO aspects to analyze')
    parser.add_argument('--output-dir', type=str, 
                       default='/SAN/bioinf/PFP/PFP/experiments/gated_fusion_ablation/modality_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    for aspect in args.aspects:
        logger.info(f"\nAnalyzing {aspect}...")
        run_modality_analysis(args.model_dir, aspect, args.output_dir)


if __name__ == "__main__":
    main()