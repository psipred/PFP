# Comprehensive Gated Fusion Ablation Study

## Executive Summary

### Performance Gains by Component

| Component | BPO Gain | CCO Gain | MFO Gain | Average |
|-----------|----------|----------|----------|----------|
| Simple Concatenation | +4.7% | +6.9% | +9.1% | +6.9% |
| Feature Transformation | +5.7% | +8.4% | +11.6% | +8.6% |
| Basic Gating | +5.5% | +8.0% | +10.5% | +8.0% |
| Cross-Modal Gating | +5.3% | +8.0% | +10.4% | +7.9% |
| Full Model (with residuals & processors) | +4.2% | +7.9% | +9.1% | +7.1% |

### Modality Importance by GO Aspect

| Aspect | Text F-max | ProtT5 F-max | ESM F-max | Avg Text Gate | Avg ProtT5 Gate | Dominant |
|--------|------------|--------------|-----------|---------------|-----------------|----------|
| BPO | 0.4683 | 0.4743 | 0.4645 | 0.544 | 0.490 | Text |
| CCO | 0.6558 | 0.7078 | 0.7014 | 0.563 | 0.500 | Text |
| MFO | 0.5988 | 0.6574 | 0.6498 | 0.577 | 0.512 | Text |

### Key Findings


1. **Largest performance gain**: 2_transformed_concat (avg 8.6% improvement)

2. **Cross-modal gating benefit**: -0.1% additional gain

3. **Modality preferences**: BPO: Balanced, CCO: Balanced, MFO: Balanced

### Visualizations

See generated plots in the output directory:
- `ablation_performance.png`: Performance progression
- `gate_distributions.png`: Gate value distributions
- `modality_importance.png`: Modality importance by aspect
