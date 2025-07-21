# Ablation Study Report for BPO

## Performance Comparison

| Model | Best F-max | Improvement | Key Insights |
|-------|------------|-------------|-------------|
| 0_baseline_esm | 0.4752 | +0.67% | ESM only baseline |
| 0_baseline_prott5 | 0.4810 | +1.89% | ProtT5 only baseline |
| 0_baseline_text | 0.4720 | +0.00% | Text only baseline |
| 1_simple_concat | 0.4961 | +5.10% | Text/ProtT5 magnitude ratio: 23.75 |
| 2_transformed_concat | 0.5013 | +6.20% | Feature similarity: 0.301 |
| 3_simple_gated | 0.4954 | +4.94% | Balanced gating |
| 4_crossmodal_gated | 0.4937 | +4.59% | Balanced gating |
| 5_full_gated | 0.4911 | +4.03% | Balanced gating |

## Interpretability Analysis


### 4_crossmodal_gated

- Average text gate: 0.461 (±0.037)
- Average ProtT5 gate: 0.438 (±0.034)

### 5_full_gated

- Average text gate: 0.525 (±0.024)
- Average ProtT5 gate: 0.451 (±0.038)
