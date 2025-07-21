# Ablation Study Report for CCO

## Performance Comparison

| Model | Best F-max | Improvement | Key Insights |
|-------|------------|-------------|-------------|
| 0_baseline_esm | 0.7048 | +7.52% | ESM only baseline |
| 0_baseline_prott5 | 0.7082 | +8.04% | ProtT5 only baseline |
| 0_baseline_text | 0.6555 | +0.00% | Text only baseline |
| 1_simple_concat | 0.7008 | +6.91% | Text/ProtT5 magnitude ratio: 23.62 |
| 2_transformed_concat | 0.7098 | +8.29% | Feature similarity: 0.320 |
| 3_simple_gated | 0.7075 | +7.94% | Balanced gating |
| 4_crossmodal_gated | 0.7090 | +8.16% | Balanced gating |
| 5_full_gated | 0.7081 | +8.02% | Balanced gating |

## Interpretability Analysis


### 4_crossmodal_gated

- Average text gate: 0.479 (±0.026)
- Average ProtT5 gate: 0.458 (±0.022)

### 5_full_gated

- Average text gate: 0.568 (±0.055)
- Average ProtT5 gate: 0.504 (±0.005)
