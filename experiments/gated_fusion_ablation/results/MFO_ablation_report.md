# Ablation Study Report for MFO

## Performance Comparison

| Model | Best F-max | Improvement | Key Insights |
|-------|------------|-------------|-------------|
| 0_baseline_esm | 0.6498 | +8.51% | ESM only baseline |
| 0_baseline_prott5 | 0.6574 | +9.77% | ProtT5 only baseline |
| 0_baseline_text | 0.5988 | +0.00% | Text only baseline |
| 1_simple_concat | 0.6534 | +9.12% | Text/ProtT5 magnitude ratio: 24.01 |
| 2_transformed_concat | 0.6685 | +11.64% | Feature similarity: 0.328 |
| 3_simple_gated | 0.6615 | +10.46% | Balanced gating |
| 4_crossmodal_gated | 0.6614 | +10.45% | Balanced gating |
| 5_full_gated | 0.6533 | +9.09% | Balanced gating |

## Interpretability Analysis


### 4_crossmodal_gated

- Average text gate: 0.490 (±0.017)
- Average ProtT5 gate: 0.457 (±0.024)

### 5_full_gated

- Average text gate: 0.577 (±0.073)
- Average ProtT5 gate: 0.512 (±0.016)
