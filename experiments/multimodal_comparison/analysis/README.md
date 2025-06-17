# Comprehensive Multi-Modal GO Prediction Results



## 1. Executive Summary {#executive-summary}

**Total Experiments Analyzed:** 48

**Best Overall Model:** ESM-only (MFO)
- F-max: 0.6876
- mAP: 0.4301
- AUROC: 0.9237

**Best Models by Aspect:**
- BPO: ESM + Text (F-max: 0.4030)
- CCO: ESM-only (F-max: 0.6678)
- MFO: ESM-only (F-max: 0.6876)

## 2. Ablation Study: Single Modality Performance {#ablation-single}

This table shows the performance of individual modalities:

| Model | BPO F-max | BPO mAP | BPO AUROC | CCO F-max | CCO mAP | CCO AUROC | MFO F-max | MFO mAP | MFO AUROC | Avg F-max |
|-------|-----------|---------|-----------|-----------|---------|-----------|-----------|---------|-----------|----------|
| Naïve Baseline | 0.1862 | 0.0209 | 0.5000 | 0.5725 | 0.0276 | 0.5000 | 0.5319 | 0.0155 | 0.5000 | 0.4302 |
| ESM-only | 0.3979 | 0.1789 | 0.8139 | 0.6678 | 0.2443 | 0.8945 | 0.6876 | 0.4301 | 0.9237 | 0.5844 |
| Text-only | 0.3850 | 0.1303 | 0.7866 | 0.6159 | 0.1212 | 0.7932 | 0.6371 | 0.3305 | 0.8780 | 0.5460 |
| Structure (k-NN + ESM) | 0.3849 | 0.1337 | 0.8237 | 0.6658 | 0.2246 | 0.8920 | 0.6800 | 0.4168 | 0.9188 | 0.5769 |

## 3. Ablation Study: Structure Configurations {#ablation-structure}

This table compares different graph construction methods:

| Configuration | Graph Type | Node Features | BPO F-max | CCO F-max | MFO F-max | Avg F-max | Avg mAP |
|---------------|------------|---------------|-----------|-----------|-----------|-----------|--------|
| Structure (Radius + One-Hot) | Radius | One-Hot | 0.3319 | 0.6253 | 0.6044 | 0.5205 | 0.1434 |
| Structure (Radius + ESM) | Radius | ESM | 0.3850 | 0.6628 | 0.6782 | 0.5753 | 0.2466 |
| Structure (k-NN + One-Hot) | k-NN | One-Hot | 0.3348 | 0.6293 | 0.5995 | 0.5212 | 0.1459 |
| Structure (k-NN + ESM) | k-NN | ESM | 0.3849 | 0.6658 | 0.6800 | 0.5769 | 0.2583 |

## 4. Ablation Study: Multi-Modal Combinations {#ablation-multimodal}

This table shows the effect of combining different modalities:

### Concatenation Fusion

| Model Combination | BPO F-max | CCO F-max | MFO F-max | Avg F-max | Δ vs Best Single |
|-------------------|-----------|-----------|-----------|-----------|------------------|
| ESM + Text | 0.4030 | 0.6655 | 0.6775 | 0.5820 | -0.4% |
| ESM + Structure | 0.3905 | 0.6663 | 0.6824 | 0.5797 | -0.8% |
| Full Model | 0.4004 | 0.6663 | 0.6828 | 0.5832 | -0.2% |

### Attention Fusion

| Model Combination | BPO F-max | CCO F-max | MFO F-max | Avg F-max | Δ vs Best Single |
|-------------------|-----------|-----------|-----------|-----------|------------------|
| ESM + Text + Attention | 0.3931 | 0.6632 | 0.6766 | 0.5776 | -1.2% |
| ESM + Structure + Attention | 0.3725 | 0.6596 | 0.6721 | 0.5681 | -2.8% |
| Full Model + Attention | 0.3837 | 0.6606 | 0.6724 | 0.5723 | -2.1% |

## 5. Comparison with Naïve Baseline {#naive-comparison}

This table shows the improvement of each model over the naïve baseline:

| Model | BPO Δ | CCO Δ | MFO Δ | Avg Δ |
|-------|-------|-------|-------|-------|
| ESM-only | +113.7% | +16.6% | +29.3% | +53.2% |
| Text-only | +106.7% | +7.6% | +19.8% | +44.7% |
| Structure (k-NN + ESM) | +106.7% | +16.3% | +27.9% | +50.3% |
| ESM + Text | +116.4% | +16.2% | +27.4% | +53.3% |
| ESM + Structure | +109.7% | +16.4% | +28.3% | +51.5% |
| Full Model | +115.0% | +16.4% | +28.4% | +53.3% |
| ESM + Text + Attention | +111.1% | +15.8% | +27.2% | +51.4% |
| ESM + Structure + Attention | +100.0% | +15.2% | +26.4% | +47.2% |
| Full Model + Attention | +106.0% | +15.4% | +26.4% | +49.3% |

## 6. Performance by GO Aspect {#aspect-performance}


### BPO Performance

| Rank | Model | F-max | mAP | AUROC | Coverage |
|------|-------|-------|-----|-------|----------|
| 1 | ESM + Text | 0.4030 | 0.1522 | 0.8346 | nan |
| 2 | Full Model | 0.4004 | 0.1502 | 0.8332 | nan |
| 3 | ESM-only | 0.3979 | 0.1789 | 0.8139 | nan |
| 4 | ESM + Text + Attention | 0.3931 | 0.1383 | 0.8313 | nan |
| 5 | ESM + Structure | 0.3905 | 0.1441 | 0.8233 | nan |
| 6 | C43 | 0.3859 | 0.1388 | 0.8231 | nan |
| 7 | Text-only | 0.3850 | 0.1303 | 0.7866 | nan |
| 8 | Structure (Radius + ESM) | 0.3850 | 0.1575 | 0.8145 | nan |
| 9 | Structure (k-NN + ESM) | 0.3849 | 0.1337 | 0.8237 | nan |
| 10 | C42 | 0.3843 | 0.1401 | 0.8235 | nan |

### CCO Performance

| Rank | Model | F-max | mAP | AUROC | Coverage |
|------|-------|-------|-----|-------|----------|
| 1 | ESM-only | 0.6678 | 0.2443 | 0.8945 | nan |
| 2 | ESM + Structure | 0.6663 | 0.2211 | 0.8909 | nan |
| 3 | Full Model | 0.6663 | 0.2067 | 0.8838 | nan |
| 4 | C43 | 0.6661 | 0.2135 | 0.8923 | nan |
| 5 | Structure (k-NN + ESM) | 0.6658 | 0.2246 | 0.8920 | nan |
| 6 | ESM + Text | 0.6655 | 0.2014 | 0.8831 | nan |
| 7 | C42 | 0.6647 | 0.1875 | 0.8840 | nan |
| 8 | C22 | 0.6636 | 0.1745 | 0.8785 | nan |
| 9 | ESM + Text + Attention | 0.6632 | 0.1748 | 0.8863 | nan |
| 10 | Structure (Radius + ESM) | 0.6628 | 0.1707 | 0.8770 | nan |

### MFO Performance

| Rank | Model | F-max | mAP | AUROC | Coverage |
|------|-------|-------|-----|-------|----------|
| 1 | ESM-only | 0.6876 | 0.4301 | 0.9237 | nan |
| 2 | Full Model | 0.6828 | 0.4221 | 0.9154 | nan |
| 3 | ESM + Structure | 0.6824 | 0.4060 | 0.9203 | nan |
| 4 | Structure (k-NN + ESM) | 0.6800 | 0.4168 | 0.9188 | nan |
| 5 | C42 | 0.6792 | 0.4004 | 0.9185 | nan |
| 6 | Structure (Radius + ESM) | 0.6782 | 0.4117 | 0.9146 | nan |
| 7 | ESM + Text | 0.6775 | 0.4162 | 0.9249 | nan |
| 8 | C22 | 0.6770 | 0.3961 | 0.9226 | nan |
| 9 | ESM + Text + Attention | 0.6766 | 0.3933 | 0.9160 | nan |
| 10 | Full Model + Attention | 0.6724 | 0.3599 | 0.9163 | nan |

## 7. Best Configurations Summary {#best-configs}

Summary of best performing configurations:

| Category | Model | Aspect | F-max | mAP | AUROC |
|----------|-------|--------|-------|-----|-------|
| Overall Best | ESM-only | MFO | 0.6876 | 0.4301 | 0.9237 |
| Best Single Modality | ESM-only | MFO | 0.6876 | 0.4301 | 0.9237 |
| Best Multi-Modal | Full Model | MFO | 0.6828 | 0.4221 | 0.9154 |

## 8. Computational Efficiency {#efficiency}

Analysis of computational requirements:

| Model Group | Avg Training Time (hours) | Avg F-max | Efficiency (F-max/hour) |
|-------------|---------------------------|-----------|-------------------------|
| Naïve | 0.01 | 0.4302 | 80.0985 |
| Single Modality | nan | 0.5652 | 0.0000 |
| Structure-based | nan | 0.5485 | 0.0000 |
| Multi-Modal | nan | 0.5771 | 0.0000 |
