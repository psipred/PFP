# Multi-Modal GO Prediction Experimental Results

Generated: 2025-06-17 11:49:05

## 1. Best Overall Performance

**Best Model:** A_ESM_only_MFO
- F-max: 0.6876
- mAP: 0.4301
- AUROC: 0.9237

## 2. Results by Model Group

### Group A: Baseline Models

- ESM-only: Average F-max = 0.5844
- Text-only: Average F-max = 0.5460

### Group B: Structure Models

- Structure (Radius + One-Hot) (10.0): Average F-max = 0.5205
- Structure (Radius + ESM) (10.0): Average F-max = 0.5753
- Structure (k-NN + One-Hot) (10.0): Average F-max = 0.5212
- Structure (k-NN + ESM)(10.0): Average F-max = 0.5769
- Radius + ESM (8.0): Average F-max = 0.5742
- Radius + ESM (12.0): Average F-max = 0.3834
- k-NN + ESM (5): Average F-max = 0.5761
- k-NN + ESM (15): Average F-max = 0.5260

**Best Structure Configuration:** Structure (k-NN + ESM)(10.0)

### Group C: Multi-Modal Models

- ESM + Text: Average F-max = 0.5820
- ESM + Structure: Average F-max = 0.5797
- Full Model: Average F-max = 0.5832
- ESM + Text + Attention: Average F-max = 0.5776
- ESM + Structure + Attention: Average F-max = 0.5681
- Full Model + Attention: Average F-max = 0.5723

## 3. Results by GO Aspect

### BPO

| Model | F-max | mAP | AUROC |
|-------|-------|-----|-------|
| ESM + Text | 0.4030 | 0.1522 | 0.8346 |
| Full Model | 0.4004 | 0.1502 | 0.8332 |
| ESM-only | 0.3979 | 0.1789 | 0.8139 |
| ESM + Text + Attention | 0.3931 | 0.1383 | 0.8313 |
| ESM + Structure | 0.3905 | 0.1441 | 0.8233 |

### CCO

| Model | F-max | mAP | AUROC |
|-------|-------|-----|-------|
| ESM-only | 0.6678 | 0.2443 | 0.8945 |
| ESM + Structure | 0.6663 | 0.2211 | 0.8909 |
| Full Model | 0.6663 | 0.2067 | 0.8838 |
| k-NN + ESM (15) | 0.6661 | 0.2135 | 0.8923 |
| Structure (k-NN + ESM)(10.0) | 0.6658 | 0.2246 | 0.8920 |

### MFO

| Model | F-max | mAP | AUROC |
|-------|-------|-----|-------|
| ESM-only | 0.6876 | 0.4301 | 0.9237 |
| Full Model | 0.6828 | 0.4221 | 0.9154 |
| ESM + Structure | 0.6824 | 0.4060 | 0.9203 |
| Structure (k-NN + ESM)(10.0) | 0.6800 | 0.4168 | 0.9188 |
| k-NN + ESM (5) | 0.6792 | 0.4004 | 0.9185 |

