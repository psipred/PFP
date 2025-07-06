# CAFA3 Evaluation Report with Information Accretion

## Executive Summary

| Aspect   |   F-max | Best Model (F)    |   Threshold (F) |   F-max (IA-weighted) | Best Model (F-IA)       |   S-min |   S-min (IA-weighted) |
|:---------|--------:|:------------------|----------------:|----------------------:|:------------------------|--------:|----------------------:|
| BPO      |  0.5688 | A_ESM_only_BPO    |            0.6  |                0.4409 | A_ESM_only_BPO          | 19.4671 |                8.2007 |
| CCO      |  0.6981 | A_ESM_only_CCO    |            0.68 |                0.5229 | L_ProtT5_Text_gated_CCO |  6.3018 |                2.1532 |
| MFO      |  0.675  | D_ProtT5_only_MFO |            0.58 |                0.5566 | D_ProtT5_only_MFO       |  4.4445 |                2.1084 |

## Information Accretion Impact Analysis

## Model Performance Summary

### BPO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| A_ESM_only_BPO             |  0.5688 |            1 |       0.4409 |                 1 | 19.4671 |            1 |       8.2007 |                 1 |       1    |
| L_ProtT5_Text_gated_BPO    |  0.5669 |            2 |       0.4369 |                 5 | 19.8297 |            3 |       8.2711 |                 2 |       3    |
| F_ESM_Text_moe_BPO         |  0.5666 |            3 |       0.4396 |                 3 | 19.8775 |            4 |       8.338  |                 4 |       3.5  |
| H_ESM_Text_contrastive_BPO |  0.5658 |            4 |       0.4404 |                 2 | 19.9108 |            6 |       8.3981 |                 7 |       4.75 |
| E_ESM_Text_gated_BPO       |  0.5648 |            5 |       0.437  |                 4 | 19.889  |            5 |       8.3918 |                 6 |       5    |
| F_ProtT5_Text_moe_BPO      |  0.5624 |            8 |       0.4313 |                 8 | 20.0847 |            9 |       8.3295 |                 3 |       7    |
| D_ProtT5_only_BPO          |  0.5641 |            6 |       0.4333 |                 6 | 20.0797 |            8 |       8.4746 |                10 |       7.5  |
| G_ESM_Text_transformer_BPO |  0.5601 |           11 |       0.4294 |                12 | 19.6938 |            2 |       8.3515 |                 5 |       7.5  |
| T_ESM_ProstT5_gated_BPO    |  0.5626 |            7 |       0.4299 |                10 | 20.1542 |           10 |       8.3986 |                 8 |       8.75 |
| R_ProstT5_Text_gated_BPO   |  0.5617 |            9 |       0.4322 |                 7 | 20.2473 |           11 |       8.4471 |                 9 |       9    |

### CCO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| D_ProtT5_only_CCO          |  0.6953 |            2 |       0.522  |                 2 |  6.3553 |            3 |       2.1563 |                 2 |       2.25 |
| E_ESM_Text_gated_CCO       |  0.6943 |            3 |       0.5154 |                 4 |  6.3018 |            1 |       2.1532 |                 1 |       2.25 |
| A_ESM_only_CCO             |  0.6981 |            1 |       0.5202 |                 3 |  6.3118 |            2 |       2.1613 |                 4 |       2.5  |
| L_ProtT5_Text_gated_CCO    |  0.6942 |            4 |       0.5229 |                 1 |  6.3604 |            4 |       2.1618 |                 5 |       3.5  |
| G_ESM_Text_transformer_CCO |  0.6906 |            5 |       0.512  |                 8 |  6.3653 |            5 |       2.161  |                 3 |       5.25 |
| H_ESM_Text_contrastive_CCO |  0.6904 |            6 |       0.5082 |                12 |  6.3902 |            6 |       2.1655 |                 6 |       7.5  |
| E_ProstT5_only_CCO         |  0.69   |            7 |       0.5138 |                 5 |  6.4748 |            9 |       2.1882 |                10 |       7.75 |
| D_ESM_Text_CCO             |  0.6876 |           10 |       0.5123 |                 7 |  6.468  |            8 |       2.1743 |                 8 |       8.25 |
| F_ESM_Text_moe_CCO         |  0.6895 |            8 |       0.5095 |                11 |  6.4321 |            7 |       2.1723 |                 7 |       8.25 |
| T_ESM_ProstT5_gated_CCO    |  0.688  |            9 |       0.5114 |                 9 |  6.5239 |           12 |       2.1766 |                 9 |       9.75 |

### MFO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| A_ESM_only_MFO             |  0.6741 |            2 |       0.5478 |                 2 |  4.4445 |            1 |       2.1084 |                 1 |       1.5  |
| F_ESM_Text_moe_MFO         |  0.6644 |            4 |       0.5281 |                 6 |  4.4472 |            2 |       2.1451 |                 2 |       3.5  |
| E_ProstT5_only_MFO         |  0.6688 |            3 |       0.5415 |                 3 |  4.5948 |            5 |       2.1895 |                 5 |       4    |
| D_ProtT5_only_MFO          |  0.675  |            1 |       0.5566 |                 1 |  4.609  |            7 |       2.2071 |                 9 |       4.5  |
| G_ESM_Text_transformer_MFO |  0.658  |            7 |       0.5237 |                 8 |  4.5085 |            3 |       2.1623 |                 3 |       5.25 |
| Q_ProstT5_Text_concat_MFO  |  0.6587 |            6 |       0.5303 |                 4 |  4.5952 |            6 |       2.1962 |                 6 |       5.5  |
| E_ESM_Text_gated_MFO       |  0.6577 |            8 |       0.5197 |                11 |  4.5137 |            4 |       2.1742 |                 4 |       6.75 |
| F_ProtT5_Text_moe_MFO      |  0.6596 |            5 |       0.5293 |                 5 |  4.6116 |            9 |       2.2119 |                10 |       7.25 |
| D_ESM_Text_MFO             |  0.6553 |           11 |       0.5227 |                 9 |  4.6283 |           10 |       2.2032 |                 8 |       9.5  |
| T_ESM_ProstT5_gated_MFO    |  0.6565 |           10 |       0.5239 |                 7 |  4.6607 |           12 |       2.2188 |                11 |      10    |

## Information Accretion Statistics

| Aspect   |   GO Terms |   Mean IC |   Std IC |   Min IC |   Max IC |
|:---------|-----------:|----------:|---------:|---------:|---------:|
| BPO      |       3992 |     5.516 |    1.25  |       -0 |    7.084 |
| CCO      |        551 |     5.156 |    1.472 |       -0 |    6.96  |
| MFO      |        677 |     5.137 |    1.202 |       -0 |    6.649 |

## BPO Detailed Results

### Top 5 Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | A_ESM_only_BPO             |      0.5688 |      0.6046 |   0.537  |          1 |        0.6  |
|      2 | L_ProtT5_Text_gated_BPO    |      0.5669 |      0.588  |   0.5472 |          1 |        0.59 |
|      3 | F_ESM_Text_moe_BPO         |      0.5666 |      0.6065 |   0.5316 |          1 |        0.62 |
|      4 | H_ESM_Text_contrastive_BPO |      0.5658 |      0.5712 |   0.5605 |          1 |        0.56 |
|      5 | E_ESM_Text_gated_BPO       |      0.5648 |      0.5823 |   0.5483 |          1 |        0.58 |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | A_ESM_only_BPO             |                 0.4409 |                 0.4814 |              0.4068 |                0.9929 |        0.6  |
|      2 | H_ESM_Text_contrastive_BPO |                 0.4404 |                 0.4487 |              0.4323 |                0.995  |        0.56 |
|      3 | F_ESM_Text_moe_BPO         |                 0.4396 |                 0.4519 |              0.428  |                0.9933 |        0.57 |
|      4 | E_ESM_Text_gated_BPO       |                 0.437  |                 0.4566 |              0.419  |                0.9929 |        0.58 |
|      5 | L_ProtT5_Text_gated_BPO    |                 0.4369 |                 0.4288 |              0.4452 |                0.9895 |        0.52 |

## Metric Descriptions

- **F-measure**: Harmonic mean of precision and recall
- **S-measure**: Semantic distance-based measure (lower is better)
- **IA-weighted**: Metrics weighted by Information Accretion (term specificity)
- **Coverage**: Fraction of proteins with at least one prediction
- **Remaining Uncertainty**: Information content not captured by predictions
- **Misinformation**: Information content of incorrect predictions
- **Threshold**: Confidence threshold used for predictions
- **Rank**: Model's position in the ranking for each metric
- **Avg Rank**: Average rank across all metrics (lower is better)
## CCO Detailed Results

### Top 5 Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | A_ESM_only_CCO             |      0.6981 |      0.7481 |   0.6545 |          1 |        0.68 |
|      2 | D_ProtT5_only_CCO          |      0.6953 |      0.7415 |   0.6546 |          1 |        0.66 |
|      3 | E_ESM_Text_gated_CCO       |      0.6943 |      0.7398 |   0.6541 |          1 |        0.67 |
|      4 | L_ProtT5_Text_gated_CCO    |      0.6942 |      0.7148 |   0.6747 |          1 |        0.61 |
|      5 | G_ESM_Text_transformer_CCO |      0.6906 |      0.7418 |   0.646  |          1 |        0.7  |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                   |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | L_ProtT5_Text_gated_CCO |                 0.5229 |                 0.5558 |              0.4937 |                0.9929 |        0.54 |
|      2 | D_ProtT5_only_CCO       |                 0.522  |                 0.5613 |              0.4877 |                0.9913 |        0.59 |
|      3 | A_ESM_only_CCO          |                 0.5202 |                 0.5658 |              0.4813 |                0.9968 |        0.61 |
|      4 | E_ESM_Text_gated_CCO    |                 0.5154 |                 0.5919 |              0.4564 |                0.9866 |        0.64 |
|      5 | E_ProstT5_only_CCO      |                 0.5138 |                 0.5512 |              0.4812 |                0.9953 |        0.58 |

## Metric Descriptions

- **F-measure**: Harmonic mean of precision and recall
- **S-measure**: Semantic distance-based measure (lower is better)
- **IA-weighted**: Metrics weighted by Information Accretion (term specificity)
- **Coverage**: Fraction of proteins with at least one prediction
- **Remaining Uncertainty**: Information content not captured by predictions
- **Misinformation**: Information content of incorrect predictions
- **Threshold**: Confidence threshold used for predictions
- **Rank**: Model's position in the ranking for each metric
- **Avg Rank**: Average rank across all metrics (lower is better)
## MFO Detailed Results

### Top 5 Models by F-measure

|   Rank | Model                 |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:----------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | D_ProtT5_only_MFO     |      0.675  |      0.6943 |   0.6567 |          1 |        0.58 |
|      2 | A_ESM_only_MFO        |      0.6741 |      0.7369 |   0.6212 |          1 |        0.61 |
|      3 | E_ProstT5_only_MFO    |      0.6688 |      0.6962 |   0.6434 |          1 |        0.57 |
|      4 | F_ESM_Text_moe_MFO    |      0.6644 |      0.689  |   0.6415 |          1 |        0.53 |
|      5 | F_ProtT5_Text_moe_MFO |      0.6596 |      0.6993 |   0.6241 |          1 |        0.53 |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                     |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:--------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | D_ProtT5_only_MFO         |                 0.5566 |                 0.5959 |              0.5222 |                0.9903 |        0.55 |
|      2 | A_ESM_only_MFO            |                 0.5478 |                 0.5877 |              0.513  |                0.9912 |        0.5  |
|      3 | E_ProstT5_only_MFO        |                 0.5415 |                 0.6033 |              0.4912 |                0.9824 |        0.56 |
|      4 | Q_ProstT5_Text_concat_MFO |                 0.5303 |                 0.551  |              0.5112 |                0.993  |        0.35 |
|      5 | F_ProtT5_Text_moe_MFO     |                 0.5293 |                 0.5115 |              0.5485 |                0.9956 |        0.29 |

## Metric Descriptions

- **F-measure**: Harmonic mean of precision and recall
- **S-measure**: Semantic distance-based measure (lower is better)
- **IA-weighted**: Metrics weighted by Information Accretion (term specificity)
- **Coverage**: Fraction of proteins with at least one prediction
- **Remaining Uncertainty**: Information content not captured by predictions
- **Misinformation**: Information content of incorrect predictions
- **Threshold**: Confidence threshold used for predictions
- **Rank**: Model's position in the ranking for each metric
- **Avg Rank**: Average rank across all metrics (lower is better)
