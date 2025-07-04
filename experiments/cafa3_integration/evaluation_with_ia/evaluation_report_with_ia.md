# CAFA3 Evaluation Report with Information Accretion

## Executive Summary

| Aspect   |   F-max | Best Model (F)   |   Threshold (F) |   F-max (IA-weighted) | Best Model (F-IA)   |   S-min |   S-min (IA-weighted) |
|:---------|--------:|:-----------------|----------------:|----------------------:|:--------------------|--------:|----------------------:|
| BPO      |  0.5688 | A_ESM_only_BPO   |            0.6  |                0.4409 | A_ESM_only_BPO      | 19.4671 |                8.2007 |
| CCO      |  0.6981 | A_ESM_only_CCO   |            0.68 |                0.5202 | A_ESM_only_CCO      |  6.3018 |                2.1532 |
| MFO      |  0.6741 | A_ESM_only_MFO   |            0.61 |                0.5478 | A_ESM_only_MFO      |  4.4445 |                2.1084 |

## Information Accretion Impact Analysis

## Model Performance Summary

### BPO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| A_ESM_only_BPO             |  0.5688 |            1 |       0.4409 |                 1 | 19.4671 |            1 |       8.2007 |                 1 |       1    |
| F_ESM_Text_moe_BPO         |  0.5666 |            2 |       0.4396 |                 3 | 19.8775 |            3 |       8.338  |                 2 |       2.5  |
| H_ESM_Text_contrastive_BPO |  0.5658 |            3 |       0.4404 |                 2 | 19.9108 |            5 |       8.3981 |                 5 |       3.75 |
| E_ESM_Text_gated_BPO       |  0.5648 |            4 |       0.437  |                 4 | 19.889  |            4 |       8.3918 |                 4 |       4    |
| G_ESM_Text_transformer_BPO |  0.5601 |            6 |       0.4294 |                 7 | 19.6938 |            2 |       8.3515 |                 3 |       4.5  |
| E_ProstT5_only_BPO         |  0.5612 |            5 |       0.4296 |                 6 | 20.005  |            6 |       8.4777 |                 6 |       5.75 |
| D_ESM_Text_BPO             |  0.5575 |            7 |       0.4304 |                 5 | 20.5495 |            8 |       8.576  |                 7 |       6.75 |
| B_Text_only_BPO            |  0.5453 |            8 |       0.4035 |                 8 | 20.4962 |            7 |       8.6104 |                 8 |       7.75 |

### CCO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| E_ESM_Text_gated_CCO       |  0.6943 |            2 |       0.5154 |                 2 |  6.3018 |            1 |       2.1532 |                 1 |       1.5  |
| A_ESM_only_CCO             |  0.6981 |            1 |       0.5202 |                 1 |  6.3118 |            2 |       2.1613 |                 3 |       1.75 |
| G_ESM_Text_transformer_CCO |  0.6906 |            3 |       0.512  |                 5 |  6.3653 |            3 |       2.161  |                 2 |       3.25 |
| H_ESM_Text_contrastive_CCO |  0.6904 |            4 |       0.5082 |                 8 |  6.3902 |            4 |       2.1655 |                 4 |       5    |
| E_ProstT5_only_CCO         |  0.69   |            5 |       0.5138 |                 3 |  6.4748 |            7 |       2.1882 |                 7 |       5.5  |
| D_ESM_Text_CCO             |  0.6876 |            7 |       0.5123 |                 4 |  6.468  |            6 |       2.1743 |                 6 |       5.75 |
| F_ESM_Text_moe_CCO         |  0.6895 |            6 |       0.5095 |                 7 |  6.4321 |            5 |       2.1723 |                 5 |       5.75 |
| Q_ProstT5_Text_concat_CCO  |  0.6874 |            8 |       0.5041 |                 9 |  6.4764 |            8 |       2.2058 |                 8 |       8.25 |
| R_ProstT5_Text_gated_CCO   |  0.6868 |            9 |       0.5101 |                 6 |  6.5185 |            9 |       2.2213 |                 9 |       8.25 |
| B_Text_only_CCO            |  0.6718 |           10 |       0.4758 |                10 |  6.7744 |           10 |       2.2784 |                10 |      10    |

### MFO

**Top 10 Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| A_ESM_only_MFO             |  0.6741 |            1 |       0.5478 |                 1 |  4.4445 |            1 |       2.1084 |                 1 |       1    |
| F_ESM_Text_moe_MFO         |  0.6644 |            3 |       0.5281 |                 4 |  4.4472 |            2 |       2.1451 |                 2 |       2.75 |
| E_ProstT5_only_MFO         |  0.6688 |            2 |       0.5415 |                 2 |  4.5948 |            5 |       2.1895 |                 5 |       3.5  |
| G_ESM_Text_transformer_MFO |  0.658  |            5 |       0.5237 |                 5 |  4.5085 |            3 |       2.1623 |                 3 |       4    |
| Q_ProstT5_Text_concat_MFO  |  0.6587 |            4 |       0.5303 |                 3 |  4.5952 |            6 |       2.1962 |                 6 |       4.75 |
| E_ESM_Text_gated_MFO       |  0.6577 |            6 |       0.5197 |                 8 |  4.5137 |            4 |       2.1742 |                 4 |       5.5  |
| D_ESM_Text_MFO             |  0.6553 |            7 |       0.5227 |                 6 |  4.6283 |            8 |       2.2032 |                 8 |       7.25 |
| H_ESM_Text_contrastive_MFO |  0.6525 |            9 |       0.5166 |                 9 |  4.6113 |            7 |       2.2023 |                 7 |       8    |
| R_ProstT5_Text_gated_MFO   |  0.6531 |            8 |       0.5208 |                 7 |  4.6781 |            9 |       2.2376 |                 9 |       8.25 |
| B_Text_only_MFO            |  0.6344 |           10 |       0.4933 |                10 |  4.9424 |           10 |       2.3472 |                10 |      10    |

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
|      2 | F_ESM_Text_moe_BPO         |      0.5666 |      0.6065 |   0.5316 |          1 |        0.62 |
|      3 | H_ESM_Text_contrastive_BPO |      0.5658 |      0.5712 |   0.5605 |          1 |        0.56 |
|      4 | E_ESM_Text_gated_BPO       |      0.5648 |      0.5823 |   0.5483 |          1 |        0.58 |
|      5 | E_ProstT5_only_BPO         |      0.5612 |      0.586  |   0.5384 |          1 |        0.56 |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | A_ESM_only_BPO             |                 0.4409 |                 0.4814 |              0.4068 |                0.9929 |        0.6  |
|      2 | H_ESM_Text_contrastive_BPO |                 0.4404 |                 0.4487 |              0.4323 |                0.995  |        0.56 |
|      3 | F_ESM_Text_moe_BPO         |                 0.4396 |                 0.4519 |              0.428  |                0.9933 |        0.57 |
|      4 | E_ESM_Text_gated_BPO       |                 0.437  |                 0.4566 |              0.419  |                0.9929 |        0.58 |
|      5 | D_ESM_Text_BPO             |                 0.4304 |                 0.4382 |              0.423  |                0.9887 |        0.57 |

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
|      2 | E_ESM_Text_gated_CCO       |      0.6943 |      0.7398 |   0.6541 |          1 |        0.67 |
|      3 | G_ESM_Text_transformer_CCO |      0.6906 |      0.7418 |   0.646  |          1 |        0.7  |
|      4 | H_ESM_Text_contrastive_CCO |      0.6904 |      0.7438 |   0.6442 |          1 |        0.75 |
|      5 | E_ProstT5_only_CCO         |      0.69   |      0.7303 |   0.6538 |          1 |        0.65 |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | A_ESM_only_CCO             |                 0.5202 |                 0.5658 |              0.4813 |                0.9968 |        0.61 |
|      2 | E_ESM_Text_gated_CCO       |                 0.5154 |                 0.5919 |              0.4564 |                0.9866 |        0.64 |
|      3 | E_ProstT5_only_CCO         |                 0.5138 |                 0.5512 |              0.4812 |                0.9953 |        0.58 |
|      4 | D_ESM_Text_CCO             |                 0.5123 |                 0.5457 |              0.4827 |                0.9937 |        0.6  |
|      5 | G_ESM_Text_transformer_CCO |                 0.512  |                 0.5263 |              0.4984 |                0.9976 |        0.57 |

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

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | A_ESM_only_MFO             |      0.6741 |      0.7369 |   0.6212 |          1 |        0.61 |
|      2 | E_ProstT5_only_MFO         |      0.6688 |      0.6962 |   0.6434 |          1 |        0.57 |
|      3 | F_ESM_Text_moe_MFO         |      0.6644 |      0.689  |   0.6415 |          1 |        0.53 |
|      4 | Q_ProstT5_Text_concat_MFO  |      0.6587 |      0.6762 |   0.6421 |          1 |        0.44 |
|      5 | G_ESM_Text_transformer_MFO |      0.658  |      0.7217 |   0.6047 |          1 |        0.6  |

### Top 5 Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | A_ESM_only_MFO             |                 0.5478 |                 0.5877 |              0.513  |                0.9912 |        0.5  |
|      2 | E_ProstT5_only_MFO         |                 0.5415 |                 0.6033 |              0.4912 |                0.9824 |        0.56 |
|      3 | Q_ProstT5_Text_concat_MFO  |                 0.5303 |                 0.551  |              0.5112 |                0.993  |        0.35 |
|      4 | F_ESM_Text_moe_MFO         |                 0.5281 |                 0.5994 |              0.472  |                0.9903 |        0.53 |
|      5 | G_ESM_Text_transformer_MFO |                 0.5237 |                 0.571  |              0.4837 |                0.9894 |        0.48 |

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
