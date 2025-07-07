# CAFA3 Evaluation Report with Information Accretion

## Executive Summary

| Aspect   |   F-max | Best Model (F)   |   Threshold (F) |   F-max (IA-weighted) | Best Model (F-IA)     |   S-min |   S-min (IA-weighted) |
|:---------|--------:|:-----------------|----------------:|----------------------:|:----------------------|--------:|----------------------:|
| BPO      |  0.5688 | ESM_only_BPO     |            0.6  |                0.4409 | ESM_only_BPO          | 19.4671 |                8.2007 |
| CCO      |  0.6981 | ESM_only_CCO     |            0.68 |                0.525  | ESM_PROTT5_concat_CCO |  6.3018 |                2.1416 |
| MFO      |  0.675  | ProtT5_only_MFO  |            0.58 |                0.5566 | ProtT5_only_MFO       |  4.4445 |                2.1084 |

## Information Accretion Impact Analysis

## Model Performance Summary

### BPO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| ESM_only_BPO               |  0.5688 |            1 |       0.4409 |                 1 | 19.4671 |            1 |       8.2007 |                 1 |       1    |
| ProtT5_Text_gated_BPO      |  0.5669 |            2 |       0.4369 |                 5 | 19.8297 |            3 |       8.2711 |                 2 |       3    |
| ESM_Text_moe_BPO           |  0.5666 |            3 |       0.4396 |                 3 | 19.8775 |            4 |       8.338  |                 4 |       3.5  |
| ESM_Text_contrastive_BPO   |  0.5658 |            4 |       0.4404 |                 2 | 19.9108 |            6 |       8.3981 |                 7 |       4.75 |
| ESM_Text_gated_BPO         |  0.5648 |            5 |       0.437  |                 4 | 19.889  |            5 |       8.3918 |                 6 |       5    |
| ProtT5_Text_moe_BPO        |  0.5624 |            8 |       0.4313 |                 9 | 20.0847 |            9 |       8.3295 |                 3 |       7.25 |
| ESM_Text_transformer_BPO   |  0.5601 |           12 |       0.4294 |                13 | 19.6938 |            2 |       8.3515 |                 5 |       8    |
| ProtT5_only_BPO            |  0.5641 |            6 |       0.4333 |                 7 | 20.0797 |            8 |       8.4746 |                13 |       8.5  |
| ESM_ProstT5_gated_BPO      |  0.5626 |            7 |       0.4299 |                11 | 20.1542 |           10 |       8.3986 |                 8 |       9    |
| ESM_PROTT5_concat_BPO      |  0.5607 |           11 |       0.4335 |                 6 | 20.1929 |           11 |       8.4032 |                 9 |       9.25 |
| ProstT5_Text_gated_BPO     |  0.5617 |            9 |       0.4322 |                 8 | 20.2473 |           12 |       8.4471 |                11 |      10    |
| ProstT5_only_BPO           |  0.5612 |           10 |       0.4296 |                12 | 20.005  |            7 |       8.4777 |                14 |      10.75 |
| ESM_PROTT5_gated_BPO       |  0.5551 |           14 |       0.4203 |                14 | 20.4139 |           13 |       8.4547 |                12 |      13.25 |
| ESM_Text_BPO               |  0.5575 |           13 |       0.4304 |                10 | 20.5495 |           15 |       8.576  |                16 |      13.5  |
| Text_only_BPO              |  0.5453 |           15 |       0.4035 |                15 | 20.4962 |           14 |       8.6104 |                17 |      15.25 |
| ESM_PROTT5_contrastive_BPO |  0.5338 |           17 |       0.3914 |                17 | 20.9783 |           17 |       8.4349 |                10 |      15.25 |
| ESM_PROTT5_transformer_BPO |  0.5409 |           16 |       0.4    |                16 | 20.6735 |           16 |       8.5683 |                15 |      15.75 |

### CCO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| ESM_PROTT5_concat_CCO      |  0.6966 |            2 |       0.525  |                 1 |  6.3039 |            2 |       2.1416 |                 1 |       1.5  |
| ESM_Text_gated_CCO         |  0.6943 |            4 |       0.5154 |                 5 |  6.3018 |            1 |       2.1532 |                 2 |       3    |
| ESM_only_CCO               |  0.6981 |            1 |       0.5202 |                 4 |  6.3118 |            3 |       2.1613 |                 5 |       3.25 |
| ProtT5_only_CCO            |  0.6953 |            3 |       0.522  |                 3 |  6.3553 |            4 |       2.1563 |                 3 |       3.25 |
| ProtT5_Text_gated_CCO      |  0.6942 |            5 |       0.5229 |                 2 |  6.3604 |            5 |       2.1618 |                 6 |       4.5  |
| ESM_Text_transformer_CCO   |  0.6906 |            6 |       0.512  |                 9 |  6.3653 |            6 |       2.161  |                 4 |       6.25 |
| ProstT5_only_CCO           |  0.69   |            8 |       0.5138 |                 6 |  6.4748 |           10 |       2.1882 |                11 |       8.75 |
| ESM_Text_contrastive_CCO   |  0.6904 |            7 |       0.5082 |                14 |  6.3902 |            7 |       2.1655 |                 7 |       8.75 |
| ESM_Text_CCO               |  0.6876 |           12 |       0.5123 |                 8 |  6.468  |            9 |       2.1743 |                 9 |       9.5  |
| ESM_Text_moe_CCO           |  0.6895 |            9 |       0.5095 |                13 |  6.4321 |            8 |       2.1723 |                 8 |       9.5  |
| ESM_PROTT5_transformer_CCO |  0.6882 |           10 |       0.5117 |                10 |  6.4843 |           12 |       2.193  |                13 |      11.25 |
| ESM_ProstT5_gated_CCO      |  0.688  |           11 |       0.5114 |                11 |  6.5239 |           14 |       2.1766 |                10 |      11.5  |
| ProtT5_Text_moe_CCO        |  0.6846 |           15 |       0.5136 |                 7 |  6.5499 |           15 |       2.1886 |                12 |      12.25 |
| ProstT5_Text_concat_CCO    |  0.6874 |           13 |       0.5041 |                16 |  6.4764 |           11 |       2.2058 |                14 |      13.5  |
| ProstT5_Text_gated_CCO     |  0.6868 |           14 |       0.5101 |                12 |  6.5185 |           13 |       2.2213 |                16 |      13.75 |
| ESM_PROTT5_gated_CCO       |  0.6798 |           16 |       0.5052 |                15 |  6.7036 |           16 |       2.2159 |                15 |      15.5  |
| Text_only_CCO              |  0.6718 |           17 |       0.4758 |                17 |  6.7744 |           17 |       2.2784 |                17 |      17    |

### MFO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| ESM_only_MFO               |  0.6741 |            2 |       0.5478 |                 2 |  4.4445 |            1 |       2.1084 |                 1 |       1.5  |
| ESM_Text_moe_MFO           |  0.6644 |            5 |       0.5281 |                 9 |  4.4472 |            2 |       2.1451 |                 2 |       4.5  |
| ProstT5_only_MFO           |  0.6688 |            4 |       0.5415 |                 4 |  4.5948 |            6 |       2.1895 |                 6 |       5    |
| ProtT5_only_MFO            |  0.675  |            1 |       0.5566 |                 1 |  4.609  |            9 |       2.2071 |                10 |       5.25 |
| ESM_PROTT5_concat_MFO      |  0.6605 |            6 |       0.5339 |                 5 |  4.582  |            5 |       2.1829 |                 5 |       5.25 |
| ESM_PROTT5_moe_MFO         |  0.6699 |            3 |       0.5423 |                 3 |  4.6072 |            8 |       2.2121 |                12 |       6.5  |
| ESM_Text_transformer_MFO   |  0.658  |            9 |       0.5237 |                11 |  4.5085 |            3 |       2.1623 |                 3 |       6.5  |
| ProstT5_Text_concat_MFO    |  0.6587 |            8 |       0.5303 |                 6 |  4.5952 |            7 |       2.1962 |                 7 |       7    |
| ESM_Text_gated_MFO         |  0.6577 |           10 |       0.5197 |                14 |  4.5137 |            4 |       2.1742 |                 4 |       8    |
| ProtT5_Text_moe_MFO        |  0.6596 |            7 |       0.5293 |                 8 |  4.6116 |           11 |       2.2119 |                11 |       9.25 |
| ESM_Text_MFO               |  0.6553 |           14 |       0.5227 |                12 |  4.6283 |           12 |       2.2032 |                 9 |      11.75 |
| ESM_PROTT5_gated_MFO       |  0.6575 |           11 |       0.5301 |                 7 |  4.7161 |           16 |       2.2557 |                16 |      12.5  |
| ESM_Text_contrastive_MFO   |  0.6525 |           16 |       0.5166 |                16 |  4.6113 |           10 |       2.2023 |                 8 |      12.5  |
| ESM_ProstT5_gated_MFO      |  0.6565 |           13 |       0.5239 |                10 |  4.6607 |           14 |       2.2188 |                13 |      12.5  |
| ProtT5_Text_gated_MFO      |  0.6566 |           12 |       0.5182 |                15 |  4.6406 |           13 |       2.2302 |                14 |      13.5  |
| ProstT5_Text_gated_MFO     |  0.6531 |           15 |       0.5208 |                13 |  4.6781 |           15 |       2.2376 |                15 |      14.5  |
| ESM_PROTT5_transformer_MFO |  0.6453 |           17 |       0.5101 |                17 |  4.8264 |           17 |       2.305  |                17 |      17    |
| Text_only_MFO              |  0.6344 |           18 |       0.4933 |                18 |  4.9424 |           18 |       2.3472 |                18 |      18    |
| Structure_MFO              |  0.1408 |           19 |       0.0343 |                19 |  8.3822 |           19 |       3.2406 |                19 |      19    |

## Information Accretion Statistics

| Aspect   |   GO Terms |   Mean IC |   Std IC |   Min IC |   Max IC |
|:---------|-----------:|----------:|---------:|---------:|---------:|
| BPO      |       3992 |     5.516 |    1.25  |       -0 |    7.084 |
| CCO      |        551 |     5.156 |    1.472 |       -0 |    6.96  |
| MFO      |        677 |     5.137 |    1.202 |       -0 |    6.649 |

## BPO Detailed Results

### All Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | ESM_only_BPO               |      0.5688 |      0.6046 |   0.537  |          1 |        0.6  |
|      2 | ProtT5_Text_gated_BPO      |      0.5669 |      0.588  |   0.5472 |          1 |        0.59 |
|      3 | ESM_Text_moe_BPO           |      0.5666 |      0.6065 |   0.5316 |          1 |        0.62 |
|      4 | ESM_Text_contrastive_BPO   |      0.5658 |      0.5712 |   0.5605 |          1 |        0.56 |
|      5 | ESM_Text_gated_BPO         |      0.5648 |      0.5823 |   0.5483 |          1 |        0.58 |
|      6 | ProtT5_only_BPO            |      0.5641 |      0.579  |   0.5501 |          1 |        0.59 |
|      7 | ESM_ProstT5_gated_BPO      |      0.5626 |      0.6054 |   0.5255 |          1 |        0.59 |
|      8 | ProtT5_Text_moe_BPO        |      0.5624 |      0.576  |   0.5493 |          1 |        0.53 |
|      9 | ProstT5_Text_gated_BPO     |      0.5617 |      0.5991 |   0.5286 |          1 |        0.59 |
|     10 | ProstT5_only_BPO           |      0.5612 |      0.586  |   0.5384 |          1 |        0.56 |
|     11 | ESM_PROTT5_concat_BPO      |      0.5607 |      0.5799 |   0.5427 |          1 |        0.6  |
|     12 | ESM_Text_transformer_BPO   |      0.5601 |      0.5911 |   0.5322 |          1 |        0.6  |
|     13 | ESM_Text_BPO               |      0.5575 |      0.5909 |   0.5277 |          1 |        0.61 |
|     14 | ESM_PROTT5_gated_BPO       |      0.5551 |      0.5902 |   0.524  |          1 |        0.66 |
|     15 | Text_only_BPO              |      0.5453 |      0.5815 |   0.5134 |          1 |        0.58 |
|     16 | ESM_PROTT5_transformer_BPO |      0.5409 |      0.57   |   0.5146 |          1 |        0.54 |
|     17 | ESM_PROTT5_contrastive_BPO |      0.5338 |      0.559  |   0.5108 |          1 |        0.5  |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | ESM_only_BPO               |                 0.4409 |                 0.4814 |              0.4068 |                0.9929 |        0.6  |
|      2 | ESM_Text_contrastive_BPO   |                 0.4404 |                 0.4487 |              0.4323 |                0.995  |        0.56 |
|      3 | ESM_Text_moe_BPO           |                 0.4396 |                 0.4519 |              0.428  |                0.9933 |        0.57 |
|      4 | ESM_Text_gated_BPO         |                 0.437  |                 0.4566 |              0.419  |                0.9929 |        0.58 |
|      5 | ProtT5_Text_gated_BPO      |                 0.4369 |                 0.4288 |              0.4452 |                0.9895 |        0.52 |
|      6 | ESM_PROTT5_concat_BPO      |                 0.4335 |                 0.4165 |              0.4518 |                0.9933 |        0.51 |
|      7 | ProtT5_only_BPO            |                 0.4333 |                 0.4231 |              0.4439 |                0.9958 |        0.54 |
|      8 | ProstT5_Text_gated_BPO     |                 0.4322 |                 0.4714 |              0.3991 |                0.9791 |        0.59 |
|      9 | ProtT5_Text_moe_BPO        |                 0.4313 |                 0.4141 |              0.45   |                0.9912 |        0.44 |
|     10 | ESM_Text_BPO               |                 0.4304 |                 0.4382 |              0.423  |                0.9887 |        0.57 |
|     11 | ESM_ProstT5_gated_BPO      |                 0.4299 |                 0.4624 |              0.4016 |                0.9858 |        0.56 |
|     12 | ProstT5_only_BPO           |                 0.4296 |                 0.4284 |              0.4309 |                0.9971 |        0.51 |
|     13 | ESM_Text_transformer_BPO   |                 0.4294 |                 0.4258 |              0.433  |                0.9941 |        0.53 |
|     14 | ESM_PROTT5_gated_BPO       |                 0.4203 |                 0.4132 |              0.4276 |                0.9787 |        0.57 |
|     15 | Text_only_BPO              |                 0.4035 |                 0.4194 |              0.3887 |                0.9983 |        0.52 |
|     16 | ESM_PROTT5_transformer_BPO |                 0.4    |                 0.4326 |              0.372  |                0.9937 |        0.52 |
|     17 | ESM_PROTT5_contrastive_BPO |                 0.3914 |                 0.3877 |              0.3952 |                0.9933 |        0.4  |

## CCO Detailed Results

### All Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | ESM_only_CCO               |      0.6981 |      0.7481 |   0.6545 |          1 |        0.68 |
|      2 | ESM_PROTT5_concat_CCO      |      0.6966 |      0.7437 |   0.655  |          1 |        0.72 |
|      3 | ProtT5_only_CCO            |      0.6953 |      0.7415 |   0.6546 |          1 |        0.66 |
|      4 | ESM_Text_gated_CCO         |      0.6943 |      0.7398 |   0.6541 |          1 |        0.67 |
|      5 | ProtT5_Text_gated_CCO      |      0.6942 |      0.7148 |   0.6747 |          1 |        0.61 |
|      6 | ESM_Text_transformer_CCO   |      0.6906 |      0.7418 |   0.646  |          1 |        0.7  |
|      7 | ESM_Text_contrastive_CCO   |      0.6904 |      0.7438 |   0.6442 |          1 |        0.75 |
|      8 | ProstT5_only_CCO           |      0.69   |      0.7303 |   0.6538 |          1 |        0.65 |
|      9 | ESM_Text_moe_CCO           |      0.6895 |      0.7417 |   0.6441 |          1 |        0.65 |
|     10 | ESM_PROTT5_transformer_CCO |      0.6882 |      0.7079 |   0.6696 |          1 |        0.66 |
|     11 | ESM_ProstT5_gated_CCO      |      0.688  |      0.7218 |   0.6574 |          1 |        0.68 |
|     12 | ESM_Text_CCO               |      0.6876 |      0.7107 |   0.6658 |          1 |        0.65 |
|     13 | ProstT5_Text_concat_CCO    |      0.6874 |      0.7465 |   0.6371 |          1 |        0.71 |
|     14 | ProstT5_Text_gated_CCO     |      0.6868 |      0.7447 |   0.6373 |          1 |        0.68 |
|     15 | ProtT5_Text_moe_CCO        |      0.6846 |      0.7142 |   0.6574 |          1 |        0.65 |
|     16 | ESM_PROTT5_gated_CCO       |      0.6798 |      0.7222 |   0.642  |          1 |        0.75 |
|     17 | Text_only_CCO              |      0.6718 |      0.724  |   0.6266 |          1 |        0.64 |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | ESM_PROTT5_concat_CCO      |                 0.525  |                 0.57   |              0.4865 |                0.9905 |        0.62 |
|      2 | ProtT5_Text_gated_CCO      |                 0.5229 |                 0.5558 |              0.4937 |                0.9929 |        0.54 |
|      3 | ProtT5_only_CCO            |                 0.522  |                 0.5613 |              0.4877 |                0.9913 |        0.59 |
|      4 | ESM_only_CCO               |                 0.5202 |                 0.5658 |              0.4813 |                0.9968 |        0.61 |
|      5 | ESM_Text_gated_CCO         |                 0.5154 |                 0.5919 |              0.4564 |                0.9866 |        0.64 |
|      6 | ProstT5_only_CCO           |                 0.5138 |                 0.5512 |              0.4812 |                0.9953 |        0.58 |
|      7 | ProtT5_Text_moe_CCO        |                 0.5136 |                 0.534  |              0.4947 |                0.9874 |        0.53 |
|      8 | ESM_Text_CCO               |                 0.5123 |                 0.5457 |              0.4827 |                0.9937 |        0.6  |
|      9 | ESM_Text_transformer_CCO   |                 0.512  |                 0.5263 |              0.4984 |                0.9976 |        0.57 |
|     10 | ESM_PROTT5_transformer_CCO |                 0.5117 |                 0.534  |              0.4912 |                0.9953 |        0.58 |
|     11 | ESM_ProstT5_gated_CCO      |                 0.5114 |                 0.5515 |              0.4766 |                0.9897 |        0.61 |
|     12 | ProstT5_Text_gated_CCO     |                 0.5101 |                 0.5256 |              0.4956 |                0.9968 |        0.51 |
|     13 | ESM_Text_moe_CCO           |                 0.5095 |                 0.5651 |              0.4639 |                0.9889 |        0.59 |
|     14 | ESM_Text_contrastive_CCO   |                 0.5082 |                 0.5682 |              0.4596 |                0.985  |        0.69 |
|     15 | ESM_PROTT5_gated_CCO       |                 0.5052 |                 0.5188 |              0.4923 |                0.9945 |        0.6  |
|     16 | ProstT5_Text_concat_CCO    |                 0.5041 |                 0.5131 |              0.4955 |                0.996  |        0.52 |
|     17 | Text_only_CCO              |                 0.4758 |                 0.5362 |              0.4276 |                0.9968 |        0.58 |

## MFO Detailed Results

### All Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | ProtT5_only_MFO            |      0.675  |      0.6943 |   0.6567 |     1      |        0.58 |
|      2 | ESM_only_MFO               |      0.6741 |      0.7369 |   0.6212 |     1      |        0.61 |
|      3 | ESM_PROTT5_moe_MFO         |      0.6699 |      0.6905 |   0.6506 |     1      |        0.48 |
|      4 | ProstT5_only_MFO           |      0.6688 |      0.6962 |   0.6434 |     1      |        0.57 |
|      5 | ESM_Text_moe_MFO           |      0.6644 |      0.689  |   0.6415 |     1      |        0.53 |
|      6 | ESM_PROTT5_concat_MFO      |      0.6605 |      0.7096 |   0.6178 |     1      |        0.61 |
|      7 | ProtT5_Text_moe_MFO        |      0.6596 |      0.6993 |   0.6241 |     1      |        0.53 |
|      8 | ProstT5_Text_concat_MFO    |      0.6587 |      0.6762 |   0.6421 |     1      |        0.44 |
|      9 | ESM_Text_transformer_MFO   |      0.658  |      0.7217 |   0.6047 |     1      |        0.6  |
|     10 | ESM_Text_gated_MFO         |      0.6577 |      0.7088 |   0.6134 |     1      |        0.58 |
|     11 | ESM_PROTT5_gated_MFO       |      0.6575 |      0.6985 |   0.6211 |     1      |        0.57 |
|     12 | ProtT5_Text_gated_MFO      |      0.6566 |      0.7197 |   0.6036 |     1      |        0.63 |
|     13 | ESM_ProstT5_gated_MFO      |      0.6565 |      0.7019 |   0.6166 |     1      |        0.57 |
|     14 | ESM_Text_MFO               |      0.6553 |      0.6802 |   0.6322 |     1      |        0.55 |
|     15 | ProstT5_Text_gated_MFO     |      0.6531 |      0.6812 |   0.6271 |     1      |        0.5  |
|     16 | ESM_Text_contrastive_MFO   |      0.6525 |      0.6694 |   0.6363 |     1      |        0.52 |
|     17 | ESM_PROTT5_transformer_MFO |      0.6453 |      0.6817 |   0.6125 |     1      |        0.53 |
|     18 | Text_only_MFO              |      0.6344 |      0.6597 |   0.611  |     1      |        0.49 |
|     19 | Structure_MFO              |      0.1408 |      0.0903 |   0.3189 |     0.7142 |        0.52 |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | ProtT5_only_MFO            |                 0.5566 |                 0.5959 |              0.5222 |                0.9903 |        0.55 |
|      2 | ESM_only_MFO               |                 0.5478 |                 0.5877 |              0.513  |                0.9912 |        0.5  |
|      3 | ESM_PROTT5_moe_MFO         |                 0.5423 |                 0.5904 |              0.5014 |                0.9894 |        0.47 |
|      4 | ProstT5_only_MFO           |                 0.5415 |                 0.6033 |              0.4912 |                0.9824 |        0.56 |
|      5 | ESM_PROTT5_concat_MFO      |                 0.5339 |                 0.5576 |              0.5121 |                0.9886 |        0.45 |
|      6 | ProstT5_Text_concat_MFO    |                 0.5303 |                 0.551  |              0.5112 |                0.993  |        0.35 |
|      7 | ESM_PROTT5_gated_MFO       |                 0.5301 |                 0.531  |              0.5292 |                0.9991 |        0.37 |
|      8 | ProtT5_Text_moe_MFO        |                 0.5293 |                 0.5115 |              0.5485 |                0.9956 |        0.29 |
|      9 | ESM_Text_moe_MFO           |                 0.5281 |                 0.5994 |              0.472  |                0.9903 |        0.53 |
|     10 | ESM_ProstT5_gated_MFO      |                 0.5239 |                 0.5673 |              0.4867 |                0.9938 |        0.46 |
|     11 | ESM_Text_transformer_MFO   |                 0.5237 |                 0.571  |              0.4837 |                0.9894 |        0.48 |
|     12 | ESM_Text_MFO               |                 0.5227 |                 0.562  |              0.4885 |                0.9921 |        0.5  |
|     13 | ProstT5_Text_gated_MFO     |                 0.5208 |                 0.5551 |              0.4905 |                0.9965 |        0.42 |
|     14 | ESM_Text_gated_MFO         |                 0.5197 |                 0.5341 |              0.5061 |                0.9974 |        0.42 |
|     15 | ProtT5_Text_gated_MFO      |                 0.5182 |                 0.5423 |              0.4961 |                0.9947 |        0.43 |
|     16 | ESM_Text_contrastive_MFO   |                 0.5166 |                 0.5279 |              0.5058 |                0.9974 |        0.43 |
|     17 | ESM_PROTT5_transformer_MFO |                 0.5101 |                 0.5183 |              0.5021 |                0.9947 |        0.36 |
|     18 | Text_only_MFO              |                 0.4933 |                 0.5552 |              0.4439 |                0.9903 |        0.48 |
|     19 | Structure_MFO              |                 0.0343 |                 0.02   |              0.1221 |                0.7142 |        0.52 |

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
