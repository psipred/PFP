# CAFA3 Evaluation Report with Information Accretion

## Executive Summary

| Aspect   |   F-max | Best Model (F)            |   Threshold (F) |   F-max (IA-weighted) | Best Model (F-IA)   |   S-min |   S-min (IA-weighted) |
|:---------|--------:|:--------------------------|----------------:|----------------------:|:--------------------|--------:|----------------------:|
| BPO      |  0.5841 | voting_t0.5_BPO           |            0.43 |                0.4617 | voting_t0.5_BPO     | 19.0257 |                8.1079 |
| CCO      |  0.7045 | weighted_average_CCO copy |            0.63 |                0.5367 | top_5_CCO           |  6.1053 |                2.1047 |
| MFO      |  0.6958 | top_5_MFO                 |            0.54 |                0.5766 | top_5_MFO           |  4.2632 |                2.0489 |

## Information Accretion Impact Analysis

## Model Performance Summary

### BPO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| voting_t0.5_BPO            |  0.5841 |            1 |       0.4617 |                 1 | 19.0268 |            2 |       8.1079 |                 1 |       1.25 |
| weighted_average_BPO copy  |  0.5825 |            2 |       0.4598 |                 2 | 19.0257 |            1 |       8.1257 |                 2 |       1.75 |
| top_5_BPO                  |  0.5824 |            3 |       0.4598 |                 3 | 19.0924 |            3 |       8.1472 |                 3 |       3    |
| selective_fusion_BPO       |  0.5787 |            5 |       0.4552 |                 4 | 19.2616 |            5 |       8.1991 |                 4 |       4.5  |
| weighted_average_BPO       |  0.5791 |            4 |       0.4552 |                 5 | 19.2561 |            4 |       8.1991 |                 5 |       4.5  |
| ESM_only_BPO               |  0.5688 |            6 |       0.4409 |                 6 | 19.4671 |            6 |       8.2007 |                 6 |       6    |
| ProtT5_Text_gated_BPO      |  0.5669 |            7 |       0.4369 |                10 | 19.8297 |            8 |       8.2711 |                 7 |       8    |
| ESM_Text_moe_BPO           |  0.5666 |            8 |       0.4396 |                 8 | 19.8775 |            9 |       8.338  |                 9 |       8.5  |
| ESM_Text_contrastive_BPO   |  0.5658 |            9 |       0.4404 |                 7 | 19.9108 |           11 |       8.3981 |                12 |       9.75 |
| ESM_Text_gated_BPO         |  0.5648 |           10 |       0.437  |                 9 | 19.889  |           10 |       8.3918 |                11 |      10    |
| ProtT5_Text_moe_BPO        |  0.5624 |           13 |       0.4313 |                14 | 20.0847 |           14 |       8.3295 |                 8 |      12.25 |
| ESM_Text_transformer_BPO   |  0.5601 |           17 |       0.4294 |                18 | 19.6938 |            7 |       8.3515 |                10 |      13    |
| ProtT5_only_BPO            |  0.5641 |           11 |       0.4333 |                12 | 20.0797 |           13 |       8.4746 |                18 |      13.5  |
| ESM_ProstT5_gated_BPO      |  0.5626 |           12 |       0.4299 |                16 | 20.1542 |           15 |       8.3986 |                13 |      14    |
| ESM_PROTT5_concat_BPO      |  0.5607 |           16 |       0.4335 |                11 | 20.1929 |           16 |       8.4032 |                14 |      14.25 |
| ProstT5_Text_gated_BPO     |  0.5617 |           14 |       0.4322 |                13 | 20.2473 |           17 |       8.4471 |                16 |      15    |
| ProstT5_only_BPO           |  0.5612 |           15 |       0.4296 |                17 | 20.005  |           12 |       8.4777 |                19 |      15.75 |
| ESM_PROTT5_gated_BPO       |  0.5551 |           19 |       0.4203 |                19 | 20.4139 |           18 |       8.4547 |                17 |      18.25 |
| ESM_Text_BPO               |  0.5575 |           18 |       0.4304 |                15 | 20.5495 |           20 |       8.576  |                21 |      18.5  |
| Text_only_BPO              |  0.5453 |           20 |       0.4035 |                21 | 20.4962 |           19 |       8.6104 |                22 |      20.5  |
| ESM_PROTT5_contrastive_BPO |  0.5338 |           22 |       0.3914 |                23 | 20.9783 |           22 |       8.4349 |                15 |      20.5  |
| ESM_PROTT5_transformer_BPO |  0.5409 |           21 |       0.4    |                22 | 20.6735 |           21 |       8.5683 |                20 |      21    |
| rank_average_BPO           |  0.5156 |           23 |       0.4061 |                20 | 25.4336 |           24 |       9.9494 |                24 |      22.75 |
| stacking_BPO               |  0.4024 |           24 |       0.2312 |                24 | 24.1748 |           23 |       9.5708 |                23 |      23.5  |
| ESM_TEXT_gated_BPO         |  0.2346 |           25 |       0.0974 |                25 | 29.0398 |           25 |      10.0745 |                25 |      25    |

### CCO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| weighted_average_CCO copy  |  0.7045 |            1 |       0.534  |                 2 |  6.1053 |            1 |       2.1085 |                 2 |       1.5  |
| top_5_CCO                  |  0.7041 |            2 |       0.5367 |                 1 |  6.1479 |            3 |       2.1047 |                 1 |       1.75 |
| voting_t0.5_CCO            |  0.703  |            3 |       0.5324 |                 3 |  6.1365 |            2 |       2.1115 |                 3 |       2.75 |
| selective_fusion_CCO       |  0.7013 |            5 |       0.5313 |                 4 |  6.2055 |            5 |       2.126  |                 4 |       4.5  |
| weighted_average_CCO       |  0.7021 |            4 |       0.5313 |                 5 |  6.1971 |            4 |       2.126  |                 5 |       4.5  |
| ESM_PROTT5_concat_CCO      |  0.6966 |            7 |       0.525  |                 6 |  6.3039 |            7 |       2.1416 |                 6 |       6.5  |
| ESM_Text_gated_CCO         |  0.6943 |            9 |       0.5154 |                10 |  6.3018 |            6 |       2.1532 |                 7 |       8    |
| ESM_only_CCO               |  0.6981 |            6 |       0.5202 |                 9 |  6.3118 |            8 |       2.1613 |                10 |       8.25 |
| ProtT5_only_CCO            |  0.6953 |            8 |       0.522  |                 8 |  6.3553 |            9 |       2.1563 |                 8 |       8.25 |
| ProtT5_Text_gated_CCO      |  0.6942 |           10 |       0.5229 |                 7 |  6.3604 |           10 |       2.1618 |                11 |       9.5  |
| ESM_Text_transformer_CCO   |  0.6906 |           11 |       0.512  |                15 |  6.3653 |           11 |       2.161  |                 9 |      11.5  |
| ProstT5_only_CCO           |  0.69   |           13 |       0.5138 |                12 |  6.4748 |           15 |       2.1882 |                16 |      14    |
| ESM_Text_contrastive_CCO   |  0.6904 |           12 |       0.5082 |                20 |  6.3902 |           12 |       2.1655 |                12 |      14    |
| ESM_Text_CCO               |  0.6876 |           17 |       0.5123 |                14 |  6.468  |           14 |       2.1743 |                14 |      14.75 |
| ESM_Text_moe_CCO           |  0.6895 |           14 |       0.5095 |                19 |  6.4321 |           13 |       2.1723 |                13 |      14.75 |
| ESM_PROTT5_transformer_CCO |  0.6882 |           15 |       0.5117 |                16 |  6.4843 |           17 |       2.193  |                18 |      16.5  |
| ESM_ProstT5_gated_CCO      |  0.688  |           16 |       0.5114 |                17 |  6.5239 |           19 |       2.1766 |                15 |      16.75 |
| ProtT5_Text_moe_CCO        |  0.6846 |           20 |       0.5136 |                13 |  6.5499 |           21 |       2.1886 |                17 |      17.75 |
| rank_average_CCO           |  0.6814 |           21 |       0.5144 |                11 |  6.5307 |           20 |       2.1938 |                19 |      17.75 |
| ProstT5_Text_concat_CCO    |  0.6874 |           18 |       0.5041 |                22 |  6.4764 |           16 |       2.2058 |                20 |      19    |
| ProstT5_Text_gated_CCO     |  0.6868 |           19 |       0.5101 |                18 |  6.5185 |           18 |       2.2213 |                22 |      19.25 |
| ESM_PROTT5_gated_CCO       |  0.6798 |           22 |       0.5052 |                21 |  6.7036 |           22 |       2.2159 |                21 |      21.5  |
| Text_only_CCO              |  0.6718 |           23 |       0.4758 |                23 |  6.7744 |           23 |       2.2784 |                23 |      23    |
| stacking_CCO               |  0.6112 |           24 |       0.4031 |                24 |  7.4264 |           24 |       2.3518 |                24 |      24    |

### MFO

**All Models by Average Rank**

| Model                      |   F-max |   F-max Rank |   F-max (IA) |   F-max (IA) Rank |   S-min |   S-min Rank |   S-min (IA) |   S-min (IA) Rank |   Avg Rank |
|:---------------------------|--------:|-------------:|-------------:|------------------:|--------:|-------------:|-------------:|------------------:|-----------:|
| top_5_MFO                  |  0.6958 |            1 |       0.5766 |                 1 |  4.2653 |            2 |       2.0489 |                 1 |       1.25 |
| weighted_average_MFO copy  |  0.6886 |            5 |       0.5698 |                 4 |  4.2632 |            1 |       2.0583 |                 2 |       3    |
| voting_t0.5_MFO            |  0.6895 |            3 |       0.5676 |                 5 |  4.2874 |            3 |       2.0624 |                 3 |       3.5  |
| weighted_average_MFO       |  0.6916 |            2 |       0.5716 |                 3 |  4.3661 |            4 |       2.0901 |                 5 |       3.5  |
| selective_fusion_MFO       |  0.6889 |            4 |       0.5716 |                 2 |  4.3902 |            5 |       2.0901 |                 4 |       3.75 |
| ESM_only_MFO               |  0.6741 |            7 |       0.5478 |                 7 |  4.4445 |            6 |       2.1084 |                 6 |       6.5  |
| ESM_Text_moe_MFO           |  0.6644 |           10 |       0.5281 |                14 |  4.4472 |            7 |       2.1451 |                 7 |       9.5  |
| ProstT5_only_MFO           |  0.6688 |            9 |       0.5415 |                 9 |  4.5948 |           11 |       2.1895 |                11 |      10    |
| ProtT5_only_MFO            |  0.675  |            6 |       0.5566 |                 6 |  4.609  |           14 |       2.2071 |                15 |      10.25 |
| ESM_PROTT5_concat_MFO      |  0.6605 |           11 |       0.5339 |                10 |  4.582  |           10 |       2.1829 |                10 |      10.25 |
| ESM_PROTT5_moe_MFO         |  0.6699 |            8 |       0.5423 |                 8 |  4.6072 |           13 |       2.2121 |                17 |      11.5  |
| ESM_Text_transformer_MFO   |  0.658  |           14 |       0.5237 |                16 |  4.5085 |            8 |       2.1623 |                 8 |      11.5  |
| ProstT5_Text_concat_MFO    |  0.6587 |           13 |       0.5303 |                11 |  4.5952 |           12 |       2.1962 |                12 |      12    |
| ESM_Text_gated_MFO         |  0.6577 |           15 |       0.5197 |                20 |  4.5137 |            9 |       2.1742 |                 9 |      13.25 |
| ProtT5_Text_moe_MFO        |  0.6596 |           12 |       0.5293 |                13 |  4.6116 |           16 |       2.2119 |                16 |      14.25 |
| ESM_Text_MFO               |  0.6553 |           19 |       0.5227 |                17 |  4.6283 |           17 |       2.2032 |                14 |      16.75 |
| ESM_ProstT5_gated_MFO      |  0.6565 |           18 |       0.5239 |                15 |  4.6607 |           19 |       2.2188 |                18 |      17.5  |
| ESM_PROTT5_gated_MFO       |  0.6575 |           16 |       0.5301 |                12 |  4.7161 |           21 |       2.2557 |                22 |      17.75 |
| ESM_Text_contrastive_MFO   |  0.6525 |           21 |       0.5166 |                22 |  4.6113 |           15 |       2.2023 |                13 |      17.75 |
| ProtT5_Text_gated_MFO      |  0.6566 |           17 |       0.5182 |                21 |  4.6406 |           18 |       2.2302 |                20 |      19    |
| ProstT5_Text_gated_MFO     |  0.6531 |           20 |       0.5208 |                19 |  4.6781 |           20 |       2.2376 |                21 |      20    |
| rank_average_MFO           |  0.6467 |           22 |       0.5226 |                18 |  4.7619 |           22 |       2.2217 |                19 |      20.25 |
| ESM_PROTT5_transformer_MFO |  0.6453 |           23 |       0.5101 |                23 |  4.8264 |           23 |       2.305  |                23 |      23    |
| Text_only_MFO              |  0.6344 |           24 |       0.4933 |                24 |  4.9424 |           24 |       2.3472 |                24 |      24    |
| stacking_MFO               |  0.446  |           25 |       0.1778 |                25 |  6.4021 |           25 |       3.0737 |                25 |      25    |
| Structure_MFO              |  0.1408 |           26 |       0.0343 |                26 |  8.3822 |           26 |       3.2406 |                26 |      26    |

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
|      1 | voting_t0.5_BPO            |      0.5841 |      0.5792 |   0.589  |     1      |        0.43 |
|      2 | weighted_average_BPO copy  |      0.5825 |      0.5774 |   0.5877 |     1      |        0.47 |
|      3 | top_5_BPO                  |      0.5824 |      0.6016 |   0.5645 |     1      |        0.53 |
|      4 | weighted_average_BPO       |      0.5791 |      0.5855 |   0.5728 |     1      |        0.52 |
|      5 | selective_fusion_BPO       |      0.5787 |      0.585  |   0.5726 |     0.9987 |        0.52 |
|      6 | ESM_only_BPO               |      0.5688 |      0.6046 |   0.537  |     1      |        0.6  |
|      7 | ProtT5_Text_gated_BPO      |      0.5669 |      0.588  |   0.5472 |     1      |        0.59 |
|      8 | ESM_Text_moe_BPO           |      0.5666 |      0.6065 |   0.5316 |     1      |        0.62 |
|      9 | ESM_Text_contrastive_BPO   |      0.5658 |      0.5712 |   0.5605 |     1      |        0.56 |
|     10 | ESM_Text_gated_BPO         |      0.5648 |      0.5823 |   0.5483 |     1      |        0.58 |
|     11 | ProtT5_only_BPO            |      0.5641 |      0.579  |   0.5501 |     1      |        0.59 |
|     12 | ESM_ProstT5_gated_BPO      |      0.5626 |      0.6054 |   0.5255 |     1      |        0.59 |
|     13 | ProtT5_Text_moe_BPO        |      0.5624 |      0.576  |   0.5493 |     1      |        0.53 |
|     14 | ProstT5_Text_gated_BPO     |      0.5617 |      0.5991 |   0.5286 |     1      |        0.59 |
|     15 | ProstT5_only_BPO           |      0.5612 |      0.586  |   0.5384 |     1      |        0.56 |
|     16 | ESM_PROTT5_concat_BPO      |      0.5607 |      0.5799 |   0.5427 |     1      |        0.6  |
|     17 | ESM_Text_transformer_BPO   |      0.5601 |      0.5911 |   0.5322 |     1      |        0.6  |
|     18 | ESM_Text_BPO               |      0.5575 |      0.5909 |   0.5277 |     1      |        0.61 |
|     19 | ESM_PROTT5_gated_BPO       |      0.5551 |      0.5902 |   0.524  |     1      |        0.66 |
|     20 | Text_only_BPO              |      0.5453 |      0.5815 |   0.5134 |     1      |        0.58 |
|     21 | ESM_PROTT5_transformer_BPO |      0.5409 |      0.57   |   0.5146 |     1      |        0.54 |
|     22 | ESM_PROTT5_contrastive_BPO |      0.5338 |      0.559  |   0.5108 |     1      |        0.5  |
|     23 | rank_average_BPO           |      0.5156 |      0.4226 |   0.6611 |     1      |        0.99 |
|     24 | stacking_BPO               |      0.4024 |      0.5059 |   0.3341 |     1      |        0.36 |
|     25 | ESM_TEXT_gated_BPO         |      0.2346 |      0.1819 |   0.3302 |     0.9904 |        0.78 |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | voting_t0.5_BPO            |                 0.4617 |                 0.4358 |              0.4908 |                1      |        0.34 |
|      2 | weighted_average_BPO copy  |                 0.4598 |                 0.4571 |              0.4626 |                1      |        0.45 |
|      3 | top_5_BPO                  |                 0.4598 |                 0.4698 |              0.4502 |                0.9996 |        0.5  |
|      4 | selective_fusion_BPO       |                 0.4552 |                 0.4564 |              0.4541 |                0.9992 |        0.5  |
|      5 | weighted_average_BPO       |                 0.4552 |                 0.4564 |              0.4541 |                0.9992 |        0.5  |
|      6 | ESM_only_BPO               |                 0.4409 |                 0.4814 |              0.4068 |                0.9929 |        0.6  |
|      7 | ESM_Text_contrastive_BPO   |                 0.4404 |                 0.4487 |              0.4323 |                0.995  |        0.56 |
|      8 | ESM_Text_moe_BPO           |                 0.4396 |                 0.4519 |              0.428  |                0.9933 |        0.57 |
|      9 | ESM_Text_gated_BPO         |                 0.437  |                 0.4566 |              0.419  |                0.9929 |        0.58 |
|     10 | ProtT5_Text_gated_BPO      |                 0.4369 |                 0.4288 |              0.4452 |                0.9895 |        0.52 |
|     11 | ESM_PROTT5_concat_BPO      |                 0.4335 |                 0.4165 |              0.4518 |                0.9933 |        0.51 |
|     12 | ProtT5_only_BPO            |                 0.4333 |                 0.4231 |              0.4439 |                0.9958 |        0.54 |
|     13 | ProstT5_Text_gated_BPO     |                 0.4322 |                 0.4714 |              0.3991 |                0.9791 |        0.59 |
|     14 | ProtT5_Text_moe_BPO        |                 0.4313 |                 0.4141 |              0.45   |                0.9912 |        0.44 |
|     15 | ESM_Text_BPO               |                 0.4304 |                 0.4382 |              0.423  |                0.9887 |        0.57 |
|     16 | ESM_ProstT5_gated_BPO      |                 0.4299 |                 0.4624 |              0.4016 |                0.9858 |        0.56 |
|     17 | ProstT5_only_BPO           |                 0.4296 |                 0.4284 |              0.4309 |                0.9971 |        0.51 |
|     18 | ESM_Text_transformer_BPO   |                 0.4294 |                 0.4258 |              0.433  |                0.9941 |        0.53 |
|     19 | ESM_PROTT5_gated_BPO       |                 0.4203 |                 0.4132 |              0.4276 |                0.9787 |        0.57 |
|     20 | rank_average_BPO           |                 0.4061 |                 0.3317 |              0.5233 |                1      |        0.99 |
|     21 | Text_only_BPO              |                 0.4035 |                 0.4194 |              0.3887 |                0.9983 |        0.52 |
|     22 | ESM_PROTT5_transformer_BPO |                 0.4    |                 0.4326 |              0.372  |                0.9937 |        0.52 |
|     23 | ESM_PROTT5_contrastive_BPO |                 0.3914 |                 0.3877 |              0.3952 |                0.9933 |        0.4  |
|     24 | stacking_BPO               |                 0.2312 |                 0.1917 |              0.291  |                1      |        0.19 |
|     25 | ESM_TEXT_gated_BPO         |                 0.0974 |                 0.0735 |              0.1442 |                0.9904 |        0.78 |

## CCO Detailed Results

### All Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | weighted_average_CCO copy  |      0.7045 |      0.7513 |   0.6632 |     1      |        0.63 |
|      2 | top_5_CCO                  |      0.7041 |      0.7207 |   0.6883 |     1      |        0.58 |
|      3 | voting_t0.5_CCO            |      0.703  |      0.7401 |   0.6694 |     1      |        0.72 |
|      4 | weighted_average_CCO       |      0.7021 |      0.7633 |   0.6499 |     1      |        0.66 |
|      5 | selective_fusion_CCO       |      0.7013 |      0.7156 |   0.6875 |     0.9992 |        0.58 |
|      6 | ESM_only_CCO               |      0.6981 |      0.7481 |   0.6545 |     1      |        0.68 |
|      7 | ESM_PROTT5_concat_CCO      |      0.6966 |      0.7437 |   0.655  |     1      |        0.72 |
|      8 | ProtT5_only_CCO            |      0.6953 |      0.7415 |   0.6546 |     1      |        0.66 |
|      9 | ESM_Text_gated_CCO         |      0.6943 |      0.7398 |   0.6541 |     1      |        0.67 |
|     10 | ProtT5_Text_gated_CCO      |      0.6942 |      0.7148 |   0.6747 |     1      |        0.61 |
|     11 | ESM_Text_transformer_CCO   |      0.6906 |      0.7418 |   0.646  |     1      |        0.7  |
|     12 | ESM_Text_contrastive_CCO   |      0.6904 |      0.7438 |   0.6442 |     1      |        0.75 |
|     13 | ProstT5_only_CCO           |      0.69   |      0.7303 |   0.6538 |     1      |        0.65 |
|     14 | ESM_Text_moe_CCO           |      0.6895 |      0.7417 |   0.6441 |     1      |        0.65 |
|     15 | ESM_PROTT5_transformer_CCO |      0.6882 |      0.7079 |   0.6696 |     1      |        0.66 |
|     16 | ESM_ProstT5_gated_CCO      |      0.688  |      0.7218 |   0.6574 |     1      |        0.68 |
|     17 | ESM_Text_CCO               |      0.6876 |      0.7107 |   0.6658 |     1      |        0.65 |
|     18 | ProstT5_Text_concat_CCO    |      0.6874 |      0.7465 |   0.6371 |     1      |        0.71 |
|     19 | ProstT5_Text_gated_CCO     |      0.6868 |      0.7447 |   0.6373 |     1      |        0.68 |
|     20 | ProtT5_Text_moe_CCO        |      0.6846 |      0.7142 |   0.6574 |     1      |        0.65 |
|     21 | rank_average_CCO           |      0.6814 |      0.6574 |   0.7072 |     1      |        0.98 |
|     22 | ESM_PROTT5_gated_CCO       |      0.6798 |      0.7222 |   0.642  |     1      |        0.75 |
|     23 | Text_only_CCO              |      0.6718 |      0.724  |   0.6266 |     1      |        0.64 |
|     24 | stacking_CCO               |      0.6112 |      0.5819 |   0.6437 |     1      |        0.37 |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | top_5_CCO                  |                 0.5367 |                 0.5712 |              0.5061 |                0.9976 |        0.54 |
|      2 | weighted_average_CCO copy  |                 0.534  |                 0.5457 |              0.5227 |                0.9992 |        0.5  |
|      3 | voting_t0.5_CCO            |                 0.5324 |                 0.5605 |              0.5071 |                0.9984 |        0.53 |
|      4 | selective_fusion_CCO       |                 0.5313 |                 0.5769 |              0.4925 |                0.9992 |        0.56 |
|      5 | weighted_average_CCO       |                 0.5313 |                 0.5769 |              0.4925 |                0.9992 |        0.56 |
|      6 | ESM_PROTT5_concat_CCO      |                 0.525  |                 0.57   |              0.4865 |                0.9905 |        0.62 |
|      7 | ProtT5_Text_gated_CCO      |                 0.5229 |                 0.5558 |              0.4937 |                0.9929 |        0.54 |
|      8 | ProtT5_only_CCO            |                 0.522  |                 0.5613 |              0.4877 |                0.9913 |        0.59 |
|      9 | ESM_only_CCO               |                 0.5202 |                 0.5658 |              0.4813 |                0.9968 |        0.61 |
|     10 | ESM_Text_gated_CCO         |                 0.5154 |                 0.5919 |              0.4564 |                0.9866 |        0.64 |
|     11 | rank_average_CCO           |                 0.5144 |                 0.5321 |              0.4978 |                1      |        0.98 |
|     12 | ProstT5_only_CCO           |                 0.5138 |                 0.5512 |              0.4812 |                0.9953 |        0.58 |
|     13 | ProtT5_Text_moe_CCO        |                 0.5136 |                 0.534  |              0.4947 |                0.9874 |        0.53 |
|     14 | ESM_Text_CCO               |                 0.5123 |                 0.5457 |              0.4827 |                0.9937 |        0.6  |
|     15 | ESM_Text_transformer_CCO   |                 0.512  |                 0.5263 |              0.4984 |                0.9976 |        0.57 |
|     16 | ESM_PROTT5_transformer_CCO |                 0.5117 |                 0.534  |              0.4912 |                0.9953 |        0.58 |
|     17 | ESM_ProstT5_gated_CCO      |                 0.5114 |                 0.5515 |              0.4766 |                0.9897 |        0.61 |
|     18 | ProstT5_Text_gated_CCO     |                 0.5101 |                 0.5256 |              0.4956 |                0.9968 |        0.51 |
|     19 | ESM_Text_moe_CCO           |                 0.5095 |                 0.5651 |              0.4639 |                0.9889 |        0.59 |
|     20 | ESM_Text_contrastive_CCO   |                 0.5082 |                 0.5682 |              0.4596 |                0.985  |        0.69 |
|     21 | ESM_PROTT5_gated_CCO       |                 0.5052 |                 0.5188 |              0.4923 |                0.9945 |        0.6  |
|     22 | ProstT5_Text_concat_CCO    |                 0.5041 |                 0.5131 |              0.4955 |                0.996  |        0.52 |
|     23 | Text_only_CCO              |                 0.4758 |                 0.5362 |              0.4276 |                0.9968 |        0.58 |
|     24 | stacking_CCO               |                 0.4031 |                 0.3914 |              0.4155 |                1      |        0.34 |

## MFO Detailed Results

### All Models by F-measure

|   Rank | Model                      |   F-measure |   Precision |   Recall |   Coverage |   Threshold |
|-------:|:---------------------------|------------:|------------:|---------:|-----------:|------------:|
|      1 | top_5_MFO                  |      0.6958 |      0.7527 |   0.6469 |     1      |        0.54 |
|      2 | weighted_average_MFO       |      0.6916 |      0.7261 |   0.6602 |     1      |        0.53 |
|      3 | voting_t0.5_MFO            |      0.6895 |      0.7348 |   0.6495 |     1      |        0.48 |
|      4 | selective_fusion_MFO       |      0.6889 |      0.7229 |   0.6579 |     0.9886 |        0.53 |
|      5 | weighted_average_MFO copy  |      0.6886 |      0.7076 |   0.6706 |     1      |        0.45 |
|      6 | ProtT5_only_MFO            |      0.675  |      0.6943 |   0.6567 |     1      |        0.58 |
|      7 | ESM_only_MFO               |      0.6741 |      0.7369 |   0.6212 |     1      |        0.61 |
|      8 | ESM_PROTT5_moe_MFO         |      0.6699 |      0.6905 |   0.6506 |     1      |        0.48 |
|      9 | ProstT5_only_MFO           |      0.6688 |      0.6962 |   0.6434 |     1      |        0.57 |
|     10 | ESM_Text_moe_MFO           |      0.6644 |      0.689  |   0.6415 |     1      |        0.53 |
|     11 | ESM_PROTT5_concat_MFO      |      0.6605 |      0.7096 |   0.6178 |     1      |        0.61 |
|     12 | ProtT5_Text_moe_MFO        |      0.6596 |      0.6993 |   0.6241 |     1      |        0.53 |
|     13 | ProstT5_Text_concat_MFO    |      0.6587 |      0.6762 |   0.6421 |     1      |        0.44 |
|     14 | ESM_Text_transformer_MFO   |      0.658  |      0.7217 |   0.6047 |     1      |        0.6  |
|     15 | ESM_Text_gated_MFO         |      0.6577 |      0.7088 |   0.6134 |     1      |        0.58 |
|     16 | ESM_PROTT5_gated_MFO       |      0.6575 |      0.6985 |   0.6211 |     1      |        0.57 |
|     17 | ProtT5_Text_gated_MFO      |      0.6566 |      0.7197 |   0.6036 |     1      |        0.63 |
|     18 | ESM_ProstT5_gated_MFO      |      0.6565 |      0.7019 |   0.6166 |     1      |        0.57 |
|     19 | ESM_Text_MFO               |      0.6553 |      0.6802 |   0.6322 |     1      |        0.55 |
|     20 | ProstT5_Text_gated_MFO     |      0.6531 |      0.6812 |   0.6271 |     1      |        0.5  |
|     21 | ESM_Text_contrastive_MFO   |      0.6525 |      0.6694 |   0.6363 |     1      |        0.52 |
|     22 | rank_average_MFO           |      0.6467 |      0.6607 |   0.6333 |     1      |        0.99 |
|     23 | ESM_PROTT5_transformer_MFO |      0.6453 |      0.6817 |   0.6125 |     1      |        0.53 |
|     24 | Text_only_MFO              |      0.6344 |      0.6597 |   0.611  |     1      |        0.49 |
|     25 | stacking_MFO               |      0.446  |      0.5679 |   0.3672 |     1      |        0.25 |
|     26 | Structure_MFO              |      0.1408 |      0.0903 |   0.3189 |     0.7142 |        0.52 |

### All Models by F-measure (IA-weighted)

|   Rank | Model                      |   F-measure (Weighted) |   Precision (Weighted) |   Recall (Weighted) |   Coverage (Weighted) |   Threshold |
|-------:|:---------------------------|-----------------------:|-----------------------:|--------------------:|----------------------:|------------:|
|      1 | top_5_MFO                  |                 0.5766 |                 0.5886 |              0.5651 |                1      |        0.38 |
|      2 | selective_fusion_MFO       |                 0.5716 |                 0.6478 |              0.5114 |                0.9886 |        0.53 |
|      3 | weighted_average_MFO       |                 0.5716 |                 0.6478 |              0.5114 |                0.9886 |        0.53 |
|      4 | weighted_average_MFO copy  |                 0.5698 |                 0.6081 |              0.5361 |                0.9991 |        0.41 |
|      5 | voting_t0.5_MFO            |                 0.5676 |                 0.6019 |              0.5369 |                0.9965 |        0.35 |
|      6 | ProtT5_only_MFO            |                 0.5566 |                 0.5959 |              0.5222 |                0.9903 |        0.55 |
|      7 | ESM_only_MFO               |                 0.5478 |                 0.5877 |              0.513  |                0.9912 |        0.5  |
|      8 | ESM_PROTT5_moe_MFO         |                 0.5423 |                 0.5904 |              0.5014 |                0.9894 |        0.47 |
|      9 | ProstT5_only_MFO           |                 0.5415 |                 0.6033 |              0.4912 |                0.9824 |        0.56 |
|     10 | ESM_PROTT5_concat_MFO      |                 0.5339 |                 0.5576 |              0.5121 |                0.9886 |        0.45 |
|     11 | ProstT5_Text_concat_MFO    |                 0.5303 |                 0.551  |              0.5112 |                0.993  |        0.35 |
|     12 | ESM_PROTT5_gated_MFO       |                 0.5301 |                 0.531  |              0.5292 |                0.9991 |        0.37 |
|     13 | ProtT5_Text_moe_MFO        |                 0.5293 |                 0.5115 |              0.5485 |                0.9956 |        0.29 |
|     14 | ESM_Text_moe_MFO           |                 0.5281 |                 0.5994 |              0.472  |                0.9903 |        0.53 |
|     15 | ESM_ProstT5_gated_MFO      |                 0.5239 |                 0.5673 |              0.4867 |                0.9938 |        0.46 |
|     16 | ESM_Text_transformer_MFO   |                 0.5237 |                 0.571  |              0.4837 |                0.9894 |        0.48 |
|     17 | ESM_Text_MFO               |                 0.5227 |                 0.562  |              0.4885 |                0.9921 |        0.5  |
|     18 | rank_average_MFO           |                 0.5226 |                 0.5748 |              0.4791 |                0.9974 |        0.99 |
|     19 | ProstT5_Text_gated_MFO     |                 0.5208 |                 0.5551 |              0.4905 |                0.9965 |        0.42 |
|     20 | ESM_Text_gated_MFO         |                 0.5197 |                 0.5341 |              0.5061 |                0.9974 |        0.42 |
|     21 | ProtT5_Text_gated_MFO      |                 0.5182 |                 0.5423 |              0.4961 |                0.9947 |        0.43 |
|     22 | ESM_Text_contrastive_MFO   |                 0.5166 |                 0.5279 |              0.5058 |                0.9974 |        0.43 |
|     23 | ESM_PROTT5_transformer_MFO |                 0.5101 |                 0.5183 |              0.5021 |                0.9947 |        0.36 |
|     24 | Text_only_MFO              |                 0.4933 |                 0.5552 |              0.4439 |                0.9903 |        0.48 |
|     25 | stacking_MFO               |                 0.1778 |                 0.2159 |              0.1511 |                1      |        0.13 |
|     26 | Structure_MFO              |                 0.0343 |                 0.02   |              0.1221 |                0.7142 |        0.52 |

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
