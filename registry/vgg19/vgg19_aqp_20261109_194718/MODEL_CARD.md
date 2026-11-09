# Model Card: vgg19_aqp_20261109_194718

## Executive Summary
- **Architecture:** VGG19
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 92.62% |
| **Compressed Accuracy** | 92.61% |
| **Accuracy Drop Margin** | 0.01% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **1.78 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **59.42%** |

## Layer Breakdown
| Layer Name   |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|--------------|--------------|-----------------|----------------------|----------------|
| features.0   |       0.2501 | 8-bit           |                 0    | 0.0%           |
| features.3   |       0.1454 | 7-bit           |                 0.25 | 23.2%          |
| features.7   |       0.1445 | 6-bit           |                 0    | 0.0%           |
| features.10  |       0.1427 | 7-bit           |                 0    | 0.0%           |
| features.14  |       0.1431 | 6-bit           |                 0    | 0.0%           |
| features.17  |       0.1448 | 7-bit           |                 0.25 | 20.1%          |
| features.20  |       0.144  | 6-bit           |                 0.75 | 55.4%          |
| features.23  |       0.1435 | 6-bit           |                 0.25 | 20.4%          |
| features.27  |       0.1501 | 4-bit           |                 0.5  | 38.5%          |
| features.30  |       0.1643 | 4-bit           |                 0.5  | 36.9%          |
| features.33  |       0.1643 | 4-bit           |                 0.75 | 51.7%          |
| features.36  |       0.1643 | 3-bit           |                 1.25 | 78.4%          |
| features.40  |       0.1642 | 3-bit           |                 1.25 | 78.2%          |
| features.43  |       0.1643 | 4-bit           |                 0.75 | 48.5%          |
| features.46  |       0.1643 | 3-bit           |                 1.5  | 90.0%          |
| features.49  |       0.1645 | 4-bit           |                 1    | 67.1%          |
| classifier.0 |       0.1395 | 3-bit           |                 1    | 74.1%          |
| classifier.3 |       0.1396 | 3-bit           |                 1.25 | 82.9%          |
| classifier.6 |       0.1681 | 4-bit           |                 0    | 0.0%           |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
