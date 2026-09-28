# Model Card: vgg19_aqp_20260928_191439

## Executive Summary
- **Architecture:** VGG19
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 14.84% |
| **Compressed Accuracy** | 14.84% |
| **Accuracy Drop Margin** | 0.00% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **4.15 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **0.00%** |

## Layer Breakdown
| Layer Name   |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|--------------|--------------|-----------------|----------------------|----------------|
| features.0   |       0.2505 | 1-bit           |                    0 | 0.0%           |
| features.3   |       0.1457 | 1-bit           |                    0 | 0.0%           |
| features.7   |       0.1454 | 1-bit           |                    0 | 0.0%           |
| features.10  |       0.1436 | 3-bit           |                    0 | 0.0%           |
| features.14  |       0.1445 | 3-bit           |                    0 | 0.0%           |
| features.17  |       0.1457 | 1-bit           |                    0 | 0.0%           |
| features.20  |       0.1448 | 2-bit           |                    0 | 0.0%           |
| features.23  |       0.1441 | 5-bit           |                    0 | 0.0%           |
| features.27  |       0.1508 | 2-bit           |                    0 | 0.0%           |
| features.30  |       0.1659 | 6-bit           |                    0 | 0.0%           |
| features.33  |       0.1673 | 4-bit           |                    0 | 0.0%           |
| features.36  |       0.1673 | 5-bit           |                    0 | 0.0%           |
| features.40  |       0.167  | 4-bit           |                    0 | 0.0%           |
| features.43  |       0.1689 | 3-bit           |                    0 | 0.0%           |
| features.46  |       0.1679 | 4-bit           |                    0 | 0.0%           |
| features.49  |       0.1706 | 6-bit           |                    0 | 0.0%           |
| classifier.0 |       0.2468 | 4-bit           |                    0 | 0.0%           |
| classifier.3 |       0.1395 | 1-bit           |                    0 | 0.0%           |
| classifier.6 |       0.1773 | 4-bit           |                    0 | 0.0%           |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
