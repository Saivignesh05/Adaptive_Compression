# Model Card: vgg19_aqp_20260928_194345

## Executive Summary
- **Architecture:** VGG19
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 93.91% |
| **Compressed Accuracy** | 93.89% |
| **Accuracy Drop Margin** | 0.02% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **6.27 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **0.00%** |

## Layer Breakdown
| Layer Name   |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|--------------|--------------|-----------------|----------------------|----------------|
| features.0   |       0.2503 | 8-bit           |                    0 | 0.0%           |
| features.3   |       0.1457 | 8-bit           |                    0 | 0.0%           |
| features.7   |       0.1454 | 8-bit           |                    0 | 0.0%           |
| features.10  |       0.1436 | 8-bit           |                    0 | 0.0%           |
| features.14  |       0.1444 | 8-bit           |                    0 | 0.0%           |
| features.17  |       0.1457 | 8-bit           |                    0 | 0.0%           |
| features.20  |       0.1448 | 7-bit           |                    0 | 0.0%           |
| features.23  |       0.1441 | 6-bit           |                    0 | 0.0%           |
| features.27  |       0.1506 | 8-bit           |                    0 | 0.0%           |
| features.30  |       0.1651 | 6-bit           |                    0 | 0.0%           |
| features.33  |       0.166  | 7-bit           |                    0 | 0.0%           |
| features.36  |       0.1661 | 6-bit           |                    0 | 0.0%           |
| features.40  |       0.166  | 7-bit           |                    0 | 0.0%           |
| features.43  |       0.1675 | 4-bit           |                    0 | 0.0%           |
| features.46  |       0.167  | 7-bit           |                    0 | 0.0%           |
| features.49  |       0.171  | 5-bit           |                    0 | 0.0%           |
| classifier.0 |       0.2465 | 5-bit           |                    0 | 0.0%           |
| classifier.3 |       0.1395 | 8-bit           |                    0 | 0.0%           |
| classifier.6 |       0.1773 | 4-bit           |                    0 | 0.0%           |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
