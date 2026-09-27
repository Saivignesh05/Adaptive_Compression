# Model Card: vgg19_aqp_20260927_193442

## Executive Summary
- **Architecture:** VGG19
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 5.47% |
| **Compressed Accuracy** | 10.16% |
| **Accuracy Drop Margin** | -4.69% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **0.00 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **100.00%** |

## Layer Breakdown
| Layer Name   |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|--------------|--------------|-----------------|----------------------|----------------|
| features.0   |       0.2501 | 1-bit           |                    3 | 100.0%         |
| features.3   |       0.1427 | 1-bit           |                    3 | 100.0%         |
| features.7   |       0.1432 | 1-bit           |                    3 | 100.0%         |
| features.10  |       0.1409 | 1-bit           |                    3 | 100.0%         |
| features.14  |       0.1429 | 1-bit           |                    3 | 100.0%         |
| features.17  |       0.1454 | 1-bit           |                    3 | 100.0%         |
| features.20  |       0.1454 | 1-bit           |                    3 | 100.0%         |
| features.23  |       0.1458 | 1-bit           |                    3 | 100.0%         |
| features.27  |       0.1528 | 1-bit           |                    3 | 100.0%         |
| features.30  |       0.1668 | 1-bit           |                    3 | 100.0%         |
| features.33  |       0.1669 | 1-bit           |                    3 | 100.0%         |
| features.36  |       0.166  | 1-bit           |                    3 | 100.0%         |
| features.40  |       0.1666 | 1-bit           |                    3 | 100.0%         |
| features.43  |       0.1664 | 1-bit           |                    3 | 100.0%         |
| features.46  |       0.1671 | 1-bit           |                    3 | 100.0%         |
| features.49  |       0.166  | 1-bit           |                    3 | 100.0%         |
| classifier.0 |       0.1466 | 1-bit           |                    3 | 100.0%         |
| classifier.3 |       0.1466 | 1-bit           |                    3 | 100.0%         |
| classifier.6 |       0.143  | 1-bit           |                    3 | 100.0%         |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
