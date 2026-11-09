# Model Card: resnet18_aqp_20261109_203523

## Executive Summary
- **Architecture:** RESNET18
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 94.06% |
| **Compressed Accuracy** | 94.05% |
| **Accuracy Drop Margin** | 0.01% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **2.15 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **59.77%** |

## Layer Breakdown
| Layer Name          |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|---------------------|--------------|-----------------|----------------------|----------------|
| conv1               |       0.2501 | 7-bit           |                 0    | 0.0%           |
| layer1.0.conv1      |       0.1427 | 5-bit           |                 0.25 | 21.3%          |
| layer1.0.conv2      |       0.1431 | 6-bit           |                 0.25 | 20.7%          |
| layer1.1.conv1      |       0.1432 | 6-bit           |                 0.25 | 21.8%          |
| layer1.1.conv2      |       0.1425 | 5-bit           |                 0.25 | 20.8%          |
| layer2.0.conv1      |       0.1413 | 6-bit           |                 0    | 0.0%           |
| layer2.0.conv2      |       0.1421 | 5-bit           |                 0.5  | 39.1%          |
| layer2.0.shortcut.0 |       0.1546 | 7-bit           |                 0    | 0.0%           |
| layer2.1.conv1      |       0.142  | 5-bit           |                 0.25 | 19.9%          |
| layer2.1.conv2      |       0.1416 | 4-bit           |                 0.25 | 20.1%          |
| layer3.0.conv1      |       0.144  | 6-bit           |                 0.25 | 20.0%          |
| layer3.0.conv2      |       0.1502 | 6-bit           |                 0.25 | 20.1%          |
| layer3.0.shortcut.0 |       0.1445 | 5-bit           |                 0    | 0.0%           |
| layer3.1.conv1      |       0.1501 | 6-bit           |                 0    | 0.0%           |
| layer3.1.conv2      |       0.1498 | 5-bit           |                 0.25 | 19.9%          |
| layer4.0.conv1      |       0.1624 | 6-bit           |                 0.25 | 19.6%          |
| layer4.0.conv2      |       0.1886 | 3-bit           |                 1.25 | 79.4%          |
| layer4.0.shortcut.0 |       0.1412 | 8-bit           |                 0.25 | 17.9%          |
| layer4.1.conv1      |       0.1885 | 3-bit           |                 1.25 | 79.4%          |
| layer4.1.conv2      |       0.1886 | 3-bit           |                 1.25 | 78.3%          |
| linear              |       0.2432 | 1-bit           |                 1.25 | 80.2%          |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
