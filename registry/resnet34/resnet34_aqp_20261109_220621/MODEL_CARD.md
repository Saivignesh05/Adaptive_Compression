# Model Card: resnet34_aqp_20261109_220621

## Executive Summary
- **Architecture:** RESNET34
- **Dataset:** CIFAR-10
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | 94.18% |
| **Compressed Accuracy** | 94.18% |
| **Accuracy Drop Margin** | 0.00% |
| **Weighted Avg Bit-Width (\(\bar{b}\))** | **3.91 bits** |
| **Huffman Avg Bit-Width** | **1.00 bits** |
| **Overall Weight Sparsity (\(S_{overall}\))** | **34.88%** |

## Layer Breakdown
| Layer Name          |   Importance | Bit Precision   |   Pruning Factor (k) | Sparsity (%)   |
|---------------------|--------------|-----------------|----------------------|----------------|
| conv1               |       0.2501 | 8-bit           |                 0    | 0.0%           |
| layer1.0.conv1      |       0.1419 | 7-bit           |                 0.5  | 41.8%          |
| layer1.0.conv2      |       0.1421 | 7-bit           |                 0.25 | 20.9%          |
| layer1.1.conv1      |       0.1411 | 4-bit           |                 0.25 | 21.1%          |
| layer1.1.conv2      |       0.141  | 4-bit           |                 0.25 | 20.4%          |
| layer1.2.conv1      |       0.1402 | 6-bit           |                 0.25 | 20.7%          |
| layer1.2.conv2      |       0.1399 | 6-bit           |                 0.25 | 20.2%          |
| layer2.0.conv1      |       0.1401 | 5-bit           |                 0.25 | 20.4%          |
| layer2.0.conv2      |       0.14   | 4-bit           |                 0.25 | 20.5%          |
| layer2.0.shortcut.0 |       0.1536 | 8-bit           |                 0    | 0.0%           |
| layer2.1.conv1      |       0.1396 | 5-bit           |                 0.25 | 19.8%          |
| layer2.1.conv2      |       0.1396 | 6-bit           |                 0.25 | 19.8%          |
| layer2.2.conv1      |       0.139  | 6-bit           |                 0.25 | 19.9%          |
| layer2.2.conv2      |       0.1391 | 6-bit           |                 0.5  | 38.3%          |
| layer2.3.conv1      |       0.1386 | 5-bit           |                 0.5  | 38.2%          |
| layer2.3.conv2      |       0.1387 | 5-bit           |                 0.25 | 19.5%          |
| layer3.0.conv1      |       0.1406 | 7-bit           |                 0    | 0.0%           |
| layer3.0.conv2      |       0.1437 | 8-bit           |                 0    | 0.0%           |
| layer3.0.shortcut.0 |       0.144  | 8-bit           |                 0    | 0.0%           |
| layer3.1.conv1      |       0.1436 | 8-bit           |                 0    | 0.0%           |
| layer3.1.conv2      |       0.1435 | 8-bit           |                 0    | 0.0%           |
| layer3.2.conv1      |       0.1431 | 8-bit           |                 0    | 0.0%           |
| layer3.2.conv2      |       0.1431 | 8-bit           |                 0    | 0.0%           |
| layer3.3.conv1      |       0.1429 | 4-bit           |                 0.25 | 19.5%          |
| layer3.3.conv2      |       0.1429 | 5-bit           |                 0.5  | 37.4%          |
| layer3.4.conv1      |       0.1427 | 4-bit           |                 1    | 67.4%          |
| layer3.4.conv2      |       0.1428 | 5-bit           |                 0.25 | 18.4%          |
| layer3.5.conv1      |       0.1427 | 3-bit           |                 0.75 | 52.5%          |
| layer3.5.conv2      |       0.1427 | 4-bit           |                 0.5  | 35.0%          |
| layer4.0.conv1      |       0.1496 | 6-bit           |                 0.75 | 53.0%          |
| layer4.0.conv2      |       0.1635 | 8-bit           |                 0    | 0.0%           |
| layer4.0.shortcut.0 |       0.1391 | 5-bit           |                 0.25 | 17.0%          |
| layer4.1.conv1      |       0.1633 | 4-bit           |                 0.75 | 50.8%          |
| layer4.1.conv2      |       0.1635 | 3-bit           |                 1.5  | 87.3%          |
| layer4.2.conv1      |       0.1633 | 5-bit           |                 0.5  | 33.0%          |
| layer4.2.conv2      |       0.1634 | 4-bit           |                 0.5  | 32.4%          |
| linear              |       0.2042 | 8-bit           |                 0    | 0.0%           |

---
*Generated automatically by Adaptive Compression ModelRegistry.*
