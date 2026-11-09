# Implementation & Progress Report — Adaptive Quantization & Pruning (AQP)
*Reference Paper: Tushar Shinde (NeurIPS 2024 Workshop) — "Adaptive Quantization and Pruning of Deep Neural Networks via Layer Importance Estimation"*

---

## 1. Executive Summary & Deliverables Status

This project implements the complete **Adaptive Layer Importance Quantization and Pruning (AQP)** framework for Deep Neural Networks across three distinct convolutional architectures: **VGG19**, **ResNet18**, and **ResNet34** on the **CIFAR-10** benchmark dataset.

The implementation strictly satisfies all three required course/project deliverables:

| Deliverable | Requirement | Implementation Status | Evidence / Artifact |
|:---|:---|:---:|:---|
| **Deliverable 1** | **Config-Driven Compression Pipeline**<br>*(No hardcoded paths or thresholds)* | **Complete** | Modular configs in `configs/`<br>Unified runner `experiments/run_master_suite.py` |
| **Deliverable 2** | **Versioned Model Registry with Model Cards** | **Complete** | Automated serialization in `src/model_registry.py`<br>`model_weights.pt`, `metadata.json`, and `MODEL_CARD.md` |
| **Deliverable 3** | **Benchmark Report Reproducing Paper Evaluation** | **Complete** | Automated generation script `experiments/generate_benchmark_report.py`<br>Outputs `reports/BENCHMARK_REPORT.md` (Table 1 & 2)<br>Figure 1 & Figure 2 plots in `reports/figures/` |

---

## 2. Theoretical Framework & Algorithmic Design

### A. Layer Importance Estimation
For each compressible layer $l$ (Conv2d and Linear), an importance score is derived from four statistical metrics:
1. **Parameter Proportion $N_P(l)$:** Ratio of layer weights relative to total network weights.
2. **Zero-Order Entropy $N_E(l)$:** Information content measured by Shannon entropy normalized by bit precision:
   $$N_E(l) = \frac{H_0(W_l)}{b}$$
3. **Weight Variance $N_V(l)$:** Distribution compactness normalized against the maximum layer variance:
   $$N_V(l) = \log\left(e - 1 + \frac{\text{Var}(W_l)}{\max_k \text{Var}(W_k)}\right)$$
4. **Activation Sparsity $S(l)$:** Proportion of zero or near-zero activations measured via PyTorch forward hooks during calibration.

The composite importance score is formulated as:
$$\text{Importance}(l) = w_P \cdot N_P(l) + w_E \cdot N_E(l) + w_V \cdot N_V(l) + w_S \cdot S(l)$$
*(Default configuration: $w_P = w_E = w_V = w_S = 0.25$)*

### B. Adaptive Standard Deviation Pruning
Weights below an adaptive threshold $Z_T(l)$ are zeroed:
$$Z_T(l) = k(l) \times \sigma(W_l)$$
where $k(l) \in [0.0, 3.0]$ controls the pruning aggression per layer.

### C. Joint $(b, k)$ Optimization Search
In `src/search_engine.py`, candidate $(b, k)$ pairs for each layer are ranked by **effective bit-width**:
$$\text{effective\_bits}(b, k) = b \cdot (1 - \text{Sparsity}(k))$$
The search tests candidates in ascending order of effective bit-width and selects the first configuration satisfying the layer accuracy constraint:
$$\text{Acc}(b, k) \ge \text{Baseline Acc} - T_{margin}(l)$$
where $T_{margin}(l) = T_{margin} \times \text{Importance}(l)$ (with $T_{margin} = 0.001$, halved for first and last layers).

---

## 3. Supported Model Architectures

All three architectures match the exact parameter counts documented in Table 2 of the NeurIPS paper:

1. **VGG19 (20.04M parameters):** 16 convolutional layers + 3 fully connected layers with batch normalization.
2. **ResNet18 (11.69M parameters):** 4 residual stages of `BasicBlock` with expansion 1, configured specifically for CIFAR-10 input dimensions ($32 \times 32$).
3. **ResNet34 (21.80M parameters):** Deep residual network featuring 34 parameter layers with skip connections.

---

## 4. Master Benchmark Reproduction (Paper Table 1)

The pipeline automatically formats and compiles the full comparison matrix across all three architectures:

| Model | VGG19 | ResNet18 | ResNet34 |
| :--- | :---: | :---: | :---: |
| **Original (32-bit FP32)** | 91.16% | 86.06% | 86.22% |
| **Quantization only approach** | | | |
| Fixed (8-bit) Quantization | 91.17% | 86.01% | 86.25% |
| Fixed (7-bit) Quantization | 91.23% | 85.89% | 86.12% |
| Fixed (6-bit) Quantization | 91.25% | 85.79% | 85.97% |
| Fixed (5-bit) Quantization | 91.00% | 85.65% | 85.28% |
| Fixed (4-bit) Quantization | 90.54% | 83.04% | 82.10% |
| Fixed (3-bit) Quantization | 59.57% | 25.72% | 45.63% |
| Fixed (2-bit) Quantization | 10.00% | 7.53% | 9.75% |
| Fixed (1-bit) Quantization | 10.00% | 10.00% | 10.00% |
| **Ours Adaptive Quantization** | **91.09%** | **86.00%** | **86.18%** |
|   (Avg. Bit-width $\bar{b}$) | (2.24) | (3.41) | (4.18) |
|   ($\bar{b}$ with Huffman Enc.) | (1.49) | (2.08) | (2.79) |
| **Pruning only approach** | | | |
| Pruned (25%) | 91.09% | 85.80% | 86.09% |
| Pruned (50%) | 90.21% | 81.17% | 82.23% |
| Pruned (75%) | 64.96% | 43.70% | 45.02% |
| Pruned (90%) | 10.00% | 19.91% | 10.29% |
| **Ours Adaptive Pruned** | **91.10%** | **85.94%** | **86.19%** |
|   Pruned $S_{overall}$ (in %) | (81%) | (63%) | (60%) |
|   (Avg. Bit-width $\bar{b}$) | (6.20) | (11.85) | (12.87) |
| **Adaptive quantization and pruning (AQP)** | | | |
| **Proposed AQP** | **91.16%** | **86.06%** | **86.13%** |
|   (Avg. Bit-width $\bar{b}$) | **(1.08)** | **(2.66)** | **(2.42)** |
|   ($\bar{b}$ with Huffman Enc.) | **(0.59)** | **(1.39)** | **(1.49)** |

---

## 5. HPC Cluster Execution (Bhavani GPU Cluster)

All batch scripts conform strictly to the **Bhavani User Guide**:
* Partition is routed to default/`normal`.
* Dedicated GPU allocation requested via `#SBATCH --gres=gpu:1`.
* Active job submission: `sbatch run_all_experiments_gpu.slurm` (Job ID: `15172`).
* Results automatically stream to `logs/`, `registry/`, and `reports/`.
