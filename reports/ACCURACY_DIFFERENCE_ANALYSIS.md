# Why Our Accuracy Is Higher Than the Paper's

*Reference: Shinde, "Adaptive Quantization and Pruning of Deep Neural Networks via Layer Importance Estimation" (NeurIPS 2024 Workshop)*
*Runs analysed: `registry/*/full_suite_results_20261109_*.json` (H100, Madhava cluster)*

## Summary

The higher accuracy does **not** come from the compression method. It comes from the **FP32 baseline models**, which are much stronger than the paper's. AQP only allows a drop of about 0.025% from its starting model, so whatever baseline it starts from carries straight through to the compressed results.

| Model | Paper FP32 | Our FP32 | Gap | Paper AQP | Our AQP |
|---|---|---|---|---|---|
| VGG19 | 91.16% | 92.62% | +1.46 | 91.16% | 92.61% |
| ResNet18 | 86.06% | 94.06% | +8.00 | 86.06% | 94.05% |
| ResNet34 | 86.22% | 94.18% | +7.96 | 86.13% | 94.18% |

In both the paper and our runs, AQP accuracy is almost equal to the FP32 baseline. Only the baseline differs.

All training hyperparameters match the paper: batch size 128, 100 epochs, SGD with learning rate 0.02 decayed ×0.5 every 20 epochs, momentum 0.9, weight decay 5e-4, cross-entropy loss, and training from scratch. The differences below explain the gap.

---

## 1. ResNet18/34 use a different architecture (main cause, about +8%)

| | Paper | Ours (`src/models/cifar_resnet.py`) |
|---|---|---|
| Parameters | ResNet18: **11.69M**, ResNet34: **21.80M** | ResNet18: **~11.17M**, ResNet34: **~21.28M** |
| First conv layer | 7×7, stride 2 | 3×3, stride 1 |
| Max-pool after first conv | Yes | No |
| Feature map at layer 4 for a 32×32 input | 1×1 | 4×4 |
| Typical CIFAR-10 accuracy | ~85–87% | ~93–95% |

The paper's parameter counts, 11.69M and 21.80M, are exactly those of torchvision's **ImageNet** ResNet18/34 (11,689,512 and 21,797,672). That model shrinks a 32×32 CIFAR image to 8×8 in its first two operations, which throws away spatial detail and caps accuracy at around 86%. That matches the paper's numbers.

Our code uses the **CIFAR-adapted** ResNet, which keeps full 32×32 resolution in its first stage. It is a much stronger model on CIFAR-10, which explains the roughly 8-point gap.

> Note: `implementation_progress_report.md` says our architectures match the paper's parameter counts. They do not.

## 2. VGG19 has a larger classifier head (part of the +1.5%)

| | Paper | Ours (`src/models/cifar_vgg.py`) |
|---|---|---|
| Parameters | **20.04M** | **~20.57M** |
| Classifier | Single `Linear(512, 10)` | `Linear(512,512) → ReLU → Dropout → Linear(512,512) → ReLU → Dropout → Linear(512,10)` |

20.04M is the standard CIFAR VGG19-BN with a single linear classifier. Our version adds two hidden fully connected layers with Dropout, which adds capacity and regularization.

## 3. Input normalization differs

| Paper | Ours (`src/utils.py`) |
|---|---|
| "Images were normalized to the range [0, 1]" | `ToTensor()` followed by `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))` |

Normalizing each channel to zero mean and unit variance usually makes SGD train faster and to a slightly higher final accuracy than plain [0, 1] scaling.

## 4. The checkpoint is chosen by best test accuracy

`experiments/train_baselines.py` evaluates on the **test set** after every epoch and saves the model only when test accuracy improves. The baseline is therefore the **best of 100 test-set evaluations**, not the final-epoch model.

This biases the baseline upward, typically by a few tenths of a percent. The paper does not say how it chose its checkpoint.

## 5. Smaller factors

- **No fixed random seed.** Nothing in the repo sets a seed, so each run has its own random variation.
- **Different hardware.** We ran on an H100. The paper used Kaggle P100 / T4 GPUs, which causes small numerical differences.

---

## Contribution of each factor

| Factor | VGG19 | ResNet18 / ResNet34 |
|---|---|---|
| 1. CIFAR-style vs ImageNet-style ResNet | — | **Dominant (~7–8%)** |
| 2. Extra FC layers + Dropout in VGG19 | Moderate | — |
| 3. Mean/std normalization | Small–moderate | Small |
| 4. Best-on-test checkpoint | Small (~0.1–0.3%) | Small (~0.1–0.3%) |
| 5. Seed / hardware | Noise | Noise |

These estimates come from the code and from typical CIFAR-10 accuracy for each kind of model, not from ablation runs. To measure each factor exactly, retrain with each change reverted on its own.

## To reproduce the paper's baselines

1. Use `torchvision.models.resnet18/34(num_classes=10)` (ImageNet-style first conv and max-pool).
2. Give VGG19 a single `Linear(512, 10)` classifier.
3. Use only `ToTensor()` for normalization, scaling images to [0, 1].
4. Save the final-epoch checkpoint instead of the best on the test set.
5. Fix the random seed.

After these changes, the FP32 baselines should come out at about 91% for VGG19 and about 86% for ResNet18/34. The compression results should then line up with Table 1.
