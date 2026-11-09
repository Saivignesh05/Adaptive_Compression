"""
Layer-wise Sensitivity Analysis for VGG19 (Paper Figure 2)
==========================================================
Replicates Figure 2 from Shinde (NeurIPS 2024):
  (a) Model Quantization: Varying low bit-precision (1-8 bit) for individual
      layers while keeping other layers at 8-bit quantization.
  (b) Model Pruning: Varying pruning percentages (0%, 25%, 50%, 75%, 90%, 99%)
      for individual layers without pruning other layers.

Output:
  - reports/figures/figure2_vgg19_sensitivity.png
  - reports/vgg19_layer_sensitivity.json

Usage:
  python experiments/run_layer_sensitivity_vgg19.py --config configs/vgg19_cifar10.yaml --device cuda
"""

import os
import sys
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, get_cifar10_dataloaders, evaluate_model, get_compressible_layers
from src.models import get_model
from src.quantizer import UniformQuantizer


def run_sensitivity_analysis(config_path: str, weights_path: str = None, device: str = "cpu", fast: bool = False, output_dir: str = "./reports"):
    config = load_config(config_path)
    model_name = config["model"]["name"]
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  VGG19 LAYER SENSITIVITY ANALYSIS (Paper Figure 2)")
    print(f"  Device: {device} | Fast mode: {fast}")
    print(f"{'=' * 65}\n")

    _, test_loader, _ = get_cifar10_dataloaders(data_dir=data_dir, batch_size=batch_size, use_synthetic=fast)

    model = get_model(model_name, num_classes=10, pretrained=False)
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    elif os.path.exists("./checkpoints/vgg19_cifar10_baseline.pt"):
        print("Loading default checkpoint ./checkpoints/vgg19_cifar10_baseline.pt...")
        model.load_state_dict(torch.load("./checkpoints/vgg19_cifar10_baseline.pt", map_location=device))

    model.to(device)
    model.eval()

    compressible = get_compressible_layers(model)
    layer_names = [name for name, _ in compressible]
    num_layers = len(layer_names)
    print(f"Found {num_layers} compressible layers in {model_name.upper()}.")

    quantizer = UniformQuantizer()
    bit_widths = [8, 7, 6, 5, 4, 3, 2, 1]
    prune_percentages = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99]

    # 1. Figure 2(a) - Quantization Sensitivity
    print("\n--- Running (a) Model Quantization Sensitivity ---")
    quant_acc_matrix = {b: [] for b in bit_widths}

    for b in bit_widths:
        print(f"Evaluating individual layers at {b}-bit (all other layers at 8-bit)...")
        for target_layer in layer_names:
            # Baseline is 8-bit for all layers, except target_layer at b
            layer_bits = {n: 8 for n in layer_names}
            layer_bits[target_layer] = b
            q_model = quantizer.quantize_model(model, layer_bits)
            acc = evaluate_model(q_model, test_loader, device=device)
            quant_acc_matrix[b].append(acc * 100)

    # 2. Figure 2(b) - Pruning Sensitivity
    print("\n--- Running (b) Model Pruning Sensitivity ---")
    prune_acc_matrix = {int(p * 100): [] for p in prune_percentages}

    for p in prune_percentages:
        p_pct = int(p * 100)
        print(f"Evaluating individual layers at {p_pct}% pruning (other layers unpruned)...")
        for target_layer in layer_names:
            p_model = copy.deepcopy(model)
            if p > 0.0:
                target_mod = dict(get_compressible_layers(p_model))[target_layer]
                prune.l1_unstructured(target_mod, name="weight", amount=p)
                prune.remove(target_mod, "weight")
            acc = evaluate_model(p_model, test_loader, device=device)
            prune_acc_matrix[p_pct].append(acc * 100)

    # Save numeric JSON
    sensitivity_data = {
        "layer_names": layer_names,
        "quantization_sensitivity": quant_acc_matrix,
        "pruning_sensitivity": prune_acc_matrix
    }
    json_path = os.path.join(output_dir, "vgg19_layer_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(sensitivity_data, f, indent=2)
    print(f"Sensitivity data saved to: {json_path}")

    # Plot Figure 2 (a & b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    layer_indices = range(1, num_layers + 1)
    layer_labels = [f"L{i}" for i in layer_indices]

    # (a) Model Quantization Plot
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'x']
    for idx, b in enumerate(bit_widths):
        ax1.plot(layer_indices, quant_acc_matrix[b], marker=markers[idx % len(markers)], label=f"{b}-bit", linewidth=1.5, markersize=4)
    ax1.set_title("(a) Model Quantization", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Quantized Model Accuracy (%)", fontsize=12)
    ax1.set_xticks(list(layer_indices))
    ax1.set_xticklabels(layer_labels, rotation=45, fontsize=8)
    ax1.set_ylim([0, 105])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9, ncol=2)

    # (b) Model Pruning Plot
    p_markers = ['o', 's', '^', 'v', 'D', 'x']
    for idx, p in enumerate(prune_percentages):
        p_pct = int(p * 100)
        ax2.plot(layer_indices, prune_acc_matrix[p_pct], marker=p_markers[idx % len(p_markers)], label=f"Pruning {p_pct}%", linewidth=1.5, markersize=4)
    ax2.set_title("(b) Model Pruning", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Pruned Model Accuracy (%)", fontsize=12)
    ax2.set_xticks(list(layer_indices))
    ax2.set_xticklabels(layer_labels, rotation=45, fontsize=8)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9, ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "figure2_vgg19_sensitivity.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n[Success] Figure 2 generated and saved to: {fig_path}")
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG19 Layer Sensitivity Analysis (Paper Figure 2)")
    parser.add_argument("--config", type=str, default="./configs/vgg19_cifar10.yaml")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./reports")
    args = parser.parse_args()

    run_sensitivity_analysis(
        config_path=args.config,
        weights_path=args.weights_path,
        device=args.device,
        fast=args.fast,
        output_dir=args.output_dir
    )

