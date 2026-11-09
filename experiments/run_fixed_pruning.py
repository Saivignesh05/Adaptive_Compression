"""
Fixed Magnitude Pruning Baselines -- VGG19 on CIFAR-10
==========================================================
Applies the same sparsity percentage (e.g., 25%, 50%, 75%, 90%)
uniformly across all compressible layers and measures the resulting accuracy.
This reproduces the "Pruning only approach" rows in Table 1 of the paper.

Output: A summary table + JSON results file comparing:
  - Original FP32 baseline
  - Fixed 25%, 50%, 75%, 90% pruned models

Usage:
    python experiments/run_fixed_pruning.py \
        --config configs/vgg19_quantization_only.yaml \
        --weights_path ./checkpoints/vgg19_cifar10_baseline.pt \
        --device cuda
"""

import os
import sys
import copy
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn.utils.prune as prune

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, get_cifar10_dataloaders, evaluate_model, get_compressible_layers
from src.models import get_model


def run_fixed_pruning(model, amount, test_loader, device="cpu"):
    """
    Prune all compressible layers uniformly by the target percentage (amount),
    then evaluate accuracy.
    """
    pruned_model = copy.deepcopy(model)
    compressible = get_compressible_layers(pruned_model)
    
    for name, module in compressible:
        # Apply L1 unstructured (magnitude) pruning
        prune.l1_unstructured(module, name="weight", amount=amount)
        # Make the pruning permanent by removing the forward pre-hook
        prune.remove(module, "weight")

    # Evaluate
    accuracy = evaluate_model(pruned_model, test_loader, device=device)

    # Compute actual overall sparsity to verify
    total_zeros = 0
    total_params = 0
    for name, layer in get_compressible_layers(pruned_model):
        weights = layer.weight.data
        total_zeros += (weights == 0).sum().item()
        total_params += weights.numel()
        
    actual_sparsity = total_zeros / total_params if total_params > 0 else 0.0

    return accuracy, actual_sparsity


def main():
    parser = argparse.ArgumentParser(
        description="Fixed Magnitude Pruning Baselines for VGG19"
    )
    parser.add_argument(
        "--config", type=str,
        default="./configs/vgg19_quantization_only.yaml",
        help="Path to YAML config"
    )
    parser.add_argument(
        "--weights_path", type=str, default=None,
        help="Path to pre-trained VGG19 checkpoint (.pt)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use synthetic mini-dataset for smoke test"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results"
    )
    args = parser.parse_args()

    # -- Load config --
    config = load_config(args.config)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")

    print(f"\n{'=' * 65}")
    print(f"  FIXED UNIFORM PRUNING BASELINES")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Dataset : {config['dataset']['name']}")
    print(f"  Device  : {args.device}")
    print(f"{'=' * 65}\n")

    # -- Load dataset --
    _, test_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_synthetic=args.fast,
    )

    # -- Load model --
    use_pretrained = config["model"].get("pretrained", False)
    model = get_model(model_name, num_classes=num_classes, pretrained=use_pretrained)

    if args.weights_path and os.path.exists(args.weights_path):
        print(f"[Info] Loading weights from: {args.weights_path}")
        model.load_state_dict(
            torch.load(args.weights_path, map_location=args.device)
        )
    else:
        print("[Info] No checkpoint provided; using initialized / hub weights.")

    model.to(args.device)
    model.eval()

    # -- FP32 Baseline --
    fp32_accuracy = evaluate_model(model, test_loader, device=args.device)
    print(f"  Original (32-bit, 0% pruned) Accuracy : {fp32_accuracy * 100:.2f}%\n")

    # Target pruning percentages from the paper Table 1
    pruning_targets = [0.25, 0.50, 0.75, 0.90]
    results = []

    results.append({
        "method": "Original (32-bit)",
        "target_sparsity": 0.0,
        "actual_sparsity": 0.0,
        "accuracy": float(fp32_accuracy),
        "accuracy_pct": round(fp32_accuracy * 100, 2),
    })

    print(f"  {'Method':<30s} {'Accuracy':>10s} {'Actual Sparsity':>18s}")
    print(f"  {'-' * 62}")
    print(f"  {'Original (32-bit)':<30s} {fp32_accuracy * 100:>9.2f}% {'0.00%':>17s}")

    for target in pruning_targets:
        start = time.time()
        acc, actual_sparsity = run_fixed_pruning(
            model, target, test_loader, device=args.device
        )
        elapsed = time.time() - start

        results.append({
            "method": f"Pruned ({int(target * 100)}%)",
            "target_sparsity": target,
            "actual_sparsity": float(actual_sparsity),
            "accuracy": float(acc),
            "accuracy_pct": round(acc * 100, 2),
        })

        print(
            f"  Pruned ({int(target * 100)}%)" + " " * 19 +
            f"{acc * 100:>9.2f}% "
            f"{actual_sparsity * 100:>16.2f}% "
            f"[{elapsed:.1f}s]"
        )

    # -- Print final summary table --
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY TABLE (Paper Table 1 -- Pruning Only)")
    print(f"{'=' * 65}")
    print(f"  {'Model':<35s} | {'VGG19':>10s}")
    print(f"  {'-' * 50}")
    for r in results:
        if r['method'] == "Original (32-bit)":
            continue
        print(f"  {r['method']:<35s} | {r['accuracy_pct']:>9.2f}%")
    print(f"{'=' * 65}\n")

    # -- Save results --
    output_dir = args.output_dir or os.path.join(
        config["registry"].get("output_dir", "./registry/vgg19"),
        "fixed_pruning"
    )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"fixed_pruning_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "baseline_accuracy": float(fp32_accuracy),
            "results": results,
        }, f, indent=2)

    print(f"  [OK] Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
