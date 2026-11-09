"""
Fixed Uniform Quantization Baselines -- VGG19 on CIFAR-10
==========================================================
Applies the SAME bit-width to ALL layers (1-bit through 8-bit) and
measures the resulting accuracy. This reproduces the "Fixed Quantization"
rows in Table 1 of Shinde (NeurIPS 2024).

Output: A summary table + JSON results file comparing:
  - Original FP32 baseline
  - Fixed 8-bit, 7-bit, 6-bit, 5-bit, 4-bit, 3-bit, 2-bit, 1-bit

Usage:
    python experiments/run_fixed_quantization.py \
        --config configs/vgg19_quantization_only.yaml \
        --weights_path ./checkpoints/vgg19_cifar10_baseline.pt \
        --device cuda
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, get_cifar10_dataloaders, evaluate_model, get_compressible_layers
from src.models import get_model
from src.quantizer import UniformQuantizer
from src.huffman import HuffmanEncoder


def run_fixed_quantization(model, bit_width, test_loader, device="cpu"):
    """
    Quantize ALL compressible layers to the same fixed bit-width,
    then evaluate accuracy.
    """
    quantizer = UniformQuantizer()

    # Build a dict mapping every compressible layer to the same bit-width
    compressible = get_compressible_layers(model)
    layer_bit_widths = {name: bit_width for name, _ in compressible}

    # Quantize
    quantized_model = quantizer.quantize_model(model, layer_bit_widths)

    # Evaluate
    accuracy = evaluate_model(quantized_model, test_loader, device=device)

    # Compute Huffman average bit-width
    total_params = sum(
        layer.weight.data.numel() for _, layer in compressible
    )
    huffman_total_bits = 0.0
    for name, layer in get_compressible_layers(quantized_model):
        n_params = layer.weight.data.numel()
        b_huff = HuffmanEncoder.compute_huffman_bit_width(layer.weight.data)
        huffman_total_bits += b_huff * n_params

    huffman_avg_bw = huffman_total_bits / total_params if total_params > 0 else 0.0

    return accuracy, huffman_avg_bw


def main():
    parser = argparse.ArgumentParser(
        description="Fixed Uniform Quantization Baselines for VGG19"
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
        help="Directory to save results (default: registry/<model>/fixed_quantization)"
    )
    args = parser.parse_args()

    # -- Load config --
    config = load_config(args.config)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")

    print(f"\n{'=' * 65}")
    print(f"  FIXED UNIFORM QUANTIZATION BASELINES")
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
    print(f"  Original (32-bit) Accuracy : {fp32_accuracy * 100:.2f}%\n")

    # -- Run Fixed Quantization for each bit-width (8 down to 1) --
    results = []
    results.append({
        "method": "Original (32-bit)",
        "bit_width": 32,
        "accuracy": float(fp32_accuracy),
        "accuracy_pct": round(fp32_accuracy * 100, 2),
        "huffman_avg_bw": 32.0,
        "compression_ratio": 1.0,
    })

    print(f"  {'Method':<30s} {'Accuracy':>10s} {'Huffman BW':>12s} {'Ratio':>8s}")
    print(f"  {'-' * 62}")
    print(f"  {'Original (32-bit)':<30s} {fp32_accuracy * 100:>9.2f}% {'32.00':>12s} {'1.00x':>8s}")

    for bit_width in range(8, 0, -1):
        start = time.time()
        acc, huffman_bw = run_fixed_quantization(
            model, bit_width, test_loader, device=args.device
        )
        elapsed = time.time() - start

        ratio = 32.0 / bit_width
        results.append({
            "method": f"Fixed ({bit_width}-bit) Quantization",
            "bit_width": bit_width,
            "accuracy": float(acc),
            "accuracy_pct": round(acc * 100, 2),
            "huffman_avg_bw": round(huffman_bw, 4),
            "compression_ratio": round(ratio, 2),
            "accuracy_drop": round((fp32_accuracy - acc) * 100, 2),
        })

        print(
            f"  Fixed ({bit_width}-bit) Quantization    "
            f"{acc * 100:>9.2f}% "
            f"{huffman_bw:>11.4f} "
            f"{ratio:>7.2f}x "
            f"[{elapsed:.1f}s]"
        )

    # -- Print final summary table --
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY TABLE (Paper Table 1 -- Quantization Only)")
    print(f"{'=' * 65}")
    print(f"  {'Model':<35s} | {'VGG19':>10s}")
    print(f"  {'-' * 50}")
    for r in results:
        print(f"  {r['method']:<35s} | {r['accuracy_pct']:>9.2f}%")
    print(f"{'=' * 65}\n")

    # -- Save results --
    output_dir = args.output_dir or os.path.join(
        config["registry"].get("output_dir", "./registry/vgg19"),
        "fixed_quantization"
    )
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"fixed_quantization_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name,
            "baseline_accuracy": float(fp32_accuracy),
            "results": results,
        }, f, indent=2)

    print(f"  [OK] Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
