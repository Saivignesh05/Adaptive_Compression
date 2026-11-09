"""
Adaptive Baselines -- VGG19 on CIFAR-10
==========================================================
Runs the three "Adaptive" methods from the paper:
1. Ours Adaptive Quantization
2. Ours Adaptive Pruned
3. Proposed AQP (Adaptive Quantization and Pruning)

Output: A summary table + JSON results file comparing the three methods.

Usage:
    python experiments/run_adaptive_baselines.py \
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, get_cifar10_dataloaders, evaluate_model
from src.models import get_model
from src.quantization_search import QuantizationOnlySearchEngine
from src.pruning_search import PruningOnlySearchEngine
from src.search_engine import AdaptiveSearchEngine

def main():
    parser = argparse.ArgumentParser(description="Adaptive Baselines for VGG19")
    parser.add_argument("--config", type=str, default="./configs/vgg19_quantization_only.yaml")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")
    calib_samples = config["dataset"].get("calibration_samples", 1000)

    print(f"\n{'=' * 65}")
    print(f"  ALL ADAPTIVE BASELINES (Quant, Pruned, AQP)")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Device  : {args.device}")
    print(f"{'=' * 65}\n")

    # -- Load dataset --
    _, test_loader, calib_loader = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=batch_size, calibration_samples=calib_samples, use_synthetic=args.fast
    )

    # -- Load model --
    use_pretrained = config["model"].get("pretrained", False)
    base_model = get_model(model_name, num_classes=num_classes, pretrained=use_pretrained)

    if args.weights_path and os.path.exists(args.weights_path):
        base_model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    
    base_model.to(args.device)
    base_model.eval()

    results = {}

    # 1. Adaptive Quantization Only
    print("\n[1/3] Running Adaptive Quantization Only...")
    model_1 = copy.deepcopy(base_model)
    engine_quant = QuantizationOnlySearchEngine(config)
    _, m_quant = engine_quant.run_quantization_search(model_1, test_loader, calib_loader, device=args.device)
    results["QuantizationOnly"] = m_quant

    # 2. Adaptive Pruning Only
    print("\n[2/3] Running Adaptive Pruning Only...")
    model_2 = copy.deepcopy(base_model)
    engine_prune = PruningOnlySearchEngine(config)
    _, m_prune = engine_prune.run_pruning_search(model_2, test_loader, calib_loader, device=args.device)
    results["PruningOnly"] = m_prune

    # 3. Proposed AQP (Adaptive Quantization + Pruning)
    print("\n[3/3] Running Proposed AQP (Quantization + Pruning)...")
    model_3 = copy.deepcopy(base_model)
    engine_aqp = AdaptiveSearchEngine(config)
    _, m_aqp = engine_aqp.run_adaptive_search(model_3, test_loader, calib_loader, device=args.device)
    results["AQP"] = m_aqp

    # -- Print Summary Tables --
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY TABLES (Paper Table 1)")
    print(f"{'=' * 65}")
    
    # Quantization
    print(f"  Ours Adaptive Quantization             |  {m_quant['compressed_accuracy'] * 100:.2f}%")
    print(f"  (Avg. Bit-width b_bar)                 |  ({m_quant.get('average_bit_width', 32.0):.2f})")
    print(f"  (b_bar with Huffman Enc.)              |  ({m_quant.get('huffman_average_bit_width', 32.0):.2f})")
    print(f"  {'-' * 50}")

    # Pruning
    print(f"  Ours Adaptive Pruned                   |  {m_prune['compressed_accuracy'] * 100:.2f}%")
    print(f"  Pruned S_overall (in %)                |  ({m_prune.get('overall_sparsity', 0.0) * 100:.0f}%)")
    print(f"  (Avg. Bit-width b_bar)                 |  ({32.0 * (1.0 - m_prune.get('overall_sparsity', 0.0)):.2f})")
    print(f"  {'-' * 50}")

    # AQP
    print(f"  Adaptive quantization and pruning      |  {m_aqp['compressed_accuracy'] * 100:.2f}%")
    print(f"  Proposed AQP                           |  {m_aqp['compressed_accuracy'] * 100:.2f}%")
    print(f"  (Avg. Bit-width b_bar)                 |  ({m_aqp.get('average_bit_width', 32.0):.2f})")
    print(f"  (b_bar with Huffman Enc.)              |  ({m_aqp.get('huffman_average_bit_width', 32.0):.2f})")
    print(f"{'=' * 65}\n")

    # Save results
    output_dir = args.output_dir or os.path.join(
        config["registry"].get("output_dir", "./registry/vgg19"),
        "adaptive_baselines"
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"adaptive_baselines_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  [OK] Results saved to: {results_path}\n")

if __name__ == "__main__":
    main()
