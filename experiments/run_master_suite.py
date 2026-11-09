"""
Master Compression Suite Orchestrator
=====================================
Executes the full suite of experiments from Shinde (NeurIPS 2024):
  1. Fixed Uniform Quantization (8-bit down to 1-bit)
  2. Fixed Uniform Pruning (25%, 50%, 75%, 90%)
  3. Adaptive Quantization Only
  4. Adaptive Pruning Only
  5. Proposed AQP (Adaptive Quantization & Pruning)

Fully config-driven with no hardcoded paths or thresholds.
Saves all model variants, metadata, and Model Cards to the registry.

Usage:
  # For a single model (e.g. ResNet18):
  python experiments/run_master_suite.py --config configs/resnet18_cifar10.yaml --device cuda

  # For fast smoke testing:
  python experiments/run_master_suite.py --config configs/vgg19_cifar10.yaml --fast

  # Run all models:
  python experiments/run_master_suite.py --all --fast
"""

import os
import sys
import copy
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn.utils.prune as prune

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, get_cifar10_dataloaders, evaluate_model, get_compressible_layers
from src.models import get_model
from src.quantizer import UniformQuantizer
from src.huffman import HuffmanEncoder
from src.quantization_search import QuantizationOnlySearchEngine
from src.pruning_search import PruningOnlySearchEngine
from src.search_engine import AdaptiveSearchEngine
from src.model_registry import ModelRegistry


def run_model_suite(config_path: str, weights_path: str = None, device: str = "cpu", fast: bool = False, output_dir: str = None, train_if_missing: bool = False):
    config = load_config(config_path)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")
    calib_samples = config["dataset"].get("calibration_samples", 1000)

    reg_dir = output_dir or config["registry"].get("output_dir", f"./registry/{model_name}")
    os.makedirs(reg_dir, exist_ok=True)
    registry = ModelRegistry(registry_dir=reg_dir)

    print(f"\n{'=' * 70}")
    print(f"  EXECUTING MASTER COMPRESSION SUITE FOR: {model_name.upper()}")
    print(f"  Config : {config_path}")
    print(f"  Device : {device} | Fast mode: {fast}")
    print(f"  Registry Output : {reg_dir}")
    print(f"{'=' * 70}\n")

    # Load data
    train_loader, test_loader, calib_loader = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=batch_size, calibration_samples=calib_samples, use_synthetic=fast
    )

    # Load Model
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    resolved_weights = weights_path
    if not resolved_weights:
        default_ckpt = f"./checkpoints/{model_name}_cifar10_baseline.pt"
        if os.path.exists(default_ckpt):
            resolved_weights = default_ckpt

    if resolved_weights and os.path.exists(resolved_weights):
        print(f"[Info] Loading weights from: {resolved_weights}")
        model.load_state_dict(torch.load(resolved_weights, map_location=device))
    elif train_if_missing and not fast:
        print(f"[Info] No checkpoint found for {model_name}. Auto-training baseline from config...")
        from experiments.train_baselines import train_baseline_from_config
        ckpt_path, _ = train_baseline_from_config(config_path, device=device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("[Info] No checkpoint found; executing with initialized model.")

    model.to(device)
    model.eval()

    fp32_acc = evaluate_model(model, test_loader, device=device)
    print(f"  FP32 Baseline Top-1 Accuracy: {fp32_acc * 100:.2f}%\n")

    suite_results = {
        "model_name": model_name,
        "baseline_accuracy": float(fp32_acc),
        "fixed_quantization": {},
        "fixed_pruning": {},
        "adaptive_quantization": {},
        "adaptive_pruning": {},
        "proposed_aqp": {}
    }

    quantizer = UniformQuantizer()
    compressible = get_compressible_layers(model)

    # 1. Fixed Quantization (8-bit down to 1-bit)
    print("\n>>> [1/5] Running Fixed Quantization Suite (8-bit -> 1-bit)...")
    for b in range(8, 0, -1):
        q_bits = {name: b for name, _ in compressible}
        q_model = quantizer.quantize_model(model, q_bits)
        acc = evaluate_model(q_model, test_loader, device=device)
        suite_results["fixed_quantization"][f"{b}-bit"] = {
            "accuracy": float(acc),
            "bit_width": float(b),
            "drop": float(fp32_acc - acc)
        }
        print(f"    Fixed {b}-bit: Accuracy = {acc * 100:.2f}% (Drop = {(fp32_acc - acc) * 100:.2f}%)")

    # 2. Fixed Pruning (25%, 50%, 75%, 90%)
    print("\n>>> [2/5] Running Fixed Pruning Suite (25%, 50%, 75%, 90%)...")
    for p_val in [0.25, 0.50, 0.75, 0.90]:
        p_pct = int(p_val * 100)
        p_model = copy.deepcopy(model)
        for _, mod in get_compressible_layers(p_model):
            prune.l1_unstructured(mod, name="weight", amount=p_val)
            prune.remove(mod, "weight")
        acc = evaluate_model(p_model, test_loader, device=device)
        suite_results["fixed_pruning"][f"{p_pct}%"] = {
            "accuracy": float(acc),
            "target_sparsity": p_val,
            "drop": float(fp32_acc - acc)
        }
        print(f"    Pruned {p_pct}%: Accuracy = {acc * 100:.2f}% (Drop = {(fp32_acc - acc) * 100:.2f}%)")

    # 3. Adaptive Quantization Only
    print("\n>>> [3/5] Running Adaptive Quantization Only...")
    engine_quant = QuantizationOnlySearchEngine(config)
    q_adapt_model, m_quant = engine_quant.run_quantization_search(copy.deepcopy(model), test_loader, calib_loader, device=device)
    suite_results["adaptive_quantization"] = m_quant
    print(f"    Adaptive Quant: Acc = {m_quant['compressed_accuracy'] * 100:.2f}% | b_bar = {m_quant.get('average_bit_width', 32.0):.2f} bits")

    # 4. Adaptive Pruning Only
    print("\n>>> [4/5] Running Adaptive Pruning Only...")
    engine_prune = PruningOnlySearchEngine(config)
    p_adapt_model, m_prune = engine_prune.run_pruning_search(copy.deepcopy(model), test_loader, calib_loader, device=device)
    suite_results["adaptive_pruning"] = m_prune
    print(f"    Adaptive Prune: Acc = {m_prune['compressed_accuracy'] * 100:.2f}% | Sparsity = {m_prune.get('overall_sparsity', 0.0) * 100:.1f}%")

    # 5. Proposed AQP (Adaptive Quantization & Pruning)
    print("\n>>> [5/5] Running Proposed AQP...")
    engine_aqp = AdaptiveSearchEngine(config)
    aqp_model, m_aqp = engine_aqp.run_adaptive_search(copy.deepcopy(model), test_loader, calib_loader, device=device)
    suite_results["proposed_aqp"] = m_aqp
    print(f"    Proposed AQP: Acc = {m_aqp['compressed_accuracy'] * 100:.2f}% | b_bar = {m_aqp.get('average_bit_width', 32.0):.2f} bits (Huffman: {m_aqp.get('huffman_average_bit_width', 32.0):.2f})")

    # Save to Model Registry (Deliverable 2: versioned model variant with MODEL_CARD.md)
    variant_dir = registry.register_model(aqp_model, model_name, m_aqp, config)
    
    # Save complete suite summary json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(reg_dir, f"full_suite_results_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(suite_results, f, indent=2)

    print(f"\n[Suite Complete] Results saved to: {summary_path}")
    print(f"[Model Registry] Versioned model & card in: {variant_dir}\n")
    return suite_results


def main():
    parser = argparse.ArgumentParser(description="Master Compression Suite Orchestrator")
    parser.add_argument("--config", type=str, default="./configs/vgg19_cifar10.yaml", help="YAML config")
    parser.add_argument("--weights_path", type=str, default=None, help="Weights checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--fast", action="store_true", help="Fast synthetic smoke run")
    parser.add_argument("--train_if_missing", action="store_true", help="Auto-train baseline checkpoint if not found")
    parser.add_argument("--all", action="store_true", help="Run across all 3 models (VGG19, ResNet18, ResNet34)")
    args = parser.parse_args()

    if args.all:
        model_configs = [
            "./configs/vgg19_cifar10.yaml",
            "./configs/resnet18_cifar10.yaml",
            "./configs/resnet34_cifar10.yaml"
        ]
        for cfg in model_configs:
            run_model_suite(config_path=cfg, device=args.device, fast=args.fast, train_if_missing=args.train_if_missing)
    else:
        run_model_suite(config_path=args.config, weights_path=args.weights_path, device=args.device, fast=args.fast, train_if_missing=args.train_if_missing)


if __name__ == "__main__":
    main()

