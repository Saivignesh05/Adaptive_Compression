"""
Quantization-Only Adaptive Compression — VGG19 on CIFAR-10
============================================================
Runs only the quantization path (no pruning) of the adaptive layer importance
pipeline described in Shinde (NeurIPS 2024).

Pipeline stages:
  1. Train / load a baseline FP32 VGG19 model on CIFAR-10.
  2. Compute per-layer importance scores via StatisticalScorer.
  3. Run greedy per-layer bit-width search (QuantizationOnlySearchEngine).
  4. Save quantized model + metrics + MODEL_CARD to the registry.

Usage:
    # With a pre-trained checkpoint:
    python experiments/run_quantization_only.py \
        --config configs/vgg19_quantization_only.yaml \
        --weights_path ./checkpoints/vgg19_cifar10_baseline.pt \
        --device cuda

    # Quick smoke test with synthetic data:
    python experiments/run_quantization_only.py \
        --config configs/vgg19_quantization_only.yaml \
        --fast

    # Train from scratch first, then quantize:
    python experiments/run_quantization_only.py \
        --config configs/vgg19_quantization_only.yaml \
        --train_baseline \
        --device cuda
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, get_cifar10_dataloaders, evaluate_model
from src.models import get_model
from src.quantization_search import QuantizationOnlySearchEngine
from src.model_registry import ModelRegistry


# ═══════════════════════════════════════════════════════════════════════
# Baseline Training (VGG19 on CIFAR-10)
# ═══════════════════════════════════════════════════════════════════════
def train_vgg19_baseline(config: dict, device: str = "cpu", save_dir: str = "./checkpoints"):
    """
    Trains VGG19 on CIFAR-10 according to the paper's implementation details:
      - SGD optimiser, lr=0.02, momentum=0.9, weight decay=5e-4
      - lr halved every 20 epochs
      - 100 epochs total
      - Batch size 128
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vgg19_cifar10_baseline.pt")

    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    lr_decay = config["training"].get("lr_decay_factor", 0.5)
    lr_step = config["training"].get("lr_decay_epochs", 20)
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 5e-4)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")

    print(f"\n{'=' * 65}")
    print(f"  TRAINING BASELINE VGG19 on CIFAR-10")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size} | Device: {device}")
    print(f"{'=' * 65}\n")

    train_loader, test_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=batch_size
    )

    use_pretrained = config["model"].get("pretrained", False)
    model = get_model(model_name, num_classes=num_classes, pretrained=use_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.stack(inputs)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total if total > 0 else 0.0
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────
        val_acc = evaluate_model(model, test_loader, device=device)
        elapsed = time.time() - start_time

        print(
            f"  Epoch [{epoch:3d}/{epochs}]  "
            f"Loss: {running_loss / len(train_loader):.4f}  "
            f"Train Acc: {train_acc * 100:.2f}%  "
            f"Val Acc: {val_acc * 100:.2f}%  "
            f"LR: {scheduler.get_last_lr()[0]:.5f}  "
            f"[{elapsed:.0f}s]"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n  Best validation accuracy: {best_acc * 100:.2f}%")
    print(f"  Checkpoint saved to: {save_path}\n")
    return save_path, best_acc


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Quantization-Only Adaptive Compression for VGG19"
    )
    parser.add_argument(
        "--config", type=str,
        default="./configs/vgg19_quantization_only.yaml",
        help="Path to quantization-only YAML config"
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
        help="Use synthetic mini-dataset for instant smoke test"
    )
    parser.add_argument(
        "--train_baseline", action="store_true",
        help="Train VGG19 from scratch before running quantization"
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config = load_config(args.config)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")
    calib_samples = config["dataset"].get("calibration_samples", 1000)

    print(f"\n{'=' * 65}")
    print(f"  ADAPTIVE QUANTIZATION-ONLY PIPELINE")
    print(f"  Model     : {model_name.upper()}")
    print(f"  Dataset   : {config['dataset']['name']}")
    print(f"  Mode      : Quantization Only (no pruning)")
    print(f"  T_margin  : {config['compression']['target_margin'] * 100:.2f}%")
    print(f"  Bit Range : {config['compression']['bit_width_range']}")
    print(f"  Device    : {args.device}")
    print(f"{'=' * 65}\n")

    # ── Step 0 (optional): Train baseline ────────────────────────────
    weights_path = args.weights_path
    if args.train_baseline:
        weights_path, _ = train_vgg19_baseline(config, device=args.device)

    # ── Load dataset ─────────────────────────────────────────────────
    train_loader, test_loader, calib_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        calibration_samples=calib_samples,
        use_synthetic=args.fast,
    )

    # ── Load model ───────────────────────────────────────────────────
    use_pretrained = config["model"].get("pretrained", False)
    model = get_model(model_name, num_classes=num_classes, pretrained=use_pretrained)

    if weights_path and os.path.exists(weights_path):
        print(f"[Info] Loading pre-trained weights from: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=args.device)
        )
    else:
        print("[Info] No checkpoint provided; using initialized / hub weights.")

    model.to(args.device)

    # ── Run Quantization-Only Search ─────────────────────────────────
    engine = QuantizationOnlySearchEngine(config)
    quantized_model, metrics = engine.run_quantization_search(
        model, test_loader, calib_loader=calib_loader, device=args.device
    )

    # ── Register model artifact ──────────────────────────────────────
    registry_dir = config["registry"].get("output_dir", "./registry/vgg19")
    registry = ModelRegistry(registry_dir=registry_dir)
    variant_path = registry.register_model(
        quantized_model, model_name, metrics, config
    )

    # ── Save a standalone results JSON ───────────────────────────────
    results_json_path = os.path.join(variant_path, "quantization_results.json")
    with open(results_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  [OK] QUANTIZATION-ONLY COMPRESSION COMPLETE")
    print(f"    Artifacts saved in : {variant_path}")
    print(f"    Results JSON       : {results_json_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
