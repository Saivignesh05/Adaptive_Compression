"""
Config-Driven Baseline Training for CIFAR-10 Models
===================================================
Trains baseline FP32 models (VGG19, ResNet18, ResNet34) on CIFAR-10
according to the paper specifications:
  - Batch size: 128
  - Optimizer: SGD (lr=0.02, momentum=0.9, weight_decay=5e-4)
  - LR Decay: factor 0.5 every 20 epochs
  - Epochs: 100

Usage:
  python experiments/train_baselines.py --config configs/vgg19_cifar10.yaml --device cuda
  python experiments/train_baselines.py --config configs/resnet18_cifar10.yaml --device cuda
  python experiments/train_baselines.py --config configs/resnet34_cifar10.yaml --device cuda
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import get_model
from src.utils import get_cifar10_dataloaders, evaluate_model, load_config


def train_baseline_from_config(config_path: str, device: str = "cpu", epochs_override: int = None, save_dir: str = "./checkpoints"):
    config = load_config(config_path)
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", 10)
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")

    training_cfg = config.get("training", {})
    lr = float(training_cfg.get("learning_rate", 0.02))
    momentum = float(training_cfg.get("momentum", 0.9))
    weight_decay = float(training_cfg.get("weight_decay", 5e-4))
    lr_decay_factor = float(training_cfg.get("lr_decay_factor", 0.5))
    lr_decay_epochs = int(training_cfg.get("lr_decay_epochs", 20))
    epochs = epochs_override if epochs_override is not None else int(training_cfg.get("epochs", 100))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_cifar10_baseline.pt")

    print(f"\n{'=' * 65}")
    print(f"  BASELINE TRAINING: {model_name.upper()} on CIFAR-10")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Device: {device}")
    print(f"  LR: {lr} (decay x{lr_decay_factor} every {lr_decay_epochs} eps) | WD: {weight_decay}")
    print(f"  Target Save Path: {save_path}")
    print(f"{'=' * 65}\n")

    train_loader, test_loader, _ = get_cifar10_dataloaders(data_dir=data_dir, batch_size=batch_size)

    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_factor)

    best_acc = 0.0

    for epoch in range(epochs):
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
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0

        val_acc = evaluate_model(model, test_loader, device=device)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{epochs:3d}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model updated ({best_acc*100:.2f}%) saved to {save_path}")

    print(f"\n[Training Complete] Best Validation Accuracy: {best_acc*100:.2f}%")
    return save_path, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven baseline training on CIFAR-10")
    parser.add_argument("--config", type=str, default="./configs/vgg19_cifar10.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_baseline_from_config(config_path=args.config, device=args.device, epochs_override=args.epochs)
