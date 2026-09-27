import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_cifar10_dataloaders(data_dir: str = "./data", batch_size: int = 128, calibration_samples: int = 1000, num_workers: int = 0, use_synthetic: bool = False):
    """
    Returns train_loader, val_loader, and calibration_loader for CIFAR-10.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    os.makedirs(data_dir, exist_ok=True)
    cifar_archive = os.path.join(data_dir, "cifar-10-batches-py")

    if use_synthetic or not os.path.exists(cifar_archive):
        try:
            train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=not use_synthetic, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=not use_synthetic, transform=transform_test)
        except Exception as e:
            print(f"[Info] Using fast synthetic dataset for execution ({e}).")
            train_dataset = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(256)]
            test_dataset = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(128)]
    else:
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Calibration subset for activation sparsity and importance calculation
    calib_indices = list(range(min(calibration_samples, len(train_dataset))))
    if isinstance(train_dataset, list):
        calib_dataset = train_dataset[:len(calib_indices)]
    else:
        calib_dataset = Subset(train_dataset, calib_indices)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, calib_loader

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> float:
    """Evaluates top-1 classification accuracy of a model."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.stack(inputs)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total if total > 0 else 0.0

def get_compressible_layers(model: nn.Module):
    """
    Extracts name and reference of compressible layers (nn.Conv2d and nn.Linear).
    """
    compressible = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            compressible.append((name, module))
    return compressible
