import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import get_model
from src.utils import get_cifar10_dataloaders, evaluate_model, load_config

def train_baseline(model_name: str = "vgg19", epochs: int = 5, lr: float = 0.01, device: str = "cpu", save_dir: str = "./checkpoints"):
    """
    Trains or initializes a baseline model on CIFAR-10 and saves checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_cifar10_baseline.pt")

    print(f"=== Training Baseline Model: {model_name} ===")
    train_loader, test_loader, _ = get_cifar10_dataloaders(batch_size=128)

    model = get_model(model_name, num_classes=10, pretrained=True)
    model.to(device)

    # Check baseline initial accuracy
    initial_acc = evaluate_model(model, test_loader, device=device)
    print(f"Initial Model Accuracy: {initial_acc * 100:.2f}%")

    if initial_acc < 0.70:
        print(f"Fine-tuning {model_name} for {epochs} epochs to ensure strong baseline...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
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

            acc = evaluate_model(model, test_loader, device=device)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {acc*100:.2f}%")

    final_acc = evaluate_model(model, test_loader, device=device)
    print(f"Saving baseline model with accuracy {final_acc * 100:.2f}% to {save_path}")
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model for Adaptive Compression")
    parser.add_argument("--model", type=str, default="vgg19", choices=["vgg19", "resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_baseline(model_name=args.model, epochs=args.epochs, lr=args.lr, device=args.device)
