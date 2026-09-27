import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, get_cifar10_dataloaders, evaluate_model
from src.models import get_model
from src.search_engine import AdaptiveSearchEngine
from src.model_registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description="Run Adaptive Compression Pipeline")
    parser.add_argument("--config", type=str, default="./configs/vgg19_cifar10.yaml", help="Path to config YAML")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to pre-trained PyTorch weights checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--fast", action="store_true", help="Use fast synthetic dataset for instant execution test")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_name = config["model"]["name"]
    batch_size = config["dataset"].get("batch_size", 128)
    data_dir = config["dataset"].get("data_dir", "./data")
    calib_samples = config["dataset"].get("calibration_samples", 1000)

    print(f"=== Running Adaptive Compression Pipeline ===")
    print(f"Model: {model_name} | Dataset: {config['dataset']['name']}")
    print(f"Target Accuracy Margin T_margin: {config['compression']['target_margin']*100:.2f}%")

    # Load dataset
    train_loader, test_loader, calib_loader = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=batch_size, calibration_samples=calib_samples, use_synthetic=args.fast
    )

    # Instantiate model
    model = get_model(model_name, num_classes=config["model"].get("num_classes", 10), pretrained=True)

    if args.weights_path and os.path.exists(args.weights_path):
        print(f"Loading weights from {args.weights_path}...")
        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))

    model.to(args.device)

    # Initialize Adaptive Search Engine
    engine = AdaptiveSearchEngine(config)
    compressed_model, metrics = engine.run_adaptive_search(
        model, test_loader, calib_loader=calib_loader, device=args.device
    )

    # Register output model artifact
    registry_dir = config["registry"].get("output_dir", "./registry")
    registry = ModelRegistry(registry_dir=registry_dir)
    variant_path = registry.register_model(compressed_model, model_name, metrics, config)

    print(f"\n[Success] Adaptive Compression completed successfully!")
    print(f"Artifacts saved in: {variant_path}")

if __name__ == "__main__":
    main()
