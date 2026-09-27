import os
import json
import time
import torch
import torch.nn as nn
from tabulate import tabulate

class ModelRegistry:
    """
    Manages versioning, serialization, and Model Card generation for compressed model variants.
    """

    def __init__(self, registry_dir: str = "./registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def register_model(self, model: nn.Module, model_name: str, metrics: dict, config: dict) -> str:
        """
        Saves quantized model weights, metadata config, and generates a formatted Model Card.
        Returns variant folder path.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        variant_id = f"{model_name}_aqp_{timestamp}"
        variant_dir = os.path.join(self.registry_dir, variant_id)
        os.makedirs(variant_dir, exist_ok=True)

        # 1. Save model weights
        weights_path = os.path.join(variant_dir, "model_weights.pt")
        torch.save(model.state_dict(), weights_path)

        # 2. Save metadata JSON
        meta = {
            "variant_id": variant_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "config": config
        }
        meta_path = os.path.join(variant_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 3. Generate Model Card
        card_content = self.generate_model_card(variant_id, model_name, metrics, config)
        card_path = os.path.join(variant_dir, "MODEL_CARD.md")
        with open(card_path, "w") as f:
            f.write(card_content)

        print(f"[ModelRegistry] Registered versioned model variant: {variant_id}")
        return variant_dir

    def generate_model_card(self, variant_id: str, model_name: str, metrics: dict, config: dict) -> str:
        """Generates a Markdown Model Card string."""
        b_widths = metrics.get("layer_bit_widths", {})
        k_facs = metrics.get("layer_k_factors", {})
        sparsities = metrics.get("layer_sparsity", {})
        importances = metrics.get("layer_importance", {})

        table_data = []
        for name in b_widths:
            table_data.append([
                name,
                f"{importances.get(name, 0.0):.4f}",
                f"{b_widths[name]}-bit",
                f"{k_facs.get(name, 0.0):.2f}",
                f"{sparsities.get(name, 0.0) * 100:.1f}%"
            ])

        layer_table = tabulate(
            table_data,
            headers=["Layer Name", "Importance", "Bit Precision", "Pruning Factor (k)", "Sparsity (%)"],
            tablefmt="github"
        )

        card = f"""# Model Card: {variant_id}

## Executive Summary
- **Architecture:** {model_name.upper()}
- **Dataset:** {config.get('dataset', {}).get('name', 'CIFAR-10')}
- **Compression Method:** Adaptive Layer Importance Quantization & Pruning (Shinde NeurIPS 2024)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **FP32 Baseline Accuracy** | {metrics.get('baseline_accuracy', 0.0)*100:.2f}% |
| **Compressed Accuracy** | {metrics.get('compressed_accuracy', 0.0)*100:.2f}% |
| **Accuracy Drop Margin** | {metrics.get('accuracy_drop', 0.0)*100:.2f}% |
| **Weighted Avg Bit-Width (\(\\bar{{b}}\))** | **{metrics.get('average_bit_width', 0.0):.2f} bits** |
| **Huffman Avg Bit-Width** | **{metrics.get('huffman_average_bit_width', 0.0):.2f} bits** |
| **Overall Weight Sparsity (\(S_{{overall}}\))** | **{metrics.get('overall_sparsity', 0.0)*100:.2f}%** |

## Layer Breakdown
{layer_table}

---
*Generated automatically by Adaptive Compression ModelRegistry.*
"""
        return card
