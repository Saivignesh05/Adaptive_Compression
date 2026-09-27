import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from .statistical_scorer import StatisticalScorer
from .quantizer import UniformQuantizer
from .pruner import AdaptivePruner
from .huffman import HuffmanEncoder
from .utils import evaluate_model, get_compressible_layers

class AdaptiveSearchEngine:
    """
    Search engine for Layer-wise Bit-width Selection & Pruning Algorithm.
    Reference: Shinde (NeurIPS 2024), Section 2.3.
    """

    def __init__(self, config: dict):
        self.config = config
        self.target_margin = config["compression"].get("target_margin", 0.001)
        self.first_last_factor = config["compression"].get("first_last_margin_factor", 0.5)
        self.weights = config["compression"].get("weights", {"w_P": 0.25, "w_E": 0.25, "w_V": 0.25, "w_S": 0.25})
        self.bit_range = config["compression"].get("bit_width_range", [1, 8])
        self.k_range = config["compression"].get("k_factor_range", [0.0, 3.0])
        self.k_step = config["compression"].get("k_factor_step", 0.25)

        self.scorer = StatisticalScorer(weights=self.weights)
        self.quantizer = UniformQuantizer()
        self.pruner = AdaptivePruner()

    def run_adaptive_search(self, model: nn.Module, test_loader, calib_loader=None, device: str = "cpu") -> Tuple[nn.Module, Dict]:
        """
        Runs the full adaptive layer-importance-guided quantization & pruning search.
        Returns:
            compressed_model: nn.Module with per-layer quantization and pruning applied
            metrics: Dict containing accuracy, average bit-width, layer-wise bit-widths, k-factors, and sparsity.
        """
        model.eval()
        model.to(device)

        # Step 1: Baseline Evaluation
        base_accuracy = evaluate_model(model, test_loader, device=device)
        print(f"[Search Engine] FP32 Baseline Accuracy: {base_accuracy * 100:.2f}%")

        # Step 2: Importance Estimation
        importance_dict = self.scorer.compute_layer_importance(model, calib_loader, device=device)
        compressible_layers = get_compressible_layers(model)
        layer_names = [name for name, _ in compressible_layers]

        if not layer_names:
            return model, {"accuracy": base_accuracy, "avg_bit_width": 32.0}

        first_layer_name = layer_names[0]
        last_layer_name = layer_names[-1]

        # Step 3: Layer Margin Budget & Importance Ranking
        ranked_layers = sorted(layer_names, key=lambda n: importance_dict[n]["importance"], reverse=True)

        layer_margins = {}
        for name in layer_names:
            imp = importance_dict[name]["importance"]
            factor = self.first_last_factor if (name == first_layer_name or name == last_layer_name) else 1.0
            layer_margins[name] = self.target_margin * imp * factor

        # Initialize per-layer search state
        current_bit_widths = {name: 8 for name in layer_names} # Start from 8-bit quantization baseline
        current_k_factors = {name: 0.0 for name in layer_names}  # Start with no pruning

        # Current working model
        compressed_model = copy.deepcopy(model)

        # Step 4: Iterative Importance-Guided Search
        print("[Search Engine] Starting iterative per-layer optimization...")
        for name in ranked_layers:
            imp_val = importance_dict[name]["importance"]
            allowed_drop = layer_margins[name]
            min_target_acc = base_accuracy - allowed_drop

            best_b = current_bit_widths[name]
            best_k = current_k_factors[name]

            # Try lowest bit-width first (from min_bit to 8)
            for candidate_b in range(self.bit_range[0], self.bit_range[1] + 1):
                # Try highest k factor (from max_k down to 0)
                k_candidates = np.arange(self.k_range[1], self.k_range[0] - self.k_step, -self.k_step)
                found_valid = False

                for candidate_k in k_candidates:
                    temp_bits = copy.deepcopy(current_bit_widths)
                    temp_ks = copy.deepcopy(current_k_factors)
                    temp_bits[name] = int(candidate_b)
                    temp_ks[name] = float(candidate_k)

                    # Apply pruning then quantization
                    test_model, _ = self.pruner.prune_model(model, temp_ks)
                    test_model = self.quantizer.quantize_model(test_model, temp_bits)

                    acc = evaluate_model(test_model, test_loader, device=device)
                    if acc >= min_target_acc:
                        best_b = int(candidate_b)
                        best_k = float(candidate_k)
                        found_valid = True
                        break

                if found_valid:
                    break

            current_bit_widths[name] = best_b
            current_k_factors[name] = best_k

        # Step 5: Construct Final Compressed Model & Metrics
        compressed_model, sparsity_dict = self.pruner.prune_model(model, current_k_factors)
        compressed_model = self.quantizer.quantize_model(compressed_model, current_bit_widths)
        final_accuracy = evaluate_model(compressed_model, test_loader, device=device)

        # Compute Weighted Average Bit-Width b_bar
        layer_np = {n: importance_dict[n]["N_P"] for n in layer_names}
        overall_sparsity = self.pruner.compute_overall_sparsity(compressed_model, layer_np)

        b_bar = sum(
            current_bit_widths[n] * (1.0 - sparsity_dict[n]) * layer_np[n]
            for n in layer_names
        )

        # Compute Huffman Encoded Avg Bit-Width
        huffman_bit_widths = []
        for name, layer in get_compressible_layers(compressed_model):
            b_huff = HuffmanEncoder.compute_huffman_bit_width(layer.weight.data)
            huffman_bit_widths.append(b_huff * layer_np[name])
        b_bar_huffman = float(sum(huffman_bit_widths))

        metrics = {
            "baseline_accuracy": float(base_accuracy),
            "compressed_accuracy": float(final_accuracy),
            "accuracy_drop": float(base_accuracy - final_accuracy),
            "average_bit_width": float(b_bar),
            "huffman_average_bit_width": float(b_bar_huffman),
            "overall_sparsity": float(overall_sparsity),
            "layer_bit_widths": current_bit_widths,
            "layer_k_factors": current_k_factors,
            "layer_sparsity": sparsity_dict,
            "layer_importance": {n: importance_dict[n]["importance"] for n in layer_names}
        }

        print(f"[Search Engine] Final Compressed Accuracy: {final_accuracy * 100:.2f}% (Drop: {(base_accuracy - final_accuracy)*100:.2f}%)")
        print(f"[Search Engine] Weighted Avg Bit-Width (b_bar): {b_bar:.2f} bits (Huffman: {b_bar_huffman:.2f} bits)")
        print(f"[Search Engine] Overall Sparsity: {overall_sparsity * 100:.2f}%")

        return compressed_model, metrics
