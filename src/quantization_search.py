"""
Quantization-Only Adaptive Search Engine
==========================================
Implements the quantization-only variant of the Adaptive Layer Importance
Estimation pipeline from Shinde (NeurIPS 2024), Section 2.3.

Key differences from the full AdaptiveSearchEngine:
  - NO pruning (k-factor always 0, sparsity always 0%)
  - Search is bit-width only: for each layer, find the minimum b in [1, 8]
    that keeps accuracy within the per-layer margin budget.
  - Weighted average bit-width b_bar uses N_P(l) directly (no sparsity term).

Paper reference formulas:
  Importance(l) = w_P * N_P(l) + w_E * N_E(l) + w_V * N_V(l) + w_S * S(l)
  b_bar = sum_l  b(l) * N_P(l)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .statistical_scorer import StatisticalScorer
from .quantizer import UniformQuantizer
from .huffman import HuffmanEncoder
from .utils import evaluate_model, get_compressible_layers


class QuantizationOnlySearchEngine:
    """
    Importance-guided per-layer bit-width search engine (quantization only).

    Algorithm outline (from the paper, quantization-only path):
      1. Evaluate FP32 baseline accuracy A_base.
      2. Compute importance I(l) for every compressible layer.
      3. Sort layers by importance ascending (least important first --
         these are the easiest to compress aggressively).
      4. For each layer l (in ascending importance order):
         a. Compute layer margin budget:
              margin(l) = T_margin * I(l) * factor
            where factor = first_last_margin_factor for the first/last
            compressible layer, 1.0 otherwise.
         b. Sweep bit-widths b = 1, 2, ..., 8:
              - Quantize only layer l with bit-width b (all others at
                their already-assigned bit-widths).
              - Evaluate accuracy A_b on the test set.
              - If A_b >= A_base - margin(l): accept b and move on.
         c. If no bit-width satisfies the margin, keep the layer at
            its current precision (default 8).
      5. Build final quantized model; report metrics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.target_margin = config["compression"].get("target_margin", 0.001)
        self.first_last_factor = config["compression"].get("first_last_margin_factor", 0.5)
        self.weights = config["compression"].get(
            "weights", {"w_P": 0.25, "w_E": 0.25, "w_V": 0.25, "w_S": 0.25}
        )
        self.bit_range = config["compression"].get("bit_width_range", [1, 8])

        self.scorer = StatisticalScorer(weights=self.weights)
        self.quantizer = UniformQuantizer()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_quantization_search(
        self,
        model: nn.Module,
        test_loader,
        calib_loader=None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict]:
        """
        Runs the adaptive per-layer quantization search (no pruning).

        Args:
            model: Pre-trained FP32 model.
            test_loader: DataLoader for accuracy evaluation.
            calib_loader: DataLoader for activation sparsity measurement.
            device: 'cpu' or 'cuda'.

        Returns:
            quantized_model: nn.Module with per-layer quantized weights.
            metrics: dict with accuracy, bit-widths, b_bar, etc.
        """
        model.eval()
        model.to(device)

        # -- Step 1: Baseline accuracy --
        base_accuracy = evaluate_model(model, test_loader, device=device)
        print(f"[Quantization Search] FP32 Baseline Accuracy: {base_accuracy * 100:.2f}%")

        # -- Step 2: Layer importance estimation --
        importance_dict = self.scorer.compute_layer_importance(
            model, calib_loader, device=device
        )
        compressible_layers = get_compressible_layers(model)
        layer_names = [name for name, _ in compressible_layers]

        if not layer_names:
            print("[Quantization Search] No compressible layers found.")
            return model, {"accuracy": base_accuracy, "avg_bit_width": 32.0}

        first_layer = layer_names[0]
        last_layer = layer_names[-1]

        # Print importance overview
        print(f"\n{'-' * 65}")
        print(f"  {'Layer':<40s} {'Importance':>10s}")
        print(f"{'-' * 65}")
        for name in layer_names:
            imp = importance_dict[name]["importance"]
            tag = ""
            if name == first_layer:
                tag = " (first)"
            elif name == last_layer:
                tag = " (last)"
            print(f"  {name + tag:<40s} {imp:>10.6f}")
        print(f"{'-' * 65}\n")

        # -- Step 3: Rank layers -- ascending importance (easiest first)
        ranked_layers = sorted(
            layer_names,
            key=lambda n: importance_dict[n]["importance"],
        )

        # Per-layer margin budgets
        layer_margins = {}
        for name in layer_names:
            imp = importance_dict[name]["importance"]
            factor = (
                self.first_last_factor
                if (name == first_layer or name == last_layer)
                else 1.0
            )
            layer_margins[name] = self.target_margin * imp * factor

        # -- Step 4: Greedy per-layer bit-width search --
        current_bit_widths: Dict[str, int] = {name: 8 for name in layer_names}

        print("[Quantization Search] Starting per-layer bit-width optimization ...")
        for idx, name in enumerate(ranked_layers, 1):
            imp_val = importance_dict[name]["importance"]
            allowed_drop = layer_margins[name]
            min_target_acc = base_accuracy - allowed_drop

            best_b = current_bit_widths[name]  # fallback: keep current

            # Sweep from lowest to highest bit-width
            for candidate_b in range(self.bit_range[0], self.bit_range[1] + 1):
                trial_bits = copy.copy(current_bit_widths)
                trial_bits[name] = candidate_b

                trial_model = self.quantizer.quantize_model(model, trial_bits)
                acc = evaluate_model(trial_model, test_loader, device=device)

                if acc >= min_target_acc:
                    best_b = candidate_b
                    break  # lowest valid bit-width found

            current_bit_widths[name] = best_b
            print(
                f"  [{idx}/{len(ranked_layers)}] {name:<40s} -> "
                f"{best_b}-bit  (importance={imp_val:.6f}, "
                f"margin={allowed_drop * 100:.4f}%)"
            )

        # -- Step 5: Build final quantized model --
        quantized_model = self.quantizer.quantize_model(model, current_bit_widths)
        final_accuracy = evaluate_model(quantized_model, test_loader, device=device)

        # Weighted average bit-width: b_bar = sum( b(l) * N_P(l) )
        layer_np = {n: importance_dict[n]["N_P"] for n in layer_names}
        b_bar = sum(
            current_bit_widths[n] * layer_np[n] for n in layer_names
        )

        # Huffman-encoded average bit-width for comparison
        huffman_bit_widths = []
        for name, layer in get_compressible_layers(quantized_model):
            b_huff = HuffmanEncoder.compute_huffman_bit_width(layer.weight.data)
            huffman_bit_widths.append(b_huff * layer_np[name])
        b_bar_huffman = float(sum(huffman_bit_widths))

        # -- Metrics dict --
        metrics = {
            "baseline_accuracy": float(base_accuracy),
            "compressed_accuracy": float(final_accuracy),
            "accuracy_drop": float(base_accuracy - final_accuracy),
            "average_bit_width": float(b_bar),
            "huffman_average_bit_width": float(b_bar_huffman),
            "overall_sparsity": 0.0,  # no pruning
            "layer_bit_widths": current_bit_widths,
            "layer_k_factors": {n: 0.0 for n in layer_names},  # no pruning
            "layer_sparsity": {n: 0.0 for n in layer_names},   # no pruning
            "layer_importance": {
                n: importance_dict[n]["importance"] for n in layer_names
            },
        }

        # -- Summary --
        print(f"\n{'=' * 65}")
        print(f"  QUANTIZATION-ONLY RESULTS")
        print(f"{'=' * 65}")
        print(f"  FP32 Baseline Accuracy  : {base_accuracy * 100:.2f}%")
        print(f"  Quantized Accuracy      : {final_accuracy * 100:.2f}%")
        print(f"  Accuracy Drop           : {(base_accuracy - final_accuracy) * 100:.4f}%")
        print(f"  Weighted Avg Bit-Width  : {b_bar:.4f} bits")
        print(f"  Huffman Avg Bit-Width   : {b_bar_huffman:.4f} bits")
        print(f"  Compression Ratio (FP32): {32.0 / max(b_bar, 1e-8):.2f}x")
        print(f"{'=' * 65}\n")

        return quantized_model, metrics
