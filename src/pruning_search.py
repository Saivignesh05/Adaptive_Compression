"""
Pruning-Only Adaptive Search Engine
=====================================
Implements the adaptive pruning-only variant.
Sweeps k-factor (from high to low) while keeping weights at FP32 (32-bit).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .statistical_scorer import StatisticalScorer
from .pruner import AdaptivePruner
from .utils import evaluate_model, get_compressible_layers

class PruningOnlySearchEngine:
    def __init__(self, config: dict):
        self.config = config
        self.target_margin = config["compression"].get("target_margin", 0.001)
        self.first_last_factor = config["compression"].get("first_last_margin_factor", 0.5)
        self.weights = config["compression"].get("weights", {"w_P": 0.25, "w_E": 0.25, "w_V": 0.25, "w_S": 0.25})
        self.k_range = config["compression"].get("k_factor_range", [0.0, 3.0])
        self.k_step = config["compression"].get("k_factor_step", 0.25)

        self.scorer = StatisticalScorer(weights=self.weights)
        self.pruner = AdaptivePruner()

    def run_pruning_search(self, model: nn.Module, test_loader, calib_loader=None, device: str = "cpu") -> Tuple[nn.Module, Dict]:
        model.eval()
        model.to(device)

        base_accuracy = evaluate_model(model, test_loader, device=device)
        
        importance_dict = self.scorer.compute_layer_importance(model, calib_loader, device=device)
        compressible_layers = get_compressible_layers(model)
        layer_names = [name for name, _ in compressible_layers]

        if not layer_names:
            return model, {"accuracy": base_accuracy, "overall_sparsity": 0.0}

        first_layer = layer_names[0]
        last_layer = layer_names[-1]

        ranked_layers = sorted(layer_names, key=lambda n: importance_dict[n]["importance"])

        layer_margins = {}
        for name in layer_names:
            imp = importance_dict[name]["importance"]
            factor = self.first_last_factor if (name == first_layer or name == last_layer) else 1.0
            layer_margins[name] = self.target_margin * imp * factor

        current_k_factors: Dict[str, float] = {name: 0.0 for name in layer_names}

        for name in ranked_layers:
            allowed_drop = layer_margins[name]
            min_target_acc = base_accuracy - allowed_drop

            best_k = current_k_factors[name]
            
            # Sweep k from high (aggressive pruning) down to low (conservative)
            k_candidates = np.arange(self.k_range[1], self.k_range[0] - self.k_step, -self.k_step)
            
            for candidate_k in k_candidates:
                trial_ks = copy.copy(current_k_factors)
                trial_ks[name] = float(candidate_k)

                trial_model, _ = self.pruner.prune_model(model, trial_ks)
                acc = evaluate_model(trial_model, test_loader, device=device)

                if acc >= min_target_acc:
                    best_k = float(candidate_k)
                    break 

            current_k_factors[name] = best_k

        compressed_model, sparsity_dict = self.pruner.prune_model(model, current_k_factors)
        final_accuracy = evaluate_model(compressed_model, test_loader, device=device)

        layer_np = {n: importance_dict[n]["N_P"] for n in layer_names}
        overall_sparsity = self.pruner.compute_overall_sparsity(compressed_model, layer_np)

        metrics = {
            "baseline_accuracy": float(base_accuracy),
            "compressed_accuracy": float(final_accuracy),
            "accuracy_drop": float(base_accuracy - final_accuracy),
            "overall_sparsity": float(overall_sparsity),
            "average_bit_width": 32.0,  # FP32
            "layer_k_factors": current_k_factors,
            "layer_sparsity": sparsity_dict,
        }

        return compressed_model, metrics
