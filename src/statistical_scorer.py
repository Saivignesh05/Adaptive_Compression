import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .utils import get_compressible_layers

class StatisticalScorer:
    """
    Computes statistical importance scores for every compressible layer (Conv2d, Linear)
    in a Neural Network based on the paper:
    Shinde (NeurIPS 2024): 'Adaptive Quantization and Pruning of Deep Neural Networks via Layer Importance Estimation'
    """

    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            self.weights = {"w_P": 0.25, "w_E": 0.25, "w_V": 0.25, "w_S": 0.25}
        else:
            self.weights = weights

    def _compute_zero_order_entropy(self, tensor: torch.Tensor, num_bins: int = 256) -> float:
        """Computes zero-order Shannon entropy H0 of weight tensor."""
        weights_flat = tensor.detach().cpu().numpy().flatten()
        if len(weights_flat) == 0:
            return 0.0
        hist, _ = np.histogram(weights_flat, bins=num_bins, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(max(0.0, entropy))

    def _compute_activation_sparsity(self, model: nn.Module, calib_loader, device: str = "cpu") -> Dict[str, float]:
        """
        Registers forward hooks on compressible layers to measure the proportion of 
        zero or near-zero (|act| < 1e-4) activations during forward passes.
        """
        activation_counts = {}
        zero_counts = {}
        hooks = []

        layers = get_compressible_layers(model)

        def make_hook(layer_name):
            def hook(module, input, output):
                act = output.detach()
                total_act = act.numel()
                zeros = (torch.abs(act) < 1e-4).sum().item()
                
                if layer_name not in activation_counts:
                    activation_counts[layer_name] = 0
                    zero_counts[layer_name] = 0
                activation_counts[layer_name] += total_act
                zero_counts[layer_name] += zeros
            return hook

        for name, layer in layers:
            hooks.append(layer.register_forward_hook(make_hook(name)))

        model.eval()
        model.to(device)
        with torch.no_grad():
            for inputs, _ in calib_loader:
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.stack(inputs)
                inputs = inputs.to(device)
                model(inputs)

        for h in hooks:
            h.remove()

        sparsity_dict = {}
        for name, _ in layers:
            tot = activation_counts.get(name, 1)
            zer = zero_counts.get(name, 0)
            sparsity_dict[name] = float(zer / tot) if tot > 0 else 0.0

        return sparsity_dict

    def compute_layer_importance(self, model: nn.Module, calib_loader=None, device: str = "cpu", current_bit_precision: int = 32) -> Dict[str, Dict[str, float]]:
        """
        Computes layer importance for all compressible layers in the model.
        Returns a dictionary mapping layer_name -> {
            'N_P': float, 'N_E': float, 'N_V': float, 'S': float, 'importance': float
        }
        """
        compressible_layers = get_compressible_layers(model)
        if not compressible_layers:
            return {}

        total_params = sum(p.numel() for name, layer in compressible_layers for p in layer.parameters() if p.requires_grad)

        # 1. Parameter Share N_P(l)
        np_dict = {}
        for name, layer in compressible_layers:
            layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            np_dict[name] = layer_params / total_params if total_params > 0 else 0.0

        # 2. Zero-Order Entropy N_E(l)
        ne_dict = {}
        for name, layer in compressible_layers:
            weight = layer.weight.data
            entropy = self._compute_zero_order_entropy(weight)
            ne_dict[name] = entropy / current_bit_precision

        # 3. Normalized Weight Variance N_V(l)
        variances = {}
        for name, layer in compressible_layers:
            var = float(torch.var(layer.weight.data).item())
            variances[name] = var
        max_var = max(variances.values()) if variances and max(variances.values()) > 0 else 1.0

        nv_dict = {}
        for name, _ in compressible_layers:
            var = variances[name]
            nv_dict[name] = float(np.log(np.e - 1.0 + (var / max_var)))

        # 4. Activation Sparsity S(l)
        if calib_loader is not None:
            s_dict = self._compute_activation_sparsity(model, calib_loader, device=device)
        else:
            s_dict = {name: 0.0 for name, _ in compressible_layers}

        # Combine into Importance(l)
        results = {}
        w_P = self.weights.get("w_P", 0.25)
        w_E = self.weights.get("w_E", 0.25)
        w_V = self.weights.get("w_V", 0.25)
        w_S = self.weights.get("w_S", 0.25)

        for name, _ in compressible_layers:
            imp = w_P * np_dict[name] + w_E * ne_dict[name] + w_V * nv_dict[name] + w_S * s_dict[name]
            results[name] = {
                "N_P": float(np_dict[name]),
                "N_E": float(ne_dict[name]),
                "N_V": float(nv_dict[name]),
                "S": float(s_dict[name]),
                "importance": float(imp)
            }

        return results
