import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple
from .utils import get_compressible_layers

class AdaptivePruner:
    """
    Layer-wise adaptive pruning scheme based on standard deviation threshold Z_T(l) = k(l) * sigma(W_l).
    Reference: Shinde (NeurIPS 2024), Section 2.2.
    """

    @staticmethod
    def prune_layer_weights(weight_tensor: torch.Tensor, k_factor: float) -> Tuple[torch.Tensor, float, float]:
        """
        Prunes weights with absolute value <= Z_T(l) = k_factor * std(weight_tensor).
        Returns pruned weight tensor, zero-threshold Z_T, and sparsity ratio.
        """
        if k_factor <= 0.0:
            return weight_tensor.clone(), 0.0, 0.0

        std_val = float(torch.std(weight_tensor).item())
        z_t = k_factor * std_val

        mask = torch.abs(weight_tensor) > z_t
        pruned_weights = weight_tensor * mask.float()

        total_num = weight_tensor.numel()
        pruned_num = (pruned_weights == 0).sum().item()
        sparsity = float(pruned_num / total_num) if total_num > 0 else 0.0

        return pruned_weights, z_t, sparsity

    def prune_model(self, model: nn.Module, layer_k_factors: Dict[str, float]) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Applies layer-wise adaptive pruning to all compressible layers in the model.
        layer_k_factors: dict mapping layer_name -> k_factor (e.g. 0.0 to 3.0)
        Returns pruned model and dict of per-layer sparsity ratios.
        """
        pruned_model = copy.deepcopy(model)
        compressible = get_compressible_layers(pruned_model)
        sparsity_dict = {}

        for name, layer in compressible:
            k = layer_k_factors.get(name, 0.0)
            p_weight, _, sparsity = self.prune_layer_weights(layer.weight.data, k)
            layer.weight.data.copy_(p_weight)
            sparsity_dict[name] = sparsity

        return pruned_model, sparsity_dict

    def compute_overall_sparsity(self, model: nn.Module, layer_np: Dict[str, float]) -> float:
        """Computes weighted average overall sparsity: S_overall = sum(S(l) * N_P(l))."""
        compressible = get_compressible_layers(model)
        overall_sparsity = 0.0

        for name, layer in compressible:
            weights = layer.weight.data
            total = weights.numel()
            zeros = (weights == 0).sum().item()
            s_l = zeros / total if total > 0 else 0.0
            n_p_l = layer_np.get(name, 0.0)
            overall_sparsity += s_l * n_p_l

        return float(overall_sparsity)
