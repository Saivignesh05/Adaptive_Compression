import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple
from .utils import get_compressible_layers

class UniformQuantizer:
    """
    Uniform Symmetric Post-Training Quantizer supporting per-layer bit precision (1 to 8 bits).
    """

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, bit_width: int) -> Tuple[torch.Tensor, float]:
        """
        Quantizes a tensor to specified bit_width (1-8 bits).
        Returns dequantized tensor (same scale) and scale factor.
        """
        if bit_width >= 32:
            return tensor.clone(), 1.0

        if bit_width == 1:
            # Binary quantization: sign(W) * mean(|W|)
            scale = float(torch.mean(torch.abs(tensor)).item())
            if scale == 0:
                scale = 1e-8
            quantized = torch.sign(tensor) * scale
            return quantized, scale

        q_max = (1 << (bit_width - 1)) - 1
        q_min = -(1 << (bit_width - 1))
        
        max_val = float(torch.max(torch.abs(tensor)).item())
        if max_val == 0:
            return tensor.clone(), 1.0

        scale = max_val / q_max
        q_tensor = torch.clamp(torch.round(tensor / scale), q_min, q_max)
        dequantized = q_tensor * scale
        return dequantized, scale

    def quantize_model(self, model: nn.Module, layer_bit_widths: Dict[str, int]) -> nn.Module:
        """
        Applies layer-wise bit-width quantization to all compressible layers in the model.
        layer_bit_widths: dict mapping layer_name -> bit_width (1 to 8, or 32 for FP32)
        """
        quantized_model = copy.deepcopy(model)
        compressible = get_compressible_layers(quantized_model)

        for name, layer in compressible:
            bit_width = layer_bit_widths.get(name, 32)
            if bit_width < 32:
                q_weight, _ = self.quantize_tensor(layer.weight.data, bit_width)
                layer.weight.data.copy_(q_weight)

        return quantized_model
