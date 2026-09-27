import torch
import torch.nn as nn
from src.quantizer import UniformQuantizer
from src.pruner import AdaptivePruner
from src.huffman import HuffmanEncoder

def test_quantizer_and_pruner():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Linear(8 * 32 * 32, 10)
    )

    quantizer = UniformQuantizer()
    pruner = AdaptivePruner()

    # Test 4-bit quantization
    q_model = quantizer.quantize_model(model, {"0": 4, "1": 2})
    assert q_model is not None

    # Test Pruning
    p_model, sparsity_dict = pruner.prune_model(model, {"0": 1.0, "1": 1.5})
    assert "0" in sparsity_dict
    assert sparsity_dict["0"] > 0.0

    # Test Huffman
    sample_tensor = torch.randn(10, 10)
    q_tensor, scale = quantizer.quantize_tensor(sample_tensor, 4)
    huff_bits = HuffmanEncoder.compute_huffman_bit_width(q_tensor, scale)
    assert huff_bits > 0.0
