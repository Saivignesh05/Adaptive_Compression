import heapq
from collections import Counter
import torch
import numpy as np

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanEncoder:
    """Simulates Huffman Coding on quantized tensor symbols to measure effective compressed bit-width."""

    @staticmethod
    def compute_huffman_bit_width(tensor: torch.Tensor, scale: float = 1.0) -> float:
        """
        Computes the average bit-width per non-zero weight parameter after Huffman encoding.
        """
        flat = tensor.detach().cpu().numpy().flatten()
        if len(flat) == 0:
            return 0.0

        # Discretize symbols by scale
        symbols = (flat / (scale if scale > 0 else 1.0)).round().astype(int)
        freq_dict = Counter(symbols)

        if len(freq_dict) <= 1:
            return 1.0

        heap = [HuffmanNode(sym, freq) for sym, freq in freq_dict.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            merged = HuffmanNode(None, n1.freq + n2.freq)
            merged.left = n1
            merged.right = n2
            heapq.heappush(heap, merged)

        root = heap[0]
        code_lengths = {}

        def traverse(node, current_len):
            if node is None:
                return
            if node.symbol is not None:
                code_lengths[node.symbol] = max(1, current_len)
                return
            traverse(node.left, current_len + 1)
            traverse(node.right, current_len + 1)

        traverse(root, 0)

        total_bits = sum(freq_dict[sym] * code_lengths[sym] for sym in freq_dict)
        avg_bit_width = total_bits / len(flat)
        return float(avg_bit_width)
