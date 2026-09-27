import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.search_engine import AdaptiveSearchEngine

def test_adaptive_search_engine_basic():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 32 * 32, 10)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyModel()
    dummy_data = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(20)]
    loader = DataLoader(dummy_data, batch_size=4)

    config = {
        "compression": {
            "target_margin": 0.05,
            "first_last_margin_factor": 0.5,
            "weights": {"w_P": 0.25, "w_E": 0.25, "w_V": 0.25, "w_S": 0.25},
            "bit_width_range": [2, 8],
            "k_factor_range": [0.0, 1.0],
            "k_factor_step": 0.5
        }
    }

    engine = AdaptiveSearchEngine(config)
    compressed_model, metrics = engine.run_adaptive_search(model, loader, calib_loader=loader, device="cpu")

    assert compressed_model is not None
    assert "average_bit_width" in metrics
    assert metrics["average_bit_width"] <= 8.0
