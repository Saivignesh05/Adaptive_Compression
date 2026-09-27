import torch
import torch.nn as nn
from src.statistical_scorer import StatisticalScorer

def test_statistical_scorer_basic():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    scorer = StatisticalScorer()
    # Dummy calib loader
    dummy_inputs = [torch.randn(4, 3, 32, 32) for _ in range(2)]
    calib_loader = [(x, torch.tensor([0, 1, 2, 3])) for x in dummy_inputs]

    imp_dict = scorer.compute_layer_importance(model, calib_loader, device="cpu")
    
    assert len(imp_dict) == 3 # 2 Conv2d + 1 Linear
    total_np = sum(v["N_P"] for v in imp_dict.values())
    assert abs(total_np - 1.0) < 1e-4

    for name, metrics in imp_dict.items():
        assert 0.0 <= metrics["importance"] <= 1.0
        assert 0.0 <= metrics["N_P"] <= 1.0
        assert metrics["N_V"] >= 0.0
