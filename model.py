"""
model.py
Simple neural network for heart disease classification.
Designed to be lightweight enough for CPU-only rural clinic nodes.
"""

import torch
import torch.nn as nn


class HeartDiseaseModel(nn.Module):
    """
    Lightweight MLP for UCI Heart Disease dataset.
    13 input features -> binary classification (disease / no disease).
    Small enough to train on CPU in seconds per round.
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64):
        super(HeartDiseaseModel, self).__init__()
        # LayerNorm instead of BatchNorm1d — works with ANY batch size including 1
        # BatchNorm crashes when a rural clinic has very few samples per batch
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_parameters(model: nn.Module) -> list:
    """Extract model parameters as a list of numpy arrays."""
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list) -> nn.Module:
    """Load a list of numpy arrays into the model parameters."""
    import numpy as np
    params_iter = zip(model.parameters(), parameters)
    for param, new_val in params_iter:
        param.data = torch.tensor(new_val, dtype=param.data.dtype)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
