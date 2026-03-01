"""
model.py
Simple neural network for heart disease classification.
Designed to be lightweight enough for CPU-only rural clinic nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeartDiseaseModel(nn.Module):
    """
    Lightweight MLP for UCI Heart Disease dataset.
    13 input features -> binary classification (disease / no disease).
    Small enough to train on CPU in seconds per round.
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64):
        super(HeartDiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


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
