"""PyTorch MLP for the cardiovascular dataset research demo."""
import torch
from torch import nn

class CardioMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.BatchNorm1d(width),
                           nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
