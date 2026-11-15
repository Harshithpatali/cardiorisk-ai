"""PyTorch MLP for the cardiovascular dataset research demo."""
import torch
from torch import nn
class CardioMLP(nn.Module):
    def __init__(self,input_dim:int,dropout:float=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(dropout),
          nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(dropout),
          nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(dropout*.5),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)
