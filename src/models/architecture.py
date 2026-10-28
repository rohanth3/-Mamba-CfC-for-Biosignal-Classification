import torch
import torch.nn as nn
from ncps.torch import CfC
from .mamba_fallback import Mamba, MambaConfig

class Pooling(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x shape: (B, L, D)
        if self.mode == 'mean':
            return x.mean(dim=1)
        elif self.mode == 'last':
            return x[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")

class MambaOnlyModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, num_layers=2, pool_mode='mean', expand_factor=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_backbone = Mamba(MambaConfig(d_model=d_model, n_layers=num_layers, expand_factor=expand_factor))
        self.pooling = Pooling(mode=pool_mode)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba_backbone(x)
        x = self.pooling(x)
        return self.head(x)

class CfCOnlyModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, pool_mode='mean'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.cfc = CfC(input_size=d_model, units=64, mode='default', return_sequences=True, batch_first=True)
        self.pooling = Pooling(mode=pool_mode)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.cfc(x)
        x = self.pooling(x)
        return self.head(x)

class MambaCfCModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, num_layers=2, cfc_units=64, pool_mode='mean'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_backbone = Mamba(MambaConfig(d_model=d_model, n_layers=num_layers))
        self.projection = nn.Linear(d_model, cfc_units)
        self.cfc = CfC(input_size=cfc_units, units=cfc_units, mode='default', return_sequences=True, batch_first=True)
        self.pooling = Pooling(mode=pool_mode)
        self.head = nn.Linear(cfc_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba_backbone(x)
        x = self.projection(x)
        x, _ = self.cfc(x)
        x = self.pooling(x)
        return self.head(x)

