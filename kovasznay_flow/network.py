"""Network definition for the Kovasznay PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Random Fourier feature mapping for 2D spatial coordinates."""

    def __init__(self, input_dim: int = 2, mapping_size: int = 50, scale: float = 1.0) -> None:
        super().__init__()
        basis = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.basis.dtype)
        x_proj = 2.0 * torch.pi * (x @ self.basis)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PINN(nn.Module):
    """Fully connected PINN with Fourier features."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = FourierEmbedding(input_dim=2, mapping_size=50, scale=1.0)

        self.hidden_layers = nn.ModuleList([nn.Linear(100, 50)])
        for _ in range(4):
            self.hidden_layers.append(nn.Linear(50, 50))

        self.output_layer = nn.Linear(50, 3)
        self.activation = nn.Tanh()

        self.w_bnd = nn.Parameter(torch.zeros(1))
        self.w_phys = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)