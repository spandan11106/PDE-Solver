"""Model definitions for the unsteady cylinder PINN."""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Random Fourier feature mapping for 3D input (t, x, y)."""

    def __init__(self, input_dim: int = 3, mapping_size: int = 50, scale: float = 1.0) -> None:
        super().__init__()
        basis = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.basis.dtype)
        x_proj = 2.0 * torch.pi * (x @ self.basis)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PINN(nn.Module):
    """PINN backbone with Fourier features and gated hidden mixing."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = FourierEmbedding(input_dim=3, mapping_size=50, scale=1.0)

        hidden_dim = 128
        embed_dim = 100

        self.encoder_u = nn.Linear(embed_dim, hidden_dim)
        self.encoder_v = nn.Linear(embed_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([nn.Linear(embed_dim, hidden_dim)])
        for _ in range(4):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 3)
        self.activation = nn.Tanh()

        self.w_bnd = nn.Parameter(torch.zeros(1))
        self.w_phys = nn.Parameter(torch.zeros(1))

    def init_weights(self) -> None:
        """Apply Xavier initialization to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed = self.embedding(x)

        u_gate = self.activation(self.encoder_u(x_embed))
        v_gate = self.activation(self.encoder_v(x_embed))

        hidden = self.activation(self.hidden_layers[0](x_embed))
        for layer in self.hidden_layers[1:]:
            z = self.activation(layer(hidden))
            hidden = (1.0 - z) * u_gate + z * v_gate

        return self.output_layer(hidden)