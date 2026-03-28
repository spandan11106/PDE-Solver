"""Analytical Kovasznay flow solution for boundary supervision and evaluation."""

from __future__ import annotations

import math

import torch


def kovasznay_solution(
    x: torch.Tensor,
    y: torch.Tensor,
    Re: float = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact velocity and pressure fields for the Kovasznay benchmark."""
    lmbda = Re / 2.0 - math.sqrt((Re**2) / 4.0 + 4.0 * math.pi**2)

    u = 1.0 - torch.exp(lmbda * x) * torch.cos(2.0 * math.pi * y)
    v = (lmbda / (2.0 * math.pi)) * torch.exp(lmbda * x) * torch.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - torch.exp(2.0 * lmbda * x))
    return u, v, p