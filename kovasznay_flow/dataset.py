"""Sampling utilities for steady Kovasznay-flow PINN training."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc

from kovasznay import kovasznay_solution


def generate_points(
    n_colloc: int = 5000,
    n_bnd: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate collocation and boundary points for the Kovasznay domain."""
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 0.5

    sampler_2d = qmc.LatinHypercube(d=2)
    lhs_samples = sampler_2d.random(n=n_colloc)
    x_c = x_min + (x_max - x_min) * lhs_samples[:, 0:1]
    y_c = y_min + (y_max - y_min) * lhs_samples[:, 1:2]
    x_colloc = torch.tensor(np.hstack([x_c, y_c]), dtype=torch.float32)

    sampler_1d = qmc.LatinHypercube(d=1)

    y_left = y_min + (y_max - y_min) * sampler_1d.random(n=n_bnd)
    x_left = np.full((n_bnd, 1), x_min)
    x_left_bnd = torch.tensor(np.hstack([x_left, y_left]), dtype=torch.float32)

    y_right = y_min + (y_max - y_min) * sampler_1d.random(n=n_bnd)
    x_right = np.full((n_bnd, 1), x_max)
    x_right_bnd = torch.tensor(np.hstack([x_right, y_right]), dtype=torch.float32)

    x_top = x_min + (x_max - x_min) * sampler_1d.random(n=n_bnd)
    y_top = np.full((n_bnd, 1), y_max)
    x_top_bnd = torch.tensor(np.hstack([x_top, y_top]), dtype=torch.float32)

    x_bottom = x_min + (x_max - x_min) * sampler_1d.random(n=n_bnd)
    y_bottom = np.full((n_bnd, 1), y_min)
    x_bottom_bnd = torch.tensor(np.hstack([x_bottom, y_bottom]), dtype=torch.float32)

    x_bnd = torch.cat([x_left_bnd, x_right_bnd, x_top_bnd, x_bottom_bnd], dim=0)
    u_bnd, v_bnd, p_bnd = kovasznay_solution(x_bnd[:, 0:1], x_bnd[:, 1:2])

    return x_colloc, x_bnd, u_bnd, v_bnd, p_bnd