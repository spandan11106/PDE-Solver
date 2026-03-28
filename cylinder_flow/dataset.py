"""Sampling utilities for unsteady cylinder flow PINN training."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc


def _sample_collocation_outside_cylinder(
    n_colloc: int,
    t_min: float,
    t_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cyl_x: float,
    cyl_y: float,
    cyl_r: float,
) -> np.ndarray:
    """Sample collocation points and reject points inside the cylinder."""
    points = []
    collected = 0
    sampler = qmc.LatinHypercube(d=3)

    while collected < n_colloc:
        candidate_count = max(2 * (n_colloc - collected), 1024)
        lhs_samples = sampler.random(n=candidate_count)

        t_c = t_min + (t_max - t_min) * lhs_samples[:, 0:1]
        x_c = x_min + (x_max - x_min) * lhs_samples[:, 1:2]
        y_c = y_min + (y_max - y_min) * lhs_samples[:, 2:3]

        mask = ((x_c - cyl_x) ** 2 + (y_c - cyl_y) ** 2) >= cyl_r**2
        accepted = np.hstack([t_c[mask], x_c[mask], y_c[mask]])
        points.append(accepted)
        collected += accepted.shape[0]

    return np.vstack(points)[:n_colloc]


def generate_points(
    n_colloc: int = 5000,
    n_bnd: int = 500,
    t_max: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate collocation points and boundary targets for cylinder flow."""
    t_min = 0.0
    x_min, x_max = -1.0, 5.0
    y_min, y_max = -2.0, 2.0

    cyl_r = 0.5
    cyl_x, cyl_y = 0.0, 0.0

    colloc_np = _sample_collocation_outside_cylinder(
        n_colloc=n_colloc,
        t_min=t_min,
        t_max=t_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cyl_x=cyl_x,
        cyl_y=cyl_y,
        cyl_r=cyl_r,
    )
    x_colloc = torch.tensor(colloc_np, dtype=torch.float32)

    sampler_2d = qmc.LatinHypercube(d=2)

    inlet_samples = sampler_2d.random(n=n_bnd)
    t_in = t_min + (t_max - t_min) * inlet_samples[:, 0:1]
    y_in = y_min + (y_max - y_min) * inlet_samples[:, 1:2]
    x_in = np.full((n_bnd, 1), x_min)
    x_inlet = np.hstack([t_in, x_in, y_in])
    u_inlet = np.hstack([np.ones((n_bnd, 1)), np.zeros((n_bnd, 1))])

    cyl_samples = sampler_2d.random(n=n_bnd)
    t_cyl = t_min + (t_max - t_min) * cyl_samples[:, 0:1]
    theta = 2.0 * np.pi * cyl_samples[:, 1:2]
    x_cyl = cyl_x + cyl_r * np.cos(theta)
    y_cyl = cyl_y + cyl_r * np.sin(theta)
    x_cylinder = np.hstack([t_cyl, x_cyl, y_cyl])
    u_cylinder = np.zeros((n_bnd, 2))

    x_bnd = torch.tensor(np.vstack([x_inlet, x_cylinder]), dtype=torch.float32)
    u_bnd = torch.tensor(np.vstack([u_inlet[:, 0:1], u_cylinder[:, 0:1]]), dtype=torch.float32)
    v_bnd = torch.tensor(np.vstack([u_inlet[:, 1:2], u_cylinder[:, 1:2]]), dtype=torch.float32)

    return x_colloc, x_bnd, u_bnd, v_bnd