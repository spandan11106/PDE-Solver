"""Loss and residual utilities for the cylinder flow PINN."""

from __future__ import annotations

import torch


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


def _residual_terms(model, x: torch.Tensor, re: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x.requires_grad_(True)
    preds = model(x)
    u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    grads_u = _grad(u, x)
    u_t, u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]

    grads_v = _grad(v, x)
    v_t, v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]

    grads_p = _grad(p, x)
    p_x, p_y = grads_p[:, 1:2], grads_p[:, 2:3]

    u_xx = _grad(u_x, x)[:, 1:2]
    u_yy = _grad(u_y, x)[:, 2:3]
    v_xx = _grad(v_x, x)[:, 1:2]
    v_yy = _grad(v_y, x)[:, 2:3]

    f_cont = u_x + v_y
    f_u = u_t + u * u_x + v * u_y + p_x - (1.0 / re) * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - (1.0 / re) * (v_xx + v_yy)
    return f_cont, f_u, f_v


def compute_loss(
    model,
    x_colloc: torch.Tensor,
    x_bnd: torch.Tensor,
    u_bnd: torch.Tensor,
    v_bnd: torch.Tensor,
    Re: float = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return boundary and physics MSE terms."""
    preds_bnd = model(x_bnd)
    u_pred, v_pred = preds_bnd[:, 0:1], preds_bnd[:, 1:2]
    mse_bnd = torch.mean((u_pred - u_bnd) ** 2) + torch.mean((v_pred - v_bnd) ** 2)

    f_cont, f_u, f_v = _residual_terms(model, x_colloc, Re)
    mse_physics = torch.mean(f_cont**2) + torch.mean(f_u**2) + torch.mean(f_v**2)
    return mse_bnd, mse_physics


def compute_pointwise_physics_residual(model, x_candidate: torch.Tensor, Re: float = 100) -> torch.Tensor:
    """Return per-point residual magnitude used by RAR sampling."""
    f_cont, f_u, f_v = _residual_terms(model, x_candidate, Re)
    return (f_cont**2 + f_u**2 + f_v**2).squeeze()