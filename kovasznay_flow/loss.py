"""Loss and residual definitions for the Kovasznay PINN."""

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
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]

    grads_v = _grad(v, x)
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]

    grads_p = _grad(p, x)
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

    u_xx = _grad(u_x, x)[:, 0:1]
    u_yy = _grad(u_y, x)[:, 1:2]
    v_xx = _grad(v_x, x)[:, 0:1]
    v_yy = _grad(v_y, x)[:, 1:2]

    f_cont = u_x + v_y
    f_u = u * u_x + v * u_y + p_x - (1.0 / re) * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - (1.0 / re) * (v_xx + v_yy)
    return f_cont, f_u, f_v


def compute_loss(
    model,
    X_colloc: torch.Tensor,
    X_bnd: torch.Tensor,
    u_bnd: torch.Tensor,
    v_bnd: torch.Tensor,
    p_bnd: torch.Tensor,
    Re: float = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute boundary and physics MSE loss terms."""
    preds_bnd = model(X_bnd)
    u_pred, v_pred, p_pred = preds_bnd[:, 0:1], preds_bnd[:, 1:2], preds_bnd[:, 2:3]

    mse_bnd = (
        torch.mean((u_pred - u_bnd) ** 2)
        + torch.mean((v_pred - v_bnd) ** 2)
        + torch.mean((p_pred - p_bnd) ** 2)
    )

    f_cont, f_u, f_v = _residual_terms(model, X_colloc, Re)
    mse_physics = torch.mean(f_cont**2) + torch.mean(f_u**2) + torch.mean(f_v**2)
    return mse_bnd, mse_physics


def compute_pointwise_physics_residual(model, X_candidate: torch.Tensor, Re: float = 20) -> torch.Tensor:
    """Compute scalar residual magnitude used for adaptive re-sampling."""
    f_cont, f_u, f_v = _residual_terms(model, X_candidate, Re)
    return (f_cont**2 + f_u**2 + f_v**2).squeeze()