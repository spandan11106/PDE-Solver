"""Evaluation and visualization script for Kovasznay-flow PINN."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from kovasznay import kovasznay_solution
from network import PINN


def l2_relative_error(pred: np.ndarray, exact: np.ndarray) -> float:
    """Compute L2-relative error between prediction and exact field."""
    return np.linalg.norm(exact.flatten() - pred.flatten()) / np.linalg.norm(exact.flatten())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN().to(device)
    model.load_state_dict(torch.load("pinn_kovasznay.pth", map_location=device))
    model.eval()

    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 0.5, 101)
    x_grid, y_grid = np.meshgrid(x, y)

    grid_points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(grid_tensor)
        u_pred = preds[:, 0].cpu().numpy().reshape(101, 101)
        v_pred = preds[:, 1].cpu().numpy().reshape(101, 101)
        p_pred = preds[:, 2].cpu().numpy().reshape(101, 101)

    x_ex = torch.tensor(grid_points[:, 0:1], dtype=torch.float32)
    y_ex = torch.tensor(grid_points[:, 1:2], dtype=torch.float32)
    u_ex_t, v_ex_t, p_ex_t = kovasznay_solution(x_ex, y_ex)

    u_exact = u_ex_t.numpy().reshape(101, 101)
    v_exact = v_ex_t.numpy().reshape(101, 101)
    p_exact = p_ex_t.numpy().reshape(101, 101)

    error_u = l2_relative_error(u_pred, u_exact)
    error_v = l2_relative_error(v_pred, v_exact)
    error_p = l2_relative_error(p_pred, p_exact)

    print("-" * 30)
    print("EVALUATION RESULTS (Re=20)")
    print("-" * 30)
    print(f"L2 Relative Error (u): {error_u:.2e} ({error_u * 100:.4f}%)")
    print(f"L2 Relative Error (v): {error_v:.2e} ({error_v * 100:.4f}%)")
    print(f"L2 Relative Error (p): {error_p:.2e} ({error_p * 100:.4f}%)")
    print("-" * 30)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fields = [(u_exact, u_pred, "u"), (v_exact, v_pred, "v"), (p_exact, p_pred, "p")]

    for i, (exact, pred, name) in enumerate(fields):
        im0 = axes[i, 0].contourf(x_grid, y_grid, exact, 50, cmap="jet")
        axes[i, 0].set_title(f"Exact {name}")
        fig.colorbar(im0, ax=axes[i, 0])

        im1 = axes[i, 1].contourf(x_grid, y_grid, pred, 50, cmap="jet")
        axes[i, 1].set_title(f"Predicted {name}")
        fig.colorbar(im1, ax=axes[i, 1])

        im2 = axes[i, 2].contourf(x_grid, y_grid, np.abs(exact - pred), 50, cmap="jet")
        axes[i, 2].set_title(f"Abs Error {name}")
        fig.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig("kovasznay_results.png")
    print("Plot saved to kovasznay_results.png")


if __name__ == "__main__":
    main()