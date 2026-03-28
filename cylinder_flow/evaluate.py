"""Evaluation and visualization for the cylinder flow PINN."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from network import PINN


RESULTS_DIR = Path("results")
REFINED_WEIGHTS = Path("pinn_cylinder_refined.pth")
BASE_WEIGHTS = Path("pinn_cylinder.pth")


def _plot_field(fig, ax, x, y, field, title: str, vmin: float, vmax: float, cyl_r: float) -> None:
    im = ax.contourf(x, y, field, levels=np.linspace(vmin, vmax, 50), cmap="jet", extend="both")
    ax.set_title(title, fontsize=13)
    ax.add_patch(plt.Circle((0, 0), cyl_r, color="dimgray", zorder=10))
    fig.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, 7))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN().to(device)
    if REFINED_WEIGHTS.exists():
        model_path = REFINED_WEIGHTS
    elif BASE_WEIGHTS.exists():
        model_path = BASE_WEIGHTS
    else:
        raise FileNotFoundError("No checkpoint found. Expected 'pinn_cylinder_refined.pth' or 'pinn_cylinder.pth'.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    t_val = 0.5
    x = np.linspace(-1.0, 5.0, 200)
    y = np.linspace(-2.0, 2.0, 150)
    x_grid, y_grid = np.meshgrid(x, y)

    t_grid = np.full((x_grid.size, 1), t_val)
    grid_points = np.hstack([t_grid, x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(grid_tensor)
        u_pred = preds[:, 0].cpu().numpy().reshape(x_grid.shape)
        v_pred = preds[:, 1].cpu().numpy().reshape(x_grid.shape)
        p_pred = preds[:, 2].cpu().numpy().reshape(x_grid.shape)

    vel_mag = np.sqrt(u_pred**2 + v_pred**2)
    cyl_r = 0.5
    mask = (x_grid**2 + y_grid**2) <= cyl_r**2
    u_pred[mask] = np.nan
    v_pred[mask] = np.nan
    p_pred[mask] = np.nan
    vel_mag[mask] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cylinder Wake Predictions at t = {t_val}s", fontsize=16, fontweight="bold")

    _plot_field(fig, axes[0, 0], x_grid, y_grid, u_pred, "X-Velocity ($u$)", vmin=-0.5, vmax=1.5, cyl_r=cyl_r)
    _plot_field(fig, axes[0, 1], x_grid, y_grid, v_pred, "Y-Velocity ($v$)", vmin=-0.8, vmax=0.8, cyl_r=cyl_r)
    _plot_field(fig, axes[1, 0], x_grid, y_grid, p_pred, "Pressure ($p$)", vmin=-0.5, vmax=0.5, cyl_r=cyl_r)
    _plot_field(
        fig,
        axes[1, 1],
        x_grid,
        y_grid,
        vel_mag,
        "Velocity Magnitude ($||V||$)",
        vmin=0.0,
        vmax=1.5,
        cyl_r=cyl_r,
    )

    plt.tight_layout()
    save_path = RESULTS_DIR / f"cylinder_eval_t{str(t_val).replace('.', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to '{save_path}'.")


if __name__ == "__main__":
    main()