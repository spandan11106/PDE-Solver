"""Residual-based adaptive refinement (RAR) for cylinder flow PINN."""

from __future__ import annotations

import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import generate_points
from loss import compute_loss, compute_pointwise_physics_residual
from network import PINN


PRETRAINED_WEIGHTS = "pinn_cylinder.pth"
REFINED_WEIGHTS = "pinn_cylinder_refined.pth"
REYNOLDS_NUMBER = 100
RAR_ITERS = 5
EPOCHS_PER_RAR = 1000
CANDIDATE_POOL_SIZE = 10000
POINTS_PER_RAR = 50
T_MAX = 0.5


def weighted_loss(model: PINN, mse_bnd: torch.Tensor, mse_phys: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * torch.exp(-model.w_bnd) * mse_bnd
        + 0.5 * model.w_bnd
        + 0.5 * torch.exp(-model.w_phys) * mse_phys
        + 0.5 * model.w_phys
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN().to(device)
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))

    x_colloc, x_bnd, u_bnd, v_bnd = generate_points(n_colloc=5000, n_bnd=500, t_max=T_MAX)
    x_colloc, x_bnd = x_colloc.to(device), x_bnd.to(device)
    u_bnd, v_bnd = u_bnd.to(device), v_bnd.to(device)

    optimizer_adam = optim.Adam(model.parameters(), lr=1e-4)

    print("\n--- Starting RAR Fine-Tuning ---")
    for rar_idx in range(RAR_ITERS):
        model.eval()

        t_c = T_MAX * torch.rand(CANDIDATE_POOL_SIZE, 1)
        x_c = -1.0 + 6.0 * torch.rand(CANDIDATE_POOL_SIZE, 1)
        y_c = -2.0 + 4.0 * torch.rand(CANDIDATE_POOL_SIZE, 1)

        mask = ((x_c - 0.0) ** 2 + (y_c - 0.0) ** 2) >= 0.5**2
        x_candidate = torch.cat(
            [t_c[mask].view(-1, 1), x_c[mask].view(-1, 1), y_c[mask].view(-1, 1)],
            dim=1,
        ).to(device)

        pointwise_res = compute_pointwise_physics_residual(model, x_candidate, Re=REYNOLDS_NUMBER)
        max_res_value, top_indices = torch.topk(pointwise_res, k=POINTS_PER_RAR)
        worst_points = x_candidate[top_indices].detach()

        x_colloc = torch.cat([x_colloc, worst_points], dim=0).detach().requires_grad_(True)

        print(
            f"\n[RAR Iteration {rar_idx + 1}/{RAR_ITERS}] Added {POINTS_PER_RAR} points. "
            f"Max Res: {max_res_value[0].item():.4e}"
        )

        model.train()
        pbar = tqdm(range(EPOCHS_PER_RAR), desc=f"RAR {rar_idx + 1} Adam", leave=False)
        for epoch in pbar:
            optimizer_adam.zero_grad()
            mse_bnd, mse_phys = compute_loss(model, x_colloc, x_bnd, u_bnd, v_bnd, Re=REYNOLDS_NUMBER)
            loss = weighted_loss(model, mse_bnd, mse_phys)
            loss.backward()
            optimizer_adam.step()

            if epoch % 10 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.2e}"})

    torch.save(model.state_dict(), REFINED_WEIGHTS)
    print(f"\nSaved refined weights to '{REFINED_WEIGHTS}'!")


if __name__ == "__main__":
    main()