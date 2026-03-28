"""Training entrypoint for the unsteady cylinder PINN."""

from __future__ import annotations

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from dataset import generate_points
from loss import compute_loss
from network import PINN


RUN_CONFIG = {
    "learning_rate": 1e-3,
    "epochs_adam": 2000,
    "epochs_lbfgs": 1000,
    "reynolds_number": 100,
    "n_colloc": 5000,
    "n_bnd": 1500,
    "t_max": 0.5,
}


def weighted_loss(model: PINN, mse_bnd: torch.Tensor, mse_phys: torch.Tensor) -> torch.Tensor:
    """Adaptive weighting between data and PDE losses."""
    return (
        0.5 * torch.exp(-model.w_bnd) * mse_bnd
        + 0.5 * model.w_bnd
        + 0.5 * torch.exp(-model.w_phys) * mse_phys
        + 0.5 * model.w_phys
    )


def main() -> None:
    wandb.init(project="pinn-cylinder", name="Unsteady-Cylinder-CPU-Window1", config=RUN_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = PINN().to(device)
    model.init_weights()

    x_colloc, x_bnd, u_bnd, v_bnd = generate_points(
        n_colloc=wandb.config.n_colloc,
        n_bnd=wandb.config.n_bnd,
        t_max=wandb.config.t_max,
    )
    x_colloc = x_colloc.to(device)
    x_bnd = x_bnd.to(device)
    u_bnd = u_bnd.to(device)
    v_bnd = v_bnd.to(device)

    optimizer_adam = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adam,
        mode="min",
        factor=0.5,
        patience=300,
    )

    print("\n--- Starting Adam Optimization ---")
    pbar_adam = tqdm(range(wandb.config.epochs_adam), desc="Adam", dynamic_ncols=True)

    for epoch in pbar_adam:
        optimizer_adam.zero_grad()
        mse_bnd, mse_phys = compute_loss(
            model,
            x_colloc,
            x_bnd,
            u_bnd,
            v_bnd,
            Re=wandb.config.reynolds_number,
        )
        loss = weighted_loss(model, mse_bnd, mse_phys)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step(loss.item())

        if epoch % 10 == 0:
            wandb.log(
                {
                    "Stage": 1,
                    "Loss": loss.item(),
                    "Bnd MSE": mse_bnd.item(),
                    "Phys MSE": mse_phys.item(),
                }
            )
            pbar_adam.set_postfix({"Loss": f"{loss.item():.2e}", "Phys": f"{mse_phys.item():.2e}"})

    pbar_adam.close()

    print("\n--- Starting L-BFGS Fine-tuning ---")
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=wandb.config.epochs_lbfgs)
    pbar_lbfgs = tqdm(total=wandb.config.epochs_lbfgs, desc="L-BFGS", dynamic_ncols=True)
    lbfgs_iter = 0

    def closure() -> torch.Tensor:
        nonlocal lbfgs_iter
        optimizer_lbfgs.zero_grad()
        mse_bnd, mse_phys = compute_loss(
            model,
            x_colloc,
            x_bnd,
            u_bnd,
            v_bnd,
            Re=wandb.config.reynolds_number,
        )
        loss = weighted_loss(model, mse_bnd, mse_phys)
        loss.backward()

        if lbfgs_iter % 10 == 0:
            pbar_lbfgs.set_postfix({"Loss": f"{loss.item():.2e}", "Phys": f"{mse_phys.item():.2e}"})

        lbfgs_iter += 1
        pbar_lbfgs.update(1)
        return loss

    optimizer_lbfgs.step(closure)
    pbar_lbfgs.close()

    torch.save(model.state_dict(), "pinn_cylinder.pth")
    print("Model saved to 'pinn_cylinder.pth'!")
    wandb.finish()


if __name__ == "__main__":
    main()