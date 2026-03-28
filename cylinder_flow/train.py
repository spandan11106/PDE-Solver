import torch 
import torch.optim as optim 
import wandb
from tqdm import tqdm

from dataset import generate_points
from network import PINN
from loss import compute_loss

wandb.init(
    project="pinn-cylinder",
    name="Unsteady-Cylinder-CPU",
    config={
        "learning_rate": 1e-3,
        "epochs_adam": 10000,   
        "epochs_lbfgs": 50000,  
        "reynolds_number": 100,
        "n_colloc": 5000,
        "n_bnd": 1500,
        "t_max": 0.5
    }
)

device = torch.device('cpu') # Forcing CPU per your instructions
print(f"Training on device: {device}")

model = PINN().to(device)

X_colloc, X_bnd, u_bnd, v_bnd = generate_points(
    n_colloc=wandb.config.n_colloc, 
    n_bnd=wandb.config.n_bnd,
    T_max=wandb.config.t_max  
)

X_colloc = X_colloc.to(device)
X_bnd = X_bnd.to(device)
u_bnd = u_bnd.to(device)
v_bnd = v_bnd.to(device)

optimizer_adam = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=300)

print("\n--- Starting Adam Optimization ---")
pbar_adam = tqdm(range(wandb.config.epochs_adam), desc="Adam", dynamic_ncols=True)

for epoch in pbar_adam:
    optimizer_adam.zero_grad()
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, Re=wandb.config.reynolds_number)
    
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss.item())
    
    if epoch % 10 == 0:
        wandb.log({"Stage": 1, "Loss": loss.item(), "Bnd MSE": mse_bnd.item(), "Phys MSE": mse_phys.item()})
        pbar_adam.set_postfix({'Loss': f"{loss.item():.2e}", 'Phys': f"{mse_phys.item():.2e}"})

pbar_adam.close()

print("\n--- Starting L-BFGS Fine-tuning ---")
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=wandb.config.epochs_lbfgs)
pbar_lbfgs = tqdm(total=wandb.config.epochs_lbfgs, desc="L-BFGS", dynamic_ncols=True)
lbfgs_iter = 0

def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, Re=wandb.config.reynolds_number)
    
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    
    if lbfgs_iter % 10 == 0:
        pbar_lbfgs.set_postfix({'Loss': f"{loss.item():.2e}", 'Phys': f"{mse_phys.item():.2e}"})
    
    lbfgs_iter += 1
    pbar_lbfgs.update(1)
    return loss

optimizer_lbfgs.step(closure)
pbar_lbfgs.close()

torch.save(model.state_dict(), 'pinn_cylinder.pth')
print("Model saved to 'pinn_cylinder.pth'!")
wandb.finish()