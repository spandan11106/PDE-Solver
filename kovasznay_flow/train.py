import torch 
import torch.optim as optim 
import wandb
from tqdm import tqdm

from dataset import generate_points
from network import PINN
from loss import compute_loss

# Initialize Weights & Biases
wandb.init(
    project="pinn-kovasznay",
    name="Re20-LHS-Fourier",
    config={
        "learning_rate": 1e-3,
        "epochs_adam": 10000,
        "epochs_lbfgs": 50000,
        "reynolds_number": 20,
        "n_colloc": 5000,
        "n_bnd": 200
    }
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Initialize Model
model = PINN().to(device)

# Generate Data 
X_colloc, X_bnd, u_bnd, v_bnd, p_bnd = generate_points(
    n_colloc=wandb.config.n_colloc, 
    n_bnd=wandb.config.n_bnd
)

X_colloc = X_colloc.to(device)
X_bnd = X_bnd.to(device)
u_bnd = u_bnd.to(device)
v_bnd = v_bnd.to(device)
p_bnd = p_bnd.to(device)

# Optimizer and Scheduler setup
optimizer_adam = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# If the loss doesn't improve for 500 epochs, cut the LR by half
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam, 
    mode='min', 
    factor=0.5, 
    patience=500
)

# --- STAGE 1: ADAM OPTIMIZATION ---
print("\n--- Starting Adam Optimization ---")
patience_plateau = 500
best_loss = float('inf')
counter = 0

# Set up the TQDM progress bar
pbar_adam = tqdm(range(wandb.config.epochs_adam), desc="Adam Optimization", dynamic_ncols=True)

for epoch in pbar_adam:
    optimizer_adam.zero_grad()
    
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=wandb.config.reynolds_number)
    
    # Adaptive weights formulation
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss.item())
    
    # Log metrics to Weights & Biases
    wandb.log({
        "Stage": 1,
        "Total Loss": loss.item(),
        "Boundary MSE": mse_bnd.item(),
        "Physics MSE": mse_phys.item(),
        "w_bnd": model.w_bnd.item(),
        "w_phys": model.w_phys.item(),
        "Learning Rate": optimizer_adam.param_groups[0]['lr'],
        "Epoch": epoch
    })
    
    # Update terminal progress bar UI
    if epoch % 10 == 0:
        pbar_adam.set_postfix({
            'Loss': f"{loss.item():.2e}", 
            'Bnd': f"{mse_bnd.item():.2e}",
            'Phys': f"{mse_phys.item():.2e}",
            'w_bnd': f"{model.w_bnd.item():.2f}"
        })
    
    # Check for early stopping / plateau to switch to L-BFGS
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1

    if counter >= patience_plateau:
        tqdm.write(f"\nPlateau detected at epoch {epoch}. Switching to L-BFGS...")
        break

pbar_adam.close()

# --- STAGE 2: L-BFGS FINE-TUNING ---
print("\n--- Starting L-BFGS Fine-tuning ---")

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    max_iter=wandb.config.epochs_lbfgs, 
    history_size=50, 
    line_search_fn="strong_wolfe"
)

lbfgs_iter = 0

pbar_lbfgs = tqdm(total=wandb.config.epochs_lbfgs, desc="L-BFGS", dynamic_ncols=True)

def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=wandb.config.reynolds_number)
    
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    
    # Log to wandb
    if lbfgs_iter % 10 == 0:
        wandb.log({
            "Stage": 2,
            "Total Loss": loss.item(),
            "Boundary MSE": mse_bnd.item(),
            "Physics MSE": mse_phys.item(),
            "w_bnd": model.w_bnd.item(),
            "w_phys": model.w_phys.item(),
            "LBFGS_Iter": lbfgs_iter
        })
        
        pbar_lbfgs.set_postfix({
            'Loss': f"{loss.item():.2e}", 
            'Phys': f"{mse_phys.item():.2e}"
        })
    
    lbfgs_iter += 1
    pbar_lbfgs.update(1)
    
    return loss

# Run L-BFGS
optimizer_lbfgs.step(closure)
pbar_lbfgs.close()

# --- FINAL EVALUATION & SAVE ---
mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=wandb.config.reynolds_number)
print(f"\nFinal Errors -> Boundary: {mse_bnd.item():.6e}, Physics: {mse_phys.item():.6e}")

# Save the model
torch.save(model.state_dict(), 'pinn_kovasznay.pth')
print("Training finished and model saved to 'pinn_kovasznay.pth'!")

# Finish the wandb run
wandb.finish()