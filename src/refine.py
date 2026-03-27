import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from dataset import generate_points
from network import PINN
from loss import compute_loss, compute_pointwise_physics_residual

# --- Configuration ---
PRETRAINED_WEIGHTS = 'pinn_kovasznay.pth'
REYNOLDS_NUMBER = 20
RAR_ITERS = 5
EPOCHS_PER_RAR = 1000       # Adam fine-tuning epochs per new batch of points
LBFGS_EPOCHS = 5000         # Final polish
CANDIDATE_POOL_SIZE = 50000 
POINTS_PER_RAR = 50 

X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 0.5

# Initialize Weights & Biases
wandb.init(
    project="pinn-kovasznay",
    name="Re20-RAR-FineTuning",
    config={
        "learning_rate_adam": 1e-4,
        "rar_iterations": RAR_ITERS,
        "epochs_per_rar": EPOCHS_PER_RAR,
        "epochs_lbfgs": LBFGS_EPOCHS,
        "reynolds_number": REYNOLDS_NUMBER,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "points_per_rar": POINTS_PER_RAR,
        "pretrained_weights": PRETRAINED_WEIGHTS
    }
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Refining on device: {device}")

# 1. Load Pre-trained Model
model = PINN().to(device)
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
print(f"Loaded base weights from {PRETRAINED_WEIGHTS}")

# 2. Setup Base Data
X_colloc, X_bnd, u_bnd, v_bnd, p_bnd = generate_points(n_colloc=5000, n_bnd=200)

X_colloc = X_colloc.to(device)
X_bnd = X_bnd.to(device)
u_bnd = u_bnd.to(device)
v_bnd = v_bnd.to(device)
p_bnd = p_bnd.to(device)

optimizer_adam = optim.Adam(model.parameters(), lr=wandb.config.learning_rate_adam)

# --- STAGE 1: RAR Loop with Adam ---
print("\n--- Starting RAR Fine-Tuning ---")

global_step = 0 # Track total epochs across all RAR iterations

for rar_idx in range(RAR_ITERS):
    # Step A: Find the highest residual points
    model.eval()
    
    # Generate dense candidate grid
    x_c = X_MIN + (X_MAX - X_MIN) * torch.rand(CANDIDATE_POOL_SIZE, 1)
    y_c = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(CANDIDATE_POOL_SIZE, 1)
    X_candidate = torch.cat([x_c, y_c], dim=1).to(device)
    
    # Evaluate pointwise residuals
    pointwise_res = compute_pointwise_physics_residual(model, X_candidate, Re=REYNOLDS_NUMBER)
    
    # Select worst points
    max_res_value, top_indices = torch.topk(pointwise_res, k=POINTS_PER_RAR)
    worst_points = X_candidate[top_indices].detach()
    
    # Append to collocation set and reset graph
    X_colloc = torch.cat([X_colloc, worst_points], dim=0)
    X_colloc = X_colloc.detach().requires_grad_(True)
    
    tqdm.write(f"\n[RAR Iteration {rar_idx+1}/{RAR_ITERS}] Added {POINTS_PER_RAR} points.")
    tqdm.write(f"Max residual found: {max_res_value[0].item():.4e} | Total Colloc Points: {X_colloc.shape[0]}")
    
    # Log the RAR event to wandb
    wandb.log({
        "RAR Iteration": rar_idx + 1,
        "Max Candidate Residual": max_res_value[0].item(),
        "Total Colloc Points": X_colloc.shape[0]
    }, step=global_step)
    
    # Step B: Train network to adapt to new points
    model.train()
    pbar_adam = tqdm(range(EPOCHS_PER_RAR), desc=f"RAR {rar_idx+1} Adam", dynamic_ncols=True, leave=False)
    
    for epoch in pbar_adam:
        optimizer_adam.zero_grad()
        
        mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=REYNOLDS_NUMBER)
        loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
               0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
               
        loss.backward()
        optimizer_adam.step()
        
        # Log metrics to Weights & Biases
        wandb.log({
            "Stage": "1_RAR_Adam",
            "Total Loss": loss.item(),
            "Boundary MSE": mse_bnd.item(),
            "Physics MSE": mse_phys.item(),
            "w_bnd": model.w_bnd.item(),
            "w_phys": model.w_phys.item(),
        }, step=global_step)
        
        if epoch % 10 == 0:
            pbar_adam.set_postfix({
                'Loss': f"{loss.item():.2e}", 
                'Phys': f"{mse_phys.item():.2e}"
            })
            
        global_step += 1
        
    pbar_adam.close()

# --- STAGE 2: Final L-BFGS Polish ---
print("\n--- Starting Final L-BFGS Polish ---")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    max_iter=LBFGS_EPOCHS, 
    history_size=50, 
    line_search_fn="strong_wolfe"
)

lbfgs_iter = 0
pbar_lbfgs = tqdm(total=LBFGS_EPOCHS, desc="L-BFGS Fine-Tuning", dynamic_ncols=True)

def closure():
    global lbfgs_iter, global_step
    optimizer_lbfgs.zero_grad()
    
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=REYNOLDS_NUMBER)
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    
    if lbfgs_iter % 10 == 0:
        wandb.log({
            "Stage": "2_RAR_LBFGS",
            "Total Loss": loss.item(),
            "Boundary MSE": mse_bnd.item(),
            "Physics MSE": mse_phys.item(),
            "w_bnd": model.w_bnd.item(),
            "w_phys": model.w_phys.item(),
        }, step=global_step)
        
        pbar_lbfgs.set_postfix({
            'Loss': f"{loss.item():.2e}", 
            'Phys': f"{mse_phys.item():.2e}"
        })
        
    lbfgs_iter += 1
    global_step += 1
    pbar_lbfgs.update(1)
        
    return loss

optimizer_lbfgs.step(closure)
pbar_lbfgs.close()

# --- Final Evaluation & Save ---
mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=REYNOLDS_NUMBER)
print(f"\nFinal Refined Errors -> Boundary: {mse_bnd.item():.6e}, Physics: {mse_phys.item():.6e}")

save_path = 'pinn_kovasznay.pth'
torch.save(model.state_dict(), save_path)
print(f"Refinement complete. Saved improved weights to '{save_path}'!")

wandb.finish()