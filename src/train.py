import torch 
import torch.optim as optim 
from dataset import generate_points
from network import PINN
from loss import compute_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PINN().to(device)

X_colloc, X_bnd, u_bnd, v_bnd, p_bnd = generate_points()

X_colloc = X_colloc.to(device)
X_bnd = X_bnd.to(device)
u_bnd = u_bnd.to(device)
v_bnd = v_bnd.to(device)
p_bnd = p_bnd.to(device)

optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

# If the loss doesn't improve for 500 epochs, cut the LR by half
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam, 
    mode='min', 
    factor=0.5, 
    patience=500
)

# --- STAGE 1: ADAM ---
print("Starting Adam Optimization...")
patience = 500
best_loss = float('inf')
counter = 0

for epoch in range(10000): # High max, but we likely won't hit it
    optimizer_adam.zero_grad()
    
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd)
    
    # Adaptive weights formulation
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss.item())
    
    # Check for plateau
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        
    if epoch % 500 == 0:
        print(f"Adam Epoch {epoch}: Total: {loss.item():.6e} | MSE_Bnd: {mse_bnd.item():.6e}, MSE_Phys: {mse_phys.item():.6e} | w_bnd: {model.w_bnd.item():.3f}, w_phys: {model.w_phys.item():.3f}")

    if counter >= patience:
        print(f"Plateau detected at epoch {epoch}. Switching to L-BFGS...")
        break

# --- STAGE 2: L-BFGS ---
print("Starting L-BFGS Fine-tuning...")

# Re-initialize optimizer for Stage 2
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    max_iter=50000, 
    history_size=50, 
    line_search_fn="strong_wolfe"
)

lbfgs_iter = 0
def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    # We only care about the total loss for the optimizer
    mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd)
    
    loss = 0.5 * torch.exp(-model.w_bnd) * mse_bnd + 0.5 * model.w_bnd + \
           0.5 * torch.exp(-model.w_phys) * mse_phys + 0.5 * model.w_phys
           
    loss.backward()
    
    if lbfgs_iter % 100 == 0:
        print(f"L-BFGS Iter {lbfgs_iter}: Total: {loss.item():.6e} | MSE_Bnd: {mse_bnd.item():.6e}, MSE_Phys: {mse_phys.item():.6e} | w_bnd: {model.w_bnd.item():.3f}, w_phys: {model.w_phys.item():.3f}")
    lbfgs_iter += 1
    
    return loss

optimizer_lbfgs.step(closure)

mse_bnd, mse_phys = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd)
print(f"Final Errors -> Bnd: {mse_bnd.item():.6e}, Phys: {mse_phys.item():.6e}")

torch.save(model.state_dict(), 'pinn_kovasznay.pth')
print("Training finished and model saved!")