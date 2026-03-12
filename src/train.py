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

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()

    mse_bnd, mse_physics = compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd)
    loss = mse_bnd + mse_physics
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f} (Bound: {mse_bnd.item():.6f}, Phys: {mse_physics.item():.6f})")

torch.save(model.state_dict(), 'pinn_kovasznay.pth')
print("Training finished and model saved!")