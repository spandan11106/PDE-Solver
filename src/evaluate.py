import torch
import numpy as np
import matplotlib.pyplot as plt
from network import PINN
from kovasznay import kovasznay_solution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the model
model = PINN().to(device)
model.load_state_dict(torch.load('pinn_kovasznay.pth', map_location=device))
model.eval()

# 2. Create a dense grid for plotting
x = np.linspace(-0.5, 1.0, 100)
y = np.linspace(-0.5, 0.5, 100)
X, Y = np.meshgrid(x, y)

# Flatten and convert to tensor for the network
grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

# 3. Get Network Predictions
with torch.no_grad():
    preds = model(grid_tensor)
    # Reshape back to grid format
    u_pred = preds[:, 0].cpu().numpy().reshape(100, 100)
    v_pred = preds[:, 1].cpu().numpy().reshape(100, 100)
    p_pred = preds[:, 2].cpu().numpy().reshape(100, 100)

# 4. Get Exact "Ground Truth"
# We pass the flattened x and y tensors to your kovasznay_solution
x_ex = torch.tensor(grid_points[:, 0:1], dtype=torch.float32)
y_ex = torch.tensor(grid_points[:, 1:2], dtype=torch.float32)
u_ex_t, v_ex_t, p_ex_t = kovasznay_solution(x_ex, y_ex)

u_exact = u_ex_t.numpy().reshape(100, 100)
v_exact = v_ex_t.numpy().reshape(100, 100)
p_exact = p_ex_t.numpy().reshape(100, 100)

# 5. Plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fields = [(u_exact, u_pred, 'u'), (v_exact, v_pred, 'v'), (p_exact, p_pred, 'p')]

for i, (exact, pred, name) in enumerate(fields):
    # Exact
    im0 = axes[i, 0].contourf(X, Y, exact, 50, cmap='jet')
    axes[i, 0].set_title(f'Exact {name}')
    fig.colorbar(im0, ax=axes[i, 0])
    
    # Predicted
    im1 = axes[i, 1].contourf(X, Y, pred, 50, cmap='jet')
    axes[i, 1].set_title(f'Predicted {name}')
    fig.colorbar(im1, ax=axes[i, 1])
    
    # Error
    im2 = axes[i, 2].contourf(X, Y, np.abs(exact - pred), 50, cmap='jet')
    axes[i, 2].set_title(f'Abs Error {name}')
    fig.colorbar(im2, ax=axes[i, 2])

plt.tight_layout()
plt.savefig('kovasznay_results.png')
print("Plot saved to kovasznay_results.png")