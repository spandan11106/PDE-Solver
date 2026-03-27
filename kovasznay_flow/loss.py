import torch 

def compute_loss(model, X_colloc, X_bnd, u_bnd, v_bnd, p_bnd, Re=20):

    # --- 1. Boundary Data Loss ---
    preds_bnd = model(X_bnd)
    u_p, v_p, p_p  = preds_bnd[:, 0:1], preds_bnd[:, 1:2], preds_bnd[:, 2:3]

    mse_bnd = torch.mean((u_p - u_bnd)**2) + \
              torch.mean((v_p - v_bnd)**2) + \
              torch.mean((p_p - p_bnd)**2)
    
    # --- 2. Physics Loss ---
    X_colloc.requires_grad_(True)
    preds = model(X_colloc)
    u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    def grad(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

    # First derivatives
    grads_u = grad(u, X_colloc)
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]

    grads_v = grad(v, X_colloc)
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    
    grads_p = grad(p, X_colloc)
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

    # Second derivatives
    u_xx = grad(u_x, X_colloc)[:, 0:1]
    u_yy = grad(u_y, X_colloc)[:, 1:2]
    v_xx = grad(v_x, X_colloc)[:, 0:1]
    v_yy = grad(v_y, X_colloc)[:, 1:2]

    # Navier-Stokes Residuals
    f_cont = u_x + v_y
    f_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)

    mse_physics = torch.mean(f_cont**2) + torch.mean(f_u**2) + torch.mean(f_v**2)

    return mse_bnd, mse_physics

def compute_pointwise_physics_residual(model, X_candidate, Re=20):
    X_candidate.requires_grad_(True)
    preds = model(X_candidate)
    u, v, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    def grad(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

    grads_u = grad(u, X_candidate)
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]

    grads_v = grad(v, X_candidate)
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    
    grads_p = grad(p, X_candidate)
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

    u_xx = grad(u_x, X_candidate)[:, 0:1]
    u_yy = grad(u_y, X_candidate)[:, 1:2]
    v_xx = grad(v_x, X_candidate)[:, 0:1]
    v_yy = grad(v_y, X_candidate)[:, 1:2]

    f_cont = u_x + v_y
    f_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)

    pointwise_residual = f_cont**2 + f_u**2 + f_v**2
    return pointwise_residual.squeeze()