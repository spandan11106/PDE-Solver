import torch 
from kovasznay import kovasznay_solution

def generate_points():
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 0.5  

    x_c = x_min + (x_max - x_min)*torch.rand(5000, 1)
    y_c = y_min + (y_max - y_min)*torch.rand(5000, 1)
    X_colloc = torch.cat([x_c, y_c], dim=1)

    x_left = torch.full((200, 1), x_min)
    y_left = y_min + (y_max - y_min)*torch.rand(200 , 1)
    X_left = torch.cat([x_left, y_left], dim=1)

    x_right = torch.full((200, 1), x_max)
    y_right = y_min + (y_max - y_min)*torch.rand(200, 1)
    X_right = torch.cat([x_right, y_right], dim=1)

    x_top = x_min + (x_max - x_min)*torch.rand(200, 1)
    y_top = torch.full((200, 1), y_max)
    X_top = torch.cat([x_top, y_top], dim=1)

    x_bottom = x_min + (x_max - x_min)*torch.rand(200, 1)
    y_bottom = torch.full((200, 1), y_min)
    X_bottom = torch.cat([x_bottom, y_bottom], dim=1)

    X_bnd = torch.cat([X_left, X_right, X_top, X_bottom], dim=0)

    u_bnd, v_bnd, p_bnd = kovasznay_solution(X_bnd[:, 0:1], X_bnd[:, 1:2]) 

    return X_colloc, X_bnd, u_bnd, v_bnd, p_bnd