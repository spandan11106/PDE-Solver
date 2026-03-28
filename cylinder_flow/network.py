import torch 
import torch.nn as nn 
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=50, scale=1.0):
        super().__init__()
        # Input dim is 3: (t, x, y)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        
    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = 2.0 * torch.pi * (x @ self.B)
        # Output shape: (N, 100)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 1. Initialize the embedding layer (input_dim=3 for t, x, y)
        self.embedding = FourierEmbedding(input_dim=3, mapping_size=50, scale=1.0)
        
        # 2. Network architecture
        self.hidden_layer = nn.ModuleList([nn.Linear(100, 50)]) 
        for _ in range(4):
            self.hidden_layer.append(nn.Linear(50, 50))

        # Output u, v, p
        self.output_layer = nn.Linear(50, 3)
        self.activation = nn.Tanh()
        
        # Adaptive Loss Weights
        self.w_bnd = torch.nn.Parameter(torch.zeros(1))
        self.w_phys = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.hidden_layer:
            x = self.activation(layer(x))
        return self.output_layer(x)