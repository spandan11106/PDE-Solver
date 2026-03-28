import torch 
import torch.nn as nn 
import numpy as np

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim=3, mapping_size=50, scale=1.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        
    def forward(self, x):
        x = x.to(self.B.dtype)
        x_proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.embedding = FourierEmbedding(input_dim=3, mapping_size=50, scale=1.0)
        
        hidden_dim = 128
        
        # Encoders for Input Injection
        self.encoder_U = nn.Linear(100, hidden_dim)
        self.encoder_V = nn.Linear(100, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(100, hidden_dim))
        for _ in range(4):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 3)
        self.activation = nn.Tanh()
        
        self.w_bnd = torch.nn.Parameter(torch.zeros(1))
        self.w_phys = torch.nn.Parameter(torch.zeros(1))
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        X_embed = self.embedding(x)
        
        U = self.activation(self.encoder_U(X_embed))
        V = self.activation(self.encoder_V(X_embed))

        H = self.activation(self.hidden_layers[0](X_embed))
        
        for layer in self.hidden_layers[1:]:
            Z = self.activation(layer(H))
            H = (1.0 - Z) * U + Z * V
            
        return self.output_layer(H)