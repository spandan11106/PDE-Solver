import torch 
import torch.nn as nn 

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.hidden_layer = nn.ModuleList([nn.Linear(2, 50)])  # Input Layer

        for _ in range(4):
            self.hidden_layer.append(nn.Linear(50, 50))

        self.output_layer = nn.Linear(50, 3)

        self.activation = nn.Tanh()
        self.w_bnd = torch.nn.Parameter(torch.zeros(1))
        self.w_phys = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # x is input tensor of shape (N, 2)

        for layer in self.hidden_layer:
            x = self.activation(layer(x))

        return self.output_layer(x)