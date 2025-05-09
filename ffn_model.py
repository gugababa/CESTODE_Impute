
import torch
import torch.nn as nn


class FFN(nn.Module):
    
    def __init__(self, param_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        
        return self.net(x)

