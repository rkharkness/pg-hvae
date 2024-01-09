import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class MLP(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        
        self.linear1 = nn.Linear(z_dim, z_dim//8)
        self.linear2 = nn.Linear(z_dim//8, 1)
        
    def forward(self, z):
        z = torch.flatten(z, start_dim=1)
        z = F.relu(self.linear1(z))
        z = self.linear2(z)
        return torch.sigmoid(z)
        
class LogisticRegression(nn.Module):
    def __init__(self, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        
        self.linear1 = nn.Linear(z_dim, 1)
        
    def forward(self, z):
        z = torch.flatten(z, start_dim=1)
        z = torch.log(z)
        z = self.linear1(z)
        return torch.sigmoid(z)
        