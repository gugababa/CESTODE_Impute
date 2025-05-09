# File containing utility functions

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class MyCustomDataset(Dataset):
    
    def __init__(self, spec, mask, B1, tsat, spec_ori, offsets):
        
        self.spec = spec
        self.mask = mask
        self.B1 = B1
        self.tsat = tsat
        self.spec_ori = spec_ori
        self.offsets = offsets
        
    def __len__(self):
        
        return len(self.spec)
    
    def __getitem__(self, idx):
        
        spec = torch.tensor(self.spec[idx,:], dtype = torch.float32)
        mask = torch.tensor(self.mask[idx,:], dtype = torch.float32)
        B1 = torch.tensor(self.B1[idx,:], dtype = torch.float32)
        tsat = torch.tensor(self.tsat[idx,:], dtype = torch.float32)
        spec_ori = torch.tensor(self.spec_ori[idx,:], dtype = torch.float32)
        offsets = torch.tensor(self.offsets[idx,:], dtype = torch.float32)
        
        return spec, B1, tsat, mask, spec_ori, offsets
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # permute self.pe to match [batch_size, seq_len, embedding_dim]
        pe = torch.permute(self.pe, (1,0,2))
        x = x + pe[:,:x.size(1),:]
        return self.dropout(x)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.elu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x

def custom_criterion_vae(X_pred, X_true, mask, mu, log_var, epoch_num, total_epochs, r = 0.5, m = 5, kl_weight = 1.0):

    kl_weight = (epoch_num % (total_epochs/m)) / (total_epochs/m)
    if kl_weight > r:
        kl_weight = 1
    
    mask = mask + 0
    # calculate reconstruction loss on observed points
    recon_loss_observed = torch.sum(torch.square(X_pred - X_true)*mask, dim = -1) / (torch.sum(mask, dim = -1) + 1e-9)
    recon_loss_observed = recon_loss_observed.mean()
    
    # calculate reconstruction loss on imputed points
    imputed_mask = 1 - mask
    recon_loss_imputed = torch.sum(torch.square(X_pred - X_true)*imputed_mask, dim = -1) / (torch.sum(imputed_mask, dim = -1) + 1e-9)
    recon_loss_imputed = recon_loss_imputed.mean()
    
    # calculate KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
    kl_div = kl_div.mean()

    # recon_loss = recon_loss_observed + recon_loss_imputed
    total_loss = recon_loss_observed + recon_loss_imputed + kl_weight * kl_div

    return total_loss, recon_loss_observed, recon_loss_imputed, kl_div, kl_weight

def custom_criterion(X_pred, X_true, mask):
    
    mask = mask + 0
    # calculate reconstruction loss on observed points
    recon_loss_observed = torch.sum(torch.abs(X_pred - X_true)*mask, dim = -1) / (torch.sum(mask, dim = -1) + 1e-9)
    recon_loss_observed = recon_loss_observed.mean()
    
    # calculate reconstruction loss on imputed points
    imputed_mask = 1 - mask
    recon_loss_imputed = torch.sum(torch.abs(X_pred - X_true)*imputed_mask, dim = -1) / (torch.sum(imputed_mask, dim = -1) + 1e-9)
    recon_loss_imputed = recon_loss_imputed.mean()
    
    total_loss = recon_loss_observed + recon_loss_imputed
    
    return total_loss, recon_loss_observed, recon_loss_imputed
    
class UncertaintyWeightedLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        # log(sigma^2) is parameterized for numerical stability
        self.log_sigma_recon = nn.Parameter(torch.tensor(0.0))  # learnable
        self.log_sigma_impute = nn.Parameter(torch.tensor(0.0))  # learnable

    def forward(self, loss_recon, loss_impute):
        precision_recon = torch.exp(-self.log_sigma_recon)
        precision_impute = torch.exp(-self.log_sigma_impute)

        precision_recon_loss = precision_recon * loss_recon + self.log_sigma_recon
        precision_impute_loss = precision_impute * loss_impute + self.log_sigma_impute
        
        weighted_loss = precision_recon_loss + precision_impute_loss
        
        return weighted_loss, precision_recon_loss, precision_impute_loss