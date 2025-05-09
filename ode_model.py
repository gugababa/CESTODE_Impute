# create the latent ODE model

import torch
import torch.nn as nn
from torchdiffeq import odeint
import math
from utils import *

# create model to simulate ordinary differental equation

class BlochODEFunc(nn.Module):
    
    def __init__(self, state_dim, param_dim, hidden_dim, latent_dim): 
        
        super(BlochODEFunc, self).__init__()
            
        self.net = nn.Sequential(
            
            nn.utils.spectral_norm(nn.Linear(state_dim + param_dim + latent_dim, hidden_dim)), # param_dim = 2 (B1, tsat), latent_dim for encoded representation
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, state_dim)))
    
        self.register_buffer('params', torch.zeros(1,param_dim), persistent = False)
        self.register_buffer('zenc', torch.zeros(1, latent_dim), persistent = False)
        self.nfe = 0
    
    def set_params(self, params):
        
        self.params = params
    
    def set_zenc(self, z_encoded):
         
         self.z_encoded = z_encoded
    
    def get_nfe(self):
        
        return self.nfe
    
    def forward(self, t, x):
        
        self.nfe += 1
        x = torch.cat([x, self.params, self.z_encoded], dim = -1)
        return self.net(x)

    
class ODEAttnEncoder(nn.Module):
    
    def __init__(self, input_dim, seq_len, feature_dim,
                 model_dim, inner_dim, ode_hidden_dim, 
                 state_dim, param_dim, latent_dim, num_heads = 2,
                 diagonal_attn_mask = True, dropout = 0.1):
        
        super(ODEAttnEncoder, self).__init__()

        # instantiate layers for self-attention based encoding
        self.diag_attn_mask = diagonal_attn_mask
        self.pos_embed = PositionalEncoding(model_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(model_dim, inner_dim, dropout)
        self.enc = nn.Linear(feature_dim * 2,model_dim)
        self.nan_embed = NanEmbed(input_dim, feature_dim, seq_len)
        
        self.multi_head_attn = nn.MultiheadAttention(model_dim, num_heads = num_heads, dropout=dropout, batch_first=True)

        # instantiate layer for neural ODE updates
        self.ode_func = BlochODEFunc(state_dim, param_dim, ode_hidden_dim, model_dim)
        self.state_dim = state_dim
        
        # instantiate layers for variational encoding 
        # self.mu = nn.Linear(state_dim, latent_dim)
        # self.log_var = nn.Linear(state_dim, latent_dim)

            
    def forward(self, data, B1, tsat, mask, offsets):
        
        batch_size, seq_len = data.shape
        data = data.unsqueeze(-1)
        mask = mask.unsqueeze(-1)

        # embed the nan values
        x = self.nan_embed(data, mask)
        
        # perform computations using attention mechanism
        x = self.enc(x)

        x = self.dropout(self.pos_embed(x))

        if self.diag_attn_mask:
            mask_time = torch.eye(seq_len).bool().to(data.device)
        else:
            mask_time = None
        
        res_x = x
        x = self.layer_norm(x)
        x, _ = self.multi_head_attn(x, x, x, attn_mask = mask_time)
        x = self.dropout(x)
        x += res_x

        enc_x = self.pos_ffn(x)

        # initialize the hidden state
        h0 = torch.zeros(batch_size, seq_len, self.state_dim).to(data.device)
        h0[:,:,2::3] = 1.0
        # set parameters for ODE function
        
        B1 = B1.unsqueeze(1).expand(-1,seq_len,-1)
        tsat = tsat.unsqueeze(1).expand(-1,seq_len,-1)
        offsets = offsets.unsqueeze(-1)

        self.ode_func.set_params(torch.cat([B1,tsat,offsets], dim = -1))
        self.ode_func.set_zenc(enc_x)
        
        t_start = 0
        t_end = torch.max(tsat)
        
        t_span = torch.tensor([t_start,t_end], dtype = torch.float32).to(data.device)
        h_final = odeint(self.ode_func, h0, t_span, method = 'dopri5')
        
        h_final = h_final[-1]
        
        return h_final
        
        # mu = self.mu(h_final)
        # log_var = self.log_var(h_final)

        # return mu, log_var

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim):
        super(Decoder,self).__init__()
        
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim ,hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        
        output = self.net(x)
        output = output.squeeze()

        return output

class AttnODE(nn.Module):
    
    def __init__(self, input_dim, seq_len, feature_dim, model_dim, inner_dim, 
                 ode_hidden_dim, state_dim, param_dim, 
                 latent_dim, n_heads = 2, diag_attn_mask = True, dropout = 0.1):
        super().__init__()
        
        self.enc = ODEAttnEncoder(input_dim, seq_len, feature_dim, model_dim, inner_dim, 
                                  ode_hidden_dim, state_dim, param_dim, 
                                  latent_dim, n_heads, diag_attn_mask, dropout)
        # self.dec = Decoder(state_dim, inner_dim)
    
    def reparametrize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return mu + eps*std
        
    def forward(self, data, B1, tsat, masks, offsets):
        
        # mu_z0, logvar_z0 = self.enc(data, B1, tsat, masks)
        z0 = self.enc(data, B1, tsat, masks, offsets)
        pred_z = z0[:,:,-1].squeeze()
        # z0 = self.reparametrize(mu_z0, logvar_z0)
        # pred_z = self.dec(z0)

        return pred_z #, mu_z0, logvar_z0


class NanEmbed(torch.nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        # create embedding weights
        self.emb_layer = nn.Linear(input_dim, output_dim)
        self.freq_embedding = nn.Embedding(seq_len, output_dim)
        self.nanVal = nn.Parameter(torch.zeros(1,output_dim))
        self.output_dim = output_dim
        self.seq_len = seq_len
        
    def forward(self, x, mask):
        # embed each feature into a larger embedding vector of size output_dim
        x = torch.nan_to_num(x, nan = -1.0) 
        emb = self.emb_layer(x)

        mask = mask.expand(-1, -1, self.output_dim)
        mask = mask.reshape(-1, self.output_dim)
        nanInds = ~mask.sum(dim = 1).bool()

        emb = emb.reshape(-1, self.output_dim)

        emb[nanInds,:] = self.nanVal

        mask = mask.reshape(-1,self.seq_len, self.output_dim)
        emb = emb.reshape(-1, self.seq_len, self.output_dim)
        
        freqEmbed = self.freq_embedding(torch.arange(self.seq_len).to(x.device))
        freqEmbed = freqEmbed.unsqueeze(0).expand(x.shape[0],-1,-1)

        emb = emb + freqEmbed

        # mask = mask.expand(-1, -1, self.output_dim)
        emb = torch.cat((emb, mask), dim = -1)
        return emb