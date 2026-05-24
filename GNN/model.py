import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from sklearn.neighbors import NearestNeighbors
import numpy as np


def mask_features(data, mask_ratio=0.3):
    x = data.x.clone()
    gene_dim = data.gene_dim

    #mask = torch.rand_like(x[:, :gene_dim]) < mask_ratio
    #x[:, :gene_dim][mask] = 0.0

    if mask_ratio > 0.:
        random_mask = torch.rand_like(x[:, :gene_dim], device=x.device) < mask_ratio 
        x[:, :gene_dim].masked_fill_(random_mask, 0.)

    return x

def loss_fn(pred, target, gene_dim):
    #return F.mse_loss(pred[:, :gene_dim], target[:, :gene_dim])
    return F.mse_loss(pred, target[:, :gene_dim])


### V1,2: sparse message passing 
class SpatialTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)

        self.out_lin = nn.Linear(dim, dim)

        # edge bias (distance encoding)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, heads),
            nn.ReLU(),
            nn.Linear(heads, heads)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        Q = self.q_lin(x).view(-1, self.heads, self.head_dim)
        K = self.k_lin(x).view(-1, self.heads, self.head_dim)
        V = self.v_lin(x).view(-1, self.heads, self.head_dim)

        # Attention scores
        attn = (Q[row] * K[col]).sum(dim=-1) / (self.head_dim ** 0.5)

        # Add spatial bias (distance)
        edge_bias = self.edge_mlp(edge_attr.unsqueeze(-1))
        attn = attn + edge_bias

        attn = softmax(attn, row)

        # Aggregate
        out = V[col] * attn.unsqueeze(-1)
        out = torch.zeros_like(Q).index_add_(0, row, out)

        out = out.reshape(-1, self.dim)
        out = self.out_lin(out)
            
        return out

"""
### V3: using MultiheadAttention in Pytorch (Claude Code)

class SpatialTransformerLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        # Replaces q_lin, k_lin, v_lin, and out_lin
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # edge bias (distance encoding)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, heads),
            nn.ReLU(),
            nn.Linear(heads, heads)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        N = x.size(0)

        # Build dense attention mask and add spatial edge bias
        # attn_mask shape: (N, N) — will be broadcast across heads
        attn_mask = torch.full((N, N), float('-inf'), device=x.device)

        # edge_bias: (num_edges, heads) → scatter into (N, N, heads)
        edge_bias = self.edge_mlp(edge_attr.unsqueeze(-1))  # (E, heads)

        # Fill valid edges with bias values; use mean across heads for the scalar mask
        # For per-head bias, we use attn_bias below instead
        attn_bias = torch.full((N, N, self.heads), float('-inf'), device=x.device)
        attn_bias[row, col] = edge_bias  # (N, N, heads)

        # MultiheadAttention expects attn_mask: (N, N) or (N*heads, N, N)
        # Reshape to (heads, N, N) then to (N*heads, N, N) for per-head bias
        attn_mask = attn_bias.permute(2, 0, 1).reshape(N * self.heads, N, N)  # (heads*N, N, N) — incorrect shape, fix:
        attn_mask = attn_bias.permute(2, 0, 1)  # (heads, N, N)

        # x needs shape (1, N, dim) for batch_first=True with a single graph
        x_in = x.unsqueeze(0)  # (1, N, dim)

        # attn_mask must be (N, N) or (batch*heads, N, N); pass (heads, N, N) → repeat for batch
        out, _ = self.attn(
            x_in, x_in, x_in,
            attn_mask=attn_mask.repeat(1, 1, 1)  # (heads, N, N)
        )

        out = out.squeeze(0)  # (N, dim)
        return out
"""

class SpatialTransformerAE(nn.Module):
    def __init__(self, in_dim, gene_dim, hidden_dim=256, latent_dim=32, heads=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Encoder
        self.enc1 = SpatialTransformerLayer(hidden_dim, heads)
        self.enc2 = SpatialTransformerLayer(hidden_dim, heads)

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        #self.dec1 = SpatialTransformerLayer(latent_dim, heads)
        #self.dec2 = SpatialTransformerLayer(latent_dim, heads)

        self.output_proj = nn.Linear(latent_dim, gene_dim)

    def forward(self, x, edge_index, edge_attr, details=False):
        # Encode
        h = F.relu(self.input_proj(x))

        h = h + self.enc1(h, edge_index, edge_attr)
        h = h + self.enc2(h, edge_index, edge_attr)

        z = self.to_latent(h)

        # Decode
        h = F.relu(z)

        #h = h + self.dec1(h, edge_index, edge_attr)
        #h = h + self.dec2(h, edge_index, edge_attr)

        out = self.output_proj(h)

        
        if details:
            return out, z
        else:
            return out
    

def fit(model, data,
        max_epochs=200,
        lr=1e-3,
        mask_ratio=0.3,
        stop_eps=1e-7,
        stop_tol=200,
        device="cpu",
        return_loss=False):

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    cnt = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        x_masked = mask_features(data, mask_ratio)

        out = model(x_masked, data.edge_index, data.edge_attr)

        loss = loss_fn(out, data.x, data.gene_dim)

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            print(f"Best : {best_loss:.4f}")
        
        if best_loss - loss.item() < stop_eps:
            cnt += 1
        else:
            cnt = 0
        if cnt >= stop_tol:
                print(f"Stopping criterion met. Final loss =  {loss.item():.4f}")
                break
        
        best_loss = min(best_loss, loss.item())

    else:
        #print(f"Maximum iterations reached. Final Loss: gene_loss =  {avg_gene_loss:.5f}, morpho_loss =  {avg_morpho_loss:.5f}")
        print(f"Maximum iterations reached. Final Loss:  {loss.item():.4f}")
    
    if return_loss:
        return loss.item()