import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from sklearn.neighbors import NearestNeighbors
import numpy as np


def mask_features(data, mask_ratio=0.3):
    x = data.x.clone()
    gene_dim = data.gene_dim

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

    def forward(self, x_m1, x_m2, edge_index, edge_attr):
        row, col = edge_index

        Q = self.q_lin(x_m1).view(-1, self.heads, self.head_dim)
        K = self.k_lin(x_m2).view(-1, self.heads, self.head_dim)
        V = self.v_lin(x_m2).view(-1, self.heads, self.head_dim)

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


class SpatialTransformerAE(nn.Module):
    def __init__(self, morpho_dim, gene_dim, hidden_dim=256, latent_dim=32, heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Input projection
        self.input_proj_morpho = nn.Linear(morpho_dim, hidden_dim)
        self.input_proj_gene = nn.Linear(gene_dim, hidden_dim)

        # Encoder
        self.enc1 = SpatialTransformerLayer(hidden_dim, heads)
        self.enc2 = SpatialTransformerLayer(hidden_dim, heads)

        self.to_latent = nn.Linear(hidden_dim*2, latent_dim)

        # Decoder
        #self.dec1 = SpatialTransformerLayer(latent_dim, heads)
        #self.dec2 = SpatialTransformerLayer(latent_dim, heads)

        self.output_proj = nn.Linear(latent_dim, gene_dim)

    def forward(self, x_m, x_g, edge_index, edge_attr, details=False):
        # Encode
        h_m = F.relu(self.input_proj_morpho(x_m))
        h_g = F.relu(self.input_proj_gene(x_g))

        h_m = h_m + self.enc1(h_m, h_g, edge_index, edge_attr)
        h_g = h_g + self.enc1(h_g, h_m, edge_index, edge_attr)
        
        h_m = h_m + self.enc2(h_m, h_g, edge_index, edge_attr)
        h_g = h_g + self.enc2(h_g, h_m, edge_index, edge_attr)

        h = torch.concatenate((h_m,h_g), dim=-1)
        z = self.to_latent(h)

        # Decode
        z = F.relu(z)
        out = self.output_proj(z)

        
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
        x_g = x_masked[:, :data.gene_dim]
        x_m = x_masked[:, data.gene_dim:]

        out = model(x_m, x_g, data.edge_index, data.edge_attr)

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