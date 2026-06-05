import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def mask_features(x, gene_dim, mask_ratio=0.3):

    x_masked = x.clone()
    if mask_ratio > 0.:
        random_mask = torch.rand_like(x[:, :gene_dim], device=x.device) < mask_ratio 
        x_masked[:, :gene_dim].masked_fill_(random_mask, 0.)

    return x_masked

def loss_fn(pred, target, gene_dim):
    return F.mse_loss(pred, target[:, :gene_dim])



class AE(nn.Module):
    def __init__(self, in_dim, gene_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.gene_dim = gene_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(latent_dim, gene_dim)

    def forward(self, x, details=False):
        z = self.encoder(x)
        out = self.decoder(z)

        if details:
            return out, z
        else:
            return out
    



def fit(
    model,
    dataset,
    gene_dim,
    max_epochs=200,
    lr=1e-3,
    mask_ratio=0.3,
    stop_eps=1e-7,
    stop_tol=20,
    batch_size=None,
    device="cuda",
    return_loss=False,
):

    if batch_size is None:
        batch_size = len(dataset)
        print(f"Batch size not provided. Using batch size = {batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    final_loss = None
    cnt = 0

    for epoch in range(max_epochs):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x in dataloader:

            x = x.to(device)

            optimizer.zero_grad()

            x_masked = mask_features(
                x,
                gene_dim,
                mask_ratio,
            )

            out = model(x_masked)

            loss = loss_fn(out, x, gene_dim)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        final_loss = epoch_loss

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {epoch_loss:.6f} | "
                f"Best: {best_loss:.6f}"
            )

        # Early stopping
        if best_loss - epoch_loss > stop_eps:
            best_loss = epoch_loss
            cnt = 0
        else:
            cnt += 1

        if cnt >= stop_tol:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best loss = {best_loss:.6f}"
            )
            break

    else:
        print(
            f"Maximum epochs reached. "
            f"Best loss = {best_loss:.6f}"
        )


    if return_loss:
        return final_loss