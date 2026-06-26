import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import scanpy as sc

from tqdm import tqdm

sys.path.append("./")
import Modified_Steamboat_CNN as MSC



parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

n_epochs = args.n_epochs
seed = args.seed


def set_random_seed(seed: int) -> None:
    """Reset seed for Numpy and PyTorch

    :param seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Using CPU")

# ─────────────────────────────────────────────
# Network Architecture
# ─────────────────────────────────────────────


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape  # exclude batch dim

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class ResNet50Encoder(nn.Module):
    def __init__(self, latent_dim, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        resnet = resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.Batch_norm = nn.BatchNorm1d(2048)
        self.fc_m = nn.Linear(2048, latent_dim).to(device)

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, 1)  # shape: (batch, 512)
        h = self.Batch_norm(h)
        h = F.relu(h)
        m = self.fc_m(h)
        return m

class Morpho_Decoder(nn.Module):
    def __init__(self, latent_dim, base_channels, output_dim):
        super().__init__()

        # Project latent vector to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 16 * 16),
            nn.ReLU()
        )


        self.decoder = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),

            # 8x8 → 16x16
            nn.ConvTranspose2d(base_channels, base_channels// 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),

            # 16x16 → 32x32
            nn.ConvTranspose2d(base_channels // 2, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),

            # 32x32 → 64x64
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(),

            nn.Conv2d(base_channels // 4, output_dim, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)                                    # (B, hidden_dim1 * 16)
        x = x.view(x.size(0), -1, 16, 16)                  # (B, hidden_dim1, 4, 4)
        x = self.decoder(x)                               # (B, output_dim, 64, 64)
        return x


class AE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.morpho_encoder = ResNet50Encoder(latent_dim)
        self.decoder = Morpho_Decoder(latent_dim, 16, 3)

    def forward(self, m):
        z = self.morpho_encoder(m)
        decoded = self.decoder(z)
        m_recon = decoded
        return m_recon, z

# ─────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────

def ae_loss(recon_x, x, loss_fun):
    # Reconstruction loss
    recon_loss = loss_fun(
        recon_x, x
    )

    return recon_loss

# ─────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────


def Train(model, max_epochs, loader, l_r):
  
  parameters = model.parameters()

  criterion = nn.MSELoss(reduction='mean')
  optimizer = optim.Adam(parameters, lr=l_r)
  scheduler = optim.lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=l_r,        # highest LR in the cycle
      total_steps = max_epochs*len(loader),
      pct_start=0.3,      # % of cycle to warm up
      anneal_strategy='cos'
  )
  save_dir = './HistoAEs/Breast Cancer/Weights'
  os.makedirs(
    save_dir,
    exist_ok=True
    )

  cnt = 0
  best_loss = np.inf
  for epoch in range(max_epochs):
      avg_loss = 0.
      i = 0
      for sample in tqdm(loader, desc=f"Epoch {epoch:2d} Training"):

        image = sample['image'].to(device)
        m_recon, _ = model(image)
        loss = ae_loss(m_recon, image, criterion)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
          print(f'batch: {i}, reconstruction_loss: {loss.item()}')

        i+=1
        scheduler.step()

      avg_loss /= len(loader)

      if epoch % 1 == 0:
        print(f'Epoch: {epoch}, reconstruction_loss: {avg_loss}')

      if avg_loss < best_loss:
        best_loss = avg_loss
        cnt = 0

      else:
        cnt += 1
        if cnt > 10:
          print('Early stopping!')
          break
  torch.save(
     model.state_dict(),
     os.path.join(save_dir,'Morpho_AE.pth')
     )

if __name__ == "__main__":

    set_random_seed(seed)

    ad = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/UNI_adata_r.h5ad")
    #adata = ad[:800].copy()
    adata = ad.copy()

    print("number of cells: ", adata.n_obs)
    print("number of genes: ", adata.n_vars)

    adata = MSC.prep_adatas(adata, norm=True, log1p=True)
    img_dir = '/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/extracted_images'
    dataset = MSC.make_dataset(adata, image_dir = img_dir, image_ext ='.png', sparse_graph=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=600,  # Adjust based on your GPU memory
        num_workers=3,
        shuffle=True,
    )

    model = AE(50).to(device)
    Train(model, n_epochs, dataloader, 1e-2)
