
#--------------------
##Import
#--------------------

import os
import pandas as pd
import numpy as np
import anndata
from PIL import Image

import albumentations as trans
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm

from huggingface_hub import login
import timm 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


#---------------------
##Parameters
#---------------------

HE_dir = "/home/mosa505e/thesis_horse/data/Mouse_Brain/Croped_Images_3"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.distributed.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = local_rank
print("device = ", device)

#Basic Transformation
base_transform = transforms.Compose([
        transforms.CenterCrop(312),
        transforms.Resize((224, 224)),
    ])

class Histo_images(Dataset):
    def __init__(self, patch_paths, transform=base_transform):
        self.transform = transform
        self.cell_ids = []
        self.paths = []

        for p in tqdm(patch_paths):
          self.paths.append(p)
          cell_id = p[:-4]
          self.cell_ids.append(cell_id)

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, index):
        p = self.paths[index]
        img = Image.open(p).convert("RGB")
        img = base_transform(img)
        img = transform(img)

        return self.cell_ids[index], img


def Embed(model, Data_loader, device):

    model.eval()  # set eval mode
    all_cell_ids = []
    embedded_vectors = []

    with torch.inference_mode():  # no_grad

        for cell_ids, frames in tqdm(Data_loader, desc='Inference'):

            # Forward pass
            outputs = model(frames.to(device))


            class_token = outputs[:, 0]    # size: 1 x 1280
            patch_tokens = outputs[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            embedding = embedding.to(torch.float16)
            # Store results
            all_cell_ids.extend(cell_ids)
            embedded_vectors.append(embedding.detach().cpu())

    # Concatenate all embedded vectors → shape: (N, embed_dim)
    embedded_vectors = torch.cat(embedded_vectors, dim=0)

    return all_cell_ids, embedded_vectors


#---------------------
##Main
#---------------------
if __name__ == "__main__":
    print("Device: ", device)

    # need to specify MLP layer and activation function for proper init
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model = model.to(device)
    model = model.eval()

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    img_list = os.listdir(HE_dir)
    patch_paths = [os.path.join(HE_dir, fn) for fn in img_list]
    Data_set = Histo_images(transform=transform, patch_paths=patch_paths)
    Data_loader = DataLoader(Data_set, batch_size=256*3, shuffle=False, drop_last=False)

    cell_ids, E_vectors = Embed(model, Data_loader, device)

    adata = anndata.AnnData(
    X = np.zeros((len(cell_ids), 1))
    )

    # Store embeddings
    adata.obs["cell_id"] = cell_ids
    adata.obsm["virchow"] = E_vectors.numpy()

    # Save to file
    adata_path = "/home/mosa505e/thesis_horse/data/Mouse_Brain/FMs_3/virchow.h5ad"
    os.makedirs(os.path.dirname(adata_path), exist_ok=True)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        adata.write(adata_path)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    

    torch.distributed.destroy_process_group()