import argparse

import torch
import scanpy as sc

import csv
import os
import sys
from filelock import FileLock

#os.chdir("/content/drive/MyDrive/Thesis/Projects/Master_Thesis")
sys.path.append("./")
import GNN_Cross as SA

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))
else:
    print("No GPU !!!")

model_name = "GNN_Cross"

###################################
# Parse arguments
###################################
parser = argparse.ArgumentParser()

parser.add_argument("--mask_rate", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

mask_rate = args.mask_rate
seed = args.seed

MaskingRate = mask_rate

###################################
# Data
###################################
#adata = sc.read_h5ad("/content/drive/MyDrive/Thesis/Projects/Data/Breast_Cancer/FMs_3/UNI_adata.h5ad")
adata = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/UNI_adata.h5ad")

adata = SA.prep_adata(adata, norm=True, log1p=True)
data = SA.build_graph(adata, k=8, morph_key='p_Morpho_Embedding')

###################################
# Train
###################################
SA.set_random_seed(seed*10)
model = SA.SpatialTransformerAE(
    morpho_dim=data.x.shape[1]-data.gene_dim,
    gene_dim=data.gene_dim,
    hidden_dim=96,
    latent_dim=64,
    heads=4
)


loss = SA.model.fit(
    model, data, 
    max_epochs=10000, mask_ratio=MaskingRate, 
    stop_eps=1e-7, stop_tol=200, 
    device=device, return_loss=True
    )

save = False
if save:
    save_dir = "/content/drive/MyDrive/Thesis/Projects/Master_Thesis/Notebooks/GNN/saved_models"

    os.makedirs(
    save_dir,
    exist_ok=True
    )

    torch.save(
    model.state_dict(),
    os.path.join(save_dir,f'MI_MR_{MaskingRate}_Seed_{seed}.pth')
    )


###################################
# CSV File
###################################

csv_file = "results.csv"
lock = FileLock("results.csv.lock")

###################################
# prevent simultaneous writes
###################################
with lock:
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Model_Name", "MaskingRate", "run_id", "loss"])

        writer.writerow([model_name, MaskingRate, seed, loss])

print(f"MR {MaskingRate} RUN {seed}: loss={loss}")