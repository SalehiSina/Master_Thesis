import argparse

import torch
import scanpy as sc

import csv
import os
import sys
from filelock import FileLock

sys.path.append("./")
import HadmardAttention as HA

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))

model_name = "Hadamard_Full"

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
adata = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/hoptimus_adata.h5ad")
adata = HA.prep_adatas(adata, norm=True, log1p=True)
dataset = HA.make_dataset(adata, sparse_graph=True)


###################################
# Train
###################################
model_type = 0

HA.set_random_seed(seed*10)
model = HA.model.Steamboat(
    features=len(adata.var_names.tolist()), 
    morpho_features=adata.obsm['p_Morpho_Embedding'].shape[1], 
    n_heads=32, model_type=model_type, 
    n_scales=2
    )
model = model.to(device)

loss = model.fit(dataset, entry_masking_rate=MaskingRate,
          device=device,
          max_epoch=10000,
          loss_fun=torch.nn.MSELoss(reduction='mean'),
          opt=torch.optim.Adam, sched= None,
          max_lr=None, opt_args=dict(lr=0.01), stop_eps=1e-7, 
          report_per=200, stop_tol=200, return_loss=True)

###################################
# CSV File
###################################

csv_file = "results_h_optimus.csv"
lock = FileLock("results_h_optimus.csv.lock")

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