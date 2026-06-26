import argparse
#import numpy as np
#import multiprocessing

import torch
from torch.utils.data import DataLoader

import scanpy as sc

import csv
import os
import sys
from filelock import FileLock

sys.path.append("./")
import Modified_Steamboat as MS

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))

model_name = "Modified Steamboat (UNI)"
return_loss = True


###################################
# Parse arguments
###################################
parser = argparse.ArgumentParser()

parser.add_argument("--mask_rate", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)

args = parser.parse_args()

mask_rate = args.mask_rate
seed = args.seed


if mask_rate == 0.9:
    save = True
else:
    save = False


MaskingRate = mask_rate

if __name__ == "__main__":

    ###################################
    # Data
    ###################################
    #adata = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/hoptimus_adata_r.h5ad")
    adata = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/UNI_adata_r.h5ad")
    adata = MS.prep_adatas(adata, norm=True, log1p=True)
    dataset = MS.make_dataset(adata, sparse_graph=True)


    ###################################
    # Train
    ###################################

    MS.set_random_seed(seed*10)
    model = MS.model.Steamboat(
        features=len(adata.var_names.tolist()), 
        morpho_features=adata.obsm['p_Morpho_Embedding'].shape[1], 
        n_heads=50,
        n_scales=2
        )
    model = model.to(device)

    loss, alpha = model.fit(
        dataset, 
        entry_masking_rate=MaskingRate,
        device=device,
        #max_epoch=10000,
        max_epoch=8000,
        loss_fun=torch.nn.MSELoss(reduction='mean'),
        opt=torch.optim.Adam,
        opt_args=dict(lr=0.01),
        stop_eps=1e-7, report_per=50, stop_tol=200, 
        return_loss=return_loss
        )

    if return_loss:

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
                    writer.writerow(["Model_Name", "MaskingRate", "RandomSeed", "loss", "alpha"])

                writer.writerow([model_name, MaskingRate, seed, loss, alpha])

        print(f"MR {MaskingRate} RUN {seed}: loss={loss} alpha={alpha}")


    if save:
            
        save_dir = "./Train/Modified_Steamboat/saved_models"

        os.makedirs(
        save_dir,
        exist_ok=True
        )

        torch.save(
        model.state_dict(),
        os.path.join(save_dir,f'MS_MR_{MaskingRate}_Seed_{seed}.pth')
        )