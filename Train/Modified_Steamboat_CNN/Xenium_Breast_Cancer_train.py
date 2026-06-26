import argparse
import numpy as np
import multiprocessing

import torch
from torch.utils.data import DataLoader

import scanpy as sc

import csv
import os
import sys
from filelock import FileLock

sys.path.append("./")
import Modified_Steamboat_CNN as MSC

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))

model_name = "Steamboat_CNN_&_hoptimus"
return_loss = True
save = False

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

if __name__ == "__main__":

    ###################################
    # Data
    ###################################
    ad = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/FMs_3/hoptimus_adata_r.h5ad")
    #adata = ad[:1000].copy()
    adata = ad.copy()
    #adata.obsm['p_Morpho_Embedding'] = np.zeros_like(adata.obsm['p_Morpho_Embedding'])

    print("number of cells: ", adata.n_obs)
    print("number of genes: ", adata.n_vars)

    adata = MSC.prep_adatas(adata, norm=True, log1p=True)
    img_dir = '/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/extracted_images'
    dataset = MSC.make_dataset(adata, image_dir = img_dir, image_ext ='.png', sparse_graph=True)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=167780//6,  # Adjust based on your GPU memory
        shuffle=True,
    )

    print(len(dataloader.dataset)*dataloader.dataset.X_global.shape[1])

    ###################################
    # Train
    ###################################

    MSC.set_random_seed(seed*10)
    model = MSC.model.Steamboat(
        features=len(adata.var_names.tolist()),
        image_size=dataset[0]['image'].shape,
        morpho_features=adata.obsm['p_Morpho_Embedding'].shape[1], 
        n_heads=32
        )
    model = model.to(device)

    loss = model.fit(
        dataloader, 
        entry_masking_rate=mask_rate,
        device=device,
        max_epoch=5000,
        loss_fun=torch.nn.MSELoss(reduction='sum'),
        opt=torch.optim.Adam,
        opt_args=dict(lr=0.01),
        stop_eps=1e-7, report_per=200, stop_tol=200, 
        return_loss=True,
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
                    writer.writerow(["Model_Name", "MaskingRate", "RandomSeed", "loss"])

                writer.writerow([model_name, MaskingRate, seed, loss])

        print(f"MR {MaskingRate} RUN {seed}: loss={loss}")


    if save:
            
        save_dir = "./Train/Modified_Steamboat_CNN/saved_models"

        os.makedirs(
        save_dir,
        exist_ok=True
        )

        torch.save(
        model.state_dict(),
        os.path.join(save_dir,f'MR_{MaskingRate}_Seed_{seed}.pth')
        )

