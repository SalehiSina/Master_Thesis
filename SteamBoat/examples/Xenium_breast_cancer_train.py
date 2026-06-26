import argparse

import torch
import scanpy as sc

import csv
import os
import sys
from filelock import FileLock

sys.path.append("./SteamBoat")
import steamboat as sf

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))

model_name = "Steamboat"
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

MaskingRate = mask_rate

if mask_rate == 0.8:
    save = True
elif mask_rate == 0.0:
    save = True
else:
    save = False


if __name__ == "__main__":

    ###################################
    # Data
    ###################################

    adata = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/ann_data.h5ad")
    adatas = []
    for i in adata.obs['region'].unique():
        adatas.append(adata[adata.obs['region'] == i])
        adatas[-1].obs['global'] = 0  #Only support one unique value for regional observation.

    adatas = sf.prep_adatas(adatas, norm=True, log1p=True)
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

    ###################################
    # Train
    ###################################
    sf.set_random_seed(seed*10)
    model = sf.model.Steamboat(adatas[0].var_names.tolist(), n_heads=64, n_scales=3)
    model = model.to(device)
    loss = model.fit(
        dataset, 
        entry_masking_rate=MaskingRate, 
        feature_masking_rate=0,
        device=device,
        max_epoch=10000,
        loss_fun=torch.nn.MSELoss(reduction='mean'),
        opt=torch.optim.Adam,
        sched= None,
        max_lr=None, 
        opt_args=dict(lr=0.01), 
        stop_eps=1e-7, 
        report_per=200, 
        stop_tol=200,
        return_loss=True
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
                    writer.writerow(["Model_Name", "MaskingRate", "run_id", "loss"])

                writer.writerow([model_name, MaskingRate, seed, loss])

        print(f"RUN {seed}: loss={loss}")


    if save:
            
        save_dir = "./SteamBoat/examples/saved_models"

        os.makedirs(
        save_dir,
        exist_ok=True
        )

        torch.save(
        model.state_dict(),
        os.path.join(save_dir,f'SB_Bias_MR_{MaskingRate}_Seed_{seed}.pth')
        )