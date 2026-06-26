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


save_dir = "./SteamBoat/examples/saved_models"

if __name__ == "__main__":

    ###################################
    # Data
    ###################################

    ad = sc.read_h5ad("/data/horse/ws/mosa505e-Multimodal_Rep/data/Breast_Cancer/ann_data.h5ad")
    adata = ad.copy()
    adatas = []
    for i in adata.obs['region'].unique():
        adatas.append(adata[adata.obs['region'] == i])
        adatas[-1].obs['global'] = 0  #Only support one unique value for regional observation.

    adatas = sf.prep_adatas(adatas, norm=True, log1p=True)
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

    ###################################
    # Inference
    ###################################
    model = sf.model.Steamboat(adatas[0].var_names.tolist(), n_heads=64, n_scales=3)
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(save_dir,'SB_Bias_MR_0.8_Seed_0.pth'), map_location=torch.device(device),weights_only=True
            ), strict=False
            )

    for i, (x, adj_list, regional_xs, regional_adj_lists) in enumerate(dataset):
        print('slide number', i+1)
        # Move tensors to the selected device
        adj_list = adj_list.squeeze(0).to(device)
        x = x.squeeze(0).to(device)
        regional_adj_lists = [regional_adj_list.to(device) for regional_adj_list in regional_adj_lists]
        regional_xs = [regional_x.to(device) for regional_x in regional_xs]

        with torch.no_grad():
            res, details = model(adj_list, x, x, regional_adj_lists, regional_xs, get_details=True)
        
    ad.obsm['attn'] = details['attn'].cpu().numpy()
    ad_path = os.path.join(save_dir,'SB_latent.h5ad')
    ad.write_h5ad(ad_path)