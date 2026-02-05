###import ---------------------------------
import torch
import scanpy as sc
import pandas as pd
import sys
from joblib import Parallel, delayed

sys.path.append("..")
import steamboat as sf

###Data ----------------------------------
h5ad_file = "../../Data/Steamboat_Data/1/Ex1_hgsc/ST_Discovery_so.h5ad"

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res


###Model parameters ----------------------
n_heads = 25
sf.set_random_seed(0)

if __name__ == "__main__":
    
    # Load dataset
    adata = sc.read_h5ad(h5ad_file)

    # Load sample metadata
    sample_metadata = pd.read_excel(
        "../../Data/Steamboat_Data/2/Ex1_hgsc/sample_metadata.xlsx",
        index_col=0,
        sheet_name='Table 2b',
        skiprows=1
    )
    sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

    # Select untreated, adnexa samples
    mask = (sample_metadata['sites_binary'] == 'Adnexa') & (sample_metadata['treatment'] == 'Untreated')
    samples_of_interest = sample_metadata.index[mask].tolist()

    all_adata = adata[adata.obs['samples'].isin(samples_of_interest)].copy()
    all_adata.obs['cell.types.nolc'] = all_adata.obs['cell.types'].str.replace('_LC', '')

    # Create the model
    model = sf.Steamboat(adata.var_names.tolist(), n_heads=n_heads, n_scales=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    del adata

    # ------------------- Parallel processing -------------------
    def process_sample(sample_id):
        temp = all_adata[all_adata.obs['samples'] == sample_id].copy()
        if temp.shape[0] < 100:
            return None
        temp.obs['global'] = 0
        return temp

    # Run in parallel across CPUs
    adatas = Parallel(n_jobs=-1)(delayed(process_sample)(i) for i in all_adata.obs['samples'].unique())
    adatas = [x for x in adatas if x is not None]

    # Normalize and log transform
    adatas = sf.prep_adatas(adatas, norm=True, log1p=True, scale=False, renorm=False)

    # Create torch dataset
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

    # ------------------- Training -------------------
    model.fit(
        dataset.to(device),
        entry_masking_rate=0.1,
        feature_masking_rate=0.1,
        max_epoch=10000,
        loss_fun=torch.nn.MSELoss(reduction='sum'),
        opt=torch.optim.Adam,
        opt_args=dict(lr=0.1),
        stop_eps=1e-3,
        report_per=200,
        stop_tol=200,
        device=device
    )

    torch.save(model.state_dict(), 'saved_models/hgsc_new.pth')
