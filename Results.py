import warnings

warnings.filterwarnings("ignore")

#############################
# Import Libraries
#############################

import torch
import torch.nn as nn
import scanpy as sc
import squidpy as sq
from tqdm.notebook import tqdm
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import argparse
import os

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = "cuda"
    print("GPU: ",torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Using CPU")

os.chdir("/content/drive/MyDrive/Thesis/Projects/Master_Thesis/")

#############################
# Parse arguments
#############################
parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--key", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--operation", type=str, required=True)

args = parser.parse_args()

ann_dir = args.dir
dataset_name = args.name
embedding_key = args.key
target = args.label
clustering_type = args.operation


#############################
# k-means clustering
#############################
def km(adata, embed_key, k):

    X = np.array(adata.obsm[embed_key])
    X_scaled = StandardScaler().fit_transform(X)
    
    # --- Fit KMeans with chosen k ---
    model = KMeans(
        n_clusters=k,
        init="k-means++",   # smart centroid seeding
        n_init="auto",       # number of random restarts, best inertia kept
        max_iter=300,
        tol=1e-4,
        random_state=42,
    )
    labels = model.fit_predict(X_scaled)
    
    print("Inertia:", model.inertia_)
    print("Iterations run:", model.n_iter_)

    adata.obs['pred_label'] = labels
    print('Done!')

    return adata
    

#############################
# data_split for classification
#############################

def split(adata, target, test_share):

    label_key = target

    # Cell indices
    cell_idx = np.arange(adata.n_obs)

    # Train/test split (stratified by ROI labels)
    train_idx, test_idx = train_test_split(
        cell_idx,
        test_size=test_share,
        random_state=42,
        stratify=adata.obs[label_key]
    )

    # Boolean masks
    train_mask = np.zeros(adata.n_obs, dtype=bool)
    test_mask = np.zeros(adata.n_obs, dtype=bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True


    # Create train/test AnnData objects if needed
    adata_train = adata[train_mask].copy()
    adata_test = adata[test_mask].copy()

    return adata_train, adata_test



#############################
# classification network
#############################


# --- Define model ---

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- Train Funcion ---
def classification(adata_train, adata_test, embed_key, target):


    X_train = adata_train.obsm[embed_key].astype(np.float32)

    X_test = adata_test.obsm[embed_key].astype(np.float32)
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(adata_train.obs[target])
    y_test = le.fit_transform(adata_test.obs[target])

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        num_classes=len(le.classes_)
        )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)


    # ----------------------------
    # Training
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in tqdm(range(epochs), desc= 'Classification Training'):
        model.train()

        optimizer.zero_grad()

        logits = model(X_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y_train).float().mean().item()
            print(f"Epoch {epoch+1:3d} | Loss={loss.item():.4f} | Train Acc={acc:.4f}")

    # ----------------------------
    # Evaluation
    # ----------------------------
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        pred = logits.argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test.numpy(), pred)

    print(f"\nTest Accuracy: {test_acc:.4f}\n")

    print(classification_report(
        y_test.numpy(),
        pred,
        target_names=le.classes_
    ))



if __name__ == "__main__":
    
    print("Loading anndata object ... \n")
    print("Location: ", ann_dir)

    adata = sc.read_h5ad(ann_dir)


    #############################
    # Dimension Reduction
    #############################

    ad = adata.copy()
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(ad.obsm[embedding_key])
    ad.obsm['reduced_dim_embedding'] = X_reduced

    if clustering_type == 'UMAP':
        #############################
        # UMAP
        #############################
        print('... UMAP ...')
        sc.pp.neighbors(ad, use_rep='reduced_dim_embedding', n_neighbors=15)
        sc.tl.umap(ad)
        sc.pl.umap(ad, color=target, title='')

        plt.savefig(f"figures/{dataset_name}_{embedding_key}_umap_{embedding_key}.png", dpi=300, bbox_inches="tight")
        
        print(f" saved in figures/{dataset_name}_{embedding_key}_umap_{embedding_key}")
        plt.close()

    #############################
    # leiden
    #############################
    elif clustering_type == 'leiden':
        print('... leiden ... ')

        if target == "annotation":
            res = 0.2
            name = "segmentation"

        elif target == "cell_type":
            res = 0.8
            name = "cell_type"
        else:
            print("Please Specify the label to be annotation or cell_type")
            print("Labels set to be annotation")
            res = 0.2
            target = "annotation"
            name = "segmentation"

        sc.pp.neighbors(ad, use_rep='reduced_dim_embedding', key_added = 'added', n_neighbors=200)
        sc.tl.leiden(ad, obsp='added_connectivities', key_added='_clusters_', resolution=res)

        sq.pl.spatial_scatter(
            ad, color=[target], title = '', shape=None, figsize=(10, 5),
            ncols=2, legend_loc='right margin', frameon=False, size=2., lw=0., wspace=0.0,
            hspace=0.0, save=f'figures/{dataset_name}_{target}.jpeg'
            )
        print(f'saved in figures/{dataset_name}_{target}')


        sq.pl.spatial_scatter(
            ad, color=['_clusters_'], title = '', shape=None, figsize=(10, 5),
            ncols=2, legend_loc=None, frameon=False, size=2., lw=0., wspace=0.0,
            hspace=0.0, save=f'figures/{dataset_name}_{embedding_key}_{name}_leiden.jpeg'
            )
        print(f'saved in figures/{dataset_name}_{embedding_key}_{name}_leiden')


    #############################
    # k-means
    #############################
    elif clustering_type == 'k-means':
        print('... k-means clustering ...')

        if target == "annotation":
            k = len(np.unique(ad.obs['annotation']))
            print('K = ', k)
            name = "segmentation"

        elif target == "cell_type":
            k = len(np.unique(ad.obs['cell_type']))
            print('K = ', k)
            name = "cell_type"
        else:
            print("Please Specify the label to be annotation or cell_type")
            print("Labels set to be annotation")
            k = len(np.unique(ad.obs['annotation']))
            print('K = ', k)
            name = "segmentation"

        ad = km(ad, embedding_key, k)


        sq.pl.spatial_scatter(
            ad, color=['pred_label'], title = '', shape=None, figsize=(10, 5),
            ncols=2, legend_loc=None, frameon=False, size=2., lw=0., wspace=0.0,
            hspace=0.0, save=f'figures/{dataset_name}_{embedding_key}_{name}_kmeans.jpeg'
            )
        print(f'saved in : figures/{dataset_name}_{embedding_key}_{name}_kmeans')

        
        ari = adjusted_rand_score(ad.obs['annotation'].to_numpy(), ad.obs['pred_label'].to_numpy())
        print(f"{dataset_name}_{embedding_key}_{name}_kmeans_ARI: ", ari)

    
    #############################
    # Supervised Classification
    #############################
    elif clustering_type == 'Supervised':
        ad_train, ad_test = split(ad, target, test_share = 0.8)
        classification(ad_train, ad_test, embedding_key, target)

    else:
        print("Please enter a valid operation")

        
    print('\n Finished!')
    