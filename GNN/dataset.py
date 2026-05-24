import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

import scanpy as sc



def prep_adata(adata: sc.AnnData, norm=True, log1p=True) -> list[sc.AnnData]:

    if norm:
        sc.pp.normalize_total(adata)
    if log1p:
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    return adata

def build_graph(adata, k=8, coord_key="spatial", gene_key="X", morph_key="morph_emb"):

    gene = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    morph = adata.obsm[morph_key]

    x = np.concatenate([gene, morph], axis=1)
    #x = gene
    x = torch.tensor(x, dtype=torch.float32)

    coords = adata.obsm[coord_key]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edge_list = []
    edge_attr = []

    for i in range(indices.shape[0]):
        for d, j in zip(distances[i][1:], indices[i][1:]):
            edge_list.append([i, j])
            edge_list.append([j, i])

            edge_attr.append(d)
            edge_attr.append(d)

    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.gene_dim = gene.shape[1]

    return data
