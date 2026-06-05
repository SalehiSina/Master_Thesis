import torch
from torch.utils.data import Dataset

import numpy as np
import scanpy as sc


def prep_adata(adata: sc.AnnData, norm=True, log1p=True) -> sc.AnnData:

    if norm:
        sc.pp.normalize_total(adata)
    if log1p:
        sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    return adata


class AEDataset(Dataset):
    def __init__(self, adata: sc.AnnData, morph_key: str = "morph_emb"):
        """AE Dataset class
        """
        super().__init__()
        self.adata = adata
        self.gene_vectors = []
        self.morph_key = morph_key
        self.morph_vectors = []
        self.X = []

        for i in range(adata.shape[0]):
            gene_vector = adata.X[i].toarray().squeeze() if not isinstance(adata.X, np.ndarray) else adata.X[i]
            if morph_key == "none":
                morph_vector = np.ndarray((adata.X.shape[0],0))
            else:
                morph_vector = adata.obsm[morph_key][i]
            self.gene_vectors.append(gene_vector)
            self.morph_vectors.append(morph_vector)
            
        for gene_vector, morph_vector in zip(self.gene_vectors, self.morph_vectors):
            self.X.append(np.concatenate([gene_vector, morph_vector]))

    def __len__(self):
        return len(self.gene_vectors)

    def __getitem__(self, index):
        return self.X[index]

