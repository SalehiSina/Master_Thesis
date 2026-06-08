from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
import scipy as sp
from PIL import Image

from torchvision import transforms

from tqdm import tqdm
import os


# ─────────────────────────────────────────────
# Default image transform
# ─────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class M_SteamboatDataset(Dataset):
    """
    Per-cell dataset where each sample returns:
      - X   : gene expression of the cell itself              [n_genes]
      - X_global  : gene expression of ALL cells              [N, n_genes]
      - image     : morphology image loaded from disk         [C, H, W]
      - adj       : neighbourhood indices for the cell        [2, k] or [k] depending on graph type
      - cell_idx  : integer index of this cell in the adata   scalar

    Design notes
    ────────────
    • X_global is stored once on the dataset and returned as a *shared reference*
      on every __getitem__ call (no copying).  In a DataLoader with num_workers > 0
      this reference is pickled per worker, so keep it as a contiguous float32
      tensor for fast serialisation.  If the full matrix is too large for RAM you
      can replace it with a memory-mapped numpy array or an HDF5 handle.

    • Images are loaded lazily (on __getitem__) so the dataset object stays small
      and multiple DataLoader workers don't share open file handles.

    • The adjacency tensor keeps the same format as the original code (either a
      [2, N*k] COO tensor for regular k-NN graphs or a [3, N*max_k] padded tensor
      with a validity mask for irregular graphs).  We slice out only the rows
      belonging to the requested cell so the DataLoader can collate them.
    """

    def __init__(
        self,
        data: list[dict],          # list with one dict per cell (see make_dataset)
        X_global: torch.Tensor,    # [N, n_genes]  — full expression matrix
        cell_image_paths: list[str],  # length-N list of absolute image paths
        sparse_graph: bool = True,
        transform=base_transform,
    ):
        super().__init__()
        self.data             = data            # list of per-cell dicts
        self.X_global         = X_global        # shared; never copied per sample
        self.cell_image_paths = cell_image_paths
        self.sparse_graph     = sparse_graph
        self.transform        = transform

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        sample = self.data[index]

        # ── 1. Local gene expression ──────────────────────────────────
        X_local = sample['X']           # [n_genes]

        # ── 2. Global gene expression (all cells) ────────────────────
        # Returned as a reference; DataLoader will stack these into
        # [batch, N, n_genes].  For very large N consider returning only
        # X_local here and passing X_global separately as a collate hook.
        X_global = self.X_global        # [N, n_genes]

        # ── 3. Morphology image from disk ────────────────────────────
        img_path = self.cell_image_paths[index]
        image = self._load_image(img_path)  # [C, H, W]

        # ── 4. Adjacency (neighbours of this cell only) ───────────────
        adj = sample['adj']             # pre-sliced in make_dataset

        # ── 5. Cell index (useful for downstream look-ups) ────────────
        cell_idx = torch.tensor(index, dtype=torch.long)

        return {
            'X_local':  X_local,    # [n_genes]
            'X_global': X_global,   # [N, n_genes]  ← global context
            'image':    image,      # [C, H, W]
            'adj':      adj,        # neighbour indices/mask
            'cell_idx': cell_idx,   # scalar
        }

    # ------------------------------------------------------------------
    def _load_image(self, path: str) -> torch.Tensor:
        """Load a single cell image; fall back to a zero tensor on error."""
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                return self.transform(img)
            return transforms.ToTensor()(img)
        except Exception as e:
            print(f"[M_SteamboatDataset] Warning: could not load image at '{path}': {e}")
            # Return a black image of the expected size so the batch can still collate
            return torch.zeros(3, 224, 224)


# ─────────────────────────────────────────────
# Preprocessing helper function
# ─────────────────────────────────────────────
def prep_adatas(
    adata: sc.AnnData,
    n_neighs: int = 8,
    norm=True,
    log1p=True,
    scale=False,
    renorm=False,
) -> sc.AnnData:
    if norm:
        sc.pp.normalize_total(adata)
    if log1p:
        sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10)
    if renorm:
        sc.pp.normalize_total(adata, target_sum=100, zero_center=False)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    return adata


# ─────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────
def make_dataset(
    adata: sc.AnnData,
    image_dir: str,
    image_col: str = 'image_id',   # adata.obs column that holds the image filename
    image_ext: str = '.png',       # extension to append if not already in image_col
    sparse_graph: bool = True,
    mask_var=False,
    obsm_key=None,
    transform=base_transform,
) -> M_SteamboatDataset:
    """
    Build an M_SteamboatDataset from a single AnnData object.

    Parameters
    ----------
    adata       : preprocessed AnnData (output of prep_adatas)
    image_dir   : directory that contains per-cell images
    image_col   : column in adata.obs with the image filename (stem or full name)
    image_ext   : extension appended when image_col contains only the stem
    sparse_graph: whether to return COO adjacency (True) or dense bool (False)
    mask_var    : obs column to subset genes, or False to use all
    obsm_key    : tuple (X_key, Morpho_key) to read from obsm instead of .X
    transform   : torchvision transform applied to each image
    """

    # ── Variable masking (HVG) ───────────────────────────────────────
    if mask_var is None:
        if 'highly_variable' in adata.var.columns:
            print("Using 'highly_variable' to mask variables. "
                  "Pass mask_var=False to use all genes.")
            mask_var = 'highly_variable'
        else:
            mask_var = False

    if mask_var:
        assert mask_var in adata.var.columns, \
            f"adata does not have '{mask_var}' in .var"

    # ── Expression matrices ───────────────────────────────────────────
    if obsm_key is None:
        X_raw = adata.X
    else:
        X_raw = adata.obsm[obsm_key[0]]

    # Convert full matrix to float32 numpy once
    if isinstance(X_raw, sp.sparse.spmatrix):
        X_np = X_raw.astype(np.float32).toarray()   # [N, n_genes]
    else:
        X_np = np.asarray(X_raw, dtype=np.float32)  # [N, n_genes]

    # X_global: shared tensor over ALL cells — the key change for req. 1
    X_global = torch.from_numpy(X_np)               # [N, n_genes]

    # ── Spatial graph  (built once; then sliced per cell below) ──────
    N = adata.shape[0]

    if sparse_graph:
        v, u = adata.obsp['spatial_connectivities'].nonzero()
        k0   = u.shape[0] / N
        k    = int(np.round(k0))

        order = np.argsort(v)
        u = u[order]
        v = v[order]

        regular = (np.abs(k - k0) < 1e-6 and
                   (v.reshape([-1, k]) == np.arange(N)[:, None]).all())

        if regular:
            # Shape [N, k]: row i holds the k neighbour indices of cell i
            adj_per_cell = u.reshape(N, k)          # numpy int array
            adj_type     = 'regular'
        else:
            ks    = np.array(adata.obsp['spatial_connectivities']
                             .sum(axis=0)).squeeze().astype(int)
            max_k = int(ks.max())
            print("Not all cells have the same number of neighbours. "
                  "Steamboat will pad with self-loops and provide a mask.")

            adj_u    = np.zeros((N, max_k), dtype=np.int64)
            adj_mask = np.zeros((N, max_k), dtype=np.int64)

            pt = 0
            for i in range(N):
                pt2 = pt + ks[i]
                adj_u[i, :ks[i]]    = u[pt:pt2]
                adj_u[i, ks[i]:]    = i          # self-loop padding
                adj_mask[i, :ks[i]] = 1
                pt = pt2

            adj_per_cell = (adj_u, adj_mask)     # tuple of [N, max_k] arrays
            adj_type     = 'irregular'
    else:
        # Dense boolean adjacency stored as float for collation convenience
        adj_dense    = (adata.obsp['spatial_connectivities'] == 1).toarray() \
                       .astype(np.float32)        # [N, N]
        adj_per_cell = adj_dense
        adj_type     = 'dense'

    # ── Image paths ───────────────────────────────────────────────────
    # Build the path list here so __getitem__ only calls os.path.join
    def _image_path(obs_row):
        fname = str(obs_row[image_col])
        if not fname.endswith(image_ext):
            fname = fname + image_ext
        return os.path.join(image_dir, fname)

    cell_image_paths = [_image_path(adata.obs.iloc[i]) for i in range(N)]

    # ── Per-cell data dicts ───────────────────────────────────────────
    data_list = []
    for i in range(N):
        cell = {}

        # Local expression vector
        cell['X'] = X_global[i]       # [n_genes]  — view into X_global

        # Adjacency slice for this cell
        if adj_type == 'regular':
            # neighbours: shape [k]
            cell['adj'] = torch.from_numpy(adj_per_cell[i])
        elif adj_type == 'irregular':
            adj_u, adj_mask = adj_per_cell
            # Stack neighbours + mask: shape [2, max_k]
            cell['adj'] = torch.from_numpy(
                np.stack([adj_u[i], adj_mask[i]], axis=0))
        else:  # dense
            # Full row of the adjacency matrix: shape [N]
            cell['adj'] = torch.from_numpy(adj_per_cell[i])

        data_list.append(cell)

    return M_SteamboatDataset(
        data=data_list,
        X_global=X_global,
        cell_image_paths=cell_image_paths,
        sparse_graph=sparse_graph,
        transform=transform,
    )