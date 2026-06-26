import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .dataset import SteamboatDataset
import os

from tqdm import tqdm

class Morpho_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, hidden_dim=1024):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class Gene_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class ScalarGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_fc = nn.Linear(2 * dim, 1)
        #self.gate_fc = nn.Linear(dim, 1)
        
    def forward(self, z_gene, z_morph):
        # z_gene, z_morph: [B, dim]
        gate_input = torch.cat([z_gene, z_morph], dim=-1)
        alpha = torch.sigmoid(self.gate_fc(gate_input))  # [B, 1]
        #alpha = torch.sigmoid(self.gate_fc(z_gene))  # [B, 1]
        #fused = z_gene + alpha * z_morph
        fused = alpha * z_gene + (1 - alpha) * z_morph
        #fused = torch.cat([alpha * z_gene, (1 - alpha) * z_morph], dim=-1)
        return fused, alpha
    
class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        """Nonnegative linear layer

        :param d_in: number of input features
        :param d_out: number of output features
        :param bias: if True, adds a learnable (unconstrained) bias term
        """
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
        self.elu = nn.ELU()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """transform weight matrix to be non-negative 

        :return: transformed weight matrix
        """
        return self.elu(self._weight) + 1

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    
class NonNegScale(nn.Module):
    def __init__(self, d) -> None:
        """Non-negative bias layer (i.e., add a non-negative vector to the output)

        :param d: number of input/output features
        """
        super().__init__()
        self._scale = torch.nn.Parameter(torch.zeros(1, d))
        self.elu = nn.ELU()

    @property
    def scale(self):
        """Transform bias to be non-negative

        :return: non-negative bias
        """
        return self.elu(self._scale) + 1

    def forward(self, x):
        return x * self.scale

class NonNegScale3(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self._scale = torch.nn.Parameter(torch.zeros(1, d, 1))
        self.elu = nn.ELU()

    @property
    def scale(self):
        return self.elu(self._scale) + 1

    def forward(self, x):
        return x * self.scale


class HadmardAttention(nn.Module):
    def __init__(self, d_in: int, d_in_M: int, n_heads: NotImplementedError,  n_scales: int = 2, d_concat: int = None, print_flag=False):
        """Hadmard attention layer

        :param d_in: number of input features
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        self.print_flag = print_flag
        super().__init__()
        self.d_in = d_in
        self.d_in_M = d_in_M
        if d_concat == None:
            self.d_concat = 1024
        else:
            self.d_concat = d_concat
        self.n_heads = n_heads
        self.n_scales = n_scales

        self.morpho_encoder = Morpho_Encoder(input_dim=self.d_in_M, output_dim=512, bias=True, hidden_dim=2048)
        self.gene_encoder = Gene_Encoder(input_dim=self.d_in, output_dim=512, bias=True)
        
        self.gate = ScalarGate(512)

        self.q = NonNegLinear(512, n_heads, bias=True)
        #self.k = NonNegLinear(self.d_concat, n_heads, bias=False)
        self.k_local = NonNegLinear(512, n_heads, bias=True)
            
        #self.w_ego = NonNegScale(n_heads)
        self.w_local = NonNegScale3(n_heads)

        self.v = NonNegLinear(n_heads, d_in, bias=True)


        self.cosine_similarity = nn.CosineSimilarity(dim=-2)

    def score_intrinsic(self, q_emb, k_emb, activation=None):
        """Score intrinsic factors. No attention to other cells/environment.

        :param q_emb: query scores
        :param k_emb: key scores
        :param activation: activation function
        :return: ego scores
        """
        scores = q_emb * k_emb
        if activation is not None:
            scores = activation(scores)
        return scores

    def score_interactive(self, q_emb, k_emb, adj_list, activation=None):
        """Score interactive factors. Attention to other cells/environment.

        :param q_emb: query scores
        :param k_emb: key scores
        :param adj_list: adjacency list
        :return: interactive scores for short or long range interaction
        """
        q = q_emb[adj_list[1, :], :] # n * g ---v-> kn * d
        k = k_emb[adj_list[0, :], :] # n * g ---u-> kn * d
        scores = q * k # nk * d
        if activation is not None:
            scores = activation(scores)
        nominal_k = scores.shape[0] // q_emb.shape[0]
        if adj_list.shape[0] == 3: # masked for unequal neighbors
            scores.masked_fill_((adj_list[2, :] == 0).reshape([-1, 1]), 0.)

        # reshape
        scores = scores.reshape([q_emb.shape[0], nominal_k, self.n_heads]) # n * k * d 
        scores = scores.transpose(-1, -2)

        # Normalize by the actual number of neighbors
        if adj_list.shape[0] == 3:
            actual_k = adj_list[2, :].reshape(q_emb.shape[0], nominal_k).sum(axis=1)
            scores = scores / actual_k[:, None, None] 
        else:
            scores = scores / nominal_k

        return scores

    def forward(self, adj_list, x, m, masked_x=None, masked_m=None, get_details=False):
        """Forward pass

        :param adj_list: adjacency list for spatial graph
        :param x: input data
        :param masked_x: masked input data, defaults to None (i.e, using x)
        :param regional_adj_lists: list of adjacency list for bipartite graph of cells - regions, defaults to None
        :param regional_xs: list of mean expression of regions, defaults to None
        :param get_details: whether to return details, defaults to False
        :return: reconstructed gene expression
        """

        if masked_x is None:
            masked_x = x

        if masked_m is None:
            masked_m = m

        #emb = self.encoder(torch.concatenate((masked_x, masked_m), axis=-1))
        encoded_m = self.morpho_encoder(masked_m)
        encoded_g = self.gene_encoder(masked_x)
        
        #emb, alpha = self.gate(encoded_g, encoded_m)
        emb, alpha = self.gate(encoded_g, encoded_m)
        #emb = torch.concatenate((encoded_g, encoded_m), axis=-1)
        
        # Get embeddings for all cells
        q_emb = self.q(emb) / emb.shape[1]

        #k_emb = self.k(emb) / emb.shape[1]

        k_local_emb = self.k_local(emb) / emb.shape[1]

        # Get raw attention scores
        #ego_score = self.w_ego(self.score_intrinsic(q_emb, k_emb))
        local_score = self.w_local((self.score_interactive(q_emb, k_local_emb, adj_list)))

        # Normalize attention scores
        sum_local_score = torch.sum(local_score, dim=-1)

        #sum_score = ego_score + sum_local_score
        sum_score = sum_local_score
        normalization_factor = sum_score.sum(axis=-1, keepdim=True) + 1e-9 # n * 1
        sum_attn = sum_score / normalization_factor
        
        res_g = self.v(sum_attn)

        if get_details:
            #ego_attnp = ego_score / normalization_factor
            local_attnp = local_score / normalization_factor[:, :, None]

            #ego_attnm = ego_attnp
            local_attnm = local_attnp.sum(axis=-1)

            return res_g, alpha, local_attnp
        else:
            return res_g, alpha

    
class Steamboat(nn.Module):
    def __init__(self, features: int, morpho_features: int, n_heads: isinstance, n_scales: int = 2):
        """Steamboat model

        :param features: feature names (usuall `adata.var_names` or a column in `adata.var` for gene symbols)
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        super().__init__()

        
        d_in = features
        d_in_M = morpho_features
        self.spatial_gather = HadmardAttention(d_in, d_in_M, n_heads, n_scales)

    def masking(self, x: torch.Tensor, entry_masking_rate: float):
        """Masking the dataset

        :param x: input data
        :param mask_rate: masking rate
        :param masking_method: full matrix or feature-wise masking
        :return: masked data
        """
        out_x = x.clone()
        if entry_masking_rate > 0.:
            random_mask = torch.rand(x.shape, device=x.device) < entry_masking_rate
            out_x.masked_fill_(random_mask, 0.)
        return out_x
    
    def forward(self, adj_list, x, m, masked_x, masked_m, get_details=False):
        return self.spatial_gather(adj_list, x, m, masked_x, masked_m, get_details)

    def fit(self, dataset: SteamboatDataset, 
            entry_masking_rate: float = 0.0,
            device:str = 'cuda', 
            *, 
            opt=None, opt_args=None, 
            loss_fun=None,
            max_epoch: int = 100, stop_eps: float = 1e-4, stop_tol: int = 10, 
            report_per: int = 10, return_loss=False):

        """Create a PyTorch Dataset from a list of adata

        :param dataset: Dataset to be trained on
        :param entry_masking_rate: Rate of masking a random entries, default 0.0
        :param feature_masking_rate: Rate of masking a full feature (can overlap with entry masking), default 0.0
        :param device: Device to be used ("cpu" or "cuda")
        :param local_entropy_penalty: entropy penalty to make the local attention more diverse
        :param opt: Optimizer for fitting
        :param opt_args: Arguments for optimizer (e.g., {'lr': 0.01})
        :param loss_fun: Loss function: Default is MSE (`nn.MSELoss`). 
        You may use MAE `nn.L1Loss`, Huber 'nn.HuberLoss`, SmoothL1 `nn.SmoothL1Loss`, or a customized loss function.
        :param max_epoch: maximum number of epochs
        :param stop_eps: Stopping criterion: minimum change (see also `stop_tol`)
        :param stop_tol: Stopping criterion: number of epochs that don't meet `stop_eps` before stopping
        :param log_dir: Directory to save logs
        :param report_per: report per how many epoch. 0 to only report before termination. negative number to never report.

        :return: self
        """
        self.train()

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()

        if loss_fun is None:
            criterion = nn.MSELoss(reduction='sum')
        else:
            criterion = loss_fun

        if opt_args is None:
            opt_args = {}
        if opt is None:
            optimizer = optim.Adam(parameters, **opt_args)
        else:
            optimizer = opt(parameters, **opt_args)

        #scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #    optimizer,
        #    milestones=[100],
        #    gamma=0.1
        #    )
        
        cnt = 0
        best_loss = np.inf
        for epoch in tqdm(range(max_epoch)):

            alpha_list = []
            avg_loss = 0.
            optimizer.zero_grad()
            for x, m, adj_list in loader:

                # Send everything to required device
                adj_list = adj_list.squeeze(0).to(device)
                x = x.squeeze(0).to(device)
                m = m.squeeze(0).to(device)

                masked_x = self.masking(x, entry_masking_rate)
                masked_m = m

                x_recon, alpha = self.forward(adj_list, x, m, masked_x, masked_m)
                alpha_list.extend(alpha.detach().cpu().tolist())

                loss = criterion(x_recon, x)
                avg_loss += loss.item()

                loss.backward()
                optimizer.step()
            
            #scheduler.step()
            alpha_m = np.mean(alpha_list)
            #if best_loss - avg_loss < stop_eps:
            if False:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                print(f"Stopping criterion met. Final loss =  {avg_loss:.5f}")

                print(f"alpha: {alpha_m:.2f}")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                print(f"Epoch {epoch + 1}: loss =  {avg_loss:.5f}")
                print(f"alpha: {alpha_m:.2f}")
            best_loss = min(best_loss, avg_loss)
        else:
            print(f"Maximum iterations reached. Final Loss:  {avg_loss:.5f}")
            print(f"alpha: {alpha_m:.2f}")
            
        self.eval()
        if return_loss:
            return avg_loss, alpha_m
        else:
            return self