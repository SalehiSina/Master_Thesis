import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .dataset import SteamboatDataset
import os


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=1024, hidden_dim2=1024):
        super(Encoder, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        """Nonegative linear layer

        :param d_in: number of input features
        :param d_out: number of output features
        :param bias: umimplemented
        :raises NotImplementedError: when bias is True
        """
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        """transform weight matrix to be non-negative 

        :return: transformed weight matrix
        """
        return self.elu(self._weight) + 1

    def forward(self, x):
        return x @ self.weight.T

    
class NonNegBias(nn.Module):
    def __init__(self, d) -> None:
        """Non-negative bias layer (i.e., add a non-negative vector to the output)

        :param d: number of input/output features
        """
        super().__init__()
        self._bias = torch.nn.Parameter(torch.zeros(1, d))
        self.elu = nn.ELU()

    @property
    def bias(self):
        """Transform bias to be non-negative

        :return: non-negative bias
        """
        return self.elu(self._bias) + 1

    def forward(self, x):
        return x + self.bias
    
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
    def __init__(self, d_in: int, d_in_M: int, n_heads: int, model_type: int=0,  n_scales: int = 2, d_out: int = None, print_flag=False):
        """Hadmard attention layer

        :param d_in: number of input features
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        :param d_out: _description_, defaults to None (meaning d_out = d_in)
        """
        self.print_flag = print_flag
        super().__init__()
        if d_out == None:
            self.d_out = d_in
        else:
            self.d_out = d_out
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.model_type = model_type

        self.encoder_m = Encoder(input_dim = d_in_M, output_dim=self.d_out)
        self.encoder_g = Encoder(input_dim = d_in, output_dim=self.d_out)
        
        self.q_m = NonNegLinear(self.d_out, n_heads, bias=False)
        self.q_g = NonNegLinear(self.d_out, n_heads, bias=False)
        
        self.k_m = NonNegLinear(self.d_out, n_heads, bias=False)
        self.k_g = NonNegLinear(self.d_out, n_heads, bias=False)
        
        self.k_local_m = NonNegLinear(self.d_out, n_heads, bias=False)
        self.k_local_g = NonNegLinear(self.d_out, n_heads, bias=False)
            
        self.w_ego_m = NonNegScale(n_heads)
        self.w_ego_g = NonNegScale(n_heads)
        self.w_ego_mg = NonNegScale(n_heads)
        self.w_ego_gm = NonNegScale(n_heads)

        self.w_local_mm = NonNegScale3(n_heads)
        self.w_local_gg = NonNegScale3(n_heads)
        self.w_local_mg = NonNegScale3(n_heads)
        self.w_local_gm = NonNegScale3(n_heads)

        self.tanh = nn.Tanh() # for clamping of the values

        #self.v_m = NonNegLinear(n_heads*2, d_in_M, bias=False)
        self.v_g = NonNegLinear(n_heads*2, d_in, bias=False)


        #self.q_emb_m = None
        #self.q_emb_g = None

        #self.k_emb_m = None
        #self.k_emb_g = None
        
        #self.k_local_emb_m = None
        #self.k_local_emb_g = None

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
            actual_k = adj_list[2, :].reshape(q_emb.shape[0], nominal_k).sum(axis=1) # TODO: memorize this
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
        model_type = self.model_type

        if masked_x is None:
            masked_x = x

        if masked_m is None:
            masked_m = m

        m_emb = self.encoder_m(masked_m)
        g_emb = self.encoder_g(masked_x)

        # Get embeddings for all cells
        q_emb_m = self.q_m(m_emb) / m_emb.shape[1]
        q_emb_g = self.q_g(g_emb) / g_emb.shape[1]

        k_emb_m = self.k_m(m_emb) / m_emb.shape[1]
        k_emb_g = self.k_g(g_emb) / g_emb.shape[1]

        k_local_emb_m = self.k_local_m(m_emb) / m_emb.shape[1]
        k_local_emb_g = self.k_local_g(g_emb) / g_emb.shape[1]

        # Get raw attention scores
        ego_score_m = self.w_ego_m(self.score_intrinsic(q_emb_m, k_emb_m))
        ego_score_g = self.w_ego_g(self.score_intrinsic(q_emb_g, k_emb_g))
        ego_score_mg = self.w_ego_m(self.score_intrinsic(q_emb_m, k_emb_g))
        ego_score_gm = self.w_ego_g(self.score_intrinsic(q_emb_g, k_emb_m))

        local_score_mm = self.w_local_mm((self.score_interactive(q_emb_m, k_local_emb_m, adj_list)))
        local_score_mg = self.w_local_mg((self.score_interactive(q_emb_m, k_local_emb_g, adj_list)))
        local_score_gm = self.w_local_gm((self.score_interactive(q_emb_g, k_local_emb_m, adj_list)))
        local_score_gg = self.w_local_gg((self.score_interactive(q_emb_g, k_local_emb_g, adj_list)))

        # Normalize attention scores
        sum_local_score_mm = torch.sum(local_score_mm, dim=-1)
        sum_local_score_mg = torch.sum(local_score_mg, dim=-1)
        sum_local_score_gm = torch.sum(local_score_gm, dim=-1)
        sum_local_score_gg = torch.sum(local_score_gg, dim=-1)

        local_score_zero = torch.concatenate((torch.zeros_like(local_score_mm), torch.zeros_like(local_score_gg)), axis=-1)
        sum_local_score_zero = torch.concatenate((torch.zeros_like(sum_local_score_mm), torch.zeros_like(sum_local_score_gg)), axis=-1)
        sum_ego_score_zero = torch.concatenate((torch.zeros_like(ego_score_m), torch.zeros_like(ego_score_g)), axis=-1)
        
        # Dynamic Part
        if model_type == 0:
            
            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = torch.concatenate((local_score_mm + local_score_mg, local_score_gm + local_score_gg), axis=-2)
            
            sum_local_score = torch.concatenate((sum_local_score_mm + sum_local_score_mg, sum_local_score_gm + sum_local_score_gg), axis=-1)
            sum_ego_score = torch.concatenate((ego_score_m + ego_score_mg, ego_score_g + ego_score_gm) , axis=-1)

        elif model_type == 1:
            
            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = torch.concatenate((local_score_mm + local_score_mg, local_score_gm + local_score_gg), axis=-2)
            
            sum_local_score = torch.concatenate((sum_local_score_mm + sum_local_score_mg, sum_local_score_gm + sum_local_score_gg), axis=-1)
            sum_ego_score = sum_ego_score_zero

        elif model_type == 2:

            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = local_score_zero
            
            sum_local_score = sum_local_score_zero
            sum_ego_score = torch.concatenate((ego_score_m + ego_score_mg, ego_score_g + ego_score_gm) , axis=-1)

        elif model_type == 3:

            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = torch.concatenate((local_score_mm, local_score_gm), axis=-2)
            
            sum_local_score = torch.concatenate((sum_local_score_mm, sum_local_score_gm), axis=-1)
            sum_ego_score = torch.concatenate((ego_score_mg, ego_score_g) , axis=-1)

        elif model_type == 4:

            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = torch.concatenate((local_score_mg, local_score_gg), axis=-2)
            
            sum_local_score = torch.concatenate((sum_local_score_mg, sum_local_score_gg), axis=-1)
            sum_ego_score = torch.concatenate((ego_score_m, ego_score_gm) , axis=-1)

        elif model_type == 5:

            if not self.print_flag:
                print("model_type: ", model_type)
                self.print_flag = True
            
            local_score = torch.concatenate((local_score_mg, local_score_gm), axis=-2)
            
            sum_local_score = torch.concatenate((sum_local_score_mg, sum_local_score_gm), axis=-1)
            sum_ego_score = torch.concatenate((ego_score_mg, ego_score_gm) , axis=-1)
        
        else:
            if not self.print_flag:
                print("model_type is not recognised. current model_type: ", 0)
                self.print_flag = True
            local_score = local_score_mm + local_score_mg + local_score_gm + local_score_gg
            
            sum_local_score = sum_local_score_mm + sum_local_score_mg + sum_local_score_gm + sum_local_score_gg
            sum_ego_score = ego_score_m + ego_score_g + ego_score_gm + ego_score_mg


        sum_score = sum_ego_score + sum_local_score
        normalization_factor = sum_score.sum(axis=-1, keepdim=True) + 1e-9 # n * 1
        sum_attn = sum_score / normalization_factor
        
        #res_m = self.v_m(sum_attn)
        res_g = self.v_g(sum_attn)
        
        #self.q_emb_m = q_emb_m
        #self.q_emb_g = q_emb_g

        #self.k_emb_m = k_emb_m
        #self.k_emb_g = k_emb_g
        
        #self.k_local_emb_m = k_local_emb_m
        #self.k_local_emb_g = k_local_emb_g
        
        ego_score = sum_ego_score

        if get_details:
            ego_attnp = ego_score / normalization_factor
            local_attnp = local_score / normalization_factor[:, :, None]

            ego_attnm = ego_attnp
            local_attnm = local_attnp.sum(axis=-1)

            #return res_m, res_g, {
            return res_g, {
                'attn': sum_attn,
                'attnp': (ego_attnp, local_attnp),
                'attnm': (ego_attnm, local_attnm)
                }
        else:
            #return res_m, res_g
            return res_g

    
class Steamboat(nn.Module):
    def __init__(self, features: int, morpho_features: int, n_heads: int, model_type: int, n_scales: int = 2):
        """Steamboat model

        :param features: feature names (usuall `adata.var_names` or a column in `adata.var` for gene symbols)
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        super().__init__()

        
        d_in = features
        d_in_M = morpho_features
        self.spatial_gather = HadmardAttention(d_in, d_in_M, n_heads, model_type, n_scales)

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
            sched = None, max_lr = None,
            max_epoch: int = 100, stop_eps: float = 1e-4, stop_tol: int = 10, 
            report_per: int = 10):

        print("This is the new version")
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

        if sched != None:
            scheduler = sched(optimizer,
                              max_lr=max_lr,        # highest LR in the cycle
                              total_steps=10000,  # must match max_epoch
                              pct_start=0.1,      # % of cycle to warm up
                              anneal_strategy='cos'
                              )

        cnt = 0
        best_loss = np.inf
        for epoch in range(max_epoch):
            #avg_gene_loss = 0.
            #avg_morpho_loss = 0.
            avg_loss = 0.
            optimizer.zero_grad()
            for x, m, adj_list in loader:
                # Send everything to required device
                adj_list = adj_list.squeeze(0).to(device)
                x = x.squeeze(0).to(device)
                m = m.squeeze(0).to(device)

                masked_x = self.masking(x, entry_masking_rate)
                #masked_m = self.masking(m, entry_masking_rate)
                masked_m = m

                x_recon = self.forward(adj_list, x, m, masked_x, masked_m)

                #gene_loss = criterion(x_recon, x)
                #morpho_loss = criterion(m_recon, m)
                #loss = gene_loss + morpho_loss

                loss = criterion(x_recon, x)
                avg_loss += loss.item()
                #avg_gene_loss += gene_loss.item()
                #avg_morpho_loss += morpho_loss.item()

                loss.backward()
                optimizer.step()

                if sched != None: 
                    scheduler.step() 

            if best_loss - avg_loss < stop_eps:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                #print(f"Epoch {epoch + 1}: gene_loss =  {avg_gene_loss:.5f}, morpho_loss =  {avg_morpho_loss:.5f}")
                print(f"Stopping criterion met. Final loss =  {avg_loss:.5f}")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                #print(f"Epoch {epoch + 1}: gene_loss =  {avg_gene_loss:.5f}, morpho_loss =  {avg_morpho_loss:.5f}")
                print(f"Epoch {epoch + 1}: loss =  {avg_loss:.5f}")
            best_loss = min(best_loss, avg_loss)
        else:
            #print(f"Maximum iterations reached. Final Loss: gene_loss =  {avg_gene_loss:.5f}, morpho_loss =  {avg_morpho_loss:.5f}")
            print(f"Maximum iterations reached. Final Loss:  {avg_loss:.5f}")
            
        self.eval()
        return self