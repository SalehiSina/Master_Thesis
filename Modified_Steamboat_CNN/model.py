import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .dataset import M_SteamboatDataset
import os

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=1024):
        super(Encoder, self).__init__()

        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1, bias=False),
            nn.Linear(hidden_dim1, output_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class imgEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, base_channel=16):
        super(imgEncoder, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, base_channel, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(base_channel),
            nn.Conv2d(base_channel, base_channel*2, kernel_size=3, padding='same', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(base_channel*2),
        )
        self.fc = nn.Linear(base_channel*2*input_dim[1]*input_dim[2], output_dim, bias=False)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


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
    def __init__(self, d_in: int, d_in_image: tuple, d_in_M: int, n_heads: NotImplementedError, d_concat: int = None, print_flag=False):
        """Hadmard attention layer

        :param d_in: number of input features
        :param d_in_image: dimensions of the input image
        :param d_in_M: dimensions of morpho features
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        self.print_flag = print_flag
        super().__init__()
        self.d_in = d_in
        self.d_in_M = d_in_M
        if d_concat == None:
            self.d_concat = d_in + d_in_M
        else:
            self.d_concat = d_concat
        self.n_heads = n_heads
        self.d_in_image = d_in_image

        self.encoder = Encoder(input_dim=self.d_in_M, output_dim=self.d_in)
        self.img_encoder = imgEncoder(input_dim=self.d_in_image, output_dim=self.d_in)

        self.q_gene = NonNegLinear(self.d_in, n_heads, bias=False)
        self.q_image = NonNegLinear(self.d_in, n_heads, bias=False)

        self.k_gene = NonNegLinear(self.d_in, n_heads, bias=False)
        self.k_morpho = NonNegLinear(self.d_in, n_heads, bias=False)
            
        
        self.w_local_gene = NonNegScale3(n_heads)
        self.w_local_morpho = NonNegScale3(n_heads)

        self.v = NonNegLinear(n_heads*2, d_in, bias=False)


        self.cosine_similarity = nn.CosineSimilarity(dim=-2)


    def score_interactive(self, q_emb, k_emb, adj, activation=None):
        """
        q_emb: the cell embedding of the batch. size: (batch_size, n_head)
        k_emb: all cells embeddings. size: (N, n_head)
        adj: adjacency matrix of the 8 nearest neighbors to each cell. size: (batch_size, 8)
        """
        # Gather neighbor embeddings for each cell in the batch
        # adj: (batch_size, 8) → expand to (batch_size, 8, n_head) for gathering
        neighbor_idx = adj.long()                                         # (batch_size, 8)
        
        neighbor_emb = k_emb[neighbor_idx]                                # (batch_size, 8, n_head)

        # Expand q_emb to broadcast over the 8 neighbors
        q_expanded = q_emb.unsqueeze(1)                                   # (batch_size, 1, n_head)

        # Element-wise product of each cell's features with each neighbor's features
        interaction = q_expanded * neighbor_emb                           # (batch_size, 8, n_head)

        if activation is not None:
            interaction = activation(interaction)

        return interaction.permute(0, 2, 1)                                # (batch_size, n_head, 8)


    def forward(self, adj, global_x, x, global_m, image, global_masked_x=None, masked_x=None, get_details=False):
        """Forward pass

        :param adj: adjacency matrix for spatial graph
        :param global_x: global input data (all cells)
        :param x: local input data (batch cell)
        :param global_m: global morpho features
        :param image: input image for cells in the batch
        :param global_masked_x: global masked input data, defaults to None (i.e, using global_x)
        :param masked_x: masked input data, defaults to None (i.e, using x)
        :param get_details: whether to return details, defaults to False
        :return: reconstructed gene expression
        """
        if global_masked_x is None:
            global_masked_x = global_x

        if masked_x is None:
            masked_x = x


        encoded_m = self.encoder(global_m)
        emb_img = self.img_encoder(image)
        
        q_gene = self.q_gene(masked_x) / masked_x.shape[1]
        q_image = self.q_image(emb_img)

        k_gene = self.k_gene(global_masked_x) / masked_x.shape[1]
        k_morpho = self.k_morpho(encoded_m)

        local_score_gene = self.w_local_gene((self.score_interactive(q_gene, k_gene, adj)))
        local_score_morpho = self.w_local_morpho((self.score_interactive(q_image, k_morpho, adj)))

        # Normalize attention scores
        sum_local_score_gene = torch.sum(local_score_gene, dim=-1)
        sum_local_score_morpho = torch.sum(local_score_morpho, dim=-1)

        sum_score = torch.cat([sum_local_score_gene, sum_local_score_morpho], axis=-1)
        normalization_factor = sum_score.sum(axis=-1, keepdim=True) + 1e-9 # n * 1
        sum_attn = sum_score / normalization_factor
        
        res_g = self.v(sum_attn)

        if get_details:
            #ego_attnp = ego_score / normalization_factor
            #local_attnp = local_score / normalization_factor[:, :, None]

            #ego_attnm = ego_attnp
            #local_attnm = local_attnp.sum(axis=-1)

            return res_g, {
                'attn': sum_attn,
                #'attnp': local_attnp,
                #'attnm': local_attnm
                }
        else:
            return res_g

    
class Steamboat(nn.Module):
    def __init__(self, features: int, image_size: tuple, morpho_features: int, n_heads: isinstance):
        """Steamboat model

        :param features: feature names (usuall `adata.var_names` or a column in `adata.var` for gene symbols)
        :param image_size: size of the input image
        :param morpho_features: number of morphological features
        :param n_heads: number of heads
        :param n_scales: number of scales (default 2, i.e., ego and local; 3 will add global)
        """
        super().__init__()

        
        d_in = features
        d_in_M = morpho_features
        self.spatial_gather = HadmardAttention(d_in, image_size, d_in_M, n_heads)

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
    
    def forward(self, adj, global_x, x, global_m, image, global_masked_x, masked_x, get_details):
        return self.spatial_gather(adj, global_x, x, global_m, image, global_masked_x, masked_x, get_details)

    def fit(self, dataloader: DataLoader, 
            entry_masking_rate: float = 0.0,
            device:str = 'cuda',
            opt=torch.optim.Adam, opt_args=dict(lr=0.01), 
            loss_fun=None,
            max_epoch: int = 100, stop_eps: float = 1e-4, stop_tol: int = 10, 
            report_per: int = 10, return_loss=False):

        """Train the model
        :param dataloader: training dataloader
        :param entry_masking_rate: masking rate for input data, defaults to 0.
        :param device: device to train on, defaults to 'cuda'
        :param opt: optimizer, defaults to torch.optim.Adam
        :param opt_args: arguments for the optimizer, defaults to dict(lr=0.01)
        :param loss_fun: loss function, defaults to None (i.e., MSELoss)
        :param max_epoch: maximum number of epochs, defaults to 100

        :return: self
        """
        self.train()

        parameters = self.parameters()

        if loss_fun is None:
            criterion = nn.MSELoss(reduction='sum')
        else:
            criterion = loss_fun

        optimizer = opt(parameters, **opt_args)

        cnt = 0
        best_loss = np.inf
        for epoch in range(max_epoch):

            avg_loss = 0.

            for sample in dataloader:

                # Send everything to required device
                adj = sample['adj'].to(device)
                x = sample['X_local'].to(device)
                image = sample['image'].to(device)
                global_x = sample['X_global'][0].to(device)
                global_m = sample['M_global'][0].to(device)

                masked_x = self.masking(x, entry_masking_rate)
                global_masked_x = self.masking(global_x, entry_masking_rate)

                x_recon = self.forward(adj, global_x, x, global_m, image, global_masked_x, masked_x, get_details=False)

                loss = criterion(x_recon, x)
                avg_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss /= (len(dataloader.dataset)*dataloader.dataset[0]['X_global'].shape[1])

            if best_loss - avg_loss < stop_eps:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                print(f"Stopping criterion met. Final loss =  {avg_loss:.5f}")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                print(f"Epoch {epoch + 1}: loss =  {avg_loss:.5f}")
            best_loss = min(best_loss, avg_loss)
        else:
            print(f"Maximum iterations reached. Final Loss:  {avg_loss:.5f}")
            
        self.eval()
        if return_loss:
            return avg_loss
        else:
            return self