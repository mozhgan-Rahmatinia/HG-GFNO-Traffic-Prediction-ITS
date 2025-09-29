import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import *

class ScalarAdjMixer(nn.Module):
    """
    A learnable mixer for combining a fixed and an adaptive adjacency matrix.
    The mixing weight is a single scalar parameter learned via a sigmoid gate.
    
    A_combined = gate * A_fix + (1 - gate) * A_adapt
    """
    def __init__(self, init_alpha: float = 0.0):
        """
        Args:
            init_alpha (float): Initial value for the mixing parameter logit.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, A_fix: torch.Tensor, A_adapt: torch.Tensor) -> torch.Tensor:
        """
        Performs the weighted sum of the two matrices.

        Args:
            A_fix (torch.Tensor): The fixed, pre-defined adjacency matrix.
            A_adapt (torch.Tensor): The dynamically learned adaptive adjacency matrix.

        Returns:
            torch.Tensor: The combined adjacency matrix.
        """
        gate = torch.sigmoid(self.alpha)          # 0<gate<1
        return gate * A_fix + (1 - gate) * A_adapt




class AVWGCN_Multi(nn.Module):
    """
    Adaptive Vertex-Weighted Graph Convolutional Network for multiple graphs.
    This layer performs Chebyshev graph convolutions where the filter weights are
    dynamically generated for each node based on its embedding.
    """
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, num_graphs, use_graph_gates=False):
        super().__init__()
        self.cheb_k, self.num_graphs, self.use_graph_gates = cheb_k, num_graphs, use_graph_gates
        self.weights_pool_list = nn.ParameterList(
            [nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out)) for _ in range(num_graphs)]
        )
        self.bias_pool_list = nn.ParameterList(
            [nn.Parameter(torch.empty(embed_dim, dim_out)) for _ in range(num_graphs)]
        )
        if use_graph_gates:
            self.alpha_logits = nn.Parameter(torch.zeros(num_graphs))
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights and biases."""
        for W in self.weights_pool_list: nn.init.xavier_uniform_(W)
        for b in self.bias_pool_list:   nn.init.zeros_(b)
        if self.use_graph_gates: nn.init.zeros_(self.alpha_logits)

    def forward(self, x, node_embeddings, fixed_supports=None):
        """
        Args:
            x (torch.Tensor): Input features of shape [Batch, Num_Nodes, Channels].
            node_embeddings (torch.Tensor): Node embeddings of shape [Num_Nodes, embed_dim].
            fixed_supports (List[torch.Tensor], optional): A list of fixed graph adjacency matrices.

        Returns:
            torch.Tensor: The output features after graph convolution.
        """
        S_adapt = _row_softmax(F.relu(node_embeddings @ node_embeddings.T))
        supports = [S_adapt] + ([] if fixed_supports is None else fixed_supports)
        cheb = [_cheb_supports_from_S(S, self.cheb_k) for S in supports]
        outs = []
        for g, sup in enumerate(cheb):
            Xg = torch.einsum("knm,bmc->bknc", sup, x).permute(0,2,1,3)
            Wg = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool_list[g])
            bg = node_embeddings @ self.bias_pool_list[g]
            outs.append(torch.einsum('bnki,nkio->bno', Xg, Wg) + bg)
        if self.use_graph_gates:
            alpha = torch.softmax(self.alpha_logits, 0).view(-1,1,1,1)
            return (alpha * torch.stack(outs)).sum(0)
        return torch.stack(outs).sum(0)






class AGCNLayer(nn.Module):
    """
    A single Adaptive Graph Convolutional Network (AGCN) layer.
    This is a wrapper around AVWGCN_Multi that manages its own node embeddings
    and fixed supports.
    """
    def __init__(self, N, len_input, nb_chev_filter, cheb_k, embed_dim=64,
                 fixed_supports=None, use_graph_gates=False):
        super().__init__()
        num_graphs = 1 + (len(fixed_supports) if fixed_supports else 0)
        self.avwgcn = AVWGCN_Multi(len_input, nb_chev_filter, cheb_k, embed_dim,
                                   num_graphs, use_graph_gates)
        self.proj_to_T = nn.Linear(nb_chev_filter, len_input)
        self.node_embeddings = nn.Parameter(torch.randn(N, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        if fixed_supports:
            self._has_fix = True
            self.fixed_supports = nn.ParameterList([nn.Parameter(S, requires_grad=False)
                                                    for S in fixed_supports])
        else:
            self._has_fix = False
            self.fixed_supports = nn.ParameterList()

    def _fixed(self):
        """Returns the list of fixed support matrices."""
        return [] if not self._has_fix else [p.data for p in self.fixed_supports]

    def forward(self, x, node_embeddings=None):
        """
        Args:
            x (torch.Tensor): Input features.
            node_embeddings (torch.Tensor, optional): Pre-computed node embeddings. 
                                                     If None, uses its own embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (output_features, node_embeddings).
        """
        E = node_embeddings if node_embeddings is not None else self.node_embeddings
        out = self.proj_to_T(self.avwgcn(x, E, self._fixed()))
        return out, E





class DoubleAGCN(nn.Module):
    """
    A block consisting of two stacked AGCN layers with normalization, dropout,
    and a gated residual connection.
    """
    def __init__(self, N, T, Ft, cheb_k, embed_dim=64, dropout=0.1,
                 fixed_supports=None, use_graph_gates=False):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(N, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.agcn1 = AGCNLayer(N, T, Ft, cheb_k, embed_dim, fixed_supports, use_graph_gates)
        self.agcn2 = AGCNLayer(N, T, Ft, cheb_k, embed_dim, fixed_supports, use_graph_gates)
        self.norm1, self.norm2 = nn.LayerNorm(T), nn.LayerNorm(T)
        self.dropout = nn.Dropout(dropout)
        self.res_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Architecture: AGCN -> Norm -> ReLU -> Dropout -> AGCN -> Gated-Residual -> Norm
        """
        res = x
        x, E = self.agcn1(x, self.node_embeddings)
        x = self.dropout(F.relu(self.norm1(x)))
        x, _ = self.agcn2(x, E)
        x = x + torch.sigmoid(self.res_gate) * res
        return self.norm2(x), E



class DataEmbedding_inverted(nn.Module):
    """
    A simple data embedding layer that projects input features to a
    higher-dimensional space (d_model).
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.value_embedding(x))




class GraphFNO1dLayer(nn.Module):
    """
    Implements a 1D Graph Fourier Neural Operator (FNO) layer.
    It performs a global convolution by operating in the spectral domain
    of the graph, defined by the Laplacian eigenvectors.
    """
    def __init__(self, d_model, modes=64, dropout=0.0):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(d_model, modes))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('U', None, persistent=False)
    
    def set_U(self, U):
        """Set the Fourier basis U."""
        self.U = U
    
    def forward(self, x):
        assert self.U is not None, "Fourier basis U must be set before forward pass."
        
        # Project to the spectral domain (Graph Fourier Transform)
        K = min(self.modes, self.U.shape[1]); U = self.U[:, :K]
        x_hat = torch.einsum('nk,bnd->bkd', U, x)

        # Apply learnable filter in the spectral domain
        x_hat = self.dropout(x_hat * self.weights[:, :K].t().unsqueeze(0))

        # Project back to the node domain (Inverse Graph Fourier Transform)
        x_rec = torch.einsum('nk,bkd->bnd', U, x_hat)

        # Return with a residual connection
        return self.out_proj(x) + x_rec




class MVEncoderLayer(nn.Module):
    """
    A single encoder layer, similar to a Transformer, but using GraphFNO1dLayer
    as the attention/mixing mechanism. Follows a Pre-Norm architecture.
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu", modes=64):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GraphFNO1dLayer(d_model, modes, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU() if activation=="gelu" else nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model),
                                 nn.Dropout(dropout))
    
    def set_U(self, U):
        """Pass the Fourier basis to the internal FNO layer."""
        self.attn.set_U(U)
    
    def forward(self, x):
        # Architecture: x + Attention(Norm(x)) -> x + FFN(Norm(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x)), None




class Encoder(nn.Module):
    """
    A stack of MVEncoderLayers.
    """
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers); self.norm = norm_layer
    
    def set_U(self, U):
        """Set the Fourier basis for all FNO layers in the stack."""
        [l.set_U(U) for l in self.attn_layers if hasattr(l, "set_U")]
    
    def forward(self, x):
        for l in self.attn_layers: x, _ = l(x)
        return (self.norm(x) if self.norm else x), None