import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming utils.py and layers.py are in the same directory
from .utils import *
from .layers import *

class GFNO_block(nn.Module):
    """
    A core block that combines a spatial Adaptive Graph Convolutional Network (AGCN)
    with a spectral Graph Fourier Neural Operator (GFNO) based encoder. This hybrid
    approach captures both local and global spatio-temporal dependencies.

    The block operates as follows:
    1. Processes input through a spatial GCN module (DoubleAGCN).
    2. Constructs a hybrid adjacency matrix from fixed and adaptive components.
    3. Computes Laplacian eigenvectors (Fourier basis) from the hybrid matrix.
    4. Uses an FNO-based encoder to perform global convolution in the spectral domain.
    5. Fuses the output with a residual connection from the original input.
    """
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, len_input, adj_mx=None, embed_dim=64, cheb_k=3,
                 dropout=0.1, use_graph_gates=False, attn_modes=64):
        super().__init__()
        self.device = DEVICE
        self.len_input = len_input
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.attn_modes = attn_modes
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.dropout = dropout
        self.use_graph_gates = use_graph_gates

        # --- Pre-process and store fixed adjacency matrices ---
        self.fixed_supports = []
        if adj_mx is not None:
            adj_list = adj_mx if isinstance(adj_mx, (list, tuple)) else [adj_mx]
            self.fixed_supports = [prep_fixed_adj(A, DEVICE) for A in adj_list]

        # --- Model Components ---
        # Spatial GCN module (lazily initialized)
        self.spatial = None

        # Temporal feature embedding into a higher dimension (d_model=512)
        self.enc_embedding = DataEmbedding_inverted(len_input, 512, 0.1)

        # Learnable mixer for combining fixed and adaptive graphs
        self.adj_mixer = ScalarAdjMixer(init_alpha=0.0)

        # Spectral encoder composed of Graph FNO layers
        encoder_layers = [MVEncoderLayer(512, 2048, 0.1, 'gelu', modes=attn_modes) for _ in range(2)]
        self.encoder = Encoder(encoder_layers, norm_layer=nn.LayerNorm(512))

        # --- Buffers for Fourier Basis ---
        # Buffer for Laplacian eigenvectors (U_k)
        self.register_buffer("Uk", None, persistent=False)
        self.eig_update_every = 5  # How often to recompute eigenvectors during training
        self.register_buffer("step_cnt", torch.tensor(0, dtype=torch.long))

        # --- Projection and Fusion Layers ---
        self.projector = nn.Linear(512, len_input) # Project from d_model back to time_len
        self.projector1 = nn.Linear(nb_time_filter, in_channels)
        self.projector2 = nn.Linear(in_channels, nb_time_filter)
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

        self.to(DEVICE)

    def set_num_nodes(self, N: int):
        """
        Lazily initializes the spatial GCN module once the number of nodes is known.
        """
        self.spatial = DoubleAGCN(
            N, self.len_input, self.nb_time_filter, self.cheb_k,
            self.embed_dim, self.dropout,
            fixed_supports=(self.fixed_supports if self.fixed_supports else None),
            use_graph_gates=self.use_graph_gates
        ).to(self.device)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Num_Nodes, In_Channels, Time_Len].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, Num_Nodes, In_Channels, Out_Time_Len].
        """
        # --- Lazy Initialization ---
        if self.spatial is None:
            N = x.size(1) if x.dim() == 4 else x.size(2)
            self.set_num_nodes(N)
        self.step_cnt += 1

        # Reshape input to [B, N, C, T] -> [B, N, T] for spatial processing
        x0 = x if x.dim() == 4 else x.permute(0, 2, 1).unsqueeze(2)
        B, N, _, T = x0.shape

        # 1. Spatial Processing: Apply DoubleAGCN to capture local features
        xs, E_nodes = self.spatial(x0.squeeze(2))

        # 2. Graph Construction: Build the combined adjacency matrix for the FNO encoder
        A_adapt = _row_softmax(F.relu(E_nodes @ E_nodes.T))
        A_fix = self.fixed_supports[0] if self.fixed_supports else make_ring_adj(N, self.device)
        A_comb = self.adj_mixer(A_fix, A_adapt)

        # 3. Fourier Basis Update: Periodically recompute Laplacian eigenvectors
        if self.Uk is None or (self.training and self.step_cnt.item() % self.eig_update_every == 0):
            with torch.no_grad():
                self.Uk = _laplacian_eigvecs(A_comb, k=min(self.attn_modes, N))
        self.encoder.set_U(self.Uk)

        # 4. Spectral Processing: Apply the FNO-based encoder for global mixing
        # Embed -> Encoder -> Project
        enc_in = self.enc_embedding(xs)
        enc_out, _ = self.encoder(enc_in)
        m_out = self.projector(enc_out)

        # 5. Fusion: Combine spectral output with a residual connection
        m_out = self.projector2(m_out.unsqueeze(3)) # [B, N, T, Ft]
        x_res = self.residual_conv(x0.permute(0, 2, 1, 3)).permute(0, 2, 3, 1)

        # Align temporal dimension if strides changed it
        if x_res.size(2) != m_out.size(2):
            m_out = m_out[:, :, :x_res.size(2), :]

        fused = self.ln(F.relu(x_res + m_out))
        
        # Final projection to output shape
        out = self.projector1(fused).permute(0, 1, 3, 2)
        return out, E_nodes




class GFNO_submodule(nn.Module):
    """
    A wrapper module for the GFNO_block that adds a final linear layer
    to project the time series output to the desired prediction length.
    """
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, num_for_predict, len_input, adj_mx,
                 attn_modes=64):
        super().__init__()
        self.Block = GFNO_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                time_strides, len_input, adj_mx, attn_modes=attn_modes)
        
        # Final projection from output time dimension to prediction length
        out_len = len_input // time_strides if time_strides > 1 else len_input
        self.final_proj = nn.Linear(out_len, num_for_predict)
        
        self.to(DEVICE)
        # Initialize the spatial GCN layer inside the block if adj_mx is provided
        if isinstance(adj_mx, np.ndarray):
            self.Block.set_num_nodes(adj_mx.shape[0])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Num_Nodes, In_Channels, Time_Len].

        Returns:
            torch.Tensor: Prediction tensor of shape [Batch, Pred_Len, Num_Nodes].
        """
        # Pass through the main processing block
        x, node_embeds = self.Block(x)  
        x = x.squeeze(2)  # Shape: [B, N, T']
 
        # Project to final prediction length
        out = self.final_proj(x)      # Shape: [B, N, pred_len]
        
        return out.permute(0, 2, 1), node_embeds   # Shape: [B, pred_len, N]



def make_model(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, adj_mx, num_for_predict, len_input,
               attn_modes=64):
    """
    Factory function to create and initialize the GFNO model.

    Returns:
        nn.Module: The initialized GFNO model.
    """
    model = GFNO_submodule(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                           time_strides, num_for_predict, len_input, adj_mx,
                           attn_modes)
    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p, -0.1, 0.1)
    return model
