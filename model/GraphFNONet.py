# -*- coding:utf-8 -*-
import math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from typing import List, Optional, Union
# ============================================================
#                 Utilities
# ============================================================
def _row_softmax(mat: torch.Tensor) -> torch.Tensor:
    return torch.softmax(mat, dim=-1)

def _prep_fixed_adj(A: Union[np.ndarray, torch.Tensor], device) -> torch.Tensor:
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A.astype(np.float32))
    A = torch.clamp(A.to(device=device, dtype=torch.float32), 0.0)
    return _row_softmax(A)

def _cheb_supports_from_S(S: torch.Tensor, K: int) -> torch.Tensor:
    N = S.size(0)
    if K == 1:
        return torch.eye(N, device=S.device, dtype=S.dtype).unsqueeze(0)
    sup = [torch.eye(N, device=S.device, dtype=S.dtype), S]
    for _ in range(2, K):
        sup.append(2 * S @ sup[-1] - sup[-2])
    return torch.stack(sup)

def _make_ring_adj(n: int, device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    A = torch.zeros((n, n), device=device)
    A[idx, (idx - 1) % n] = 1.0
    A[idx, (idx + 1) % n] = 1.0
    return A

def _laplacian_eigvecs(A: torch.Tensor, k: int) -> torch.Tensor:
    A = torch.clamp(0.5 * (A + A.T), 0.0)
    deg = torch.clamp(A.sum(-1), min=1e-6)
    Dm12 = torch.diag(torch.pow(deg, -0.5))
    L = torch.eye(A.size(0), device=A.device) - Dm12 @ A @ Dm12
    _, evecs = torch.linalg.eigh(L)
    return evecs[:, :min(k, A.size(0))].contiguous()

def _to_torch_float(a, device):
    if a is None: return None
    if torch.is_tensor(a): return a.to(device=device, dtype=torch.float32)
    return torch.tensor(a, dtype=torch.float32, device=device)
# ============================================================
#            Scalar mixer  (A_comb = σ(α) A_fix + …)
# ============================================================
class ScalarAdjMixer(nn.Module):
    def __init__(self, init_alpha: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, A_fix: torch.Tensor, A_adapt: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)          # 0<gate<1
        return gate * A_fix + (1 - gate) * A_adapt
# ============================================================
#            Adaptive Chebyshev GCN  
# ============================================================
class AVWGCN_Multi(nn.Module):
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
        for W in self.weights_pool_list: nn.init.xavier_uniform_(W)
        for b in self.bias_pool_list:   nn.init.zeros_(b)
        if self.use_graph_gates: nn.init.zeros_(self.alpha_logits)

    def forward(self, x, node_embeddings, fixed_supports=None):
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
# ============================================================
#                         AGCN layer
# ============================================================
class AGCNLayer(nn.Module):
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
        return [] if not self._has_fix else [p.data for p in self.fixed_supports]

    def forward(self, x, node_embeddings=None):
        E = node_embeddings if node_embeddings is not None else self.node_embeddings
        out = self.proj_to_T(self.avwgcn(x, E, self._fixed()))
        return out, E
# ============================================================
class DoubleAGCN(nn.Module):
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
        res = x
        x, E = self.agcn1(x, self.node_embeddings)
        x = self.dropout(F.relu(self.norm1(x)))
        x, _ = self.agcn2(x, E)
        x = x + torch.sigmoid(self.res_gate) * res
        return self.norm2(x), E
# ============================================================
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.value_embedding(x))
# ---------------- Graph-FNO layer ---------------------------
class GraphFNO1dLayer(nn.Module):
    def __init__(self, d_model, modes=64, dropout=0.0):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(d_model, modes))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('U', None, persistent=False)
    def set_U(self, U): self.U = U
    def forward(self, x):
        assert self.U is not None
        K = min(self.modes, self.U.shape[1]); U = self.U[:, :K]
        x_hat = torch.einsum('nk,bnd->bkd', U, x)
        x_hat = self.dropout(x_hat * self.weights[:, :K].t().unsqueeze(0))
        x_rec = torch.einsum('nk,bkd->bnd', U, x_hat)
        return self.out_proj(x) + x_rec
# ------------------------------------------------------------
class MVEncoderLayer(nn.Module):
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
    def set_U(self, U): self.attn.set_U(U)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x)), None
# ------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers); self.norm = norm_layer
    def set_U(self, U): [l.set_U(U) for l in self.attn_layers if hasattr(l, "set_U")]
    def forward(self, x):
        for l in self.attn_layers: x, _ = l(x)
        return (self.norm(x) if self.norm else x), None
# ============================================================
#                     MGCN Block
# ============================================================
class MGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, len_input, adj_mx=None, embed_dim=64, cheb_k=3,
                 dropout=0.1, use_graph_gates=False, attn_modes=64):
        super().__init__()
        self.device = DEVICE
        self.len_input, self.nb_time_filter = len_input, nb_time_filter
        self.time_strides, self.attn_modes = time_strides, attn_modes
        self.embed_dim, self.cheb_k = embed_dim, cheb_k
        # fixed supports
        self.fixed_supports = []
        if adj_mx is not None:
            if isinstance(adj_mx, (list, tuple)):
                self.fixed_supports = [_prep_fixed_adj(A, DEVICE) for A in adj_mx]
            else:
                self.fixed_supports = [_prep_fixed_adj(adj_mx, DEVICE)]
        # spatial GCN (set later)
        self.spatial = None
        # embedding→512
        self.enc_embedding = DataEmbedding_inverted(len_input, 512, 0.1)
        # scalar mixer
        self.adj_mixer = ScalarAdjMixer(init_alpha=0.0)
        # encoder
        layers = [MVEncoderLayer(512, 2048, 0.1, 'gelu', modes=attn_modes) for _ in range(2)]
        self.encoder = Encoder(layers, norm_layer=nn.LayerNorm(512))
        self.register_buffer("Uk", None, persistent=False)
        self.eig_update_every = 5
        self.register_buffer("step_cnt", torch.tensor(0, dtype=torch.long))
        # projections
        self.projector = nn.Linear(512, len_input)
        self.projector1 = nn.Linear(nb_time_filter, in_channels)
        self.projector2 = nn.Linear(in_channels, nb_time_filter)
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, (1,1), (1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        self.dropout, self.use_graph_gates = dropout, use_graph_gates
        self.to(DEVICE)

    # ---------------- helpers ----------------
    def set_num_nodes(self, N: int):
        self.spatial = DoubleAGCN(N, self.len_input, self.nb_time_filter, self.cheb_k,
                                  self.embed_dim, self.dropout,
                                  fixed_supports=(self.fixed_supports if self.fixed_supports else None),
                                  use_graph_gates=self.use_graph_gates).to(self.device)

    # ---------------- main forward ----------------
    def forward(self, x):
        if self.spatial is None:
            N = x.size(1) if x.dim()==4 else x.size(2)
            self.set_num_nodes(N)
        self.step_cnt += 1
        x0 = x if x.dim()==4 else x.permute(0,2,1).unsqueeze(2)  # [B,N,1,T]
        B,N,_,T = x0.shape
        # 1) spatial
        xs, E_nodes = self.spatial(x0.squeeze(2))
        
        # E_det = E_nodes.detach()
        E_det = E_nodes
        A_adapt = _row_softmax(F.relu(E_det @ E_det.T))          # without gradient 
        A_fix = self.fixed_supports[0] if self.fixed_supports else _make_ring_adj(N, self.device)
        A_comb = self.adj_mixer(A_fix, A_adapt)
        # update U occasionally
        if (self.Uk is None) or (self.training and self.step_cnt.item() % self.eig_update_every == 0):
            with torch.no_grad():
                self.Uk = _laplacian_eigvecs(A_comb, k=min(self.attn_modes, N))
        self.encoder.set_U(self.Uk)
        # 2) encoder
        enc_out, _ = self.encoder(self.enc_embedding(xs))        # [B,N,512]
        # 3) project back to T
        m_out = self.projector(enc_out).permute(0,2,1)[:, :, :N].permute(0,2,1)
        # 4) fuse with residual
        m_out = self.projector2(m_out.unsqueeze(3))              # [B,N,T, Ft]
        x_res = self.residual_conv(x0.permute(0,2,1,3)).permute(0,2,3,1)
        if x_res.size(2) != m_out.size(2): m_out = m_out[:, :, :x_res.size(2), :]
        fused = self.ln(F.relu(x_res + m_out))
        out = self.projector1(fused).permute(0,1,3,2)            # [B,N,C,T']
        return out
# ============================================================
#               Submodule & factory helpers
# ============================================================
class MGCN_submodule(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, num_for_predict, len_input, adj_mx,
                 attn_modes=64):
        super().__init__()
        self.Block = MGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                time_strides, len_input, adj_mx, attn_modes=attn_modes)
        self.projector3 = nn.Linear(len_input // time_strides if time_strides>1 else len_input,
                                    num_for_predict)
        self.to(DEVICE)
        if isinstance(adj_mx, np.ndarray): self.Block.set_num_nodes(adj_mx.shape[0])

    def forward(self, x):
        x = self.Block(x).squeeze(2)                     # [B,N,T']
        return self.projector3(x).permute(0,2,1)         # [B,pred_len,N]

def make_model(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, adj_mx, num_for_predict, len_input,
               attn_modes=64):
    model = MGCN_submodule(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                           time_strides, num_for_predict, len_input, adj_mx,
                           attn_modes)
    for p in model.parameters():
        nn.init.xavier_uniform_(p) if p.dim()>1 else nn.init.uniform_(p, -0.1, 0.1)
    return model
# ============================================================
#                   Sanity check
# ============================================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, C, T = 2, 16, 1, 12
    A = np.random.rand(N, N).astype(np.float32); np.fill_diagonal(A, 0.0)
    model = make_model(DEVICE, C, 3, 32, 32, 1, A, 6, T, attn_modes=min(64, N//2)).to(DEVICE)
    x = torch.randn(B, N, C, T, device=DEVICE)
    y = model(x)
    print("Output shape:", y.shape)    # torch.Size([2, 6, 16])



