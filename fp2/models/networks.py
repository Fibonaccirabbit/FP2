"""Networks of FP2: the DiT velocity network, the DiT text classifier and the condition MLP.

The DiT is adapted from CleanDiffuser (https://github.com/CleanDiffuserTeam/CleanDiffuser). Module names are
kept so that checkpoints stay loadable.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Embedding of the flow time t (b,) -> (b, dim)."""

    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim, self.max_positions = dim, max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = (1 / self.max_positions) ** (freqs / (self.dim // 2 - 1))
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


def sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Embedding of the time step index within a segment (horizon,) -> (horizon, dim)."""
    half_dim = dim // 2
    freqs = torch.exp(torch.arange(half_dim, device=positions.device) * -(math.log(10000) / (half_dim - 1)))
    emb = positions[:, None] * freqs[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(approximate="tanh"), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x, emb, key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class FinalLayer1d(nn.Module):

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, emb):
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT1d(nn.Module):
    """f(x_t, t, c): (b, horizon, in_dim) -> (b, horizon, in_dim); c is a (b, emb_dim) condition embedding."""

    def __init__(self, in_dim: int, emb_dim: int, d_model: int = 384, n_heads: int = 6, depth: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.map_noise = PositionalEmbedding(emb_dim)
        self.x_proj = nn.Linear(in_dim, d_model)
        self.map_emb = nn.Sequential(nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())
        self.d_model = d_model
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def xavier(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(xavier)
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)
        for block in self.blocks:  # adaLN-Zero
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        self.zero_final_layer()

    def zero_final_layer(self):
        for layer in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, condition=None, attention_mask: Optional[torch.Tensor] = None):
        """attention_mask: (b, horizon), False for padded time steps."""
        x = self.x_proj(x) + sinusoidal_embedding(torch.arange(x.shape[1], device=x.device), self.d_model)[None]
        emb = self.map_noise(t)
        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        for block in self.blocks:
            x = block(x, emb, key_padding_mask)
        return self.final_layer(x, emb)


class ClassifierDiT(DiT1d):
    """Text classifier network: DiT features of (x_t, t), averaged over time, -> text embedding (b, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, d_model: int = 192, n_heads: int = 6, depth: int = 2):
        super().__init__(in_dim, emb_dim, d_model, n_heads, depth)
        self.final_layer = FinalLayer1d(d_model, d_model // 2)
        self.zero_final_layer()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, out_dim))

    def forward(self, x, t, condition=None):
        return self.proj(super().forward(x, t, condition).mean(1))


class Mlp(nn.Module):

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, act: nn.Module):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.mlp = nn.Sequential(*[nn.Sequential(nn.Linear(dims[i], dims[i + 1]), act) for i in range(len(hidden_dims))],
                                 nn.Linear(dims[-1], out_dim))

    def forward(self, x):
        return self.mlp(x)


class ConditionMLP(nn.Module):
    """MLP embedding of the numerical condition c = [preference, return-to-go], (b, in_dim) -> (b, out_dim).

    During training each sample's condition embedding is zeroed with probability ``dropout`` (CFG).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], act: nn.Module, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout
        self.mlp = Mlp(in_dim, hidden_dims, out_dim, act)

    def forward(self, condition):
        out = self.mlp(condition)
        if self.training:
            out = out * (torch.rand(condition.shape[0], 1, device=condition.device) > self.dropout).float()
        return out
