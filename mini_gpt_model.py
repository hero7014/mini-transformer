"""从零组合的教学版 GPT：手写 Attention 和 FFN 的核心计算。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        def project(weight):
            return (x @ weight).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = project(self.W_Q), project(self.W_K), project(self.W_V)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        heads = weights @ v
        concat = heads.transpose(1, 2).contiguous().reshape(B, T, D)
        return concat @ self.W_O


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return F.gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x):
        h = x + self.attn(self.ln1(x))
        return h + self.ffn(self.ln2(h))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 hidden_dim: int, num_layers: int, context_length: int):
        super().__init__()
        self.context_length = context_length
        self.token_table = nn.Parameter(torch.randn(vocab_size, d_model) * 0.1)
        self.position_table = nn.Parameter(torch.randn(context_length, d_model) * 0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.W_LM = nn.Parameter(torch.randn(d_model, vocab_size) * 0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        if T > self.context_length:
            raise ValueError(f"序列长度 {T} 超过 context_length={self.context_length}")
        token_vectors = self.token_table[ids]
        position_ids = torch.arange(T, device=ids.device)
        x = token_vectors + self.position_table[position_ids].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x) @ self.W_LM  # (B, T, vocab_size)
