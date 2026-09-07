"""Shared layers: RMSNorm, RoPE, SwiGLU, causal attention, TTT-Linear."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.shape[-2]
        cos = self.cos[:length].to(q.device, q.dtype)[None, None]
        sin = self.sin[:length].to(q.device, q.dtype)[None, None]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


def _ln_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta


def _ln_fused_l2_bwd(
    x: torch.Tensor, target: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    dim = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_x_hat = (y - target) * gamma
    return (
        (1.0 / dim)
        * (dim * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True) - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True))
        / std
    )


class TTTLinearMixer(nn.Module):
    """TTT-Linear: hidden state is a per-head linear model updated on each mini-batch.

    Dual-form update matches the official ttt-lm-pytorch layer, without KV cache.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.mini_batch = cfg.ttt_mini_batch
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.heads, 1, self.head_dim))
        self.ttt_norm_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.ttt_base_lr = cfg.ttt_base_lr
        self.learnable_ttt_lr = nn.Linear(cfg.hidden_size, self.heads, bias=True)
        token_idx = 1.0 / torch.arange(1, self.mini_batch + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(self.mini_batch))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        pad = (self.mini_batch - length % self.mini_batch) % self.mini_batch
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        padded = x.shape[1]
        chunks = padded // self.mini_batch

        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        q, k = self.rope(q, k)

        # Official ttt-lm-pytorch: η = token_idx ⊗ (base_lr · σ(gate) / head_dim)
        gate = torch.sigmoid(self.learnable_ttt_lr(x)).transpose(1, 2)
        ttt_lr_eta = (self.ttt_base_lr * gate / self.head_dim).unsqueeze(-2)
        token_scale = (self.token_idx + self.learnable_token_idx).clamp_min(0.0)
        gamma = self.ttt_norm_weight.view(self.heads, 1, self.head_dim)
        beta = self.ttt_norm_bias.view(self.heads, 1, self.head_dim)

        w = self.W1.unsqueeze(0).expand(batch, -1, -1, -1)
        b = self.b1.unsqueeze(0).expand(batch, -1, -1, -1)
        outputs = []
        for i in range(chunks):
            sl = slice(i * self.mini_batch, (i + 1) * self.mini_batch)
            xq = q[:, :, sl]
            xk = k[:, :, sl]
            xv = v[:, :, sl]
            token_eta = token_scale.view(1, 1, self.mini_batch, 1)
            eta = token_eta * ttt_lr_eta[:, :, :, sl]

            z1 = xk @ w + b
            target = xv - xk
            grad_z = _ln_fused_l2_bwd(z1, target, gamma, beta)
            attn = torch.tril(xq @ xk.transpose(-2, -1))
            b_bar = b - torch.tril(eta) @ grad_z
            z_bar = xq @ w - (eta * attn) @ grad_z + b_bar
            last_eta = eta[:, :, -1, :, None]
            w = w - (last_eta * xk).transpose(-1, -2) @ grad_z
            b = b - torch.sum(last_eta * grad_z, dim=-2, keepdim=True)
            z_bar = _ln_fwd(z_bar, gamma, beta)
            outputs.append(xq + z_bar)

        y = torch.cat(outputs, dim=2)[:, :, :length]
        y = y.transpose(1, 2).reshape(batch, length, width)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mixer = TTTLinearMixer(cfg) if cfg.mixer == "ttt_linear" else CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
