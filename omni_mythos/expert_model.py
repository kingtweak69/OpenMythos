"""OmniMythos Expert Backbone — 1024-dim domain specialist.

Architecture: 1 prelude attn → 1 Mamba3 SISO → 1 GDN2 → 1 MLA → 1 coda attn
No embed/lm_head — pure backbone for MoE upcycle.
Input/output: [B, T, 1024] hidden states (projected from backbone's 2048 dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from mamba_ssm.modules.mamba3 import Mamba3
from gdn2 import GatedDeltaNet2


@dataclass
class ExpertConfig:
    dim: int = 1024
    n_heads: int = 8
    # MLA
    kv_lora_rank: int = 256
    q_lora_rank: int = 768
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64
    # misc
    ffn_mult: float = 1.333
    mamba_headdim: int = 64
    mamba_d_state: int = 64
    # for standalone training
    vocab_size: int = 32000
    rope_theta: float = 500000.0
    max_seq_len: int = 1_000_000
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms * self.weight).to(x.dtype)


class FFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def make_ffn(cfg):
    return FFN(cfg.dim, int(cfg.dim * cfg.ffn_mult))


def precompute_freqs_cis(head_dim, max_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    B, H, T, D = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    xc = xc * freqs_cis.view(1, 1, T, D // 2)
    return torch.view_as_real(xc).reshape(B, H, T, D).to(x.dtype)


class SoftmaxAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, freqs_cis):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class SoftmaxBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn  = SoftmaxAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn   = make_ffn(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class MLAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.nope    = cfg.qk_nope_head_dim
        self.rope    = cfg.qk_rope_head_dim
        self.v_dim   = cfg.v_head_dim
        self.qk_dim  = self.nope + self.rope

        self.wq_a    = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm  = RMSNorm(cfg.q_lora_rank)
        self.wq_b    = nn.Linear(cfg.q_lora_rank, cfg.n_heads * self.qk_dim, bias=False)
        self.wkv_a   = nn.Linear(cfg.dim, cfg.kv_lora_rank + self.rope, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.wkv_b   = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * (self.nope + self.v_dim), bias=False)
        self.wo      = nn.Linear(cfg.n_heads * self.v_dim, cfg.dim, bias=False)
        self.kv_lora_rank = cfg.kv_lora_rank

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        H = self.n_heads

        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, H, self.qk_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        q_pe = apply_rope(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        c_kv, k_pe = kv.split([self.kv_lora_rank, self.rope], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(1), freqs_cis)
        kv = self.wkv_b(self.kv_norm(c_kv)).view(B, T, H, self.nope + self.v_dim).transpose(1, 2)
        k_nope, v = kv.split([self.nope, self.v_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, H * self.v_dim))


class MLABlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn  = MLAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn   = make_ffn(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class Mamba3Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.mixer = Mamba3(
            d_model=cfg.dim,
            d_state=cfg.mamba_d_state,
            headdim=cfg.mamba_headdim,
            expand=1,
            ngroups=1,
            is_mimo=False,
            chunk_size=64,
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn   = make_ffn(cfg)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GDN2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.mixer = GatedDeltaNet2(
            hidden_size=cfg.dim,
            head_dim=cfg.mamba_headdim,
            num_heads=cfg.dim // cfg.mamba_headdim,
            mode="chunk",
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn   = make_ffn(cfg)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ExpertBackbone(nn.Module):
    """Pure backbone — no embed/lm_head. Used as MoE expert."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg = cfg

        self.prelude  = SoftmaxBlock(cfg)
        self.m3       = Mamba3Block(cfg)
        self.gdn      = GDN2Block(cfg)
        self.mla      = MLABlock(cfg)
        self.coda     = SoftmaxBlock(cfg)
        self.norm_f   = RMSNorm(cfg.dim)

        head_dim = cfg.dim // cfg.n_heads
        self.freqs_softmax = precompute_freqs_cis(head_dim, 8192, cfg.rope_theta)
        self.freqs_mla     = precompute_freqs_cis(cfg.qk_rope_head_dim, 8192, cfg.rope_theta)

    def _freqs(self, buf, T, device):
        if buf.shape[0] < T:
            head_dim = buf.shape[1] * 2
            buf = precompute_freqs_cis(head_dim, min(2 * T, self.cfg.max_seq_len), self.cfg.rope_theta)
        if buf.device != device:
            buf = buf.to(device)
        return buf[:T], buf

    def forward(self, x):
        """x: [B, T, dim] → [B, T, dim]"""
        T = x.shape[1]
        fs, self.freqs_softmax = self._freqs(self.freqs_softmax, T, x.device)
        fm, self.freqs_mla     = self._freqs(self.freqs_mla,     T, x.device)

        x = self.prelude(x, fs)
        x = self.m3(x)
        x = self.gdn(x)
        x = self.mla(x, fm)
        x = self.coda(x, fs)
        return self.norm_f(x)


class ExpertLM(nn.Module):
    """Standalone trainable version with embed + lm_head for pretraining."""

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.cfg      = cfg
        self.embed    = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.backbone = ExpertBackbone(cfg)
        self.lm_head  = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.backbone(x)
        return self.lm_head(x)
