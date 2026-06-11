"""OmniMythos v3 — dense backbone with interleaved Mamba3 + GDN2 recurrent core.

Recurrent inner loop pattern: M3 → GDN2 → M3 → MLA → GDN2 → M3 → GDN2 → MLA

Components:
  - Mamba3Block: official mamba_ssm Mamba3 MIMO
  - GDN2Block: official NVIDIA GatedDeltaNet2 from FLA
  - MLA: DeepSeek-style multi-head latent attention
  - SoftmaxAttention: standard MHA with RoPE
  - HyperConnections, LTIInjection, LoopEmbedding, LoRADepthAdapter, ACTHalting
  - ModalityBlock: cross-attention bridge for audio/vision encoders

MoE upscale: swap make_ffn() return. Audio/image heads are plain Linear; swap at upscale time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from mamba_ssm.modules.mamba3 import Mamba3
from gdn2 import GatedDeltaNet2


# =============================================================================
# Config
# =============================================================================

@dataclass
class MythosConfig:
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    max_seq_len: int = 1_000_000
    # MLA
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    # structure
    prelude_attn_layers: int = 4
    coda_attn_layers: int = 4
    max_loop_iters: int = 16
    act_threshold: float = 0.99
    hyper_n_streams: int = 4
    # multimodal
    audio_vocab: int = 1024
    audio_n_codebooks: int = 8
    image_vocab: int = 8192
    audio_encoder_dim: int = 1280
    vision_encoder_dim: int = 1152
    # misc
    rope_theta: float = 500000.0
    lora_rank: int = 16
    ffn_mult: float = 1.333
    tie_embeddings: bool = True
    mamba_headdim: int = 64
    mamba_d_state: int = 128


# =============================================================================
# Basics
# =============================================================================

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


# =============================================================================
# RoPE
# =============================================================================

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


# =============================================================================
# Attention blocks
# =============================================================================

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


class MLADenseBlock(nn.Module):
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


# =============================================================================
# Mamba3 block
# =============================================================================

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
            is_mimo=True,
            mimo_rank=4,
            chunk_size=64,
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn   = make_ffn(cfg)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# GDN2 block (official GatedDeltaNet2)
# =============================================================================

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


# =============================================================================
# Recurrent-depth machinery
# =============================================================================

class HyperConnections(nn.Module):
    def __init__(self, dim, n_streams):
        super().__init__()
        self.n = n_streams
        self.static_beta   = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.static_alpha  = nn.Parameter(torch.eye(n_streams))
        self.static_alpha0 = nn.Parameter(torch.ones(n_streams) / n_streams)
        self.norm       = RMSNorm(dim)
        self.dyn_beta   = nn.Linear(dim, 1, bias=False)
        self.dyn_alpha  = nn.Linear(dim, n_streams, bias=False)
        self.dyn_alpha0 = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.dyn_beta.weight)
        nn.init.zeros_(self.dyn_alpha.weight)
        nn.init.zeros_(self.dyn_alpha0.weight)

    def expand(self, x):
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1]).contiguous()

    def width(self, streams):
        ns = self.norm(streams)
        beta = self.static_beta + torch.tanh(self.dyn_beta(ns)).squeeze(-1)
        return (beta.unsqueeze(-1) * streams).sum(dim=-2)

    def depth(self, streams, block_out):
        ns = self.norm(streams)
        alpha  = self.static_alpha  + torch.tanh(self.dyn_alpha(ns))
        alpha0 = self.static_alpha0 + torch.tanh(self.dyn_alpha0(ns)).squeeze(-1)
        mixed  = torch.einsum("btij,btjd->btid", alpha, streams)
        return mixed + alpha0.unsqueeze(-1) * block_out.unsqueeze(-2)

    def collapse(self, streams):
        return streams.sum(dim=-2)


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A  = nn.Parameter(torch.zeros(dim))
        self.gate_e = nn.Parameter(torch.zeros(dim))

    def forward(self, h, e, block_out):
        decay = torch.exp(-F.softplus(self.log_A))
        return decay * h + torch.sigmoid(self.gate_e) * e + block_out


class LoopEmbedding(nn.Module):
    def __init__(self, dim, max_loops):
        super().__init__()
        self.emb = nn.Embedding(max_loops, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, h, t):
        return h + self.emb.weight[t].to(h.dtype)


class LoRADepthAdapter(nn.Module):
    def __init__(self, dim, rank, max_loops):
        super().__init__()
        self.down      = nn.Linear(dim, rank, bias=False)
        self.up        = nn.Linear(rank, dim, bias=False)
        self.loop_gate = nn.Embedding(max_loops, rank)
        nn.init.zeros_(self.up.weight)
        nn.init.ones_(self.loop_gate.weight)

    def forward(self, x, t):
        return self.up(self.down(x) * self.loop_gate.weight[t].to(x.dtype))


class ACTHalting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.halt = nn.Linear(dim, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -2.0)

    def forward(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# =============================================================================
# Recurrent core — M3 → GDN2 → M3 → MLA → GDN2 → M3 → GDN2 → MLA
# =============================================================================

class RecurrentDenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.m3_0    = Mamba3Block(cfg)
        self.gdn_0   = GDN2Block(cfg)
        self.m3_1    = Mamba3Block(cfg)
        self.mla_mid = MLADenseBlock(cfg)
        self.gdn_1   = GDN2Block(cfg)
        self.m3_2    = Mamba3Block(cfg)
        self.gdn_2   = GDN2Block(cfg)
        self.mla_exit = MLADenseBlock(cfg)

        self.hyper     = HyperConnections(cfg.dim, cfg.hyper_n_streams)
        self.injection = LTIInjection(cfg.dim)
        self.act       = ACTHalting(cfg.dim)
        self.lora      = LoRADepthAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.loop_emb  = LoopEmbedding(cfg.dim, cfg.max_loop_iters)
        self.norm      = RMSNorm(cfg.dim)

    def inner(self, x, freqs_cis, t):
        x = self.m3_0(x)
        x = self.gdn_0(x)
        x = self.m3_1(x)
        x = self.mla_mid(x, freqs_cis)
        x = self.gdn_1(x)
        x = self.m3_2(x)
        x = self.gdn_2(x)
        x = self.mla_exit(x, freqs_cis)
        return x + self.lora(x, t)

    def forward(self, h, e, freqs_cis, n_loops=None):
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        streams      = self.hyper.expand(h)
        halted       = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device, dtype=torch.float32)
        h_out        = torch.zeros(B, T, D, device=h.device, dtype=torch.float32)

        for t in range(n_loops):
            x = self.hyper.width(streams)
            x = self.loop_emb(x, t)
            x = self.injection(x, e, torch.zeros_like(x))
            x = self.norm(x)
            block_out = self.inner(x, freqs_cis, t)
            streams   = self.hyper.depth(streams, block_out)

            h_cur = self.hyper.collapse(streams)
            p     = self.act(h_cur)

            still     = (~halted).float()
            remainder = (1.0 - cumulative_p).clamp(min=0)
            last      = t == n_loops - 1
            weight    = torch.where(
                (cumulative_p + p >= self.cfg.act_threshold) | last, remainder, p
            ) * still
            h_out        = h_out + weight.unsqueeze(-1) * h_cur.float()
            cumulative_p = cumulative_p + p * still
            halted       = halted | (cumulative_p >= self.cfg.act_threshold)
            if bool(halted.all()) and not self.training:
                break
        return h_out.to(e.dtype)


# =============================================================================
# Multimodal bridges
# =============================================================================

class CrossAttention(nn.Module):
    def __init__(self, dim, encoder_dim, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(encoder_dim, dim, bias=False)
        self.wv = nn.Linear(encoder_dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, encoder_out):
        B, T, _ = x.shape
        S = encoder_out.shape[1]
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(encoder_out).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(encoder_out).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class ModalityBlock(nn.Module):
    def __init__(self, cfg, encoder_dim):
        super().__init__()
        self.norm1      = RMSNorm(cfg.dim)
        self.cross_attn = CrossAttention(cfg.dim, encoder_dim, cfg.n_heads)
        self.norm2      = RMSNorm(cfg.dim)
        self.ffn        = make_ffn(cfg)

    def forward(self, x, encoder_out):
        x = x + self.cross_attn(self.norm1(x), encoder_out)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Full model
# =============================================================================

class OmniMythosDense(nn.Module):
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg   = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.audio_encoder  = ModalityBlock(cfg, cfg.audio_encoder_dim)
        self.vision_encoder = ModalityBlock(cfg, cfg.vision_encoder_dim)

        self.prelude_attn = nn.ModuleList([SoftmaxBlock(cfg) for _ in range(cfg.prelude_attn_layers)])
        self.recurrent    = RecurrentDenseBlock(cfg)
        self.coda_attn    = nn.ModuleList([SoftmaxBlock(cfg) for _ in range(cfg.coda_attn_layers)])

        self.norm_f  = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # MoE upscale points
        self.audio_head = nn.Linear(cfg.dim, cfg.audio_vocab * cfg.audio_n_codebooks)
        self.image_head = nn.Linear(cfg.dim, cfg.image_vocab, bias=False)

        head_dim = cfg.dim // cfg.n_heads
        self.freqs_softmax = precompute_freqs_cis(head_dim, 8192, cfg.rope_theta)
        self.freqs_mla     = precompute_freqs_cis(cfg.qk_rope_head_dim, 8192, cfg.rope_theta)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.audio_head.weight, std=0.02)
        nn.init.zeros_(self.audio_head.bias)
        nn.init.normal_(self.image_head.weight, std=0.02)

    def _freqs(self, buf, T, device):
        if buf.shape[0] < T:
            head_dim = buf.shape[1] * 2
            buf = precompute_freqs_cis(head_dim, min(2 * T, self.cfg.max_seq_len), self.cfg.rope_theta)
        if buf.device != device:
            buf = buf.to(device)
        return buf[:T], buf

    def forward(self, input_ids, audio_features=None, vision_features=None, n_loops=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)

        fs, self.freqs_softmax = self._freqs(self.freqs_softmax, T, x.device)
        fm, self.freqs_mla     = self._freqs(self.freqs_mla,     T, x.device)

        for blk in self.prelude_attn:
            x = blk(x, fs)

        if audio_features is not None:
            x = self.audio_encoder(x, audio_features)
        if vision_features is not None:
            x = self.vision_encoder(x, vision_features)

        e = x
        x = self.recurrent(x, e, fm, n_loops)

        for blk in self.coda_attn:
            x = blk(x, fs)

        x = self.norm_f(x)
        return self.lm_head(x), self.audio_head(x), self.image_head(x)
