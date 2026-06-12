"""
Pure PyTorch Mamba3 — no CUDA compilation required.
Drop-in replacement for mamba_ssm.modules.mamba3.Mamba3
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


def build_rope_freqs(num_angles: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(num_angles, device=device, dtype=torch.float32)
    return 1.0 / (10000.0 ** (i / num_angles))


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


def mamba3_siso_scan(x, B_proj, C_proj, ADT, DT, trap, D_skip):
    B_batch, L, H, P = x.shape
    D_state = B_proj.shape[-1]
    device, dtype = x.device, x.dtype

    h = torch.zeros(B_batch, H, P, D_state, device=device, dtype=torch.float32)
    Bx_prev = torch.zeros_like(h)
    ys = []

    for t in range(L):
        x_t   = x[:, t]
        B_t   = B_proj[:, t]
        C_t   = C_proj[:, t]
        adt_t = ADT[:, t]
        dt_t  = DT[:, t]
        tr_t  = trap[:, t]

        decay = torch.exp(adt_t).unsqueeze(-1).unsqueeze(-1)
        Bx_curr = torch.einsum("bhp,bhd->bhpd", x_t.float(), B_t.float())
        dt_e = dt_t.unsqueeze(-1).unsqueeze(-1)
        tr_e = tr_t.unsqueeze(-1).unsqueeze(-1)
        Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + Bx_prev)
        h = decay * h + dt_e * Bx_blended
        y_t = torch.einsum("bhd,bhpd->bhp", C_t.float(), h)
        y_t = y_t + D_skip.unsqueeze(0).unsqueeze(-1) * x_t.float()
        ys.append(y_t.to(dtype))
        Bx_prev = Bx_curr

    return torch.stack(ys, dim=1)


def mamba3_mimo_scan(x, B_proj, C_proj, ADT, DT, trap, D_skip, mimo_x, mimo_o):
    B_batch, L, H, P = x.shape
    D_state = B_proj.shape[-1]
    device, dtype = x.device, x.dtype

    h = torch.zeros(B_batch, H, D_state, device=device, dtype=torch.float32)
    Bx_prev = torch.zeros_like(h)
    ys = []

    for t in range(L):
        x_t   = x[:, t]
        B_t   = B_proj[:, t]
        C_t   = C_proj[:, t]
        adt_t = ADT[:, t]
        dt_t  = DT[:, t]
        tr_t  = trap[:, t]

        decay = torch.exp(adt_t)
        x_r = torch.einsum("bhp,hrp->bhr", x_t.float(), mimo_x.float())
        Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_t.float())
        tr_e = tr_t.unsqueeze(-1)
        Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + Bx_prev)
        h = decay.unsqueeze(-1) * h + dt_t.unsqueeze(-1) * Bx_blended
        y_r_scalar = torch.einsum("brhd,bhd->brh", C_t.float(), h)
        skip = D_skip.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)
        y_pre = y_r_scalar + skip
        y_t = torch.einsum("brh,hrp->bhp", y_pre, mimo_o.float())
        ys.append(y_t.to(dtype))
        Bx_prev = Bx_curr

    return torch.stack(ys, dim=1)


class Mamba3(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,  # ignored, kept for API compat
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model   = d_model
        self.d_state   = d_state
        self.expand    = expand
        self.headdim   = headdim
        self.A_floor   = A_floor
        self.is_mimo   = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.num_bc_heads = ngroups

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        assert rope_fraction in [0.5, 1.0]
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        d_in_proj = (
            2 * self.d_inner
            + 2 * d_state * ngroups * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.B_bias._no_weight_decay = True
        self.C_bias._no_weight_decay = True

        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs)
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )

        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, L, _ = u.shape

        zxBCdtAtrap = self.in_proj(u)

        (z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angle_raw) = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        B_raw = rearrange(B_raw, "b l (r g n) -> b l r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)
        C_raw = rearrange(C_raw, "b l (r g n) -> b l r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)

        A   = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)
        DT  = F.softplus(dd_dt.float() + self.dt_bias)
        ADT = A * DT
        trap = torch.sigmoid(trap_raw.float())

        B_normed = self.B_norm(B_raw.float())
        C_normed = self.C_norm(C_raw.float())

        B_exp = B_normed.expand(-1, -1, -1, self.nheads, -1)
        C_exp = C_normed.expand(-1, -1, -1, self.nheads, -1)

        B_bias_t = rearrange(self.B_bias, "h r d -> r h d")
        C_bias_t = rearrange(self.C_bias, "h r d -> r h d")
        B_exp = B_exp + B_bias_t
        C_exp = C_exp + C_bias_t

        angle_increments = angle_raw.float().unsqueeze(2) * DT.float().unsqueeze(-1)
        cumulative_angles = torch.cumsum(angle_increments, dim=1)
        angles_for_rot = cumulative_angles.unsqueeze(2).expand(
            batch, L, self.mimo_rank, self.nheads, self.num_rope_angles
        )

        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)
        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)

        if self.is_mimo:
            y = mamba3_mimo_scan(x, B_proj, C_proj, ADT, DT, trap,
                                  self.D, self.mimo_x, self.mimo_o)
            y = y * F.silu(z.float())
        else:
            y = mamba3_siso_scan(x, B_proj[:, :, 0], C_proj[:, :, 0],
                                  ADT, DT, trap, self.D)
            y = y * F.silu(z.float())

        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y.to(x.dtype))
