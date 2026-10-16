"""B3 · Anima: Cosmos-Predict2-2B DiT + LLM adapter (Qwen3-0.6B → Cosmos 텍스트 공간).

diffusers 원본: models/transformers/transformer_cosmos.py (CosmosTransformer3DModel)
              + models/condition_embedders/condition_embedder_anima.py (AnimaTextConditioner).
파라미터 이름은 diffusers 와 같다 (text_conditioner.* / transformer.* 두 서브모듈).

전체 흐름 (파이프라인 한 step)
  Qwen3-0.6B 마지막 hidden [B, Ls, 1024] ──┐
  T5 토큰 id [B, Lt] → 학습된 임베딩 (질의) ─┴→ LLM adapter → 텍스트 임베딩 [B, max(Lt, 512), 1024]
  latent [B, 16, 1, h, w] + 시간 t ──────────────→ Cosmos DiT (cross-attn 으로 텍스트를 읽음) → 속도 예측

1. LLM adapter (AnimaTextConditioner) — Anima 고유 부품
   - 질의 = T5 토큰 임베딩. 블록마다 self-attn(질의끼리) → cross-attn(Qwen hidden 을 읽음) → GELU MLP, pre-norm residual.
   - q/k 에 head 별 RMSNorm, 그리고 RoPE. cross-attn 에서도 q 는 T5 위치, k 는 Qwen 위치로 각각 회전한다.
   - 끝: out_proj → RMSNorm → T5 padding 위치 0 → 길이가 512 보다 짧으면 0 행으로 채움 (DiT 에서는 mask 없이 읽힘).
2. Cosmos DiT
   - 입력 채널 끝에 padding mask 채널을 붙이고 (1,2,2) 3D patch → Linear (bias 없음).
   - 위치: 3D RoPE. head 채널을 [t | h | w] 로 나누고 (h, w 는 head_dim//6*2, t 는 나머지), 축마다 NTK 배율을 곱한 theta.
   - 시간: sincos(cos 먼저) → t_embedder (Linear → SiLU → Linear 3C) = 모든 블록이 공유하는 base 변조 temb,
            sincos 에 RMSNorm 한 것 = embedded_timestep.
   - adaLN-LoRA: 블록의 norm 마다 SiLU(embedded_timestep) → C→r→3C 저랭크 Linear 로 shift/scale/gate 를 만들고 temb 를 더한다.
   - 블록 = [self-attn(3D RoPE, qk RMSNorm) → cross-attn(텍스트, RoPE 없음) → GELU FFN], 각각 adaLN 변조 + gate residual.
   - 출력: adaLN(shift/scale, temb 앞 2C 를 더함) → Linear → unpatchify.
   - Cosmos-Predict2 비디오 경로(5D timestep = 프레임별 t, fps 로 시간축 RoPE 늘이기, learnable 절대 위치)도 구현.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------- config

@dataclass
class AdapterConfig:
    source_dim: int = 1024              # Qwen3-0.6B hidden
    target_dim: int = 1024              # 출력 = DiT text_embed_dim
    model_dim: int = 1024
    n_layer: int = 6
    n_head: int = 16
    mlp_ratio: float = 4.0
    vocab_size: int = 32128             # T5 vocab
    use_self_attention: bool = True
    use_layer_norm: bool = False        # False → RMSNorm(1e-6), True → LayerNorm(1e-5, affine)
    min_seq_len: int = 512              # 짧으면 0 행으로 채움
    rope_theta: float = 10000.0
    # ablation 스위치 (기본값 = 원본)
    use_rope: bool = True               # False: adapter attention 에서 RoPE 생략


@dataclass
class CosmosConfig:
    in_ch: int = 16
    out_ch: int = 16
    n_head: int = 16
    head_dim: int = 128
    n_layer: int = 28
    mlp_ratio: float = 4.0
    text_dim: int = 1024
    adaln_lora_dim: int = 256
    max_size: tuple = (128, 240, 240)
    patch: tuple = (1, 2, 2)
    rope_scale: tuple = (1.0, 4.0, 4.0)     # (t, h, w) NTK 배율 (Anima / Predict2-2B T2I)
    concat_padding_mask: bool = True
    learnable_pos: bool = False             # extra_pos_embed_type == "learnable"
    base_fps: int = 24
    # ablation 스위치 (기본값 = 원본)
    use_rope: bool = True                   # False: self-attn 에서 3D RoPE 생략
    use_qk_norm: bool = True                # False: q/k RMSNorm 생략
    use_adaln_lora: bool = True             # False: 블록 변조에 공유 temb 를 안 더함 (저랭크 경로만)
    use_cross_attn: bool = True             # False: 텍스트 cross-attn 블록을 건너뜀 (무조건 생성)

    @property
    def d_model(self) -> int:
        return self.n_head * self.head_dim


@dataclass
class AnimaConfig:
    dit: CosmosConfig = field(default_factory=CosmosConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)


# ---------------------------------------------------------------- 공통 부품

class RMSNorm(nn.Module):
    """diffusers RMSNorm: x * rsqrt(mean(x^2) + eps) * w (분산은 fp32)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


def attend(q, k, v, mask=None, probs_hook=None):
    """명시적 softmax attention (SDPA 와 같은 mask 규칙). q/k/v [B, H, S, D].

    mask: bool → False 칸 제외, float → 점수에 더함 (SDPA 와 동일). [B, 1, 1, Sk] 로 broadcast.
    전부 가려진 행(빈 프롬프트의 Qwen mask 등)은 torch SDPA math 경로처럼 확률 0 → 출력 0.
    """
    s = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask
    p = s.float().softmax(-1)
    if mask is not None and mask.dtype == torch.bool:
        p = p.nan_to_num(0.0)
    p = p.to(q.dtype)
    if probs_hook is not None:
        p = probs_hook(p)
    return p @ v


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------- 1. LLM adapter

class TextAttention(nn.Module):
    """adapter attention: head 별 q/k RMSNorm(1e-6) + RoPE (q, k 각자 자기 위치)."""

    def __init__(self, query_dim, context_dim, n_head, head_dim):
        super().__init__()
        inner = n_head * head_dim
        self.h, self.hd = n_head, head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)
        self.attn_probs = nn.Identity()  # hook 지점: [B, H, Lq, Lk]

    def forward(self, x, mask=None, ctx=None, pos=None, ctx_pos=None):
        ctx = x if ctx is None else ctx
        B, Lq, Lk = x.shape[0], x.shape[1], ctx.shape[1]
        q = self.q_norm(self.q_proj(x).view(B, Lq, self.h, self.hd))
        k = self.k_norm(self.k_proj(ctx).view(B, Lk, self.h, self.hd))
        v = self.v_proj(ctx).view(B, Lk, self.h, self.hd)
        if pos is not None:                       # cos/sin [1, L, hd] → head 축에 broadcast
            cos, sin = (t.unsqueeze(2) for t in pos)
            q = q * cos + rotate_half(q) * sin
            cos, sin = (t.unsqueeze(2) for t in ctx_pos)
            k = k * cos + rotate_half(k) * sin
        o = attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask, self.attn_probs)
        return self.o_proj(o.transpose(1, 2).flatten(2, 3))


class TextBlock(nn.Module):
    def __init__(self, c: AdapterConfig):
        super().__init__()
        self.c = c
        d, hd = c.model_dim, c.model_dim // c.n_head

        def norm():
            return nn.LayerNorm(d) if c.use_layer_norm else nn.RMSNorm(d, eps=1e-6)
        if c.use_self_attention:
            self.norm_self_attn = norm()
            self.self_attn = TextAttention(d, d, c.n_head, hd)
        self.norm_cross_attn = norm()
        self.cross_attn = TextAttention(d, c.source_dim, c.n_head, hd)
        self.norm_mlp = norm()
        inner = int(d * c.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, inner), nn.GELU(), nn.Linear(inner, d))

    def forward(self, x, src, tgt_mask, src_mask, pos, src_pos):
        if self.c.use_self_attention:
            x = x + self.self_attn(self.norm_self_attn(x), tgt_mask, pos=pos, ctx_pos=pos)
        x = x + self.cross_attn(self.norm_cross_attn(x), src_mask, ctx=src, pos=pos, ctx_pos=src_pos)
        return x + self.mlp(self.norm_mlp(x))


class TextConditioner(nn.Module):
    """T5 토큰 질의가 Qwen3 hidden 을 읽어 Cosmos 텍스트 임베딩을 만든다."""

    def __init__(self, c: AdapterConfig):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.target_dim)
        self.in_proj = nn.Linear(c.target_dim, c.model_dim) if c.model_dim != c.target_dim else nn.Identity()
        hd = c.model_dim // c.n_head
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).to(torch.float32) / hd))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.blocks = nn.ModuleList(TextBlock(c) for _ in range(c.n_layer))
        self.out_proj = nn.Linear(c.model_dim, c.target_dim)
        self.norm = nn.RMSNorm(c.target_dim, eps=1e-6)

    def _rope(self, n, ref):
        pos = torch.arange(n, device=ref.device).float()
        freqs = (pos[:, None] * self.inv_freq.float().to(ref.device)[None])[None]   # [1, L, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(ref.dtype), emb.sin().to(ref.dtype)

    @staticmethod
    def _mask(m):
        if m is None:
            return None
        m = m.to(torch.bool)
        return m[:, None, None, :] if m.ndim == 2 else m

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None,
                source_attention_mask=None):
        tm, sm = self._mask(target_attention_mask), self._mask(source_attention_mask)
        x = self.in_proj(self.embed(target_input_ids).to(source_hidden_states.dtype))
        pos = src_pos = None
        if self.config.use_rope:
            pos = self._rope(x.shape[1], x)
            src_pos = self._rope(source_hidden_states.shape[1], x)
        for b in self.blocks:
            x = b(x, source_hidden_states, tm, sm, pos, src_pos)
        x = self.norm(self.out_proj(x))
        if tm is not None:
            x = x * tm.squeeze(1).squeeze(1).to(x).unsqueeze(-1)
        if x.shape[1] < self.config.min_seq_len:
            x = F.pad(x, (0, 0, 0, self.config.min_seq_len - x.shape[1]))
        return x


# ---------------------------------------------------------------- 2. Cosmos DiT 부품

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, d, patch):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_ch * patch[0] * patch[1] * patch[2], d, bias=False)

    def forward(self, x):
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch
        x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)      # patch 안 순서 = (c, pt, ph, pw)
        return self.proj(x)                                      # [B, T', H', W', D]


class RoPE3D(nn.Module):
    """head 채널을 [t | h | w] (각 절반씩, rotate_half 배치) 로 나눈 3D RoPE. (cos, sin) [THW, head_dim]."""

    def __init__(self, c: CosmosConfig):
        super().__init__()
        self.c = c
        self.max_size = [s // p for s, p in zip(c.max_size, c.patch)]
        d = c.head_dim
        self.dim_h = d // 6 * 2
        self.dim_w = d // 6 * 2
        self.dim_t = d - self.dim_h - self.dim_w
        self.h_ntk = c.rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk = c.rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk = c.rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, x, fps=None):
        _, _, T, H, W = x.shape
        pt, ph, pw = self.c.patch
        n = [T // pt, H // ph, W // pw]
        dev = x.device
        seq = torch.arange(max(self.max_size), device=dev, dtype=torch.float32)

        def freqs(dim, ntk):
            r = torch.arange(0, dim, 2, device=dev, dtype=torch.float32)[: dim // 2] / dim
            return 1.0 / ((10000.0 * ntk) ** r)
        fh, fw, ft = freqs(self.dim_h, self.h_ntk), freqs(self.dim_w, self.w_ntk), freqs(self.dim_t, self.t_ntk)
        eh = torch.outer(seq[: n[1]], fh)[None, :, None, :].repeat(n[0], 1, n[2], 1)
        ew = torch.outer(seq[: n[2]], fw)[None, None, :, :].repeat(n[0], n[1], 1, 1)
        tpos = seq[: n[0]] if fps is None else seq[: n[0]] / fps * self.c.base_fps   # 비디오: fps 로 시간축 보정
        et = torch.outer(tpos, ft)[:, None, None, :].repeat(1, n[1], n[2], 1)
        f = torch.cat([et, eh, ew] * 2, dim=-1).flatten(0, 2).float()
        return torch.cos(f), torch.sin(f)


class LearnablePos(nn.Module):
    """축별 학습 위치 (t + h + w) 를 RMS 크기로 정규화 (Predict2 비디오 일부 config)."""

    def __init__(self, c: CosmosConfig, eps=1e-6):
        super().__init__()
        self.c, self.eps = c, eps
        m = [s // p for s, p in zip(c.max_size, c.patch)]
        self.pos_emb_t = nn.Parameter(torch.zeros(m[0], c.d_model))
        self.pos_emb_h = nn.Parameter(torch.zeros(m[1], c.d_model))
        self.pos_emb_w = nn.Parameter(torch.zeros(m[2], c.d_model))

    def forward(self, x):
        B, _, T, H, W = x.shape
        pt, ph, pw = self.c.patch
        n = [T // pt, H // ph, W // pw]
        et = self.pos_emb_t[: n[0]][None, :, None, None, :].repeat(B, 1, n[1], n[2], 1)
        eh = self.pos_emb_h[: n[1]][None, None, :, None, :].repeat(B, n[0], 1, n[2], 1)
        ew = self.pos_emb_w[: n[2]][None, None, None, :, :].repeat(B, n[0], n[1], 1, 1)
        e = (et + eh + ew).flatten(1, 3)
        norm = torch.linalg.vector_norm(e, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=math.sqrt(norm.numel() / e.numel()))
        return (e / norm).type_as(x)


def timestep_sincos(t, dim, max_period=10000):
    """diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos | sin]."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear_1 = nn.Linear(d, d, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(d, 3 * d, bias=False)

    def forward(self, x):
        return self.linear_2(self.activation(self.linear_1(x)))


class TimeEmbed(nn.Module):
    """→ (temb [.., 3C]: 모든 블록 공유 base 변조, embedded_timestep [.., C]: 블록별 LoRA 입력)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.t_embedder = TimestepMLP(d)
        self.norm = RMSNorm(d, eps=1e-6)

    def forward(self, x, t):
        proj = timestep_sincos(t, self.d).type_as(x)
        return self.t_embedder(proj), self.norm(proj)


class AdaLNZero(nn.Module):
    """adaLN-LoRA: SiLU(e) → C→r→3C (+ temb) → shift, scale, gate."""

    def __init__(self, d, r, c: CosmosConfig):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()
        self.linear_1 = nn.Linear(d, r, bias=False)
        self.linear_2 = nn.Linear(r, 3 * d, bias=False)
        self.shift_probe = nn.Identity()   # hook 지점: shift [B, C] (5D timestep 이면 [B, N, C])
        self.scale_probe = nn.Identity()   # hook 지점: scale
        self.gate_probe = nn.Identity()    # hook 지점: gate

    def forward(self, x, e, temb=None):
        e = self.linear_2(self.linear_1(self.activation(e)))
        if temb is not None and self.c.use_adaln_lora:
            e = e + temb
        shift, scale, gate = e.chunk(3, dim=-1)
        shift, scale, gate = self.shift_probe(shift), self.scale_probe(scale), self.gate_probe(gate)
        x = self.norm(x)
        if e.ndim == 2:
            shift, scale, gate = (t.unsqueeze(1) for t in (shift, scale, gate))
        return x * (1 + scale) + shift, gate


class AdaLNOut(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.d = d
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(d, r, bias=False)
        self.linear_2 = nn.Linear(r, 2 * d, bias=False)
        self.shift_probe = nn.Identity()
        self.scale_probe = nn.Identity()

    def forward(self, x, e, temb=None):
        e = self.linear_2(self.linear_1(self.activation(e)))
        if temb is not None:
            e = e + temb[..., : 2 * self.d]
        shift, scale = e.chunk(2, dim=-1)
        shift, scale = self.shift_probe(shift), self.scale_probe(scale)
        x = self.norm(x)
        if e.ndim == 2:
            shift, scale = (t.unsqueeze(1) for t in (shift, scale))
        return x * (1 + scale) + shift


class Attention(nn.Module):
    """DiT attention (bias 없음). q/k RMSNorm(head_dim, eps 1e-5), self-attn 이면 3D RoPE."""

    def __init__(self, d, ctx_dim, c: CosmosConfig):
        super().__init__()
        self.c = c
        inner = c.n_head * c.head_dim
        self.to_q = nn.Linear(d, inner, bias=False)
        self.to_k = nn.Linear(ctx_dim, inner, bias=False)
        self.to_v = nn.Linear(ctx_dim, inner, bias=False)
        self.norm_q = RMSNorm(c.head_dim, eps=1e-5)
        self.norm_k = RMSNorm(c.head_dim, eps=1e-5)
        self.to_out = nn.ModuleList([nn.Linear(inner, d, bias=False), nn.Identity()])   # 1 = dropout 자리
        self.attn_probs = nn.Identity()   # hook 지점: self [B, H, N, N] / cross [B, H, N, L]

    def forward(self, x, ctx=None, mask=None, rope=None):
        ctx = x if ctx is None else ctx
        h = self.c.n_head
        q, k, v = (f(i).unflatten(2, (h, -1)).transpose(1, 2)
                   for f, i in ((self.to_q, x), (self.to_k, ctx), (self.to_v, ctx)))
        if self.c.use_qk_norm:
            q, k = self.norm_q(q), self.norm_k(k)
        if rope is not None:
            cos, sin = (t[None, None].to(q.device) for t in rope)
            q = (q.float() * cos + rotate_half(q).float() * sin).to(q.dtype)
            k = (k.float() * cos + rotate_half(k).float() * sin).to(k.dtype)
        o = attend(q, k, v, mask, self.attn_probs)
        o = o.transpose(1, 2).flatten(2, 3).type_as(q)
        return self.to_out[1](self.to_out[0](o))


class GELUProj(nn.Module):
    def __init__(self, d, inner):
        super().__init__()
        self.proj = nn.Linear(d, inner, bias=False)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        inner = int(d * mult)
        self.net = nn.ModuleList([GELUProj(d, inner), nn.Identity(), nn.Linear(inner, d, bias=False)])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


class Block(nn.Module):
    def __init__(self, c: CosmosConfig):
        super().__init__()
        self.c = c
        d = c.d_model
        self.norm1 = AdaLNZero(d, c.adaln_lora_dim, c)
        self.attn1 = Attention(d, d, c)
        self.norm2 = AdaLNZero(d, c.adaln_lora_dim, c)
        self.attn2 = Attention(d, c.text_dim, c)
        self.norm3 = AdaLNZero(d, c.adaln_lora_dim, c)
        self.ff = FeedForward(d, c.mlp_ratio)

    def forward(self, x, text, e, temb, rope=None, extra_pos=None, mask=None):
        if extra_pos is not None:                  # learnable 위치는 블록마다 다시 더한다
            x = x + extra_pos
        h, gate = self.norm1(x, e, temb)
        x = x + gate * self.attn1(h, rope=rope if self.c.use_rope else None)
        if self.c.use_cross_attn:
            h, gate = self.norm2(x, e, temb)
            x = x + gate * self.attn2(h, ctx=text, mask=mask)
        h, gate = self.norm3(x, e, temb)
        return x + gate * self.ff(h)


class CosmosDiT(nn.Module):
    def __init__(self, c: CosmosConfig):
        super().__init__()
        self.config = c
        d = c.d_model
        self.patch_embed = PatchEmbed(c.in_ch + int(c.concat_padding_mask), d, c.patch)
        self.rope = RoPE3D(c)
        self.learnable_pos_embed = LearnablePos(c) if c.learnable_pos else None
        self.time_embed = TimeEmbed(d)
        self.transformer_blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm_out = AdaLNOut(d, c.adaln_lora_dim)
        self.proj_out = nn.Linear(d, c.patch[0] * c.patch[1] * c.patch[2] * c.out_ch, bias=False)

    def forward(self, hidden_states, timestep, encoder_hidden_states, attention_mask=None, fps=None,
                condition_mask=None, padding_mask=None):
        c = self.config
        x = hidden_states
        B, _, T, H, W = x.shape
        if condition_mask is not None:
            x = torch.cat([x, condition_mask], dim=1)
        if c.concat_padding_mask:              # 원본: torchvision resize(NEAREST) = interpolate(mode="nearest")
            if padding_mask is None:
                padding_mask = x.new_zeros(1, 1, H, W)
            pm = F.interpolate(padding_mask, size=list(x.shape[-2:]), mode="nearest")
            x = torch.cat([x, pm.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1)
        mask = attention_mask[:, None, None, :] if attention_mask is not None else None

        rope = self.rope(x, fps=fps)
        extra_pos = self.learnable_pos_embed(x) if self.learnable_pos_embed is not None else None

        pt, ph, pw = c.patch
        nt, nh, nw = T // pt, H // ph, W // pw
        x = self.patch_embed(x).flatten(1, 3)                       # [B, THW, D]

        if timestep.ndim == 1:
            temb, e = self.time_embed(x, timestep)                  # [B, 3D], [B, D]
        elif timestep.ndim == 5:                                    # 프레임별 t [B, 1, T, 1, 1] (p_t = 1 가정)
            assert timestep.shape == (B, 1, T, 1, 1)
            temb, e = self.time_embed(x, timestep.flatten())
            temb, e = (t.view(B, nt, 1, 1, -1).expand(-1, -1, nh, nw, -1).flatten(1, 3) for t in (temb, e))
        else:
            raise ValueError(f"timestep shape {tuple(timestep.shape)}")

        for b in self.transformer_blocks:
            x = b(x, encoder_hidden_states, e, temb, rope, extra_pos, mask)

        x = self.proj_out(self.norm_out(x, e, temb))
        # unpatchify: 채널 묶음 순서가 (ph, pw, pt, C) 라 patchify 의 역순이 아님 (원본 그대로)
        x = x.unflatten(2, (ph, pw, pt, -1)).unflatten(1, (nt, nh, nw))
        x = x.permute(0, 7, 1, 6, 2, 4, 3, 5)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


# ---------------------------------------------------------------- 전체

class Anima(nn.Module):
    """LLM adapter → Cosmos DiT. forward 인자는 파이프라인의 한 denoise step 과 같다."""

    def __init__(self, c: AnimaConfig):
        super().__init__()
        self.config = c
        self.text_conditioner = TextConditioner(c.adapter)
        self.transformer = CosmosDiT(c.dit)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def encode_text(self, source_hidden_states, target_input_ids, target_attention_mask=None,
                    source_attention_mask=None):
        return self.text_conditioner(source_hidden_states, target_input_ids,
                                     target_attention_mask, source_attention_mask)

    def forward(self, hidden_states, timestep, source_hidden_states, target_input_ids,
                target_attention_mask=None, source_attention_mask=None, padding_mask=None,
                attention_mask=None, fps=None, condition_mask=None):
        text = self.encode_text(source_hidden_states, target_input_ids, target_attention_mask, source_attention_mask)
        return self.transformer(hidden_states, timestep, text, attention_mask=attention_mask, fps=fps,
                                condition_mask=condition_mask, padding_mask=padding_mask)


# ---------------------------------------------------------------- 검증

def config_from_hf(cfg, adapter_cfg=None) -> AnimaConfig:
    """cfg = CosmosTransformer3DModel config (dict / FrozenDict), adapter_cfg = AnimaTextConditioner config."""
    c = dict(cfg)
    assert not c.get("img_context_dim_in"), "Cosmos 2.5 이미지 context 경로는 미구현 (Anima 는 None)"
    assert not c.get("use_crossattn_projection", False), "crossattn_proj 미구현 (Anima 는 False)"
    assert c.get("extra_pos_embed_type") in (None, "learnable")
    dit = CosmosConfig(
        in_ch=c["in_channels"], out_ch=c["out_channels"], n_head=c["num_attention_heads"],
        head_dim=c["attention_head_dim"], n_layer=c["num_layers"], mlp_ratio=c["mlp_ratio"],
        text_dim=c["text_embed_dim"], adaln_lora_dim=c["adaln_lora_dim"], max_size=tuple(c["max_size"]),
        patch=tuple(c["patch_size"]), rope_scale=tuple(c["rope_scale"]),
        concat_padding_mask=c["concat_padding_mask"], learnable_pos=c.get("extra_pos_embed_type") == "learnable")
    a = dict(adapter_cfg) if adapter_cfg is not None else {}
    adapter = AdapterConfig(
        source_dim=a.get("source_dim", 1024), target_dim=a.get("target_dim", dit.text_dim),
        model_dim=a.get("model_dim", 1024), n_layer=a.get("num_layers", 6), n_head=a.get("num_attention_heads", 16),
        mlp_ratio=a.get("mlp_ratio", 4.0), vocab_size=a.get("target_vocab_size", 32128),
        use_self_attention=a.get("use_self_attention", True), use_layer_norm=a.get("use_layer_norm", False),
        min_seq_len=a.get("min_sequence_length", 512))
    assert adapter.target_dim == dit.text_dim, "adapter 출력 = DiT text_embed_dim 이어야 한다"
    return AnimaConfig(dit=dit, adapter=adapter)


def verify_pair(ref) -> dict:
    """ref = (text_conditioner=AnimaTextConditioner, transformer=CosmosTransformer3DModel) 를 묶은 모듈. 이름이 같아 name_map 불필요."""
    cfg = config_from_hf(ref.transformer.config, ref.text_conditioner.config)
    mine = Anima(cfg)
    pairs = [("text_conditioner", "text_conditioner")]
    for i in range(cfg.adapter.n_layer):
        pairs.append((f"text_conditioner.blocks.{i}", f"text_conditioner.blocks.{i}"))
    pairs += [("transformer.rope", "transformer.rope"), ("transformer.patch_embed", "transformer.patch_embed"),
              ("transformer.time_embed", "transformer.time_embed")]
    if cfg.dit.learnable_pos:
        pairs.append(("transformer.learnable_pos_embed", "transformer.learnable_pos_embed"))
    for i in range(cfg.dit.n_layer):
        p = f"transformer.transformer_blocks.{i}"
        pairs += [(f"{p}.attn1", f"{p}.attn1"), (f"{p}.attn2", f"{p}.attn2"), (f"{p}.ff", f"{p}.ff"), (p, p)]
    pairs.append(("transformer.norm_out", "transformer.norm_out"))
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
