"""B2 · Krea-2 (12B single-stream DiT): 텍스트·이미지 토큰이 한 줄로 같은 블록 가중치를 공유.

원본: diffusers `Krea2Transformer2DModel` (transformer_krea2.py). 파라미터 이름이 원본과 같다 (name_map 불필요).

흐름
1. 텍스트 조건 = 텍스트 인코더의 여러 층(12개)에서 뽑은 hidden state 묶음 [B, L, 층수, D_txt].
   - text_fusion.layerwise_blocks: 토큰마다 "층 축"을 따라 attention (토큰끼리는 안 섞임, 마스크·RoPE 없음)
   - projector: Linear(층수 → 1) 로 층 축을 접음 → [B, L, D_txt]
   - refiner_blocks: 이제 토큰 축을 따라 attention (padding 토큰은 key 에서 제외, RoPE 없음)
   - txt_in: RMSNorm → Linear → GELU(tanh) → Linear 로 transformer 폭에 맞춤
2. 이미지: 2×2 patchify 된 latent → img_in (Linear). 시퀀스 = [텍스트 | 이미지] 를 이어 붙인 한 줄.
3. 시간 조건: cos-first 정현파(t×1000) → MLP = temb [B, 1, H].
   time_mod_proj(GELU(temb)) 로 6H 벡터 하나를 만들어 **모든 블록이 공유**, 블록마다 학습된 표(scale_shift_table)만 더함.
   순서: (prescale, preshift, pregate, postscale, postshift, postgate).
4. 블록 (pre-norm, zero-centered RMSNorm = x/rms·(1+w)):
   h += pregate · Attn((1+prescale)·norm1(h) + preshift)
   h += postgate · SwiGLU((1+postscale)·norm2(h) + postshift)
   Attn = GQA + q/k RMSNorm(head 별) + 3축 RoPE(t,h,w; 인접 쌍 회전) + sigmoid 출력 gate.
   텍스트 토큰의 RoPE 좌표는 전부 0 → 텍스트는 위치 없는 "집합" 처럼 취급된다.
5. 출력: 이미지 토큰만 잘라 final_layer: (1+scale)·norm(h) + shift (modulation 은 temb + 표, GELU/proj 없이) → Linear.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Krea2Config:
    in_channels: int = 64             # vae 채널 × patch²
    n_layer: int = 28
    head_dim: int = 128
    n_head: int = 48
    n_kv_head: int = 12
    d_ff: int = 16384
    t_freq_dim: int = 256
    text_dim: int = 2560
    n_text_layers: int = 12           # 텍스트 인코더에서 뽑아 쌓은 층 수
    text_n_head: int = 20
    text_n_kv_head: int = 20
    text_d_ff: int = 6912
    n_layerwise: int = 2
    n_refiner: int = 2
    axes_dims_rope: tuple = (32, 48, 48)
    rope_theta: float = 1000.0
    norm_eps: float = 1e-5
    # ablation 스위치 (기본값 = 원본)
    use_layerwise_fusion: bool = True   # False: 층 축 attention 생략 (projector 로 바로 접음)
    use_text_refiner: bool = True       # False: 토큰 축 refiner 생략
    use_rope: bool = True               # False: 본 블록에서 위치 정보 없음
    use_qk_norm: bool = True            # False: q/k RMSNorm 생략
    use_attn_gate: bool = True          # False: sigmoid 출력 gate 생략
    use_block_table: bool = True        # False: 블록별 scale_shift_table 무시 (공유 modulation 만)
    use_text_mask: bool = True          # False: padding 토큰도 key 로 봄

    @property
    def d_model(self) -> int:
        return self.head_dim * self.n_head


# ---------------------------------------------------------------- 부품

class Krea2RMSNorm(nn.Module):
    """zero-centered RMSNorm: 실효 배율 = 1 + weight. fp32 에서 계산."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        return F.rms_norm(x.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps).to(dtype)


def rope_cos_sin(ids: torch.Tensor, axes_dim, theta: float):
    """ids [S, 3] → cos, sin [S, head_dim]. 축마다 dim 만큼, 인접 두 채널이 같은 주파수 (repeat_interleave)."""
    pos = ids.float()
    fdt = torch.float32 if ids.device.type in ("mps", "npu", "neuron") else torch.float64
    cos_out, sin_out = [], []
    for i in range(ids.shape[-1]):
        d = axes_dim[i]
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=fdt, device=ids.device) / d))
        f = torch.outer(pos[:, i], inv)
        cos_out.append(f.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(f.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


class Krea2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: float, axes_dim):
        super().__init__()
        self.theta, self.axes_dim = theta, list(axes_dim)

    def forward(self, ids):
        return rope_cos_sin(ids, self.axes_dim, self.theta)


def apply_rope(x, cos, sin):
    """x [B, S, H, D]. (x0, x1) 쌍을 회전: (x0, x1) → (x0·c − x1·s, x1·c + x0·s). fp32 로 계산."""
    re, im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rot = torch.stack([-im, re], dim=-1).flatten(3)
    cos, sin = cos[None, :, None, :].to(x.device), sin[None, :, None, :].to(x.device)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class Krea2Attention(nn.Module):
    """GQA + q/k RMSNorm + (선택) RoPE + sigmoid 출력 gate. bias 없음."""

    def __init__(self, dim: int, n_head: int, n_kv_head: int, eps: float, c: Krea2Config):
        super().__init__()
        self.h, self.kvh, self.hd = n_head, n_kv_head, dim // n_head
        self.c = c
        self.to_q = nn.Linear(dim, self.hd * n_head, bias=False)
        self.to_k = nn.Linear(dim, self.hd * n_kv_head, bias=False)
        self.to_v = nn.Linear(dim, self.hd * n_kv_head, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)
        self.norm_q = Krea2RMSNorm(self.hd, eps)
        self.norm_k = Krea2RMSNorm(self.hd, eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, Sq, Sk] (양방향, padding key 는 0)
        self.gate_probe = nn.Identity()   # hook 지점: sigmoid 출력 gate [B, S, dim]

    def forward(self, x, mask=None, rope=None):
        q = self.to_q(x).unflatten(-1, (self.h, self.hd))          # [B, S, H, hd]
        k = self.to_k(x).unflatten(-1, (self.kvh, self.hd))
        v = self.to_v(x).unflatten(-1, (self.kvh, self.hd))
        gate = self.to_gate(x)
        if self.c.use_qk_norm:
            q, k = self.norm_q(q), self.norm_k(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        g = self.h // self.kvh
        if g > 1:  # 원본도 kv head 를 반복 (enable_gqa 대신)
            k, v = k.repeat_interleave(g, dim=2), v.repeat_interleave(g, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))            # [B, H, S, hd]
        scores = (q @ k.transpose(-2, -1)) * self.hd ** -0.5
        if mask is not None:  # SDPA 와 같은 의미: bool 은 True=볼 수 있음, float 은 더함
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        probs = self.attn_probs(scores.float().softmax(-1).to(q.dtype))
        y = (probs @ v).transpose(1, 2).flatten(2)
        if self.c.use_attn_gate:
            y = y * self.gate_probe(torch.sigmoid(gate))
        return self.to_out[0](y)


class Krea2SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------- 텍스트 융합

class Krea2TextFusionBlock(nn.Module):
    """시간 조건·RoPE 없는 평범한 pre-norm 블록."""

    def __init__(self, c: Krea2Config):
        super().__init__()
        d = c.text_dim
        self.norm1 = Krea2RMSNorm(d, c.norm_eps)
        self.norm2 = Krea2RMSNorm(d, c.norm_eps)
        self.attn = Krea2Attention(d, c.text_n_head, c.text_n_kv_head, c.norm_eps, c)
        self.ff = Krea2SwiGLU(d, c.text_d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class Krea2TextFusion(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.c = c
        self.layerwise_blocks = nn.ModuleList(Krea2TextFusionBlock(c) for _ in range(c.n_layerwise))
        self.projector = nn.Linear(c.n_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(Krea2TextFusionBlock(c) for _ in range(c.n_refiner))

    def forward(self, e, mask=None):
        B, L, NL, D = e.shape
        h = e.reshape(B * L, NL, D)              # 토큰마다 층 축을 시퀀스로
        if self.c.use_layerwise_fusion:
            for blk in self.layerwise_blocks:
                h = blk(h.contiguous())
        h = self.projector(h.reshape(B, L, NL, D).permute(0, 1, 3, 2)).squeeze(-1)   # [B, L, D]
        if self.c.use_text_refiner:
            for blk in self.refiner_blocks:
                h = blk(h, mask)
        return h


class Krea2TextProjection(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.norm = Krea2RMSNorm(c.text_dim, c.norm_eps)
        self.linear_1 = nn.Linear(c.text_dim, c.d_model)
        self.linear_2 = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(self.norm(x)), approximate="tanh"))


class Krea2TimestepEmbedding(nn.Module):
    """[cos | sin](t·1000·freq), freq = exp(−ln(1e4)·i/half) → Linear → GELU(tanh) → Linear. 출력 [B, 1, H]."""

    def __init__(self, c: Krea2Config):
        super().__init__()
        self.dim = c.t_freq_dim
        self.linear_1 = nn.Linear(c.t_freq_dim, c.d_model)
        self.linear_2 = nn.Linear(c.d_model, c.d_model)

    def forward(self, t, dtype):
        half = self.dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (t.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


# ---------------------------------------------------------------- 본 블록

class Krea2TransformerBlock(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        d = c.d_model
        self.c = c
        self.scale_shift_table = nn.Parameter(torch.zeros(6, d))
        self.norm1 = Krea2RMSNorm(d, c.norm_eps)
        self.norm2 = Krea2RMSNorm(d, c.norm_eps)
        self.attn = Krea2Attention(d, c.n_head, c.n_kv_head, c.norm_eps, c)
        self.ff = Krea2SwiGLU(d, c.d_ff)
        self.mod_probe = nn.Identity()   # hook 지점: [B, 1, 6, H] = (prescale, preshift, pregate, postscale, postshift, postgate)

    def forward(self, h, temb_mod, rope, mask=None):
        mod = temb_mod.unflatten(-1, (6, -1))
        if self.c.use_block_table:
            mod = mod + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod_probe(mod).unbind(-2)
        h = h + pregate * self.attn((1.0 + prescale) * self.norm1(h) + preshift, mask, rope)
        return h + postgate * self.ff((1.0 + postscale) * self.norm2(h) + postshift)


class Krea2FinalLayer(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, c.d_model))
        self.norm = Krea2RMSNorm(c.d_model, c.norm_eps)
        self.linear = nn.Linear(c.d_model, c.in_channels)
        self.mod_probe = nn.Identity()   # hook 지점: [B, 2, H] = (scale, shift)

    def forward(self, h, temb):
        scale, shift = self.mod_probe(temb + self.scale_shift_table).chunk(2, dim=1)
        return self.linear((1.0 + scale) * self.norm(h) + shift)


class Krea2(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        assert sum(c.axes_dims_rope) == c.head_dim
        self.config = c
        d = c.d_model
        self.img_in = nn.Linear(c.in_channels, d)
        self.time_embed = Krea2TimestepEmbedding(c)
        self.time_mod_proj = nn.Linear(d, 6 * d)
        self.text_fusion = Krea2TextFusion(c)
        self.txt_in = Krea2TextProjection(c)
        self.rotary_emb = Krea2RotaryPosEmbed(c.rope_theta, c.axes_dims_rope)
        self.transformer_blocks = nn.ModuleList(Krea2TransformerBlock(c) for _ in range(c.n_layer))
        self.final_layer = Krea2FinalLayer(c)

    def forward(self, hidden_states, encoder_hidden_states, timestep, position_ids, encoder_attention_mask=None):
        """hidden_states [B, N_img, C] (patchify 된 latent), encoder_hidden_states [B, L, 층수, D_txt],
        timestep [B] ∈ [0, 1], position_ids [L + N_img, 3] (텍스트 행 = 0), mask [B, L] bool → 속도 [B, N_img, C]."""
        c = self.config
        B, N, _ = hidden_states.shape
        L = encoder_hidden_states.shape[1]
        temb = self.time_embed(timestep, hidden_states.dtype)                   # [B, 1, H]
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))         # [B, 1, 6H] 모든 블록 공유

        txt_mask = mask = None
        if encoder_attention_mask is not None and c.use_text_mask:              # key padding mask [B, 1, 1, S]
            txt_mask = encoder_attention_mask[:, None, None, :]
            ones = encoder_attention_mask.new_ones((B, N))
            mask = torch.cat([encoder_attention_mask, ones], dim=1)[:, None, None, :]

        e = self.txt_in(self.text_fusion(encoder_hidden_states, txt_mask))
        h = torch.cat([e, self.img_in(hidden_states)], dim=1)                   # [텍스트 | 이미지]
        rope = self.rotary_emb(position_ids) if c.use_rope else None
        for blk in self.transformer_blocks:
            h = blk(h, temb_mod, rope, mask)
        return self.final_layer(h[:, L:], temb)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> Krea2Config:
    """c = diffusers config (FrozenDict) 또는 transformer/config.json dict."""
    return Krea2Config(
        in_channels=c["in_channels"], n_layer=c["num_layers"], head_dim=c["attention_head_dim"],
        n_head=c["num_attention_heads"], n_kv_head=c["num_key_value_heads"], d_ff=c["intermediate_size"],
        t_freq_dim=c["timestep_embed_dim"], text_dim=c["text_hidden_dim"], n_text_layers=c["num_text_layers"],
        text_n_head=c["text_num_attention_heads"], text_n_kv_head=c["text_num_key_value_heads"],
        text_d_ff=c["text_intermediate_size"], n_layerwise=c["num_layerwise_text_blocks"],
        n_refiner=c["num_refiner_text_blocks"], axes_dims_rope=tuple(c["axes_dims_rope"]),
        rope_theta=float(c["rope_theta"]), norm_eps=float(c["norm_eps"]))


def verify_pair(ref) -> dict:
    """ref = diffusers Krea2Transformer2DModel (이름이 같아 name_map 불필요, 입력 이름도 같음)"""
    c = config_from_hf(ref.config)
    mine = Krea2(c)
    pairs = [("time_embed", "time_embed"), ("time_mod_proj", "time_mod_proj"), ("rotary_emb", "rotary_emb")]
    pairs += [(f"text_fusion.layerwise_blocks.{i}", f"text_fusion.layerwise_blocks.{i}") for i in range(c.n_layerwise)]
    pairs += [(f"text_fusion.refiner_blocks.{i}", f"text_fusion.refiner_blocks.{i}") for i in range(c.n_refiner)]
    pairs += [("text_fusion", "text_fusion"), ("txt_in", "txt_in"), ("img_in", "img_in")]
    for i in range(c.n_layer):
        pairs += [(f"transformer_blocks.{i}.attn", f"transformer_blocks.{i}.attn"),
                  (f"transformer_blocks.{i}", f"transformer_blocks.{i}")]
    pairs += [("final_layer", "final_layer")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
