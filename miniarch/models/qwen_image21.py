"""C2 · Qwen-Image-2.1: Single-Stream DiT (텍스트 + 조건 이미지 + 타깃 이미지를 한 줄로) + 편집 조건.

원본: diffusers `QwenImage21Transformer2DModel` (transformer_qwenimage21.py, v0.41.0).
파라미터 이름은 diffusers 와 같다 (name_map 불필요). VL 인코더(Qwen3-VL)는 구현하지 않고 그 hidden state 를 입력으로 받는다.

한 줄 시퀀스 만들기
- VL 인코더 시퀀스(img_mask)에서 이미지 칸 하나 = latent 2×2 = 토큰 4개. 이미지 칸을 4배로 늘리고 그 자리에
  latent(img_in 통과)를 끼워 넣는다. 조건 이미지(편집용 참조 이미지)는 텍스트 사이 원래 자리에, 타깃 이미지는 맨 끝에.
- 텍스트: txt_in = zero-center RMSNorm(가중치+1) → Linear → GELU(tanh) → Linear.
- 3축 RoPE (frame, height, width): 텍스트는 세 축 모두 같은 위치로 한 칸씩 전진, 이미지 블록은 frame 을 고정하고
  h/w 를 0 중심 격자로 둔다 (음수 위치 사용). 블록 하나가 끝나면 위치를 max(h, w) 만큼 건너뛴다. 복소수 interleaved 쌍 회전.

block-causal attention
- 허용 = (key 위치 ≤ query 위치) 또는 (같은 이미지 블록). 즉 텍스트는 causal, 이미지 블록 안은 양방향,
  블록끼리는 앞쪽만 본다. 타깃 이미지는 맨 끝이라 전부를 본다. padding 텍스트는 key 에서만 빠진다.
- 원본은 segment 별 SDPA 를 여러 번 부르지만 여기서는 같은 마스크를 한 번에 만든 명시적 softmax 로 계산한다.

조건(modulation)
- timestep → 정현파(×1000, [cos|sin]) → MLP → temb. 모델 전체가 Linear 하나(`modulation`)를 공유해
  [mod1.scale, mod1.gate, mod2.scale, mod2.gate] 를 만든다 (shift 없음). 게이트는 tanh 를 거친다.
- causal_condition: 텍스트·조건 이미지 토큰은 t=0 행으로, 타깃 토큰만 실제 t 행으로 modulation
  → prefix 출력이 timestep·타깃과 무관해져 KV cache 가 가능하다.
- 마지막: LayerNorm × (1 + scale) (scale 만, 같은 행 선택) → proj_out. 출력은 결합 시퀀스 전체 [B, N, C].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IMG_TOKENS_PER_SLOT = 4   # VL 이미지 칸 하나 = latent 2×2


@dataclass
class QwenImage21Config:
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = 64
    n_layer: int = 32
    n_head: int = 32
    head_dim: int = 128
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple = (16, 56, 56)     # frame, height, width (합 = head_dim)
    rope_theta: int = 10000
    eps: float = 1e-6
    t_freq_dim: int = 256
    time_factor: float = 1000.0
    causal_condition: bool = True            # 원본 config 값. False: 모든 토큰이 실제 t 로 modulation
    # ablation 스위치 (기본값 = 원본)
    block_causal: bool = True                # False: 전체 양방향 attention (padding key 만 제외)
    use_qk_norm: bool = True                 # False: q/k RMSNorm 생략
    use_rope: bool = True                    # False: 위치 정보 없음
    tanh_gate: bool = True                   # False: 게이트를 tanh 없이 그대로 곱함

    def __post_init__(self):
        self.out_channels = self.out_channels or self.in_channels
        self.axes_dims_rope = tuple(self.axes_dims_rope)
        assert sum(self.axes_dims_rope) == self.head_dim, "RoPE 축 차원 합 = head_dim"

    @property
    def d_model(self) -> int:
        return self.n_head * self.head_dim


# ---------------------------------------------------------------- 임베딩 / norm

class TemporalTimesteps(nn.Module):
    """[cos | sin] 정현파. t 에 time_factor(1000) 를 곱하고, 주파수 지수는 / half (half-1 아님)."""

    def __init__(self, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.dim, self.time_factor = dim, time_factor
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        t = self.time_factor * t.float()
        args = t[:, None] * self.freqs[None].to(t.device)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb.to(t.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, d, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class TimeEmbed(nn.Module):
    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.time_proj = TemporalTimesteps(c.t_freq_dim, time_factor=c.time_factor)
        self.timestep_embedder = TimestepEmbedding(c.t_freq_dim, c.d_model)

    def forward(self, t, hidden_states):
        return self.timestep_embedder(self.time_proj(t).to(dtype=hidden_states.dtype))


class ZeroCenterRMSNorm(nn.Module):
    """저장된 가중치 w 에 대해 실제 scale = w + 1 (fp32 계산)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        r = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * r * (self.weight.float() + 1)).to(dt)


class RMSNorm(nn.Module):
    """diffusers RMSNorm (q/k norm, head_dim 단위)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class TextProjection(nn.Module):
    """VL 인코더 hidden → 모델 폭 (connector)."""

    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(c.context_in_dim, c.eps)
        self.in_layer = nn.Linear(c.context_in_dim, c.d_model, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):
        return self.out_layer(self.act(self.in_layer(self.text_norm(x))))


# ---------------------------------------------------------------- 3축 RoPE

class Rope3D(nn.Module):
    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.theta, self.axes = c.rope_theta, list(c.axes_dims_rope)

    def _polar(self, idx: torch.Tensor, dim: int) -> torch.Tensor:
        # 원본 표와 같은 식: outer(위치, 1/theta^(2i/dim)) → polar (CPU 에서 계산)
        inv = 1.0 / torch.pow(self.theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        f = torch.outer(idx, inv)
        return torch.polar(torch.ones_like(f), f)

    def positions(self, img_shapes, image_pad_mask):
        """[N] frame / height / width 위치 (long, CPU)."""
        frame, img_h, img_w = [], [], []
        cursor, pos = 0, 0
        total = image_pad_mask.shape[-1]
        is_img = image_pad_mask.tolist()
        for _, h, w in img_shapes:
            start = is_img.index(True, cursor)
            n_txt = start - cursor
            frame.extend(range(pos, pos + n_txt))            # 텍스트: 한 칸씩 전진
            pos += n_txt
            cursor = start + h * w
            frame.extend([pos] * (h * w))                     # 이미지: frame 고정
            pos += max(h, w)
            img_h.extend([y for y in range(-(h - h // 2), h // 2) for _ in range(w)])   # 0 중심 격자
            img_w.extend([x for _ in range(h) for x in range(-(w - w // 2), w // 2)])
        if cursor < total:
            frame.extend(range(pos, pos + total - cursor))
        frame = torch.tensor(frame, dtype=torch.long)
        hh, ww = frame.clone(), frame.clone()
        m = image_pad_mask.cpu()
        hh[m] = torch.tensor(img_h, dtype=torch.long)
        ww[m] = torch.tensor(img_w, dtype=torch.long)
        return frame, hh, ww

    def forward(self, img_shapes, image_pad_mask, device):
        idx = self.positions(img_shapes, image_pad_mask)
        out = torch.cat([self._polar(i, d) for i, d in zip(idx, self.axes)], dim=-1)   # [N, head_dim/2] complex
        return out.to(device)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x [B, N, H, D], freqs [N, D/2] complex. (x0,x1),(x2,x3).. 쌍을 복소수로 보고 곱한다."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs.unsqueeze(1)).flatten(3).type_as(x)


# ---------------------------------------------------------------- 블록

class Attention(nn.Module):
    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.c, self.heads = c, c.n_head
        d = c.d_model
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(d, d, bias=False), nn.Dropout(0.0)])
        self.norm_q = RMSNorm(c.head_dim, c.eps)
        self.norm_k = RMSNorm(c.head_dim, c.eps)
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, N, N] (텍스트·조건·타깃 전부, 막힌 칸은 0)

    def forward(self, x, rope, allow):
        q = self.to_q(x).unflatten(-1, (self.heads, -1))      # [B, N, H, D]
        k = self.to_k(x).unflatten(-1, (self.heads, -1))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        if self.c.use_qk_norm:
            q, k = self.norm_q(q).to(v.dtype), self.norm_k(k).to(v.dtype)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))      # [B, H, N, D]
        s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
        s = s.masked_fill(~allow[:, None], float("-inf"))
        p = self.attn_probs(s.float().softmax(dim=-1).to(q.dtype))
        o = (p @ v).transpose(1, 2).flatten(2, 3).type_as(q)
        return self.to_out[1](self.to_out[0](o))


class SwiGLU(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.proj = nn.Linear(d, inter, bias=False)
        self.out = nn.Linear(inter, d, bias=False)
        self.gate_layer = nn.Linear(d, inter, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        return self.out(self.activation_fn(self.gate_layer(x)) * self.proj(x))


def select_rows(params: torch.Tensor, target_mask: torch.Tensor | None) -> torch.Tensor:
    """params [B(+1), D] → [B, N, D]. causal_condition 이면 마지막 행(t=0)을 비타깃 토큰에, 앞 B 행을 타깃 토큰에."""
    if target_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_mask.view(1, -1, 1), real, zero)


class Block(nn.Module):
    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.c = c
        d = c.d_model
        self.img_norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.attn = Attention(c)
        self.img_norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.img_mlp = SwiGLU(d, d * c.mlp_ratio)
        # hook 지점: 토큰별로 고른 modulation [B, N, D] (gate 는 tanh 전)
        self.scale1_probe, self.gate1_probe = nn.Identity(), nn.Identity()
        self.scale2_probe, self.gate2_probe = nn.Identity(), nn.Identity()

    def _gate(self, g):
        return g.tanh() if self.c.tanh_gate else g

    def forward(self, x, modulation, rope, allow, rows):
        mod1, mod2 = modulation.chunk(2, dim=-1)              # 공유 modulation 을 둘로
        s1, g1 = mod1.chunk(2, dim=-1)
        s1, g1 = self.scale1_probe(select_rows(s1, rows)), self.gate1_probe(select_rows(g1, rows))
        x = x + self._gate(g1) * self.attn(self.img_norm1(x) * (1 + s1), rope, allow)
        s2, g2 = mod2.chunk(2, dim=-1)
        s2, g2 = self.scale2_probe(select_rows(s2, rows)), self.gate2_probe(select_rows(g2, rows))
        x = x + self._gate(g2) * self.img_mlp(self.img_norm2(x) * (1 + s2))
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class AdaNormOut(nn.Module):
    """최종 norm: scale 만 (shift 없음)."""

    def __init__(self, c: QwenImage21Config):
        super().__init__()
        d = c.d_model
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d, c.eps, elementwise_affine=False, bias=False)

    def forward(self, x, temb, rows):
        scale = select_rows(self.linear(self.silu(temb).to(x.dtype)), rows)
        return self.norm(x) * (1 + scale)


# ---------------------------------------------------------------- 모델

def build_token_metadata(image_pad_mask: torch.Tensor, img_shapes):
    """image_ids [N] (텍스트 -1, 이미지 블록마다 0,1,..), target_mask [N] (마지막 블록 = 타깃)."""
    pos = image_pad_mask.nonzero(as_tuple=True)[0]
    lens = [math.prod(s) for s in img_shapes]
    assert sum(lens) == pos.numel(), "img_shapes 토큰 수 ≠ image_pad_mask 이미지 칸 수"
    ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
    ids[pos] = torch.tensor([b for b, n in enumerate(lens) for _ in range(n)], device=ids.device)
    tgt = torch.zeros_like(image_pad_mask)
    tgt[pos[-lens[-1]:]] = True
    return ids, tgt


class QwenImage21(nn.Module):
    def __init__(self, c: QwenImage21Config):
        super().__init__()
        self.config = c
        d = c.d_model
        self.pos_embed = Rope3D(c)
        self.time_text_embed = TimeEmbed(c)
        self.txt_in = TextProjection(c)
        self.img_in = nn.Linear(c.in_channels * c.patch_size ** 2, d, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 4 * d, bias=False))   # 모든 블록 공유
        self.transformer_blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm_out = AdaNormOut(c)
        self.proj_out = nn.Linear(d, c.patch_size ** 2 * c.out_channels, bias=False)
        self.mask_probe = nn.Identity()   # hook 지점: attention 허용 마스크 [B, N, N] bool
        self.last_meta: dict = {}         # 분석용: image_ids, target_mask, key_valid

    def attention_mask(self, image_ids, key_valid, B):
        N = image_ids.shape[0]
        i = torch.arange(N, device=image_ids.device)
        if self.config.block_causal:
            same = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
            allow = (i[None, :] <= i[:, None]) | same
        else:
            allow = torch.ones(N, N, dtype=torch.bool, device=image_ids.device)
        allow = allow[None].expand(B, N, N)
        if key_valid is not None:
            allow = allow & key_valid[:, None, :]
        return self.mask_probe(allow)

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                encoder_hidden_states_mask=None):
        """hidden_states [B, 이미지 토큰(조건들 → 타깃), C_in], encoder_hidden_states [B, L, ctx],
        img_mask [B, L + 타깃칸] (VL 시퀀스의 이미지 칸), encoder_hidden_states_mask [B, L]. 반환 [B, N, C_out]."""
        c = self.config
        B = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        enc = self.txt_in(encoder_hidden_states)

        # 이미지 칸을 4배로 늘리고 그 자리에 latent 를 넣는다 (배치 전체가 같은 배치(layout))
        repeats = torch.where(img_mask, IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        n_target = math.prod(img_shapes[0][-1])
        joint = torch.cat([enc, enc.new_zeros(B, n_target // IMG_TOKENS_PER_SLOT, enc.shape[2])], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = hidden_states

        rope = self.pos_embed(img_shapes[0], image_pad_mask, hidden_states.device) if c.use_rope else None
        image_ids, target_mask = build_token_metadata(image_pad_mask, img_shapes[0])
        prefix_len = int((~target_mask).sum())
        assert bool(target_mask[prefix_len:].all()), "타깃 이미지는 시퀀스 맨 끝이어야 한다"

        t = timestep.to(hidden_states.dtype)
        if c.causal_condition:
            t = torch.cat([t, t.new_zeros(1)], dim=0)         # 마지막 행 = t=0 (prefix 토큰용)
            rows = target_mask
        else:
            rows = None
        temb = self.time_text_embed(t, hidden_states)
        modulation = self.modulation(temb)

        # padding 텍스트는 key 에서 제외. 결합 시퀀스의 텍스트 위치 = VL 시퀀스의 비이미지 위치 (순서대로)
        key_valid = None
        if encoder_hidden_states_mask is not None:
            key_valid = torch.ones(B, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device)
            txt_pos = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_txt = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            key_valid[:, txt_pos] = encoder_hidden_states_mask.bool()[:, vlm_txt]
        allow = self.attention_mask(image_ids, key_valid, B)
        self.last_meta = dict(image_ids=image_ids, target_mask=target_mask, key_valid=key_valid)

        for blk in self.transformer_blocks:
            joint = blk(joint, modulation, rope, allow, rows)
        return self.proj_out(self.norm_out(joint, temb, rows))


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> QwenImage21Config:
    """c = diffusers QwenImage21Transformer2DModel.config (transformer/config.json)"""
    return QwenImage21Config(
        patch_size=c.patch_size, in_channels=c.in_channels, out_channels=c.out_channels,
        n_layer=c.num_layers, n_head=c.num_attention_heads, head_dim=c.attention_head_dim,
        context_in_dim=c.context_in_dim, mlp_ratio=c.mlp_ratio, axes_dims_rope=tuple(c.axes_dims_rope),
        eps=c.eps, causal_condition=c.causal_condition)


def verify_pair(ref) -> dict:
    """ref = diffusers QwenImage21Transformer2DModel (이름이 같아 name_map 불필요)"""
    mine = QwenImage21(config_from_hf(ref.config))
    pairs = [(n, n) for n in ("img_in", "txt_in", "time_text_embed", "modulation")]
    for i in range(ref.config.num_layers):
        pairs += [(f"transformer_blocks.{i}.attn",) * 2, (f"transformer_blocks.{i}",) * 2]
    pairs += [("norm_out", "norm_out")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
