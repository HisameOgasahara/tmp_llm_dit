"""B1 · Qwen-Image: MM-DiT (dual stream) 이미지 생성 transformer.

원본: diffusers `QwenImageTransformer2DModel` (src/diffusers/models/transformers/transformer_qwenimage.py, v0.41.0).
Qwen-Image / 2512 (생성), Qwen-Image-Edit-2511 (zero_cond_t), Qwen-Image-Layered (layer3d RoPE + 추가 t 조건)이
모두 이 클래스를 쓴다. 파라미터 이름은 원본 state_dict 와 같다. VAE·텍스트 인코더(Qwen2.5-VL)는 제외:
입력은 이미 2×2 patch 로 묶인 latent [B, N, in_ch] 와 텍스트 hidden [B, T, joint_attention_dim].

블록 구성 (dual stream = 이미지·텍스트가 가중치를 따로 갖고 attention 만 함께)
1. 조건: timestep → 사인 임베딩(256, ×1000, [cos|sin]) → MLP → temb. 블록마다 img_mod / txt_mod 가
   SiLU(temb) → Linear(6d) 로 [shift1, scale1, gate1, shift2, scale2, gate2] 를 만든다 (adaLN).
2. 각 stream: LayerNorm(affine 없음) → x·(1+scale)+shift → q/k/v (stream 별 Linear) → head 별 RMSNorm(QK-norm)
   → RoPE → [텍스트, 이미지] 순서로 이어 붙여 하나의 양방향 attention (텍스트 padding 은 key 에서만 가림)
   → 다시 나눠 stream 별 출력 Linear → gate1 곱해 residual. MLP(GELU-tanh, 4배) 도 같은 방식 + gate2.
3. MSRoPE (3축 RoPE): head_dim 을 [frame | 높이 | 너비] 채널로 나눈다 (원본 16/56/56). 인접 두 채널이 한 쌍(복소수).
   - 이미지: 높이·너비 위치를 가운데 0 기준으로 (h=3 → -2,-1,0). 여러 장이면 i번째 이미지의 frame 위치 = i.
   - 텍스트: 세 축 모두 같은 위치 m, m+1, ... (m = 이미지들의 max(h//2, w//2)) → 이미지 대각선 뒤에 놓인 셈.
   - Layered: 마지막 이미지(조건 이미지)의 frame 위치 = -1, m 은 layer 개수 이상.
4. 편집(zero_cond_t, Edit-2511): batch 를 [t, 0] 두 벌로 만들어, 첫 이미지(생성 대상) 토큰은 t 의 modulation,
   나머지(참조 이미지) 토큰은 t=0 의 modulation 을 쓴다. 텍스트·최종층은 t 쪽만.
5. 최종: AdaLayerNormContinuous ([scale, shift] 순서!) → proj_out (patch·patch·out_ch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class QwenImageConfig:
    patch_size: int = 2
    in_channels: int = 64            # = 16 latent 채널 × 2×2 patch
    out_channels: int = 16
    num_layers: int = 60
    n_head: int = 24
    head_dim: int = 128
    joint_attention_dim: int = 3584  # 텍스트 인코더 hidden 크기
    axes_dims_rope: tuple = (16, 56, 56)
    rope_theta: int = 10000
    zero_cond_t: bool = False        # Edit-2511: 참조 이미지 토큰은 t=0 modulation
    use_additional_t_cond: bool = False   # Layered: Embedding(2) 를 temb 에 더함
    use_layer3d_rope: bool = False        # Layered: 마지막 이미지 frame 위치 = -1
    eps: float = 1e-6
    mlp_ratio: int = 4
    # ablation 스위치 (기본값 = 원본)
    use_qk_norm: bool = True         # False: head 별 RMSNorm 생략
    use_rope: bool = True            # False: 위치 정보 없음
    center_rope: bool = True         # False: 이미지 위치를 0 부터 (원본 scale_rope=False 경로)
    use_text_mask: bool = True       # False: 텍스트 padding 도 key 로 봄
    joint_attention: bool = True     # False: 이미지는 이미지끼리, 텍스트는 텍스트끼리만 (교차 정보 없음)
    use_txt_norm: bool = True        # False: 텍스트 입력 RMSNorm 생략

    @property
    def d_model(self) -> int:
        return self.n_head * self.head_dim


# ---------------------------------------------------------------- 기본 부품

class RMSNorm(nn.Module):
    """diffusers RMSNorm 과 같은 계산 순서 (weight 가 fp32 면 출력도 fp32 로 남음)."""

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


def timestep_proj(t: torch.Tensor, dim: int = 256, scale: float = 1000.0) -> torch.Tensor:
    """Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000) → [cos | sin]."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch, d):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, d)
        self.linear_2 = nn.Linear(d, d)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, d, use_additional_t_cond=False):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, d)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, d)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        emb = self.timestep_embedder(timestep_proj(timestep).to(dtype=hidden_states.dtype))
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("use_additional_t_cond=True 이면 additional_t_cond 가 필요")
            emb = emb + self.addition_t_embedding(addition_t_cond).to(dtype=hidden_states.dtype)
        return emb


# ---------------------------------------------------------------- MSRoPE

def _axis_freqs(pos: torch.Tensor, dim: int, theta) -> torch.Tensor:
    """정수 위치 [L] → 복소 회전 e^{i·pos·ω} [L, dim/2] (원본 rope_params 와 같은 식)."""
    inv = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    f = torch.outer(pos, inv)
    return torch.polar(torch.ones_like(f), f)


class QwenEmbedRope(nn.Module):
    """img_shapes → (이미지 freqs [N, D/2], 텍스트 freqs [T, D/2]) 복소수. 파라미터 없음.

    원본은 위치 0..4095 / -4096..-1 표를 미리 만들어 잘라 쓰지만, 같은 정수 위치를 직접 계산한다 (CPU 에서 → device).
    """

    def __init__(self, c: QwenImageConfig):
        super().__init__()
        self.c = c

    def _pos(self, n):
        if self.c.center_rope:
            return torch.arange(-(n - n // 2), n // 2)   # 예: 3 → [-2, -1, 0], 4 → [-2, -1, 0, 1]
        return torch.arange(n)

    def _image(self, frame, h, w, frame_pos):
        a0, a1, a2 = self.c.axes_dims_rope
        th = self.c.rope_theta
        ff = _axis_freqs(frame_pos, a0, th).view(frame, 1, 1, -1).expand(frame, h, w, -1)
        fh = _axis_freqs(self._pos(h), a1, th).view(1, h, 1, -1).expand(frame, h, w, -1)
        fw = _axis_freqs(self._pos(w), a2, th).view(1, 1, w, -1).expand(frame, h, w, -1)
        return torch.cat([ff, fh, fw], dim=-1).reshape(frame * h * w, -1)

    def forward(self, img_shapes, txt_len: int, device=None):
        c = self.c
        shapes = img_shapes[0] if isinstance(img_shapes, list) else img_shapes   # batch 는 첫 샘플 기준 (원본과 같음)
        if not isinstance(shapes, list):
            shapes = [shapes]
        last = len(shapes) - 1
        vid, m = [], 0
        for i, (frame, h, w) in enumerate(shapes):
            if c.use_layer3d_rope and i == last:      # Layered 조건 이미지: frame 위치 -1
                assert frame == 1
                fpos = torch.tensor([-1])
            else:
                fpos = torch.arange(i, i + frame)
            vid.append(self._image(frame, h, w, fpos))
            m = max(h // 2, w // 2, m) if c.center_rope else max(h, w, m)
        if c.use_layer3d_rope:
            m = max(m, last)
        tpos = torch.arange(m, m + int(txt_len))
        a0, a1, a2 = c.axes_dims_rope
        txt = torch.cat([_axis_freqs(tpos, a, c.rope_theta) for a in (a0, a1, a2)], dim=1)
        return torch.cat(vid, dim=0).to(device), txt.to(device)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x [B, S, H, D], freqs 복소 [S, D/2]. 인접 채널 (2k, 2k+1) 을 한 복소수로 회전."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs.unsqueeze(1)).flatten(3).type_as(x)


# ---------------------------------------------------------------- 블록

class JointAttention(nn.Module):
    """이미지(to_*) · 텍스트(add_*) 가 따로 q/k/v 를 만들고, [텍스트, 이미지] 로 이어 한 번에 attention."""

    def __init__(self, c: QwenImageConfig):
        super().__init__()
        d, hd = c.d_model, c.head_dim
        self.c = c
        self.to_q, self.to_k, self.to_v = (nn.Linear(d, d) for _ in range(3))
        self.add_q_proj, self.add_k_proj, self.add_v_proj = (nn.Linear(d, d) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(d, d)])     # 원본 to_out = [Linear, Dropout]
        self.to_add_out = nn.Linear(d, d)
        self.norm_q, self.norm_k = RMSNorm(hd, c.eps), RMSNorm(hd, c.eps)
        self.norm_added_q, self.norm_added_k = RMSNorm(hd, c.eps), RMSNorm(hd, c.eps)
        self.attn_probs = nn.Identity()  # hook 지점: [B, H, T+N, T+N], 행·열 순서 = [텍스트 T | 이미지 N]

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask=None, image_rotary_emb=None):
        c, hd = self.c, self.c.head_dim
        B, N, _ = hidden_states.shape
        T = encoder_hidden_states.shape[1]
        iq, ik, iv = (f(hidden_states).unflatten(-1, (-1, hd)) for f in (self.to_q, self.to_k, self.to_v))
        tq, tk, tv = (f(encoder_hidden_states).unflatten(-1, (-1, hd))
                      for f in (self.add_q_proj, self.add_k_proj, self.add_v_proj))
        if c.use_qk_norm:
            iq, ik = self.norm_q(iq), self.norm_k(ik)
            tq, tk = self.norm_added_q(tq), self.norm_added_k(tk)
        if c.use_rope and image_rotary_emb is not None:
            img_f, txt_f = image_rotary_emb
            iq, ik = apply_rope(iq, img_f), apply_rope(ik, img_f)
            tq, tk = apply_rope(tq, txt_f), apply_rope(tk, txt_f)
        q, k, v = (torch.cat([t, i], dim=1).transpose(1, 2) for t, i in ((tq, iq), (tk, ik), (tv, iv)))  # [B,H,S,D]

        scores = (q @ k.transpose(-2, -1)) * hd ** -0.5
        keep = None
        if encoder_hidden_states_mask is not None:   # key 쪽 텍스트 padding 만 가림 (이미지는 항상 보임)
            img_ok = torch.ones(B, N, dtype=torch.bool, device=q.device)
            keep = torch.cat([encoder_hidden_states_mask.to(torch.bool), img_ok], dim=1)[:, None, None, :]
        if not c.joint_attention:                   # ablation: 같은 stream 끼리만
            is_txt = torch.arange(T + N, device=q.device) < T
            same = (is_txt[:, None] == is_txt[None, :])[None, None]
            keep = same if keep is None else keep & same
        if keep is not None:
            scores = scores.masked_fill(~keep, float("-inf"))
        probs = scores.float().softmax(dim=-1)
        if not c.joint_attention:
            probs = probs.nan_to_num(0.0)           # 볼 key 가 하나도 없는 텍스트 행
        probs = self.attn_probs(probs.to(q.dtype))
        out = (probs @ v).transpose(1, 2).flatten(2).to(q.dtype)
        txt_o, img_o = out[:, :T], out[:, T:]
        return self.to_out[0](img_o.contiguous()), self.to_add_out(txt_o.contiguous())


class GeluProj(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    """원본 FeedForward(activation_fn="gelu-approximate"): net = [GELU(proj), Dropout, Linear]."""

    def __init__(self, d, mult=4):
        super().__init__()
        self.net = nn.ModuleList([GeluProj(d, d * mult), nn.Dropout(0.0), nn.Linear(d * mult, d)])

    def forward(self, x):
        for f in self.net:
            x = f(x)
        return x


def modulate(x, mod, index=None):
    """mod [B', 3d] → (x·(1+scale)+shift, gate). index [B, L] 가 있으면 토큰별로 앞 절반(0)/뒤 절반(1) 선택."""
    shift, scale, gate = mod.chunk(3, dim=-1)
    if index is not None:
        b = shift.size(0) // 2
        sel = index.unsqueeze(-1) == 0

        def pick(a):
            return torch.where(sel, a[:b].unsqueeze(1), a[b:].unsqueeze(1))
        shift, scale, gate = pick(shift), pick(scale), pick(gate)
    else:
        shift, scale, gate = shift.unsqueeze(1), scale.unsqueeze(1), gate.unsqueeze(1)
    return x * (1 + scale) + shift, gate


class QwenImageBlock(nn.Module):
    def __init__(self, c: QwenImageConfig):
        super().__init__()
        d = c.d_model
        self.zero_cond_t = c.zero_cond_t
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.img_norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.attn = JointAttention(c)
        self.img_norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.img_mlp = FeedForward(d, c.mlp_ratio)
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.txt_norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.txt_norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=c.eps)
        self.txt_mlp = FeedForward(d, c.mlp_ratio)
        # hook 지점: [B 또는 2B, 6d] = [shift1 | scale1 | gate1 | shift2 | scale2 | gate2] (1 = attn, 2 = MLP)
        self.img_mod_probe = nn.Identity()
        self.txt_mod_probe = nn.Identity()

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb,
                image_rotary_emb=None, modulate_index=None):
        img_mod = self.img_mod_probe(self.img_mod(temb))
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod = self.txt_mod_probe(self.txt_mod(temb))
        img_mod1, img_mod2 = img_mod.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod.chunk(2, dim=-1)

        img_m, img_g1 = modulate(self.img_norm1(hidden_states), img_mod1, modulate_index)
        txt_m, txt_g1 = modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_a, txt_a = self.attn(img_m, txt_m, encoder_hidden_states_mask, image_rotary_emb)
        hidden_states = hidden_states + img_g1 * img_a
        encoder_hidden_states = encoder_hidden_states + txt_g1 * txt_a

        img_m2, img_g2 = modulate(self.img_norm2(hidden_states), img_mod2, modulate_index)
        hidden_states = hidden_states + img_g2 * self.img_mlp(img_m2)
        txt_m2, txt_g2 = modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_g2 * self.txt_mlp(txt_m2)

        if encoder_hidden_states.dtype == torch.float16:   # fp16 overflow 방지 (원본과 같음)
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states        # 원본과 같은 순서 (텍스트, 이미지)


class AdaLayerNormContinuous(nn.Module):
    """최종층: LayerNorm(affine 없음) 후 [scale, shift] (블록과 순서가 반대) 로 modulation."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d, eps=eps, elementwise_affine=False)

    def forward(self, x, cond):
        emb = self.linear(F.silu(cond).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class QwenImage(nn.Module):
    def __init__(self, c: QwenImageConfig):
        super().__init__()
        self.config = c
        d = c.d_model
        assert sum(c.axes_dims_rope) == c.head_dim
        self.pos_embed = QwenEmbedRope(c)
        self.time_text_embed = TimestepProjEmbeddings(d, c.use_additional_t_cond)
        self.txt_norm = RMSNorm(c.joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(c.in_channels, d)
        self.txt_in = nn.Linear(c.joint_attention_dim, d)
        self.transformer_blocks = nn.ModuleList(QwenImageBlock(c) for _ in range(c.num_layers))
        self.norm_out = AdaLayerNormContinuous(d, eps=1e-6)
        self.proj_out = nn.Linear(d, c.patch_size * c.patch_size * c.out_channels)

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask=None, timestep=None,
                img_shapes=None, additional_t_cond=None):
        """hidden_states [B, N, in_ch] (patch 로 묶인 latent), encoder_hidden_states [B, T, joint_dim],
        encoder_hidden_states_mask [B, T] (1 = 유효), timestep [B] (0~1), img_shapes [[(f, h, w), ...]] × B.
        반환: [B, N, patch·patch·out_ch] (velocity 예측)."""
        c = self.config
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        modulate_index = None
        if c.zero_cond_t:   # [t, 0] 두 벌. 샘플마다 첫 이미지 토큰 = 0번(t), 나머지 이미지 = 1번(t=0)
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(s[0]) + [1] * sum(prod(x) for x in s[1:]) for s in img_shapes],
                device=timestep.device, dtype=torch.int)

        if c.use_txt_norm:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        mask = None
        if encoder_hidden_states_mask is not None and c.use_text_mask:
            mask = encoder_hidden_states_mask.to(torch.bool)

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)
        rope = self.pos_embed(img_shapes, encoder_hidden_states.shape[1], device=hidden_states.device)

        for blk in self.transformer_blocks:
            encoder_hidden_states, hidden_states = blk(hidden_states, encoder_hidden_states, mask, temb,
                                                       rope, modulate_index)
        if c.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------- 검증

def config_from_hf(cfg) -> QwenImageConfig:
    """diffusers config (FrozenDict) 또는 transformer/config.json dict."""
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    assert not g("guidance_embeds", False), "guidance_embeds 경로는 원본에서도 쓰이지 않음 (미구현)"
    in_ch = g("in_channels", 64)
    return QwenImageConfig(
        patch_size=g("patch_size", 2), in_channels=in_ch, out_channels=g("out_channels", 16) or in_ch,
        num_layers=g("num_layers", 60), n_head=g("num_attention_heads", 24), head_dim=g("attention_head_dim", 128),
        joint_attention_dim=g("joint_attention_dim", 3584), axes_dims_rope=tuple(g("axes_dims_rope", (16, 56, 56))),
        zero_cond_t=bool(g("zero_cond_t", False)), use_additional_t_cond=bool(g("use_additional_t_cond", False)),
        use_layer3d_rope=bool(g("use_layer3d_rope", False)))


def verify_pair(ref) -> dict:
    """ref = diffusers QwenImageTransformer2DModel (이름이 같아 name_map 불필요, 입력 이름도 같음)"""
    c = config_from_hf(ref.config)
    mine = QwenImage(c)
    pairs = [("img_in", "img_in"), ("txt_in", "txt_in"), ("time_text_embed", "time_text_embed")]
    for i in range(c.num_layers):
        b = f"transformer_blocks.{i}"
        pairs += [(f"{b}.attn", f"{b}.attn"), (f"{b}.img_mlp", f"{b}.img_mlp"),
                  (f"{b}.txt_mlp", f"{b}.txt_mlp"), (b, b)]
    pairs.append(("norm_out", "norm_out"))
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
