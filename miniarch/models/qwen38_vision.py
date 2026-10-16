"""C0 · Qwen3.8-27B 내장 ViT + M-RoPE: 이미지·비디오를 LM 토큰으로 바꾸고 3축 위치를 붙인다.

HF 원본: transformers/models/qwen3_5 (Qwen3_5VisionModel + Qwen3_5Model.get_rope_index).
텍스트 쪽은 A6a 의 miniarch/models/qwen36.py 를 그대로 쓴다. 파라미터 이름은 HF 와 같다.

ViT (Qwen3_5VisionModel)
1. patch embed: 입력은 이미 잘린 patch 열 [N, C·tp·p·p]. 시간 tp 프레임 × p×p 픽셀을 Conv3d(커널=stride) 한 번으로
   D 차원 벡터 하나로 만든다 (사실상 Linear). 이미지는 같은 프레임을 tp 번 복제해서 넣는다 (전처리 쪽 일).
   patch 순서는 raster 가 아니라 2×2 merge 블록 순서 (블록 행, 블록 열, 블록 안 행, 블록 안 열).
2. 절대 위치: 학습된 정사각 표 (side×side) 를 각 이미지 격자 (h, w) 로 bilinear 보간 (align_corners=True) 해서 더한다.
   patch 마다 표의 4칸을 가중합.
3. 블록 × depth: pre-LayerNorm, 양방향 attention (bias 있는 fused qkv), GELU(tanh) MLP.
   - 2D RoPE: head_dim 을 [h 주파수 | w 주파수] × 2 로 채워 전체 채널을 회전 (부분 회전 없음).
   - attention 범위: 프레임 하나 (cu_seqlens = 프레임별 h·w). 이미지끼리, 비디오 프레임끼리 섞이지 않는다.
     (Qwen2.5-VL 의 window attention 은 Qwen3.5 부터 없다 → 모든 층이 프레임 단위 full attention.)
4. patch merger: LayerNorm 후 merge 블록 안 2×2 = 4 patch 를 이어붙여 [4D] → MLP(GELU) → LM 폭.

LM 쪽 M-RoPE (get_rope_index)
- 텍스트 토큰: 세 축(t, h, w) 모두 같은 1D 위치.
- 이미지/비디오 토큰: 시작 위치 s 에서 t 축 = s + 프레임 번호, h 축 = s + 행, w 축 = s + 열 (merge 후 격자).
  다음 텍스트는 s + max(h, w)/merge 부터 → 이미지가 차지하는 위치 폭이 토큰 수보다 훨씬 작다.
- 비디오는 프레임마다 타임스탬프 텍스트로 나뉘므로 프레임마다 t=1 짜리 격자로 쪼개 따로 매긴다.
- 회전 채널 배치는 interleaved: 주파수 i 를 i mod 3 = 0/1/2 → t/h/w 축 위치로 (mrope_section 개수만큼, 나머지는 t).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .common import attention, rotate_half
from .qwen36 import Qwen36, config_from_hf as text_config_from_hf


@dataclass
class Qwen38VisionConfig:
    d_model: int = 1152
    n_layer: int = 27
    n_head: int = 16
    d_ff: int = 4304
    act: str = "gelu_pytorch_tanh"            # MLP 활성화 ("gelu" 면 정확한 erf GELU)
    in_ch: int = 3
    patch: int = 16
    t_patch: int = 2                          # 시간축 patch (프레임 2장을 묶음)
    merge: int = 2                            # 2×2 patch → LM 토큰 1개
    out_dim: int = 5120                       # LM 폭
    n_pos_emb: int = 2304                     # 학습 위치 표 칸 수 (48×48)
    rope_theta: float = 10000.0
    ln_eps: float = 1e-6
    # ablation (기본값 = 원본)
    use_abs_pos: bool = True                  # 보간한 학습 위치 임베딩
    use_rope: bool = True                     # 2D RoPE
    attn_scope: str = "frame"                 # "frame"(원본) / "clip"(비디오 프레임끼리도) / "all"(배치 전체)


@dataclass
class VisionOutput:
    last_hidden_state: torch.Tensor           # merger 전 [N, D]
    pooler_output: torch.Tensor               # merger 후 LM 토큰 [N / merge², out_dim]


# ---------------------------------------------------------------- 격자 → 위치 (가중치 없는 전처리)

def vision_rot_pos_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """patch 마다 (행, 열) [N, 2], merge 블록 순서. 비디오는 프레임마다 반복."""
    out = []
    for t, h, w in grid_thw.tolist():
        hp, wp = torch.meshgrid(torch.arange(h, device=grid_thw.device), torch.arange(w, device=grid_thw.device),
                                indexing="ij")
        blk = (h // merge, merge, w // merge, merge)
        hp, wp = hp.reshape(blk).transpose(1, 2).flatten(), wp.reshape(blk).transpose(1, 2).flatten()
        out.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def _bilinear_axis(idx: torch.Tensor, size: int, side: int):
    """한 축: 격자 위치 idx (길이 size) → 표 (길이 side) 의 이웃 두 칸과 가중치 (align_corners=True, 끝은 clamp)."""
    # 원본과 같은 연산: float32 ÷ (원소별 long 텐서). 파이썬 스칼라로 나누면 CUDA 가 역수 곱으로 바꿔
    # 1 ulp 차이 → floor 가 바뀌면 고른 표 칸(pos_embed hook 출력)이 달라질 수 있다.
    src = idx.to(torch.float32) * (side - 1) / torch.full_like(idx, max(size - 1, 1), dtype=torch.long)
    floor = torch.floor(src)
    off = torch.arange(0, 2, device=idx.device)
    taps = (floor.long()[:, None] + off).clamp(0, side - 1)
    w = (1 - (src[:, None] - floor[:, None] - off).abs()).clamp(min=0)
    return taps, w


def pos_interp(grid_thw: torch.Tensor, side: int, merge: int):
    """학습 위치 표 보간용 (표 index [N, 4], 가중치 [N, 4]). patch 순서는 vision_rot_pos_ids 와 같다."""
    idx, wts = [], []
    for t, h, w in grid_thw.tolist():
        rc = vision_rot_pos_ids(grid_thw.new_tensor([[1, h, w]]), merge)
        ht, hw = _bilinear_axis(rc[:, 0], h, side)
        wt, ww = _bilinear_axis(rc[:, 1], w, side)
        idx.append((ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4).repeat(t, 1))
        wts.append((hw[:, :, None] * ww[:, None, :]).reshape(-1, 4).repeat(t, 1))
    return torch.cat(idx), torch.cat(wts)


def attn_segments(grid_thw: torch.Tensor, scope: str = "frame") -> list[int]:
    """attention 이 서로 보는 구간 길이들 (패킹된 patch 열 기준)."""
    if scope == "frame":
        return [h * w for t, h, w in grid_thw.tolist() for _ in range(t)]
    if scope == "clip":
        return [t * h * w for t, h, w in grid_thw.tolist()]
    return [int(grid_thw.prod(-1).sum())]


# ---------------------------------------------------------------- ViT 부품

class PatchEmbed(nn.Module):
    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.c = c
        k = (c.t_patch, c.patch, c.patch)
        self.proj = nn.Conv3d(c.in_ch, c.d_model, kernel_size=k, stride=k, bias=True)

    def forward(self, x):
        c = self.c
        x = x.view(-1, c.in_ch, c.t_patch, c.patch, c.patch)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, c.d_model)


class VisionRotary(nn.Module):
    """2D 축 RoPE: head_dim/4 개 주파수를 행·열에 각각 → [h | w] 를 두 번 이어 head_dim 전체를 채운다."""

    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        dim = (c.d_model // c.n_head) // 2
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x, pos):                # pos [N, 2] → cos, sin [N, head_dim]
        f = pos[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        f = torch.cat([f[:, 0], f[:, 1]], dim=-1)
        f = torch.cat([f, f], dim=-1)
        return f.cos(), f.sin()


class VisionAttention(nn.Module):
    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.h = c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=True)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.attn_probs = nn.Identity()       # hook 지점: 구간마다 [1, H, n, n] (마지막 구간이 남음)

    def forward(self, x, segs, cos=None, sin=None):
        N = x.shape[0]
        q, k, v = self.qkv(x).reshape(N, 3, self.h, -1).permute(1, 0, 2, 3).unbind(0)   # [N, H, d]
        if cos is not None:
            dt = q.dtype
            cs, sn = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
            q, k = q.float(), k.float()
            q, k = (q * cs + rotate_half(q) * sn).to(dt), (k * cs + rotate_half(k) * sn).to(dt)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))                    # [1, H, N, d]
        ys = [attention(a, b, c, causal=False, probs_hook=self.attn_probs)
              for a, b, c in zip(q.split(segs, 2), k.split(segs, 2), v.split(segs, 2))]
        y = torch.cat(ys, dim=2).transpose(1, 2).reshape(N, -1)
        return self.proj(y)


class VisionMLP(nn.Module):
    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(c.d_model, c.d_ff, bias=True)
        self.linear_fc2 = nn.Linear(c.d_ff, c.d_model, bias=True)
        self.approx = "tanh" if "tanh" in c.act else "none"

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate=self.approx))


class VisionBlock(nn.Module):
    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.attn = VisionAttention(c)
        self.mlp = VisionMLP(c)

    def forward(self, x, segs, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), segs, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """LayerNorm(patch 단위) → 이웃 merge² patch 이어붙이기 → MLP → LM 폭."""

    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.d = c.d_model * c.merge ** 2
        self.norm = nn.LayerNorm(c.d_model, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.d, self.d)
        self.linear_fc2 = nn.Linear(self.d, c.out_dim)

    def forward(self, x):
        x = self.norm(x).view(-1, self.d)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen38Vision(nn.Module):
    def __init__(self, c: Qwen38VisionConfig):
        super().__init__()
        self.config = c
        self.side = int(c.n_pos_emb ** 0.5)
        self.patch_embed = PatchEmbed(c)
        self.pos_embed = nn.Embedding(c.n_pos_emb, c.d_model)
        self.rotary_pos_emb = VisionRotary(c)
        self.blocks = nn.ModuleList(VisionBlock(c) for _ in range(c.n_layer))
        self.merger = PatchMerger(c)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> VisionOutput:
        c = self.config
        x = self.patch_embed(hidden_states)                                   # [N, D]
        if c.use_abs_pos:
            idx, w = pos_interp(grid_thw, self.side, c.merge)
            x = x + (self.pos_embed(idx) * w[:, :, None]).sum(1).to(x.dtype)
        cos = sin = None
        if c.use_rope:
            cos, sin = self.rotary_pos_emb(x, vision_rot_pos_ids(grid_thw, c.merge))
        segs = attn_segments(grid_thw, c.attn_scope)
        for blk in self.blocks:
            x = blk(x, segs, cos, sin)
        return VisionOutput(x, self.merger(x))


# ---------------------------------------------------------------- LM 쪽 M-RoPE

def rope_index(mm_token_type_ids: torch.Tensor, image_grid_thw=None, video_grid_thw=None,
               attention_mask=None, merge: int = 2):
    """토큰 종류 (0 텍스트, 1 이미지, 2 비디오) → 3축 위치 [3, B, T], 위치 delta [B, 1] (원본 get_rope_index 와 같은 규칙)."""
    if video_grid_thw is not None:   # 비디오는 프레임마다 t=1 격자로 쪼갬 (프레임 사이에 타임스탬프 텍스트)
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    grids = {1: iter(image_grid_thw.tolist()) if image_grid_thw is not None else None,
             2: iter(video_grid_thw.tolist()) if video_grid_thw is not None else None}
    B, T = mm_token_type_ids.shape
    dev = mm_token_type_ids.device
    pos = torch.zeros(3, B, T, dtype=torch.long, device=dev)
    deltas = []
    for b in range(B):
        keep = attention_mask[b].bool() if attention_mask is not None else torch.ones(T, dtype=torch.bool, device=dev)
        types = mm_token_type_ids[b][keep].tolist()
        cur, chunks = 0, []
        for kind, run in itertools.groupby(types):
            n = len(list(run))
            if kind == 0:
                chunks.append(torch.arange(cur, cur + n, device=dev).expand(3, -1))
                cur += n
            else:
                t, h, w = next(grids[kind])
                h, w = h // merge, w // merge
                tt, hh, ww = torch.meshgrid(torch.arange(t, device=dev), torch.arange(h, device=dev),
                                            torch.arange(w, device=dev), indexing="ij")
                chunks.append(torch.stack([tt, hh, ww]).reshape(3, -1) + cur)
                cur += max(h, w)
        p = torch.cat(chunks, dim=1)
        pos[:, b, keep] = p
        deltas.append(int(p.max()) + 1 - len(types))
    return pos, torch.tensor(deltas, device=dev).unsqueeze(1)


def mrope_cos_sin(pos: torch.Tensor, inv_freq: torch.Tensor, section=(11, 11, 10)):
    """pos [3, B, T] → cos, sin [B, T, rot]. interleaved: 주파수 i 는 i%3 축 (h·w 는 section 개수까지만, 나머지 t)."""
    f = pos[..., None].float() * inv_freq.to(pos.device, torch.float)       # [3, B, T, F]
    out = f[0].clone()
    for ax in (1, 2):
        sl = slice(ax, section[ax] * 3, 3)
        out[..., sl] = f[ax, ..., sl]
    out = torch.cat([out, out], dim=-1)
    return out.cos(), out.sin()


@dataclass
class Qwen38VLConfig:
    vision: Qwen38VisionConfig = field(default_factory=Qwen38VisionConfig)
    text: object = None                        # qwen36.Qwen36Config
    image_token_id: int = 248056
    video_token_id: int = 248057
    mrope_section: tuple = (11, 11, 10)
    use_mrope: bool = True                     # ablation: False 면 이미지 토큰도 1D 위치


class Qwen38VL(nn.Module):
    """미니 VLM: ViT 출력으로 이미지/비디오 자리 토큰 임베딩을 바꿔 끼우고, 3축 위치로 Qwen36 층을 돌린다.
    이름은 HF Qwen3_5Model 과 같다 (visual.*, language_model.*)."""

    def __init__(self, c: Qwen38VLConfig):
        super().__init__()
        self.config = c
        self.visual = Qwen38Vision(c.vision)
        self.language_model = Qwen36(c.text)

    def forward(self, input_ids, mm_token_type_ids, pixel_values=None, image_grid_thw=None,
                pixel_values_videos=None, video_grid_thw=None):
        c, lm = self.config, self.language_model
        x = lm.embed_tokens(input_ids)
        for pv, g, tok in ((pixel_values, image_grid_thw, c.image_token_id),
                           (pixel_values_videos, video_grid_thw, c.video_token_id)):
            if pv is not None:
                feat = self.visual(pv, g).pooler_output
                x = x.masked_scatter((input_ids == tok)[..., None].to(x.device), feat.to(x.dtype))
        B, T = input_ids.shape
        if c.use_mrope:
            pos, _ = rope_index(mm_token_type_ids, image_grid_thw,
                                video_grid_thw.clone() if video_grid_thw is not None else None,
                                merge=c.vision.merge)
        else:
            pos = torch.arange(T, device=input_ids.device).view(1, 1, -1).expand(3, B, -1)
        cos, sin = mrope_cos_sin(pos, lm.inv_freq, c.mrope_section)
        cos, sin = cos[:, None].to(x.dtype), sin[:, None].to(x.dtype)       # [B, 1, T, rot] (head 축 broadcast)
        for layer in lm.layers:
            x = layer(x, cos, sin)
        return lm.norm(x)


# ---------------------------------------------------------------- 검증

def _int(v):
    return v if isinstance(v, int) else int(v[0])


def config_from_hf(c) -> Qwen38VisionConfig:
    rp = getattr(c, "rope_parameters", None) or {}
    return Qwen38VisionConfig(
        d_model=c.hidden_size, n_layer=c.depth, n_head=c.num_heads, d_ff=c.intermediate_size, act=c.hidden_act,
        in_ch=c.in_channels, patch=_int(c.patch_size), t_patch=_int(c.temporal_patch_size),
        merge=c.spatial_merge_size, out_dim=c.out_hidden_size, n_pos_emb=c.num_position_embeddings,
        rope_theta=rp.get("rope_theta", 10000.0))


def vl_config_from_hf(c) -> Qwen38VLConfig:
    """c = transformers Qwen3_5Config (text_config + vision_config)."""
    rp = getattr(c.text_config, "rope_parameters", None) or {}
    return Qwen38VLConfig(vision=config_from_hf(c.vision_config), text=text_config_from_hf(c.text_config),
                          image_token_id=c.image_token_id, video_token_id=c.video_token_id,
                          mrope_section=tuple(rp.get("mrope_section", (11, 11, 10))))


def verify_pair(ref) -> dict:
    """ref = transformers Qwen3_5VisionModel (이름이 같아 name_map 불필요). 최종 비교는 merger 출력."""
    mine = Qwen38Vision(config_from_hf(ref.config))
    pairs = [("patch_embed", "patch_embed"), ("pos_embed", "pos_embed"), ("rotary_pos_emb", "rotary_pos_emb")]
    for i in range(ref.config.depth):
        pairs += [(f"blocks.{i}.attn", f"blocks.{i}.attn"), (f"blocks.{i}", f"blocks.{i}")]
    pairs += [("merger", "merger")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs,
                out_fn_ref=lambda o: o.pooler_output, out_fn_mine=lambda o: o.pooler_output)
