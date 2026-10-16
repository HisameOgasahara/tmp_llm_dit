"""B4 · MiniMax-H3: 비디오 + 오디오를 한 줄로 이어 붙여 같이 denoise 하는 3D DiT (단일 스트림).

원본: DiffSynth-Studio `diffsynth/models/minimax_h3_dit.py` (`MiniMaxH3DiT`, 실제 50층 / hidden 5376 / head 56×128).
파라미터 이름은 원본·실제 체크포인트(MiniMaxAI/MiniMax-H3 `FL2VA/transformer`) 키와 같다.

한 줄 시퀀스 (batch 1, 길이 S = 64 의 배수로 padding)
  [text | 조건 keyframe 프레임 | audio | video | pad]
- video latent [1, C, T, H, W] → (1, 2, 2) patch → 행 하나 = C·4 채널 → video_patch_proj
- audio latent [채널 2, 32, T_a] → 채널별로 시간축을 펼친 행(채널-우선) → audio_patch_proj
- text: 인코더 hidden → condition_proj → token refiner (RoPE·조건 없는 작은 transformer 2층 + final norm)
- 각 modality 는 자기 자리(position_ids)에만 index_add 로 들어가고 나머지 칸은 0.

위치 (3축 RoPE, head_dim 앞쪽 6·n 채널만 회전, 나머지는 그대로)
- 토큰마다 (t, h, w) 실수 좌표. 축마다 n(=16)개 주파수 → [t | h | w] 를 두 번 이어 rotate_half 방식으로 회전.
- text: t = 0..L-1. video: t 는 text 뒤에서 시작해 5프레임 묶음 간격(1,4,4,4,4)×5/3 으로 증가, h/w 는 화면 비율로 정한 격자×32.
- audio: t = text 뒤 0..T_a-1 (채널마다 반복), w 는 채널 0 이면 격자 왼쪽 끝, 채널 1 이면 오른쪽 끝. pad 는 0.

블록 (모든 modality 가 같은 가중치를 쓰는 단일 스트림, pre-norm RMSNorm)
- attention: fused qkv (head 마다 [q|k|v] 순으로 interleave) → q/k RMSNorm → RoPE → 구간별 양방향 attention.
  구간(cu_seqlens) = [0, 실제 길이) 와 [실제 길이, S) → pad 는 pad 끼리만 본다.
- MLP: SwiGLU (fc1 출력 앞 절반 = gate, 뒤 절반 = up), bias 없음.
- adaLN: timestep 임베딩 하나에서 modality 3개(video 0 / text 1 / audio 2) 각각의 shift·scale·gate × (attn, mlp) 를 만든다.
  토큰마다 자기 timestep(고유값 index) × 3 + modality tag 행을 골라 쓴다 → text 도 timestep 으로 변조된다.
  토큰별 timestep: video·audio 따로, 조건 keyframe 은 max(t, 0.999), 편집에서 고정한 칸은 1.0 (깨끗함).
- final layer: RMSNorm + (modality 구분 없는) shift/scale → video_out / audio_out 을 모든 행에 적용 후 자기 자리만 꺼냄.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_MODALITY = 3          # adaLN 행: 0 = video(조건 포함), 1 = text, 2 = audio. pad(-1) 는 0 으로 clamp
PATCH = (1, 2, 2)


@dataclass
class MiniMaxH3Config:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple = PATCH
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16            # 축마다 주파수 수 → 회전 채널 = 6 × 이것
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # ablation 스위치 (기본값 = 원본)
    use_rope: bool = True                  # False: 본 블록에서 3축 RoPE 를 끔 (위치 정보 없음)
    use_qk_norm: bool = True               # False: q/k RMSNorm 생략
    use_token_refiner: bool = True         # False: condition_proj 출력을 바로 시퀀스에 넣음
    modality_adaln: bool = True            # False: 모든 토큰이 video(0) 행의 변조를 공유
    use_gate: bool = True                  # False: adaLN gate = 1 (residual 에 그대로 더함)

    @property
    def video_patch_dim(self) -> int:
        p = self.patch_size
        return self.latents_dim * p[0] * p[1] * p[2]


# ---------------------------------------------------------------- 입력 포장 (DiffSynth 파이프라인과 같은 규칙)

def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    """[1, C, T, H, W] → [T·(H/2)·(W/2), C·4]. 행 순서 t → h → w, 채널 순서 c → (pt, ph, pw)."""
    b, c, ft, fh, fw = latent.shape
    t, h, w = ft // PATCH[0], fh // PATCH[1], fw // PATCH[2]
    x = latent.reshape(b, c, t, PATCH[0], h, PATCH[1], w, PATCH[2])
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * PATCH[0] * PATCH[1] * PATCH[2]).contiguous()


def unpatchify_video(rows: torch.Tensor, ft: int, fh: int, fw: int, channel: int) -> torch.Tensor:
    t, h, w = ft // PATCH[0], fh // PATCH[1], fw // PATCH[2]
    x = rows.reshape(-1, t, h, w, channel, *PATCH)
    return torch.einsum("nthwcrpq->nctrhpwq", x).reshape(-1, channel, ft, fh, fw).contiguous()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """[채널, D, T] → [채널·T, D] (채널-우선)."""
    ac, ld, steps = latent.shape
    return latent.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def unpack_audio(rows: torch.Tensor, audio_channel: int, steps: int) -> torch.Tensor:
    return rows.reshape(audio_channel, steps, -1).permute(0, 2, 1).contiguous()


_INTERP, _T_GROUP, _FRAME_PER_TOKEN, _FRAME_RESCALE = 32, 5, (1, 4, 4, 4, 4), 5.0 / 3.0


def _axis(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n, origin):
    spans = torch.tensor([_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_span(n):
    spans = np.ones(int(n), dtype=np.float64) * _FRAME_RESCALE
    for i in range(_T_GROUP):
        spans[i::_T_GROUP] *= _FRAME_PER_TOKEN[i]
    return float(spans.sum())


def build_inputs(video_latents, audio_latents, prompt_embeds, text_token_tags, timestep_video, timestep_audio, *,
                 keyframe_cond_anchor=None, keyframe_indices=(), input_latents_video=None, denoise_mask_video=None,
                 input_latents_audio=None, denoise_mask_audio=None, imgvid_cond_noise_aug=0.999, seq_align=64):
    """DiffSynth `MiniMaxH3Unit_PackedSequenceBuilder._build_packed_fl2va` + `model_fn_minimax_h3` 의 입력 준비를 옮긴 것.

    반환 = 원본 MiniMaxH3DiT.forward 의 keyword 인자 dict (t2va / fl2va / 편집(retake) 경로).
    video_latents [1, C, T, H, W], audio_latents [채널, D, T_a], prompt_embeds [L, text_dim], timestep ∈ [0, 1000].
    """
    t_video = 1.0 - float(timestep_video) / 1000
    t_audio = 1.0 - float(timestep_audio) / 1000
    dtype, device = video_latents.dtype, video_latents.device
    latent_t, latent_h, latent_w = video_latents.shape[2:]
    audio_channel, audio_t = audio_latents.shape[0], audio_latents.shape[-1]
    text_len = prompt_embeds.shape[0]

    # ---- packed 배치 [text | cond | audio | video | pad]
    frame_rows = (latent_h // 2) * (latent_w // 2)
    video_rows_n, audio_rows_n = latent_t * frame_rows, audio_t * audio_channel
    keyframe_indices = list(keyframe_indices) if keyframe_cond_anchor is not None else []
    cond_rows_n = len(keyframe_indices) * frame_rows
    used = text_len + cond_rows_n + audio_rows_n + video_rows_n
    seq_len = (used + seq_align - 1) // seq_align * seq_align
    cond_sl = slice(text_len, text_len + cond_rows_n)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows_n)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows_n)
    img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid, w_grid = _axis(latent_h, 2, sqrt_area), _axis(latent_w, 2, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    span = _temporal_span(latent_t)
    for i, idx in enumerate(keyframe_indices):
        cond_t = float(text_len) if idx == 0 else float(text_len) + span - _FRAME_RESCALE
        s0 = cond_sl.start + i * frame_rows
        g[s0:s0 + frame_rows, 0] = cond_t
        g[s0:s0 + frame_rows, 1:] = frame
    vg = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    vg[:, :, 0] = _video_t_grid(latent_t, float(text_len))[:, None]
    vg[:, :, 1:] = frame[None]
    g[video_sl] = vg.reshape(-1, 3)
    g[audio_sl, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
    g[audio_sl, 2] = torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
                                torch.full((audio_rows_n - audio_t,), float(w_grid[-1]), dtype=torch.float64)])
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[:text_len] = 1
    token_tags[audio_sl] = 2
    token_tags[img_pos] = 0
    token_tags[text_pos] = text_token_tags.cpu()

    # ---- latent → 행, 토큰별 timestep
    if input_latents_video is not None:   # 편집: 고정 칸은 원본 latent 로, timestep 1.0
        video_latents = video_latents * denoise_mask_video + input_latents_video * (1.0 - denoise_mask_video)
        denoise_mask_video = patchify_video(denoise_mask_video)[:, 0]
    if input_latents_audio is not None:
        audio_latents = audio_latents * denoise_mask_audio + input_latents_audio * (1.0 - denoise_mask_audio)
        denoise_mask_audio = pack_audio(denoise_mask_audio)[:, 0]
    img_pos, audio_pos, text_pos = img_pos.to(device), audio_pos.to(device), text_pos.to(device)
    video_rows, audio_rows = patchify_video(video_latents), pack_audio(audio_latents)
    x = torch.zeros(1, seq_len, video_rows.shape[1], dtype=dtype, device=device)
    x[0].index_copy_(0, img_pos[cond_rows_n:], video_rows)
    if keyframe_cond_anchor is not None:
        x[0].index_copy_(0, img_pos[:cond_rows_n], keyframe_cond_anchor.to(device, dtype))
    audio_x = torch.zeros(1, seq_len, audio_rows.shape[1], dtype=dtype, device=device)
    audio_x[0].index_copy_(0, audio_pos, audio_rows.to(dtype))
    timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
    timesteps[audio_pos] = t_audio
    if input_latents_video is not None:
        timesteps[img_pos[cond_rows_n:][denoise_mask_video.to(device) == 0]] = 1.0
    if input_latents_audio is not None:
        timesteps[audio_pos[denoise_mask_audio.to(device) == 0]] = 1.0
    timesteps[img_pos[:cond_rows_n]] = max(t_video, imgvid_cond_noise_aug)
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
    cu = torch.tensor([0, used, seq_len], dtype=torch.int32, device=device)
    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    g, token_tags = g.to(device), token_tags.to(device)
    return dict(
        x=x, audio_x=audio_x, img_position_ids=g[None], unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices, update_mask=None, token_tags=token_tags, prompt_embeds=prompt_embeds,
        img_pos_info={"position_ids": img_pos}, audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos}, img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        skip_mask_out_condition=True)


# ---------------------------------------------------------------- 부품

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, freqs):
    """x [S, H, D], freqs [S, R] (R ≤ D): 앞 R 채널만 회전."""
    r = freqs.shape[-1]
    x_rot, x_pass = x[..., :r], x[..., r:]
    cos, sin = freqs.cos().to(x.dtype).unsqueeze(1), freqs.sin().to(x.dtype).unsqueeze(1)
    return torch.cat((x_rot * cos + rotate_half(x_rot) * sin, x_pass), dim=-1)


def segment_mask(cu_seqlens, n, device):
    """cu_seqlens 로 나눈 구간끼리만 True 인 [n, n] (block-diagonal)."""
    pos = torch.arange(n, device=device)
    seg = torch.searchsorted(cu_seqlens.to(device=device, dtype=torch.long), pos, right=True)
    return seg[:, None] == seg[None, :]


class Rope3D(nn.Module):
    """(t, h, w) → 축마다 n 개 주파수, [t|h|w|t|h|w]. inv_freq 는 체크포인트에 있지만 원본처럼 매번 다시 계산."""

    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.n, self.theta = c.rope_inv_freq_len, c.rope_theta
        self.inv_freq = nn.Parameter(self._inv_freq())   # forward 에서 안 씀 (원본과 동일)

    def _inv_freq(self, device=None):
        return 1.0 / (self.theta ** (torch.arange(0, self.n, dtype=torch.float32, device=device) / self.n))

    def forward(self, pos3d):
        pos = pos3d[0].to(torch.float32)                                  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self._inv_freq(pos.device).view(1, 1, -1)
        half = torch.cat(per_axis.unbind(dim=1), dim=-1)                  # [S, 3n]
        return torch.cat((half, half), dim=-1)


class TimeEmbedder(nn.Module):
    """[cos | sin] (지수 분모 = half), Linear → SiLU → Linear."""

    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.dim = c.timestep_input_dim
        self.proj_in = nn.Linear(c.timestep_input_dim, c.time_embed_hidden_size)
        self.proj_out = nn.Linear(c.time_embed_hidden_size, c.time_embed_dim)

    def forward(self, t, dtype):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.proj_out(F.silu(self.proj_in(emb.to(dtype))))


class Attention(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.h, self.d = c.num_attention_heads, c.attention_head_dim
        self.use_qk_norm = c.use_qk_norm
        inner = self.h * self.d
        self.qkv_proj = nn.Linear(c.hidden_size, 3 * inner, bias=False)
        self.q_norm = nn.RMSNorm(self.d, eps=c.qk_norm_eps)
        self.k_norm = nn.RMSNorm(self.d, eps=c.qk_norm_eps)
        self.out_proj = nn.Linear(inner, c.hidden_size, bias=False)
        self.attn_probs = nn.Identity()   # hook 지점: [1, H, S, S] (구간 밖 = 0)

    def forward(self, x, rope_freqs, mask):
        n = x.shape[0]
        qkv = self.qkv_proj(x).view(n, self.h, 3, self.d)                 # head 마다 [q|k|v]
        q, k, v = qkv.unbind(dim=2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope_freqs is not None:
            q, k = apply_rope(q, rope_freqs), apply_rope(k, rope_freqs)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))     # [1, H, S, d]
        scores = (q @ k.transpose(-2, -1)) * self.d ** -0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = self.attn_probs(scores.float().softmax(-1).to(q.dtype))
        out = (probs @ v).squeeze(0).transpose(0, 1).reshape(n, self.h * self.d)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden_size, 2 * c.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(c.ffn_hidden_size, c.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class AdalnProj(nn.Module):
    """SiLU(t_emb) → Linear → [고유 timestep × modality 행, expand × hidden] → expand 개로 나눔."""

    def __init__(self, c: MiniMaxH3Config, expand, n_modality):
        super().__init__()
        self.expand, self.m = expand, n_modality
        self.linear = nn.Linear(c.time_embed_dim, expand * c.hidden_size * n_modality)

    def forward(self, t_emb):
        x = self.linear(F.silu(t_emb))
        return x.view(x.shape[0] * self.m, -1).chunk(self.expand, dim=-1)


def modulate(x, shift, scale, idx):
    return (x * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)).to(x.dtype)


class RefinerBlock(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.attn = Attention(c)
        self.mlp = MLP(c)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), None, mask)
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.blocks = nn.ModuleList(RefinerBlock(c) for _ in range(c.token_refiner_num_layers))
        self.final_norm = nn.RMSNorm(c.hidden_size, eps=c.final_norm_eps)

    def forward(self, x, mask):
        for b in self.blocks:
            x = b(x, mask)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.use_gate = c.use_gate
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.attn = Attention(c)
        self.mlp = MLP(c)
        self.adaln_proj = AdalnProj(c, 6, N_MODALITY)
        self.mod_probe = nn.Identity()   # hook 지점: [6, S, hidden] 토큰별 (shift, scale, gate) × (attn, mlp)

    def forward(self, x, t_emb, idx, rope_freqs, mask):
        sa, sca, ga, sm, scm, gm = self.adaln_proj(t_emb)
        if not self.use_gate:
            ga, gm = torch.ones_like(ga), torch.ones_like(gm)
        self.mod_probe(torch.stack([t.index_select(0, idx) for t in (sa, sca, ga, sm, scm, gm)]))
        h = self.attn(modulate(self.norm1(x), sa, sca, idx), rope_freqs, mask)
        x = (x + ga.index_select(0, idx) * h).to(x.dtype)
        h = self.mlp(modulate(self.norm2(x), sm, scm, idx))
        return (x + gm.index_select(0, idx) * h).to(x.dtype)


class FinalLayer(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.final_norm_eps)
        self.adaln_proj = AdalnProj(c, 2, 1)       # modality 구분 없음
        self.video_out = nn.Linear(c.hidden_size, c.video_patch_dim)
        self.audio_out = nn.Linear(c.hidden_size, c.audio_latents_dim)
        self.mod_probe = nn.Identity()   # hook 지점: [2, S, hidden] (shift, scale)

    def forward(self, x, t_emb, t_index):
        shift, scale = self.adaln_proj(t_emb)
        self.mod_probe(torch.stack([shift.index_select(0, t_index), scale.index_select(0, t_index)]))
        h = modulate(self.norm(x), shift, scale, t_index)
        return self.video_out(h), self.audio_out(h)


class MiniMaxH3(nn.Module):
    def __init__(self, c: MiniMaxH3Config):
        super().__init__()
        self.config = c
        d = c.hidden_size
        self.video_patch_proj = nn.Linear(c.video_patch_dim, d)
        self.audio_patch_proj = nn.Linear(c.audio_latents_dim, d)
        self.condition_proj = nn.Linear(c.text_dim, d)
        self.time_embedder = TimeEmbedder(c)
        self.rope = Rope3D(c)
        self.token_refiner = TokenRefiner(c)
        self.blocks = nn.ModuleList(DiTBlock(c) for _ in range(c.num_layers))
        self.final_layer = FinalLayer(c)

    def forward(self, x, audio_x, pos3d, t_unique, t_index, token_tags, text, img_pos, audio_pos, text_pos,
                cu_seqlens, text_cu_seqlens, out_img_pos=None, update_mask=None, update_audio_mask=None,
                control_hints=None):
        """packed 한 줄 입력 (build_inputs 참고) → (video 행 [len(out_img_pos), C·4], audio 행 [len(audio_pos), D]).

        x [1, S, C·4] / audio_x [1, S, D]: 자기 자리 행만 의미 있음. pos3d [1, S, 3]. t_unique [M], t_index [S] (∈ 0..M-1).
        token_tags [S]: 0 video / 1 text / 2 audio / -1 pad. text [L, text_dim].
        """
        c = self.config
        dev, dtype, n = x.device, text.dtype, x.shape[1]
        img_pos, audio_pos, text_pos = (p.view(-1).long().to(dev) for p in (img_pos, audio_pos, text_pos))
        out_img_pos = img_pos if out_img_pos is None else out_img_pos.view(-1).long().to(dev)
        t_index = t_index.view(-1).long().to(dev)
        tags = token_tags.view(-1).long().to(dev).clamp(min=0)
        rope = self.rope(pos3d).to(dev) if c.use_rope else None

        # 임베딩: modality 마다 자기 투영 → 자기 자리에 더함
        video = self.video_patch_proj(x.view(-1, x.shape[-1]).index_select(0, img_pos).to(dtype))
        audio = self.audio_patch_proj(audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).to(dtype))
        txt = self.condition_proj(text.to(dev))
        if c.use_token_refiner:
            txt = self.token_refiner(txt, segment_mask(text_cu_seqlens, txt.shape[0], dev))
        h = torch.zeros(n, c.hidden_size, device=dev, dtype=dtype)
        h.index_add_(0, text_pos, txt.to(dtype)[: text_pos.shape[0]])
        h.index_add_(0, img_pos, video.to(dtype)[: img_pos.shape[0]])
        h.index_add_(0, audio_pos, audio.to(dtype)[: audio_pos.shape[0]])
        t_emb = self.time_embedder(t_unique.view(-1).to(dev), dtype)

        idx = t_index * N_MODALITY + (tags if c.modality_adaln else torch.zeros_like(tags))
        mask = segment_mask(cu_seqlens, n, dev)
        for i, b in enumerate(self.blocks):
            h = b(h, t_emb, idx, rope, mask)
            if control_hints is not None and i in control_hints:
                h = h + control_hints[i].to(h.device, h.dtype)

        v, a = self.final_layer(h, t_emb, t_index)
        v, a = v.index_select(0, out_img_pos), a.index_select(0, audio_pos)
        if update_mask is not None:
            v = v * update_mask.view(-1).to(dev).unsqueeze(-1)
        if update_audio_mask is not None:
            a = a * update_audio_mask.view(-1).to(dev).unsqueeze(-1)
        return v, a


# ---------------------------------------------------------------- 검증

def config_from_hf(cfg) -> MiniMaxH3Config:
    """FL2VA/transformer/config.json (dict 또는 속성 객체) → config. adaln_out_features 는 6·hidden·3 으로 정해짐."""
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    c = MiniMaxH3Config(**{k: get(k) for k in (
        "num_layers", "token_refiner_num_layers", "hidden_size", "num_attention_heads", "attention_head_dim",
        "ffn_hidden_size", "latents_dim", "audio_latents_dim", "text_dim", "timestep_input_dim",
        "time_embed_hidden_size", "time_embed_dim", "rope_inv_freq_len", "norm_eps", "qk_norm_eps",
        "final_norm_eps") if get(k) is not None})
    c.patch_size = tuple(get("patch_size", PATCH))
    assert c.patch_size == PATCH, "원본은 (1, 2, 2) patch 만 지원"
    for k, want in (("adaln_out_features", 6 * c.hidden_size * N_MODALITY), ("final_adaln_out_features", 2 * c.hidden_size)):
        assert get(k, want) == want, f"{k}={get(k)} != {want}"
    return c


def config_from_module(ref) -> MiniMaxH3Config:
    """DiffSynth MiniMaxH3DiT 는 config 를 저장하지 않으므로 모듈 shape 에서 읽는다."""
    b0 = ref.blocks[0] if len(ref.blocks) else ref.token_refiner.blocks[0]
    te = ref.time_embedder
    return MiniMaxH3Config(
        num_layers=len(ref.blocks), token_refiner_num_layers=len(ref.token_refiner.blocks),
        hidden_size=ref.hidden_size, num_attention_heads=ref.num_attention_heads,
        attention_head_dim=b0.attn.head_dim, ffn_hidden_size=b0.mlp.fc2.in_features,
        latents_dim=ref.num_channels_latents, audio_latents_dim=ref.audio_patch_proj.in_features,
        text_dim=ref.condition_proj.in_features, timestep_input_dim=te.frequency_embedding_size,
        time_embed_hidden_size=te.proj_in.out_features, time_embed_dim=te.proj_out.out_features,
        rope_inv_freq_len=ref.rope.inv_freq_len, norm_eps=b0.norm1.eps, qk_norm_eps=b0.attn.q_norm.eps,
        final_norm_eps=ref.final_layer.norm.eps)


def mine_inputs(d: dict) -> dict:
    """원본 forward 인자 dict → 내 forward 인자."""
    skip = d.get("skip_mask_out_condition", False)
    return dict(
        x=d["x"], audio_x=d["audio_x"], pos3d=d["img_position_ids"], t_unique=d["unique_timesteps"],
        t_index=d["inverse_indices"], token_tags=d["token_tags"], text=d["prompt_embeds"],
        img_pos=d["img_pos_info"]["position_ids"], audio_pos=d["audio_pos_info"]["position_ids"],
        text_pos=d["text_pos_info"]["position_ids"], cu_seqlens=d["packed_seq_params"]["cu_seqlens_q"],
        text_cu_seqlens=d["refiner_packed_seq_params"]["cu_seqlens_q"],
        out_img_pos=d["img_pos_for_infer_output_info"]["position_ids"],
        update_mask=None if skip else d["update_mask"],
        update_audio_mask=None if skip else d.get("update_audio_mask"),
        control_hints=d.get("control_hints"))


def joined(out) -> torch.Tensor:
    """(video 행, audio 행) → 한 벡터 (둘 다 비교)."""
    return torch.cat([out[0].reshape(-1), out[1].reshape(-1)])


def verify_pair(ref) -> dict:
    """ref = DiffSynth MiniMaxH3DiT (파라미터 이름이 같아 name_map 불필요)"""
    c = config_from_module(ref)
    mine = MiniMaxH3(c)
    pairs = [("time_embedder", "time_embedder"), ("video_patch_proj", "video_patch_proj"),
             ("audio_patch_proj", "audio_patch_proj"), ("condition_proj", "condition_proj")]
    pairs += [(f"token_refiner.blocks.{i}", f"token_refiner.blocks.{i}") for i in range(c.token_refiner_num_layers)]
    pairs.append(("token_refiner", "token_refiner"))
    for i in range(c.num_layers):
        pairs += [(f"blocks.{i}.attn", f"blocks.{i}.attn"), (f"blocks.{i}.mlp", f"blocks.{i}.mlp"),
                  (f"blocks.{i}", f"blocks.{i}")]
    pairs.append(("final_layer", "final_layer"))
    return dict(mine=mine, name_map=None, hook_pairs=pairs, mine_input_fn=mine_inputs,
                out_fn_ref=joined, out_fn_mine=joined)
