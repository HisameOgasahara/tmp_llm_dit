"""C1 · Cosmos3 (Omni): Mixture-of-Transformers — reasoner(이해, AR) 탑 + generator(확산) 탑.

diffusers 원본: src/diffusers/models/transformers/transformer_cosmos3.py (`Cosmos3OmniTransformer`).
파라미터 이름은 원본(= 실제 체크포인트 nvidia/Cosmos3-*/transformer) 과 같다.

한 줄 요약: 층마다 파라미터가 두 벌이다. 시퀀스는 [텍스트(reasoner) | 영상·소리·행동 latent(generator)] 로
이어 붙인 한 줄(batch 없음, packed)이고, 토큰은 자기 쪽 파라미터로만 계산되며 attention 에서만 만난다.

층 구성 (pre-norm residual 두 번, 두 탑이 같은 모양)
- reasoner 탑: input_layernorm → to_q/k/v → QK-norm → RoPE → **causal** attention (reasoner 토큰끼리만)
               → to_out → residual → post_attention_layernorm → mlp
- generator 탑: *_moe_gen norm, add_q/k/v_proj, norm_added_q/k, to_add_out, mlp_moe_gen
               query 는 generator 토큰, key/value 는 [reasoner ; generator] 전부 (**양방향**)
  → 정보는 reasoner → generator 한 방향으로만 흐른다. reasoner 출력은 생성 토큰과 무관 (= 순수 LLM 그대로).
  Edge(Nemotron) 계열은 reasoner QK-norm 이 없고, generator 가 볼 reasoner key 에만 별도 norm(k_norm_und_for_gen).
- 위치: 3D M-RoPE (t, h, w) 를 주파수 채널에 t,h,w,t,h,w... 로 끼워 넣는 interleaved 방식. head_dim 전체 회전.
  텍스트는 세 축이 같은 번호, latent 는 (시간, 행, 열). 두 탑이 같은 cos/sin 을 나눠 쓴다.

입력 / 출력 (generator 쪽)
- 영상 latent [1, C, T, H, W] → p×p patch (H, W 가 p 배수가 아니면 0 padding) → proj_in
- 소리 latent [C, T] → audio_proj_in + audio_modality_embed
- 행동 [T, D] → embodiment(domain) 마다 다른 가중치의 DomainAwareLinear + action_modality_embed
- 시간 조건: adaLN 이 아니라 **noisy 프레임 토큰에 timestep 임베딩을 더하기만** 한다 (조건 프레임 = 깨끗한 토큰은 그대로).
  → 이미지→영상, 편집, 행동 조건부 생성이 같은 경로 (noisy_frame_indexes 로 구분)
- 출력: noisy 토큰만 proj_out / audio_proj_out / action_proj_out → 속도(velocity) 예측. 조건 프레임 자리는 0.
- lm_head 는 reasoner 다음 토큰 예측용 (확산 forward 에서는 쓰지 않음 → logits() 로 따로 제공).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import rotate_half


@dataclass
class Cosmos3Config:
    vocab_size: int = 151936
    d_model: int = 4096
    n_layer: int = 36
    n_head: int = 32
    n_kv_head: int = 8
    head_dim: int = 128
    d_ff: int = 12288
    hidden_act: str = "silu"                 # "silu"(Qwen3 계열: Nano/Super) / "relu2"(Nemotron 계열: Edge)
    attention_bias: bool = False
    rms_eps: float = 1e-6
    qk_norm_for_text: bool = True            # reasoner q/k 의 head 별 RMSNorm
    use_und_k_norm_for_gen: bool = False     # generator 가 보는 reasoner key 에만 norm (qk_norm_for_text=False 일 때)
    rope_theta: float = 5_000_000.0
    rope_axes_dim: tuple = (24, 20, 20)      # M-RoPE 주파수 채널 수 (t, h, w), 합 = head_dim / 2
    # generator 입출력
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192              # = p * p * latent_channel
    timestep_scale: float = 0.001
    t_freq_dim: int = 256
    action_gen: bool = False
    action_dim: int | None = None
    num_embodiment_domains: int = 32
    sound_gen: bool = False
    sound_dim: int | None = None
    # ablation 스위치 (기본값 = 원본)
    und_causal: bool = True                  # False: reasoner 도 양방향
    gen_sees_und: bool = True                # False: generator 가 reasoner key/value 를 안 봄 (텍스트 조건 끊김)
    und_sees_gen: bool = False               # True: reasoner 도 generator 토큰을 봄 (양방향 정보 흐름)
    share_towers: bool = False               # True: generator 도 reasoner 파라미터 사용 (MoT → 단일 dense transformer)
    use_timestep: bool = True                # False: timestep 임베딩을 더하지 않음
    mrope_interleaved: bool = True           # False: [ttt..hhh..www] 덩어리 배치

    @property
    def nemotron(self) -> bool:
        return self.hidden_act == "relu2"


@dataclass
class Cosmos3Output:
    sample: list                 # 영상 속도 예측 [1, C, T, H, W] (조건 프레임 = 0)
    sound: list | None
    action: list | None
    und_hidden: torch.Tensor     # reasoner 최종 norm 출력 [N_und, d]
    gen_hidden: torch.Tensor     # generator 최종 norm 출력 [N_gen, d]

    def flat(self) -> torch.Tensor:
        return flat_preds(self.sample, self.sound, self.action)


def flat_preds(sample, sound=None, action=None) -> torch.Tensor:
    """모달리티별 예측 리스트를 한 줄 텐서로 (비교용)."""
    ts = list(sample) + list(sound or []) + list(action or [])
    return torch.cat([t.flatten() for t in ts])


# ---------------------------------------------------------------- 부품

class RMSNorm(nn.Module):
    """diffusers RMSNorm(기본) / Cosmos3NemotronRMSNorm(relu2 계열). fp32 에서는 같은 값, 저정밀 cast 순서만 다름."""

    def __init__(self, dim: int, eps: float, nemotron: bool = False):
        super().__init__()
        self.eps, self.nemotron = eps, nemotron
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        if self.nemotron:
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight.float() * x).to(dtype)
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class MLP(nn.Module):
    """silu: SwiGLU, relu2: down(relu(up(x))²) (gate_proj 없음)."""

    def __init__(self, d, d_ff, act):
        super().__init__()
        self.act = act
        if act == "silu":
            self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        if self.act == "relu2":
            return self.down_proj(torch.relu(self.up_proj(x)).square())
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DomainAwareLinear(nn.Module):
    """embodiment(로봇 종류) 마다 다른 weight/bias. 표(nn.Embedding)에서 domain 행을 꺼내 bmm."""

    def __init__(self, d_in, d_out, n_domains):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.fc = nn.Embedding(n_domains, d_out * d_in)
        self.bias = nn.Embedding(n_domains, d_out)

    def forward(self, x, domain_id):
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        w = self.fc(domain_id).view(-1, self.d_in, self.d_out)
        b = self.bias(domain_id).view(-1, self.d_out)
        return torch.bmm(x.unsqueeze(1), w).squeeze(1) + b      # x [N, d_in] (토큰마다 domain 하나)


class TimestepEmbedding(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d)
        self.linear_2 = nn.Linear(d, d)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


def timestep_freqs(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    """[cos | sin] 정현파 (diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0)."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    args = t[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MRoPE(nn.Module):
    """3D M-RoPE. position_ids [3, B, N] (t, h, w) → cos, sin [B, N, head_dim]."""

    def __init__(self, c: Cosmos3Config):
        super().__init__()
        hd = c.head_dim
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)),
                             persistent=False)
        self.axes, self.interleaved = tuple(c.rope_axes_dim), c.mrope_interleaved

    def forward(self, position_ids, dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
        freqs = position_ids[..., None].float() * self.inv_freq.float().to(position_ids.device)   # [3, B, N, hd/2]
        out = freqs[0].clone()
        if self.interleaved:   # 채널 i: i%3==1 → h, i%3==2 → w (각 축 채널 수까지), 나머지 t
            for ax, off in ((1, 1), (2, 2)):
                idx = slice(off, self.axes[ax] * 3, 3)
                out[..., idx] = freqs[ax, ..., idx]
        else:                  # [t 채널 | h 채널 | w 채널]
            a0, a1, a2 = self.axes
            out[..., a0:a0 + a1] = freqs[1, ..., a0:a0 + a1]
            out[..., a0 + a1:a0 + a1 + a2] = freqs[2, ..., a0 + a1:a0 + a1 + a2]
        emb = torch.cat((out, out), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope(x, cos, sin):
    """x [N, H, D], cos/sin [N, D]"""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def _attend(q, k, v, keep, hook):
    """q [H, Tq, D], k/v [Hkv, Tk, D] (GQA: head h → kv head h // g), keep [Tq, Tk] bool 또는 None."""
    g = q.shape[0] // k.shape[0]
    k, v = k.repeat_interleave(g, 0), v.repeat_interleave(g, 0)
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    if keep is not None:
        s = s.masked_fill(~keep, float("-inf"))
    p = hook(s.float().softmax(-1).to(q.dtype))
    return p @ v


# ---------------------------------------------------------------- 블록

class PackedMoTAttention(nn.Module):
    def __init__(self, c: Cosmos3Config):
        super().__init__()
        self.c = c
        d, H, Hk, D, b = c.d_model, c.n_head, c.n_kv_head, c.head_dim, c.attention_bias

        def norm():
            return RMSNorm(D, c.rms_eps, c.nemotron)
        # reasoner(und) 탑
        self.to_q, self.to_k, self.to_v = nn.Linear(d, H * D, b), nn.Linear(d, Hk * D, b), nn.Linear(d, Hk * D, b)
        self.to_out = nn.Linear(H * D, d, b)
        self.norm_q = norm() if c.qk_norm_for_text else nn.Identity()
        self.norm_k = norm() if c.qk_norm_for_text else nn.Identity()
        self.k_norm_und_for_gen = norm() if (c.use_und_k_norm_for_gen and not c.qk_norm_for_text) else None
        # generator 탑
        self.add_q_proj, self.add_k_proj = nn.Linear(d, H * D, b), nn.Linear(d, Hk * D, b)
        self.add_v_proj, self.to_add_out = nn.Linear(d, Hk * D, b), nn.Linear(H * D, d, b)
        self.norm_added_q, self.norm_added_k = norm(), norm()
        self.und_probs = nn.Identity()   # hook 지점: reasoner attention [H, N_und, N_und (+N_gen)]
        self.gen_probs = nn.Identity()   # hook 지점: generator attention [H, N_gen, N_und + N_gen]

    def forward(self, und, gen, rot):
        c = self.c
        H, Hk, D = c.n_head, c.n_kv_head, c.head_dim
        if c.share_towers:
            gq, gk, gv, go, gnq, gnk = self.to_q, self.to_k, self.to_v, self.to_out, self.norm_q, self.norm_k
        else:
            gq, gk, gv, go = self.add_q_proj, self.add_k_proj, self.add_v_proj, self.to_add_out
            gnq, gnk = self.norm_added_q, self.norm_added_k
        q_u, k_u, v_u = self.to_q(und).view(-1, H, D), self.to_k(und).view(-1, Hk, D), self.to_v(und).view(-1, Hk, D)
        q_g, k_g, v_g = gq(gen).view(-1, H, D), gk(gen).view(-1, Hk, D), gv(gen).view(-1, Hk, D)
        q_u, k_u = self.norm_q(q_u), self.norm_k(k_u)
        k_ug = self.k_norm_und_for_gen(k_u) if self.k_norm_und_for_gen is not None else k_u
        q_g, k_g = gnq(q_g), gnk(k_g)
        cu, su, cg, sg = rot
        q_u, k_u, k_ug = _rope(q_u, cu, su), _rope(k_u, cu, su), _rope(k_ug, cu, su)
        q_g, k_g = _rope(q_g, cg, sg), _rope(k_g, cg, sg)

        t = lambda x: x.transpose(0, 1)   # [N, H, D] → [H, N, D]
        nu, ng = und.shape[0], gen.shape[0]
        # reasoner: 자기들끼리 causal
        keep = torch.ones(nu, nu, dtype=torch.bool, device=und.device).tril() if c.und_causal else None
        ku, vu = k_u, v_u
        if c.und_sees_gen:
            keep = torch.cat([keep if keep is not None else torch.ones(nu, nu, dtype=torch.bool, device=und.device),
                              torch.ones(nu, ng, dtype=torch.bool, device=und.device)], 1)
            ku, vu = torch.cat([k_u, k_g]), torch.cat([v_u, v_g])
        o_u = _attend(t(q_u), t(ku), t(vu), keep, self.und_probs)
        # generator: [reasoner ; generator] 전부 양방향
        kg, vg = (torch.cat([k_ug, k_g]), torch.cat([v_u, v_g])) if c.gen_sees_und else (k_g, v_g)
        o_g = _attend(t(q_g), t(kg), t(vg), None, self.gen_probs)
        return self.to_out(t(o_u).flatten(-2)), go(t(o_g).flatten(-2))


class MoTDecoderLayer(nn.Module):
    def __init__(self, c: Cosmos3Config):
        super().__init__()
        self.c = c
        self.self_attn = PackedMoTAttention(c)
        self.mlp = MLP(c.d_model, c.d_ff, c.hidden_act)
        self.mlp_moe_gen = MLP(c.d_model, c.d_ff, c.hidden_act)
        n = lambda: RMSNorm(c.d_model, c.rms_eps, c.nemotron)
        self.input_layernorm, self.input_layernorm_moe_gen = n(), n()
        self.post_attention_layernorm, self.post_attention_layernorm_moe_gen = n(), n()

    def forward(self, und, gen, rot):
        if self.c.share_towers:
            n1g, n2g, mlpg = self.input_layernorm, self.post_attention_layernorm, self.mlp
        else:
            n1g, n2g, mlpg = self.input_layernorm_moe_gen, self.post_attention_layernorm_moe_gen, self.mlp_moe_gen
        a_u, a_g = self.self_attn(self.input_layernorm(und), n1g(gen), rot)
        und, gen = und + a_u, gen + a_g
        return und + self.mlp(self.post_attention_layernorm(und)), gen + mlpg(n2g(gen))


# ---------------------------------------------------------------- 전체

class Cosmos3(nn.Module):
    def __init__(self, c: Cosmos3Config):
        super().__init__()
        self.config = c
        d = c.d_model
        self.embed_tokens = nn.Embedding(c.vocab_size, d)
        self.layers = nn.ModuleList(MoTDecoderLayer(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(d, c.rms_eps, c.nemotron)
        self.norm_moe_gen = RMSNorm(d, c.rms_eps, c.nemotron)
        self.rotary_emb = MRoPE(c)
        self.lm_head = nn.Linear(d, c.vocab_size, bias=False)
        self.proj_in = nn.Linear(c.patch_latent_dim, d)
        self.proj_out = nn.Linear(d, c.patch_latent_dim)
        self.time_embedder = TimestepEmbedding(c.t_freq_dim, d)
        if c.action_gen:
            self.action_proj_in = DomainAwareLinear(c.action_dim, d, c.num_embodiment_domains)
            self.action_proj_out = DomainAwareLinear(d, c.action_dim, c.num_embodiment_domains)
            self.action_modality_embed = nn.Parameter(torch.zeros(d))
        if c.sound_gen:
            self.audio_proj_in = nn.Linear(c.sound_dim, d)
            self.audio_proj_out = nn.Linear(d, c.sound_dim)
            self.audio_modality_embed = nn.Parameter(torch.zeros(d))

    def logits(self, und_hidden):
        """reasoner 다음 토큰 예측 (AR 탑의 LM head)."""
        return self.lm_head(und_hidden)

    # ---- packing
    def _add_time(self, tokens, timesteps, noisy, shapes):
        """noisy 프레임 토큰에만 timestep 임베딩을 더한다. timesteps 는 noisy 토큰 하나당 하나."""
        emb = self.time_embedder(timestep_freqs(timesteps * self.config.timestep_scale, self.config.t_freq_dim)
                                 .to(self.time_embedder.linear_1.weight.dtype)).to(tokens.dtype)
        if not self.config.use_timestep:
            return tokens
        idx, start = [], 0
        for nz, shp in zip(noisy, shapes):
            sp = math.prod(shp[1:])
            idx.append(((nz * sp).unsqueeze(-1) + torch.arange(sp, device=tokens.device) + start).flatten())
            start += shp[0] * sp
        idx = torch.cat(idx).unsqueeze(-1).expand(-1, tokens.shape[1])
        return tokens.scatter_add(0, idx, emb)

    def _patchify(self, latents):
        p, C = self.config.latent_patch_size, self.config.latent_channel
        out, shapes = [], []
        for x in latents:
            x = x.squeeze(0)                                # [C, T, H, W]
            _, T, H, W = x.shape
            shapes.append((T, H, W))
            Hp, Wp = -(-H // p) * p, -(-W // p) * p
            if (Hp, Wp) != (H, W):                          # p 배수가 아니면 오른쪽·아래 0 padding
                pad = torch.zeros((C, T, Hp, Wp), device=x.device, dtype=x.dtype)
                pad[:, :, :H, :W] = x
                x = pad
            x = x.reshape(C, T, Hp // p, p, Wp // p, p)
            out.append(torch.einsum("cthpwq->thwpqc", x).reshape(-1, p * p * C))
        return torch.cat(out), shapes

    def _unpatchify(self, preds, token_shapes, noisy, orig_shapes):
        p, C = self.config.latent_patch_size, self.config.latent_channel
        res, start = [], 0
        for shp, nz, (_, H, W) in zip(token_shapes, noisy, orig_shapes):
            hp, wp = -(-H // p), -(-W // p)
            out = torch.zeros((C, shp[0], H, W), device=preds.device, dtype=preds.dtype)
            n = len(nz) * hp * wp
            if n > 0:
                x = preds[start:start + n].reshape(len(nz), hp, wp, p, p, C)
                x = torch.einsum("thwpqc->cthpwq", x).reshape(C, len(nz), hp * p, wp * p)
                out[:, nz] = x[:, :, :H, :W]
                start += n
            res.append(out.unsqueeze(0))
        return res

    @staticmethod
    def _unpack_rows(preds, shapes, noisy, dim, transpose):
        res, start = [], 0
        for shp, nz in zip(shapes, noisy):
            out = torch.zeros((dim, shp[0]) if transpose else (shp[0], dim), device=preds.device, dtype=preds.dtype)
            if len(nz) > 0:
                if transpose:
                    out[:, nz] = preds[start:start + len(nz)].T
                else:
                    out[nz] = preds[start:start + len(nz)]
                start += len(nz)
            res.append(out)
        return res

    # ---- forward
    def forward(self, input_ids, text_indexes, position_ids, und_len, sequence_length,
                vision_tokens, vision_token_shapes, vision_sequence_indexes, vision_mse_loss_indexes,
                vision_timesteps, vision_noisy_frame_indexes,
                sound_tokens=None, sound_token_shapes=None, sound_sequence_indexes=None, sound_mse_loss_indexes=None,
                sound_timesteps=None, sound_noisy_frame_indexes=None,
                action_tokens=None, action_token_shapes=None, action_sequence_indexes=None,
                action_mse_loss_indexes=None, action_timesteps=None, action_noisy_frame_indexes=None,
                action_domain_ids=None) -> Cosmos3Output:
        c = self.config
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

        txt = self.embed_tokens(input_ids)
        dtype = txt.dtype
        h = txt.new_zeros((sequence_length, c.d_model))
        h[text_indexes] = txt

        v, orig_shapes = self._patchify(vision_tokens)
        v = self._add_time(self.proj_in(v), vision_timesteps, vision_noisy_frame_indexes, vision_token_shapes)
        h[vision_sequence_indexes] = v

        if has_sound:
            s = torch.cat([x[:, :shp[0]].permute(1, 0) for x, shp in zip(sound_tokens, sound_token_shapes)]).to(dtype)
            s = self.audio_proj_in(s) + self.audio_modality_embed
            h[sound_sequence_indexes] = self._add_time(s, sound_timesteps, sound_noisy_frame_indexes, sound_token_shapes)

        if has_action:
            a = torch.cat([x[:shp[0]] for x, shp in zip(action_tokens, action_token_shapes)]).to(dtype)
            dom = torch.cat([d.reshape(1).expand(shp[0]) for d, shp in zip(action_domain_ids, action_token_shapes)])
            a = self.action_proj_in(a, dom.to(a.device)) + self.action_modality_embed
            if action_mse_loss_indexes.numel() > 0:
                a = self._add_time(a, action_timesteps, action_noisy_frame_indexes, action_token_shapes)
            h[action_sequence_indexes] = a

        pos = position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
        cos, sin = self.rotary_emb(pos, h.dtype)
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        rot = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])

        und, gen = h[:und_len], h[und_len:]
        for layer in self.layers:
            und, gen = layer(und, gen, rot)
        und, gen = self.norm(und), self.norm_moe_gen(gen)
        last = torch.cat([und, gen])

        sample = self._unpatchify(self.proj_out(last[vision_mse_loss_indexes]), vision_token_shapes,
                                  vision_noisy_frame_indexes, orig_shapes)
        sound = action = None
        if has_sound:
            sound = self._unpack_rows(self.audio_proj_out(last[sound_mse_loss_indexes]), sound_token_shapes,
                                      sound_noisy_frame_indexes, c.sound_dim, transpose=True)
        if has_action:
            dom = torch.cat([d.reshape(1).expand(len(nz)) for d, nz in zip(action_domain_ids, action_noisy_frame_indexes)])
            pa = self.action_proj_out(last[action_mse_loss_indexes], dom.to(last.device))
            action = self._unpack_rows(pa, action_token_shapes, action_noisy_frame_indexes, c.action_dim, transpose=False)
        return Cosmos3Output(sample, sound, action, und, gen)


# ---------------------------------------------------------------- 검증

def config_from_hf(cfg, **switches) -> Cosmos3Config:
    """diffusers Cosmos3OmniTransformer.config (FrozenDict) 또는 transformer/config.json dict."""
    g = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    axes = g("rope_axes_dim") or (g("rope_scaling") or {}).get("mrope_section") or [24, 20, 20]
    return Cosmos3Config(
        vocab_size=g("vocab_size", 151936), d_model=g("hidden_size", 4096), n_layer=g("num_hidden_layers", 36),
        n_head=g("num_attention_heads", 32), n_kv_head=g("num_key_value_heads", 8), head_dim=g("head_dim", 128),
        d_ff=g("intermediate_size", 12288), hidden_act=g("hidden_act", "silu"),
        attention_bias=g("attention_bias", False), rms_eps=g("rms_norm_eps", 1e-6),
        qk_norm_for_text=g("qk_norm_for_text", True), use_und_k_norm_for_gen=g("use_und_k_norm_for_gen", False),
        rope_theta=float(g("rope_theta", 5e6)), rope_axes_dim=tuple(axes),
        latent_channel=g("latent_channel", 48), latent_patch_size=g("latent_patch_size", 2),
        patch_latent_dim=g("patch_latent_dim", 192), timestep_scale=g("timestep_scale", 0.001),
        action_gen=bool(g("action_gen", False)), action_dim=g("action_dim"),
        num_embodiment_domains=g("num_embodiment_domains", 32),
        sound_gen=bool(g("sound_gen", False)), sound_dim=g("sound_dim"), **switches)


def _flat_any(o) -> torch.Tensor:
    """원본 출력(Cosmos3OmniTransformerOutput 또는 stage 의 (flat, out) tuple) → 한 줄 텐서."""
    if isinstance(o, tuple) and torch.is_tensor(o[0]):
        return o[0]
    return flat_preds(o.sample, o.sound, o.action)


def verify_pair(ref) -> dict:
    """ref = diffusers Cosmos3OmniTransformer (이름이 같아 name_map 불필요)"""
    c = config_from_hf(ref.config)
    mine = Cosmos3(c)
    pairs = [("rotary_emb", "rotary_emb"), ("proj_in", "proj_in"), ("time_embedder", "time_embedder")]
    for i in range(c.n_layer):
        pairs += [(f"layers.{i}.self_attn.to_out", f"layers.{i}.self_attn.to_out"),
                  (f"layers.{i}.self_attn.to_add_out", f"layers.{i}.self_attn.to_add_out"),
                  (f"layers.{i}.mlp_moe_gen", f"layers.{i}.mlp_moe_gen"),
                  (f"layers.{i}", f"layers.{i}")]          # tuple 출력 → 첫 텐서(reasoner) 비교
    pairs += [("norm", "norm"), ("norm_moe_gen", "norm_moe_gen"), ("proj_out", "proj_out")]
    if c.sound_gen:
        pairs.append(("audio_proj_out", "audio_proj_out"))
    if c.action_gen:
        pairs += [("action_proj_in", "action_proj_in"), ("action_proj_out", "action_proj_out")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs, out_fn_ref=_flat_any, out_fn_mine=lambda o: o.flat())
