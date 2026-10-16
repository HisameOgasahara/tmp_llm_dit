"""A9 · StepFun Step-5 텍스트 backbone (Step-5-Preview).

원본: HF stepfun-ai/Step-5-Preview-BF16 `inference/model.py` + `kernel.py` (transformers main 에는 아직 없음).
파라미터 이름은 원본(= 실제 체크포인트 키 model.* / lm_head.*)과 같다. vision encoder 와 MTP 층은 뺐다.

블록 구성 (한 층 = pre-norm residual 두 번, residual 은 fp32 로 들고 다님)
1. Attention — 층 종류가 둘이다 (실제 설정: sliding 3 : full 1 반복).
   공통: GQA(q head 64 / kv group 4), head 마다 zero-centered RMSNorm(x·(1+w)) 을 q, k 에 걸고
         head 앞쪽 일부 채널만 NEOX 방식(d ↔ d + span/2 쌍)으로 회전하는 partial RoPE.
         회전 비율·theta 는 층마다 다르다 (sliding 1.0 / 1e4, full 1/3 / 1e7). llama3 rope scaling 은 full 층에만.
         출력은 head 별 sigmoid gate(g_proj) 를 곱한 뒤 o_proj.
   - sliding : 최근 window 개 토큰만 보는 causal attention.
   - full    : DSA(sparse) — 토큰을 region(8개) 단위로 묶어 고른 region 만 본다.
       indexer: 작은 q(head 16 = kv group 4 × 4), 단일 k, 압축 logit z 를 hidden 에서 바로 투영.
                q 는 zero-centered RMSNorm, k 는 LayerNorm, 둘 다 앞 rope_dim 채널만 RoPE.
       region 요약: region 안 토큰의 k 를 채널마다 softmax(z) 가중평균 (완성된 region 만).
       점수: kv group 마다 Σ_h w_h·ReLU(q_h·요약) (w = 투영 · (group 당 head 수)^-0.5).
       선택: 질의가 속한 region 이전의 region 중 점수 top-k (동점은 region 번호가 작은 쪽) + 현재 region(causal 부분).
       kv group 마다 고른 region 이 달라 같은 group 의 q head 들만 같은 칸을 본다.
2. FFN — moe_layer_list 에 없는 앞쪽 층은 dense clamped SwiGLU, 나머지는 MoE + shared expert.
   clamped SwiGLU: silu(gate).clamp(max=L) · up.clamp(-L, L) (L = swiglu_limits[층]).
   MoE: fp32 sigmoid 점수 + router_bias 로 top-k 선택, 가중치는 bias 뺀 점수를 합 1 로 정규화 × scaling.
        routed 와 shared 는 각각 계산해 fp32 로 더한다.

원본은 요청들을 이어 붙인 packed 입력([총 토큰])을 쓰지만 여기서는 [B, T] 로 계산한다.
causal 모델이라 오른쪽 padding 은 앞쪽 출력에 영향이 없으므로, seq_lens 를 주면 padding 위치 출력만 0 으로 만든다.
비교용 hook 은 probe.* (packed [총 토큰, D] 로 다시 모은 값) 에 걸린다.
원본 설정의 head_dim_full/sliding_attention 등 층 종류별 GQA 키는 원본 코드가 읽지 않는다 (값도 같음) → 하나로 둔다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput

SLIDING, FULL = "sliding_attention", "full_attention"


@dataclass
class Step5ModelConfig:
    vocab_size: int = 128
    d_model: int = 64
    n_layer: int = 4
    layer_types: list = field(default_factory=lambda: [SLIDING, SLIDING, SLIDING, FULL])
    # attention
    n_head: int = 4
    n_kv: int = 2                         # num_attention_groups
    head_dim: int = 24
    rope_theta: list = field(default_factory=lambda: [1e4, 1e4, 1e4, 1e7])
    partial_rotary: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1 / 3])
    rope_scaling: dict | None = None      # {"rope_type": "llama3", ...}
    yarn_only_types: list = field(default_factory=lambda: [FULL])
    window: int = 8
    rms_eps: float = 1e-5
    attn_gate: bool = True                # use_head_wise_attn_gate (g_proj 유무)
    # DSA (sparse_config)
    sparse: bool = True
    sparse_types: list = field(default_factory=lambda: [FULL])
    proxy_dim: int = 32
    index_rope_dim: int = 8
    index_heads: int = 4                  # 전체 indexer q head (= kv group × group 당 head)
    index_topk: int = 3
    region: int = 4
    # FFN
    d_ff: int = 96                        # dense 층 intermediate_size
    moe_layers: list = field(default_factory=lambda: [1, 2, 3])
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 32
    shared_inter: int = 48
    router_scale: float = 3.0
    norm_expert_weight: bool = True
    swiglu_limits: list = field(default_factory=list)          # 비면 7.0
    swiglu_limits_shared: list = field(default_factory=list)
    fp32_residual: bool = True
    # ablation 스위치 (기본값 = 원본)
    use_sparse: bool = True               # False: full 층을 dense causal attention 으로
    use_sliding: bool = True              # False: sliding 층도 window 없이 causal 전체
    use_attn_gate: bool = True            # False: head gate 생략 (g_proj 는 남지만 안 씀)
    use_partial_rope: bool = True         # False: 모든 층에서 head 전체 채널 회전
    use_rope_scaling: bool = True         # False: llama3 scaling 끔
    use_moe: bool = True                  # False: MoE 층도 dense SwiGLU (d_ff)
    use_shared_expert: bool = True
    use_router_bias: bool = True          # False: 선택에 router_bias 안 씀
    use_swiglu_limit: bool = True         # False: clamp 없는 SwiGLU

    def __post_init__(self):
        n = self.n_layer
        assert len(self.layer_types) == n and len(self.rope_theta) >= n and len(self.partial_rotary) >= n
        assert self.n_head % self.n_kv == 0 and self.index_heads % self.n_kv == 0

    def is_sparse(self, i):
        return self.sparse and self.layer_types[i] in self.sparse_types

    def is_moe(self, i):
        return i in self.moe_layers

    def limit(self, i, shared=False):
        lst = self.swiglu_limits_shared if shared else self.swiglu_limits
        return float(lst[i]) if lst else 7.0

    def scaling_for(self, i):
        if self.rope_scaling is None:
            return None
        if self.yarn_only_types and self.layer_types[i] not in self.yarn_only_types:
            return None
        return self.rope_scaling


# ---------------------------------------------------------------- 기본 부품

class ZeroRMSNorm(nn.Module):
    """x / rms(x) · (1 + w), fp32 로 계산 후 입력 dtype 으로 (원본 Step5RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (normed * (self.weight.float() + 1.0)).to(dtype)


def llama3_scale(inv_freq: torch.Tensor, s: dict) -> torch.Tensor:
    """원본 _apply_llama3_scaling: 긴 파장은 factor 로 늘리고, 중간 대역은 보간."""
    factor, lo, hi = float(s["factor"]), float(s["low_freq_factor"]), float(s["high_freq_factor"])
    orig = float(s["original_max_position_embeddings"])
    low_wl, high_wl = orig / lo, orig / hi
    wl = 2 * math.pi / inv_freq
    scaled = torch.where(wl > low_wl, inv_freq / factor, inv_freq)
    smooth = (orig / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    return torch.where((wl >= high_wl) & (wl <= low_wl), smoothed, scaled)


def rope_table(span: int, theta: float, scaling, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [T, span/2]. 원본 build_rope_cache 와 같은 식 (fp32)."""
    inv = 1.0 / (theta ** (torch.arange(0, span, 2, device=device, dtype=torch.float32) / span))
    if scaling:
        inv = llama3_scale(inv, scaling)
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., T, H, D]: 앞 2·half 채널만 (d, d+half) 쌍 회전, 나머지 그대로. cos/sin [T, half]."""
    half = cos.shape[-1]
    if half == 0:
        return x
    c, s = cos[:, None].float(), sin[:, None].float()
    re, im = x[..., :half].float(), x[..., half:2 * half].float()
    rot = torch.cat((re * c - im * s, re * s + im * c), -1).to(x.dtype)
    return torch.cat((rot, x[..., 2 * half:]), -1)


def clamped_swiglu(gate, up, limit):
    """silu(gate).clamp(max=L) · up.clamp(-L, L), fp32 계산. limit=None → clamp 없음."""
    if limit is None:
        return (F.silu(gate.float()) * up.float()).to(gate.dtype)
    return (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(gate.dtype)


# ---------------------------------------------------------------- attention

class Step5Attention(nn.Module):
    def __init__(self, c: Step5ModelConfig, i: int):
        super().__init__()
        self.c, self.i = c, i
        H, G, Dh, D = c.n_head, c.n_kv, c.head_dim, c.d_model
        self.H, self.G, self.Dh = H, G, Dh
        self.q_proj = nn.Linear(D, H * Dh, bias=False)
        self.k_proj = nn.Linear(D, G * Dh, bias=False)
        self.v_proj = nn.Linear(D, G * Dh, bias=False)
        self.o_proj = nn.Linear(H * Dh, D, bias=False)
        self.q_norm = ZeroRMSNorm(Dh, c.rms_eps)
        self.k_norm = ZeroRMSNorm(Dh, c.rms_eps)
        if c.attn_gate:
            self.g_proj = nn.Linear(D, H, bias=False)          # head 마다 scalar gate
        span = int(round(Dh * (c.partial_rotary[i] if c.use_partial_rope else 1.0)))
        self.span = span - span % 2
        self.theta = float(c.rope_theta[i])
        self.scaling = c.scaling_for(i) if c.use_rope_scaling else None
        self.kind = c.layer_types[i]
        self.sparse = c.is_sparse(i)
        if self.sparse:
            P, NI = c.proxy_dim, c.index_heads
            self.sparse_indexer_q = nn.Linear(D, NI * P, bias=False)
            self.sparse_indexer_k = nn.Linear(D, P, bias=False)          # 단일 key head
            self.sparse_indexer_z = nn.Linear(D, P, bias=False)          # region 압축 logit
            self.sparse_indexer_w = nn.Linear(D, NI, bias=False)         # head 별 가중치
            self.sparse_indexer_q_norm = ZeroRMSNorm(P, c.rms_eps)
            self.sparse_indexer_k_norm = nn.LayerNorm(P, eps=c.rms_eps)
            self.register_buffer("ssmax_s", torch.zeros(H))              # 체크포인트에 있지만 안 쓰임
            self.select_probe = nn.Identity()    # hook: 고른 region [B, G, T, region 수] (bool)
            self.score_probe = nn.Identity()     # hook: indexer 점수 [B, G, T, region 수]
        self.attn_probs = nn.Identity()          # hook: attention 확률 [B, H, T, T]

    # ---- DSA indexer
    def _index_norm_rope(self, x, w, b, cos, sin, sub_mean):
        """원본 indexer_norm_rope: head 별 정규화 후 앞 rope_dim 채널 회전.
        q: RMSNorm(1+w), k: LayerNorm (분산 = E[x²] - mean², 원본 커널과 같은 식)."""
        v = x.float()
        n = v.shape[-1]
        ss = (v * v).sum(-1, keepdim=True)
        if sub_mean:
            mean = v.sum(-1, keepdim=True) / n
            y = (v - mean) * torch.rsqrt(ss / n - mean * mean + self.c.rms_eps) * w.float() + b.float()
        else:
            y = v * torch.rsqrt(ss / n + self.c.rms_eps) * (w.float() + 1.0)
        return rope_neox(y.to(x.dtype), cos, sin)

    def select_regions(self, h):
        """질의마다 고른 과거 region mask [B, G, T, ceil(T/R)] (현재 region 은 포함 안 함)."""
        c = self.c
        B, T, _ = h.shape
        G, P, R = self.G, c.proxy_dim, c.region
        hpg = c.index_heads // G
        cos, sin = rope_table(c.index_rope_dim, self.theta, self.scaling, T, h.device)
        iq = self.sparse_indexer_q(h).view(B, T, c.index_heads, P)
        ik = self.sparse_indexer_k(h).view(B, T, 1, P)
        iz = self.sparse_indexer_z(h).view(B, T, P)
        qn = self.sparse_indexer_q_norm
        iq = self._index_norm_rope(iq, qn.weight.to(iq.dtype), None, cos, sin, False)
        kn = self.sparse_indexer_k_norm
        ik = self._index_norm_rope(ik, kn.weight.to(ik.dtype), kn.bias.to(ik.dtype), cos, sin, True)[:, :, 0]
        wts = F.linear(h, self.sparse_indexer_w.weight.to(h.dtype)).float().view(B, T, G, hpg) * float(hpg) ** -0.5
        iq = iq.view(B, T, G, hpg, P).float()

        nR, nRp = T // R, -(-T // R)
        sel_full = torch.zeros(B, G, T, nRp, dtype=torch.bool, device=h.device)
        if nR == 0:
            return sel_full
        # region 요약: 채널마다 softmax(z) 가중평균 (shift 는 region 안 전체 최댓값 하나)
        kr = ik[:, :nR * R].reshape(B, nR, R, P).float()
        zr = iz[:, :nR * R].reshape(B, nR, R, P).float()
        e = torch.exp(zr - zr.amax(dim=(2, 3), keepdim=True))
        summ = (e * kr).sum(2) / e.sum(2).clamp_min(1e-20)                     # [B, nR, P]
        # 점수: Σ_h w_h · ReLU(q_h · 요약)
        score = torch.zeros(B, G, T, nR, device=h.device)
        for j in range(hpg):
            s = torch.einsum("btgp,brp->bgtr", iq[:, :, :, j], summ)
            score = score + s.clamp_min(0) * wts[:, :, :, j].permute(0, 2, 1)[..., None]
        self.score_probe(score)
        # 선택: 현재 region 이전 region 중 top-k, 동점은 번호 작은 쪽 (stable 정렬)
        visible = torch.arange(T, device=h.device) // R                          # [T]
        valid = torch.arange(nR, device=h.device)[None, :] < visible[:, None]   # [T, nR]
        key = score.masked_fill(~valid, float("-inf"))
        order = torch.sort(key, dim=-1, descending=True, stable=True).indices
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(nR, device=h.device).expand_as(order).contiguous())
        sel = valid & (rank < visible.clamp_max(c.index_topk)[:, None])
        sel_full[..., :nR] = sel
        return self.select_probe(sel_full)

    def forward(self, h):
        c = self.c
        B, T, _ = h.shape
        H, G, Dh = self.H, self.G, self.Dh
        q = self.q_norm(self.q_proj(h).view(B, T, H, Dh))
        k = self.k_norm(self.k_proj(h).view(B, T, G, Dh))
        v = self.v_proj(h).view(B, T, G, Dh)
        cos, sin = rope_table(self.span, self.theta, self.scaling, T, h.device)
        q, k = rope_neox(q, cos, sin), rope_neox(k, cos, sin)

        pos = torch.arange(T, device=h.device)
        dist = pos[:, None] - pos[None, :]
        allowed = (dist >= 0)[None, None]                                       # [1, 1, T, T]
        if self.sparse and c.use_sparse:
            sel = self.select_regions(h)                                        # [B, G, T, nRp]
            r = pos // c.region
            same = (r[:, None] == r[None, :])[None, None]
            past = sel.gather(-1, r.view(1, 1, 1, T).expand(B, G, T, T))        # 고른 region 의 토큰
            allowed = allowed & (same | past)
            allowed = allowed.repeat_interleave(H // G, dim=1)                  # group → q head
        elif self.kind == SLIDING and c.use_sliding:
            allowed = allowed & (dist < c.window)[None, None]

        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(H // G, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(H // G, dim=1)
        s = (qh.float() @ kh.float().transpose(-2, -1)) * Dh ** -0.5
        p = self.attn_probs(s.masked_fill(~allowed, float("-inf")).softmax(-1))
        o = (p @ vh.float()).to(q.dtype).transpose(1, 2)                         # [B, T, H, Dh]
        if c.attn_gate and c.use_attn_gate:
            o = o * self.g_proj(h).sigmoid().unsqueeze(-1).to(o.dtype)
        return self.o_proj(o.reshape(B, T, H * Dh))


# ---------------------------------------------------------------- FFN

class Step5MLP(nn.Module):
    def __init__(self, d: int, inter: int, limit):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)
        self.limit = limit

    def forward(self, x):
        return self.down_proj(clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit))


class StackedExpertWeight(nn.Module):
    """expert 가중치를 [E, out, in] 하나로 (체크포인트 moe.gate_proj.weight 와 같은 모양)."""

    def __init__(self, n: int, out: int, inp: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, out, inp) * inp ** -0.5)


class Step5MoE(nn.Module):
    """sigmoid router (fp32) + 선택용 bias. routed expert 출력만 돌려준다 (shared 는 층이 더함)."""

    def __init__(self, c: Step5ModelConfig, i: int):
        super().__init__()
        self.c = c
        D, E = c.d_model, c.n_experts
        self.gate = nn.Linear(D, E, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.gate_proj = StackedExpertWeight(E, c.moe_inter, D)
        self.up_proj = StackedExpertWeight(E, c.moe_inter, D)
        self.down_proj = StackedExpertWeight(E, D, c.moe_inter)
        self.limit = c.limit(i) if c.use_swiglu_limit else None
        self.route_probe = nn.Identity()     # hook: 고른 expert [N, k]

    def route(self, x):
        c = self.c
        prob = torch.sigmoid(F.linear(x.float(), self.gate.weight.float()))
        if c.use_router_bias:
            top, ids = torch.topk(prob + self.router_bias.view(1, -1), c.top_k, dim=-1)
            w = top - self.router_bias[ids]           # 원본과 같이 (p + b) - b
        else:
            w, ids = torch.topk(prob, c.top_k, dim=-1)
        if c.norm_expert_weight:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return w * c.router_scale, self.route_probe(ids)

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        w, ids = self.route(x)
        slot = torch.zeros(x.shape[0], self.c.top_k, shp[-1], device=x.device, dtype=x.dtype)
        for e in range(self.c.n_experts):
            rows, slots = (ids == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            t = x[rows]
            hdn = clamped_swiglu(F.linear(t, self.gate_proj.weight[e]), F.linear(t, self.up_proj.weight[e]), self.limit)
            slot[rows, slots] = F.linear(hdn, self.down_proj.weight[e]).to(x.dtype)
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        for k in range(self.c.top_k):                 # slot 순서대로 fp32 누적 (원본 weighted_topk_gather)
            out = out + slot[:, k].float() * w[:, k].unsqueeze(-1)
        return out.to(x.dtype).view(shp)


# ---------------------------------------------------------------- 층 / 모델

class Step5DecoderLayer(nn.Module):
    def __init__(self, c: Step5ModelConfig, i: int):
        super().__init__()
        self.c = c
        self.self_attn = Step5Attention(c, i)
        self.input_layernorm = ZeroRMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = ZeroRMSNorm(c.d_model, c.rms_eps)
        lim = (lambda s: c.limit(i, s) if c.use_swiglu_limit else None)
        self.is_moe = c.is_moe(i) and c.use_moe
        if self.is_moe:
            self.moe = Step5MoE(c, i)
            if c.use_shared_expert:
                self.share_expert = Step5MLP(c.d_model, c.shared_inter, lim(True))
        else:
            self.mlp = Step5MLP(c.d_model, c.d_ff, lim(False))
        # hook 지점: 원본 packed 출력과 비교하려고 padding 을 뺀 [총 토큰, D]
        self.probe = nn.ModuleDict({k: nn.Identity() for k in ("attn", "moe", "shared", "mlp", "out")})

    def forward(self, residual, pack):
        h = self.input_layernorm(residual).to(self.self_attn.q_proj.weight.dtype)
        a = self.self_attn(h)
        self.probe["attn"](pack(a))
        residual = residual + a.float()
        h = self.post_attention_layernorm(residual).to(a.dtype)
        if self.is_moe:
            f = self.moe(h)
            self.probe["moe"](pack(f))
            f = f.float()
            if hasattr(self, "share_expert"):
                s = self.share_expert(h)
                self.probe["shared"](pack(s))
                f = f + s.float()
        else:
            f = self.mlp(h)
            self.probe["mlp"](pack(f))
        out = residual + f.float()
        self.probe["out"](pack(out))
        return out


class Step5Model(nn.Module):
    def __init__(self, c: Step5ModelConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Step5DecoderLayer(c, i) for i in range(c.n_layer))
        self.norm = ZeroRMSNorm(c.d_model, c.rms_eps)
        self.norm_probe = nn.Identity()

    def forward(self, ids, pack):
        residual = self.embed_tokens(ids).float()
        for layer in self.layers:
            residual = layer(residual, pack)
        x = residual.to(self.embed_tokens.weight.dtype) if self.c.fp32_residual else residual
        h = self.norm(x)
        self.norm_probe(pack(h))
        return h


class Step5(nn.Module):
    """Step5ForCausalLM 텍스트 부분. forward(input_ids [B, T], seq_lens=None) -> LMOutput."""

    def __init__(self, c: Step5ModelConfig):
        super().__init__()
        self.config = c
        self.model = Step5Model(c)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, input_ids, seq_lens=None):
        B, T = input_ids.shape
        if seq_lens is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)
        else:
            mask = torch.arange(T, device=input_ids.device)[None] < seq_lens.to(input_ids.device).view(-1, 1)
        h = self.model(input_ids, lambda x: x[mask])
        logits = self.lm_head(h.to(self.lm_head.weight.dtype)).float()
        if seq_lens is not None:   # 오른쪽 padding 위치는 0 (원본은 그 토큰을 아예 계산하지 않음)
            h = h.masked_fill(~mask[..., None], 0)
            logits = logits.masked_fill(~mask[..., None], 0)
        return LMOutput(last_hidden_state=h, logits=logits)


# ---------------------------------------------------------------- 원본 config / 검증 쌍

def _get(c, *names, default=None):
    for n in names:
        v = c.get(n) if isinstance(c, dict) else getattr(c, n, None)
        if v is not None:
            return v
    return default


def config_from_hf(c, **overrides) -> Step5ModelConfig:
    """원본 Step5Config 객체 또는 config.json(dict, text_config 포함 가능)을 받는다. MTP 항목은 잘라낸다."""
    c = _get(c, "text_config", default=c)
    n = _get(c, "num_hidden_layers")
    per = lambda v, d: (list(v) if isinstance(v, (list, tuple)) else [float(v)] * n)[:n] if v is not None else [d] * n
    sp = _get(c, "sparse_config")
    sg = (lambda k, d=None: _get(sp, k, default=d)) if sp is not None else (lambda k, d=None: d)
    use_moe = _get(c, "use_moe", default=True)
    kw = dict(
        vocab_size=_get(c, "vocab_size"), d_model=_get(c, "hidden_size"), n_layer=n,
        layer_types=list(_get(c, "layer_types"))[:n],
        n_head=_get(c, "num_attention_heads"), n_kv=_get(c, "num_attention_groups"), head_dim=_get(c, "head_dim"),
        rope_theta=per(_get(c, "rope_theta"), 10000.0), partial_rotary=per(_get(c, "partial_rotary_factors"), 1.0),
        rope_scaling=_get(c, "rope_scaling"), yarn_only_types=list(_get(c, "yarn_only_types", default=[])),
        window=_get(c, "sliding_window", default=512), rms_eps=_get(c, "rms_norm_eps", default=1e-5),
        attn_gate=bool(_get(c, "use_head_wise_attn_gate", default=True)),
        sparse=bool(sp is not None and sg("enabled", True)),
        sparse_types=list(sg("apply_to_layer_types", None) or [FULL]),
        proxy_dim=sg("proxy_dim", 256), index_rope_dim=sg("sparse_indexer_rope_dim", 32),
        index_heads=sg("sparse_indexer_num_heads", 16), index_topk=sg("topk", 512), region=sg("region_block_size", 8),
        d_ff=_get(c, "intermediate_size"), moe_layers=list(_get(c, "moe_layer_list", default=[])) if use_moe else [],
        n_experts=_get(c, "moe_num_experts"), top_k=_get(c, "moe_top_k", "num_experts_per_tok"),
        moe_inter=_get(c, "moe_intermediate_size"), shared_inter=_get(c, "share_expert_dim"),
        router_scale=float(_get(c, "moe_router_scaling_factor", default=1.0)),
        norm_expert_weight=bool(_get(c, "norm_expert_weight", default=True)),
        swiglu_limits=list(_get(c, "swiglu_limits", default=[]) or [])[:n],
        swiglu_limits_shared=list(_get(c, "swiglu_limits_shared", default=[]) or [])[:n],
        fp32_residual=bool(_get(c, "fp32_residual_connection", default=True)))
    kw.update(overrides)
    return Step5ModelConfig(**kw)


def verify_pair(ref) -> dict:
    """ref = 원본 Step5ForCausalLM (stage 모듈이 forward 를 [B, T] 입력용으로 감싸 둠). 이름이 같아 name_map 불필요.

    원본 층 출력은 packed [총 토큰, D] 이므로 내 쪽은 probe.* (padding 뺀 값) 와 비교한다.
    최종 출력은 양쪽 모두 [B, T, V] logits (padding 위치 0).
    """
    mine = Step5(config_from_hf(ref.config))
    pairs = []
    for i, layer in enumerate(mine.model.layers):
        p = f"model.layers.{i}"
        pairs.append((f"{p}.self_attn", f"{p}.probe.attn"))
        if layer.is_moe:
            pairs += [(f"{p}.moe", f"{p}.probe.moe"), (f"{p}.share_expert", f"{p}.probe.shared")]
        else:
            pairs.append((f"{p}.mlp", f"{p}.probe.mlp"))
        pairs.append((p, f"{p}.probe.out"))
    pairs.append(("model.norm", "model.norm_probe"))
    return dict(mine=mine, name_map=None, hook_pairs=pairs, out_fn_mine=lambda o: o.logits)
