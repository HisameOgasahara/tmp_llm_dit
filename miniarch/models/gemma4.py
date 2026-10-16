"""A10 · Gemma 4 텍스트 backbone: sliding 5 : global 1 하이브리드 + 경량화 부품들 (PLE, KV 공유, MoE 병렬 블록).

HF 원본: transformers/models/gemma4 (Gemma4TextModel, gemma-4-31B / 26B-A4B / E4B / E2B 공통),
         transformers/models/gemma4_unified (gemma-4-12B, 텍스트 부분은 PLE·MoE 없는 같은 계산).
파라미터 이름은 HF 와 같다 (layer_scalar 는 HF 처럼 persistent buffer).

한 층의 흐름 (Gemma 의 "sandwich" norm: 부품 앞뒤로 RMSNorm)
    h = h + post_attn_norm(Attn(input_norm(h)))
    h = h + post_ff_norm( MLP(pre_ff_norm(h))  [+ MoE 병렬 가지] )
    h = h + post_ple_norm(W_up(gelu(W_gate h) * ple_i))      ← E 계열만 (Per-Layer Embeddings)
    h = h * layer_scalar

Attention
- 층 종류가 둘: sliding(창 W 안만 봄, head_dim 256) / global(전체, head_dim 512, KV head 수도 따로).
- QK-norm (가중치 있는 RMSNorm) + V-norm (가중치 없는 RMSNorm). 그래서 softmax 스케일이 1.0 (1/sqrt(d) 없음).
- attention_k_eq_v (31B·26B·12B): global 층은 v_proj 가 없고 k_proj 출력을 V 로도 쓴다.
  K = rope(k_norm(k)), V = v_norm(k) → 같은 투영이지만 norm/rope 가 달라서 K ≠ V.
- RoPE 는 층 종류마다 다르다: sliding = 기본 RoPE(θ=1e4), global = "proportional" RoPE(θ=1e6, 25%).
  proportional: 주파수 지수의 분모는 head_dim 전체인데 앞 25% 주파수만 남기고 나머지는 0 (= 회전 없음).
  rotate_half 배치라 회전하는 채널은 [0, r) 와 [d/2, d/2 + r) 두 덩어리다.
- KV 공유 (E 계열): 마지막 num_kv_shared_layers 층은 k/v 투영이 없고, 공유 구간 직전의 같은 종류 층이
  만든 K, V (norm·rope 적용 후) 를 그대로 가져다 쓴다. 이 층들의 MLP 는 2배 폭 (use_double_wide_mlp).

MLP / MoE
- GeGLU: down(gelu_tanh(gate x) * up x).
- 26B-A4B: dense MLP 와 MoE 를 **둘 다** 돌려 더한다. 각 가지 출력에 norm 을 따로 건다.
  router: 가중치 없는 RMSNorm → × scale × d^-0.5 → softmax → top-k → 재정규화 → × per_expert_scale[e].
  MoE 입력은 MLP 입력과 같은 residual 을 별도 norm(pre_feedforward_layernorm_2) 으로 정규화한 것.

PLE (Per-Layer Embeddings, E 계열)
- 토큰마다 층별 작은 벡터 ple_i (dim 256): 토큰 id 전용 임베딩 표 + 주 임베딩의 선형 투영(→ norm) 을 더해 1/√2.
- 각 층 끝에서 hidden 으로 만든 게이트에 ple_i 를 곱해 residual 에 더한다 (층마다 토큰 정보를 다시 주입).

임베딩 × sqrt(d_model), 최종 logit 은 cap·tanh(logit/cap) 로 부드럽게 자른다 (softcapping, cap=30).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput

ACTS = {
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
    "silu": F.silu,
}


def _default_rope():
    return {"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}}


@dataclass
class Gemma4Config:
    vocab_size: int = 128
    d_model: int = 64
    n_layer: int = 6
    layer_types: list[str] | None = None        # None → sliding 5 : full 1 반복, 마지막 층은 항상 full
    # attention
    n_head: int = 4
    n_kv_head: int = 2                          # sliding 층 KV head
    head_dim: int = 16                          # sliding 층 head_dim
    global_head_dim: int = 32                   # full 층 head_dim
    n_global_kv_head: int | None = None         # full 층 KV head (None → n_kv_head). 원본은 k_eq_v 일 때만 따로 둔다
    sliding_window: int = 512
    attention_k_eq_v: bool = False              # full 층: v_proj 없이 k_proj 출력을 V 로
    attention_bias: bool = False
    rope_parameters: dict = field(default_factory=_default_rope)   # 층 종류별 {rope_type, rope_theta, partial_rotary_factor}
    num_kv_shared_layers: int = 0               # 마지막 몇 층이 앞층 K/V 를 공유
    # MLP / MoE
    d_ff: int = 128
    act: str = "gelu_pytorch_tanh"
    double_wide_mlp: bool = False               # KV 공유 층의 MLP 폭 2배
    enable_moe: bool = False                    # dense MLP 옆에 MoE 가지 추가 (26B-A4B)
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 32
    # PLE
    ple_dim: int = 0                            # hidden_size_per_layer_input (0 = 끔)
    ple_vocab: int = 128
    # 기타
    rms_eps: float = 1e-6
    final_logit_softcap: float | None = None
    tie_embeddings: bool = True
    # ablation 스위치 (기본값 = 원본)
    qk_norm: bool = True                        # False: q_norm / k_norm 생략 (스케일 1.0 은 그대로)
    v_norm: bool = True                         # False: V-norm 생략
    sandwich_norm: bool = True                  # False: post_attention / post_feedforward norm 생략 (pre-norm 만)
    scale_embed: bool = True                    # False: 임베딩 × sqrt(d) 생략
    share_kv: bool = True                       # False: 공유 층도 자기 k/v 투영을 가짐 (가중치가 새로 생김)
    use_dense_branch: bool = True               # MoE 층에서 dense MLP 가지 사용
    use_moe_branch: bool = True                 # MoE 층에서 expert 가지 사용
    use_ple: bool = True                        # False: PLE 주입 생략 (가중치는 그대로)
    logit_softcap: bool = True                  # False: logit softcapping 생략

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(self.n_layer)]
        self.layer_types = list(self.layer_types)
        assert len(self.layer_types) == self.n_layer
        if self.layer_types[-1] != "full_attention":   # 원본도 마지막 층을 full 로 강제
            self.layer_types[-1] = "full_attention"
        self.rope_parameters = copy.deepcopy(self.rope_parameters)

    # ---- 층별 해석 (원본 규칙 그대로)
    def is_full(self, i):
        return self.layer_types[i] == "full_attention"

    def layer_head_dim(self, i):
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def layer_kv_heads(self, i):
        if self.is_full(i) and self.n_global_kv_head is not None:
            return self.n_global_kv_head
        return self.n_kv_head

    @property
    def first_shared(self):
        return self.n_layer - self.num_kv_shared_layers

    def attn_is_shared(self, i):                # 원본 attention: i >= first >= 0
        return self.share_kv and i >= self.first_shared >= 0

    def mlp_is_shared(self, i):                 # 원본 MLP: i >= first > 0 (부등호가 attention 과 다름)
        return i >= self.first_shared > 0

    def stores_kv(self, i):
        """공유 구간 직전, 같은 종류의 마지막 층이 K/V 를 넘겨준다."""
        if not self.share_kv or self.attn_is_shared(i):
            return False
        prev = self.layer_types[:self.first_shared]
        return i == len(prev) - 1 - prev[::-1].index(self.layer_types[i])


# ---------------------------------------------------------------- norm / RoPE

class RMSNorm(nn.Module):
    """Gemma4RMSNorm: x · (mean(x²)+eps)^-0.5 · w (fp32). with_scale=False 면 가중치 없음 (V-norm, router)."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            h = h * self.weight.float()
        return h.type_as(x)


def rope_inv_freq(p: dict, head_dim: int) -> torch.Tensor:
    """층 종류별 inv_freq [head_dim/2]."""
    rt = p.get("rope_type", "default")
    base = p["rope_theta"]
    if rt == "default":                          # partial_rotary_factor 는 무시 (원본 Gemma4 기본 함수와 같음)
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    assert rt == "proportional", f"rope_type {rt} 미구현"
    angles = int(p.get("partial_rotary_factor", 1.0) * head_dim // 2)   # 회전하는 주파수 개수
    inv = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / head_dim))
    nope = head_dim // 2 - angles
    if nope > 0:                                 # 나머지 주파수 0 → cos 1, sin 0 (회전 없음)
        inv = torch.cat([inv, torch.zeros(nope, dtype=torch.float32)])
    return inv / p.get("factor", 1.0)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x [B, T, H, D], cos/sin [T, D] (head 축으로 broadcast)"""
    cos, sin = cos[:, None, :], sin[:, None, :]
    return x * cos + rotate_half(x) * sin


class ScaledEmbedding(nn.Embedding):
    """임베딩 × scale (scale 은 weight dtype 으로 변환 후 곱함, 원본과 같음)."""

    def __init__(self, n, d, scale: float):
        super().__init__(n, d)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, ids):
        return super().forward(ids) * self.embed_scale.to(self.weight.dtype)


# ---------------------------------------------------------------- attention

class Attention(nn.Module):
    def __init__(self, c: Gemma4Config, i: int):
        super().__init__()
        self.c = c
        self.layer_type = c.layer_types[i]
        self.is_sliding = not c.is_full(i)
        self.h, self.kv, self.d = c.n_head, c.layer_kv_heads(i), c.layer_head_dim(i)
        self.is_shared = c.attn_is_shared(i)
        self.store_kv = c.stores_kv(i)
        b = c.attention_bias
        self.q_proj = nn.Linear(c.d_model, self.h * self.d, bias=b)
        self.q_norm = RMSNorm(self.d, c.rms_eps)
        if not self.is_shared:                  # 공유 층은 k/v 가중치가 없다
            self.k_norm = RMSNorm(self.d, c.rms_eps)
            self.v_norm = RMSNorm(self.d, c.rms_eps, with_scale=False)
            self.k_proj = nn.Linear(c.d_model, self.kv * self.d, bias=b)
            k_eq_v = c.attention_k_eq_v and not self.is_sliding
            self.v_proj = None if k_eq_v else nn.Linear(c.d_model, self.kv * self.d, bias=b)
        self.o_proj = nn.Linear(self.h * self.d, c.d_model, bias=b)
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T]
        self.k_probe = nn.Identity()      # hook 지점: rope 적용된 K [B, KV, T, D] (공유 층은 받아 온 K)

    def forward(self, x, cos, sin, allow, shared: dict):
        B, T, _ = x.shape
        c = self.c
        q = self.q_proj(x).view(B, T, -1, self.d)
        if c.qk_norm:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin).transpose(1, 2)
        if self.is_shared:
            k, v = shared[self.layer_type]
        else:
            k = self.k_proj(x).view(B, T, -1, self.d)
            v = self.v_proj(x).view(B, T, -1, self.d) if self.v_proj is not None else k   # k_eq_v: 정규화 전 k
            if c.qk_norm:
                k = self.k_norm(k)
            k = apply_rope(k, cos, sin).transpose(1, 2)
            if c.v_norm:
                v = self.v_norm(v)
            v = v.transpose(1, 2)
        if self.store_kv:
            shared[self.layer_type] = (k, v)
        k = self.k_probe(k)

        rep = self.h // k.shape[1]
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * 1.0      # QK-norm 이 스케일을 대신 → scaling = 1
        scores = scores.masked_fill(~allow, float("-inf"))
        probs = self.attn_probs(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype))
        out = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------- MLP / MoE

class MLP(nn.Module):
    def __init__(self, c: Gemma4Config, i: int):
        super().__init__()
        inter = c.d_ff * (2 if c.double_wide_mlp and c.mlp_is_shared(i) else 1)
        self.gate_proj = nn.Linear(c.d_model, inter, bias=False)
        self.up_proj = nn.Linear(c.d_model, inter, bias=False)
        self.down_proj = nn.Linear(inter, c.d_model, bias=False)
        self.act = ACTS[c.act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """가중치 없는 RMSNorm → × scale × d^-0.5 → softmax → top-k 재정규화 → × per_expert_scale."""

    def __init__(self, c: Gemma4Config):
        super().__init__()
        self.top_k = c.top_k
        self.scalar_root_size = c.d_model ** -0.5
        self.norm = RMSNorm(c.d_model, c.rms_eps, with_scale=False)
        self.proj = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(c.d_model))
        self.per_expert_scale = nn.Parameter(torch.ones(c.n_experts))
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]

    def forward(self, x):                  # x [N, D]
        h = self.norm(x)
        h = h * self.scale * self.scalar_root_size
        probs = F.softmax(self.proj(h), dim=-1, dtype=torch.float32)
        w, idx = torch.topk(probs, k=self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        w = w * self.per_expert_scale[idx]
        self.route_probe(idx)
        return probs, w, idx


class Experts(nn.Module):
    """3D 가중치 (HF 와 같은 이름·모양: gate_up_proj [E, 2I, D], down_proj [E, D, I])."""

    def __init__(self, c: Gemma4Config):
        super().__init__()
        self.n = c.n_experts
        self.gate_up_proj = nn.Parameter(torch.randn(c.n_experts, 2 * c.moe_inter, c.d_model) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(c.n_experts, c.d_model, c.moe_inter) * 0.02)
        self.act = ACTS[c.act]

    def forward(self, x, idx, w):          # x [N, D], idx/w [N, k]
        out = torch.zeros_like(x)
        for e in range(self.n):            # 원본과 같은 순서 (expert 번호 → slot → 토큰)
            slot, tok = torch.where(idx.T == e)
            if tok.numel() == 0:
                continue
            g, u = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(self.act(g) * u, self.down_proj[e]) * w[tok, slot, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: Gemma4Config, i: int):
        super().__init__()
        self.c = c
        d, eps = c.d_model, c.rms_eps
        self.self_attn = Attention(c, i)
        self.mlp = MLP(c, i)
        self.input_layernorm = RMSNorm(d, eps)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.pre_feedforward_layernorm = RMSNorm(d, eps)
        self.post_feedforward_layernorm = RMSNorm(d, eps)
        self.register_buffer("layer_scalar", torch.ones(1))     # HF 와 같이 persistent (체크포인트에 있음)
        if c.ple_dim:
            self.per_layer_input_gate = nn.Linear(d, c.ple_dim, bias=False)
            self.per_layer_projection = nn.Linear(c.ple_dim, d, bias=False)
            self.post_per_layer_input_norm = RMSNorm(d, eps)
            self.act = ACTS[c.act]
        if c.enable_moe:
            self.router = Router(c)
            self.experts = Experts(c)
            self.post_feedforward_layernorm_1 = RMSNorm(d, eps)
            self.post_feedforward_layernorm_2 = RMSNorm(d, eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(d, eps)

    def forward(self, h, ple, cos, sin, allow, shared):
        c = self.c
        a = self.self_attn(self.input_layernorm(h), cos, sin, allow, shared)
        if c.sandwich_norm:
            a = self.post_attention_layernorm(a)
        h = h + a

        res = h
        y = self.mlp(self.pre_feedforward_layernorm(res))
        if c.enable_moe:
            y1 = self.post_feedforward_layernorm_1(y)
            flat = res.reshape(-1, res.shape[-1])            # router·expert 입력은 MLP 입력과 같은 residual
            _, w, idx = self.router(flat)
            y2 = self.experts(self.pre_feedforward_layernorm_2(flat), idx, w).reshape(res.shape)
            y2 = self.post_feedforward_layernorm_2(y2)
            if c.use_dense_branch and c.use_moe_branch:
                y = y1 + y2
            else:
                y = y1 if c.use_dense_branch else y2
        if c.sandwich_norm:
            y = self.post_feedforward_layernorm(y)
        h = res + y

        if c.ple_dim and c.use_ple:
            g = self.act(self.per_layer_input_gate(h)) * ple
            h = h + self.post_per_layer_input_norm(self.per_layer_projection(g))
        return h * self.layer_scalar


class Gemma4(nn.Module):
    def __init__(self, c: Gemma4Config):
        super().__init__()
        self.config = c
        d = c.d_model
        self.embed_tokens = ScaledEmbedding(c.vocab_size, d, d ** 0.5 if c.scale_embed else 1.0)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(d, c.rms_eps)
        self.lm_head = None if c.tie_embeddings else nn.Linear(d, c.vocab_size, bias=False)
        if c.ple_dim:
            L, P = c.n_layer, c.ple_dim
            self.embed_tokens_per_layer = ScaledEmbedding(c.ple_vocab, L * P, P ** 0.5)
            self.per_layer_model_projection = nn.Linear(d, L * P, bias=False)
            self.per_layer_projection_norm = RMSNorm(P, c.rms_eps)
        for t in sorted(set(c.layer_types)):
            hd = c.global_head_dim if t == "full_attention" else c.head_dim
            self.register_buffer(f"{t}_inv_freq", rope_inv_freq(c.rope_parameters[t], hd), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def per_layer_inputs(self, ids, x):
        """PLE: (norm(투영(x)·d^-0.5) + 토큰 표(ids)) · 2^-0.5 → [B, T, L, P]"""
        c = self.config
        B, T = ids.shape
        tok = self.embed_tokens_per_layer(ids).reshape(B, T, c.n_layer, c.ple_dim)
        proj = self.per_layer_model_projection(x) * c.d_model ** -0.5
        proj = self.per_layer_projection_norm(proj.reshape(B, T, c.n_layer, c.ple_dim))
        return (proj + tok) * 2.0 ** -0.5

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        c = self.config
        T = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        ple = self.per_layer_inputs(input_ids, x) if c.ple_dim else None

        pos = torch.arange(T, device=x.device).float()
        i, j = pos[:, None], pos[None, :]
        rope, masks = {}, {}
        for t in set(c.layer_types):
            freqs = pos[:, None] * getattr(self, f"{t}_inv_freq").float()[None]
            emb = torch.cat((freqs, freqs), dim=-1)
            rope[t] = (emb.cos().to(x.dtype), emb.sin().to(x.dtype))
            allow = j <= i
            if t == "sliding_attention":
                allow = allow & (i - j < c.sliding_window)
            masks[t] = allow

        h, shared = x, {}
        for n, layer in enumerate(self.layers):
            t = c.layer_types[n]
            h = layer(h, ple[:, :, n] if ple is not None else None, *rope[t], masks[t], shared)
        h = self.norm(h)
        logits = self.lm_head(h) if self.lm_head is not None else F.linear(h, self.embed_tokens.weight)
        if c.final_logit_softcap is not None and c.logit_softcap:
            logits = torch.tanh(logits / c.final_logit_softcap) * c.final_logit_softcap
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def _get(c, k, default=None):
    """gemma4_unified 텍스트 config 는 PLE·MoE 항목 자체가 없다 → 기본값."""
    try:
        v = getattr(c, k)
    except AttributeError:
        return default
    return default if v is None or isinstance(v, BaseException) else v


def config_from_hf(c) -> Gemma4Config:
    """Gemma4TextConfig / Gemma4UnifiedTextConfig → Gemma4Config.

    global 층의 head_dim·KV head 는 per_layer_config 에만 있다 (전역 config 에서 읽으면 원본이 예외를 낸다).
    """
    types = list(c.layer_types)
    n = c.num_hidden_layers
    dims = {}
    for i, t in enumerate(types):
        lc = c.per_layer_config[i]
        dims.setdefault(t, (lc.head_dim, lc.num_key_value_heads))
        assert dims[t] == (lc.head_dim, lc.num_key_value_heads), "같은 종류 층은 head_dim·KV head 가 같아야 함"
    hd_s, kv_s = dims.get("sliding_attention", dims["full_attention"])
    hd_f, kv_f = dims["full_attention"]
    assert _get(c, "use_bidirectional_attention") != "all", "텍스트 causal 경로만 구현"
    rp = copy.deepcopy(dict(c.rope_parameters))
    for t in set(types):
        assert rp[t].get("rope_type", "default") in ("default", "proportional"), rp[t]
    moe = bool(_get(c, "enable_moe_block", False))
    return Gemma4Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=n, layer_types=types,
        n_head=c.num_attention_heads, n_kv_head=kv_s, head_dim=hd_s, global_head_dim=hd_f, n_global_kv_head=kv_f,
        sliding_window=c.sliding_window, attention_k_eq_v=bool(_get(c, "attention_k_eq_v", False)),
        attention_bias=bool(c.attention_bias), rope_parameters={t: rp[t] for t in set(types)},
        num_kv_shared_layers=_get(c, "num_kv_shared_layers", 0),
        d_ff=c.intermediate_size, act=c.hidden_activation, double_wide_mlp=bool(_get(c, "use_double_wide_mlp", False)),
        enable_moe=moe, n_experts=_get(c, "num_experts", 8) if moe else 8,
        top_k=_get(c, "top_k_experts", 2) if moe else 2, moe_inter=_get(c, "moe_intermediate_size", 32) if moe else 32,
        ple_dim=_get(c, "hidden_size_per_layer_input", 0), ple_vocab=_get(c, "vocab_size_per_layer_input", c.vocab_size),
        rms_eps=c.rms_norm_eps, final_logit_softcap=_get(c, "final_logit_softcapping"),
        tie_embeddings=bool(_get(c, "tie_word_embeddings", True)))


def verify_pair(ref) -> dict:
    """ref = Gemma4TextModel / Gemma4UnifiedTextModel (이름이 같아 name_map 불필요)
    또는 Gemma4ForCausalLM 계열 (앞의 "model." 을 떼고 logits 를 비교 → softcapping 까지 검증)."""
    is_lm = hasattr(ref, "lm_head") and hasattr(ref, "model")
    cfg = config_from_hf(ref.config)
    name_map, extra, pre = None, {}, ""
    if is_lm:
        tied = ref.lm_head.weight.data_ptr() == ref.model.embed_tokens.weight.data_ptr()
        cfg.tie_embeddings = tied
        pre = "model."

        def name_map(k):
            if k == "lm_head.weight":
                return None if tied else k
            return k[len(pre):] if k.startswith(pre) else k
        extra = dict(out_fn_ref=lambda o: o.logits, out_fn_mine=lambda o: o.logits)
    mine = Gemma4(cfg)
    pairs = []
    for i in range(cfg.n_layer):
        pairs += [(f"{pre}layers.{i}.self_attn", f"layers.{i}.self_attn"), (f"{pre}layers.{i}.mlp", f"layers.{i}.mlp")]
        if cfg.enable_moe:
            pairs += [(f"{pre}layers.{i}.router", f"layers.{i}.router"),
                      (f"{pre}layers.{i}.experts", f"layers.{i}.experts")]
        pairs.append((f"{pre}layers.{i}", f"layers.{i}"))
    return dict(mine=mine, name_map=name_map, hook_pairs=pairs, **extra)
