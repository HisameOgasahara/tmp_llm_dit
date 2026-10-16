"""A6a · Qwen3.6 하이브리드: Gated DeltaNet (선형 attention) 3 : Gated Attention 1.

HF 원본: transformers/models/qwen3_5 (Qwen3.5 / 3.6 / 3.8-27B 가 이 구조를 쓴다).
플래그십 Qwen3.8-2.4T-A95B (qwen3_5_moe) 는 같은 하이브리드에서 MLP 만 MoE 로 바뀐다 → n_experts > 0.
파라미터 이름은 HF 와 같다.

핵심 부품
- Gated DeltaNet: 고정 크기 메모리 행렬 S [Dk, Dv] 를 토큰마다 갱신한다.
      S ← S · exp(g)                       (망각: g ≤ 0, 토큰·head 마다 다름)
      S ← S + k ⊗ β (v − Sᵀk)              (delta rule: 이미 기억하는 부분을 빼고 차이만 기록)
      o = Sᵀ q
  softmax attention 과 달리 시퀀스 길이와 무관한 상태 크기 → 긴 문맥에 싸지만 기억 용량이 유한하다.
- Gated Attention: 일반 softmax attention 출력에 head 별 sigmoid 게이트를 곱한다 (+ QK-norm, partial RoPE).
- Zero-centered RMSNorm: weight 를 0 으로 두고 (1 + w) 를 곱한다.
- MoE (플래그십): softmax 점수 → top-k → 고른 점수끼리 다시 정규화, + sigmoid 게이트를 곱한 shared expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, attention, rotate_half


@dataclass
class Qwen36Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    layer_types: list[str] | None = None      # None 이면 full_attention_interval 로 생성
    full_attention_interval: int = 4          # 4 → [선형, 선형, 선형, full] 반복
    # Gated Attention (full)
    n_head: int = 4
    n_kv_head: int = 2
    head_dim: int = 64
    partial_rotary: float = 0.25              # head_dim 의 앞 25% 에만 RoPE
    rope_theta: float = 10000.0
    # Gated DeltaNet (linear)
    lin_k_heads: int = 2
    lin_v_heads: int = 4
    lin_k_dim: int = 64
    lin_v_dim: int = 64
    conv_kernel: int = 4
    delta_mode: str = "chunk"                 # "chunk"(학습용, 빠름) / "recurrent"(토큰별, 해석용)
    chunk_size: int = 64
    # MLP / MoE
    d_ff: int = 704                           # dense MLP 폭 (n_experts = 0 일 때)
    n_experts: int = 0                        # 0 = dense (3.6 / 3.8-27B), >0 = MoE (3.8-2.4T)
    top_k: int = 8
    moe_inter: int = 128                      # routed expert 폭
    shared_inter: int = 128                   # shared expert 폭
    use_shared_expert: bool = True            # ablation: shared expert 끄기
    # 공통
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.layer_types is None:
            k = self.full_attention_interval
            self.layer_types = ["linear_attention" if (i + 1) % k else "full_attention" for i in range(self.n_layer)]
        assert len(self.layer_types) == self.n_layer


# ---------------------------------------------------------------- norms

class ZeroCenteredRMSNorm(nn.Module):
    """출력 = norm(x) · (1 + w). w 를 0 으로 초기화 → 시작 시 항등 스케일, weight decay 가 1 쪽으로 당긴다."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """norm(x) · w · silu(z). DeltaNet 출력에 입력 의존 게이트 z 를 건다."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dtype)
        return (h * F.silu(z.float())).to(dtype)


# ---------------------------------------------------------------- Gated delta rule

def l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def delta_rule_recurrent(q, k, v, g, beta, state=None, return_states: bool = False):
    """토큰 하나씩 메모리를 갱신하는 정의 그대로의 구현. 입력 [B, H, T, D], g/beta [B, H, T] (fp32).

    return_states=True 면 매 시점 메모리 S_t [B, H, T, Dk, Dv] 도 돌려준다 (해석용).
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, device=v.device, dtype=v.dtype) if state is None else state
    out, states = torch.zeros_like(v), []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]                     # 망각
        kv_mem = (S * k[:, :, t, :, None]).sum(-2)                    # 지금 기억이 k_t 에 대해 내놓는 값 Sᵀk
        delta = (v[:, :, t] - kv_mem) * beta[:, :, t, None]          # 예측 오차 × 쓰기 강도
        S = S + k[:, :, t, :, None] * delta[:, :, None, :]            # 오차만 기록
        out[:, :, t] = (S * q[:, :, t, :, None]).sum(-2)              # 읽기: Sᵀq
        if return_states:
            states.append(S)
    return out, S, (torch.stack(states, 2) if return_states else None)


def delta_rule_chunk(q, k, v, g, beta, chunk_size: int = 64, state=None):
    """같은 수식을 chunk 단위 행렬곱으로 계산 (HF/FLA 의 chunk 알고리즘). 입력 형식은 recurrent 와 같다.

    chunk 내부: delta rule 의 연쇄 갱신을 하삼각 연립방정식(UT transform)으로 한 번에 푼다.
    chunk 사이: 메모리 S 를 순차적으로 넘긴다.
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    N = (T + pad) // C

    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(B, H, N, C, -1) for x in (q, k, k_beta, v_beta))
    g = g.reshape(B, H, N, C)

    upper = torch.ones(C, C, dtype=torch.bool, device=q.device).triu(1)
    G = g.cumsum(-1)                                                  # chunk 안 누적 망각 (log)
    D = (G[..., :, None] - G[..., None, :]).masked_fill(upper, float("-inf")).exp()  # j→i 사이 망각

    A = (k_beta @ k.transpose(-1, -2)) * D                           # delta rule 연쇄 계수
    attn = (q @ k.transpose(-1, -2)) * D                              # chunk 내부 "attention"
    u = torch.linalg.solve_triangular(A, v_beta, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(A, k_beta * G.exp()[..., None], upper=False, unitriangular=True)

    S = torch.zeros(B, H, Dk, Dv, device=v.device, dtype=v.dtype) if state is None else state
    out = torch.zeros_like(u)
    q = q * G.exp()[..., None]
    k = k * (G[..., -1:] - G).exp()[..., None]
    chunk_decay = G[..., -1].exp()[..., None, None]
    for i in range(N):
        v_new = u[:, :, i] - w[:, :, i] @ S                            # 이전 chunk 기억으로 예측한 부분 제거
        out[:, :, i] = q[:, :, i] @ S + attn[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    return out.reshape(B, H, -1, Dv)[:, :, :T], S


class GatedDeltaNet(nn.Module):
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.c = c
        self.nk, self.nv, self.dk, self.dv = c.lin_k_heads, c.lin_v_heads, c.lin_k_dim, c.lin_v_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(c.d_model, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.d_model, self.value_dim, bias=False)    # 출력 게이트
        self.in_proj_b = nn.Linear(c.d_model, self.nv, bias=False)           # 쓰기 강도 β
        self.in_proj_a = nn.Linear(c.d_model, self.nv, bias=False)           # 망각 속도
        # q, k, v 에 짧은 causal depthwise conv (직전 몇 토큰을 섞음)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, c.conv_kernel, groups=conv_dim, bias=False,
                                padding=c.conv_kernel - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.A_log = nn.Parameter(torch.empty(self.nv).uniform_(0.01, 16).log())
        self.norm = GatedRMSNorm(self.dv, c.rms_eps)
        self.out_proj = nn.Linear(self.value_dim, c.d_model, bias=False)
        # hook 지점 (해석용): 망각 g [B, T, Hv], 쓰기 강도 β [B, T, Hv]
        self.decay_probe = nn.Identity()
        self.beta_probe = nn.Identity()

    def forward(self, x, return_states: bool = False):
        B, T, _ = x.shape
        qkv = self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[..., :T]      # 앞쪽 padding → causal
        qkv = F.silu(qkv).transpose(1, 2)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.nk, self.dk)
        k = k.reshape(B, T, self.nk, self.dk)
        v = v.reshape(B, T, self.nv, self.dv)
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)

        beta = self.beta_probe(self.in_proj_b(x).sigmoid())
        g = self.decay_probe(-self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias))
        if self.nv > self.nk:   # value head 가 더 많으면 q, k 를 공유 (GQA 와 같은 발상)
            q = q.repeat_interleave(self.nv // self.nk, dim=2)
            k = k.repeat_interleave(self.nv // self.nk, dim=2)

        # [B, H, T, D] fp32, q/k 는 L2 정규화 후 q 만 1/sqrt(Dk) 스케일
        q, k, v = (t.transpose(1, 2).float() for t in (q, k, v))
        g, beta = g.transpose(1, 2).float(), beta.transpose(1, 2).float()
        q, k = l2norm(q), l2norm(k)
        q = q * self.dk ** -0.5

        states = None
        if self.c.delta_mode == "recurrent" or return_states:
            o, _, states = delta_rule_recurrent(q, k, v, g, beta, return_states=return_states)
        else:
            o, _ = delta_rule_chunk(q, k, v, g, beta, self.c.chunk_size)
        o = o.transpose(1, 2).to(x.dtype)                                    # [B, T, Hv, Dv]
        o = self.norm(o.reshape(-1, self.dv), z.reshape(-1, self.dv)).reshape(B, T, -1)
        out = self.out_proj(o)
        return (out, states) if return_states else out


# ---------------------------------------------------------------- Gated Attention

def apply_partial_rope(x, cos, sin):
    """head_dim 중 앞 rotary_dim(= cos 폭) 만 회전하고 나머지는 그대로 (위치 정보 없는 채널)."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.h, self.kv, self.d = c.n_head, c.n_kv_head, c.head_dim
        self.q_proj = nn.Linear(c.d_model, self.h * self.d * 2, bias=False)  # head 마다 [query | gate]
        self.k_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.d_model, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.d_model, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.d, c.rms_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.d, c.rms_eps)
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T]
        self.gate_probe = nn.Identity()   # hook 지점: sigmoid 게이트 [B, T, H*D]

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.h, 2 * self.d).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.kv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = attention(q, k, v, causal=True, probs_hook=self.attn_probs)
        y = y.transpose(1, 2).reshape(B, T, -1)
        y = y * self.gate_probe(torch.sigmoid(gate.reshape(B, T, -1)))
        return self.o_proj(y)


# ---------------------------------------------------------------- 블록 / 모델

class MLP(nn.Module):
    def __init__(self, c: Qwen36Config, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or c.d_ff
        self.gate_proj = nn.Linear(c.d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(c.d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """softmax 점수 → top-k → 고른 점수끼리 합 1 로 재정규화 (HF Qwen3_5MoeTopKRouter)."""
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.top_k = c.top_k
        self.weight = nn.Parameter(torch.zeros(c.n_experts, c.d_model))
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]

    def forward(self, x):                  # x [N, D]
        probs = F.linear(x, self.weight).float().softmax(-1)
        w, idx = probs.topk(self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        self.route_probe(idx)
        return w.to(x.dtype), idx


class Experts(nn.Module):
    """expert 가중치를 3D 텐서로 보관 (HF 와 같은 이름·모양: gate_up_proj [E, 2I, D], down_proj [E, D, I])."""
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.n = c.n_experts
        self.gate_up_proj = nn.Parameter(torch.empty(c.n_experts, 2 * c.moe_inter, c.d_model))
        self.down_proj = nn.Parameter(torch.empty(c.n_experts, c.d_model, c.moe_inter))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x, idx, w):          # x [N, D], idx/w [N, k]
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():    # 원본과 같은 순서(expert 번호 순)로 누적
            tok, pos = torch.where(idx == e)
            gate, up = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e]) * w[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out


class SparseMoE(nn.Module):
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.c = c
        self.gate = Router(c)
        self.experts = Experts(c)
        self.shared_expert = MLP(c, c.shared_inter)
        self.shared_expert_gate = nn.Linear(c.d_model, 1, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        w, idx = self.gate(x)
        out = self.experts(x, idx, w)
        if self.c.use_shared_expert:
            out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return out.view(B, T, D)


class Layer(nn.Module):
    def __init__(self, c: Qwen36Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = GatedAttention(c)
        self.input_layernorm = ZeroCenteredRMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(c.d_model, c.rms_eps)
        self.mlp = SparseMoE(c) if c.n_experts else MLP(c)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        h = self.linear_attn(h) if self.layer_type == "linear_attention" else self.self_attn(h, cos, sin)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen36(nn.Module):
    def __init__(self, c: Qwen36Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, t) for t in c.layer_types)
        self.norm = ZeroCenteredRMSNorm(c.d_model, c.rms_eps)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        rot = int(c.head_dim * c.partial_rotary)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, rot, 2, dtype=torch.float) / rot))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        T = input_ids.shape[1]
        freqs = torch.arange(T, device=input_ids.device).float()[:, None] * self.inv_freq[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        h = self.norm(x)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> Qwen36Config:
    rp = getattr(c, "rope_parameters", None) or {}
    return Qwen36Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers, layer_types=list(c.layer_types),
        n_head=c.num_attention_heads, n_kv_head=c.num_key_value_heads, head_dim=c.head_dim,
        partial_rotary=rp.get("partial_rotary_factor", 0.25), rope_theta=rp.get("rope_theta", 10000.0),
        lin_k_heads=c.linear_num_key_heads, lin_v_heads=c.linear_num_value_heads,
        lin_k_dim=c.linear_key_head_dim, lin_v_dim=c.linear_value_head_dim, conv_kernel=c.linear_conv_kernel_dim,
        d_ff=getattr(c, "intermediate_size", None) or 704, rms_eps=c.rms_norm_eps,
        n_experts=getattr(c, "num_experts", 0), top_k=getattr(c, "num_experts_per_tok", 8),
        moe_inter=getattr(c, "moe_intermediate_size", 128),
        shared_inter=getattr(c, "shared_expert_intermediate_size", 128))


def verify_pair(ref) -> dict:
    """ref = transformers Qwen3_5TextModel / Qwen3_5MoeTextModel (이름이 같아 name_map 불필요)"""
    mine = Qwen36(config_from_hf(ref.config))
    pairs = []
    for i in range(ref.config.num_hidden_layers):
        pairs += [(f"layers.{i}.mlp", f"layers.{i}.mlp"), (f"layers.{i}", f"layers.{i}")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
