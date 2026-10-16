"""A3·A4·A5·A7 · DeepSeek-V4 블록 (V4-Flash / V4-Pro, V4.1 의 뼈대).

HF 원본: transformers/models/deepseek_v4 (순수 PyTorch). 파라미터 이름은 HF 와 같다.
V4.1 전용 부품(Engram, 층 간 KV/indexer 공유)은 deepseek_v41.py 에서 이 위에 얹는다.

블록 구성 (한 층)
1. Attention — 층 종류가 셋이다.
   - sliding : 최근 window 개 토큰만 보는 attention
   - CSA     : sliding + "압축 KV" (토큰 m 개를 1 개로 요약, 이웃 window 와 겹치게) 중
               Lightning Indexer 가 고른 top-k 만 추가로 봄
   - HCA     : sliding + 아주 강하게 압축한 KV (m' 개 → 1 개) 전부를 추가로 봄
   공통: K = V 인 단일 KV head (MQA), q 는 저랭크(q_a → q_b), head 마다 학습되는 attention sink,
        RoPE 는 head 의 뒤쪽 rope_dim 채널에만 (interleaved), 출력도 저랭크 그룹 투영(o_a → o_b).
2. MoE — sqrt(softplus) 점수 + top-k (앞쪽 층은 토큰 id 로 expert 를 고정하는 hash 라우팅) + shared expert.
3. mHC (manifold-constrained Hyper-Connections) — residual 을 hc_mult 개 줄기로 나눠 들고 다니며,
   층마다 (pre: 줄기 → 입력 합치기, post: 출력 → 줄기 퍼뜨리기, comb: 줄기끼리 섞기) 를 입력에서 계산.
   comb 은 Sinkhorn 으로 이중확률행렬로 만들어 norm 이 폭발하지 않게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput

SLIDING, CSA, HCA = "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"


@dataclass
class DeepseekV4Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    layer_types: list[str] | None = None        # None → [HCA, HCA, CSA, HCA, CSA, ...] (HF 기본)
    mlp_layer_types: list[str] | None = None    # None → 앞 n_hash 층 hash_moe, 나머지 moe
    n_hash: int = 1
    # attention
    n_head: int = 4
    head_dim: int = 64
    rope_dim: int = 16                          # head 의 뒤쪽 rope_dim 채널에만 RoPE
    q_lora_rank: int = 64
    o_groups: int = 2
    o_lora_rank: int = 64
    window: int = 16                            # sliding window (자기 자신 포함)
    csa_rate: int = 4                           # CSA: 4 토큰 → 압축 1개 (window 두 개가 겹침)
    hca_rate: int = 16                          # HCA: 16 토큰 → 압축 1개
    rope_theta: float = 10000.0                 # sliding 층
    compress_rope_theta: float = 160000.0       # CSA/HCA 층 (압축 위치는 간격이 넓어서 theta 가 큼)
    index_heads: int = 4
    index_dim: int = 32
    index_topk: int = 8
    # mHC
    hc_mult: int = 4
    hc_iters: int = 20
    hc_eps: float = 1e-6
    # MoE
    n_experts: int = 8
    top_k: int = 2
    moe_inter: int = 128                        # routed / shared expert 의 중간 차원
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # 기타
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    # ablation 스위치 (기본값 = 원본)
    use_mhc: bool = True                        # False: 일반 residual
    use_moe: bool = True                        # False: shared expert 하나만 (dense SwiGLU)

    def __post_init__(self):
        n = self.n_layer
        if self.layer_types is None:
            inter = [CSA if i % 2 else HCA for i in range(max(n - 2, 0))]
            self.layer_types = [HCA] * min(n, 2) + inter
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["hash_moe"] * min(n, self.n_hash) + ["moe"] * max(0, n - self.n_hash)
        assert len(self.layer_types) == n and len(self.mlp_layer_types) == n


# ---------------------------------------------------------------- 기본 부품

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)  # weight 를 fp32 로 곱한 뒤 한 번만 cast


def unweighted_rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def rope_inv_freq(rope_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor):
    """positions [..., T] → cos, sin [..., T, rope_dim/2] (interleaved 쌍마다 하나의 각도)"""
    f = positions[..., None].float() * inv_freq.to(positions.device)
    return f.cos(), f.sin()


def apply_rope_tail(x, cos, sin):
    """x [..., T, D] 의 뒤쪽 rope 채널을 interleaved 방식((x0,x1) 쌍 회전)으로 회전. cos/sin [..., T, r/2]."""
    cos, sin = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
    r = cos.shape[-1]
    nope, rope = x[..., :-r], x[..., -r:]
    rot = torch.stack((-rope[..., 1::2], rope[..., 0::2]), dim=-1).flatten(-2)
    return torch.cat([nope, (rope.float() * cos + rot.float() * sin).to(x.dtype)], dim=-1)


# ---------------------------------------------------------------- KV 압축 / Indexer

def compress_windows(kv, gate, position_bias, rate: int, overlap: bool, dim: int):
    """토큰 rate 개씩 묶어 softmax(gate + bias) 가중합으로 요약 → [B, W, dim].

    overlap=False (HCA): window 하나 = 토큰 rate 개.
    overlap=True  (CSA): kv/gate 가 2*dim 폭. 앞 절반(Ca)은 '다음' window 에, 뒤 절반(Cb)은 '현재' window 에 기여
                         → 압축 하나가 이전 window 와 현재 window 의 2*rate 토큰을 겹쳐서 본다.
    """
    B, T, _ = kv.shape
    W = T // rate
    if W == 0:
        return kv.new_zeros(B, 0, dim)
    kv = kv[:, : W * rate].view(B, W, rate, -1)
    gate = gate[:, : W * rate].view(B, W, rate, -1) + position_bias
    if overlap:
        new_kv = kv.new_zeros(B, W, 2 * rate, dim)
        new_gate = gate.new_full((B, W, 2 * rate, dim), float("-inf"))   # 첫 window 의 '이전' 칸은 weight 0
        new_kv[:, :, rate:], new_gate[:, :, rate:] = kv[..., dim:], gate[..., dim:]
        if W > 1:
            new_kv[:, 1:, :rate], new_gate[:, 1:, :rate] = kv[:, :-1, :, :dim], gate[:, :-1, :, :dim]
        kv, gate = new_kv, new_gate
    return (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)


class Compressor(nn.Module):
    """KV 압축기. CSA 는 indexer 를 함께 가진다."""

    def __init__(self, c: DeepseekV4Config, kind: str):
        super().__init__()
        self.kind = kind
        self.rate = c.csa_rate if kind == CSA else c.hca_rate
        width = 2 * c.head_dim if kind == CSA else c.head_dim
        self.head_dim = c.head_dim
        self.kv_proj = nn.Linear(c.d_model, width, bias=False)
        self.gate_proj = nn.Linear(c.d_model, width, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.rate, width))
        self.kv_norm = RMSNorm(c.head_dim, c.rms_eps)
        self.indexer = Indexer(c) if kind == CSA else None

    def forward(self, x, q_residual, inv_freq):
        """→ (압축 KV [B, W, D], 압축 칸에 대한 mask [B, T, W] (True = 볼 수 있음))"""
        B, T, _ = x.shape
        ckv = compress_windows(self.kv_proj(x), self.gate_proj(x), self.position_bias,
                               self.rate, self.kind == CSA, self.head_dim)
        ckv = self.kv_norm(ckv)
        W = ckv.shape[1]
        cos, sin = rope_cos_sin(torch.arange(W, device=x.device) * self.rate, inv_freq)
        ckv = apply_rope_tail(ckv, cos, sin)
        # 인과성: 질의 t 는 window w 가 완전히 끝난 뒤(t+1 ≥ (w+1)·rate)에만 그 압축을 볼 수 있다
        t = torch.arange(T, device=x.device)
        ready = torch.arange(W, device=x.device)[None, :] < ((t + 1) // self.rate)[:, None]   # [T, W]
        mask = ready.expand(B, T, W)
        if self.indexer is not None and W > 0:
            idx = self.indexer(x, q_residual, inv_freq, ready)                               # [B, T, k]
            sel = torch.zeros(B, T, W + 1, dtype=torch.bool, device=x.device)
            sel.scatter_(-1, torch.where(idx >= 0, idx, W), True)
            mask = sel[..., :W]
        return ckv, mask


class Indexer(nn.Module):
    """Lightning Indexer: 작은 별도 압축 key 와 질의로 점수 Σ_h w_h·ReLU(q_h·k) 를 매겨 top-k 압축 칸만 고른다."""

    def __init__(self, c: DeepseekV4Config):
        super().__init__()
        self.rate, self.heads, self.dim, self.topk = c.csa_rate, c.index_heads, c.index_dim, c.index_topk
        self.kv_proj = nn.Linear(c.d_model, 2 * self.dim, bias=False)
        self.gate_proj = nn.Linear(c.d_model, 2 * self.dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.rate, 2 * self.dim))
        self.kv_norm = RMSNorm(self.dim, c.rms_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, self.heads * self.dim, bias=False)
        self.scorer = nn.Module()
        self.scorer.weights_proj = nn.Linear(c.d_model, self.heads, bias=False)
        self.score_probe = nn.Identity()   # hook 지점: indexer 점수 [B, T, W]

    def forward(self, x, q_residual, inv_freq, ready):
        B, T, _ = x.shape
        k = self.kv_norm(compress_windows(self.kv_proj(x), self.gate_proj(x), self.position_bias,
                                          self.rate, True, self.dim))
        W = k.shape[1]
        cos, sin = rope_cos_sin(torch.arange(W, device=x.device) * self.rate, inv_freq)
        k = apply_rope_tail(k, cos, sin)                                                     # [B, W, d]
        cq, sq = rope_cos_sin(torch.arange(T, device=x.device), inv_freq)
        q = self.q_b_proj(q_residual).view(B, T, self.heads, self.dim).transpose(1, 2)
        q = apply_rope_tail(q, cq, sq).transpose(1, 2)                                       # [B, T, h, d]
        s = F.relu(torch.einsum("bthd,bwd->bthw", q.float(), k.float())) * self.dim ** -0.5
        w = self.scorer.weights_proj(x).float() * self.heads ** -0.5                         # [B, T, h]
        scores = self.score_probe((s * w[..., None]).sum(2))                                 # [B, T, W]
        scores = scores.masked_fill(~ready, float("-inf"))
        idx = scores.topk(min(self.topk, W), dim=-1).indices
        return torch.where(idx >= ((torch.arange(T, device=x.device) + 1) // self.rate)[None, :, None], -1, idx)


# ---------------------------------------------------------------- Attention

class GroupedLinear(nn.Module):
    """head 들을 g 그룹으로 나눠 그룹별로 따로 투영 (block-diagonal). weight [g·out_per_group, in_per_group]."""

    def __init__(self, in_per_group, out_features, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_features, in_per_group))

    def forward(self, x):                                   # x [..., g, in_per_group]
        w = self.weight.view(self.groups, -1, x.shape[-1])  # [g, out/g, in]
        return torch.einsum("...gi,goi->...go", x, w)


class Attention(nn.Module):
    def __init__(self, c: DeepseekV4Config, layer_type: str):
        super().__init__()
        self.c, self.layer_type = c, layer_type
        self.h, self.d = c.n_head, c.head_dim
        self.q_a_proj = nn.Linear(c.d_model, c.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(c.q_lora_rank, c.rms_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, self.h * self.d, bias=False)
        self.kv_proj = nn.Linear(c.d_model, self.d, bias=False)             # K = V, head 1개
        self.kv_norm = RMSNorm(self.d, c.rms_eps)
        self.o_a_proj = GroupedLinear(self.h * self.d // c.o_groups, c.o_groups * c.o_lora_rank, c.o_groups)
        self.o_b_proj = nn.Linear(c.o_groups * c.o_lora_rank, c.d_model, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.h))                      # head 별 "아무것도 안 보기" 로짓
        self.compressor = Compressor(c, layer_type) if layer_type != SLIDING else None
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T + W] (앞 T = 원래 토큰, 뒤 W = 압축 칸)
        self.sink_probe = nn.Identity()   # hook 지점: sink 로 간 확률 [B, H, T]

    def forward(self, x, inv_main, inv_comp):
        B, T, _ = x.shape
        inv = inv_main if self.layer_type == SLIDING else inv_comp
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), inv)
        q_res = self.q_a_norm(self.q_a_proj(x))
        q = unweighted_rmsnorm(self.q_b_proj(q_res).view(B, T, self.h, self.d), self.c.rms_eps).transpose(1, 2)
        q = apply_rope_tail(q, cos, sin)                                    # [B, H, T, D]
        kv = apply_rope_tail(self.kv_norm(self.kv_proj(x)), cos, sin)       # [B, T, D]

        i, j = torch.arange(T, device=x.device)[:, None], torch.arange(T, device=x.device)[None, :]
        allow = ((j <= i) & (i - j < self.c.window)).expand(B, T, T)        # sliding window
        if self.compressor is not None:
            ckv, cmask = self.compressor(x, q_res, inv)
            kv = torch.cat([kv, ckv], dim=1)
            allow = torch.cat([allow, cmask], dim=-1)

        logits = torch.einsum("bhtd,bsd->bhts", q, kv) * self.d ** -0.5
        logits = logits.masked_fill(~allow[:, None], float("-inf"))
        logits = torch.cat([logits, self.sinks.view(1, -1, 1, 1).expand(B, -1, T, 1).to(logits.dtype)], dim=-1)
        probs = (logits - logits.max(-1, keepdim=True).values).softmax(-1)
        self.sink_probe(probs[..., -1])
        probs = self.attn_probs(probs[..., :-1])
        out = torch.einsum("bhts,bsd->bhtd", probs.to(kv.dtype), kv)
        out = apply_rope_tail(out, cos, -sin)          # V 에도 RoPE 가 들어갔으므로 질의 위치로 되돌려 상대위치화
        out = out.transpose(1, 2).reshape(B, T, self.c.o_groups, -1)
        return self.o_b_proj(self.o_a_proj(out).flatten(2))


# ---------------------------------------------------------------- MoE

class SwiGLU(nn.Module):
    def __init__(self, d, inter, limit):
        super().__init__()
        self.limit = limit
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        g = self.gate_proj(x).clamp(max=self.limit)
        u = self.up_proj(x).clamp(-self.limit, self.limit)
        return self.down_proj(F.silu(g) * u)


class Router(nn.Module):
    def __init__(self, c: DeepseekV4Config, hashed: bool):
        super().__init__()
        self.c, self.hashed = c, hashed
        self.weight = nn.Parameter(torch.empty(c.n_experts, c.d_model))
        nn.init.normal_(self.weight, std=0.02)
        if hashed:   # 토큰 id → expert 고정 표 (학습 안 함)
            self.register_buffer("tid2eid", torch.randint(0, c.n_experts, (c.vocab_size, c.top_k)))
        else:        # aux-loss-free 부하 분산용 bias (선택에만 쓰고 가중치에는 안 씀)
            self.register_buffer("e_score_correction_bias", torch.zeros(c.n_experts))
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]
        self._last_idx = None

    def forward(self, x, input_ids):
        scores = F.softplus(F.linear(x, self.weight)).sqrt()                 # sqrt(softplus) 점수
        if self.hashed:
            idx = self.tid2eid[input_ids.reshape(-1)].long()
        else:
            idx = torch.topk(scores + self.e_score_correction_bias, self.c.top_k, dim=-1, sorted=False).indices
        self._last_idx = self.route_probe(idx).detach()
        w = scores.gather(1, idx)
        return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * self.c.route_scale


class Experts(nn.Module):
    def __init__(self, c: DeepseekV4Config):
        super().__init__()
        self.c = c
        self.gate_up_proj = nn.Parameter(torch.randn(c.n_experts, 2 * c.moe_inter, c.d_model) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(c.n_experts, c.d_model, c.moe_inter) * 0.02)

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        L = self.c.swiglu_limit
        for e in range(self.c.n_experts):
            tok, slot = torch.where(idx == e)
            if tok.numel() == 0:
                continue
            g, u = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.silu(g.clamp(max=L)) * u.clamp(-L, L)
            out.index_add_(0, tok, (F.linear(h, self.down_proj[e]) * w[tok, slot, None]).to(out.dtype))
        return out


class MoE(nn.Module):
    def __init__(self, c: DeepseekV4Config, mlp_type: str):
        super().__init__()
        self.c = c
        if c.use_moe:
            self.gate = Router(c, hashed=(mlp_type == "hash_moe"))
            self.experts = Experts(c)
        self.shared_experts = SwiGLU(c.d_model, c.moe_inter, c.swiglu_limit)

    def forward(self, x, input_ids):
        out = self.shared_experts(x)
        if self.c.use_moe:
            flat = x.reshape(-1, x.shape[-1])
            idx, w = self.gate(flat, input_ids)
            out = out + self.experts(flat, idx, w).view_as(x)
        return out


# ---------------------------------------------------------------- mHC

class HyperConnection(nn.Module):
    """줄기 N 개 [B,T,N,D] → (post [B,T,N], comb [B,T,N,N], 블록 입력 [B,T,D])."""

    def __init__(self, c: DeepseekV4Config):
        super().__init__()
        self.c, n = c, c.hc_mult
        self.fn = nn.Parameter(torch.randn((2 + n) * n, n * c.d_model) * 0.02)
        self.base = nn.Parameter(torch.zeros((2 + n) * n))
        self.scale = nn.Parameter(torch.ones(3))
        self.comb_probe = nn.Identity()   # hook 지점: Sinkhorn 후 줄기 혼합 행렬

    def forward(self, h):
        B, T, n, _ = h.shape
        f = F.linear(unweighted_rmsnorm(h.reshape(B, T, -1).float(), self.c.rms_eps), self.fn.float())
        pre_w, post_w, comb_w = f.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.base.split([n, n, n * n])
        s_pre, s_post, s_comb = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * s_pre + pre_b) + self.c.hc_eps
        post = 2 * torch.sigmoid(post_w * s_post + post_b)
        comb = torch.softmax(comb_w.view(B, T, n, n) * s_comb + comb_b.view(n, n), dim=-1) + self.c.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.c.hc_eps)          # Sinkhorn: 열/행 번갈아 정규화
        for _ in range(self.c.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.c.hc_eps)
        comb = self.comb_probe(comb)
        return post, comb, (pre.unsqueeze(-1) * h).sum(2).to(h.dtype)


class HyperHead(nn.Module):
    """마지막에 줄기 N 개를 하나로 합친다."""

    def __init__(self, c: DeepseekV4Config):
        super().__init__()
        self.c, n = c, c.hc_mult
        self.hc_fn = nn.Parameter(torch.randn(n, n * c.d_model) * 0.02)
        self.hc_base = nn.Parameter(torch.zeros(n))
        self.hc_scale = nn.Parameter(torch.ones(1))

    def forward(self, h):
        f = F.linear(unweighted_rmsnorm(h.flatten(2).float(), self.c.rms_eps), self.hc_fn.float())
        pre = torch.sigmoid(f * self.hc_scale.float() + self.hc_base.float()) + self.c.hc_eps
        return (pre.unsqueeze(-1) * h).sum(2).to(h.dtype)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: DeepseekV4Config, i: int):
        super().__init__()
        self.c = c
        self.self_attn = Attention(c, c.layer_types[i])
        self.mlp = MoE(c, c.mlp_layer_types[i])
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps)
        if c.use_mhc:
            self.attn_hc = HyperConnection(c)
            self.ffn_hc = HyperConnection(c)

    def forward(self, h, input_ids, inv_main, inv_comp):
        if not self.c.use_mhc:                         # 일반 pre-norm residual (ablation)
            h = h + self.self_attn(self.input_layernorm(h), inv_main, inv_comp)
            return h + self.mlp(self.post_attention_layernorm(h), input_ids)
        post, comb, x = self.attn_hc(h)
        a = self.self_attn(self.input_layernorm(x), inv_main, inv_comp)
        # 새 줄기 = post·(블록 출력) + combᵀ·(기존 줄기)
        h = post.to(h.dtype).unsqueeze(-1) * a.unsqueeze(-2) + comb.to(h.dtype).transpose(-1, -2) @ h
        post, comb, x = self.ffn_hc(h)
        m = self.mlp(self.post_attention_layernorm(x), input_ids)
        return post.to(h.dtype).unsqueeze(-1) * m.unsqueeze(-2) + comb.to(h.dtype).transpose(-1, -2) @ h


class DeepseekV4(nn.Module):
    def __init__(self, c: DeepseekV4Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        self.hc_head = HyperHead(c) if c.use_mhc else None
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.register_buffer("inv_main", rope_inv_freq(c.rope_dim, c.rope_theta), persistent=False)
        self.register_buffer("inv_comp", rope_inv_freq(c.rope_dim, c.compress_rope_theta), persistent=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, GroupedLinear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        x = self.embed_tokens(input_ids)
        h = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous() if self.config.use_mhc else x
        for layer in self.layers:
            h = layer(h, input_ids, self.inv_main, self.inv_comp)
        h = self.norm(self.hc_head(h) if self.config.use_mhc else h)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)

    @torch.no_grad()
    def update_router_bias(self, rate: float = 1e-3) -> None:
        """aux-loss-free 부하 분산: 직전 forward 에서 적게 뽑힌 expert 의 bias 를 올리고 많이 뽑힌 쪽은 내린다.

        각 Router 의 route_probe 에 기록된 선택 결과가 필요하므로, 학습 루프에서 forward 직후 호출한다.
        """
        for m in self.modules():
            if isinstance(m, Router) and not m.hashed and getattr(m, "_last_idx", None) is not None:
                load = torch.bincount(m._last_idx.flatten(), minlength=self.config.n_experts).float()
                m.e_score_correction_bias += rate * torch.sign(load.mean() - load)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> DeepseekV4Config:
    rp = c.rope_parameters
    return DeepseekV4Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers,
        layer_types=list(c.layer_types), mlp_layer_types=list(c.mlp_layer_types),
        n_head=c.num_attention_heads, head_dim=c.head_dim, rope_dim=c.qk_rope_head_dim, q_lora_rank=c.q_lora_rank,
        o_groups=c.o_groups, o_lora_rank=c.o_lora_rank, window=c.sliding_window,
        csa_rate=c.compress_rates[CSA], hca_rate=c.compress_rates[HCA],
        rope_theta=rp["main"]["rope_theta"], compress_rope_theta=rp["compress"]["rope_theta"],
        index_heads=c.index_n_heads, index_dim=c.index_head_dim, index_topk=c.index_topk,
        hc_mult=c.hc_mult, hc_iters=c.hc_sinkhorn_iters, hc_eps=c.hc_eps,
        n_experts=c.n_routed_experts, top_k=c.num_experts_per_tok, moe_inter=c.moe_intermediate_size,
        route_scale=c.routed_scaling_factor, swiglu_limit=c.swiglu_limit, rms_eps=c.rms_norm_eps)


def verify_pair(ref) -> dict:
    """ref = transformers DeepseekV4Model (이름이 같아 name_map 불필요)"""
    mine = DeepseekV4(config_from_hf(ref.config))
    return dict(mine=mine, name_map=None,
                hook_pairs=[(f"layers.{i}", f"layers.{i}") for i in range(ref.config.num_hidden_layers)])
