"""A3·A5a · GLM-5.3 블록: MLA + DSA(Lightning Indexer, 층 간 top-k 공유) + 세분화 MoE.

HF 원본: transformers/models/glm_moe_dsa (GLM-5 / 5.1 / 5.2 / 5.3 공통, Flash 는 구조가 달라 제외).
파라미터 이름은 HF 와 같다 (lm_head 는 HF GlmMoeDsaModel 에 없으므로 검증 때는 tie).

블록 구성 (한 층 = pre-norm residual 두 번)
1. MLA (Multi-head Latent Attention)
   - q: hidden → q_a_proj(저랭크 q_lora_rank) → RMSNorm → q_b_proj → head 마다 [nope | rope] 채널
   - kv: hidden → kv_a_proj_with_mqa → [잠재 c (kv_lora_rank) | 공유 k_rope (rope_dim)]
         c 를 RMSNorm 후 kv_b_proj 로 head 마다 [k_nope | v] 로 펼친다 (cache 에는 작은 c 와 k_rope 만 저장).
   - decoupled RoPE: 위치 정보는 rope 채널에만 (interleaved 쌍 회전), k_rope 는 모든 head 가 공유.
2. DSA (DeepSeek Sparse Attention)
   - Lightning Indexer: 작은 질의(q 잠재에서)·단일 key 로 점수 Σ_h w_h·ReLU(q_h·k) 를 매겨
     질의마다 이전 토큰 중 top-k 만 고르고, MLA 는 고른 토큰만 본다.
   - 층 간 공유: indexer_types 가 "shared" 인 층은 indexer 없이 직전 "full" 층의 선택을 그대로 쓴다
     (원본 GLM-5.3: 앞 3층 full, 이후 4층마다 full).
3. MoE — sigmoid 점수 + 그룹 제한 top-k (aux-loss-free bias 는 선택에만), 선택 가중치 정규화 × scale,
   + shared expert. 앞쪽 first_k_dense 층은 dense SwiGLU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, RMSNorm


@dataclass
class GLM53Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 4
    mlp_layer_types: list[str] | None = None    # None → 앞 first_k_dense 층 "dense", 나머지 "sparse"
    first_k_dense: int = 3
    indexer_types: list[str] | None = None      # None → index_topk_freq / offset 로 생성 (HF 와 같은 규칙)
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    # MLA
    n_head: int = 4
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_dim: int = 32
    qk_rope_dim: int = 16
    v_head_dim: int = 32
    rope_theta: float = 10000.0
    attn_scale: float | None = None             # None → (nope + rope) ** -0.5
    attention_bias: bool = False
    # DSA indexer
    index_heads: int = 4
    index_dim: int = 32                         # 앞 qk_rope_dim 채널에만 RoPE
    index_topk: int = 16
    # MLP / MoE
    d_ff: int = 512                             # dense 층
    moe_inter: int = 64                         # routed / shared expert 중간 차원
    n_shared: int = 1
    n_experts: int = 8
    top_k: int = 2
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    route_scale: float = 2.5
    # 기타
    rms_eps: float = 1e-5                       # 층 norm. q/kv 잠재 norm 은 1e-6 고정 (HF 와 동일)
    tie_embeddings: bool = True
    # ablation 스위치 (기본값 = 원본)
    use_dsa: bool = True                        # False: indexer 를 건너뛰고 dense causal attention
    share_indexer: bool = True                  # False: 모든 층이 자기 indexer 를 돌림 (shared 층에 indexer 새로 생김)
    use_moe: bool = True                        # False: shared expert 하나만 (dense SwiGLU)
    use_router_bias: bool = True                # False: expert 선택에 e_score_correction_bias 를 안 씀

    def __post_init__(self):
        n = self.n_layer
        if self.mlp_layer_types is None:
            k = min(self.first_k_dense, n)
            self.mlp_layer_types = ["dense"] * k + ["sparse"] * (n - k)
        if self.indexer_types is None:
            f, o = max(self.index_topk_freq, 1), self.index_skip_topk_offset
            self.indexer_types = ["full" if max(i - o + 1, 0) % f == 0 else "shared" for i in range(n)]
        assert len(self.mlp_layer_types) == n and len(self.indexer_types) == n
        assert self.indexer_types[0] == "full", "첫 층은 indexer 를 직접 돌려야 한다"


# ---------------------------------------------------------------- RoPE (interleaved)

def rope_inv_freq(rope_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))


def apply_rope_interleave(x, cos, sin):
    """(x0,x1),(x2,x3).. 쌍을 회전. 출력은 [짝수 채널 결과 | 홀수 채널 결과] 순서 (HF 와 같은 배치).

    q, k 를 같은 방식으로 재배치하므로 내적은 원래 interleaved RoPE 와 같다. cos/sin 폭 = rope_dim/2.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------- DSA indexer

class Indexer(nn.Module):
    """Lightning Indexer. 질의 t 마다 causal 범위 안에서 점수 top-k 토큰 index [B, T, k] 를 돌려준다."""

    def __init__(self, c: GLM53Config):
        super().__init__()
        self.c = c
        self.heads, self.dim, self.rope = c.index_heads, c.index_dim, c.qk_rope_dim
        self.wq_b = nn.Linear(c.q_lora_rank, self.heads * self.dim, bias=False)   # q 잠재 재사용
        self.wk = nn.Linear(c.d_model, self.dim, bias=False)                      # 단일 key head
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)                            # RMSNorm 이 아니라 LayerNorm
        self.weights_proj = nn.Linear(c.d_model, self.heads, bias=False)          # head 별 가중치 w_h
        self.score_probe = nn.Identity()   # hook 지점: mask 전 indexer 점수 [B, T, T]
        self.topk_probe = nn.Identity()    # hook 지점: 고른 토큰 index [B, T, k]

    @torch.no_grad()   # HF 와 같이 indexer 로는 gradient 가 흐르지 않는다
    def forward(self, x, q_resid, cos, sin, causal):
        B, T, _ = x.shape
        q = self.wq_b(q_resid).view(B, T, self.heads, self.dim)
        q_rot, q_pass = q.split([self.rope, self.dim - self.rope], dim=-1)        # 여기는 rope 가 앞쪽
        k = self.k_norm(self.wk(x)).unsqueeze(2)                                  # [B, T, 1, d]
        k_rot, k_pass = k.split([self.rope, self.dim - self.rope], dim=-1)
        cq, sq = cos[:, None, :], sin[:, None, :]                                 # [T, 1, r/2]
        q = torch.cat([apply_rope_interleave(q_rot, cq, sq), q_pass], dim=-1)     # [B, T, h, d]
        k = torch.cat([apply_rope_interleave(k_rot, cq, sq), k_pass], dim=-1).squeeze(2)  # [B, T, d]

        s = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.dim ** -0.5  # [B, T, h, T]
        s = F.relu(s)
        w = self.weights_proj(x).float() * self.heads ** -0.5                     # [B, T, h]
        scores = self.score_probe(torch.matmul(w.unsqueeze(-2), s).squeeze(-2))   # [B, T, T]
        scores = scores.masked_fill(~causal, float("-inf"))
        idx = scores.topk(min(self.c.index_topk, T), dim=-1).indices
        return self.topk_probe(idx)


# ---------------------------------------------------------------- MLA

class Attention(nn.Module):
    def __init__(self, c: GLM53Config, has_indexer: bool):
        super().__init__()
        self.c = c
        self.h, self.nope, self.rope, self.vd = c.n_head, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim
        self.qk = self.nope + self.rope
        self.scale = c.attn_scale if c.attn_scale is not None else self.qk ** -0.5
        b = c.attention_bias
        self.q_a_proj = nn.Linear(c.d_model, c.q_lora_rank, bias=b)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, 1e-6)
        self.q_b_proj = nn.Linear(c.q_lora_rank, self.h * self.qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(c.d_model, c.kv_lora_rank + self.rope, bias=b)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, 1e-6)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, self.h * (self.nope + self.vd), bias=False)
        self.o_proj = nn.Linear(self.h * self.vd, c.d_model, bias=b)
        self.indexer = Indexer(c) if has_indexer else None
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T] (고르지 않은 토큰은 0)
        self.sel_probe = nn.Identity()    # hook 지점: 실제로 볼 수 있는 칸 (causal ∧ top-k) [B, T, T] bool

    def forward(self, x, cos, sin, prev_topk=None):
        B, T, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(B, T, self.h, self.qk).transpose(1, 2)
        q_pass, q_rot = q.split([self.nope, self.rope], dim=-1)                   # MLA 는 rope 가 뒤쪽
        kv_pass, k_rot = self.kv_a_proj_with_mqa(x).split([self.c.kv_lora_rank, self.rope], dim=-1)
        c_kv = self.kv_a_layernorm(kv_pass)                                       # 잠재 c [B, T, r]
        k_rot = k_rot.view(B, 1, T, self.rope)
        q_rot, k_rot = apply_rope_interleave(q_rot, cos, sin), apply_rope_interleave(k_rot, cos, sin)
        query = torch.cat((q_pass, q_rot), dim=-1)                                # [B, H, T, nope+rope]

        kv = self.kv_b_proj(c_kv).view(B, T, self.h, self.nope + self.vd).transpose(1, 2)
        k_nope, value = kv.split([self.nope, self.vd], dim=-1)
        key = torch.cat((k_nope, k_rot.expand(-1, self.h, -1, -1)), dim=-1)      # k_rope 는 head 공유

        i = torch.arange(T, device=x.device)
        causal = (i[None, :] <= i[:, None]).expand(B, T, T)
        topk, allow = prev_topk, causal
        if self.c.use_dsa:
            if self.indexer is not None:
                topk = self.indexer(x, q_resid, cos, sin, causal)
            sel = torch.zeros(B, T, T, dtype=torch.bool, device=x.device).scatter(-1, topk.long(), True)
            allow = causal & sel
        allow = self.sel_probe(allow)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        scores = scores.masked_fill(~allow[:, None], float("-inf"))
        probs = self.attn_probs(F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype))
        out = torch.matmul(probs, value).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), topk


# ---------------------------------------------------------------- MLP / MoE

class MLP(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """sigmoid 점수 → (bias 더한 점수로) 그룹 top-k → 그룹 안 expert top-k. 가중치는 bias 없는 점수에서."""

    def __init__(self, c: GLM53Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.empty(c.n_experts, c.d_model))
        nn.init.normal_(self.weight, std=0.02)
        # aux-loss-free 부하 분산용 bias (선택에만 쓰고 가중치에는 안 씀). HF 처럼 state_dict 에 들어간다.
        self.register_buffer("e_score_correction_bias", torch.zeros(c.n_experts, dtype=torch.float32))
        self.score_probe = nn.Identity()   # hook 지점: sigmoid 점수 [N, E]
        self.route_probe = nn.Identity()   # hook 지점: 선택된 expert [N, k]
        self._last_idx = None

    def forward(self, x):
        c = self.c
        E, G = c.n_experts, c.n_group
        logits = F.linear(x.float(), self.weight.float())
        scores = self.score_probe(logits.sigmoid())
        choice = scores + self.e_score_correction_bias if c.use_router_bias else scores
        # 그룹 점수 = 그룹 안 상위 2 개 합 → 상위 topk_group 그룹 밖 expert 는 후보에서 제외
        group_scores = choice.view(-1, G, E // G).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=c.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, G, E // G).reshape(-1, E)
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
        idx = torch.topk(choice, k=c.top_k, dim=-1, sorted=False)[1]
        self._last_idx = self.route_probe(idx).detach()
        w = scores.gather(1, idx)
        if c.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * c.route_scale


class Experts(nn.Module):
    def __init__(self, c: GLM53Config):
        super().__init__()
        self.c = c
        self.gate_up_proj = nn.Parameter(torch.randn(c.n_experts, 2 * c.moe_inter, c.d_model) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(c.n_experts, c.d_model, c.moe_inter) * 0.02)

    def forward(self, x, idx, w):
        out = torch.zeros_like(x)
        for e in range(self.c.n_experts):
            slot, tok = torch.where(idx.T == e)          # HF 와 같은 순서 (slot 우선)
            if tok.numel() == 0:
                continue
            g, u = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.linear(F.silu(g) * u, self.down_proj[e]) * w[tok, slot, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out


class MoE(nn.Module):
    def __init__(self, c: GLM53Config):
        super().__init__()
        self.c = c
        if c.use_moe:
            self.experts = Experts(c)
            self.gate = Router(c)
        self.shared_experts = MLP(c.d_model, c.moe_inter * c.n_shared)

    def forward(self, x):
        if not self.c.use_moe:
            return self.shared_experts(x)
        flat = x.reshape(-1, x.shape[-1])
        idx, w = self.gate(flat)
        return self.experts(flat, idx, w).view_as(x) + self.shared_experts(x)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: GLM53Config, i: int):
        super().__init__()
        has_indexer = c.indexer_types[i] == "full" or not c.share_indexer
        self.self_attn = Attention(c, has_indexer)
        self.mlp = MoE(c) if c.mlp_layer_types[i] == "sparse" else MLP(c.d_model, c.d_ff)
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps)

    def forward(self, h, cos, sin, prev_topk=None):
        a, topk = self.self_attn(self.input_layernorm(h), cos, sin, prev_topk)
        h = h + a
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h, topk                                  # topk 는 다음 "shared" 층으로 넘어간다


class GLM53(nn.Module):
    def __init__(self, c: GLM53Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.register_buffer("inv_freq", rope_inv_freq(c.qk_rope_dim, c.rope_theta), persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        T = input_ids.shape[1]
        freqs = torch.arange(T, device=input_ids.device).float()[:, None] * self.inv_freq[None]
        cos, sin = freqs.cos(), freqs.sin()             # [T, rope/2] (interleaved 쌍마다 각도 하나)
        h = self.embed_tokens(input_ids)
        topk = None
        for layer in self.layers:
            h, topk = layer(h, cos, sin, topk)
        h = self.norm(h)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)

    @torch.no_grad()
    def update_router_bias(self, rate: float = 1e-3) -> None:
        """aux-loss-free 부하 분산: 직전 forward 에서 적게 뽑힌 expert 의 bias 를 올리고 많이 뽑힌 쪽은 내린다."""
        for m in self.modules():
            if isinstance(m, Router) and m._last_idx is not None:
                load = torch.bincount(m._last_idx.flatten(), minlength=self.config.n_experts).float()
                m.e_score_correction_bias += rate * torch.sign(load.mean() - load)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> GLM53Config:
    rp = c.rope_parameters or {}
    assert rp.get("rope_type", "default") == "default", "default RoPE 만 구현 (GLM-5.3 원본도 default)"
    scale = (c.qk_nope_head_dim + c.qk_rope_head_dim) ** -0.5     # HF yarn_apply_mscale 는 default 면 그대로
    assert c.hidden_act == "silu" and c.num_key_value_heads == c.num_attention_heads
    return GLM53Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=c.num_hidden_layers,
        mlp_layer_types=list(c.mlp_layer_types), indexer_types=list(c.indexer_types),
        n_head=c.num_attention_heads, q_lora_rank=c.q_lora_rank, kv_lora_rank=c.kv_lora_rank,
        qk_nope_dim=c.qk_nope_head_dim, qk_rope_dim=c.qk_rope_head_dim, v_head_dim=c.v_head_dim,
        rope_theta=float(rp.get("rope_theta", 10000.0)), attn_scale=scale, attention_bias=c.attention_bias,
        index_heads=c.index_n_heads, index_dim=c.index_head_dim, index_topk=c.index_topk,
        d_ff=c.intermediate_size, moe_inter=c.moe_intermediate_size, n_shared=c.n_shared_experts,
        n_experts=c.n_routed_experts, top_k=c.num_experts_per_tok, n_group=c.n_group, topk_group=c.topk_group,
        norm_topk_prob=c.norm_topk_prob, route_scale=c.routed_scaling_factor, rms_eps=c.rms_norm_eps)


def verify_pair(ref) -> dict:
    """ref = transformers GlmMoeDsaModel (이름이 같아 name_map 불필요)"""
    mine = GLM53(config_from_hf(ref.config))
    pairs = []
    for i in range(ref.config.num_hidden_layers):
        pairs += [(f"layers.{i}.self_attn", f"layers.{i}.self_attn"), (f"layers.{i}.mlp", f"layers.{i}.mlp"),
                  (f"layers.{i}", f"layers.{i}")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
