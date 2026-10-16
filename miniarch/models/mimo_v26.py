"""A2 · MiMo-V2.6 (Flash / Pro) 텍스트 backbone.

HF 원본: XiaomiMiMo/MiMo-V2.6-Flash-RL 등의 원격 코드 modeling_mimo_v2.py (MiMoV2Model).
파라미터 이름은 원본과 같다 (vision/audio 인코더, MTP 층, DFlash draft 는 제외).

블록 구성 (한 층 = pre-norm attention + pre-norm MLP/MoE)
1. Attention — 층 종류가 둘이다.
   - full    : 일반 causal attention (Flash 는 1 + 5개마다 1층, Pro 는 7개마다 1층)
   - sliding : 최근 window 개 토큰(자기 자신 포함)만 보는 attention. 나머지 층 전부.
   두 종류가 head 수, KV head 수(GQA), head_dim, RoPE theta 를 각자 따로 가진다.
   공통 부품
   - q/k head_dim 과 v head_dim 이 다르다 (예: 192 / 128). o_proj 입력은 head 수 × v_dim.
   - partial RoPE: head 앞쪽 rope_dim(= head_dim × 0.334) 채널만 회전 (rotate_half 방식), 나머지는 위치 정보 없음.
   - value scale: v 에 상수(0.707 등)를 곱한 뒤 attention.
   - attention sink (gpt-oss 방식): head 마다 학습되는 로짓 하나를 softmax 분모에만 추가.
     "아무것도 안 보기" 칸이라 확률 일부가 거기로 빠지고, 나머지 토큰 확률 합은 1 보다 작아진다.
     원본 설정은 sliding 층에만 sink 가 있다. QK-norm 은 없다.
   - 실제 체크포인트는 q/k/v 를 하나로 합친 qkv_proj ("fused_qkv"), 분리형("split")도 지원.
2. MLP — 첫 층은 dense SwiGLU, 나머지는 MoE.
   MoE: sigmoid 점수 + (선택용) 보정 bias → 그룹 top-k → expert top-k, 가중치는 bias 없는 점수를 정규화.
   shared expert 는 없다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .common import LMOutput, RMSNorm, rope_cos_sin, rope_inv_freq, rotate_half

FULL, SWA = "full_attention", "sliding_attention"


@dataclass
class MiMoV26Config:
    vocab_size: int = 128
    d_model: int = 256
    n_layer: int = 8
    layer_types: list[str] | None = None      # None → 0번 층 + hybrid_block 개마다 마지막 층이 full, 나머지 sliding
    hybrid_block: int = 8                     # Pro: 8 (sliding 7 : full 1), Flash: 6
    mlp_types: list[str] | None = None        # None → 0번 층 dense, 나머지 moe
    # full attention 층
    n_head: int = 8
    n_kv_head: int = 2
    head_dim: int = 48                        # q/k
    v_head_dim: int = 32
    rope_theta: float = 1e7
    full_sink: bool = False
    # sliding attention 층
    swa_n_head: int = 8
    swa_n_kv_head: int = 2
    swa_head_dim: int = 48
    swa_v_head_dim: int = 32
    swa_rope_theta: float = 1e4
    swa_sink: bool = True
    window: int = 32                          # 자기 자신 포함 최근 window 토큰
    # 공통 attention
    partial_rotary: float = 0.334             # head 앞쪽 int(head_dim × 0.334) 채널만 RoPE
    v_scale: float | None = 0.707
    qkv_layout: str = "fused_qkv"             # "fused_qkv"(실제 체크포인트) / "split"
    attn_bias: bool = False                   # q/k/v bias (o_proj 는 항상 없음)
    # MLP / MoE
    d_ff: int = 512                           # dense 층
    moe_inter: int = 64                       # expert 하나의 중간 차원
    n_experts: int = 8
    top_k: int = 2
    n_group: int = 1                          # expert 그룹 수 (실제 설정 1 = 그룹 제한 없음)
    topk_group: int = 1                       # 토큰마다 고를 그룹 수
    norm_topk: bool = True
    route_scale: float = 1.0
    # 기타
    rms_eps: float = 1e-6
    tie_embeddings: bool = False
    # ablation 스위치 (기본값 = 원본)
    use_sink: bool = True                     # False: sink 제거 (일반 softmax)
    use_swa: bool = True                      # False: sliding 층도 window 없이 전체 causal
    use_value_scale: bool = True              # False: v 스케일 생략
    use_moe: bool = True                      # False: moe 층도 dense SwiGLU(d_ff)

    def __post_init__(self):
        n = self.n_layer
        if self.layer_types is None:
            self.layer_types = [FULL if i == 0 or (i + 1) % self.hybrid_block == 0 else SWA for i in range(n)]
        if self.mlp_types is None:
            self.mlp_types = ["dense"] + ["moe"] * (n - 1)
        assert len(self.layer_types) == n and len(self.mlp_types) == n
        assert self.n_experts % self.n_group == 0


# ---------------------------------------------------------------- Attention

def apply_partial_rope(x, cos, sin):
    """head_dim 중 앞 rope_dim(= cos 폭) 만 회전하고 나머지는 그대로. x [B, H, T, D], cos/sin [T, r]."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


class Attention(nn.Module):
    def __init__(self, c: MiMoV26Config, swa: bool):
        super().__init__()
        self.c, self.swa = c, swa
        if swa:
            self.h, self.kv, self.d, self.dv = c.swa_n_head, c.swa_n_kv_head, c.swa_head_dim, c.swa_v_head_dim
        else:
            self.h, self.kv, self.d, self.dv = c.n_head, c.n_kv_head, c.head_dim, c.v_head_dim
        self.rope_dim = int(self.d * c.partial_rotary)
        assert self.rope_dim % 2 == 0, "rope_dim 은 짝수여야 함"
        self.q_size, self.k_size, self.v_size = self.h * self.d, self.kv * self.d, self.kv * self.dv
        if c.qkv_layout == "fused_qkv":
            self.qkv_proj = nn.Linear(c.d_model, self.q_size + self.k_size + self.v_size, bias=c.attn_bias)
        else:
            self.q_proj = nn.Linear(c.d_model, self.q_size, bias=c.attn_bias)
            self.k_proj = nn.Linear(c.d_model, self.k_size, bias=c.attn_bias)
            self.v_proj = nn.Linear(c.d_model, self.v_size, bias=c.attn_bias)
        self.o_proj = nn.Linear(self.h * self.dv, c.d_model, bias=False)
        has_sink = c.use_sink and (c.swa_sink if swa else c.full_sink)
        # head 별 sink 로짓 (원본은 추론 코드라 requires_grad=False, 여기선 학습 가능)
        self.attention_sink_bias = nn.Parameter(torch.zeros(self.h)) if has_sink else None
        self.attn_probs = nn.Identity()   # hook 지점: [B, H, T, T] (sink 칸 제외)
        if has_sink:
            self.sink_probe = nn.Identity()   # hook 지점: sink 로 간 확률 [B, H, T]

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        if self.c.qkv_layout == "fused_qkv":
            q, k, v = self.qkv_proj(x).split([self.q_size, self.k_size, self.v_size], dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.kv, self.d).transpose(1, 2)
        v = v.view(B, T, self.kv, self.dv).transpose(1, 2)
        if self.c.use_value_scale and self.c.v_scale is not None:
            v = v * self.c.v_scale
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        rep = self.h // self.kv                                            # GQA: head i 는 KV head i // rep
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.d ** -0.5
        i, j = torch.arange(T, device=x.device)[:, None], torch.arange(T, device=x.device)[None, :]
        allow = j <= i
        if self.swa and self.c.use_swa:
            allow = allow & (i - j < self.c.window)
        scores = scores.masked_fill(~allow, float("-inf"))
        if self.attention_sink_bias is not None:                           # 분모에만 들어가는 sink 칸
            sink = self.attention_sink_bias.view(1, -1, 1, 1).expand(B, -1, T, 1).to(scores.dtype)
            scores = torch.cat([scores, sink], dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.attention_sink_bias is not None:
            self.sink_probe(probs[..., -1])
            probs = probs[..., :-1]
        probs = self.attn_probs(probs)
        y = (probs @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(y)


# ---------------------------------------------------------------- MLP / MoE

class MLP(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """sigmoid 점수 → (점수 + bias) 로 그룹 top-k → 고른 그룹 안에서 expert top-k. 가중치는 bias 없는 점수."""

    def __init__(self, c: MiMoV26Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.randn(c.n_experts, c.d_model) * 0.02)
        # aux-loss-free 부하 분산 bias (선택에만 쓰고 가중치에는 안 씀, 학습 안 함)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(c.n_experts), requires_grad=False)
        self.score_probe = nn.Identity()   # hook 지점: sigmoid 점수 [N, E]
        self.group_probe = nn.Identity()   # hook 지점: 고른 그룹 [N, topk_group]
        self.route_probe = nn.Identity()   # hook 지점: 고른 expert [N, k]

    def forward(self, x):                                                  # x [N, d]
        c, N = self.c, x.shape[0]
        scores = self.score_probe(F.linear(x.float(), self.weight.float()).sigmoid())
        choice = scores + self.e_score_correction_bias.unsqueeze(0)
        # 그룹 점수 = 그룹 안 상위 2 개 합
        group_scores = choice.view(N, c.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = self.group_probe(torch.topk(group_scores, k=c.topk_group, dim=-1, sorted=False)[1])
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(N, c.n_group, c.n_experts // c.n_group).reshape(N, -1)
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
        idx = self.route_probe(torch.topk(choice, k=c.top_k, dim=-1, sorted=False)[1])
        w = scores.gather(1, idx)
        if c.top_k > 1 and c.norm_topk:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * c.route_scale


class MoE(nn.Module):
    def __init__(self, c: MiMoV26Config):
        super().__init__()
        self.experts = nn.ModuleList(MLP(c.d_model, c.moe_inter) for _ in range(c.n_experts))
        self.gate = Gate(c)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        idx, w = self.gate(flat)
        out = torch.zeros_like(flat, dtype=w.dtype)                       # fp32 누적 (원본과 같음)
        for e, expert in enumerate(self.experts):
            tok, slot = torch.where(idx == e)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, expert(flat[tok]) * w[tok, slot].unsqueeze(-1))
        return out.to(x.dtype).view(shape)


# ---------------------------------------------------------------- 층 / 모델

class Layer(nn.Module):
    def __init__(self, c: MiMoV26Config, i: int):
        super().__init__()
        self.attention_type = c.layer_types[i]
        self.self_attn = Attention(c, swa=self.attention_type == SWA)
        self.mlp = MoE(c) if (c.mlp_types[i] == "moe" and c.use_moe) else MLP(c.d_model, c.d_ff)
        self.input_layernorm = RMSNorm(c.d_model, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(c.d_model, c.rms_eps)

    def forward(self, x, rope_full, rope_swa):
        cos, sin = rope_swa if self.attention_type == SWA else rope_full
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class MiMoV26(nn.Module):
    def __init__(self, c: MiMoV26Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.rms_eps)
        self.lm_head = None if c.tie_embeddings else nn.Linear(c.d_model, c.vocab_size, bias=False)
        # full / sliding 층은 head_dim 과 theta 가 달라 RoPE 주파수도 따로
        self.register_buffer("inv_full", rope_inv_freq(int(c.head_dim * c.partial_rotary), c.rope_theta),
                             persistent=False)
        self.register_buffer("inv_swa", rope_inv_freq(int(c.swa_head_dim * c.partial_rotary), c.swa_rope_theta),
                             persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> LMOutput:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        rope_full, rope_swa = rope_cos_sin(pos, self.inv_full), rope_cos_sin(pos, self.inv_swa)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, rope_full, rope_swa)
        h = self.norm(x)
        logits = self.lm_head(h) if self.lm_head is not None else h @ self.embed_tokens.weight.T
        return LMOutput(h, logits)


# ---------------------------------------------------------------- 검증

def config_from_hf(c) -> MiMoV26Config:
    assert getattr(c, "hidden_act", "silu") == "silu"
    assert getattr(c, "scoring_func", "sigmoid") == "sigmoid" and getattr(c, "topk_method", "noaux_tc") == "noaux_tc"
    n = c.num_hidden_layers
    has_moe = getattr(c, "n_routed_experts", None) is not None
    rp = getattr(c, "rope_parameters", None) or {}
    return MiMoV26Config(
        vocab_size=c.vocab_size, d_model=c.hidden_size, n_layer=n,
        layer_types=[SWA if c.hybrid_layer_pattern[i] == 1 else FULL for i in range(n)],
        mlp_types=["moe" if has_moe and c.moe_layer_freq[i] else "dense" for i in range(n)],
        n_head=c.num_attention_heads, n_kv_head=c.num_key_value_heads, head_dim=c.head_dim, v_head_dim=c.v_head_dim,
        rope_theta=rp.get("rope_theta", c.rope_theta), full_sink=bool(c.add_full_attention_sink_bias),
        swa_n_head=c.swa_num_attention_heads, swa_n_kv_head=c.swa_num_key_value_heads,
        swa_head_dim=c.swa_head_dim, swa_v_head_dim=c.swa_v_head_dim, swa_rope_theta=c.swa_rope_theta,
        swa_sink=bool(c.add_swa_attention_sink_bias), window=c.sliding_window or 0,
        partial_rotary=c.partial_rotary_factor, v_scale=c.attention_value_scale,
        qkv_layout=c.attention_projection_layout or "split", attn_bias=bool(c.attention_bias),
        d_ff=c.intermediate_size, moe_inter=c.moe_intermediate_size,
        n_experts=c.n_routed_experts if has_moe else 1, top_k=c.num_experts_per_tok or 1,
        n_group=c.n_group or 1, topk_group=c.topk_group or 1, norm_topk=bool(c.norm_topk_prob),
        route_scale=c.routed_scaling_factor if c.routed_scaling_factor is not None else 1.0,
        rms_eps=c.layernorm_epsilon, tie_embeddings=bool(getattr(c, "tie_word_embeddings", False)))


def verify_pair(ref) -> dict:
    """ref = 원격 코드 MiMoV2Model (이름이 같아 name_map 불필요).

    원본은 lm_head 가 없는 backbone 이라 비교용으로는 tie_embeddings=True (복사 안 된 lm_head 가 안 생기게).
    """
    mine = MiMoV26(replace(config_from_hf(ref.config), tie_embeddings=True))
    pairs = []
    for i in range(ref.config.num_hidden_layers):
        pairs += [(f"layers.{i}.self_attn", f"layers.{i}.self_attn"), (f"layers.{i}.mlp", f"layers.{i}.mlp"),
                  (f"layers.{i}", f"layers.{i}")]
    return dict(mine=mine, name_map=None, hook_pairs=pairs)
